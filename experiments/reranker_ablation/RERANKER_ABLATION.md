# Reranker ablation — results

Corpus: clync Work (20 queries). pre_topk=50, topk=10, RRF k=60. Embedder BAAI/bge-m3, reranker BAAI/bge-reranker-v2-m3.

Benchmark backing [ADR 0001](../../docs/adr/0001-drop-cross-encoder-reranker.md). Reproduce (needs the local clync DB + the gitignored data/ artifacts): `uv run python ablation.py candidates && … embed && … run && … judge-targets`, judge the pairs, then `… report`.

## 1. Divergence gate (label-free): does the reranker even change the top-10?

| metric | mean | read |
|---|---|---|
| overlap@10 (Jaccard of sets) | 0.858 | 1.0 = identical members |
| RBO (p=0.9, top-weighted) | 0.461 | 1.0 = identical order |
| Kendall-τ (shared items) | 0.413 | 1.0 = same order |
| top-1 changed | 4/20 (20%) | divergence signal only — NOT a quality metric (RAG never feeds k=1) |
| mean rank displacement (shared) | 1.87 | positions moved |
| positions changed / 10 | 7.35 | |

**Gate verdict:** The reranker materially reorders the top-10 → quality judging warranted.

## 2. Performance

| stage | p50 ms | p95 ms |
|---|---|---|
| first-stage (embed+dense+sparse+RRF) | 54 | 251 |
| rerank (50 pairs) | 3835 | 4129 |
| **added by reranker** | **3835** | **4129** |

## 3. Quality — set membership at the truncation you FEED the LLM

A reranker's job in RAG is to pack relevant docs into the top-k you actually feed the model. When the LLM reads the whole fed set, order *within* it is a weak lever (only 'lost-in-the-middle' effects), so the decision metric is **Recall@feed-k**, not rank order. Relevant = Sonnet grade ≥ 2.

| feed-k | Recall Arm A | Recall Arm B | Δ | note |
|---|---|---|---|---|
| 3 | 0.768 | 0.882 | +0.114 | aggressive feed budget |
| 5 | 0.890 | 0.952 | +0.062 | aggressive feed budget |
| 10 | 1.000 | 1.000 | +0.000 | = retrieval depth; both arms already hold every relevant conv → reranker cannot add members |

Rank-sensitive metrics (discount for read-all-k consumption): nDCG@10 A=0.939 B=0.973 Δ=+0.035 (9W/3L, sign-test p=0.146).

### Recall@3 (grade≥2) by language
| lang | n | A | B | Δ |
|---|---|---|---|---|
| en | 16 | 0.710 | 0.853 | +0.143 |
| ja | 4 | 1.000 | 1.000 | +0.000 |

## Decision — conditional on feed-k

The reranker's value is entirely a function of how many retrieved convs you feed the LLM. Top-1 change-rate and rank order within the fed set are **not** decision metrics (the consumer reads all of what it's given); set membership at feed-k is.

- **Feed-k = 10 (retrieve 10, read all):** Δrecall = +0.000. The reranker only reorders a set the LLM fully consumes, for +3835 ms/query. **No value.**
- **Feed-k = 3 (retrieve 50 → rerank → feed 3):** Δrecall@3 = +0.114. Here it packs more relevant convs into a tight budget — a real gain, but it costs +3835 ms/query and buys quality by feeding *less*.

**Recommendation: DROP the reranker for a read-all (feed-k≈10) design** — it is pure latency (+3835 ms) for Δrecall=+0.000. It becomes worth revisiting *only* if clync commits to a small feed-k (≤3–5), where Δrecall@3=+0.114 is meaningful — and only after cutting the 3835 ms rerank cost (smaller pre_topk, batching, ONNX/GPU). Offline screen, n=20, directional only; an online interleaving A/B is the real arbiter if adopted.
