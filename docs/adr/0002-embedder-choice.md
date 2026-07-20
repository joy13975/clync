# ADR 0002 — Embedder: BGE-M3 (not Qwen3-Embedding)

- **Status:** Accepted
- **Date:** 2026-07-20
- **Scope:** The embedder for clync hybrid search.
- **Benchmark:** [`experiments/embedder_ablation/EMBEDDER_ABLATION.md`](../../experiments/embedder_ablation/EMBEDDER_ABLATION.md) (harness: `experiments/embedder_ablation/embedder_ablation.py`).

## Context

BGE-M3 was originally chosen for clync by **reasoning**, not measurement: one model
emits dense + learned-sparse in a single pass, its sparse head does CJK lexical
matching that BM25/tsvector can't, and (pre-[ADR 0001](0001-drop-cross-encoder-reranker.md))
"its reranker closes the dense-quality gap." But the public benchmarks actually
favor the alternative — **Qwen3-Embedding beats BGE-M3 on pure dense retrieval
(MMTEB)** — and ADR 0001 removed the reranker pillar. So we measured the two
head-to-head on the real clync corpus, reusing the reranker ablation's 20-query
set + Sonnet judgments (identical corpus and chunking).

Fair-footprint comparison: **Qwen3-Embedding-0.6B** (≈ BGE-M3's 569M params) — the
size viable for an always-on local tool. The 4B/8B Qwen variants top the
leaderboard but are 8–16 GB resident; not tested (the 0.6B already lost, and the
big variants are non-viable for the always-on local use case).

## Decision

**Use BGE-M3 as clync's embedder.** On this corpus it wins on quality *and* speed:

| metric (feed-k=5) | BGE-M3 dense | Qwen3-0.6B dense |
|---|---|---|
| anchor recall@5 (label-free) | **1.000** | 0.850 |
| graded recall@5 (Sonnet ≥2) | **0.925** | 0.838 |
| per-query latency (p50) | **27 ms** | 34 ms |
| corpus index time (240 chunks) | **7.3 s** | 22.4 s |
| peak RSS | 4.5 GB | 2.9 GB |

The head-to-head is **pure dense** (BGE dense vs Qwen dense), so it isolates
embedder quality. BGE wins — the opposite of the public-benchmark expectation, at
this size. Of the 65 conversations Qwen surfaced into some top-10 that BGE did
not, blind judging found **exactly 1 relevant** (grade 3) — Qwen's extra
retrievals are almost all noise. The two embedders genuinely diverge (top-10
Jaccard 0.61, never an identical set), but both reach anchor recall@10 = 1.0, so
the disagreement lives in the tail and the decision bites at small feed-k, where
BGE is ahead.

Qwen's only win is a lower memory footprint (2.9 vs 4.5 GB); it is slower to
index (3×) and slightly slower per query, and it is dense-only (no sparse head).

## Consequences

- **clync embedder = BGE-M3** (`BAAI/bge-m3`), dense as the retrieval backbone.
- **Open sub-finding — sparse did not help here, but the test was unfair to it.**
  The dense+sparse *hybrid* did **not** beat BGE dense-only on this query set
  (anchor recall@5 0.950 vs 1.000; graded@5 0.883 vs 0.925 — marginally *worse*).
  But this query set is semantic-paraphrase-heavy and does **not** stress the
  exact-term / identifier / CJK-lexical matching that learned-sparse exists for.
  So: keep dense as the backbone; the dense+sparse+typed-metadata hybrid design
  still stands for production, but **whether learned-sparse earns its place needs
  its own focused ablation** (exact codes, proper nouns, CJK keyword queries)
  before committing. Do **not** drop sparse on this evidence alone.
- **Directional only:** n=20, EN-dominant (16 EN / 4 JA), ZH unevaluable (corpus
  has no ZH content). A larger, exact-term-inclusive query set would firm this up.

## Addendum (2026-07-20) — int8 quantization is not a win on this hardware

Question: can a quantized/optimized BGE-M3 hold quality for less RAM/time? We
measured `gpahal/bge-m3-onnx-int8` (int8 ONNX Runtime, CPU) as extra arms in the
same harness. (fp8 was ruled out first: Apple Silicon/MPS has no fp8 compute
path, so it would only emulate — no benefit.)

| | fp16 (PyTorch/MPS) | int8 (ONNX/CPU) |
|---|---|---|
| anchor recall@3 / @5 / @10 | 0.950 / 1.000 / 1.000 | 0.950 / 0.950 / 1.000 |
| per-query latency p50 | 27 ms | **9 ms** |
| corpus index (240 chunks) | **7 s** | 74 s |
| peak RSS | **4.5 GB** | 6.8 GB |
| top-10 Jaccard vs fp16 | — | 0.88 dense / 0.90 hybrid |

**Verdict: quality holds, but int8 does NOT deliver the goal (less RAM/time) on
this Mac.** It is 3× faster *per query* (though per-query was never the
bottleneck) but 10× slower to index and uses **more** peak RAM, not less. The
model weights were never the memory bottleneck (~1.1 GB of the fp16 footprint);
int8's peak RSS is dominated by the batch-16 indexing pass — the ONNX Runtime CPU
arena plus the **unused ColBERT output tensor** the export computes anyway.

**Decision: keep fp16 BGE-M3 on MPS as the default.** If RAM later becomes the
binding constraint, the real lever is a **ColBERT-stripped ONNX export**
(dense+sparse heads only) with a small indexing batch — not off-the-shelf int8.
Benchmark: [`experiments/embedder_ablation/EMBEDDER_ABLATION.md`](../../experiments/embedder_ablation/EMBEDDER_ABLATION.md) §1–4.
