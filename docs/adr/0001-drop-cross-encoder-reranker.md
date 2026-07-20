# ADR 0001 — Drop the cross-encoder reranker from clync search (default design)

- **Status:** Accepted
- **Date:** 2026-07-19
- **Scope:** The planned clync hybrid-search build (BGE-M3 dense + learned-sparse + RRF).
- **Benchmark:** [`experiments/reranker_ablation/RERANKER_ABLATION.md`](../../experiments/reranker_ablation/RERANKER_ABLATION.md) (harness: `experiments/reranker_ablation/ablation.py`).

## Context

The hybrid-search design proposed adding a `bge-reranker-v2-m3` cross-encoder as a
second stage: retrieve a top-50 shortlist with the first-stage hybrid
(dense cosine + learned-sparse dot, fused via RRF), then re-score those 50 and
keep the top-10. A reranker adds real cost — it cannot be precomputed/indexed
(a cross-encoder scores every query×doc pair at query time). We needed to know
whether it earns that cost on **this** corpus before wiring it into production.

We ran an offline ablation on the real Work corpus (20 queries, 16 EN / 4 JA;
ZH unevaluable — the sole ZH conversation is a body-less server-side stub).
Both arms drew from the **identical** 50-shortlist so only the reranker varied.
Relevance graded 0–3 by blind, clean-context Sonnet judges (one grade per
unique query×conversation pair, blind to arm and rank).

A key framing correction happened during review: the initial decision rule
leaned on **nDCG@10** and **top-1-rescue rate**. Both are the wrong metric for
RAG. When the retrieved set is fed to an LLM that reads *all* of it, rank order
*within* the fed set is a weak lever (only soft "lost-in-the-middle" effects) and
k=1 is never the feed size. The decision metric is **recall at the truncation
you actually feed the LLM** (set membership), not rank order.

## Decision

**Do not include the cross-encoder reranker in the default clync search.**

The default design retrieves a small set and lets the consumer (Claude Code) read
all of it. In that regime the reranker adds nothing:

| feed-k | Recall@k (grade≥2) first-stage | + reranker | Δ |
|---|---|---|---|
| 10 | 1.000 | 1.000 | **+0.000** |
| 5 | 0.890 | 0.952 | +0.062 |
| 3 | 0.768 | 0.882 | +0.114 |

At feed-k=10 the first-stage hybrid already contains **every** relevant
conversation; the reranker only reorders a set the LLM fully consumes, for a
measured **+3.8 s/query** on MPS. The value it does produce is confined to
aggressive truncation (feed-k≤3–5), and even there it is **EN-only** — JA
recall@3 is already 1.000 from the first stage, so the reranker has nothing to add
for Japanese.

## Consequences

- **v1 clync search ships first-stage hybrid only** (BGE-M3 dense + learned-sparse
  + RRF). Lower latency, one fewer model at query time.
- **Revisit condition (not now):** adopt the reranker *only if* clync commits to a
  small feed-k (≤3–5, e.g. under a tight context budget) **and** the rerank latency
  is cut (smaller pre_topk, batching, ONNX/GPU export). Absent both, it is latency
  for nothing.
- This weakens one pillar of the BGE-M3 embedder choice ("its reranker closes the
  gap to stronger dense models"). The embedder decision is revisited separately
  (see the embedder ablation).
- **This is a directional offline screen (n=20).** If a reranker is ever
  reconsidered, an online interleaving A/B on live queries is the real arbiter.
