# Embedder ablation — BGE-M3 (fp16 & int8) vs Qwen3-Embedding-0.6B

Corpus: clync Work, identical to the reranker ablation (20 queries, same chunks). topk=10, RRF k=60. Dense scored by cosine; BGE sparse by learned-weight dot; hybrid = dense+sparse RRF. int8 = ONNX-Runtime quantized BGE-M3 on CPU (same tokenizer/heads as fp16).

Backs [ADR 0002](../../docs/adr/0002-embedder-choice.md).

## 1. Speed & footprint

| model | params | dim | load | corpus embed | per-query p50 | per-query p95 | peak RSS |
|---|---|---|---|---|---|---|---|
| BAAI/bge-m3 (fp16, PyTorch/MPS) | 569M | 1024 | 2.0s | 7.3s (240 chunks) | 27 ms | 36 ms | 4.50 GB |
| gpahal/bge-m3-onnx-int8 (int8, ONNX/CPU) | — (ONNX) | 1024 | 3.6s | 74.2s (240 chunks) | 9 ms | 10 ms | 6.81 GB |
| Qwen/Qwen3-Embedding-0.6B (fp16, PyTorch/MPS) | 596M | 1024 | 8.9s | 22.4s (240 chunks) | 34 ms | 87 ms | 2.90 GB |

_per-query = single query encoded alone (online-realistic); BGE per-query includes dense **and** sparse in one pass. fp16 runs on MPS, int8 on CPU._

## 2. Quality — anchor recall@k (label-free)
Does the embedder surface each query's known source conversation into the top-k? Free ground truth, unbiased across embedders.

| feed-k | BGE fp16 dense | BGE int8 dense | Qwen3-0.6B dense | BGE fp16 hybrid | BGE int8 hybrid |
|---|---|---|---|---|---|
| 3 | 0.950 | 0.950 | 0.850 | 0.950 | 0.950 |
| 5 | 1.000 | 0.950 | 0.850 | 0.950 | 0.950 |
| 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## 3. Quality — graded recall@k (Sonnet grade ≥ 2)
Relevant = graded ≥2. **Coverage:** 0 of 276 (query,conv) pairs unjudged (treated as grade 0). Full judgment coverage.

| feed-k | BGE fp16 dense | BGE int8 dense | Qwen3-0.6B dense | BGE fp16 hybrid | BGE int8 hybrid |
|---|---|---|---|---|---|
| 3 | 0.805 | 0.815 | 0.768 | 0.763 | 0.763 |
| 5 | 0.925 | 0.887 | 0.838 | 0.883 | 0.889 |
| 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## 4. Top-10 set-membership divergence between arms

How different are the conversations each arm surfaces (set overlap, order-free)? The int8-vs-fp16 rows show how much quantization perturbs retrieval — near-1.0 Jaccard means quantization is effectively lossless for ranking.

| arm pair | mean overlap@10 (/10) | mean Jaccard | queries with identical set |
|---|---|---|---|
| BGE fp16 dense vs BGE int8 dense | 9.30 | 0.876 | 8/20 |
| BGE fp16 hybrid vs BGE int8 hybrid | 9.45 | 0.903 | 11/20 |
| BGE fp16 dense vs Qwen3-0.6B dense | 7.50 | 0.608 | 0/20 |
| BGE fp16 hybrid vs Qwen3-0.6B dense | 7.35 | 0.591 | 0/20 |
| BGE fp16 dense vs BGE fp16 hybrid | 8.55 | 0.756 | 1/20 |

### Anchor recall@10 by language
| lang | n | BGE fp16 dense | BGE int8 dense | Qwen3-0.6B dense | BGE fp16 hybrid | BGE int8 hybrid |
|---|---|---|---|---|---|---|
| en | 16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ja | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
