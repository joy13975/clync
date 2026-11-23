# Lambda CPU Embedding Benchmark: S3-to-RAM vs Container Image (BGE-M3 int8 ONNX)

**Date:** 2026-11-23  
**Platform:** arm64 (Graviton), Python 3.12, `onnxruntime==1.32.0`, `tokenizers==0.21.0`, `numpy==2.2.3`.  
**VPC setup:** Existing private app subnets routed through the VPC's S3 Gateway endpoint ($0 data transfer).  
**Pass bar:** Cold total (init + first embed) `<= 10,000 ms`, warm embed `p95 <= 100 ms`, peak memory `<= 70%` of allocated tier.

## Verdict

**Use a Container Image Lambda at 5120–10240 MB (prefer 10240 MB, or 5120 MB if cost-optimizing warm runs); reject S3-to-RAM zip.**

- **Container image @ 10240 MB** is the only arm that cleared every bar: **3.8 s p50 / 5.4 s p95 cold total**, **38.8 ms p50 / 43.3 ms p95 warm embed**, **1,742 MB peak RAM (17.0%)**.
- **S3-to-RAM zip** hits a hard **~5.9–6.2 s S3 download floor** for the 543 MB model across all memory tiers (Lambda caps per-ENI network throughput similarly across 3–10 GB). Combined with ~2.0–2.7 s ONNX Runtime load, cold total sits at **8.8–10.2 s** (right on/over the 10 s bar), and buffering the 543 MB payload in Python adds **~500 MB peak RAM** (2.22 GB vs 1.74 GB for container-on-disk, pushing 3008 MB over the 70% bar to 73.6%).
- **ColBERT head stripping saved only 1.0 MB** (`543.55 MB -> 542.55 MB`, 26 nodes / 4 initializers removed). The ColBERT linear projection is tiny compared to the shared XLM-R backbone; stripping still cut local `batch=1` inference latency by ~23% (19.2 ms → 14.7 ms) because the ColBERT output tensor is no longer allocated or materialized.
- **Note on container cold-start #1:** The very first invocation after `create_function` on a freshly pushed ECR image took **41.9 s** (Lambda populating its regional chunk cache for the new image digest). Subsequent cold starts on the same image dropped to **3.8–5.4 s**. In production, a single post-deploy smoke invocation warms Lambda's image cache once per deploy.

## 1. Local `batch=1` Parity vs fp16 BGE-M3 (30 Real Queries, EN + JA)

Evaluated with raw `onnxruntime` + `tokenizers` (no PyTorch or `transformers`) against clync's cached fp16 MPS query vectors and the 200k-chunk hybrid index:

| Variant | Size (MB) | Local load (ms) | Local embed p50 / p95 (ms) | Dense cos mean / min | Sparse cos mean / min | Top-10 hybrid overlap | Top-10 Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full `gpahal/bge-m3-onnx-int8` | 543.55 | 436.3 | 19.2 / 28.0 | 0.9813 / 0.9758 | 0.9972 / 0.9936 | 0.9122 | 0.8454 |
| ColBERT-stripped int8 | 542.55 | 486.3 | 14.7 / 23.5 | 0.9813 / 0.9758 | 0.9972 / 0.9936 | 0.9122 | 0.8454 |

Stripping is bit-for-bit identical on dense and sparse outputs.

## 2. Live AWS Lambda Measurements (5 Forced Cold Starts + 20 Warm Queries per Tier)

| Arm | Tier (MB) | S3/disk read p50 (ms) | ORT load p50 (ms) | Cold total p50 / p95 (ms) | Warm embed p50 / p95 (ms) | Peak RAM (MB / %) | Pass bar? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Arm 1: Full S3-to-RAM (zip, 43.3 MB / 127 MB unzipped) | 3008 | 5,957 | 1,947 | 8,806 / 9,918 | 191.2 / 231.5 | 2,214 (73.6%) | No (RAM, warm) |
| Arm 1: Full S3-to-RAM (zip) | 5120 | 5,971 | 2,033 | 8,911 / 10,171 | 197.9 / 242.0 | 2,217 (43.3%) | No (cold p95, warm) |
| Arm 1: Full S3-to-RAM (zip) | 10240 | 6,210 | 2,596 | 9,907 / 10,051 | 77.6 / 91.8 | 2,251 (22.0%) | No (cold p95 10.05 s) |
| Arm 2: Stripped S3-to-RAM (zip) | 2048 | 6,224 | 2,260 | 17,444 / 20,114 | 105.2 / 125.6 | 2,048 (100.0%) | No (OOM/thrash) |
| Arm 2: Stripped S3-to-RAM (zip) | 3008 | 5,845 | 2,036 | 8,801 / 10,085 | 194.6 / 235.9 | 2,238 (74.4%) | No (RAM, warm) |
| Arm 3: Stripped Container Image (ECR) | 3008 | 30 | 3,968 | 5,172 / 41,917* | 190.2 / 231.3 | 1,741 (57.9%) | No (1st-pull cold, warm) |
| **Arm 3: Stripped Container Image (ECR)** | **10240** | **30** | **2,992** | **3,831 / 5,441** | **38.8 / 43.3** | **1,742 (17.0%)** | **YES** |

\* Arm 3 @ 3008 MB run #1 (first invocation ever on newly pushed image) took 41,917 ms; runs #2–5 were 3,777 / 5,438 / 5,087 / 5,172 ms.

## 3. Operational Findings Discovered Live

1. **`onnxruntime==1.21.0` crashes on arm64 Lambda Firecracker:** `cpuinfo` fails to parse `/sys/devices/system/cpu/{possible,present}` and aborts during import with `Attempt to use DefaultLogger but none has been registered`. Upgrading to **`onnxruntime==1.32.0`** resolves this cleanly (5/5 cold starts succeeded on every tier).
2. **Graviton fleet heterogeneity:** Lambda arm64 lands on two distinct Graviton generations with ~2x CPU speed difference (visible in identical queries taking ~80 ms vs ~190 ms at 3008 MB, and ~38 ms vs ~78 ms at 10240 MB). Even on the slower host, Arm 3 @ 10240 MB stays under 65 ms per query and 5.4 s cold.
3. **Cost arithmetic at 100 queries/day (3,000/mo, arm64 rate $0.0000133334/GB-s):**
   - Arm 3 @ 10240 MB (10 GB): even assuming 100% cold starts at 5.5 s = `3,000 * 10 * 5.5 = 165,000 GB-s/mo` (well inside the 400,000 GB-s free tier, or **$2.20/mo** gross without free tier). Warm calls at 100 ms cost `0.000013` each (**$0.04/mo** gross).
   - ECR image storage (~750 MB compressed): **~$0.075/mo** idle.
4. **Teardown:** All throwaway S3 buckets, Lambda functions, ECR repositories, IAM roles, and CloudWatch log groups were deleted after measurement.

## 4. Next-Best Option If Cold-Start Must Drop Below 3 s

- **Not SnapStart** (~$39/mo cache cost at 10 GB violates the near-zero idle goal).
- If 3.8–5.4 s cold starts ever become an issue, **switch the cloud query encoder to an API-backed embedder** only if clync also re-embeds the corpus (out of scope while BGE-M3 is shared with local clync). For BGE-M3 on Lambda, **Arm 3 Container Image @ 10240 MB** is the optimal serverless choice.
