# ADR 0005 — HNSW indexes and exact-slice routing for hybrid search

- **Status:** Accepted
- **Date:** 2026-11-23
- **Scope:** Local `clync` hybrid retrieval (`search.py`) and Dream recall coverage (`clync.py` / `dream.py`).
- **Benchmark harness:** `experiments/hnsw_bench/` (reads private query corpus from `~/.local/share/clync/bench/`).

## Context

[ADR 0003](0003-postgres-only-and-cc-ingest.md) stored BGE-M3 dense (`vector(1024)`) and
learned-sparse (`sparsevec(250002)`) embeddings in Postgres `chunks` with no vector index.
At ~200,000 chunks, every unfiltered `search_history` or `search_transcripts` call ran full
sequential scans on both legs (~390–445 ms warm SQL, up to ~2.5 s cold), and `dream.recall`
added a ~290–330 ms sequential scan over `messages` for its unread-units coverage count.

We measured HNSW parameters and query routing against an exact (no-index) oracle built from
**770 real recorded search calls** (629 distinct queries: 471 `search_transcripts` — 446
faceted, 131 `recall_insights`, 128 `recall_raw`, 40 `dream` dig probes; 7 CJK) on a clone
of the ~200k-chunk corpus.

## Decision

### 1. HNSW on `halfvec(1024)` (dense) and `sparsevec(250002)` (sparse)

Create two HNSW indexes in `search.ensure_index_schema()`:

- `idx_chunks_dense_hnsw`: `USING hnsw ((dense::halfvec(1024)) halfvec_cosine_ops)`
- `idx_chunks_sparse_hnsw`: `USING hnsw (sparse sparsevec_ip_ops)`

The full-precision `vector(1024)` column stays untouched so `dense_sim` scoring, `min_dense_sim`
filtering, and exact-slice scans preserve fp32 precision while the HNSW graph uses half-precision
storage (**454 MB** vs **1,368 MB** for `vector(1024)` — 3× smaller with equal or higher recall).
`sparsevec` HNSW is **194 MB**.

Parameters: **`m = 32`, `ef_construction = 200`, `hnsw.ef_search = 800`,
`hnsw.iterative_scan = 'strict_order'`**.

- `m = 32, ef_construction = 200` eliminated the 0.0 min-recall tail seen on sparse HNSW at
  `m = 16`.
- `hnsw.ef_search = 800` is checked against `PRE_TOPK..1000` (`PGVECTOR_EF_SEARCH_MAX`): below
  `PRE_TOPK` a leg returns short; above `1000` pgvector silently resets `ef_search` to `40`.
- `_sparse_literal()` enforces pgvector's 1,000-nnz HNSW cap (`SPARSE_MAX_NNZ = 1000`),
  keeping the heaviest weights and logging the dropped count to stderr.
- `_hnsw_build_memory_kb()` sizes `maintenance_work_mem` per index before `CREATE INDEX` so
  builds stay in memory (verified 0 spill notices at 900 MB on 200k rows: 306 s dense, 166 s
  sparse). Incremental 50-chunk delete+reinsert moves from 57 ms to 1,456 ms (~29 ms/chunk).

### 2. Route small facet selections to exact scan (`EXACT_SELECTION_MAX = 30000`)

Before running the hybrid query, `_selection_is_small()` runs a bounded count (`LIMIT 30001`).

- **Selection ≤ 30,000 rows (`exact=True`):** Bypass HNSW using `+ 0` on the float8 distance
  expressions (`(dense <=> %(qd)s::vector) + 0`, `(sparse <#> %(qs)s::sparsevec) + 0`). Using
  `+ 0` instead of a `MATERIALIZED` CTE avoids temp-file spills of wide TOASTed vectors
  (which caused a 1,316 ms tail at 22k rows) while preserving btree facet scans.
- **Selection > 30,000 rows (`exact=False`):** Set transaction-local `hnsw.ef_search = 800`,
  `hnsw.iterative_scan = 'strict_order'`, and disable `seqscan`, `bitmapscan`, and `sort` so
  the planner cannot fall back to a btree+sort scan on filtered slices. `scored AS MATERIALIZED`
  (≤ 100 rows) keeps `enable_sort = off` from forcing a full `idx_chunks_unit` scan on the
  outer `WindowAgg`.

### 3. pgvector version floor and vacuum warning

- Hard floor: **`PGVECTOR_MIN_VERSION = (0, 8, 0)`** (required for `hnsw.iterative_scan` and
  `sparsevec` HNSW; `ensure_index_schema` upgrades installed extensions and raises below 0.8.0).
- Warning threshold: **`PGVECTOR_SAFE_VACUUM_VERSION = (0, 8, 4)`** (0.8.3/0.8.4 fixed HNSW
  graph corruption under `VACUUM`). Both Homebrew PG17 bottles and Aurora PostgreSQL 17.10 ship
  pgvector `0.8.2` (Homebrew's `0.8.7` bottle dropped PG17), so clync runs on `0.8.2`, emits a
  stderr warning when building HNSW indexes, and surfaces the note in `clync doctor` (pointing
  to `clync index --full` if vacuum ever triggers an HNSW error) without repeating on every
  read command.

### 4. Partial covering index for `dream.recall` coverage count

`clync.RAW_SCHEMA` adds `idx_messages_nonempty ON messages(unit_id, sender) WHERE btrim(coalesce(embed_text, text, '')) <> ''`
(2.2 MB, predicate defined once in `clync.NONEMPTY_MSG_SQL` and reused by `dream.py`). This
replaces a 290–330 ms sequential scan + 4-batch hash semi-join on `messages` with a ~33–42 ms
Index-Only Scan during every `search_history` call.

## Measured Results (200k-chunk clone, 770 real queries)

### Sweep across `ef_search` and `EXACT_SELECTION_MAX` (`m = 32, ef_construction = 200`)

| Config | HNSW / exact | SQL p50 / p95 / max (ms) | Unit recall@10 mean (HNSW subset) | Dense / sparse leg recall@50 | Fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (no HNSW, 193-query subset) | 0 / 193 | 53.6 / 410.8 / 445.8 | 1.0000 | 1.0000 / 1.0000 | — |
| `ef=400`, `exact_max=20000` | 345 / 425 | 23.8 / 92.0 / 149.3 | 0.9608 (0.9203) | 0.9528 / 0.9205 | 0 / 0 |
| `ef=800`, `exact_max=10000` | 364 / 406 | 59.4 / 99.2 / 129.0 | 0.9751 (0.9521) | 0.9734 / 0.9471 | 0 / 0 |
| `ef=800`, `exact_max=20000` | 345 / 425 | 60.1 / 105.2 / 348.8 | 0.9763 (0.9519) | 0.9738 / 0.9455 | 0 / 0 |
| `ef=800`, `exact_max=30000` (`MATERIALIZED` exact) | 263 / 507 | 60.5 / 194.4 / 1316.0 | 0.9908 (0.9758) | 0.9864 / 0.9777 | 0 / 0 |
| **`ef=800`, `exact_max=30000` (`+ 0` fence, shipped)** | **263 / 507** | **54.1 / 99.6 / 119.0** | **0.9906 (0.9758)** | **0.9864 / 0.9777** | **0 / 0** |

On the shipped configuration:
- HNSW-served queries (n=263): SQL p50 **72.5 ms**, p95 **105.0 ms**, max **119.0 ms**
  (dense leg p50 43.6 ms / max 78.0 ms; sparse leg p50 25.2 ms / max 44.2 ms).
- Exact-served queries (n=507): SQL p50 **48.5 ms**, p95 **97.5 ms**, max **108.9 ms**. Every
  exact query matches the oracle's dense and sparse shortlists identically (one query showed
  0.90 unit recall only from `now()` recency-decay drift at rank 10 between oracle capture and
  replay 5 hours later).
- Worst 5 HNSW unit recalls (0.60–0.71): caused by either low-weight sparse queries where inner
  product scores cluster near `-0.08..-0.02` (`q0326`, `q0702`, `q0768`, dense recall still
  0.94–1.00) or `per_unit > 1` slices where near-tied RRF scores swap one borderline unit at
  rank 10 (`q0114`, `q0234`, leg recalls 0.98–1.00).

### End-to-end `dream.recall` (`search_history`: embed + insights + raw + coverage)

Measured across 5 fresh child processes (5 cold + 100 warm real queries):

| Stage | Cold total p50 / max (ms) | Warm total p50 / p95 / max (ms) | Warm SQL p50 / p95 / max (ms) |
| --- | ---: | ---: | ---: |
| Before (no HNSW, no `idx_messages_nonempty`) | 5,339 / 9,637 | 743.7 / 789.5 / 1,140.9 | 392.1 / 427.8 / 646.8 |
| After HNSW only | 5,807 / 9,520 | 466.9 / 638.1 / 1,164.4 | 104.7 / 222.0 / 587.6 |
| **After HNSW + `idx_messages_nonempty` (shipped)** | **4,735 / 4,795** | **174.5 / 234.2 / 453.6** | **100.5 / 137.7 / 169.2** |

Warm `search_history` p50 drops **4.3×** (743.7 ms → 174.5 ms) and max drops **2.5×**
(1,140.9 ms → 453.6 ms, with remaining max dominated by occasional MPS embed spikes up to
285 ms; warm SQL max is 169.2 ms).

## Consequences

- Disk footprint increases by ~650 MB on a 200k-chunk store (`454 MB` dense HNSW + `194 MB`
  sparse HNSW + `2.2 MB` partial messages index).
- First upgrade via `clync.ensure_cluster()` / `search.ensure_index_schema()` builds the two
  HNSW indexes once (~8 min total on 200k chunks); subsequent runs are idempotent no-ops.
- When Homebrew ships pgvector ≥ 0.8.4 for PG17 (or clync moves to PG18+), the stderr vacuum
  warning clears automatically without code changes.
