"""Sweep ef_search and EXACT_SELECTION_MAX across the full real query set."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import clync  # noqa: E402
import measure  # noqa: E402
import search  # noqa: E402


def eval_config(entries: list[dict], emb: dict, gold: dict, ef: int, exact_max: int) -> dict:
    search.HNSW_EF_SEARCH = ef
    search.EXACT_SELECTION_MAX = exact_max
    rows = []
    with clync.connect_pg() as con:
        for e in entries:
            obs = measure.observe(search, e, emb)
            dist = measure.exact_distances(e, emb, obs["legs"])
            ref = gold[e["id"]]
            measure.replay(con, obs["statements"])
            con.rollback()
            t0 = time.perf_counter()
            plan = measure.replay(con, obs["statements"], explain=True)
            wall_ms = (time.perf_counter() - t0) * 1000
            con.rollback()
            rows.append({
                "id": e["id"],
                "origin": e["origin"],
                "kwargs": e["kwargs"],
                "ms": plan["Execution Time"],
                "wall_ms": wall_ms,
                "legs": measure.legs_from_plan(plan["Plan"]),
                "recall_dense": measure.leg_recall(ref["legs"]["dense"], obs["legs"]["dense"], dist["dense"]),
                "recall_sparse": measure.leg_recall(ref["legs"]["sparse"], obs["legs"]["sparse"], dist["sparse"]),
                "recall_units": measure.unit_recall(ref["results"], obs["results"]),
                "exact": not any("hnsw.ef_search" in s for s, _ in obs["statements"]),
            })

    def vals(key, rs=rows):
        return [r[key] for r in rs if r[key] is not None]

    def summ(vs):
        if not vs:
            return {}
        return {
            "n": len(vs),
            "p50": round(statistics.median(vs), 1),
            "p95": round(measure.percentile(vs, 0.95), 1),
            "max": round(max(vs), 1),
        }

    hnsw_rows = [r for r in rows if not r["exact"]]
    exact_rows = [r for r in rows if r["exact"]]
    no_hnsw_dense = sum(
        1 for r in hnsw_rows
        if "dense" in r["legs"] and not any("hnsw" in ix or "exp_dense" in ix for ix in r["legs"]["dense"]["indexes"])
    )
    no_hnsw_sparse = sum(
        1 for r in hnsw_rows
        if "sparse" in r["legs"] and not any("hnsw" in ix or "exp_sparse" in ix for ix in r["legs"]["sparse"]["indexes"])
    )
    worst_rec = sorted([r for r in rows if r["recall_units"] is not None], key=lambda r: r["recall_units"])[:5]
    worst_ms = sorted(rows, key=lambda r: -r["ms"])[:5]
    return {
        "ef": ef,
        "exact_max": exact_max,
        "n_total": len(rows),
        "n_hnsw": len(hnsw_rows),
        "n_exact": len(exact_rows),
        "no_hnsw_dense": no_hnsw_dense,
        "no_hnsw_sparse": no_hnsw_sparse,
        "sql_ms": summ(vals("ms")),
        "hnsw_sql_ms": summ(vals("ms", hnsw_rows)),
        "exact_sql_ms": summ(vals("ms", exact_rows)),
        "dense_leg_hnsw_ms": summ([r["legs"]["dense"]["ms"] for r in hnsw_rows if "dense" in r["legs"]]),
        "sparse_leg_hnsw_ms": summ([r["legs"]["sparse"]["ms"] for r in hnsw_rows if "sparse" in r["legs"]]),
        "recall_units_mean": round(statistics.mean(vals("recall_units")), 4),
        "recall_units_hnsw_mean": round(statistics.mean(vals("recall_units", hnsw_rows)), 4),
        "recall_units_exact_min": min(vals("recall_units", exact_rows)),
        "recall_dense_hnsw": round(statistics.mean(vals("recall_dense", hnsw_rows)), 4),
        "recall_sparse_hnsw": round(statistics.mean(vals("recall_sparse", hnsw_rows)), 4),
        "worst_5_recall": [
            (r["id"], r["origin"], round(r["recall_units"], 3),
             round(r["recall_dense"] if r["recall_dense"] is not None else -1, 3),
             round(r["recall_sparse"] if r["recall_sparse"] is not None else -1, 3),
             r["exact"], r["kwargs"])
            for r in worst_rec
        ],
        "worst_5_ms": [
            (r["id"], r["origin"], round(r["ms"], 1), r["exact"], r["legs"], r["kwargs"])
            for r in worst_ms
        ],
    }


def main() -> None:
    entries, emb = measure.load_inputs()
    gold = json.loads(measure.ORACLE.read_text())
    results = []
    for ef, em in [(400, 20000), (800, 20000), (800, 10000), (800, 30000)]:
        res = eval_config(entries, emb, gold, ef, em)
        results.append(res)
        print(json.dumps(res, indent=1), flush=True)
    (measure.BENCH / "sweep-m32.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
