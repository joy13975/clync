"""Measurement half of the HNSW benchmark: oracle recording, replay timing, recall."""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from bench import BENCH, EMBEDDINGS, ORACLE, QUERIES


def load_inputs() -> tuple[list[dict], dict]:
    entries = json.loads(QUERIES.read_text())
    raw = json.loads(EMBEDDINGS.read_text())
    return entries, {q: ([v[0]], [{int(k): w for k, w in v[1].items()}])
                     for q, v in raw.items()}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    return {"n": len(values), "p50": round(statistics.median(values), 1),
            "p95": round(percentile(values, 0.95), 1), "max": round(max(values), 1)}


class Recorder:
    """A connection proxy recording every statement `hybrid_search` executes, so the
    benchmark replays the code's REAL SQL and settings, not a restated copy."""

    def __init__(self, real, log):
        self.real, self.log = real, log

    def __enter__(self):
        self.real.__enter__()
        return self

    def __exit__(self, *exc):
        return self.real.__exit__(*exc)

    def execute(self, sql, params=None):
        self.log.append((sql, params))
        return self.real.execute(sql, params)

    def __getattr__(self, name):
        return getattr(self.real, name)


def _leg_sql(main_sql: str, leg: str) -> str:
    """The recorded main statement narrowed to one leg's shortlist (CTE `d` / `s`)."""
    return main_sql.rsplit("\nSELECT", 1)[0] + f"\nSELECT chunk_id, rnk FROM {leg}"


def _main_index(statements: list) -> int:
    return max(i for i, (sql, _) in enumerate(statements) if sql.lstrip().startswith("WITH"))


def observe(search, entry: dict, embeddings: dict) -> dict:
    """Run one recorded call through the code under test; return its statements, its
    final rows (topk widened to 100 so recall@10 can be judged by score), and each leg's
    shortlist chunk ids."""
    import clync
    log: list = []
    real_connect = search.connect_pg
    search.connect_pg = lambda: Recorder(real_connect(), log)
    try:
        results = search.hybrid_search(
            entry["query"], query_embedding=embeddings[entry["query"]],
            **dict(entry["kwargs"], topk=100))
    finally:
        search.connect_pg = real_connect
    main = _main_index(log)
    legs = {}
    with clync.connect_pg() as con:
        for name, cte in (("dense", "d"), ("sparse", "s")):
            if not re.search(rf"(?:WITH|,)\s*{cte} AS \(", log[main][0]):
                legs[name] = []
                continue
            for sql, params in log[:main]:
                con.execute(sql, params)
            legs[name] = [r["chunk_id"] for r in con.execute(
                _leg_sql(log[main][0], cte), log[main][1]).fetchall()]
            con.rollback()
    return {"statements": log, "results": [(r["unit_id"], r["score"]) for r in results],
            "legs": legs}


def exact_distances(entry: dict, embeddings: dict, ids: dict) -> dict:
    """Full-precision distance of each shortlisted chunk to the query, whatever index
    (or halfvec approximation) ranked it. Recall is judged on these, never on ids."""
    import clync
    import search
    (qd,), (qs,) = embeddings[entry["query"]]
    out = {"dense": {}, "sparse": {}}
    with clync.connect_pg() as con:
        out["dense"] = {r["chunk_id"]: r["d"] for r in con.execute(
            "SELECT chunk_id, dense <=> %s::vector AS d FROM chunks WHERE chunk_id = ANY(%s)",
            (search._dense_literal(qd), ids["dense"])).fetchall()}
        try:
            sparse_lit = search._sparse_literal(qs)
        except ValueError:
            return out
        out["sparse"] = {r["chunk_id"]: r["d"] for r in con.execute(
            "SELECT chunk_id, sparse <#> %s::sparsevec AS d FROM chunks "
            "WHERE chunk_id = ANY(%s)", (sparse_lit, ids["sparse"])).fetchall()}
    return out


def has_ann_index() -> bool:
    import clync
    with clync.connect_pg() as con:
        return con.execute("SELECT 1 FROM pg_indexes WHERE tablename='chunks' AND "
                           "indexdef ILIKE '%%hnsw%%'").fetchone() is not None


def oracle() -> None:
    """Exact results, recorded on a database with NO vector index (every leg a full scan)."""
    import search
    if has_ann_index():
        raise SystemExit("record the oracle on a database without ANN indexes")
    entries, embeddings = load_inputs()
    out = {}
    for n, e in enumerate(entries, 1):
        obs = observe(search, e, embeddings)
        out[e["id"]] = {"results": obs["results"], "ranked": obs["legs"],
                        "legs": exact_distances(e, embeddings, obs["legs"])}
        print(f"oracle {n}/{len(entries)}", file=sys.stderr, flush=True)
    ORACLE.write_text(json.dumps(out))


def leg_recall(oracle_leg: dict, arm_ids: list[str], arm_dist: dict) -> float | None:
    """Fraction of the arm's shortlist no farther than the oracle's cut-off, so ties at
    the cut-off cannot make recall unstable."""
    if not oracle_leg:
        return None
    cutoff = max(oracle_leg.values())
    return sum(1 for c in arm_ids if arm_dist[c] <= cutoff + 1e-9) / len(oracle_leg)


def _best_by_unit(rows: list) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for u, s in rows:
        if u not in best or s > best[u]:
            best[u] = s
    return sorted(best.items(), key=lambda kv: -kv[1])


def unit_recall(oracle_rows: list, arm_rows: list, k: int = 10) -> float | None:
    """Recall of the arm's top-k distinct units against the oracle's, judged on the
    oracle's highest score per unit so per_unit > 1 rows cannot overwrite a unit's best
    score and RRF ties at the cut-off stay stable."""
    if not oracle_rows:
        return None
    oracle_u = _best_by_unit(oracle_rows)
    arm_u = _best_by_unit(arm_rows)
    score = dict(oracle_u)
    want = min(k, len(oracle_u))
    cutoff = oracle_u[want - 1][1]
    return sum(1 for u, _ in arm_u[:want] if score.get(u, -1) >= cutoff - 1e-12) / want


def _values(node: dict, key: str):
    if key in node:
        yield node[key]
    for child in node.get("Plans", []):
        yield from _values(child, key)


def legs_from_plan(plan: dict) -> dict:
    """Per-leg server time (ms) and the indexes each leg scanned, from EXPLAIN JSON.
    A leg is the outermost LIMIT whose subtree orders by its distance operator."""
    found: dict = {}

    def walk(node):
        if node["Node Type"] == "Limit":
            text = json.dumps(node)
            for leg, op in (("dense", "<=>"), ("sparse", "<#>")):
                if leg not in found and op in text:
                    found[leg] = {"ms": node["Actual Total Time"],
                                  "indexes": sorted(set(_values(node, "Index Name"))),
                                  "nodes": sorted(set(_values(node, "Node Type")))}
        for child in node.get("Plans", []):
            walk(child)

    walk(plan)
    return found


def replay(con, statements: list, *, explain: bool = False):
    main = _main_index(statements)
    for i, (sql, params) in enumerate(statements):
        if i == main and explain:
            return con.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql,
                               params).fetchone()["QUERY PLAN"][0]
        con.execute(sql, params)


def run(label: str, reps: int, every: int = 1) -> None:
    import clync
    import search
    entries, embeddings = load_inputs()
    entries = entries[::every]
    gold = json.loads(ORACLE.read_text())
    rows = []
    with clync.connect_pg() as con:
        for n, e in enumerate(entries, 1):
            obs = observe(search, e, embeddings)
            dist = exact_distances(e, embeddings, obs["legs"])
            ref = gold[e["id"]]
            replay(con, obs["statements"])      # warm-up, discarded
            con.rollback()
            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                replay(con, obs["statements"])
                times.append((time.perf_counter() - t0) * 1000)
                con.rollback()
            plan = replay(con, obs["statements"], explain=True)
            con.rollback()
            rows.append({
                "id": e["id"], "origin": e["origin"], "query": e["query"],
                "kwargs": e["kwargs"], "ms": statistics.median(times),
                "legs": legs_from_plan(plan["Plan"]),
                "recall_dense": leg_recall(ref["legs"]["dense"], obs["legs"]["dense"],
                                           dist["dense"]),
                "recall_sparse": leg_recall(ref["legs"]["sparse"], obs["legs"]["sparse"],
                                            dist["sparse"]),
                "recall_units": unit_recall(ref["results"], obs["results"]),
                "oracle_dense_rows": len(ref["legs"]["dense"])})
            print(f"{label} {n}/{len(entries)} {rows[-1]['ms']:.0f}ms", file=sys.stderr,
                  flush=True)
    (BENCH / f"run-{label}.json").write_text(json.dumps(rows, ensure_ascii=False))
    report(rows)


def report(rows: list[dict]) -> None:
    def vals(key, rs=rows):
        return [r[key] for r in rs if r[key] is not None]

    def leg_ms(leg):
        return [r["legs"][leg]["ms"] for r in rows if leg in r["legs"]]

    out = {"sql_ms": summarize(vals("ms")),
           "dense_leg_ms": summarize(leg_ms("dense")),
           "sparse_leg_ms": summarize(leg_ms("sparse")),
           "recall_dense_mean": round(statistics.mean(vals("recall_dense")), 4),
           "recall_sparse_mean": round(statistics.mean(vals("recall_sparse")), 4),
           "recall_units_mean": round(statistics.mean(vals("recall_units")), 4),
           "by_origin": {}}
    for origin in sorted({r["origin"] for r in rows}):
        rs = [r for r in rows if r["origin"] == origin]
        out["by_origin"][origin] = {
            "sql_ms": summarize(vals("ms", rs)),
            "recall_units_mean": round(statistics.mean(vals("recall_units", rs)), 4)}
    print(json.dumps(out, indent=1))


def e2e_child(offset: int, count: int) -> None:
    """One fresh process: `count` search_history-shaped recalls (embed + two hybrid_search
    calls, via `dream.recall`). The first is the cold call (imports, model load, first
    connection); the rest are warm. Prints JSON."""
    import clync
    import dream
    import search
    queries = sorted({e["query"] for e in json.loads(QUERIES.read_text())
                      if e["origin"] == "recall_raw"})
    picked = [queries[(offset + 7 * i) % len(queries)] for i in range(count)]
    spent = {"embed": 0.0, "sql": 0.0}

    def timed(fn, bucket):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                spent[bucket] += (time.perf_counter() - t0) * 1000
        return wrapper

    search._embed = timed(search._embed, "embed")
    search.hybrid_search = timed(search.hybrid_search, "sql")
    calls = []
    for q in picked:
        spent.update(embed=0.0, sql=0.0)
        con = clync.connect()
        t0 = time.perf_counter()
        dream.recall(con, q, limit=10, include_evidence=True, include_raw=True)
        calls.append({"total_ms": (time.perf_counter() - t0) * 1000, **spent})
        con.close()
    print(json.dumps({"calls": calls}))


def e2e(label: str, children: int, count: int) -> None:
    cold, warm = [], []
    for k in range(children):
        out = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "bench.py"), "e2e-child",
             "--offset", str(k * 3), "--count", str(count)],
            capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
        calls = json.loads(out)["calls"]
        cold.append(calls[0])
        warm += calls[1:]
    result = {"cold_total_ms": [round(c["total_ms"]) for c in cold],
              "cold": {k: summarize([c[k] for c in cold]) for k in ("total_ms", "embed", "sql")},
              "warm": {k: summarize([c[k] for c in warm]) for k in ("total_ms", "embed", "sql")}}
    (BENCH / f"e2e-{label}.json").write_text(json.dumps(result))
    print(json.dumps(result, indent=1))
