"""HNSW benchmark harness for clync hybrid search (ADR 0005).

Private inputs live OUTSIDE the repo, in `$CLYNC_BENCH_HOME` (default
`~/.local/share/clync/bench/`): the real query set and its cached embeddings. The
benchmark database is a clone of the live store selected with `CLYNC_PG_DB`
(never run experiments against the live database).

    extract   build queries.json from real recorded search calls
    embed     cache each query's BGE-M3 embedding (queries.emb.json)
    oracle    record exact (no-index) results + per-leg shortlists (oracle.json)
    run       time + score the current code path against the oracle
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BENCH = Path(os.environ.get("CLYNC_BENCH_HOME",
                            Path.home() / ".local/share/clync/bench"))
QUERIES = BENCH / "queries.json"
EMBEDDINGS = BENCH / "queries.emb.json"
ORACLE = BENCH / "oracle.json"
CC_PROJECTS = Path.home() / ".claude/projects"

SEARCH_TOOLS = ("mcp__clync__search_transcripts", "mcp__clync__search_history",
                "mcp__clync__search_insights")
TRANSCRIPT_ARGS = ("source", "project", "model", "repo", "worktree", "branch", "session",
                   "sender", "since", "until", "per_unit")


def _recorded_calls() -> list[tuple[str, dict]]:
    """Every (tool, input) a Claude Code session sent to a clync archive search tool."""
    pattern = "|".join(SEARCH_TOOLS)
    files = subprocess.run(
        ["rg", "-l", "--no-messages", rf'"name":"({pattern})"', str(CC_PROJECTS)],
        capture_output=True, text=True, check=False).stdout.split()
    calls = []
    for path in files:
        for line in open(path, errors="replace"):
            if "mcp__clync__search_" not in line:
                continue
            try:
                message = json.loads(line).get("message")
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                continue
            calls += [(b["name"], b["input"]) for b in message["content"]
                      if b.get("type") == "tool_use" and b.get("name") in SEARCH_TOOLS]
    return calls


def _int(value):
    return None if value is None else int(value)


def _transcript_kwargs(args: dict) -> dict | None:
    if not args.get("query", "").strip() or args.get("sort", "relevance") != "relevance":
        return None
    kw = {k: args[k] for k in TRANSCRIPT_ARGS if args.get(k) not in (None, "")}
    kw["topk"] = _int(args.get("limit")) or 10
    if "per_unit" in kw:
        kw["per_unit"] = int(kw["per_unit"])
    return kw


def extract(n_dig: int) -> None:
    """Distinct real queries, keeping their facets. The recall surfaces
    (search_history / search_insights) issue two hybrid_search calls each
    (dream.recall); a dream-dig probe carries `exclude_attempted` + the relevance
    floor (dream.candidates)."""
    import dream
    import clync
    seen: set[str] = set()
    entries: list[dict] = []

    def add(origin: str, query: str, kwargs: dict, group: str) -> None:
        key = json.dumps([origin, query, kwargs], sort_keys=True)
        if key not in seen:
            seen.add(key)
            entries.append({"id": f"q{len(entries):04d}", "group": group, "origin": origin,
                            "query": query, "kwargs": kwargs})

    for tool, args in _recorded_calls():
        query = args.get("query", "")
        if not query.strip():
            continue
        if tool.endswith("search_transcripts"):
            kw = _transcript_kwargs(args)
            if kw:
                add("transcripts", query, kw, f"t:{query}")
            continue
        limit = _int(args.get("limit")) or 10
        add("recall_insights", query,
            {"topk": limit * 4, "source": dream.DREAM_SOURCE,
             "min_dense_sim": dream.MIN_DENSE_SIM}, f"h:{query}")
        if tool.endswith("search_history"):
            add("recall_raw", query, {"topk": limit, "source": "all"}, f"h:{query}")
    with clync.connect_pg() as con:
        probes = con.execute(
            "SELECT DISTINCT ON (query) query, topic_id FROM dream_queries "
            "WHERE topic_id IS NOT NULL ORDER BY query, asked_at DESC").fetchall()
    step = max(1, len(probes) // n_dig)
    for row in probes[::step][:n_dig]:
        add("dig", row["query"],
            {"topk": 10, "source": "all", "exclude_attempted": row["topic_id"],
             "min_dense_sim": dream.MIN_DENSE_SIM}, f"d:{row['query']}")
    BENCH.mkdir(parents=True, exist_ok=True)
    QUERIES.write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    by_origin: dict[str, int] = {}
    for e in entries:
        by_origin[e["origin"]] = by_origin.get(e["origin"], 0) + 1
    faceted = sum(1 for e in entries if e["origin"] == "transcripts"
                  and set(e["kwargs"]) - {"topk", "per_unit"})
    cjk = sum(1 for e in entries if any(
        "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in e["query"]))
    print(json.dumps({"calls": len(entries),
                      "distinct_queries": len({e["query"] for e in entries}),
                      "by_origin": by_origin, "faceted_transcripts": faceted,
                      "cjk": cjk}, indent=1))


def embed() -> None:
    import search
    entries = json.loads(QUERIES.read_text())
    cache: dict[str, list] = {}
    for e in entries:
        if e["query"] not in cache:
            (qd,), (qs,) = search._embed([e["query"]], max_length=512)
            cache[e["query"]] = [qd, {str(k): v for k, v in qs.items()}]
    EMBEDDINGS.write_text(json.dumps(cache))
    print(f"embedded {len(cache)} distinct queries")




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract").add_argument("--dig", type=int, default=40)
    sub.add_parser("embed")
    sub.add_parser("oracle")
    run_p = sub.add_parser("run")
    run_p.add_argument("label")
    run_p.add_argument("--reps", type=int, default=3)
    run_p.add_argument("--every", type=int, default=1)
    e2e_p = sub.add_parser("e2e")
    e2e_p.add_argument("label")
    e2e_p.add_argument("--children", type=int, default=5)
    e2e_p.add_argument("--count", type=int, default=21)
    child_p = sub.add_parser("e2e-child")
    child_p.add_argument("--offset", type=int, default=0)
    child_p.add_argument("--count", type=int, default=21)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import measure
    handlers = {"oracle": lambda: measure.oracle(),
                "run": lambda: measure.run(args.label, args.reps, args.every),
                "e2e": lambda: measure.e2e(args.label, args.children, args.count),
                "e2e-child": lambda: measure.e2e_child(args.offset, args.count)}
    if args.cmd == "extract":
        extract(args.dig)
    elif args.cmd == "embed":
        embed()
    else:
        handlers[args.cmd]()


if __name__ == "__main__":
    main()
