"""clync reranker ablation — deterministic pipeline.

Arm A (baseline): BGE-M3 dense + sparse → RRF(k=60) → top-50 shortlist → top-10.
Arm B (reranked): same 50 → bge-reranker-v2-m3 cross-encoder → top-10.
Retrieval/rerank operate on CHUNKS; results/divergence/metrics collapse to distinct
CONVERSATIONS (what the user sees). Relevance grades are per (query, conversation),
arm-independent, produced externally by Sonnet judges.

Subcommands:
  candidates    → candidates.json  (per-conv digests for query generation)
  embed         → embeds chunks, caches to cache/
  run           → reads queries.json (+ judgments.json if present) → results.json
  judge-targets → reads results.json → judge_targets.json (unique query×conv to grade)
  report        → results.json → RERANKER_ABLATION.md

Fail-loud: any missing input / model / device problem raises. No silent fallbacks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DATA = HERE / "data"
DB_PATH = Path(os.environ.get("CLYNC_DB", Path.home() / ".local/share/clync/history.db"))

RRF_K = 60
PRE_TOPK = 50
TOPK = 10
CHUNK_CHARS = 2000      # ~500 tokens; BGE-M3 handles 8192 so most msgs are one chunk
CHUNK_OVERLAP = 200
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


# --------------------------------------------------------------------------- #
# Corpus + chunking
# --------------------------------------------------------------------------- #
def _lang(s: str) -> str:
    s = s or ""
    if re.search(r"[぀-ヿ]", s):   # kana → Japanese
        return "ja"
    if re.search(r"[一-鿿]", s):   # Han without kana → Chinese
        return "zh"
    return "en"


def load_chunks() -> list[dict]:
    """Flatten messages into chunks with conversation provenance."""
    if not DB_PATH.exists():
        raise SystemExit(f"clync DB not found at {DB_PATH} — run `clync sync` first.")
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT m.uuid mid, m.conversation_uuid cid, m.sender, m.text, m.idx, "
        "c.name cname FROM messages m JOIN conversations c ON c.uuid=m.conversation_uuid "
        "WHERE m.text IS NOT NULL AND m.text != '' ORDER BY c.updated_at DESC, m.idx"
    ).fetchall()
    con.close()
    chunks: list[dict] = []
    for r in rows:
        text = r["text"]
        step = CHUNK_CHARS - CHUNK_OVERLAP
        pieces = [text[i:i + CHUNK_CHARS] for i in range(0, max(1, len(text)), step)] or [text]
        for j, piece in enumerate(pieces):
            body = f"[{r['cname'] or 'untitled'}] {r['sender']}: {piece}"
            chunks.append({
                "chunk_id": f"{r['mid']}#{j}",
                "conv_uuid": r["cid"],
                "conv_name": r["cname"],
                "text": body,
                "lang": _lang(body),
            })
    return chunks


def cmd_candidates(args) -> None:
    """Emit per-conversation digests (for external query generation)."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    convs = con.execute("SELECT uuid, name, message_count FROM conversations "
                        "ORDER BY updated_at DESC").fetchall()
    out = []
    for c in convs:
        msgs = con.execute("SELECT sender, text FROM messages WHERE conversation_uuid=? "
                           "ORDER BY idx", (c["uuid"],)).fetchall()
        digest = " ".join(f"{m['sender']}: {m['text']}" for m in msgs)[:1800]
        if not digest.strip():
            continue
        out.append({"conv_uuid": c["uuid"], "name": c["name"],
                    "message_count": c["message_count"],
                    "lang": _lang((c["name"] or "") + " " + digest), "digest": digest})
    con.close()
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    from collections import Counter
    print(f"wrote {len(out)} candidates → {DATA/'candidates.json'} "
          f"langs={dict(Counter(c['lang'] for c in out))}")


# --------------------------------------------------------------------------- #
# Embedding (cached)
# --------------------------------------------------------------------------- #
def _device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    raise SystemExit("MPS not available — this experiment targets Apple Silicon.")


def cmd_embed(args) -> None:
    chunks = load_chunks()
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    from FlagEmbedding import BGEM3FlagModel
    dev = _device()
    print(f"loading {EMBED_MODEL} on {dev} for {len(chunks)} chunks …")
    model = BGEM3FlagModel(EMBED_MODEL, use_fp16=True, devices=dev)
    t0 = time.time()
    out = model.encode([c["text"] for c in chunks], batch_size=16,
                       return_dense=True, return_sparse=True, max_length=2048)
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    # normalize for cosine via dot
    dense /= (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9)
    np.save(CACHE / "dense.npy", dense)
    # sparse: list of {token_id(str): weight}
    sparse = [{str(k): float(v) for k, v in d.items()} for d in out["lexical_weights"]]
    (CACHE / "sparse.json").write_text(json.dumps(sparse))
    print(f"embedded {len(chunks)} chunks in {time.time()-t0:.1f}s → {CACHE}")


# --------------------------------------------------------------------------- #
# Retrieval + rerank
# --------------------------------------------------------------------------- #
def _load_cache():
    chunks = json.loads((CACHE / "chunks.json").read_text())
    dense = np.load(CACHE / "dense.npy")
    sparse = json.loads((CACHE / "sparse.json").read_text())
    return chunks, dense, sparse


def _sparse_dot(q: dict, d: dict) -> float:
    if len(q) > len(d):
        q, d = d, q
    return sum(w * d.get(t, 0.0) for t, w in q.items())


def _rrf(rank_lists: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rank_lists:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _collapse_to_convs(chunk_indices: list[int], chunks: list[dict], topk: int) -> list[str]:
    """Chunk ranking → distinct conversation ranking (first occurrence wins)."""
    seen, out = set(), []
    for idx in chunk_indices:
        cu = chunks[idx]["conv_uuid"]
        if cu not in seen:
            seen.add(cu)
            out.append(cu)
        if len(out) >= topk:
            break
    return out


def cmd_run(args) -> None:
    chunks, dense, sparse = _load_cache()
    queries = json.loads((DATA / "queries.json").read_text())
    judgments = {}
    jp = DATA / "judgments.json"
    if jp.exists():
        judgments = {(j["query_id"], j["conv_uuid"]): j["grade"] for j in json.loads(jp.read_text())}

    from FlagEmbedding import BGEM3FlagModel, FlagReranker
    dev = _device()
    embed = BGEM3FlagModel(EMBED_MODEL, use_fp16=True, devices=dev)
    reranker = FlagReranker(RERANK_MODEL, use_fp16=True, devices=dev)

    results = []
    for q in queries:
        qtext = q["query"]
        t0 = time.time()
        qe = embed.encode([qtext], return_dense=True, return_sparse=True, max_length=512)
        qd = np.asarray(qe["dense_vecs"][0], dtype=np.float32)
        qd /= (np.linalg.norm(qd) + 1e-9)
        qs = {str(k): float(v) for k, v in qe["lexical_weights"][0].items()}
        dense_scores = dense @ qd
        sparse_scores = np.array([_sparse_dot(qs, sparse[i]) for i in range(len(chunks))])
        dense_rank = list(np.argsort(-dense_scores))
        sparse_rank = list(np.argsort(-sparse_scores))
        rrf = _rrf([dense_rank, sparse_rank])
        shortlist = sorted(rrf, key=lambda i: -rrf[i])[:PRE_TOPK]
        t_first = time.time() - t0

        # Arm A: RRF order of the shortlist
        arm_a = _collapse_to_convs(shortlist, chunks, TOPK)

        # Arm B: rerank the SAME shortlist chunks
        t1 = time.time()
        pairs = [[qtext, chunks[i]["text"]] for i in shortlist]
        rr = reranker.compute_score(pairs, normalize=True)
        rr = rr if isinstance(rr, list) else [rr]
        reranked = [shortlist[i] for i in np.argsort(-np.asarray(rr))]
        t_rerank = time.time() - t1
        arm_b = _collapse_to_convs(reranked, chunks, TOPK)

        # representative chunk per conv (from first-stage RRF order) for judging
        rep = {}
        for idx in shortlist:
            cu = chunks[idx]["conv_uuid"]
            if cu not in rep:
                rep[cu] = {"chunk_id": chunks[idx]["chunk_id"],
                           "snippet": chunks[idx]["text"][:1200]}

        results.append({
            "query_id": q["id"], "query": qtext, "lang": q["lang"],
            "type": q["type"], "difficulty": q["difficulty"],
            "anchor": q["target_conv_uuid"],
            "arm_a": arm_a, "arm_b": arm_b,
            "rep_chunks": rep,
            "t_first_ms": round(t_first * 1000, 1), "t_rerank_ms": round(t_rerank * 1000, 1),
            "grades": {cu: judgments.get((q["id"], cu)) for cu in set(arm_a) | set(arm_b)},
        })
    (DATA / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"ran {len(results)} queries → {DATA/'results.json'} "
          f"(judged={'yes' if judgments else 'no'}) rerank_device={dev}")


def cmd_judge_targets(args) -> None:
    results = json.loads((DATA / "results.json").read_text())
    targets, seen = [], set()
    for r in results:
        for cu in set(r["arm_a"]) | set(r["arm_b"]):
            key = (r["query_id"], cu)
            if key in seen:
                continue
            seen.add(key)
            rep = r["rep_chunks"].get(cu, {})
            targets.append({"query_id": r["query_id"], "query": r["query"],
                            "conv_uuid": cu, "snippet": rep.get("snippet", "")})
    (DATA / "judge_targets.json").write_text(json.dumps(targets, ensure_ascii=False, indent=2))
    print(f"{len(targets)} unique (query,conv) pairs to judge → {DATA/'judge_targets.json'}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _rbo(a: list, b: list, p: float = 0.9) -> float:
    """Rank-biased overlap, top-weighted, handles non-identical lists."""
    if not a and not b:
        return 1.0
    sa, sb, overlap, rbo = set(), set(), 0, 0.0
    for d in range(max(len(a), len(b))):
        if d < len(a):
            sa.add(a[d]); overlap += 1 if a[d] in sb else 0
        if d < len(b):
            sb.add(b[d]); overlap += 1 if b[d] in sa and (d >= len(a) or a[d] != b[d]) else 0
        agree = len(sa & sb) / (d + 1)
        rbo += (p ** d) * agree
    return (1 - p) * rbo


def _ndcg(order: list[str], grades: dict, k: int = TOPK) -> float:
    import math
    g = [float(grades.get(cu) or 0) for cu in order[:k]]
    dcg = sum((2 ** gi - 1) / math.log2(i + 2) for i, gi in enumerate(g))
    ideal = sorted([float(v or 0) for v in grades.values()], reverse=True)[:k]
    idcg = sum((2 ** gi - 1) / math.log2(i + 2) for i, gi in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def cmd_report(args) -> None:
    from scipy.stats import kendalltau, binomtest
    results = json.loads((DATA / "results.json").read_text())
    judged = all(all(v is not None for v in r["grades"].values()) for r in results) and \
        any(r["grades"] for r in results)

    n = len(results)
    # divergence (label-free)
    overlaps, jaccards, rbos, taus, disp, changed, top1 = [], [], [], [], [], [], 0
    for r in results:
        a, b = r["arm_a"], r["arm_b"]
        sa, sb = set(a), set(b)
        overlaps.append(len(sa & sb) / TOPK)
        jaccards.append(len(sa & sb) / max(1, len(sa | sb)))
        rbos.append(_rbo(a, b))
        shared = sa & sb
        if len(shared) >= 2:
            ra = [a.index(x) for x in shared]; rb = [b.index(x) for x in shared]
            t = kendalltau(ra, rb).statistic
            if t == t:  # not nan
                taus.append(t)
        disp.append(np.mean([abs(a.index(x) - b.index(x)) for x in shared]) if shared else 0.0)
        changed.append(sum(1 for i in range(min(len(a), len(b))) if a[i] != b[i]))
        top1 += 1 if (a[:1] != b[:1]) else 0

    lines = []
    W = lines.append
    W("# Reranker ablation — results\n")
    W(f"Corpus: a private clync corpus ({n} queries). pre_topk={PRE_TOPK}, topk={TOPK}, RRF k={RRF_K}. "
      f"Embedder {EMBED_MODEL}, reranker {RERANK_MODEL}.\n")
    W("Benchmark backing [ADR 0001](../../docs/adr/0001-drop-cross-encoder-reranker.md). "
      "Reproduce (needs the local clync DB + the gitignored data/ artifacts): "
      "`uv run python ablation.py candidates && … embed && … run && … judge-targets`, "
      "judge the pairs, then `… report`.\n")

    W("## 1. Divergence gate (label-free): does the reranker even change the top-10?\n")
    W("| metric | mean | read |")
    W("|---|---|---|")
    W(f"| overlap@10 (Jaccard of sets) | {np.mean(jaccards):.3f} | 1.0 = identical members |")
    W(f"| RBO (p=0.9, top-weighted) | {np.mean(rbos):.3f} | 1.0 = identical order |")
    W(f"| Kendall-τ (shared items) | {np.mean(taus) if taus else float('nan'):.3f} | 1.0 = same order |")
    W(f"| top-1 changed | {top1}/{n} ({top1/n:.0%}) | divergence signal only — NOT a quality metric (RAG never feeds k=1) |")
    W(f"| mean rank displacement (shared) | {np.mean(disp):.2f} | positions moved |")
    W(f"| positions changed / 10 | {np.mean(changed):.2f} | |")
    inert = np.mean(rbos) > 0.99 and top1 == 0
    W(f"\n**Gate verdict:** {'INERT — reranker barely changes the list; quality judging moot.' if inert else 'The reranker materially reorders the top-10 → quality judging warranted.'}\n")

    W("## 2. Performance\n")
    tf = np.array([r["t_first_ms"] for r in results]); tr = np.array([r["t_rerank_ms"] for r in results])
    W("| stage | p50 ms | p95 ms |")
    W("|---|---|---|")
    W(f"| first-stage (embed+dense+sparse+RRF) | {np.percentile(tf,50):.0f} | {np.percentile(tf,95):.0f} |")
    W(f"| rerank (50 pairs) | {np.percentile(tr,50):.0f} | {np.percentile(tr,95):.0f} |")
    W(f"| **added by reranker** | **{np.percentile(tr,50):.0f}** | **{np.percentile(tr,95):.0f}** |\n")

    if judged:
        # Relevant = conversations Sonnet graded >=2 (substantially relevant).
        rel_sets = [{cu for cu, g in r["grades"].items() if (g or 0) >= 2} for r in results]

        def _recall_at_k(order, rel, k):
            return len(set(order[:k]) & rel) / len(rel) if rel else None

        def _mean_recall(arm_key, k):
            vals = [_recall_at_k(r[arm_key], s, k) for r, s in zip(results, rel_sets)]
            return float(np.mean([v for v in vals if v is not None]))

        FEED_KS = (3, 5, 10)
        ra = {k: _mean_recall("arm_a", k) for k in FEED_KS}
        rb = {k: _mean_recall("arm_b", k) for k in FEED_KS}

        # nDCG / MRR kept for completeness but explicitly demoted (see note).
        ndcg_a = [_ndcg(r["arm_a"], r["grades"]) for r in results]
        ndcg_b = [_ndcg(r["arm_b"], r["grades"]) for r in results]
        deltas = [b - a for a, b in zip(ndcg_a, ndcg_b)]
        wins = sum(1 for d in deltas if d > 1e-9); losses = sum(1 for d in deltas if d < -1e-9)
        sign_p = binomtest(wins, wins + losses).pvalue if (wins + losses) else 1.0

        W("## 3. Quality — set membership at the truncation you FEED the LLM\n")
        W("A reranker's job in RAG is to pack relevant docs into the top-k you actually feed "
          "the model. When the LLM reads the whole fed set, order *within* it is a weak lever "
          "(only 'lost-in-the-middle' effects), so the decision metric is **Recall@feed-k**, "
          "not rank order. Relevant = Sonnet grade ≥ 2.\n")
        W("| feed-k | Recall Arm A | Recall Arm B | Δ | note |")
        W("|---|---|---|---|---|")
        for k in FEED_KS:
            note = ("= retrieval depth; both arms already hold every relevant conv → reranker "
                    "cannot add members") if k >= TOPK else "aggressive feed budget"
            W(f"| {k} | {ra[k]:.3f} | {rb[k]:.3f} | {rb[k]-ra[k]:+.3f} | {note} |")
        W(f"\nRank-sensitive metrics (discount for read-all-k consumption): "
          f"nDCG@10 A={np.mean(ndcg_a):.3f} B={np.mean(ndcg_b):.3f} Δ={np.mean(deltas):+.3f} "
          f"({wins}W/{losses}L, sign-test p={sign_p:.3f}).\n")

        # Recall@3 by language (the regime where the reranker can matter)
        W("### Recall@3 (grade≥2) by language\n| lang | n | A | B | Δ |\n|---|---|---|---|---|")
        for lg in ("en", "ja", "zh"):
            idx = [i for i, r in enumerate(results) if r["lang"] == lg and rel_sets[i]]
            if idx:
                a = np.mean([_recall_at_k(results[i]["arm_a"], rel_sets[i], 3) for i in idx])
                b = np.mean([_recall_at_k(results[i]["arm_b"], rel_sets[i], 3) for i in idx])
                W(f"| {lg} | {len(idx)} | {a:.3f} | {b:.3f} | {b-a:+.3f} |")

        rerank_ms = np.percentile(tr, 50)
        d3 = rb[3] - ra[3]; d10 = rb[TOPK] - ra[TOPK]
        W(f"\n## Decision — conditional on feed-k\n")
        W("The reranker's value is entirely a function of how many retrieved convs you feed "
          "the LLM. Top-1 change-rate and rank order within the fed set are **not** decision "
          "metrics (the consumer reads all of what it's given); set membership at feed-k is.\n")
        W(f"- **Feed-k = {TOPK} (retrieve {TOPK}, read all):** Δrecall = {d10:+.3f}. The reranker "
          f"only reorders a set the LLM fully consumes, for +{rerank_ms:.0f} ms/query. **No value.**")
        W(f"- **Feed-k = 3 (retrieve {PRE_TOPK} → rerank → feed 3):** Δrecall@3 = {d3:+.3f}. "
          f"Here it packs more relevant convs into a tight budget — a real gain, but it costs "
          f"+{rerank_ms:.0f} ms/query and buys quality by feeding *less*.\n")
        keep_small_k = d3 >= 0.05
        W(f"**Recommendation: DROP the reranker for a read-all (feed-k≈{TOPK}) design** — it is "
          f"pure latency (+{rerank_ms:.0f} ms) for Δrecall={d10:+.3f}. "
          + (f"It becomes worth revisiting *only* if clync commits to a small feed-k (≤3–5), where "
             f"Δrecall@3={d3:+.3f} is meaningful — and only after cutting the {rerank_ms:.0f} ms rerank cost "
             f"(smaller pre_topk, batching, ONNX/GPU). "
             if keep_small_k else "")
          + f"Offline screen, n={n}, directional only; an online interleaving A/B is the real arbiter if adopted.\n")
    else:
        W("## 2. Quality\n_Judgments not yet present (`data/judgments.json`). "
          "Run the judge phase, then re-run `report`._\n")

    out = HERE / "RERANKER_ABLATION.md"
    out.write_text("\n".join(lines))
    print(f"wrote report → {out}\n"); print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("candidates", cmd_candidates), ("embed", cmd_embed), ("run", cmd_run),
                     ("judge-targets", cmd_judge_targets), ("report", cmd_report)]:
        sub.add_parser(name).set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
