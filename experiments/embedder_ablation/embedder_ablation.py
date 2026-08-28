"""clync embedder ablation — BGE-M3 vs Qwen3-Embedding-0.6B, deterministic.

Question: which embedder retrieves the right past conversations on THIS corpus?
Public benchmarks favor Qwen3 on pure dense; BGE-M3 adds a learned-sparse head in
the same model. We measure on the real clync corpus, reusing the reranker
ablation's query set + Sonnet judgments (shared ground truth).

Arms (each → distinct-conversation top-10):
  bge_dense   — BGE-M3 dense cosine only
  qwen_dense  — Qwen3-Embedding-0.6B dense cosine only   (Qwen is dense-only)
  bge_hybrid  — BGE-M3 dense + learned-sparse → RRF(k=60) (clync's chosen config)

Comparisons the report answers:
  - bge_dense vs qwen_dense  → pure embedder quality, head-to-head
  - bge_hybrid vs bge_dense  → what the learned-sparse head adds
  - bge_hybrid vs qwen_dense → the actual system decision (best config each)

Metrics: anchor recall@{3,5,10} (label-free), graded recall@{3,5,10} (grade>=2,
reused judgments), SPEED (load, per-query encode p50/p95, corpus throughput, peak
RSS, params), and top-10 SET-membership divergence between arms.

Corpus + chunking are byte-identical to the reranker ablation (same load_chunks,
CHUNK_CHARS, RRF_K) so the two experiments compare on the same footing.
Fail-loud: any missing input / model / device problem raises. No silent fallbacks.

Subcommands:
  embed --model {bge|qwen}  → embed corpus + queries, cache vectors + speed meta
  run                       → cached vectors → results_embed.json (3 arms/query)
  report                    → results_embed.json → EMBEDDER_ABLATION.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import sqlite3
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DATA = HERE / "data"
# Shared ground truth lives with the reranker ablation (same 20 queries + judges).
SHARED = HERE.parent / "reranker_ablation" / "data"
DB_PATH = Path(os.environ.get("CLYNC_DB", Path.home() / ".local/share/clync/history.db"))

RRF_K = 60
TOPK = 10
CHUNK_CHARS = 2000
CHUNK_OVERLAP = 200
BGE_MODEL = "BAAI/bge-m3"
QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
INT8_MODEL = "gpahal/bge-m3-onnx-int8"   # int8 ONNX BGE-M3, all 3 heads, CPU
XLMR_SPECIAL_IDS = {0, 1, 2, 3}          # <s>, <pad>, </s>, <unk> — dropped from sparse
# Qwen3-Embedding expects a task instruction on the QUERY side only (docs stay raw).
QWEN_TASK = "Given a search query, retrieve past conversation passages relevant to it"


# --------------------------------------------------------------------------- #
# Corpus + chunking  (mirrors reranker_ablation.ablation — identical corpus)
# --------------------------------------------------------------------------- #
def _lang(s: str) -> str:
    s = s or ""
    if re.search(r"[぀-ヿ]", s):   # kana → Japanese
        return "ja"
    if re.search(r"[一-鿿]", s):   # Han without kana → Chinese
        return "zh"
    return "en"


def load_chunks() -> list[dict]:
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
            chunks.append({"chunk_id": f"{r['mid']}#{j}", "conv_uuid": r["cid"],
                           "conv_name": r["cname"], "text": body, "lang": _lang(body)})
    return chunks


def _device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    raise SystemExit("MPS not available — this experiment targets Apple Silicon.")


def _queries() -> list[dict]:
    return json.loads((SHARED / "queries.json").read_text())


# --------------------------------------------------------------------------- #
# Embedding (cached, one model per process so peak RSS is per-model)
# --------------------------------------------------------------------------- #
def _rss_bytes() -> int:
    # macOS: ru_maxrss is bytes; Linux: kilobytes. This experiment is macOS.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def cmd_embed(args) -> None:
    model_key = args.model
    chunks = load_chunks()
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    queries = _queries()
    qtexts = [q["query"] for q in queries]
    ctexts = [c["text"] for c in chunks]
    dev = _device()
    meta = {"model": model_key, "n_chunks": len(chunks), "n_queries": len(queries)}

    if model_key == "bge":
        from FlagEmbedding import BGEM3FlagModel
        t0 = time.time()
        model = BGEM3FlagModel(BGE_MODEL, use_fp16=True, devices=dev)
        meta["load_ms"] = round((time.time() - t0) * 1000, 1)
        meta["n_params"] = int(sum(p.numel() for p in model.model.parameters()))

        t0 = time.time()
        out = model.encode(ctexts, batch_size=16, return_dense=True,
                           return_sparse=True, max_length=2048)
        meta["corpus_embed_s"] = round(time.time() - t0, 2)
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        dense /= (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9)
        np.save(CACHE / "bge_dense.npy", dense)
        (CACHE / "bge_sparse.json").write_text(json.dumps(
            [{str(k): float(v) for k, v in d.items()} for d in out["lexical_weights"]]))
        meta["dim"] = int(dense.shape[1])

        # per-query latency, one at a time (online-realistic), dense + sparse
        per_ms, qdense, qsparse = [], [], []
        for qt in qtexts:
            t0 = time.time()
            qo = model.encode([qt], return_dense=True, return_sparse=True, max_length=512)
            per_ms.append(round((time.time() - t0) * 1000, 1))
            qd = np.asarray(qo["dense_vecs"][0], dtype=np.float32)
            qd /= (np.linalg.norm(qd) + 1e-9)
            qdense.append(qd)
            qsparse.append({str(k): float(v) for k, v in qo["lexical_weights"][0].items()})
        np.save(CACHE / "bge_qdense.npy", np.asarray(qdense, dtype=np.float32))
        (CACHE / "bge_qsparse.json").write_text(json.dumps(qsparse))
        meta["per_query_ms"] = per_ms

    elif model_key == "qwen":
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        model = SentenceTransformer(QWEN_MODEL, device=dev,
                                    model_kwargs={"torch_dtype": "float16"})
        meta["load_ms"] = round((time.time() - t0) * 1000, 1)
        meta["n_params"] = int(sum(p.numel() for p in model.parameters()))

        t0 = time.time()
        dense = model.encode(ctexts, batch_size=16, normalize_embeddings=True,
                             convert_to_numpy=True).astype(np.float32)
        meta["corpus_embed_s"] = round(time.time() - t0, 2)
        np.save(CACHE / "qwen_dense.npy", dense)
        meta["dim"] = int(dense.shape[1])

        q_prompt = f"Instruct: {QWEN_TASK}\nQuery: "
        per_ms, qdense = [], []
        for qt in qtexts:
            t0 = time.time()
            qd = model.encode([qt], prompt=q_prompt, normalize_embeddings=True,
                              convert_to_numpy=True)[0].astype(np.float32)
            per_ms.append(round((time.time() - t0) * 1000, 1))
            qdense.append(qd)
        np.save(CACHE / "qwen_qdense.npy", np.asarray(qdense, dtype=np.float32))
        meta["per_query_ms"] = per_ms
    elif model_key == "int8":
        from optimum.onnxruntime import ORTModelForCustomTasks
        from transformers import AutoTokenizer
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(BGE_MODEL)
        model = ORTModelForCustomTasks.from_pretrained(
            INT8_MODEL, file_name="model_quantized.onnx")   # CPU ONNX Runtime
        meta["load_ms"] = round((time.time() - t0) * 1000, 1)
        meta["n_params"] = None   # ONNX graph — footprint read from peak RSS instead

        def _encode(texts, max_length):
            inp = tok(texts, padding=True, truncation=True, max_length=max_length,
                      return_tensors="np")
            out = model(**inp)
            dense = np.asarray(out["dense_vecs"], dtype=np.float32)
            dense /= (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9)
            ids, amask = inp["input_ids"], inp["attention_mask"]
            sv = np.asarray(out["sparse_vecs"], dtype=np.float32)[..., 0]  # (B, seq)
            sparse = []
            for i in range(len(texts)):
                d = {}
                for pos in range(ids.shape[1]):
                    if amask[i, pos] == 0:
                        continue
                    tid = int(ids[i, pos])
                    if tid in XLMR_SPECIAL_IDS:
                        continue
                    w = float(sv[i, pos])
                    if w > d.get(str(tid), 0.0):   # keep max weight per token
                        d[str(tid)] = w
                sparse.append(d)
            return dense, sparse

        t0 = time.time()
        cd, cs = [], []
        for b in range(0, len(ctexts), 16):
            dd, ss = _encode(ctexts[b:b + 16], 2048)
            cd.append(dd); cs.extend(ss)
        dense = np.vstack(cd)
        meta["corpus_embed_s"] = round(time.time() - t0, 2)
        np.save(CACHE / "int8_dense.npy", dense)
        (CACHE / "int8_sparse.json").write_text(json.dumps(cs))
        meta["dim"] = int(dense.shape[1])

        per_ms, qdense, qsparse = [], [], []
        for qt in qtexts:
            t0 = time.time()
            dd, ss = _encode([qt], 512)
            per_ms.append(round((time.time() - t0) * 1000, 1))
            qdense.append(dd[0]); qsparse.append(ss[0])
        np.save(CACHE / "int8_qdense.npy", np.asarray(qdense, dtype=np.float32))
        (CACHE / "int8_qsparse.json").write_text(json.dumps(qsparse))
        meta["per_query_ms"] = per_ms
    else:
        raise SystemExit(f"unknown --model {model_key!r} (expected bge|qwen|int8)")

    meta["peak_rss_bytes"] = _rss_bytes()
    (CACHE / f"{model_key}_meta.json").write_text(json.dumps(meta, indent=2))
    params = f"{meta['n_params']/1e6:.0f}M" if meta["n_params"] else "n/a"
    print(f"[{model_key}] load={meta['load_ms']}ms corpus={meta['corpus_embed_s']}s "
          f"per_query_p50={np.percentile(meta['per_query_ms'],50):.0f}ms "
          f"params={params} dim={meta['dim']} "
          f"peakRSS={meta['peak_rss_bytes']/1e9:.2f}GB → {CACHE}")


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
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


def _collapse(chunk_order: list[int], chunks: list[dict], topk: int) -> list[str]:
    seen, out = set(), []
    for idx in chunk_order:
        cu = chunks[idx]["conv_uuid"]
        if cu not in seen:
            seen.add(cu); out.append(cu)
        if len(out) >= topk:
            break
    return out


MODEL_KEYS = ("bge", "int8", "qwen")   # display order


def _avail_models() -> list[str]:
    return [m for m in MODEL_KEYS if (CACHE / f"{m}_meta.json").exists()]


def cmd_run(args) -> None:
    chunks = json.loads((CACHE / "chunks.json").read_text())
    n = len(chunks)
    models = _avail_models()
    if not models:
        raise SystemExit("no embeddings cached — run `embed --model {bge|int8|qwen}` first.")
    cache = {}
    for m in models:
        d = np.load(CACHE / f"{m}_dense.npy")
        if d.shape[0] != n:
            raise SystemExit(f"{m} corpus cache ({d.shape[0]}) != {n} chunks — re-embed.")
        entry = {"d": d, "qd": np.load(CACHE / f"{m}_qdense.npy")}
        sp = CACHE / f"{m}_sparse.json"
        if sp.exists():
            entry["s"] = json.loads(sp.read_text())
            entry["qs"] = json.loads((CACHE / f"{m}_qsparse.json").read_text())
        cache[m] = entry

    queries = _queries()
    results = []
    for i, q in enumerate(queries):
        arms = {}
        for m in models:
            drank = list(np.argsort(-(cache[m]["d"] @ cache[m]["qd"][i])))
            arms[f"{m}_dense"] = _collapse(drank, chunks, TOPK)
            if "s" in cache[m]:
                ss = np.array([_sparse_dot(cache[m]["qs"][i], cache[m]["s"][j]) for j in range(n)])
                rrf = _rrf([drank, list(np.argsort(-ss))])
                arms[f"{m}_hybrid"] = _collapse(sorted(rrf, key=lambda x: -rrf[x]), chunks, TOPK)
        results.append({"query_id": q["id"], "lang": q["lang"], "type": q["type"],
                        "difficulty": q["difficulty"], "anchor": q["target_conv_uuid"],
                        "arms": arms})
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "results_embed.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"ran {len(results)} queries × {len(results[0]['arms'])} arms "
          f"({', '.join(results[0]['arms'])}) → {DATA/'results_embed.json'}")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
ARM_ORDER = ["bge_dense", "int8_dense", "qwen_dense", "bge_hybrid", "int8_hybrid"]
ARM_LABEL = {"bge_dense": "BGE fp16 dense", "int8_dense": "BGE int8 dense",
             "qwen_dense": "Qwen3-0.6B dense", "bge_hybrid": "BGE fp16 hybrid",
             "int8_hybrid": "BGE int8 hybrid"}
MODEL_LABEL = {"bge": "BAAI/bge-m3 (fp16, PyTorch/MPS)",
               "int8": "gpahal/bge-m3-onnx-int8 (int8, ONNX/CPU)",
               "qwen": "Qwen/Qwen3-Embedding-0.6B (fp16, PyTorch/MPS)"}


def _recall(order, rel, k):
    return len(set(order[:k]) & rel) / len(rel) if rel else None


def cmd_report(args) -> None:
    results = json.loads((DATA / "results_embed.json").read_text())
    n = len(results)
    models = _avail_models()
    meta = {m: json.loads((CACHE / f"{m}_meta.json").read_text()) for m in models}
    arms = [a for a in ARM_ORDER if a in results[0]["arms"]]
    judgments = {}
    jp = SHARED / "judgments.json"
    if jp.exists():
        judgments = {(j["query_id"], j["conv_uuid"]): j["grade"]
                     for j in json.loads(jp.read_text())}

    L = []
    W = L.append
    W("# Embedder ablation — BGE-M3 (fp16 & int8) vs Qwen3-Embedding-0.6B\n")
    W(f"Corpus: a private clync corpus, identical to the reranker ablation ({n} queries, "
      f"same chunks). topk={TOPK}, RRF k={RRF_K}. Dense scored by cosine; BGE sparse "
      f"by learned-weight dot; hybrid = dense+sparse RRF. int8 = ONNX-Runtime "
      f"quantized BGE-M3 on CPU (same tokenizer/heads as fp16).\n")
    W("Backs [ADR 0002](../../docs/adr/0002-embedder-choice.md).\n")

    # ---- Speed / footprint ----
    W("## 1. Speed & footprint\n")
    W("| model | params | dim | load | corpus embed | per-query p50 | per-query p95 | peak RSS |")
    W("|---|---|---|---|---|---|---|---|")
    for m in models:
        d = meta[m]
        pm = d["per_query_ms"]
        params = f"{d['n_params']/1e6:.0f}M" if d["n_params"] else "— (ONNX)"
        W(f"| {MODEL_LABEL[m]} | {params} | {d['dim']} | {d['load_ms']/1000:.1f}s | "
          f"{d['corpus_embed_s']:.1f}s ({d['n_chunks']} chunks) | {np.percentile(pm,50):.0f} ms | "
          f"{np.percentile(pm,95):.0f} ms | {d['peak_rss_bytes']/1e9:.2f} GB |")
    W("\n_per-query = single query encoded alone (online-realistic); BGE per-query "
      "includes dense **and** sparse in one pass. fp16 runs on MPS, int8 on CPU._\n")

    # ---- Quality tables ----
    def recall_table(rel_fn, title, note):
        W(f"## {title}\n{note}\n")
        W("| feed-k | " + " | ".join(ARM_LABEL[a] for a in arms) + " |")
        W("|---|" + "---|" * len(arms))
        for k in (3, 5, 10):
            cells = []
            for a in arms:
                vals = [v for v in (_recall(r["arms"][a], rel_fn(r), k) for r in results) if v is not None]
                cells.append(f"{np.mean(vals):.3f}")
            W(f"| {k} | " + " | ".join(cells) + " |")
        W("")

    recall_table(lambda r: {r["anchor"]}, "2. Quality — anchor recall@k (label-free)",
                 "Does the embedder surface each query's known source conversation into the top-k? "
                 "Free ground truth, unbiased across embedders.")

    def rel_graded(r):
        return {cu for cu in set().union(*r["arms"].values())
                if (judgments.get((r["query_id"], cu)) or 0) >= 2}
    all_pairs = {(r["query_id"], cu) for r in results for a in arms for cu in r["arms"][a]}
    unjudged = sorted(p for p in all_pairs if p not in judgments)
    if judgments:
        recall_table(rel_graded, "3. Quality — graded recall@k (Sonnet grade ≥ 2)",
                     f"Relevant = graded ≥2. **Coverage:** {len(unjudged)} of {len(all_pairs)} "
                     f"(query,conv) pairs unjudged (treated as grade 0). "
                     + ("Top-up judging recommended before trusting graded deltas."
                        if unjudged else "Full judgment coverage."))
    else:
        W("## 3. Quality — graded recall@k\n_No judgments.json found locally._\n")

    # ---- top-10 membership divergence between arms ----
    W("## 4. Top-10 set-membership divergence between arms\n")
    W("How different are the conversations each arm surfaces (set overlap, order-free)? "
      "The int8-vs-fp16 rows show how much quantization perturbs retrieval — near-1.0 "
      "Jaccard means quantization is effectively lossless for ranking.\n")
    W("| arm pair | mean overlap@10 (/10) | mean Jaccard | queries with identical set |")
    W("|---|---|---|---|")
    candidate_pairs = [("bge_dense", "int8_dense"), ("bge_hybrid", "int8_hybrid"),
                       ("bge_dense", "qwen_dense"), ("bge_hybrid", "qwen_dense"),
                       ("bge_dense", "bge_hybrid")]
    for a, b in candidate_pairs:
        if a not in arms or b not in arms:
            continue
        ov, ja, same = [], [], 0
        for r in results:
            sa, sb = set(r["arms"][a]), set(r["arms"][b])
            ov.append(len(sa & sb)); ja.append(len(sa & sb) / len(sa | sb))
            same += 1 if sa == sb else 0
        W(f"| {ARM_LABEL[a]} vs {ARM_LABEL[b]} | {np.mean(ov):.2f} | {np.mean(ja):.3f} | {same}/{n} |")
    W("")

    # ---- by language: anchor recall@10 ----
    W("### Anchor recall@10 by language\n| lang | n | " + " | ".join(ARM_LABEL[a] for a in arms) + " |")
    W("|---|---|" + "---|" * len(arms))
    for lg in ("en", "ja"):
        idx = [r for r in results if r["lang"] == lg]
        if not idx:
            continue
        cells = [f"{np.mean([_recall(r['arms'][a], {r['anchor']}, 10) for r in idx]):.3f}" for a in arms]
        W(f"| {lg} | {len(idx)} | " + " | ".join(cells) + " |")
    W("")

    out = HERE / "EMBEDDER_ABLATION.md"
    out.write_text("\n".join(L))
    print(f"wrote report → {out}\n"); print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed"); e.add_argument("--model", required=True, choices=["bge", "qwen", "int8"])
    e.set_defaults(func=cmd_embed)
    for name, fn in [("run", cmd_run), ("report", cmd_report)]:
        sub.add_parser(name).set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
