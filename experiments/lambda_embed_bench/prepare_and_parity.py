"""Strip the unused ColBERT head from gpahal/bge-m3-onnx-int8 and verify batch=1
embedding + hybrid-search ranking parity against clync's fp16 BGE-M3."""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/hnsw_bench"))

XLMR_SPECIAL_IDS = {0, 1, 2, 3}
BENCH = Path(os.environ.get("CLYNC_BENCH_HOME", Path.home() / ".local/share/clync/bench"))
OUT_DIR = BENCH / "onnx"


def strip_colbert(src: Path, dst: Path) -> dict:
    model = onnx.load(str(src))
    graph = model.graph
    orig_nodes, orig_inits = len(graph.node), len(graph.initializer)
    keep_outputs = [o for o in graph.output if o.name in ("dense_vecs", "sparse_vecs")]
    del graph.output[:]
    graph.output.extend(keep_outputs)

    producers = {out: idx for idx, node in enumerate(graph.node) for out in node.output if out}
    needed = {o.name for o in graph.output}
    stack = list(needed)
    kept_idxs: set[int] = set()
    while stack:
        tensor = stack.pop()
        ni = producers.get(tensor)
        if ni is None or ni in kept_idxs:
            continue
        kept_idxs.add(ni)
        for inp in graph.node[ni].input:
            if inp and inp not in needed:
                needed.add(inp)
                stack.append(inp)

    kept_nodes = [n for idx, n in enumerate(graph.node) if idx in kept_idxs]
    kept_inits = [i for i in graph.initializer if i.name in needed]
    del graph.node[:]
    graph.node.extend(kept_nodes)
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return {
        "nodes": f"{orig_nodes} -> {len(kept_nodes)}",
        "inits": f"{orig_inits} -> {len(kept_inits)}",
        "full_mb": round(src.stat().st_size / (1024 * 1024), 2),
        "stripped_mb": round(dst.stat().st_size / (1024 * 1024), 2),
    }


def encode_one(sess: ort.InferenceSession, tok: Tokenizer, text: str) -> tuple[np.ndarray, dict[int, float]]:
    enc = tok.encode(text)
    ids = np.array([enc.ids], dtype=np.int64)
    amask = np.array([enc.attention_mask], dtype=np.int64)
    dense, sparse = sess.run(["dense_vecs", "sparse_vecs"], {"input_ids": ids, "attention_mask": amask})
    d = dense[0].astype(np.float32)
    d /= np.linalg.norm(d) + 1e-9
    sv = sparse[0, :, 0].astype(np.float32)
    sp: dict[int, float] = {}
    for pos, tid in enumerate(enc.ids):
        if tid in XLMR_SPECIAL_IDS:
            continue
        w = float(sv[pos])
        if w > sp.get(tid, 0.0):
            sp[tid] = w
    return d, sp


def evaluate_model(model_path: Path, tok_path: Path, queries: list[dict], emb_fp16: dict, label: str) -> dict:
    import search
    tok = Tokenizer.from_file(str(tok_path))
    tok.enable_truncation(max_length=512)
    t0 = time.perf_counter()
    raw = model_path.read_bytes()
    sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1000

    dense_cos, sparse_cos, embed_ms, jaccards, recall10 = [], [], [], [], []
    for e in queries:
        q = e["query"]
        t1 = time.perf_counter()
        d, sp = encode_one(sess, tok, q)
        embed_ms.append((time.perf_counter() - t1) * 1000)

        fp_d = np.array(emb_fp16[q][0], dtype=np.float32)
        fp_d /= np.linalg.norm(fp_d) + 1e-9
        dense_cos.append(float(np.dot(d, fp_d)))

        fp_sp = {int(k): float(v) for k, v in emb_fp16[q][1].items() if float(v) > 0}
        keys = set(sp) | set(fp_sp)
        num = sum(sp.get(k, 0.0) * fp_sp.get(k, 0.0) for k in keys)
        den = (sum(v * v for v in sp.values()) ** 0.5) * (sum(v * v for v in fp_sp.values()) ** 0.5)
        sparse_cos.append(num / den if den > 0 else 1.0)

        # Top-10 hybrid ranking parity on the benchmark DB
        kw = dict(e["kwargs"], topk=10)
        res_fp = search.hybrid_search(q, query_embedding=([emb_fp16[q][0]], [fp_sp]), **kw)
        res_q8 = search.hybrid_search(q, query_embedding=([d.tolist()], [sp]), **kw)
        u_fp = {r["unit_id"] for r in res_fp}
        u_q8 = {r["unit_id"] for r in res_q8}
        if u_fp:
            jaccards.append(len(u_fp & u_q8) / len(u_fp | u_q8))
            recall10.append(len(u_fp & u_q8) / len(u_fp))

    return {
        "label": label,
        "n_queries": len(queries),
        "local_load_ms": round(load_ms, 1),
        "embed_ms_p50": round(float(np.median(embed_ms)), 1),
        "embed_ms_p95": round(float(np.percentile(embed_ms, 95)), 1),
        "dense_cos_mean": round(float(np.mean(dense_cos)), 5),
        "dense_cos_min": round(float(np.min(dense_cos)), 5),
        "sparse_cos_mean": round(float(np.mean(sparse_cos)), 5),
        "sparse_cos_min": round(float(np.min(sparse_cos)), 5),
        "top10_jaccard_mean": round(float(np.mean(jaccards)), 4),
        "top10_overlap_mean": round(float(np.mean(recall10)), 4),
    }


def main() -> None:
    src_model = Path(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--gpahal--bge-m3-onnx-int8/snapshots/*/model_quantized.onnx"))[0])
    src_tok = Path(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/*/tokenizer.json"))[0])
    stripped = OUT_DIR / "bge_m3_int8_dense_sparse.onnx"

    strip_info = strip_colbert(src_model, stripped)
    print("STRIP:", json.dumps(strip_info), flush=True)

    all_q = json.loads((BENCH / "queries.json").read_text())
    emb_fp16 = json.loads((BENCH / "queries.emb.json").read_text())

    # Select 30 distinct real recall_raw/transcripts queries including all CJK queries
    picked, seen = [], set()
    for e in all_q:
        if e["origin"] not in ("recall_raw", "transcripts"):
            continue
        q = e["query"]
        if q in seen:
            continue
        if any("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in q):
            seen.add(q)
            picked.append(e)
    for e in all_q:
        if len(picked) >= 30:
            break
        if e["origin"] != "recall_raw":
            continue
        q = e["query"]
        if q in seen:
            continue
        seen.add(q)
        picked.append(e)

    full_res = evaluate_model(src_model, src_tok, picked, emb_fp16, "full_int8")
    print("PARITY_FULL:", json.dumps(full_res), flush=True)
    strip_res = evaluate_model(stripped, src_tok, picked, emb_fp16, "stripped_int8")
    print("PARITY_STRIPPED:", json.dumps(strip_res), flush=True)

    (OUT_DIR / "local_parity.json").write_text(json.dumps({
        "strip": strip_info, "full": full_res, "stripped": strip_res,
    }, indent=1))


if __name__ == "__main__":
    main()
