"""Lambda handler for S3-to-RAM vs container-image ONNX BGE-M3 embedding benchmark."""
from __future__ import annotations

import os
import resource
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

XLMR_SPECIAL_IDS = {0, 1, 2, 3}

_SESS: ort.InferenceSession | None = None
_TOK: Tokenizer | None = None
_INIT_STATS: dict = {}
_COLD_CONSUMED = False


def _download_parallel_bytes(bucket: str, key: str, part_mb: int = 32, workers: int = 8) -> bytes:
    """Download an S3 object directly into a bytearray in RAM using parallel ranged GETs."""
    s3 = boto3.client("s3")
    size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    buf = bytearray(size)
    mv = memoryview(buf)
    chunk = part_mb * 1024 * 1024
    ranges = [(off, min(off + chunk - 1, size - 1)) for off in range(0, size, chunk)]

    def fetch(bounds: tuple[int, int]) -> None:
        start, end = bounds
        resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
        body = resp["Body"].read()
        mv[start : end + 1] = body

    with ThreadPoolExecutor(max_workers=min(workers, len(ranges))) as pool:
        list(pool.map(fetch, ranges))
    return bytes(buf)


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _init() -> None:
    global _SESS, _TOK, _INIT_STATS
    if _SESS is not None:
        return
    mode = os.environ["LOAD_MODE"]  # "s3_ram" or "local_disk"
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    if mode == "s3_ram":
        bucket = os.environ["MODEL_BUCKET"]
        model_key = os.environ["MODEL_KEY"]
        tok_key = os.environ["TOKENIZER_KEY"]
        part_mb = int(os.environ.get("S3_PART_MB", "32"))
        workers = int(os.environ.get("S3_WORKERS", "8"))
        s3 = boto3.client("s3")
        tok_bytes = s3.get_object(Bucket=bucket, Key=tok_key)["Body"].read()
        model_input = _download_parallel_bytes(bucket, model_key, part_mb=part_mb, workers=workers)
        model_bytes_len = len(model_input)
    elif mode == "local_disk":
        tok_bytes = open(os.environ["TOKENIZER_PATH"], "rb").read()
        model_input = os.environ["MODEL_PATH"]
        model_bytes_len = os.path.getsize(model_input)
    else:
        raise ValueError(f"Unknown LOAD_MODE={mode!r}")
    dl_ms = (time.perf_counter() - t0) * 1000
    rss_after_dl = _rss_mb()

    t1 = time.perf_counter()
    _TOK = Tokenizer.from_buffer(tok_bytes)
    _TOK.enable_truncation(max_length=512)
    so = ort.SessionOptions()
    # Keep default intra_op threads (Lambda vCPUs) unless overridden
    if os.environ.get("ORT_THREADS"):
        so.intra_op_num_threads = int(os.environ["ORT_THREADS"])
    so.inter_op_num_threads = 1
    _SESS = ort.InferenceSession(model_input, sess_options=so, providers=["CPUExecutionProvider"])
    # Release raw bytes buffer immediately so transient spike drops after session load
    del model_input
    ort_ms = (time.perf_counter() - t1) * 1000
    rss_after_ort = _rss_mb()

    _INIT_STATS = {
        "mode": mode,
        "model_mb": round(model_bytes_len / (1024 * 1024), 2),
        "init_download_ms": round(dl_ms, 1),
        "init_ort_load_ms": round(ort_ms, 1),
        "init_total_ms": round((time.perf_counter() - t_total) * 1000, 1),
        "rss_after_dl_mb": rss_after_dl,
        "rss_after_ort_mb": rss_after_ort,
    }


# Trigger during Lambda INIT phase so Lambda Init Duration captures model load
if os.environ.get("EAGER_INIT", "1") == "1":
    _init()


def _encode(text: str) -> tuple[list[float], dict[int, float]]:
    assert _SESS is not None and _TOK is not None
    enc = _TOK.encode(text)
    ids = np.array([enc.ids], dtype=np.int64)
    amask = np.array([enc.attention_mask], dtype=np.int64)
    dense, sparse = _SESS.run(["dense_vecs", "sparse_vecs"], {"input_ids": ids, "attention_mask": amask})
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
    return d.tolist(), sp


def handler(event: dict, context) -> dict:
    global _COLD_CONSUMED
    t0 = time.perf_counter()
    if _SESS is None:
        _init()
    is_cold = not _COLD_CONSUMED
    _COLD_CONSUMED = True

    text = event["query"]
    t_emb = time.perf_counter()
    dense, sparse = _encode(text)
    embed_ms = (time.perf_counter() - t_emb) * 1000

    return {
        "is_cold": is_cold,
        "embed_ms": round(embed_ms, 1),
        "handler_ms": round((time.perf_counter() - t0) * 1000, 1),
        "rss_mb": _rss_mb(),
        "dense_dim": len(dense),
        "sparse_nnz": len(sparse),
        "dense_head": [round(x, 6) for x in dense[:4]],
        "init": _INIT_STATS if is_cold else None,
    }
