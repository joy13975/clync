"""Per-owner AWS Lambda handler for clync-share (direct lambda:InvokeFunction).

Verifies every caller with a short-lived SigV4-presigned sts:GetCallerIdentity
request, decrypts the per-owner data key via KMS once per warm container, runs
HNSW hybrid search over encrypted Postgres rows, and serves:
  - read-only `ask` via Bedrock live redaction (with per-call logging-off guard)
    and opaque random citation handles;
  - owner-only full recall (`search_history`, `search_transcripts`,
    `get_conversation`, `resolve_citation`, `ingest`).
"""
from __future__ import annotations

import base64
import gzip
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import boto3
import numpy as np
import psycopg
from psycopg.rows import dict_row

from clync_cipher import (
    DENSE_DIM,
    SPARSE_DIM,
    ClyncCipher,
    dense_literal,
    make_citation_handle,
    sparse_literal,
)

OWNER = os.environ["CLYNC_OWNER"]
OWNER_SCHEMA = os.environ["CLYNC_DB_SCHEMA"]
DB_HOST = os.environ["CLYNC_DB_HOST"]
DB_PORT = int(os.environ.get("CLYNC_DB_PORT", "5432"))
DB_NAME = os.environ.get("CLYNC_DB_NAME", "clync")
DB_USER = os.environ["CLYNC_DB_USER"]
AWS_REGION = os.environ["AWS_REGION"]
ALLOWED_ACCOUNT_ID = os.environ["CLYNC_ALLOWED_ACCOUNT_ID"]
OWNER_USER_IDS = {x.strip() for x in os.environ.get("CLYNC_OWNER_USER_IDS", "").split(",") if x.strip()}
ALLOWED_ROLE_PREFIXES = tuple(
    x.strip() for x in os.environ.get("CLYNC_ALLOWED_ROLE_PREFIXES", "").split(",") if x.strip()
)
ENABLE_SHARE_MCP = os.environ.get("CLYNC_ENABLE_SHARE_MCP", "true").lower() == "true"
ENABLE_OWNER_MCP = os.environ.get("CLYNC_ENABLE_OWNER_MCP", "true").lower() == "true"
DATA_KEY_CT_B64 = os.environ["CLYNC_DATA_KEY_CT_B64"]
KMS_KEY_ID = os.environ["CLYNC_KMS_KEY_ID"]
STAGING_BUCKET = os.environ.get("CLYNC_STAGING_BUCKET", "")
BEDROCK_REGION = os.environ["CLYNC_BEDROCK_REGION"]
BEDROCK_MODEL_ID = os.environ["CLYNC_BEDROCK_MODEL_ID"]
MODEL_DIR = os.environ.get("CLYNC_MODEL_DIR", "/var/task/model")

PRE_TOPK = 50
RRF_K = 60
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 800
EXACT_SELECTION_MAX = 30000
MAX_STS_SKEW_SECS = 90
ERROR_KEY = "error"
XLMR_SPECIAL_IDS = {0, 1, 2, 3}
STS_NS = {"s": "https://sts.amazonaws.com/doc/2011-06-15/"}
SCHEMA_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

_CIPHER: ClyncCipher | None = None
_ORT_SESS: Any = None
_TOKENIZER: Any = None
_DB_TOKEN_CACHE: tuple[float, str] | None = None

REDACTION_SYSTEM_PROMPT = """You answer a teammate's question based ONLY on the owner's retrieved work notes below.
Strict privacy & redaction rules (MUST obey every rule):
1. NEVER quote the owner or assistant verbatim. Synthesize and paraphrase in neutral, professional engineering prose.
2. NEVER include profanity, frustration, scolding, personal/non-work anecdotes, private preferences, employee/HR matters, credentials, API keys, tokens, URLs with secrets, or internal personal paths.
3. If a retrieved snippet is purely private/non-work or contains nothing relevant to the work question, ignore it.
4. Ground every factual claim by citing the provided bracketed slot tags like [S1], [S2] immediately after the claim. Only cite slots you actually used.
5. If the retrieved material does not answer the question, say so plainly without guessing.
"""


def _validate_schema_name() -> str:
    if not SCHEMA_IDENT_RE.match(OWNER_SCHEMA):
        raise ValueError(f"invalid schema identifier: {OWNER_SCHEMA!r}")
    return OWNER_SCHEMA


def _get_cipher() -> ClyncCipher:
    global _CIPHER
    if _CIPHER is None:
        kms = boto3.client("kms", region_name=AWS_REGION)
        resp = kms.decrypt(
            KeyId=KMS_KEY_ID,
            CiphertextBlob=base64.b64decode(DATA_KEY_CT_B64),
            EncryptionContext={"clync_owner": OWNER},
        )
        _CIPHER = ClyncCipher(resp["Plaintext"])
    return _CIPHER


def _get_embedder() -> tuple[Any, Any]:
    global _ORT_SESS, _TOKENIZER
    if _ORT_SESS is None or _TOKENIZER is None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ORT_SESS = ort.InferenceSession(
            os.path.join(MODEL_DIR, "bge_m3_int8_dense_sparse.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        tok.enable_truncation(max_length=512)
        _TOKENIZER = tok
    return _ORT_SESS, _TOKENIZER


def _embed_query(text: str) -> tuple[list[float], dict[int, float]]:
    sess, tok = _get_embedder()
    enc = tok.encode(text)
    ids = np.asarray([enc.ids], dtype=np.int64)
    mask = np.asarray([enc.attention_mask], dtype=np.int64)
    outs = sess.run(None, {"input_ids": ids, "attention_mask": mask})
    dense_raw, sparse_raw = outs[0][0], outs[1][0]
    norm = float(np.linalg.norm(dense_raw))
    dense = (dense_raw / norm if norm > 0 else dense_raw).astype(np.float32).tolist()
    weights: dict[int, float] = {}
    for tid, w in zip(enc.ids, sparse_raw):
        wf = float(w)
        if wf > 0 and tid not in XLMR_SPECIAL_IDS:
            prev = weights.get(tid, 0.0)
            if wf > prev:
                weights[tid] = wf
    return dense, weights


def _get_db_token() -> str:
    global _DB_TOKEN_CACHE
    now = time.monotonic()
    if _DB_TOKEN_CACHE is not None and _DB_TOKEN_CACHE[0] > now:
        return _DB_TOKEN_CACHE[1]
    rds = boto3.client("rds", region_name=AWS_REGION)
    token = rds.generate_db_auth_token(DBHostname=DB_HOST, Port=DB_PORT, DBUsername=DB_USER, Region=AWS_REGION)
    _DB_TOKEN_CACHE = (now + 600.0, token)
    return token


def _connect_db() -> psycopg.Connection:
    schema = _validate_schema_name()
    con = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=_get_db_token(),
        sslmode="require",
        row_factory=dict_row,
        options=f"-c search_path={schema},public",
    )
    return con


def _verify_caller(auth: dict[str, Any] | None) -> dict[str, Any]:
    """Replay the client's SigV4-signed sts:GetCallerIdentity request and check IAM policy."""
    if not isinstance(auth, dict):
        raise PermissionError("missing auth proof")
    url = str(auth.get("url", ""))
    headers = auth.get("headers") or {}
    parsed = urllib.parse.urlparse(url)
    expected_host = f"sts.{AWS_REGION}.amazonaws.com"
    if (
        parsed.scheme != "https"
        or parsed.hostname != expected_host
        or parsed.port is not None
        or parsed.path not in ("", "/")
        or parsed.query != "Action=GetCallerIdentity&Version=2011-06-15"
    ):
        raise PermissionError("invalid STS verification URL")
    lower_headers = {str(k).lower(): str(v) for k, v in headers.items()}
    amz_date = lower_headers.get("x-amz-date", "")
    try:
        signed_at = datetime.strptime(amz_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise PermissionError("invalid x-amz-date header") from exc
    skew = abs((datetime.now(timezone.utc) - signed_at).total_seconds())
    if skew > MAX_STS_SKEW_SECS:
        raise PermissionError(f"stale STS proof ({skew:.0f}s skew)")
    safe_headers = {
        k: v
        for k, v in headers.items()
        if str(k).lower() in ("host", "x-amz-date", "x-amz-security-token", "authorization")
    }
    req = urllib.request.Request(url, headers=safe_headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8")
    except Exception as exc:
        raise PermissionError(f"STS caller verification failed: {exc}") from exc
    root = ET.fromstring(body)
    account = root.findtext(".//s:Account", namespaces=STS_NS) or ""
    arn = root.findtext(".//s:Arn", namespaces=STS_NS) or ""
    user_id = root.findtext(".//s:UserId", namespaces=STS_NS) or ""
    if account != ALLOWED_ACCOUNT_ID:
        raise PermissionError(f"caller account {account!r} is not allowed")
    is_owner = user_id in OWNER_USER_IDS
    is_share_allowed = is_owner or any(arn.startswith(p) for p in ALLOWED_ROLE_PREFIXES)
    if not is_share_allowed:
        raise PermissionError(f"caller role {arn!r} is not authorized")
    return {"account": account, "arn": arn, "user_id": user_id, "is_owner": is_owner}


def _assert_bedrock_logging_off() -> None:
    """Fail loud if Bedrock model invocation logging is enabled in BEDROCK_REGION."""
    bedrock = boto3.client("bedrock", region_name=BEDROCK_REGION)
    resp = bedrock.get_model_invocation_logging_configuration()
    cfg = resp.get("loggingConfig") or {}
    if cfg.get("textDataDeliveryEnabled") or cfg.get("s3Config") or cfg.get("cloudWatchConfig"):
        raise RuntimeError(
            f"Bedrock invocation logging is ENABLED in {BEDROCK_REGION}; refusing to send decrypted transcripts"
        )


def _ensure_owner_schema(con: psycopg.Connection, *, create_hnsw: bool = False) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS units (
            unit_id       text PRIMARY KEY,
            source_hmac   text NOT NULL,
            kind_hmac     text NOT NULL,
            repo_hmac     text,
            project_hmac  text,
            updated_at    timestamptz,
            meta_ct       text NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            unit_id       text NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
            msg_idx       int NOT NULL,
            sender_hmac   text,
            created_at    timestamptz,
            payload_ct    text NOT NULL,
            PRIMARY KEY (unit_id, msg_idx)
        );
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id      text PRIMARY KEY,
            unit_id       text NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
            source_hmac   text NOT NULL,
            repo_hmac     text,
            project_hmac  text,
            sender_hmac   text,
            lang_hmac     text,
            msg_idx       int,
            chunk_idx     int,
            updated_at    timestamptz,
            payload_ct    text NOT NULL,
            dense         vector({DENSE_DIM}) NOT NULL,
            sparse        sparsevec({SPARSE_DIM}) NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_unit ON chunks(unit_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_hmac);
        CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_hmac);
        CREATE INDEX IF NOT EXISTS idx_chunks_updated ON chunks(updated_at);
        CREATE TABLE IF NOT EXISTS citations (
            handle        text PRIMARY KEY,
            created_at    timestamptz NOT NULL DEFAULT now(),
            caller_hmac   text NOT NULL,
            payload_ct    text NOT NULL
        );
        """
    )
    if create_hnsw:
        existing = {
            r["indexname"]
            for r in con.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() AND tablename = 'chunks'"
            ).fetchall()
        }
        if "idx_chunks_dense_hnsw" not in existing:
            con.execute(
                f"CREATE INDEX idx_chunks_dense_hnsw ON chunks USING hnsw "
                f"((dense::halfvec({DENSE_DIM})) halfvec_cosine_ops) "
                f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
            )
        if "idx_chunks_sparse_hnsw" not in existing:
            con.execute(
                f"CREATE INDEX idx_chunks_sparse_hnsw ON chunks USING hnsw "
                f"(sparse sparsevec_ip_ops) "
                f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
            )
    con.commit()


def _lang(text: str) -> str:
    if re.search(r"[぀-ヿ]", text):
        return "ja"
    if re.search(r"[一-鿿]", text):
        return "zh"
    return "en"


def _hybrid_search(
    con: psycopg.Connection,
    cipher: ClyncCipher,
    query: str,
    *,
    topk: int = 10,
    source: str = "all",
    repo: str | None = None,
    sender: str | None = None,
    per_unit: int = 1,
) -> list[dict[str, Any]]:
    qd_plain, qs_plain = _embed_query(query)
    qd_rot = cipher.rotate_dense(qd_plain)
    qs_perm = cipher.permute_sparse(qs_plain)
    params: dict[str, Any] = {
        "qd": dense_literal(qd_rot),
        "qlang": cipher.hmac_facet("lang", _lang(query)),
        "per_unit": max(1, int(per_unit)),
    }
    conds: list[str] = []
    if source == "dream":
        conds.append("source_hmac = %(src_dream)s")
        params["src_dream"] = cipher.hmac_facet("source", "dream")
    elif source == "all":
        conds.append("source_hmac <> %(src_dream)s")
        params["src_dream"] = cipher.hmac_facet("source", "dream")
    else:
        conds.append("source_hmac = %(src_one)s")
        params["src_one"] = cipher.hmac_facet("source", source)
    if repo:
        conds.append("repo_hmac = %(repo)s")
        params["repo"] = cipher.hmac_facet("repo", repo)
    if sender:
        role_map = {
            "user": ["human", "user"],
            "assistant": ["assistant"],
        }
        senders = role_map.get(sender, [sender])
        conds.append("sender_hmac = ANY(%(senders)s)")
        params["senders"] = [cipher.hmac_facet("sender", s) for s in senders]
    where = ("WHERE " + " AND ".join(conds)) if conds else ""

    n_sel = con.execute(
        f"SELECT count(*) AS n FROM (SELECT 1 FROM chunks {where} LIMIT {EXACT_SELECTION_MAX + 1}) s",
        params,
    ).fetchone()["n"]
    exact = n_sel <= EXACT_SELECTION_MAX
    with_sparse = bool(qs_perm)
    if with_sparse:
        params["qs"] = sparse_literal(qs_perm)

    if not exact:
        con.execute(
            "SELECT set_config('hnsw.ef_search', %s, true), "
            "set_config('hnsw.iterative_scan', 'strict_order', true), "
            "set_config('enable_seqscan', 'off', true), "
            "set_config('enable_bitmapscan', 'off', true), "
            "set_config('enable_sort', 'off', true)",
            (str(HNSW_EF_SEARCH),),
        )
        d_dist = f"(dense::halfvec({DENSE_DIM})) <=> (%(qd)s::halfvec({DENSE_DIM}))"
        s_dist = "sparse <#> %(qs)s::sparsevec"
    else:
        d_dist = "(dense <=> %(qd)s::vector) + 0"
        s_dist = "(sparse <#> %(qs)s::sparsevec) + 0"

    ctes = [
        f"d AS (SELECT chunk_id, row_number() OVER (ORDER BY distance) rnk "
        f"FROM (SELECT chunk_id, {d_dist} AS distance FROM chunks {where} ORDER BY distance LIMIT {PRE_TOPK}) sl)"
    ]
    unions = ["SELECT * FROM d"]
    if with_sparse:
        ctes.append(
            f"s AS (SELECT chunk_id, row_number() OVER (ORDER BY distance) rnk "
            f"FROM (SELECT chunk_id, {s_dist} AS distance FROM chunks {where} ORDER BY distance LIMIT {PRE_TOPK}) sl)"
        )
        unions.append("SELECT * FROM s")
    ctes.append(
        f"fused AS (SELECT chunk_id, SUM(1.0 / ({RRF_K} + rnk)) AS rrf "
        f"FROM ({' UNION ALL '.join(unions)}) u GROUP BY chunk_id)"
    )
    ctes.append(
        "scored AS MATERIALIZED ("
        "SELECT c.chunk_id, c.unit_id, c.msg_idx, c.chunk_idx, c.payload_ct, "
        "1 - (c.dense <=> %(qd)s::vector) AS dense_sim, "
        "f.rrf * (1.0 "
        "+ CASE WHEN c.updated_at IS NULL THEN 0 "
        "       ELSE 0.10 * exp(-GREATEST(0, EXTRACT(EPOCH FROM (now() - c.updated_at))) / (86400.0 * 180)) END "
        "+ CASE WHEN c.lang_hmac = %(qlang)s THEN 0.05 ELSE 0 END) AS score "
        "FROM fused f JOIN chunks c ON c.chunk_id = f.chunk_id)"
    )
    ctes.append(
        "ranked AS (SELECT chunk_id, unit_id, msg_idx, chunk_idx, payload_ct, score, dense_sim, "
        "row_number() OVER (PARTITION BY unit_id ORDER BY score DESC) rn FROM scored)"
    )
    sql = (
        "WITH "
        + ",\n".join(ctes)
        + "\nSELECT chunk_id, unit_id, msg_idx, chunk_idx, payload_ct, score, dense_sim "
        "FROM ranked WHERE rn <= %(per_unit)s"
    )
    rows = con.execute(sql, params).fetchall()
    con.rollback()

    out = []
    for r in rows:
        payload = json.loads(cipher.decrypt_text(r["payload_ct"], aad=f"chunks:{r['chunk_id']}"))
        out.append(
            {
                "chunk_id": r["chunk_id"],
                "unit_id": r["unit_id"],
                "unit_name": payload.get("unit_name") or "",
                "source": payload.get("source") or "",
                "sender": payload.get("sender"),
                "lang": payload.get("lang"),
                "text": payload.get("text") or "",
                "msg_idx": r["msg_idx"],
                "chunk_idx": r["chunk_idx"],
                "score": float(r["score"]),
                "dense_sim": float(r["dense_sim"]),
            }
        )
    out.sort(key=lambda x: -x["score"])
    return out[:topk]


def _handle_ask(con: psycopg.Connection, cipher: ClyncCipher, caller: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    question = str(params.get("question", "")).strip()
    if not question:
        raise ValueError("question is required")
    limit = min(15, max(1, int(params.get("limit", 8))))
    insights = _hybrid_search(con, cipher, question, topk=min(5, limit), source="dream")
    raw = _hybrid_search(con, cipher, question, topk=limit, source="all")
    hits = [h for h in insights if h["dense_sim"] >= 0.45] + raw
    if not hits:
        return {"owner": OWNER, "answer": f"No relevant work notes found in {OWNER}'s archive.", "citations": []}

    _assert_bedrock_logging_off()
    slots: dict[str, dict[str, Any]] = {}
    context_lines = []
    for idx, h in enumerate(hits, 1):
        slot = f"S{idx}"
        slots[slot] = h
        kind = "distilled_insight" if h["source"] == "dream" else h["source"]
        context_lines.append(f"[{slot}] (source_type={kind}, sender={h['sender']}): {h['text'][:1500]}")

    user_prompt = (
        f"Question for {OWNER}: {question}\n\n"
        f"Retrieved snippets (CONFIDENTIAL — DO NOT QUOTE VERBATIM):\n"
        + "\n\n".join(context_lines)
    )
    br = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    resp = br.converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": REDACTION_SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"maxTokens": 1024},
    )
    answer = "".join(
        block.get("text", "") for block in resp["output"]["message"]["content"] if "text" in block
    ).strip()

    cited_slots = []
    for m in re.finditer(r"\[(S\d+)\]", answer):
        slot = m.group(1)
        if slot in slots and slot not in cited_slots:
            cited_slots.append(slot)
    if not cited_slots:
        cited_slots = list(slots.keys())[:3]

    slot_to_handle: dict[str, str] = {}
    citations: list[dict[str, str]] = []
    caller_hmac = cipher.hmac_facet("caller", caller["arn"])
    for slot in cited_slots:
        h = slots[slot]
        handle = make_citation_handle(OWNER)
        slot_to_handle[slot] = handle
        record = {
            "unit_id": h["unit_id"],
            "chunk_id": h["chunk_id"],
            "msg_idx": h["msg_idx"],
            "source": h["source"],
            "unit_name": h["unit_name"],
            "question": question,
            "caller_arn": caller["arn"],
        }
        payload_ct = cipher.encrypt_text(json.dumps(record), aad=f"citations:{handle}")
        con.execute(
            "INSERT INTO citations (handle, caller_hmac, payload_ct) VALUES (%s, %s, %s)",
            (handle, caller_hmac, payload_ct),
        )
        citations.append({"handle": handle, "source_type": "insight" if h["source"] == "dream" else h["source"]})
    con.commit()

    def _replace_slot(m: re.Match) -> str:
        slot = m.group(1)
        return f"[{slot_to_handle[slot]}]" if slot in slot_to_handle else ""

    final_answer = re.sub(r"\[(S\d+)\]", _replace_slot, answer)
    return {"owner": OWNER, "answer": final_answer, "citations": citations}


def _handle_resolve_citation(con: psycopg.Connection, cipher: ClyncCipher, params: dict[str, Any]) -> dict[str, Any]:
    handle = str(params.get("handle", "")).strip()
    row = con.execute("SELECT handle, created_at, payload_ct FROM citations WHERE handle = %s", (handle,)).fetchone()
    if not row:
        raise ValueError(f"citation handle not found: {handle!r}")
    meta = json.loads(cipher.decrypt_text(row["payload_ct"], aad=f"citations:{handle}"))
    unit_id = meta["unit_id"]
    center = meta.get("msg_idx")
    if center is not None:
        msgs = con.execute(
            "SELECT msg_idx, payload_ct FROM messages WHERE unit_id = %s AND msg_idx BETWEEN %s AND %s ORDER BY msg_idx",
            (unit_id, max(0, int(center) - 4), int(center) + 4),
        ).fetchall()
    else:
        msgs = con.execute(
            "SELECT msg_idx, payload_ct FROM messages WHERE unit_id = %s ORDER BY msg_idx LIMIT 12",
            (unit_id,),
        ).fetchall()
    lines = [
        f"Citation: {handle}",
        f"Unit    : [{meta.get('source')}] {meta.get('unit_name')} (uuid={unit_id}, msg_idx={center})",
        f"Asked by: {meta.get('caller_arn')} at {row['created_at'].isoformat()}",
        f"Question: {meta.get('question')}",
        "",
    ]
    for m in msgs:
        mp = json.loads(cipher.decrypt_text(m["payload_ct"], aad=f"messages:{unit_id}:{m['msg_idx']}"))
        marker = " <-- cited" if m["msg_idx"] == center else ""
        lines.append(f"## [{m['msg_idx']}] {mp.get('sender')}{marker}\n{mp.get('text', '')[:1200]}\n")
    return {"handle": handle, "meta": meta, "text": "\n".join(lines).strip()}


def _handle_get_conversation(con: psycopg.Connection, cipher: ClyncCipher, params: dict[str, Any]) -> dict[str, Any]:
    unit_id = str(params.get("uuid", "")).strip()
    urow = con.execute("SELECT unit_id, meta_ct FROM units WHERE unit_id = %s", (unit_id,)).fetchone()
    if not urow:
        raise ValueError(f"conversation not found: {unit_id!r}")
    umeta = json.loads(cipher.decrypt_text(urow["meta_ct"], aad=f"units:{unit_id}"))
    conds = ["unit_id = %s"]
    vals: list[Any] = [unit_id]
    if params.get("from_idx") is not None:
        conds.append("msg_idx >= %s")
        vals.append(int(params["from_idx"]))
    if params.get("to_idx") is not None:
        conds.append("msg_idx <= %s")
        vals.append(int(params["to_idx"]))
    if params.get("sender"):
        role_map = {"user": ["human", "user"], "assistant": ["assistant"]}
        senders = role_map.get(params["sender"], [params["sender"]])
        conds.append("sender_hmac = ANY(%s)")
        vals.append([cipher.hmac_facet("sender", s) for s in senders])
    msgs = con.execute(
        f"SELECT msg_idx, payload_ct FROM messages WHERE {' AND '.join(conds)} ORDER BY msg_idx",
        vals,
    ).fetchall()
    lines = [f"# {umeta.get('title') or '(untitled)'} [{umeta.get('source')}] (uuid={unit_id})"]
    for m in msgs:
        mp = json.loads(cipher.decrypt_text(m["payload_ct"], aad=f"messages:{unit_id}:{m['msg_idx']}"))
        lines.append(f"\n## [{m['msg_idx']}] {mp.get('sender')}\n{mp.get('text', '')}")
    return {"text": "\n".join(lines)}


def _handle_ingest(con: psycopg.Connection, params: dict[str, Any]) -> dict[str, Any]:
    s3_key = params.get("s3_key")
    ensure_hnsw = bool(params.get("ensure_hnsw", False))
    _ensure_owner_schema(con, create_hnsw=False)
    counts = {"units": 0, "messages": 0, "chunks": 0}
    if s3_key:
        if not STAGING_BUCKET or not str(s3_key).startswith(f"{OWNER}/"):
            raise PermissionError(f"s3_key {s3_key!r} outside owner prefix '{OWNER}/'")
        s3 = boto3.client("s3", region_name=AWS_REGION)
        raw_gz = s3.get_object(Bucket=STAGING_BUCKET, Key=s3_key)["Body"].read()
        lines = gzip.decompress(raw_gz).decode("utf-8").splitlines()
        with con.cursor() as cur:
            for line in lines:
                if not line.strip():
                    continue
                rec = json.loads(line)
                kind = rec["t"]
                if kind == "unit":
                    cur.execute(
                        "INSERT INTO units (unit_id, source_hmac, kind_hmac, repo_hmac, project_hmac, updated_at, meta_ct) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s) "
                        "ON CONFLICT (unit_id) DO UPDATE SET "
                        "source_hmac=EXCLUDED.source_hmac, kind_hmac=EXCLUDED.kind_hmac, "
                        "repo_hmac=EXCLUDED.repo_hmac, project_hmac=EXCLUDED.project_hmac, "
                        "updated_at=EXCLUDED.updated_at, meta_ct=EXCLUDED.meta_ct",
                        (
                            rec["unit_id"],
                            rec["source_hmac"],
                            rec["kind_hmac"],
                            rec.get("repo_hmac"),
                            rec.get("project_hmac"),
                            rec.get("updated_at"),
                            rec["meta_ct"],
                        ),
                    )
                    counts["units"] += 1
                elif kind == "msg":
                    cur.execute(
                        "INSERT INTO messages (unit_id, msg_idx, sender_hmac, created_at, payload_ct) "
                        "VALUES (%s,%s,%s,%s,%s) "
                        "ON CONFLICT (unit_id, msg_idx) DO UPDATE SET "
                        "sender_hmac=EXCLUDED.sender_hmac, created_at=EXCLUDED.created_at, payload_ct=EXCLUDED.payload_ct",
                        (
                            rec["unit_id"],
                            rec["msg_idx"],
                            rec.get("sender_hmac"),
                            rec.get("created_at"),
                            rec["payload_ct"],
                        ),
                    )
                    counts["messages"] += 1
                elif kind == "chunk":
                    cur.execute(
                        f"INSERT INTO chunks (chunk_id, unit_id, source_hmac, repo_hmac, project_hmac, "
                        f"sender_hmac, lang_hmac, msg_idx, chunk_idx, updated_at, payload_ct, dense, sparse) "
                        f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector({DENSE_DIM}),%s::sparsevec({SPARSE_DIM})) "
                        f"ON CONFLICT (chunk_id) DO UPDATE SET "
                        f"source_hmac=EXCLUDED.source_hmac, repo_hmac=EXCLUDED.repo_hmac, "
                        f"project_hmac=EXCLUDED.project_hmac, sender_hmac=EXCLUDED.sender_hmac, "
                        f"lang_hmac=EXCLUDED.lang_hmac, updated_at=EXCLUDED.updated_at, "
                        f"payload_ct=EXCLUDED.payload_ct, dense=EXCLUDED.dense, sparse=EXCLUDED.sparse",
                        (
                            rec["chunk_id"],
                            rec["unit_id"],
                            rec["source_hmac"],
                            rec.get("repo_hmac"),
                            rec.get("project_hmac"),
                            rec.get("sender_hmac"),
                            rec.get("lang_hmac"),
                            rec.get("msg_idx"),
                            rec.get("chunk_idx"),
                            rec.get("updated_at"),
                            rec["payload_ct"],
                            rec["dense"],
                            rec["sparse"],
                        ),
                    )
                    counts["chunks"] += 1
        con.commit()
        s3.delete_object(Bucket=STAGING_BUCKET, Key=s3_key)
    if ensure_hnsw:
        _ensure_owner_schema(con, create_hnsw=True)
        con.execute("ANALYZE chunks")
        con.commit()
    return {"ok": True, "ingested": counts, "hnsw_ensured": ensure_hnsw}


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    action = str((event or {}).get("action", "")).strip()
    params = (event or {}).get("params") or {}
    try:
        caller = _verify_caller((event or {}).get("auth"))
    except PermissionError as exc:
        return {ERROR_KEY: f"unauthorized: {exc}"}

    modes = []
    tools = []
    if ENABLE_SHARE_MCP:
        modes.append("ask")
        tools.append(f"ask_{OWNER}")
    if ENABLE_OWNER_MCP and caller["is_owner"]:
        modes.append("owner")
        tools.extend(["search_history", "search_transcripts", "get_conversation", "resolve_citation"])

    if action == "handshake":
        return {
            "owner": OWNER,
            "modes": modes,
            "tools": tools,
            "caller_arn": caller["arn"],
            "caller_user_id": caller["user_id"],
        }

    owner_actions = {"search_history", "search_transcripts", "get_conversation", "resolve_citation", "ingest"}
    if action in owner_actions:
        if not caller["is_owner"]:
            return {ERROR_KEY: f"action {action!r} requires owner identity (caller={caller['arn']})"}
        if action != "ingest" and not ENABLE_OWNER_MCP:
            return {ERROR_KEY: "owner MCP interface is disabled on this endpoint"}
    elif action == "ask":
        if not ENABLE_SHARE_MCP:
            return {ERROR_KEY: "read-only ask interface is disabled on this endpoint"}
    else:
        return {ERROR_KEY: f"unknown action {action!r}"}

    cipher = _get_cipher()
    with _connect_db() as con:
        try:
            if action == "ingest":
                return _handle_ingest(con, params)
            if action == "ask":
                return _handle_ask(con, cipher, caller, params)
            if action == "resolve_citation":
                return _handle_resolve_citation(con, cipher, params)
            if action == "get_conversation":
                return _handle_get_conversation(con, cipher, params)
            if action == "search_transcripts":
                hits = _hybrid_search(
                    con,
                    cipher,
                    str(params.get("query", "")),
                    topk=int(params.get("limit", 10)),
                    source=str(params.get("source", "all")),
                    repo=params.get("repo"),
                    sender=params.get("sender"),
                    per_unit=int(params.get("per_unit", 1)),
                )
                if not hits:
                    return {"text": f"No matches for {params.get('query', '')!r}.", "hits": []}
                lines = [f"{len(hits)} match(es) in {OWNER}'s cloud archive:\n"]
                for h in hits:
                    snippet = h["text"].replace("\n", " ")[:300]
                    lines.append(
                        f"- [{h['source']}] {h['unit_name'] or '(untitled)'} "
                        f"(uuid={h['unit_id']}, sender={h['sender']}, msg_idx={h['msg_idx']}, score={h['score']:.4f})\n"
                        f"  {snippet}"
                    )
                return {"text": "\n".join(lines), "hits": hits}
            if action == "search_history":
                q = str(params.get("query", ""))
                lim = int(params.get("limit", 10))
                ins = _hybrid_search(con, cipher, q, topk=lim, source="dream")
                raw = _hybrid_search(con, cipher, q, topk=lim, source="all")
                lines = ["=== INSIGHTS ==="]
                if ins:
                    for h in ins:
                        lines.append(f"- {h['text'][:300]} (uuid={h['unit_id']}, score={h['score']:.4f})")
                else:
                    lines.append("(no matching insights)")
                lines.append("\n=== RAW TRANSCRIPTS ===")
                if raw:
                    for h in raw:
                        snippet = h["text"].replace("\n", " ")[:300]
                        lines.append(
                            f"- [{h['source']}] {h['unit_name'] or '(untitled)'} "
                            f"(uuid={h['unit_id']}, sender={h['sender']}, msg_idx={h['msg_idx']}, score={h['score']:.4f})\n"
                            f"  {snippet}"
                        )
                else:
                    lines.append("(no matching transcripts)")
                return {"text": "\n".join(lines), "insights": ins, "raw": raw}
        except (ValueError, PermissionError) as exc:
            return {ERROR_KEY: f"{type(exc).__name__}: {exc}"}
    return {ERROR_KEY: "unhandled action"}
