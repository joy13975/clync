"""Contained hybrid search + indexing for clync.

clync runs its OWN Postgres 17 + pgvector cluster — a private `initdb` cluster
under ~/.local/share/clync/pg, on a non-default port, fully isolated from any
system Postgres. Synced messages (from the SQLite raw store) are chunked and
embedded with BGE-M3 (dense + learned sparse; ADR 0002) and served via a hybrid
dense + sparse + typed-metadata search with RRF fusion — no reranker (ADR 0001).

Heavy deps (FlagEmbedding/torch, psycopg) live in the `search` extra:
    uv sync --extra search

Fail-loud: missing binaries / pgvector / model / cluster problems raise. No
silent fallback to stale or partial results.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Reuse the SQLite raw store + shared config from the core module (SSOT).
from clync import DB_PATH, DEFAULT_TOPK, connect

# --------------------------------------------------------------------------- #
# Config (all overridable; nothing personal hardcoded)
# --------------------------------------------------------------------------- #
DATA_HOME = DB_PATH.parent                         # ~/.local/share/clync
PG_DIR = DATA_HOME / "pg"
PG_DATA = PG_DIR / "data"
PG_LOG = PG_DIR / "postmaster.log"
# pgvector is installed against PG17's share dir; the cluster MUST use those
# binaries (the PATH `initdb` may be a different major without pgvector).
PG_BIN = Path(os.environ.get("CLYNC_PG_BIN", "/opt/homebrew/opt/postgresql@17/bin"))
PG_PORT = int(os.environ.get("CLYNC_PG_PORT", "54329"))
PG_DB = os.environ.get("CLYNC_PG_DB", "clync")
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", PG_DB):
    raise SystemExit(f"invalid $CLYNC_PG_DB {PG_DB!r} — must match [A-Za-z_][A-Za-z0-9_]*")
PG_USER = os.environ.get("USER", "postgres")

EMBED_MODEL = "BAAI/bge-m3"
DENSE_DIM = 1024
SPARSE_DIM = 250002          # XLM-R vocab; BGE-M3 lexical weights are token ids
CHUNK_CHARS = 2000
CHUNK_OVERLAP = 200
RRF_K = 60
PRE_TOPK = 50                # per-signal shortlist before fusion
TOPK = DEFAULT_TOPK          # default result count (SSOT: clync.DEFAULT_TOPK)

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    text PRIMARY KEY,
    conv_uuid   text NOT NULL,
    conv_name   text,
    sender      text,
    msg_idx     int,
    chunk_idx   int,
    lang        text,                       -- typed metadata (en/ja/zh)
    created_at  timestamptz,
    updated_at  timestamptz,                -- conversation recency (metadata)
    text        text NOT NULL,
    dense       vector({DENSE_DIM}) NOT NULL,
    sparse      sparsevec({SPARSE_DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_conv ON chunks(conv_uuid);
CREATE INDEX IF NOT EXISTS idx_chunks_lang ON chunks(lang);
CREATE INDEX IF NOT EXISTS idx_chunks_updated ON chunks(updated_at);
-- Per-unit (conversation OR project-doc) index watermark, for incremental
-- re-embedding. The watermark is an opaque sha256 over the EXACT bytes embedded for
-- that unit (message/doc text plus the rendered `[Project] name` prefix), so a
-- change to any embedded input (doc edit, project rename, message edit) flips it and
-- the unit re-embeds on the default incremental run. It lives in the `updated_at`
-- column, which historically held an API timestamp -- keeping the column name means
-- a live cluster needs no schema migration, since only the watermark's derivation
-- changed, not its role. Always non-null, so it can never NULL-abort the INSERT.
CREATE TABLE IF NOT EXISTS indexed_convs (
    conv_uuid   text PRIMARY KEY,
    updated_at  text NOT NULL,          -- content signature (see note above)
    indexed_at  text NOT NULL
);
"""


# --------------------------------------------------------------------------- #
# Cluster lifecycle (contained; never touches a system cluster)
# --------------------------------------------------------------------------- #
def _pg(binary: str) -> str:
    path = PG_BIN / binary
    if not path.exists():
        raise SystemExit(
            f"Postgres 17 binary not found: {path}\n"
            f"clync runs its own PG17+pgvector cluster. Install with:\n"
            f"  brew install postgresql@17 pgvector\n"
            f"or set $CLYNC_PG_BIN to the PG17 bin dir.")
    return str(path)


def _vector_control_present() -> bool:
    # pgvector ships vector.control into PG17's extension share dir.
    share = PG_BIN.parent / "share" / "postgresql@17" / "extension" / "vector.control"
    alt = Path("/opt/homebrew/share/postgresql@17/extension/vector.control")
    return share.exists() or alt.exists()


def cluster_running() -> bool:
    r = subprocess.run([_pg("pg_ctl"), "-D", str(PG_DATA), "status"],
                       capture_output=True, text=True)
    return r.returncode == 0


def start_cluster() -> None:
    if cluster_running():
        return
    PG_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [_pg("pg_ctl"), "-D", str(PG_DATA), "-l", str(PG_LOG),
         "-o", f"-p {PG_PORT} -c listen_addresses=localhost "
               f"-c unix_socket_directories={PG_DIR}",
         "-w", "start"],
        check=True)


def stop_cluster() -> bool:
    """Stop the contained cluster if it's running. Returns True iff a running
    cluster was actually stopped; False (no-op) if there's no cluster data or the
    PG17 binary is absent — that's not an error."""
    if not (PG_DATA / "PG_VERSION").exists() or not (PG_BIN / "pg_ctl").exists():
        return False
    if not cluster_running():
        return False
    subprocess.run([_pg("pg_ctl"), "-D", str(PG_DATA), "-m", "fast", "stop"],
                   check=True)
    return True


def ensure_cluster() -> None:
    """Idempotent: initdb (if absent) → start → create db + extension + schema."""
    if not _vector_control_present():
        raise SystemExit(
            "pgvector not found for PG17. Install with:  brew install pgvector")
    if not (PG_DATA / "PG_VERSION").exists():
        PG_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [_pg("initdb"), "-D", str(PG_DATA), "-U", PG_USER,
             "--encoding=UTF8", "--locale=en_US.UTF-8", "-A", "trust"],
            check=True, capture_output=True)
    start_cluster()
    # create db if missing (connect to the always-present `postgres` db first)
    exists = subprocess.run(
        [_pg("psql"), "-h", "localhost", "-p", str(PG_PORT), "-d", "postgres",
         "-tAc", f"SELECT 1 FROM pg_database WHERE datname='{PG_DB}'"],
        capture_output=True, text=True, check=True).stdout.strip()
    if exists != "1":
        subprocess.run([_pg("psql"), "-h", "localhost", "-p", str(PG_PORT),
                        "-d", "postgres", "-c", f'CREATE DATABASE "{PG_DB}"'],
                       check=True, capture_output=True)
    with connect_pg() as con:
        with con.cursor() as cur:
            for stmt in (s.strip() for s in SCHEMA.split(";")):
                if stmt:
                    cur.execute(stmt)     # psycopg3 runs one statement per execute
        con.commit()


def connect_pg():
    import psycopg
    return psycopg.connect(host="localhost", port=PG_PORT, dbname=PG_DB, user=PG_USER)


def available() -> bool:
    """SSOT for 'is the `search` extra installed'. `import search` is NOT a valid
    check — this module imports fine without the extra (its heavy deps are lazy),
    so callers must gate on this, not on catching ImportError from `import search`."""
    import importlib.util
    return all(importlib.util.find_spec(m) is not None
               for m in ("psycopg", "FlagEmbedding", "numpy"))


def index_status() -> dict:
    """Structured health for `clync doctor` — REPORTS problems as data, never
    raises and never leaks the schema to the caller. Keys: available, pg_bin,
    cluster_running, chunks (int|None), error (str|None)."""
    st = {"available": available(), "pg_bin": (PG_BIN / "pg_ctl").exists(),
          "cluster_running": False, "chunks": None, "error": None}
    if not st["available"] or not st["pg_bin"]:
        return st
    import psycopg
    try:
        st["cluster_running"] = cluster_running()
        if st["cluster_running"]:
            with connect_pg() as pg:
                st["chunks"] = pg.execute("SELECT count(*) FROM chunks").fetchone()[0]
    except (psycopg.Error, OSError) as e:   # cluster down / schema absent / pg_ctl unrunnable — report, don't crash doctor
        st["error"] = str(e)
    return st


# --------------------------------------------------------------------------- #
# Corpus + chunking (mirrors the ablation harness — identical chunking)
# --------------------------------------------------------------------------- #
def _lang(s: str) -> str:
    s = s or ""
    if re.search(r"[぀-ヿ]", s):   # kana → Japanese
        return "ja"
    if re.search(r"[一-鿿]", s):   # Han without kana → Chinese
        return "zh"
    return "en"


def _pieces(text: str) -> list[str]:
    step = CHUNK_CHARS - CHUNK_OVERLAP
    # range(0, max(1, len(text)), step) always includes index 0 -> >=1 piece.
    return [text[i:i + CHUNK_CHARS] for i in range(0, max(1, len(text)), step)]


# The chunks-table column shape (all columns except the two embedding vectors).
# SSOT for BOTH the row dicts the loaders build (_chunk_row) AND the INSERT column
# list in build_index — add/rename a column in exactly one place.
CHUNK_COLS = ("chunk_id", "conv_uuid", "conv_name", "sender", "msg_idx",
              "chunk_idx", "lang", "created_at", "updated_at", "text")


def _chunk_row(*, chunk_id: str, conv_uuid: str, conv_name: str, sender: str,
               msg_idx: int, chunk_idx: int, created_at, updated_at,
               text: str) -> dict:
    """Build one chunks-table row dict — the single definition of the row shape,
    shared by both loaders. `lang` is derived from the embedded `text`."""
    return {"chunk_id": chunk_id, "conv_uuid": conv_uuid, "conv_name": conv_name,
            "sender": sender, "msg_idx": msg_idx, "chunk_idx": chunk_idx,
            "lang": _lang(text), "created_at": created_at, "updated_at": updated_at,
            "text": text}


def _load_chunks(sqlite_con) -> list[dict]:
    # LEFT JOIN projects so a project-associated chat carries its project name into
    # both the embedded text and the displayed result (e.g. "[Work] CNX sour guy…").
    q = ("SELECT m.uuid mid, m.conversation_uuid cid, m.sender, m.text, m.idx, "
         "m.created_at, c.name cname, c.updated_at cupd, pr.name pname "
         "FROM messages m JOIN conversations c ON c.uuid=m.conversation_uuid "
         "LEFT JOIN projects pr ON pr.uuid=c.project_uuid "
         "WHERE m.text IS NOT NULL AND m.text != ''")
    chunks = []
    for r in sqlite_con.execute(q).fetchall():
        prefix = f"[{r['pname']}] " if r["pname"] else ""
        name = f"{prefix}{r['cname'] or 'untitled'}"
        for j, piece in enumerate(_pieces(r["text"])):
            chunks.append(_chunk_row(
                chunk_id=f"{r['mid']}#{j}", conv_uuid=r["cid"], conv_name=name,
                sender=r["sender"], msg_idx=r["idx"], chunk_idx=j,
                created_at=r["created_at"], updated_at=r["cupd"],
                text=f"{name} | {r['sender']}: {piece}"))
    return chunks


def _load_doc_chunks(sqlite_con) -> list[dict]:
    """Project knowledge docs as retrievable units — indexed into the same `chunks`
    table as conversations (conv_uuid = doc uuid), so hybrid_search returns them
    alongside chats with no query-side change."""
    q = ("SELECT d.uuid did, d.file_name fname, d.content, d.created_at, pr.name pname "
         "FROM project_docs d JOIN projects pr ON pr.uuid=d.project_uuid "
         "WHERE d.content IS NOT NULL AND d.content != ''")
    chunks = []
    for r in sqlite_con.execute(q).fetchall():
        name = f"[{r['pname']}] {r['fname'] or 'document'}"
        for j, piece in enumerate(_pieces(r["content"])):
            chunks.append(_chunk_row(
                chunk_id=f"{r['did']}#{j}", conv_uuid=r["did"], conv_name=name,
                sender="project_doc", msg_idx=0, chunk_idx=j,
                created_at=r["created_at"], updated_at=r["created_at"],
                text=f"{name}: {piece}"))
    return chunks


# --------------------------------------------------------------------------- #
# Embedding (lazy, warm-cached in-process)
# --------------------------------------------------------------------------- #
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from FlagEmbedding import BGEM3FlagModel
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _MODEL = BGEM3FlagModel(EMBED_MODEL, use_fp16=True, devices=dev)
    return _MODEL


def _embed(texts: list[str], max_length: int) -> tuple[list[list[float]], list[dict]]:
    out = _model().encode(texts, batch_size=16, return_dense=True,
                          return_sparse=True, max_length=max_length)
    import numpy as np
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    dense /= (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9)
    sparse = [{int(k): float(v) for k, v in d.items()} for d in out["lexical_weights"]]
    return dense.tolist(), sparse


def _dense_literal(vec) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


def _sparse_literal(weights: dict) -> str:
    # pgvector sparsevec: 1-based indices. BGE token ids are 0-based → +1.
    items = sorted((int(t) + 1, w) for t, w in weights.items() if w > 0)
    if not items:                       # a chunk with no positive weights is a bug
        raise ValueError("empty sparse vector")
    body = ",".join(f"{i}:{w:.7g}" for i, w in items)
    return f"{{{body}}}/{SPARSE_DIM}"


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def _unit_signature(chunks: list[dict]) -> str:
    """Deterministic content signature for one unit (conversation or project doc):
    a sha256 over the EXACT bytes embedded for it — each chunk's id plus its `text`,
    where `text` already carries the rendered `[Project] name` prefix. Any change to
    the embedded content — an in-place doc edit, a project rename (changes the
    prefix), a message edit, an added/removed chunk — flips the signature, so the
    unit re-embeds on the default incremental run. A unit with no chunks hashes to a
    stable constant. Never NULL, so it can never NULL-abort the indexed_convs INSERT
    nor collapse the incremental staleness test (a real value always != None)."""
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda c: c["chunk_id"]):
        h.update(c["chunk_id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(c["text"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def build_index(full: bool = False) -> dict:
    """Embed new/changed conversations AND project knowledge docs into the contained
    cluster. A unit is (re)embedded whenever the exact bytes folded into its chunks
    change — tracked by a content signature over those bytes (`_unit_signature`),
    NOT an API timestamp — so an in-place doc edit, a project rename, or a message
    edit all mark the unit stale on the default incremental run. `full` re-embeds
    everything. Both kinds are indexed into the same `chunks` table as retrievable
    units (indexed_convs.conv_uuid tracks either). Returns stats."""
    ensure_cluster()
    sq = connect()
    # The universe of units is EVERY conversation + project doc (incl. ones whose
    # text is now empty -> zero chunks), so an emptied unit still gets purged and its
    # watermark advanced rather than reindexing forever.
    universe = {r[0] for r in sq.execute("SELECT uuid FROM conversations").fetchall()}
    universe |= {r[0] for r in sq.execute("SELECT uuid FROM project_docs").fetchall()}
    # Load every unit's chunks (local SQLite read — cheap; embedding is the only
    # expensive step and is confined to the stale units below). The chunk `text` is
    # the exact byte string that gets embedded, so a signature over it captures every
    # embedded-content change with no dependence on any API timestamp.
    chunks = _load_chunks(sq) + _load_doc_chunks(sq)
    sq.close()
    by_unit: dict[str, list[dict]] = {}
    for c in chunks:
        by_unit.setdefault(c["conv_uuid"], []).append(c)
    sigs = {u: _unit_signature(by_unit.get(u, [])) for u in universe}
    with connect_pg() as pg:
        indexed = {r[0]: r[1] for r in
                   pg.execute("SELECT conv_uuid, updated_at FROM indexed_convs").fetchall()}
        if full:
            stale = set(universe)
        else:
            stale = {u for u in universe if indexed.get(u) != sigs[u]}
        # drop index rows for units that vanished from the raw store entirely
        gone = set(indexed) - universe
        # Gate on stale/gone, NOT on chunks: a stale unit that now yields zero
        # chunks (emptied messages / empty doc) still needs its old chunks purged
        # and its watermark advanced — otherwise stale rows linger + it reindexes
        # every run.
        if not stale and not gone:
            return {"reindexed_convs": 0, "chunks": 0, "removed_convs": 0}

        # Embed only the stale units' chunks (the whole corpus is already loaded).
        stale_chunks = [c for c in chunks if c["conv_uuid"] in stale]
        dense, sparse = ([], [])
        if stale_chunks:
            dense, sparse = _embed([c["text"] for c in stale_chunks], max_length=2048)

        cols = ", ".join(CHUNK_COLS)
        placeholders = ",".join(["%s"] * len(CHUNK_COLS)) + ",%s::vector,%s::sparsevec"
        now = datetime.now(timezone.utc).isoformat()
        with pg.cursor() as cur:
            for u in stale | gone:
                cur.execute("DELETE FROM chunks WHERE conv_uuid=%s", (u,))
            for c, d, s in zip(stale_chunks, dense, sparse):
                cur.execute(
                    f"INSERT INTO chunks ({cols}, dense, sparse) VALUES ({placeholders})",
                    tuple(c[k] for k in CHUNK_COLS)
                    + (_dense_literal(d), _sparse_literal(s)))
            for u in gone:
                cur.execute("DELETE FROM indexed_convs WHERE conv_uuid=%s", (u,))
            for u in stale:
                cur.execute(
                    "INSERT INTO indexed_convs (conv_uuid, updated_at, indexed_at) "
                    "VALUES (%s,%s,%s) ON CONFLICT(conv_uuid) DO UPDATE SET "
                    "updated_at=excluded.updated_at, indexed_at=excluded.indexed_at",
                    (u, sigs[u], now))
        pg.commit()
    # reindexed_convs counts all reindexed UNITS (conversations + project docs).
    return {"reindexed_convs": len(stale), "chunks": len(stale_chunks),
            "removed_convs": len(gone)}


# --------------------------------------------------------------------------- #
# Hybrid search — dense + sparse RRF, typed-metadata boost, collapse to convs
# --------------------------------------------------------------------------- #
def hybrid_search(query: str, topk: int = TOPK, lang: str | None = None) -> list[dict]:
    """Return up to `topk` conversations ranked by hybrid relevance. Each result
    is the best-matching chunk of a distinct conversation. An empty/degenerate
    query returns [] rather than raising."""
    if not query.strip():
        return []
    ensure_cluster()
    (qd,), (qs,) = _embed([query], max_length=512)
    qlang = _lang(query)
    # A query with no positive sparse weights (rare — punctuation/stopword-only)
    # still has a dense vector: fall back to dense-only rather than crash.
    try:
        qs_lit = _sparse_literal(qs)
    except ValueError:
        qs_lit = None
    lang_filter = "WHERE lang = %(lang)s" if lang else ""
    params: dict = {"qd": _dense_literal(qd), "qlang": qlang, "lang": lang}

    # RRF over the per-signal shortlists, then a small typed-metadata boost
    # (recency + query/chunk language agreement), then collapse to conversations.
    ctes = [f"""d AS (
        SELECT chunk_id, row_number() OVER (ORDER BY dense <=> %(qd)s::vector) rnk
        FROM chunks {lang_filter} ORDER BY dense <=> %(qd)s::vector LIMIT {PRE_TOPK})"""]
    unions = ["SELECT * FROM d"]
    if qs_lit is not None:
        params["qs"] = qs_lit
        ctes.append(f"""s AS (
        SELECT chunk_id, row_number() OVER (ORDER BY sparse <#> %(qs)s::sparsevec) rnk
        FROM chunks {lang_filter} ORDER BY sparse <#> %(qs)s::sparsevec LIMIT {PRE_TOPK})""")
        unions.append("SELECT * FROM s")
    ctes.append(f"""fused AS (
        SELECT chunk_id, SUM(1.0/({RRF_K}+rnk)) AS rrf
        FROM ({' UNION ALL '.join(unions)}) u GROUP BY chunk_id)""")
    ctes.append("""scored AS (
        SELECT c.conv_uuid, c.conv_name, c.lang, c.text,
               f.rrf * (1.0
                   -- recency: 6-month 1/e time-constant (~4-month half-life);
                   -- a missing updated_at contributes 0, not a max boost.
                   + CASE WHEN c.updated_at IS NULL THEN 0
                          ELSE 0.10 * exp(-GREATEST(0, EXTRACT(EPOCH FROM (now() - c.updated_at)))
                                          / (86400.0*180)) END
                   + CASE WHEN c.lang = %(qlang)s THEN 0.05 ELSE 0 END) AS score
        FROM fused f JOIN chunks c ON c.chunk_id = f.chunk_id)""")
    sql = ("WITH " + ",\n".join(ctes)
           + "\nSELECT DISTINCT ON (conv_uuid) conv_uuid, conv_name, lang, text, score"
             "\nFROM scored ORDER BY conv_uuid, score DESC")
    with connect_pg() as pg:
        rows = pg.execute(sql, params).fetchall()
    cols = ["conv_uuid", "conv_name", "lang", "text", "score"]
    results = [dict(zip(cols, r)) for r in rows]
    results.sort(key=lambda r: -r["score"])
    return results[:topk]
