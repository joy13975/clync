"""Hybrid search + indexing over clync's contained Postgres store.

The store (cluster lifecycle, connection, raw `units`/`messages` schema) is owned
by clync.py; this module owns the *index*: it creates the vector `chunks` /
`indexed_units` tables (it pins the embedding dims), embeds `messages` with BGE-M3
(dense + learned sparse; ADR 0002), and serves a **faceted** hybrid dense + sparse
+ typed-metadata search with RRF fusion — no reranker (ADR 0001).

Fail-loud: a missing PG17/pgvector toolchain, model, or cluster problem raises. No
silent fallback to stale or partial results.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from datetime import datetime

# The store + shared config live in the core module (SSOT). Search is core now
# (ADR 0003): its deps are no longer an optional extra.
import cc
from clync import (DEFAULT_TOPK, PG_BIN, RAW_SOURCES, connect_pg, limit_resources,
                   require_rebuild, resolve_sources, sql_statements,
                   _vector_control_present)

EMBED_MODEL = "BAAI/bge-m3"
# Where the embedder runs. Unset picks the GPU when Metal is available; "cpu" keeps
# the sync off the GPU entirely, which is the only real way to bound GPU use.
EMBED_DEVICE = os.environ.get("CLYNC_EMBED_DEVICE")
DENSE_DIM = 1024
SPARSE_DIM = 250002          # XLM-R vocab; BGE-M3 lexical weights are token ids
CHUNK_CHARS = 2000
CHUNK_OVERLAP = 200
EMBED_CALL_TEXTS = 256       # texts per embed call (see `_embed`)
RRF_K = 60
PRE_TOPK = 50                # per-signal shortlist before fusion
TOPK = DEFAULT_TOPK          # default result count (SSOT: clync.DEFAULT_TOPK)

# ANN serving of the two legs (ADR 0005 has the measurements behind each value).
PGVECTOR_MIN_VERSION = (0, 8, 0)         # required for hnsw.iterative_scan and sparsevec HNSW
PGVECTOR_SAFE_VACUUM_VERSION = (0, 8, 4) # 0.8.3/0.8.4 fixed HNSW graph corruption under VACUUM
PGVECTOR_EF_SEARCH_MAX = 1000            # above this pgvector silently reverts ef_search to 40
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 800
EXACT_SELECTION_MAX = 30000              # selections at or below this run exact to keep recall=1.0
SPARSE_MAX_NNZ = 1000                    # pgvector's limit for sparsevec in an HNSW index
HNSW_ELEMENT_OVERHEAD_BYTES = 2000       # per-element graph link overhead at m=32 (verified spill-free at 900 MB / 200k rows)
DENSE_KEY = f"(dense::halfvec({DENSE_DIM}))"
# name -> (index expression + opclass, estimated bytes per indexed value)
ANN_INDEXES = {
    "idx_chunks_dense_hnsw": (f"({DENSE_KEY} halfvec_cosine_ops)", 2 * DENSE_DIM),
    "idx_chunks_sparse_hnsw": ("(sparse sparsevec_ip_ops)", 16 + 8 * SPARSE_MAX_NNZ),
}


def check_ef_search(ef: int) -> None:
    """Fail loud on an ef_search pgvector would not honour as given: below the shortlist
    size a leg returns short, and above its maximum the extension silently reverts to 40."""
    if not PRE_TOPK <= ef <= PGVECTOR_EF_SEARCH_MAX:
        raise ValueError(
            f"hnsw ef_search={ef} must be within {PRE_TOPK}..{PGVECTOR_EF_SEARCH_MAX}: "
            f"below the shortlist size a leg returns short; above the maximum pgvector "
            f"silently reverts to its default of 40")


check_ef_search(HNSW_EF_SEARCH)

INDEX_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     text PRIMARY KEY,
    unit_id      text NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
    kind         text,
    source       text,
    unit_name    text,
    title        text,
    sender       text,
    msg_idx      int,
    chunk_idx    int,
    lang         text,
    created_at   timestamptz,
    updated_at   timestamptz,                 -- unit recency (metadata boost)
    model        text,
    project_uuid text,
    project_name text,
    repo         text,
    worktree     text,
    git_branch   text,
    entrypoint   text,
    text         text NOT NULL,
    dense        vector({DENSE_DIM}) NOT NULL,
    sparse       sparsevec({SPARSE_DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_unit    ON chunks(unit_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source  ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_chunks_lang    ON chunks(lang);
CREATE INDEX IF NOT EXISTS idx_chunks_updated ON chunks(updated_at);
CREATE INDEX IF NOT EXISTS idx_chunks_repo    ON chunks(repo);
CREATE INDEX IF NOT EXISTS idx_chunks_pname   ON chunks(project_name);
-- Per-unit watermark for the UNIT-derived columns only (`_unit_meta_signature`);
-- re-embedding is decided per chunk against `chunks.text`, which is the exact
-- string embedded.
CREATE TABLE IF NOT EXISTS indexed_units (
    unit_id    text PRIMARY KEY REFERENCES units(unit_id) ON DELETE CASCADE,
    signature  text NOT NULL,
    indexed_at text NOT NULL
);
"""


def ensure_index_schema(*, rebuild: bool = False) -> None:
    """Create the vector index tables (idempotent). Assumes the cluster + `vector`
    extension exist (clync.ensure_cluster provisions them).

    The index tables are DERIVED and rebuildable, so ANY stale layout — the
    pre-ADR-0003 conversation-keyed index (`indexed_convs`), or a `chunks`
    column set that no longer matches INDEX_SCHEMA — needs dropping and recreating
    (watermarks included, so everything re-embeds); CREATE IF NOT EXISTS alone
    would silently keep the old shape and the first insert would fail. But that
    drop is data loss, so it never happens implicitly: a stale shape RAISES unless
    `rebuild=True` (see `clync.require_rebuild`)."""
    with connect_pg() as con:
        have = {r["column_name"] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='chunks'").fetchall()}
        legacy = con.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name='indexed_convs'").fetchone()
        if legacy or (have and have != set(CHUNK_COLS) | {"dense", "sparse"}):
            require_rebuild(rebuild, layer="search index",
                            cost="every embedding (a full re-embed of the corpus)")
            con.execute("DROP TABLE IF EXISTS chunks")
            con.execute("DROP TABLE IF EXISTS indexed_convs")
            con.execute("DROP TABLE IF EXISTS indexed_units")
        for stmt in sql_statements(INDEX_SCHEMA):
            con.execute(stmt)
        pgvector_version = _ensure_pgvector(con)
        _ensure_ann_indexes(con, pgvector_version)
        for table in ("chunks", "indexed_units"):
            constraint = table + "_unit_id_fkey"
            exists = "SELECT 1 FROM pg_constraint WHERE conrelid=%s::regclass AND conname=%s"
            if not con.execute(exists, (table, constraint)).fetchone():
                con.execute(f"LOCK TABLE {table} IN SHARE ROW EXCLUSIVE MODE")
                if not con.execute(exists, (table, constraint)).fetchone():
                    con.execute(f"DELETE FROM {table} WHERE NOT EXISTS "
                                f"(SELECT 1 FROM units WHERE units.unit_id={table}.unit_id)")
                    con.execute(f"ALTER TABLE {table} ADD CONSTRAINT {constraint} "
                                "FOREIGN KEY (unit_id) REFERENCES units(unit_id) ON DELETE CASCADE")
        con.commit()


def _pgvector_version_tuple(version_str: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version_str.split("."))


def _pgvector_vacuum_warning(version_str: str) -> str | None:
    if _pgvector_version_tuple(version_str) < PGVECTOR_SAFE_VACUUM_VERSION:
        return (
            f"pgvector {version_str} is below "
            f"{'.'.join(map(str, PGVECTOR_SAFE_VACUUM_VERSION))} (VACUUM HNSW fixes); "
            "run `clync index --full` if an index error occurs after heavy re-indexing"
        )
    return None


def _ensure_pgvector(con) -> str:
    """Update the extension to the installed binary and refuse < PGVECTOR_MIN_VERSION."""
    row = con.execute("SELECT installed_version, default_version FROM pg_available_extensions "
                      "WHERE name = 'vector'").fetchone()
    if row["installed_version"] != row["default_version"]:
        con.execute("ALTER EXTENSION vector UPDATE")
    version_str = row["default_version"]
    if _pgvector_version_tuple(version_str) < PGVECTOR_MIN_VERSION:
        raise RuntimeError(
            f"pgvector {version_str} is older than "
            f"{'.'.join(map(str, PGVECTOR_MIN_VERSION))}, which HNSW iterative scans need.")
    return version_str


def _hnsw_build_memory_kb(elements: int, value_bytes: int) -> int:
    """maintenance_work_mem (kB) so the HNSW graph builds in memory without spilling."""
    graph = elements * (value_bytes + 8 + HNSW_ELEMENT_OVERHEAD_BYTES)
    return (graph * 6 // 5 + 4 * 1024 * 1024) // 1024


def _ensure_ann_indexes(con, pgvector_version: str) -> None:
    """Create each missing HNSW index with an in-memory build budget."""
    existing = {r["indexname"] for r in con.execute(
        "SELECT indexname FROM pg_indexes WHERE schemaname='public' AND tablename='chunks'")}
    missing = [name for name in ANN_INDEXES if name not in existing]
    if not missing:
        return
    warning = _pgvector_vacuum_warning(pgvector_version)
    if warning:
        print(f"clync: {warning}", file=sys.stderr)
    rows = con.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    for name in missing:
        key, value_bytes = ANN_INDEXES[name]
        con.execute("SELECT set_config('maintenance_work_mem', %s, true)",
                    (f"{_hnsw_build_memory_kb(rows, value_bytes)}kB",))
        con.execute(
            f"CREATE INDEX {name} ON chunks USING hnsw {key} "
            f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
        )


def available() -> bool:
    """Can hybrid search actually run on this machine? Deps are core now, so this
    is the PG17 + pgvector toolchain check (the thing that can genuinely be absent),
    not an import gate."""
    import importlib.util
    deps = all(importlib.util.find_spec(m) is not None
               for m in ("psycopg", "FlagEmbedding", "numpy"))
    return deps and (PG_BIN / "pg_ctl").exists() and _vector_control_present()


def index_status() -> dict:
    """Structured health for `clync doctor` — reports problems as data, never
    raises. Keys: available, cluster_running, chunks (int|None), pgvector_warning, error."""
    import clync
    st = {"available": available(),
          "cluster_running": False, "chunks": None, "pgvector_warning": None, "error": None}
    if not st["available"]:
        return st
    import psycopg
    try:
        st["cluster_running"] = clync.cluster_running()
        if st["cluster_running"]:
            with connect_pg() as pg:
                st["chunks"] = pg.execute("SELECT count(*) AS n FROM chunks"
                                          ).fetchone()["n"]
                ext = pg.execute("SELECT extversion FROM pg_extension "
                                 "WHERE extname = 'vector'").fetchone()
                if ext:
                    st["pgvector_warning"] = _pgvector_vacuum_warning(ext["extversion"])
    except (psycopg.Error, OSError) as e:   # cluster down / schema absent — report
        st["error"] = str(e)
    return st


# --------------------------------------------------------------------------- #
# Corpus + chunking
# --------------------------------------------------------------------------- #
def _lang(s: str) -> str:
    s = s or ""
    if re.search(r"[぀-ヿ]", s):   # kana -> Japanese
        return "ja"
    if re.search(r"[一-鿿]", s):   # Han without kana -> Chinese
        return "zh"
    return "en"


def _pieces(text: str) -> list[str]:
    step = CHUNK_CHARS - CHUNK_OVERLAP
    return [text[i:i + CHUNK_CHARS] for i in range(0, max(1, len(text)), step)]


def _display_name(kind, title, project_name, repo) -> str:
    """The human-facing unit label shown in results and folded into embed text."""
    if kind in ("cc_session", "codex_session", "opencode_session"):
        base = title or "session"
        return f"[{repo}] {base}" if repo else base
    base = title or ("document" if kind == "project_doc" else "untitled")
    return f"[{project_name}] {base}" if project_name else base


# All chunk columns except the two embedding vectors — SSOT for both the row dicts
# the loader builds and the INSERT column list in build_index.
CHUNK_COLS = ("chunk_id", "unit_id", "kind", "source", "unit_name", "title",
              "sender", "msg_idx", "chunk_idx", "lang", "created_at", "updated_at",
              "model", "project_uuid", "project_name", "repo", "worktree",
              "git_branch", "entrypoint", "text")
# The subset of CHUNK_COLS a chunk inherits from its UNIT rather than its own
# message — refreshed without re-embedding (see `_unit_meta_signature`).
_UNIT_COLS = ("kind", "source", "unit_name", "title", "updated_at", "model",
              "project_uuid", "project_name", "repo", "worktree", "git_branch",
              "entrypoint")


def _load_chunks(con) -> list[dict]:
    """Every retrievable unit's messages -> chunk rows. The `text` field is the
    exact string embedded: `<display name> | <sender>: <piece>` (project docs use
    `<display name>: <piece>`). A message's tighter `embed_text` overrides `text`;
    an empty `embed_text` (pure tool output) contributes no chunk."""
    q = ("SELECT m.msg_id, m.unit_id, m.sender, m.idx, "
         "COALESCE(m.embed_text, m.text) AS etext, m.created_at, "
         "u.kind, u.source, u.title, u.updated_at, u.model, u.project_uuid, "
         "u.project_name, u.repo, u.worktree, u.git_branch, u.entrypoint "
         "FROM messages m JOIN units u ON u.unit_id = m.unit_id "
         "WHERE COALESCE(m.embed_text, m.text) IS NOT NULL "
         "AND COALESCE(m.embed_text, m.text) != ''")
    chunks: list[dict] = []
    for r in con.execute(q).fetchall():
        name = _display_name(r["kind"], r["title"], r["project_name"], r["repo"])
        prefix = f"{name}: " if r["kind"] == "project_doc" else f"{name} | {r['sender']}: "
        for j, piece in enumerate(_pieces(r["etext"])):
            text = f"{prefix}{piece}"
            # Message identity is (unit_id, msg_id) — a cc resume stores the same
            # event uuid under both sessions — so the chunk id must be unit-scoped.
            chunks.append({
                "chunk_id": f"{r['unit_id']}:{r['msg_id']}#{j}", "unit_id": r["unit_id"],
                "kind": r["kind"], "source": r["source"], "unit_name": name,
                "title": r["title"], "sender": r["sender"], "msg_idx": r["idx"],
                "chunk_idx": j, "lang": _lang(text), "created_at": r["created_at"],
                "updated_at": r["updated_at"], "model": r["model"],
                "project_uuid": r["project_uuid"], "project_name": r["project_name"],
                "repo": r["repo"], "worktree": r["worktree"],
                "git_branch": r["git_branch"], "entrypoint": r["entrypoint"],
                "text": text})
    return chunks


# --------------------------------------------------------------------------- #
# Embedding (lazy, warm-cached in-process)
# --------------------------------------------------------------------------- #
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        # Before the imports: the thread pools below are sized at load time.
        threads = int(limit_resources()["OMP_NUM_THREADS"])
        from FlagEmbedding import BGEM3FlagModel
        import torch
        torch.set_num_threads(threads)          # OpenMP honours the env; torch's own
        torch.set_num_interop_threads(threads)  # two pools do not, and default to all
        dev = EMBED_DEVICE or ("mps" if torch.backends.mps.is_available() else "cpu")
        _MODEL = BGEM3FlagModel(EMBED_MODEL, use_fp16=True, devices=dev)
    return _MODEL


def _embed(texts: list[str], max_length: int) -> tuple[list[list[float]], list[dict]]:
    """Embed `texts`, in calls of at most `EMBED_CALL_TEXTS`, reporting progress once
    the work spans more than one call.

    `encode` tokenises and pads whatever list it is handed before any forward pass,
    so the call size — not the batch size it uses internally — sets the peak. The
    bound lives here rather than at the callers so none of them can lift it.

    Progress goes to stderr because the calling thread blocks in the GPU dispatch
    while it works: the process reports near 0% CPU throughout, so progress is the
    only liveness signal a long run has."""
    model = _model()          # loads torch under the CPU cap; numpy imports after,
    import numpy as np        # so Accelerate's BLAS pool reads the cap at load too
    dense_parts, sparse = [], []
    for i in range(0, len(texts), EMBED_CALL_TEXTS):
        out = model.encode(texts[i:i + EMBED_CALL_TEXTS], batch_size=16,
                           return_dense=True, return_sparse=True,
                           max_length=max_length)
        dense_parts.append(np.asarray(out["dense_vecs"], dtype=np.float32))
        sparse += [{int(k): float(v) for k, v in d.items()}
                   for d in out["lexical_weights"]]
        if len(texts) > EMBED_CALL_TEXTS:
            print(f"embedded {len(sparse)}/{len(texts)}", file=sys.stderr, flush=True)
    dense = np.vstack(dense_parts)
    dense /= (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9)
    return dense.tolist(), sparse


def _dense_literal(vec) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


def _sparse_literal(weights: dict) -> str:
    items = sorted((int(t) + 1, w) for t, w in weights.items() if w > 0)  # 0- -> 1-based
    if not items:
        raise ValueError("empty sparse vector")
    if len(items) > SPARSE_MAX_NNZ:
        # The HNSW index refuses a wider vector, and one chunk must not stop the corpus
        # indexing: keep the heaviest tokens (ties by token id) and say so.
        print(f"sparse vector has {len(items)} tokens; keeping the {SPARSE_MAX_NNZ} "
              f"heaviest, dropping {len(items) - SPARSE_MAX_NNZ}", file=sys.stderr)
        items = sorted(sorted(items, key=lambda item: (-item[1], item[0]))[:SPARSE_MAX_NNZ])
    body = ",".join(f"{i}:{w:.7g}" for i, w in items)
    return f"{{{body}}}/{SPARSE_DIM}"


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def _unit_meta_signature(chunks: list[dict]) -> str:
    """Signature over the UNIT-derived columns a chunk row carries. Those values are
    identical across a unit's chunks by construction, so one chunk answers for all.

    Re-embedding is decided per chunk, by `text` (below). This covers the other
    half: a bumped `updated_at` changes no chunk's text but does change every row's
    metadata, and `updated_at` feeds the recency boost per chunk — so a unit whose
    rows are left half-refreshed scores its own chunks inconsistently."""
    first = chunks[0] if chunks else {}
    h = hashlib.sha256()
    for col in _UNIT_COLS:
        h.update(f"{col}={first.get(col)!r}".encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def build_index(full: bool = False) -> dict:
    """Embed new/changed CHUNKS into the contained cluster. `full` re-embeds
    everything. Returns stats.

    Staleness is per chunk, and a chunk's `text` is the exact string embedded, so
    the stored text IS the signature: a message the change never touched keeps
    byte-identical text and keeps its vectors. Deciding it per UNIT instead re-embeds
    a whole session for one edited message.

    Columns a chunk carries but does not embed are refreshed without embedding:
    unit-derived ones per unit (`_unit_meta_signature`), message-derived position
    ones (`msg_idx`, `created_at`) per chunk when a text-stable message is renumbered.

    The store must already be provisioned (callers do it at command entry)."""
    with connect_pg() as con:
        universe = {r["unit_id"] for r in
                    con.execute("SELECT unit_id FROM units").fetchall()}
        chunks = _load_chunks(con)
        by_unit: dict[str, list[dict]] = {}
        for c in chunks:
            by_unit.setdefault(c["unit_id"], []).append(c)
        stored = {r["chunk_id"]: (r["text"], r["msg_idx"], r["created_at"]) for r in
                  con.execute(
                      "SELECT chunk_id, text, msg_idx, created_at FROM chunks").fetchall()}
        metas = {u: _unit_meta_signature(by_unit.get(u, [])) for u in universe}
        indexed = {r["unit_id"]: r["signature"] for r in
                   con.execute("SELECT unit_id, signature FROM indexed_units").fetchall()}

        stale_chunks = chunks if full else [
            c for c in chunks if c["chunk_id"] not in stored
            or stored[c["chunk_id"]][0] != c["text"]]
        dropped = set(stored) - {c["chunk_id"] for c in chunks}
        # A chunk's `msg_idx`/`created_at` come from its message, not its `text` and
        # not its unit, so neither the per-chunk text check nor the per-unit metadata
        # signature catches a text-stable message that was renumbered — reposition it
        # in place, no re-embed.
        reposition = [] if full else [
            c for c in chunks if c["chunk_id"] in stored
            and stored[c["chunk_id"]][0] == c["text"]
            and stored[c["chunk_id"]][1:] != (c["msg_idx"], c["created_at"])]
        refresh = set() if full else {u for u in universe if indexed.get(u) != metas[u]}
        gone = set(indexed) - universe
        if not stale_chunks and not dropped and not reposition and not refresh and not gone:
            return {"reindexed_units": 0, "chunks": 0, "removed_units": 0,
                    "dropped_chunks": 0, "repositioned_chunks": 0}

        dense, sparse = ([], [])
        if stale_chunks:
            dense, sparse = _embed([c["text"] for c in stale_chunks], max_length=2048)

        cols = ", ".join(CHUNK_COLS)
        placeholders = ",".join(["%s"] * len(CHUNK_COLS)) + ",%s::vector,%s::sparsevec"
        now = datetime.now().astimezone().isoformat()
        for u in gone:
            con.execute("DELETE FROM chunks WHERE unit_id=%s", (u,))
        for chunk_id in dropped:
            con.execute("DELETE FROM chunks WHERE chunk_id=%s", (chunk_id,))
        # Replace rather than upsert: a chunk id is stable, so its row is rewritten
        # whole with the vectors that match the text now stored alongside them.
        for c, d, s in zip(stale_chunks, dense, sparse):
            con.execute("DELETE FROM chunks WHERE chunk_id=%s", (c["chunk_id"],))
            con.execute(
                f"INSERT INTO chunks ({cols}, dense, sparse) VALUES ({placeholders})",
                tuple(c[k] for k in CHUNK_COLS)
                + (_dense_literal(d), _sparse_literal(s)))
        for c in reposition:
            con.execute("UPDATE chunks SET msg_idx=%s, created_at=%s WHERE chunk_id=%s",
                        (c["msg_idx"], c["created_at"], c["chunk_id"]))
        assignments = ", ".join(f"{col}=%s" for col in _UNIT_COLS)
        for u in refresh:
            first = (by_unit.get(u) or [{}])[0]
            con.execute(f"UPDATE chunks SET {assignments} WHERE unit_id=%s",
                        tuple(first.get(col) for col in _UNIT_COLS) + (u,))
        for u in gone:
            con.execute("DELETE FROM indexed_units WHERE unit_id=%s", (u,))
        # `refresh` is every unit whose metadata signature differs, new units included;
        # a full run re-embeds all of them and so rewrites every watermark.
        for u in (universe if full else refresh):
            con.execute(
                "INSERT INTO indexed_units (unit_id, signature, indexed_at) "
                "VALUES (%s,%s,%s) ON CONFLICT(unit_id) DO UPDATE SET "
                "signature=EXCLUDED.signature, indexed_at=EXCLUDED.indexed_at",
                (u, metas[u], now))
        con.commit()
    reindexed = {c["unit_id"] for c in stale_chunks} | refresh
    return {"reindexed_units": len(reindexed), "chunks": len(stale_chunks),
            "removed_units": len(gone), "dropped_chunks": len(dropped),
            "repositioned_chunks": len(reposition)}


# --------------------------------------------------------------------------- #
# Faceted hybrid search
# --------------------------------------------------------------------------- #
# Every source-specific facet the WHERE builder understands, mapped to the SET of
# raw sources whose units can satisfy it. This registry is the SSOT read by BOTH
# `_validate_facets` and `_facet_where` (they take the same `facets` dict), so a
# facet added to the WHERE builder cannot be silently un-guarded: an unregistered
# key fails loud on the first call instead of validating and then ANDing an
# unsatisfiable predicate into an empty, confident "no matches". (`session` was
# exactly that hole: it reached the WHERE builder but not the hand-enumerated
# guard, so source='dream' + session=... returned nothing with no error.)
# A facet maps to a SET because a facet can be carried by more than one source:
# repo/worktree come from a local session's cwd (Claude Code AND Codex CLI), and
# `model` is carried by every conversational source (ChatGPT records model_slug).
FACET_SOURCES = {
    "repo": frozenset({"claude_code", "codex_cli", "opencode", "pi"}),
    "worktree": frozenset({"claude_code", "codex_cli"}),
    "branch": frozenset({"claude_code"}),
    "session": frozenset({"claude_code", "codex_cli", "opencode", "pi"}),
    "project": frozenset({"claude_ai", "chatgpt", "artiport"}),
    "model": frozenset({"claude_ai", "claude_code", "codex_cli", "chatgpt", "opencode", "pi"}),
    "sender": frozenset(RAW_SOURCES),
}


def _check_registered(facets: dict) -> None:
    unknown = sorted(set(facets) - set(FACET_SOURCES))
    if unknown:
        raise ValueError(f"facet(s) {unknown} are not registered in FACET_SOURCES — "
                         f"register each with the source whose units carry it")


def _validate_facets(source: str, facets: dict) -> None:
    """Fail loud on a facet that contradicts the chosen source (ADR 0003).
    `facets` must carry EVERY key in `FACET_SOURCES` (see the registry note)."""
    resolve_sources(source)   # raises on an unknown source (the value-set SSOT)
    _check_registered(facets)
    if source == "dream":
        # Dream units carry NO raw-source facets, so any of them silently matches
        # nothing. Fail loud rather than return a confusing empty result.
        bad = [k for k, v in facets.items() if v]
        if bad:
            raise ValueError(f"facet(s) {bad} do not apply to source='dream' "
                             f"(derived knowledge has no repo/project facets)")
    elif source in RAW_SOURCES:
        # A single raw source: every set facet must be satisfiable BY that source.
        bad = [k for k, v in facets.items() if v and source not in FACET_SOURCES[k]]
        if bad:
            raise ValueError(
                f"facet(s) {bad} do not apply to source='{source}' — "
                f"{bad[0]!r} is satisfied only by source(s) "
                f"{sorted(FACET_SOURCES[bad[0]])}")


def _facet_where(source, facets: dict, since, until, params: dict) -> str:
    """Build the shared WHERE fragment (on `chunks`) from the given facets, filling
    `params`. Returns '' or 'WHERE ...'. Rejects a facet key that is not in
    `FACET_SOURCES` — the registry both guards read (same dict, same keys)."""
    _check_registered(facets)
    conds = []

    def add(cond: str, **kw):
        conds.append(cond)
        params.update(kw)

    # 'all' == raw only, resolved by clync.resolve_sources — the ONE owner of that
    # policy (cmd_list shares it): derived dream units never leak into an
    # unqualified search, so the dig can't read its own output.
    add("source = ANY(%(sources)s)", sources=resolve_sources(source))
    project = facets.get("project")
    if project == "any":
        add("project_uuid IS NOT NULL")
    elif project == "none":
        add("project_uuid IS NULL AND kind IN ('chat', 'chatgpt', 'artifact')")
    elif project:
        add("(project_name = %(project)s OR project_uuid = %(project)s)", project=project)
    if facets.get("model"):
        add("model = %(model)s", model=facets["model"])
    if facets.get("repo"):
        add("repo = %(repo)s", repo=facets["repo"])
    if facets.get("worktree"):
        add("worktree = %(worktree)s", worktree=facets["worktree"])
    if facets.get("branch"):
        add("git_branch = %(branch)s", branch=facets["branch"])
    if facets.get("session"):
        add("(unit_id = %(session)s OR title ILIKE %(session_like)s)",
            session=facets["session"], session_like=f"%{facets['session']}%")
    if facets.get("sender"):
        add("sender = ANY(%(senders)s)",
            senders=cc.resolve_sender_role(facets["sender"]))
    if since:
        add("updated_at >= %(since)s", since=since)
    if until:
        add("updated_at <= %(until)s", until=until)
    return ("WHERE " + " AND ".join(conds)) if conds else ""


def _row_to_result(r: dict) -> dict:
    # `dense_sim` is the raw cosine similarity of the query to the chunk's dense
    # vector — an ABSOLUTE relevance signal. The RRF `score` is rank-only (topk
    # rows always come back), so it cannot express "nothing here matches";
    # dream.recall's relevance floor needs dense_sim for exactly that.
    # `msg_idx` is WHERE in the unit the match actually is. Retrieval is the only
    # layer that knows it, so dropping it here forces every consumer to guess a
    # window — measured consequence: 93% of matches fell outside the window the
    # dream layer rendered (it started at idx 0), median match at idx 1284.
    # `sender` is WHOSE words the hit is. A reader that cannot see it has to guess
    # whether a quote is the user's own position or something said to them.
    return {"unit_id": r["unit_id"], "unit_name": r["unit_name"],
            "source": r["source"], "lang": r["lang"], "text": r["text"],
            "sender": r["sender"],
            "msg_idx": r["msg_idx"], "chunk_idx": r["chunk_idx"],
            "score": float(r["score"]), "dense_sim": float(r["dense_sim"])}


def _selection_is_small(pg, where: str, params: dict) -> bool:
    """Does the facet selection hold at most `EXACT_SELECTION_MAX` chunks? The count stops
    one row past the limit, so a large selection costs no more than reading that far."""
    n = pg.execute(f"SELECT count(*) AS n FROM (SELECT 1 FROM chunks {where} "
                   f"LIMIT {EXACT_SELECTION_MAX + 1}) selection", params).fetchone()["n"]
    return n <= EXACT_SELECTION_MAX


def _leg_ctes(where: str, exact: bool, with_sparse: bool) -> tuple[list[str], list[str]]:
    """Build dense (`d`) and sparse (`s`) shortlist CTEs.

    Exact selections bypass HNSW via `+ 0` on float8 distance rather than a
    MATERIALIZED CTE, avoiding temp-file spills of wide TOASTed vectors while
    preserving btree facet index scans."""
    if exact:
        dense_distance = "(dense <=> %(qd)s::vector) + 0"
        sparse_distance = "(sparse <#> %(qs)s::sparsevec) + 0"
    else:
        dense_distance = f"{DENSE_KEY} <=> (%(qd)s::halfvec({DENSE_DIM}))"
        sparse_distance = "sparse <#> %(qs)s::sparsevec"
    legs = [("d", dense_distance)]
    if with_sparse:
        legs.append(("s", sparse_distance))
    ctes = [
        f"""{name} AS (
            SELECT chunk_id, row_number() OVER (ORDER BY distance) rnk
            FROM (SELECT chunk_id, {distance} AS distance
                  FROM chunks {where} ORDER BY distance LIMIT {PRE_TOPK}) shortlist)"""
        for name, distance in legs
    ]
    return ctes, [f"SELECT * FROM {name}" for name, _ in legs]


def hybrid_search(query: str = "", topk: int = TOPK, *, source: str = "all",
                  project: str | None = None, model: str | None = None,
                  repo: str | None = None, worktree: str | None = None,
                  branch: str | None = None, session: str | None = None,
                  sender: str | None = None,
                  since: str | None = None, until: str | None = None,
                  sort: str = "relevance", lang: str | None = None,
                  per_unit: int = 1,
                  exclude_attempted: str | None = None,
                  min_dense_sim: float | None = None,
                  query_embedding: tuple[list[list[float]], list[dict]] | None = None
                  ) -> list[dict]:
    """Faceted hybrid search. With a `query`, ranks by fused dense+sparse RRF +
    typed-metadata boost, collapsed to `per_unit` best chunks per unit — at the
    default of 1, rows ALWAYS have unit semantics, so the `topk` budget spans
    DISTINCT units and can never be consumed by one unit's chunks. With an
    EMPTY query, degrades to a metadata browse (units matching the facets,
    newest first). A facet contradicting `source` fails loud.

    `per_unit` above 1 lifts that ceiling for a caller that wants several
    positions inside one unit: a long session holds many separate episodes, and at
    a ceiling of 1 the second is unreachable at any `topk`. `topk` still caps the
    total, so raising `per_unit` trades breadth of units for depth within them.

    `sender` is a ROLE (see `cc.SENDER_ROLES`), not a stored sender string — the
    sources spell the user differently, and a role spans every spelling. It needs
    a content query: a browse ranks units and has no matched message to attribute.

    `exclude_attempted` (a dream topic_id) is the dream dig's exclusion,
    applied HERE — in the SQL WHERE, before the ANN shortlists and the topk
    cut — because this boundary owns both the budget and the exclusion: a
    chunk whose `msg_idx` falls inside one of the topic's already-rendered
    `dream_attempted` ranges never occupies a shortlist or result slot, so
    the collapsed row per unit is that unit's best UNREAD hit, and an empty
    result genuinely means no unread hit anywhere. (The previous shape —
    uncollapsed rows trimmed to topk chunks, then range-filtered in Python —
    let one verbose unit consume the whole budget and made the backfill
    misread a saturated window as an exhausted pool.) Positionless chunks
    are excluded too: a hit that cannot be positioned cannot be dug.

    `min_dense_sim` is the relevance floor, likewise applied in SQL before the
    collapse and the cut: RRF scores are rank-only (topk rows always come back,
    so they cannot express "nothing here matches"), and a below-floor chunk is
    retrieval filler that must neither occupy a result slot nor keep its unit
    in a dig pool. Filtering it in Python after the cut re-creates both.

    Read-only, and deliberately does NOT provision the store: provisioning takes
    DDL locks, and a read that takes DDL locks deadlocks against its own caller's
    open transaction. Callers provision once at command entry."""
    facets = {"project": project, "model": model, "repo": repo,
              "worktree": worktree, "branch": branch, "session": session,
              "sender": sender}
    _validate_facets(source, facets)
    if per_unit < 1:
        raise ValueError(f"per_unit must be at least 1 (got {per_unit})")

    if not query.strip() or sort == "recency":
        if exclude_attempted or min_dense_sim is not None or sender:
            raise ValueError(
                "exclude_attempted / min_dense_sim / sender require a relevance-ranked "
                "content query; browse mode has no hit positions, vectors or author")
        return _browse(topk, source, facets, since, until, query=query)

    (qd,), (qs,) = (query_embedding if query_embedding is not None
                    else _embed([query], max_length=512))
    qlang = _lang(query)
    try:
        qs_lit = _sparse_literal(qs)
    except ValueError:
        qs_lit = None
    params: dict = {"qd": _dense_literal(qd), "qlang": qlang}
    if lang:
        params["lang"] = lang
    where = _facet_where(source, facets, since, until, params)
    if lang:
        where = (where + (" AND " if where else "WHERE ") + "lang = %(lang)s")
    if exclude_attempted:
        params["xatopic"] = exclude_attempted
        # Expiring artifacts must not become permanent distilled Dream content.
        where = (where + (" AND " if where else "WHERE ")
                 + "kind <> 'artifact' AND msg_idx IS NOT NULL AND NOT EXISTS ("
                   "SELECT 1 FROM dream_attempted a "
                   "WHERE a.topic_id = %(xatopic)s "
                   "AND a.unit_id = chunks.unit_id "
                   "AND chunks.msg_idx BETWEEN a.from_idx AND a.to_idx)")

    if qs_lit is not None:
        params["qs"] = qs_lit
    check_ef_search(HNSW_EF_SEARCH)
    with connect_pg() as pg:
        exact = _selection_is_small(pg, where, params)
        if not exact:
            pg.execute(
                "SELECT set_config('hnsw.ef_search', %s, true), "
                "set_config('hnsw.iterative_scan', 'strict_order', true), "
                "set_config('enable_seqscan', 'off', true), "
                "set_config('enable_bitmapscan', 'off', true), "
                "set_config('enable_sort', 'off', true)", (str(HNSW_EF_SEARCH),))
        ctes, unions = _leg_ctes(where, exact, qs_lit is not None)
        ctes.append(f"""fused AS (
            SELECT chunk_id, SUM(1.0/({RRF_K}+rnk)) AS rrf
            FROM ({' UNION ALL '.join(unions)}) u GROUP BY chunk_id)""")
        # Materializing `scored` (<= 2*PRE_TOPK rows) prevents enable_sort=off from
        # turning the outer WindowAgg into a full idx_chunks_unit scan on `chunks`.
        ctes.append("""scored AS MATERIALIZED (
            SELECT c.unit_id, c.unit_name, c.source, c.lang, c.text, c.sender,
                   c.msg_idx, c.chunk_idx,
                   1 - (c.dense <=> %(qd)s::vector) AS dense_sim,
                   f.rrf * (1.0
                       + CASE WHEN c.updated_at IS NULL THEN 0
                              ELSE 0.10 * exp(-GREATEST(0, EXTRACT(EPOCH FROM (now() - c.updated_at)))
                                              / (86400.0*180)) END
                       + CASE WHEN c.lang = %(qlang)s THEN 0.05 ELSE 0 END) AS score
            FROM fused f JOIN chunks c ON c.chunk_id = f.chunk_id)""")
        cols = ("unit_id, unit_name, source, lang, text, sender,"
                " msg_idx, chunk_idx, score, dense_sim")
        floor = ""
        if min_dense_sim is not None:
            params["minsim"] = min_dense_sim
            floor = " WHERE dense_sim >= %(minsim)s"
        params["per_unit"] = per_unit
        ctes.append(f"""ranked AS (
            SELECT {cols}, row_number() OVER (PARTITION BY unit_id ORDER BY score DESC) rn
            FROM scored{floor})""")
        sql = ("WITH " + ",\n".join(ctes)
               + f"\nSELECT {cols} FROM ranked WHERE rn <= %(per_unit)s")
        rows = pg.execute(sql, params).fetchall()
    results = [_row_to_result(r) for r in rows]
    results.sort(key=lambda r: -r["score"])
    return results[:topk]


def _browse(topk, source, facets: dict, since, until, *, query: str) -> list[dict]:
    """Metadata browse: units matching the facets, newest first (no content
    vector). Backs the empty-query and sort='recency' modes."""
    params: dict = {}
    where = _facet_where(source, facets, since, until, params)
    # For recency-sorted content queries we still narrow by lexical title match if a
    # query was given, but ranking is purely recency here.
    if query.strip():
        where = (where + (" AND " if where else "WHERE ")
                 + "(title ILIKE %(q_like)s)")
        params["q_like"] = f"%{query.strip()}%"
    sql = (f"SELECT unit_id, kind, source, title, project_name, repo, summary, "
           f"COALESCE(updated_at, created_at) AS ts "
           f"FROM units {where} ORDER BY ts DESC NULLS LAST LIMIT {int(topk)}")
    with connect_pg() as pg:
        rows = pg.execute(sql, params).fetchall()
    out = []
    for r in rows:
        name = _display_name(r["kind"], r["title"], r["project_name"], r["repo"])
        out.append({"unit_id": r["unit_id"], "unit_name": name,
                    "source": r["source"], "lang": None,
                    "text": r["summary"] or r["title"] or "", "score": 0.0,
                    # browse mode ranks by recency, so there is no matched chunk and
                    # thus no position or author: None, never a 0 that would read as
                    # "idx 0" nor a sender the row cannot have.
                    "sender": None, "msg_idx": None, "chunk_idx": None,
                    "dense_sim": None})   # browse mode has no content vector
    return out
