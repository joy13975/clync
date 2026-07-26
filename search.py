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
import re
from datetime import datetime

# The store + shared config live in the core module (SSOT). Search is core now
# (ADR 0003): its deps are no longer an optional extra.
from clync import (DEFAULT_TOPK, PG_BIN, connect_pg, ensure_cluster,
                   resolve_sources, sql_statements, _vector_control_present)

EMBED_MODEL = "BAAI/bge-m3"
DENSE_DIM = 1024
SPARSE_DIM = 250002          # XLM-R vocab; BGE-M3 lexical weights are token ids
CHUNK_CHARS = 2000
CHUNK_OVERLAP = 200
RRF_K = 60
PRE_TOPK = 50                # per-signal shortlist before fusion
TOPK = DEFAULT_TOPK          # default result count (SSOT: clync.DEFAULT_TOPK)

INDEX_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     text PRIMARY KEY,
    unit_id      text NOT NULL,
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
-- Per-unit content-signature watermark for incremental re-embedding: a sha256
-- over the EXACT bytes embedded for the unit. Any change to embedded content (a
-- message edit, a rename that changes the display prefix, an added/removed chunk)
-- flips it and the unit re-embeds on the default incremental run.
CREATE TABLE IF NOT EXISTS indexed_units (
    unit_id    text PRIMARY KEY,
    signature  text NOT NULL,
    indexed_at text NOT NULL
);
"""


def ensure_index_schema() -> None:
    """Create the vector index tables (idempotent). Assumes the cluster + `vector`
    extension exist (clync.ensure_cluster provisions them).

    The index tables are DERIVED and rebuildable, so ANY stale layout — the
    pre-ADR-0003 conversation-keyed index (`indexed_convs`), or a `chunks`
    column set that no longer matches INDEX_SCHEMA — is dropped and recreated
    (watermarks included, so everything re-embeds). CREATE IF NOT EXISTS alone
    would silently keep the old shape and the first insert would fail."""
    with connect_pg() as con:
        have = {r["column_name"] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='chunks'").fetchall()}
        legacy = con.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name='indexed_convs'").fetchone()
        if legacy or (have and have != set(CHUNK_COLS) | {"dense", "sparse"}):
            con.execute("DROP TABLE IF EXISTS chunks")
            con.execute("DROP TABLE IF EXISTS indexed_convs")
            con.execute("DROP TABLE IF EXISTS indexed_units")
        for stmt in sql_statements(INDEX_SCHEMA):
            con.execute(stmt)
        con.commit()


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
    raises. Keys: available, cluster_running, chunks (int|None), error."""
    import clync
    st = {"available": available(),
          "cluster_running": False, "chunks": None, "error": None}
    if not st["available"]:
        return st
    import psycopg
    try:
        st["cluster_running"] = clync.cluster_running()
        if st["cluster_running"]:
            with connect_pg() as pg:
                st["chunks"] = pg.execute("SELECT count(*) AS n FROM chunks"
                                          ).fetchone()["n"]
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
    if kind == "cc_session":
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
    items = sorted((int(t) + 1, w) for t, w in weights.items() if w > 0)  # 0- -> 1-based
    if not items:
        raise ValueError("empty sparse vector")
    body = ",".join(f"{i}:{w:.7g}" for i, w in items)
    return f"{{{body}}}/{SPARSE_DIM}"


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def _unit_signature(chunks: list[dict]) -> str:
    """Deterministic content signature over the exact bytes embedded for a unit."""
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda c: c["chunk_id"]):
        h.update(c["chunk_id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(c["text"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def build_index(full: bool = False) -> dict:
    """Embed new/changed units into the contained cluster. A unit re-embeds
    whenever the exact bytes folded into its chunks change (a content signature,
    not a timestamp). `full` re-embeds everything. Returns stats."""
    ensure_cluster()
    with connect_pg() as con:
        universe = {r["unit_id"] for r in
                    con.execute("SELECT unit_id FROM units").fetchall()}
        chunks = _load_chunks(con)
        by_unit: dict[str, list[dict]] = {}
        for c in chunks:
            by_unit.setdefault(c["unit_id"], []).append(c)
        sigs = {u: _unit_signature(by_unit.get(u, [])) for u in universe}
        indexed = {r["unit_id"]: r["signature"] for r in
                   con.execute("SELECT unit_id, signature FROM indexed_units").fetchall()}
        stale = set(universe) if full else {u for u in universe
                                            if indexed.get(u) != sigs[u]}
        gone = set(indexed) - universe
        if not stale and not gone:
            return {"reindexed_units": 0, "chunks": 0, "removed_units": 0}

        stale_chunks = [c for c in chunks if c["unit_id"] in stale]
        dense, sparse = ([], [])
        if stale_chunks:
            dense, sparse = _embed([c["text"] for c in stale_chunks], max_length=2048)

        cols = ", ".join(CHUNK_COLS)
        placeholders = ",".join(["%s"] * len(CHUNK_COLS)) + ",%s::vector,%s::sparsevec"
        now = datetime.now().astimezone().isoformat()
        for u in stale | gone:
            con.execute("DELETE FROM chunks WHERE unit_id=%s", (u,))
        for c, d, s in zip(stale_chunks, dense, sparse):
            con.execute(
                f"INSERT INTO chunks ({cols}, dense, sparse) VALUES ({placeholders})",
                tuple(c[k] for k in CHUNK_COLS)
                + (_dense_literal(d), _sparse_literal(s)))
        for u in gone:
            con.execute("DELETE FROM indexed_units WHERE unit_id=%s", (u,))
        for u in stale:
            con.execute(
                "INSERT INTO indexed_units (unit_id, signature, indexed_at) "
                "VALUES (%s,%s,%s) ON CONFLICT(unit_id) DO UPDATE SET "
                "signature=EXCLUDED.signature, indexed_at=EXCLUDED.indexed_at",
                (u, sigs[u], now))
        con.commit()
    return {"reindexed_units": len(stale), "chunks": len(stale_chunks),
            "removed_units": len(gone)}


# --------------------------------------------------------------------------- #
# Faceted hybrid search
# --------------------------------------------------------------------------- #
def _validate_facets(source: str, *, project, model, repo, worktree, branch) -> None:
    """Fail loud on a facet that contradicts the chosen source (ADR 0003)."""
    resolve_sources(source)   # raises on an unknown source (the value-set SSOT)
    cc_facets = {"repo": repo, "worktree": worktree, "branch": branch}
    ai_facets = {"project": project, "model": model}
    if source == "dream":
        # Dream units carry NO raw-source facets, so any of them silently matches
        # nothing. Fail loud rather than return a confusing empty result.
        bad = [k for k, v in {**cc_facets, **ai_facets}.items() if v]
        if bad:
            raise ValueError(f"facet(s) {bad} do not apply to source='dream' "
                             f"(derived knowledge has no repo/project facets)")
    if source == "claude_ai":
        bad = [k for k, v in cc_facets.items() if v]
        if bad:
            raise ValueError(f"facet(s) {bad} apply only to Claude Code sessions, "
                             f"not source='claude_ai'")
    if source == "claude_code":
        bad = [k for k, v in ai_facets.items() if v]
        if bad:
            raise ValueError(f"facet(s) {bad} apply only to claude.ai chats, "
                             f"not source='claude_code'")


def _facet_where(source, project, model, repo, worktree, branch, session,
                 since, until, params: dict) -> str:
    """Build the shared WHERE fragment (on `chunks`) from the given facets, filling
    `params`. Returns '' or 'WHERE ...'."""
    conds = []

    def add(cond: str, **kw):
        conds.append(cond)
        params.update(kw)

    # 'all' == raw only, resolved by clync.resolve_sources — the ONE owner of that
    # policy (cmd_list shares it): derived dream units never leak into an
    # unqualified search, so the dig can't read its own output.
    add("source = ANY(%(sources)s)", sources=resolve_sources(source))
    if project == "any":
        add("project_uuid IS NOT NULL")
    elif project == "none":
        add("project_uuid IS NULL AND kind = 'chat'")
    elif project:
        add("(project_name = %(project)s OR project_uuid = %(project)s)", project=project)
    if model:
        add("model = %(model)s", model=model)
    if repo:
        add("repo = %(repo)s", repo=repo)
    if worktree:
        add("worktree = %(worktree)s", worktree=worktree)
    if branch:
        add("git_branch = %(branch)s", branch=branch)
    if session:
        add("(unit_id = %(session)s OR title ILIKE %(session_like)s)",
            session=session, session_like=f"%{session}%")
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
    return {"unit_id": r["unit_id"], "unit_name": r["unit_name"],
            "source": r["source"], "lang": r["lang"], "text": r["text"],
            "score": float(r["score"]), "dense_sim": float(r["dense_sim"])}


def hybrid_search(query: str = "", topk: int = TOPK, *, source: str = "all",
                  project: str | None = None, model: str | None = None,
                  repo: str | None = None, worktree: str | None = None,
                  branch: str | None = None, session: str | None = None,
                  since: str | None = None, until: str | None = None,
                  sort: str = "relevance", lang: str | None = None) -> list[dict]:
    """Faceted hybrid search. With a `query`, ranks by fused dense+sparse RRF +
    typed-metadata boost, collapsed to one best chunk per unit. With an EMPTY
    query, degrades to a metadata browse (units matching the facets, newest first).
    A facet contradicting `source` fails loud."""
    _validate_facets(source, project=project, model=model, repo=repo,
                     worktree=worktree, branch=branch)
    ensure_cluster()

    if not query.strip() or sort == "recency":
        return _browse(topk, source, project, model, repo, worktree, branch,
                       session, since, until, query=query)

    (qd,), (qs,) = _embed([query], max_length=512)
    qlang = _lang(query)
    try:
        qs_lit = _sparse_literal(qs)
    except ValueError:
        qs_lit = None
    params: dict = {"qd": _dense_literal(qd), "qlang": qlang}
    if lang:
        params["lang"] = lang
    where = _facet_where(source, project, model, repo, worktree, branch, session,
                         since, until, params)
    if lang:
        where = (where + (" AND " if where else "WHERE ") + "lang = %(lang)s")

    ctes = [f"""d AS (
        SELECT chunk_id, row_number() OVER (ORDER BY dense <=> %(qd)s::vector) rnk
        FROM chunks {where} ORDER BY dense <=> %(qd)s::vector LIMIT {PRE_TOPK})"""]
    unions = ["SELECT * FROM d"]
    if qs_lit is not None:
        params["qs"] = qs_lit
        ctes.append(f"""s AS (
        SELECT chunk_id, row_number() OVER (ORDER BY sparse <#> %(qs)s::sparsevec) rnk
        FROM chunks {where} ORDER BY sparse <#> %(qs)s::sparsevec LIMIT {PRE_TOPK})""")
        unions.append("SELECT * FROM s")
    ctes.append(f"""fused AS (
        SELECT chunk_id, SUM(1.0/({RRF_K}+rnk)) AS rrf
        FROM ({' UNION ALL '.join(unions)}) u GROUP BY chunk_id)""")
    ctes.append("""scored AS (
        SELECT c.unit_id, c.unit_name, c.source, c.lang, c.text,
               1 - (c.dense <=> %(qd)s::vector) AS dense_sim,
               f.rrf * (1.0
                   + CASE WHEN c.updated_at IS NULL THEN 0
                          ELSE 0.10 * exp(-GREATEST(0, EXTRACT(EPOCH FROM (now() - c.updated_at)))
                                          / (86400.0*180)) END
                   + CASE WHEN c.lang = %(qlang)s THEN 0.05 ELSE 0 END) AS score
        FROM fused f JOIN chunks c ON c.chunk_id = f.chunk_id)""")
    sql = ("WITH " + ",\n".join(ctes)
           + "\nSELECT DISTINCT ON (unit_id) unit_id, unit_name, source, lang, text,"
             " score, dense_sim"
             "\nFROM scored ORDER BY unit_id, score DESC")
    with connect_pg() as pg:
        rows = pg.execute(sql, params).fetchall()
    results = [_row_to_result(r) for r in rows]
    results.sort(key=lambda r: -r["score"])
    return results[:topk]


def _browse(topk, source, project, model, repo, worktree, branch, session,
            since, until, *, query: str) -> list[dict]:
    """Metadata browse: units matching the facets, newest first (no content
    vector). Backs the empty-query and sort='recency' modes."""
    params: dict = {}
    where = _facet_where(source, project, model, repo, worktree, branch, session,
                         since, until, params)
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
                    "dense_sim": None})   # browse mode has no content vector
    return out
