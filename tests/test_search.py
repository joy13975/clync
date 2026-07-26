"""Search + index integration: real Postgres (throwaway db), embedder mocked so
the SQL / RRF-fusion / watermark / faceting logic is what's under test, not BGE-M3."""
from __future__ import annotations

import pytest

import clync

# valid timestamptz values (units.updated_at is a real timestamp)
T1, T2, T3 = "2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00"


def _seed(uuid, name, updated_at, texts, org="org1", project_uuid=None):
    """Insert a claude.ai chat unit + its messages."""
    con = clync.connect()
    full = {"uuid": uuid, "name": name, "updated_at": updated_at,
            "created_at": T1, "project_uuid": project_uuid,
            "chat_messages": [{"uuid": f"{uuid}-m{i}", "sender": "human",
                               "content": [{"type": "text", "text": t}],
                               "created_at": T1}
                              for i, t in enumerate(texts)]}
    clync.upsert_conversation(con, org, full)
    # mirror _sync_org: backfill project_name so the "in project X" facet is queryable
    con.execute("UPDATE units u SET project_name = p.name FROM projects p "
                "WHERE u.unit_id = %s AND u.project_uuid = p.uuid", (uuid,))
    con.commit()
    con.close()


def _seed_project(project_uuid, name, docs, org="org1"):
    """Insert a project + its knowledge docs. docs: [(uuid, file_name, content)]."""
    con = clync.connect()
    project = {"uuid": project_uuid, "name": name}
    clync.upsert_project(con, org, project)
    clync.upsert_project_docs(con, org, project,
        [{"uuid": du, "file_name": fn, "content": c, "created_at": T1}
         for du, fn, c in docs])
    con.commit()
    con.close()


def _seed_cc(unit_id, title, repo, texts, worktree=None, branch=None):
    """Insert a Claude Code session unit directly (bypasses file parsing)."""
    import cc
    msgs = [cc.CCMessage(msg_id=f"{unit_id}-m{i}", idx=i, sender="user",
                         role_detail="text", text=t, created_at=T1)
            for i, t in enumerate(texts)]
    s = cc.CCSession(session_id=unit_id, title=title, summary=None,
                     created_at=T1, updated_at=T1, repo=repo, cwd=f"/x/{repo}",
                     worktree=worktree, git_branch=branch, cc_version="2.1.0",
                     entrypoint="cli", model="claude-opus-4-8", messages=msgs)
    con = clync.connect()
    clync._upsert_cc_session(con, s)
    con.commit()
    con.close()


def _delete(unit_id):
    con = clync.connect()
    con.execute("DELETE FROM units WHERE unit_id=%s", (unit_id,))   # cascades messages
    con.commit()
    con.close()


def _pg_units(search):
    with search.connect_pg() as pg:
        return {r["unit_id"] for r in pg.execute("SELECT DISTINCT unit_id FROM chunks")}


def _pg_rows(search):
    """{unit_id: (unit_name, concatenated text)} — the embedded content per unit."""
    out: dict[str, tuple[str, str]] = {}
    with search.connect_pg() as pg:
        for r in pg.execute("SELECT unit_id, unit_name, text FROM chunks ORDER BY chunk_id"):
            prev = out.get(r["unit_id"], (r["unit_name"], ""))
            out[r["unit_id"]] = (r["unit_name"], prev[1] + r["text"])
    return out


def test_build_index_watermark_gone_and_stale_zero_chunk(pg_test_db, mock_embed):
    search = pg_test_db
    _seed("A", "Alpha", T1, ["alpha body"])
    _seed("B", "Beta", T1, ["beta body"])
    st = search.build_index(full=False)
    assert st["reindexed_units"] == 2
    assert _pg_units(search) == {"A", "B"}

    assert search.build_index(full=False)["reindexed_units"] == 0    # watermark skips

    _seed("A", "Alpha", T2, ["alpha rewritten"])                     # stale
    _delete("B")                                                     # gone
    st = search.build_index(full=False)
    assert st["reindexed_units"] == 1 and st["removed_units"] == 1
    assert _pg_units(search) == {"A"}

    # a stale unit that now yields ZERO chunks must still be purged AND advance its
    # watermark (else it reindexes forever).
    _seed("A", "Alpha", T3, [""])
    assert search.build_index(full=False)["reindexed_units"] == 1
    assert _pg_units(search) == set()
    assert search.build_index(full=False)["reindexed_units"] == 0


def test_hybrid_search_collapse_topk_and_lang(pg_test_db, mock_embed):
    search = pg_test_db
    _seed("A", "Alpha", T1, ["first chunk", "second chunk"])   # 2 chunks, one unit
    _seed("J", "日本語", T1, ["これは日本語の会話です"])          # ja unit
    search.build_index(full=False)

    hits = search.hybrid_search("anything", topk=10)
    assert len(hits) == 2                                   # DISTINCT ON collapsed A's chunks
    assert len({h["unit_id"] for h in hits}) == 2
    assert len(search.hybrid_search("anything", topk=1)) == 1   # topk respected

    ja = search.hybrid_search("anything", topk=10, lang="ja")
    assert [h["unit_id"] for h in ja] == ["J"]            # lang filter restricts


def test_faceted_source_repo_and_project(pg_test_db, mock_embed):
    search = pg_test_db
    _seed_project("p1", "Work", [("doc1", "audit.md", "cnx audit body")])
    _seed("chat1", "planning", T1, ["chat body"], project_uuid="p1")
    _seed("chat2", "loose", T1, ["other body"])              # no project
    _seed_cc("ccs1", "clync design", "clync", ["postgres schema body"])
    _seed_cc("ccs2", "meristem fix", "meristem", ["dispatch body"], branch="main")
    search.build_index(full=False)

    def ids(**kw):
        return {h["unit_id"] for h in search.hybrid_search("body", topk=20, **kw)}

    assert ids(source="claude_code") == {"ccs1", "ccs2"}
    assert ids(source="claude_ai") == {"doc1", "chat1", "chat2"}
    assert ids(source="claude_code", repo="clync") == {"ccs1"}
    # project membership spans kinds: a project's chat AND its knowledge doc match
    assert ids(source="claude_ai", project="any") == {"chat1", "doc1"}
    assert ids(source="claude_ai", project="Work") == {"chat1", "doc1"}  # by project name
    assert ids(source="claude_ai", project="none") == {"chat2"}     # chats not in a project
    assert ids(source="claude_code", branch="main") == {"ccs2"}


def test_resolve_sources_owns_the_all_means_raw_rule():
    """ONE resolution of the --source facet: 'all' is RAW only, at every surface
    (`search`'s WHERE builder AND `clync list` resolve through this) — dream
    units must never leak into an unqualified listing or search."""
    import pytest
    assert clync.resolve_sources("all") == list(clync.RAW_SOURCES)
    assert clync.resolve_sources("dream") == ["dream"]
    assert clync.resolve_sources("claude_ai") == ["claude_ai"]
    with pytest.raises(ValueError, match="source must be one of"):
        clync.resolve_sources("bogus")


def test_faceted_source_mismatch_fails_loud(pg_test_db, mock_embed):
    import pytest
    search = pg_test_db
    with pytest.raises(ValueError):
        search.hybrid_search("x", source="claude_ai", repo="clync")
    with pytest.raises(ValueError):
        search.hybrid_search("x", source="claude_code", project="Work")


def test_empty_query_recency_browse(pg_test_db, mock_embed):
    search = pg_test_db
    _seed("A", "Alpha", T1, ["a"])
    _seed("B", "Beta", T3, ["b"])          # newer
    search.build_index(full=False)
    # empty query -> metadata browse, newest first, no content vector needed
    hits = search.hybrid_search("", sort="recency", topk=10)
    assert [h["unit_id"] for h in hits] == ["B", "A"]


def test_project_docs_and_project_chats_are_indexed(pg_test_db, mock_embed):
    search = pg_test_db
    _seed_project("p1", "Work", [("doc1", "audit.md", "CNX infrastructure audit")])
    _seed("chat1", "CNX sour guy dance", T1, ["about the cnx guy"], project_uuid="p1")
    st = search.build_index(full=False)
    assert st["reindexed_units"] == 2

    hits = {h["unit_id"]: h["unit_name"] for h in search.hybrid_search("cnx", topk=10)}
    assert "doc1" in hits and hits["doc1"] == "[Work] audit.md"
    assert "chat1" in hits and hits["chat1"] == "[Work] CNX sour guy dance"


def test_project_rename_reindexes_on_incremental(pg_test_db, mock_embed):
    # an embedded input owned OUTSIDE the unit's own messages (the project name in
    # the `[Project]` prefix) must still mark the unit stale on the DEFAULT run.
    search = pg_test_db
    _seed_project("p1", "Work", [("doc1", "audit.md", "audit body")])
    _seed("chat1", "planning", T1, ["some chat body"], project_uuid="p1")
    assert search.build_index(full=False)["reindexed_units"] == 2
    assert search.build_index(full=False)["reindexed_units"] == 0

    # Mirror a real sync after an upstream rename: the project is re-listed with the
    # new name (upsert_project + upsert_project_docs re-inserts the doc unit), and
    # chat units' project_name is backfilled.
    _seed_project("p1", "Audit", [("doc1", "audit.md", "audit body")])
    con = clync.connect()
    con.execute("UPDATE units u SET project_name=p.name FROM projects p "
                "WHERE u.project_uuid=p.uuid AND u.kind='chat'")
    con.commit()
    con.close()

    assert search.build_index(full=False)["reindexed_units"] == 2
    rows = _pg_rows(search)
    assert rows["chat1"][0] == "[Audit] planning"
    assert rows["doc1"][0] == "[Audit] audit.md"
    assert search.build_index(full=False)["reindexed_units"] == 0


def test_doc_inplace_edit_reindexes_on_incremental(pg_test_db, mock_embed):
    search = pg_test_db
    _seed_project("p1", "Work", [("doc1", "notes.md", "original content")])
    assert search.build_index(full=False)["reindexed_units"] == 1
    assert search.build_index(full=False)["reindexed_units"] == 0

    _seed_project("p1", "Work", [("doc1", "notes.md", "rewritten content")])
    assert search.build_index(full=False)["reindexed_units"] == 1
    assert "rewritten content" in _pg_rows(search)["doc1"][1]
    assert search.build_index(full=False)["reindexed_units"] == 0


def test_legacy_conversation_keyed_index_is_replaced(pg_test_db, mock_embed):
    # A cluster from the SQLite-SSOT era carries `indexed_convs` + a chunks table
    # WITHOUT unit_id. ensure_index_schema must replace that derived layout (both
    # tables are rebuildable) — CREATE IF NOT EXISTS alone would silently keep the
    # stale shape and the first chunk insert would fail.
    search = pg_test_db
    con = clync.connect()
    con.execute("DROP TABLE chunks")
    con.execute("DROP TABLE indexed_units")
    con.execute("CREATE TABLE chunks (chunk_id text PRIMARY KEY, conv_uuid text, "
                "text text)")                                  # legacy shape
    con.execute("CREATE TABLE indexed_convs (conv_uuid text PRIMARY KEY)")
    con.commit()
    con.close()

    # Refusal first: the rebuild re-embeds the whole corpus, so it is never an
    # implicit side effect of provisioning (a read used to trigger it and deadlock).
    with pytest.raises(clync.StaleDerivedSchema, match="clync migrate"):
        search.ensure_index_schema()
    search.ensure_index_schema(rebuild=True)                        # explicit opt-in
    _seed("A", "Alpha", T1, ["alpha body"])
    assert search.build_index(full=False)["reindexed_units"] == 1   # new shape works
    assert _pg_units(search) == {"A"}


def test_retrieval_carries_the_matched_position_out(pg_test_db, mock_embed):
    """`msg_idx` is WHERE the match is, and only retrieval knows it. A consumer that
    has to guess instead renders the wrong part of the unit: measured on the real
    corpus, 98% of matches fell outside the window the dream layer showed, because it
    started at idx 0 while the median match sat at idx 1039. So the position must
    come OUT of hybrid_search, and be the position of the actually-matched chunk."""
    search = pg_test_db
    # one long unit; the distinctive phrase sits deep inside it, never near idx 0
    bodies = [f"filler turn number {i} about unrelated scheduling chatter"
              for i in range(30)]
    bodies[23] = "the marmoset calibration procedure requires a torque wrench"
    _seed("long", "Long conversation", T1, bodies)
    search.build_index(full=False)

    hits = search.hybrid_search("marmoset calibration torque", topk=3)
    assert hits[0]["unit_id"] == "long"
    assert hits[0]["msg_idx"] == 23, (
        f"matched position is wrong: got {hits[0]['msg_idx']}, want 23")
    assert hits[0]["chunk_idx"] is not None


def test_browse_reports_no_matched_position_rather_than_zero(pg_test_db, mock_embed):
    # A recency browse has no matched chunk. Reporting 0 would read as "the match is
    # at the start of the unit" and silently re-create the render-the-opening bug.
    search = pg_test_db
    _seed("A", "Alpha", T1, ["alpha body"])
    search.build_index(full=False)
    assert search.hybrid_search("", topk=3)[0]["msg_idx"] is None
    assert search.hybrid_search("alpha", topk=3, sort="recency")[0]["msg_idx"] is None


def test_read_path_never_provisions_so_it_cannot_ddl(pg_test_db, mock_embed):
    """A read must not provision the store. Provisioning takes DDL locks, and a
    read that takes DDL locks (a) deadlocks against its own caller's still-open
    read transaction — observed live as a permanent hang, no timeout, no error —
    and (b) when it does not deadlock, silently destroys the derived layer it just
    decided was stale. So: with a DELIBERATELY STALE derived shape, a search still
    answers and the stale shape is still there afterwards. If someone reinstates
    provisioning inside the read path, this fails (loudly, on the refusal) instead
    of hanging the suite."""
    search = pg_test_db
    _seed("A", "Alpha", T1, ["alpha body"])
    search.build_index(full=False)
    con = clync.connect()
    con.execute("ALTER TABLE dream_insights ADD COLUMN zz_stale text")   # stale now
    con.commit()
    con.close()

    assert search.hybrid_search("alpha", topk=5)          # read still answers

    con = clync.connect()
    try:
        assert con.execute(
            "SELECT 1 FROM information_schema.columns WHERE table_name="
            "'dream_insights' AND column_name='zz_stale'").fetchone(), \
            "the read path mutated the schema"
    finally:
        con.close()


def test_stale_chunks_column_set_is_replaced(pg_test_db, mock_embed):
    # A `chunks` layout whose column set drifted from INDEX_SCHEMA (e.g. a
    # cluster built at an earlier commit of this branch, before the
    # conv_name -> unit_name rename) must be dropped and rebuilt — watermarks
    # included, so every unit re-embeds into the fresh shape.
    search = pg_test_db
    con = clync.connect()
    con.execute("DROP TABLE chunks")
    con.execute("CREATE TABLE chunks (chunk_id text PRIMARY KEY, unit_id text, "
                "conv_name text, text text)")                  # drifted shape
    con.execute("INSERT INTO indexed_units VALUES ('stale-unit', 'sig', 'ts')")
    con.commit()
    con.close()

    with pytest.raises(clync.StaleDerivedSchema, match="clync migrate"):
        search.ensure_index_schema()
    search.ensure_index_schema(rebuild=True)                        # explicit opt-in
    _seed("A", "Alpha", T1, ["alpha body"])
    assert search.build_index(full=False)["reindexed_units"] == 1
    assert _pg_units(search) == {"A"}
    con = clync.connect()
    try:  # the stale watermark is gone with the stale shape
        assert con.execute("SELECT COUNT(*) n FROM indexed_units "
                           "WHERE unit_id='stale-unit'").fetchone()["n"] == 0
    finally:
        con.close()


def test_resumed_cc_sessions_index_without_chunk_collision(pg_test_db, mock_embed):
    # Per-unit message identity: two sessions can store the SAME event uuid (a cc
    # resume replay), so chunk ids must be unit-scoped — both units index fully.
    search = pg_test_db
    import cc
    for unit_id in ("sess-A", "sess-B"):
        msgs = [cc.CCMessage(msg_id="shared-uuid", idx=0, sender="user",
                             role_detail="text", text="shared prefix", created_at=T1)]
        s = cc.CCSession(session_id=unit_id, title=unit_id, summary=None,
                         created_at=T1, updated_at=T1, repo="clync", cwd="/x/clync",
                         worktree=None, git_branch=None, cc_version="2.1.0",
                         entrypoint="cli", model=None, messages=msgs)
        con = clync.connect()
        clync._upsert_cc_session(con, s)
        con.commit()
        con.close()
    assert search.build_index(full=False)["reindexed_units"] == 2
    assert _pg_units(search) == {"sess-A", "sess-B"}


def test_null_updated_at(pg_test_db, mock_embed):
    # a null updated_at must NOT (1) be silently omitted from the incremental index
    # nor (2) abort a full rebuild — the signature watermark is never null.
    search = pg_test_db
    _seed("A", "Alpha", None, ["alpha body"])
    assert search.build_index(full=False)["reindexed_units"] == 1
    assert "A" in _pg_units(search)
    assert search.hybrid_search("anything", topk=10)
    search.build_index(full=True)
    assert "A" in _pg_units(search)
