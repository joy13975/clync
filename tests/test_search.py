"""Search + index integration: real Postgres (throwaway db), embedder mocked so
the SQL / RRF-fusion / watermark logic is what's under test, not BGE-M3."""
from __future__ import annotations

import clync

# valid timestamptz values (the chunks.updated_at column is a real timestamp)
T1, T2, T3 = "2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00"


def _seed(uuid, name, updated_at, texts, org="org1", project_uuid=None):
    """Insert a conversation + its messages into the (temp) SQLite store."""
    con = clync.connect()
    full = {"uuid": uuid, "name": name, "updated_at": updated_at,
            "project_uuid": project_uuid,
            "chat_messages": [{"uuid": f"{uuid}-m{i}", "sender": "human",
                               "content": [{"type": "text", "text": t}],
                               "created_at": "2026-01-01"}
                              for i, t in enumerate(texts)]}
    clync.upsert_conversation(con, org, full)
    con.commit()
    con.close()


def _seed_project(project_uuid, name, docs, org="org1"):
    """Insert a project + its knowledge docs. docs: [(uuid, file_name, content)]."""
    con = clync.connect()
    clync.upsert_project(con, org, {"uuid": project_uuid, "name": name})
    clync.upsert_project_docs(con, project_uuid,
        [{"uuid": du, "file_name": fn, "content": c, "created_at": "2026-01-01"}
         for du, fn, c in docs])
    con.commit()
    con.close()


def _delete(uuid):
    con = clync.connect()
    con.execute("DELETE FROM messages WHERE conversation_uuid=?", (uuid,))
    con.execute("DELETE FROM conversations WHERE uuid=?", (uuid,))
    con.commit()
    con.close()


def _pg_chunks(search):
    with search.connect_pg() as pg:
        return {r[0] for r in pg.execute("SELECT DISTINCT conv_uuid FROM chunks")}


def _pg_rows(search):
    """{conv_uuid: (conv_name, concatenated text)} — the embedded content per unit."""
    out: dict[str, tuple[str, str]] = {}
    with search.connect_pg() as pg:
        for cid, name, text in pg.execute(
                "SELECT conv_uuid, conv_name, text FROM chunks ORDER BY chunk_id"):
            prev = out.get(cid, (name, ""))
            out[cid] = (name, prev[1] + text)
    return out


def test_build_index_watermark_gone_and_stale_zero_chunk(sqlite_db, pg_test_db, mock_embed):
    search = pg_test_db

    _seed("A", "Alpha", T1, ["alpha body"])
    _seed("B", "Beta", T1, ["beta body"])
    st = search.build_index(full=False)
    assert st["reindexed_convs"] == 2
    assert _pg_chunks(search) == {"A", "B"}

    # nothing changed -> watermark skips everything
    assert search.build_index(full=False)["reindexed_convs"] == 0

    # A bumped (stale) + B removed from source (gone)
    _seed("A", "Alpha", T2, ["alpha rewritten"])
    _delete("B")
    st = search.build_index(full=False)
    assert st["reindexed_convs"] == 1 and st["removed_convs"] == 1
    assert _pg_chunks(search) == {"A"}                      # B purged

    # HARDEN BUG: a stale conv that now yields ZERO chunks must still be purged
    # AND have its watermark advanced (else it reindexes forever).
    _seed("A", "Alpha", T3, [""])                        # emptied -> 0 chunks
    st = search.build_index(full=False)
    assert st["reindexed_convs"] == 1
    assert _pg_chunks(search) == set()                     # A's old chunks gone
    assert search.build_index(full=False)["reindexed_convs"] == 0   # watermark advanced


def test_hybrid_search_collapse_topk_and_lang(sqlite_db, pg_test_db, mock_embed):
    search = pg_test_db
    _seed("A", "Alpha", T1, ["first chunk", "second chunk"])   # 2 chunks, one conv
    _seed("J", "日本語", T1, ["これは日本語の会話です"])          # ja conv
    search.build_index(full=False)

    hits = search.hybrid_search("anything", topk=10)
    assert len(hits) == 2                                   # DISTINCT ON collapsed A's 2 chunks
    assert len({h["conv_uuid"] for h in hits}) == 2

    assert len(search.hybrid_search("anything", topk=1)) == 1   # topk respected

    ja = search.hybrid_search("anything", topk=10, lang="ja")
    assert [h["conv_uuid"] for h in ja] == ["J"]            # lang filter restricts


def test_project_docs_and_project_chats_are_indexed(sqlite_db, pg_test_db, mock_embed):
    search = pg_test_db
    # a project, a project knowledge doc, and a chat associated with that project
    _seed_project("p1", "Work", [("doc1", "audit.md", "CNX infrastructure audit")])
    _seed("chat1", "CNX sour guy dance", T1, ["about the cnx guy"], project_uuid="p1")
    st = search.build_index(full=False)
    assert st["reindexed_convs"] == 2                       # 1 chat + 1 doc, both indexed

    hits = {h["conv_uuid"]: h["conv_name"] for h in search.hybrid_search("cnx", topk=10)}
    assert "doc1" in hits and hits["doc1"] == "[Work] audit.md"        # doc searchable
    assert "chat1" in hits and hits["chat1"] == "[Work] CNX sour guy dance"  # project prefix


def test_project_rename_reindexes_on_incremental(sqlite_db, pg_test_db, mock_embed):
    # CLASS: an embedded input owned OUTSIDE the unit's own watermark (the project
    # name folded into every chat + doc via the `[Project]` prefix) must still mark
    # the unit stale on the DEFAULT incremental run — the content-signature watermark.
    search = pg_test_db
    _seed_project("p1", "Work", [("doc1", "audit.md", "audit body")])
    _seed("chat1", "planning", T1, ["some chat body"], project_uuid="p1")
    assert search.build_index(full=False)["reindexed_convs"] == 2
    assert search.build_index(full=False)["reindexed_convs"] == 0    # steady state

    # Rename the project (chat.updated_at and doc.created_at are UNCHANGED).
    con = clync.connect()
    clync.upsert_project(con, "org1", {"uuid": "p1", "name": "Audit"})
    con.commit()
    con.close()

    # Incremental (default) must re-embed both dependent units and refresh the prefix.
    assert search.build_index(full=False)["reindexed_convs"] == 2
    rows = _pg_rows(search)
    assert rows["chat1"][0] == "[Audit] planning"
    assert rows["doc1"][0] == "[Audit] audit.md"
    assert search.build_index(full=False)["reindexed_convs"] == 0    # steady again


def test_doc_inplace_edit_reindexes_on_incremental(sqlite_db, pg_test_db, mock_embed):
    # CLASS: a doc edited in place, keeping the same uuid + created_at, must reindex
    # on the incremental run (the old timestamp watermark missed this).
    search = pg_test_db
    _seed_project("p1", "Work", [("doc1", "notes.md", "original content")])
    assert search.build_index(full=False)["reindexed_convs"] == 1
    assert search.build_index(full=False)["reindexed_convs"] == 0

    _seed_project("p1", "Work", [("doc1", "notes.md", "rewritten content")])  # same uuid+created_at
    assert search.build_index(full=False)["reindexed_convs"] == 1
    assert "rewritten content" in _pg_rows(search)["doc1"][1]
    assert search.build_index(full=False)["reindexed_convs"] == 0


def test_null_conversation_updated_at(sqlite_db, pg_test_db, mock_embed):
    # CLASS: a null updated_at must NOT (1) be silently omitted from the incremental
    # index nor (2) abort a full rebuild — the signature watermark is never null.
    search = pg_test_db
    _seed("A", "Alpha", None, ["alpha body"])           # updated_at = NULL
    assert search.build_index(full=False)["reindexed_convs"] == 1
    assert "A" in _pg_chunks(search)                    # indexed, not silently dropped
    assert search.hybrid_search("anything", topk=10)    # searchable
    search.build_index(full=True)                       # must not NOT-NULL abort
    assert "A" in _pg_chunks(search)
