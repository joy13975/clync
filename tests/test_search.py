"""Search + index integration: real Postgres (throwaway db), embedder mocked so
the SQL / RRF-fusion / watermark logic is what's under test, not BGE-M3."""
from __future__ import annotations

import clync

# valid timestamptz values (the chunks.updated_at column is a real timestamp)
T1, T2, T3 = "2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00"


def _seed(uuid, name, updated_at, texts, org="org1"):
    """Insert a conversation + its messages into the (temp) SQLite store."""
    con = clync.connect()
    full = {"uuid": uuid, "name": name, "updated_at": updated_at,
            "chat_messages": [{"uuid": f"{uuid}-m{i}", "sender": "human",
                               "content": [{"type": "text", "text": t}],
                               "created_at": "2026-01-01"}
                              for i, t in enumerate(texts)]}
    clync.upsert_conversation(con, org, full)
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
