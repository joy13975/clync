"""Sync with the HTTP client mocked — proves conversations/messages land in the
DB with text extracted, and that the incremental watermark skips unchanged convs."""
from __future__ import annotations

import clync


class _FakeClient:
    def __init__(self, listing, details):
        self._listing, self._details = listing, details
        self.fetched: list[str] = []

    def list_conversations(self, org_uuid):
        return self._listing

    def get_conversation(self, org_uuid, cid):
        self.fetched.append(cid)
        return self._details[cid]

    def get_file(self, url):                       # pragma: no cover - no files in fixtures
        raise AssertionError("no file downloads expected in these fixtures")


def _wire(monkeypatch, client):
    monkeypatch.setattr(clync, "read_auth_cookies", lambda profile: {})
    monkeypatch.setattr(clync, "ClaudeClient", lambda cookies, profile: client)
    monkeypatch.setattr(clync, "resolve_org",
                        lambda c, ref: {"uuid": "org1", "name": "Test Org"})


def _conv(uuid, updated_at, text):
    return {"uuid": uuid, "name": uuid, "updated_at": updated_at,
            "chat_messages": [{"uuid": f"{uuid}-m0", "sender": "human",
                               "content": [{"type": "text", "text": text}],
                               "created_at": "2026-01-01"}]}


def test_sync_stores_conversations_with_extracted_text(sqlite_db, monkeypatch):
    listing = [{"uuid": "c1", "updated_at": "t1"}, {"uuid": "c2", "updated_at": "t1"}]
    details = {"c1": _conv("c1", "t1", "hello from one"),
               "c2": _conv("c2", "t1", "hello from two")}
    client = _FakeClient(listing, details)
    _wire(monkeypatch, client)

    stats = clync.run_sync("prof", None, full=False, download_files=False)
    assert stats["fetched"] == 2 and stats["skipped"] == 0

    con = clync.connect()
    assert con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 2
    txt = con.execute("SELECT text FROM messages WHERE conversation_uuid='c1'").fetchone()[0]
    con.close()
    assert txt == "hello from one"                 # text extracted from content blocks


def test_sync_incremental_skips_unchanged(sqlite_db, monkeypatch):
    listing = [{"uuid": "c1", "updated_at": "t1"}]
    details = {"c1": _conv("c1", "t1", "body")}
    client = _FakeClient(listing, details)
    _wire(monkeypatch, client)

    clync.run_sync("prof", None, full=False, download_files=False)      # first: fetches c1
    assert client.fetched == ["c1"]

    stats = clync.run_sync("prof", None, full=False, download_files=False)  # second: unchanged
    assert stats["skipped"] == 1 and stats["fetched"] == 0
    assert client.fetched == ["c1"]                # get_conversation NOT called again
