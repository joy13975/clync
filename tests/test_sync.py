"""claude.ai sync with the HTTP client mocked — proves chats/messages/projects/docs
land in the unified `units`/`messages` store, multi-org is covered, project_name is
backfilled, and the watermark skips unchanged convs."""
from __future__ import annotations

import clync


class _FakeClient:
    """Org-aware fake. per_org: {org_uuid: {convs:[full...], projects:[...],
    docs:{project_uuid:[doc...]}}}."""
    def __init__(self, per_org):
        self.per_org = per_org
        self.fetched: list[str] = []

    def list_conversations(self, org):
        return [{"uuid": c["uuid"], "updated_at": c["updated_at"]}
                for c in self.per_org[org]["convs"]]

    def get_conversation(self, org, cid):
        self.fetched.append(cid)
        return next(c for c in self.per_org[org]["convs"] if c["uuid"] == cid)

    def list_projects(self, org):
        return self.per_org[org].get("projects", [])

    def list_project_docs(self, org, puid):
        return self.per_org[org].get("docs", {}).get(puid, [])

    def get_file(self, url):                    # pragma: no cover - no files in fixtures
        raise AssertionError("no file downloads expected in these fixtures")


def _wire(monkeypatch, client, orgs):
    monkeypatch.setattr(clync, "read_auth_cookies", lambda profile: {})
    monkeypatch.setattr(clync, "ClaudeClient", lambda cookies, profile: client)
    monkeypatch.setattr(clync, "resolve_orgs", lambda c, ref: orgs)


def _conv(uuid, updated_at, text, project_uuid=None):
    return {"uuid": uuid, "name": uuid, "updated_at": updated_at,
            "created_at": "2026-01-01T00:00:00+00:00", "project_uuid": project_uuid,
            "chat_messages": [{"uuid": f"{uuid}-m0", "sender": "human",
                               "content": [{"type": "text", "text": text}],
                               "created_at": "2026-01-01T00:00:00+00:00"}]}


def _one(con, sql, *a):
    return con.execute(sql, a).fetchone()


def test_sync_stores_chats_projects_docs_as_units(pg_test_db, monkeypatch):
    per_org = {"org1": {
        "convs": [_conv("c1", "2026-02-01T00:00:00+00:00", "hello one", project_uuid="p1"),
                  _conv("c2", "2026-02-01T00:00:00+00:00", "hello two")],
        "projects": [{"uuid": "p1", "name": "Work", "description": "d",
                      "created_at": "2026-01-01T00:00:00+00:00",
                      "updated_at": "2026-01-02T00:00:00+00:00"}],
        "docs": {"p1": [{"uuid": "d1", "file_name": "audit.md",
                         "content": "CNX guy notes",
                         "created_at": "2026-01-01T00:00:00+00:00"}]},
    }}
    client = _FakeClient(per_org)
    _wire(monkeypatch, client, [{"uuid": "org1", "name": "Org2"}])

    stats = clync.run_sync("prof", None, full=False, download_files=False)
    assert stats["fetched"] == 2 and stats["projects"] == 1 and stats["project_docs"] == 1

    con = clync.connect()
    try:
        assert _one(con, "SELECT COUNT(*) n FROM units WHERE kind='chat'")["n"] == 2
        assert _one(con, "SELECT text FROM messages WHERE unit_id='c1'")["text"] == "hello one"
        # project association + backfilled project_name (the "in project X" facet)
        row = _one(con, "SELECT project_uuid, project_name FROM units WHERE unit_id='c1'")
        assert row["project_uuid"] == "p1" and row["project_name"] == "Work"
        # the knowledge doc is a retrievable project_doc unit + one message
        assert _one(con, "SELECT kind FROM units WHERE unit_id='d1'")["kind"] == "project_doc"
        assert _one(con, "SELECT text FROM messages WHERE unit_id='d1'"
                    )["text"] == "CNX guy notes"
    finally:
        con.close()


def test_sync_covers_all_orgs(pg_test_db, monkeypatch):
    per_org = {"orgA": {"convs": [_conv("a1", "2026-01-01T00:00:00+00:00", "from A")]},
               "orgB": {"convs": [_conv("b1", "2026-01-01T00:00:00+00:00", "from B")]}}
    client = _FakeClient(per_org)
    _wire(monkeypatch, client,
          [{"uuid": "orgA", "name": "A"}, {"uuid": "orgB", "name": "B"}])

    stats = clync.run_sync("prof", None, full=False, download_files=False)
    assert stats["fetched"] == 2                        # both orgs, not just the first
    con = clync.connect()
    try:
        orgs = {r["org_uuid"] for r in con.execute(
            "SELECT DISTINCT org_uuid FROM units WHERE kind='chat'").fetchall()}
    finally:
        con.close()
    assert orgs == {"orgA", "orgB"}


def test_sync_purges_project_deleted_upstream(pg_test_db, monkeypatch):
    # A whole project (with docs) deleted upstream must be purged locally, not
    # linger as orphan units that stay indexed/served.
    per_org = {"org1": {
        "convs": [],
        "projects": [{"uuid": "p1", "name": "Keep"},
                     {"uuid": "p2", "name": "Doomed"}],
        "docs": {"p1": [{"uuid": "d1", "file_name": "k.md", "content": "keep me",
                         "created_at": "2026-01-01T00:00:00+00:00"}],
                 "p2": [{"uuid": "d2", "file_name": "x.md", "content": "delete me",
                         "created_at": "2026-01-01T00:00:00+00:00"}]},
    }}
    client = _FakeClient(per_org)
    _wire(monkeypatch, client, [{"uuid": "org1", "name": "Org2"}])

    clync.run_sync("prof", None, full=False, download_files=False)
    con = clync.connect()
    try:
        assert {r["uuid"] for r in con.execute("SELECT uuid FROM projects")} == {"p1", "p2"}
        assert {r["unit_id"] for r in con.execute(
            "SELECT unit_id FROM units WHERE kind='project_doc'")} == {"d1", "d2"}
    finally:
        con.close()

    # p2 vanishes upstream -> next sync no longer lists it
    per_org["org1"]["projects"] = [{"uuid": "p1", "name": "Keep"}]
    per_org["org1"]["docs"].pop("p2")
    clync.run_sync("prof", None, full=False, download_files=False)

    con = clync.connect()
    try:
        assert {r["uuid"] for r in con.execute("SELECT uuid FROM projects")} == {"p1"}
        assert {r["unit_id"] for r in con.execute(
            "SELECT unit_id FROM units WHERE kind='project_doc'")} == {"d1"}
        # the doc's message went with it (FK cascade)
        assert con.execute("SELECT COUNT(*) n FROM messages WHERE unit_id='d2'"
                           ).fetchone()["n"] == 0
    finally:
        con.close()


def test_sync_incremental_skips_unchanged(pg_test_db, monkeypatch):
    per_org = {"org1": {"convs": [_conv("c1", "2026-01-01T00:00:00+00:00", "body")]}}
    client = _FakeClient(per_org)
    _wire(monkeypatch, client, [{"uuid": "org1", "name": "Org2"}])

    clync.run_sync("prof", None, full=False, download_files=False)      # first: fetches c1
    assert client.fetched == ["c1"]
    stats = clync.run_sync("prof", None, full=False, download_files=False)  # unchanged
    assert stats["skipped"] == 1 and stats["fetched"] == 0
    assert client.fetched == ["c1"]                     # get_conversation NOT called again


def test_sync_scrubs_nul_bytes_everywhere(pg_test_db, monkeypatch):
    """Regression (review round 3): a NUL byte anywhere in a claude.ai payload
    (pasted binary in message text, attachment extracted_content, even the
    conversation name) must never abort the sync. Postgres rejects 0x00 in
    `text` and its u0000 escape in jsonb, so BOTH parameter paths scrub it:
    connect_pg's str dumpers and _json's recursive scrub."""
    conv = _conv("c1", "2026-01-01T00:00:00+00:00", "log dump: a\x00b")
    conv["name"] = "bin\x00ary"
    conv["chat_messages"][0]["attachments"] = [
        {"file_name": "x.log", "extracted_content": "head\x00tail"}]
    per_org = {"org1": {"convs": [conv]}}
    _wire(monkeypatch, _FakeClient(per_org), [{"uuid": "org1", "name": "Org2"}])

    stats = clync.run_sync("prof", None, full=False, download_files=False)
    assert stats["fetched"] == 1

    con = clync.connect()
    try:
        u = _one(con, "SELECT title, raw FROM units WHERE unit_id='c1'")
        assert u["title"] == "binary"
        assert u["raw"]["name"] == "binary"               # raw jsonb scrubbed too
        m = _one(con, "SELECT text, raw FROM messages WHERE unit_id='c1'")
        assert "log dump: ab" in m["text"] and "\x00" not in m["text"]
        assert m["raw"]["attachments"][0]["extracted_content"] == "headtail"
    finally:
        con.close()
