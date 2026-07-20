"""Sync with the HTTP client mocked — proves conversations/messages/projects/docs
land in the DB, multi-org is covered, and the watermark skips unchanged convs."""
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
            "project_uuid": project_uuid,
            "chat_messages": [{"uuid": f"{uuid}-m0", "sender": "human",
                               "content": [{"type": "text", "text": text}],
                               "created_at": "2026-01-01"}]}


def test_sync_stores_conversations_projects_and_docs(sqlite_db, monkeypatch):
    per_org = {"org1": {
        "convs": [_conv("c1", "t1", "hello one", project_uuid="p1"),
                  _conv("c2", "t1", "hello two")],
        "projects": [{"uuid": "p1", "name": "Work", "description": "d",
                      "created_at": "2026-01-01", "updated_at": "2026-01-02"}],
        "docs": {"p1": [{"uuid": "d1", "file_name": "audit.md",
                         "content": "CNX guy notes", "created_at": "2026-01-01"}]},
    }}
    client = _FakeClient(per_org)
    _wire(monkeypatch, client, [{"uuid": "org1", "name": "Org2"}])

    stats = clync.run_sync("prof", None, full=False, download_files=False)
    assert stats["fetched"] == 2 and stats["projects"] == 1 and stats["project_docs"] == 1

    con = clync.connect()
    assert con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 2
    assert con.execute("SELECT text FROM messages WHERE conversation_uuid='c1'"
                       ).fetchone()[0] == "hello one"
    assert con.execute("SELECT project_uuid FROM conversations WHERE uuid='c1'"
                       ).fetchone()[0] == "p1"          # project chat association kept
    assert con.execute("SELECT content FROM project_docs WHERE uuid='d1'"
                       ).fetchone()[0] == "CNX guy notes"
    con.close()


def test_sync_covers_all_orgs(sqlite_db, monkeypatch):
    per_org = {"orgA": {"convs": [_conv("a1", "t1", "from A")]},
               "orgB": {"convs": [_conv("b1", "t1", "from B")]}}
    client = _FakeClient(per_org)
    _wire(monkeypatch, client,
          [{"uuid": "orgA", "name": "A"}, {"uuid": "orgB", "name": "B"}])

    stats = clync.run_sync("prof", None, full=False, download_files=False)
    assert stats["fetched"] == 2                        # both orgs synced, not just the first
    con = clync.connect()
    orgs = {r[0] for r in con.execute("SELECT DISTINCT org_uuid FROM conversations")}
    con.close()
    assert orgs == {"orgA", "orgB"}


def test_sync_purges_project_deleted_upstream(sqlite_db, monkeypatch):
    # A whole project (with docs) deleted upstream must be purged locally, not
    # linger as orphan rows that stay indexed/served.
    per_org = {"org1": {
        "convs": [],
        "projects": [{"uuid": "p1", "name": "Keep"},
                     {"uuid": "p2", "name": "Doomed"}],
        "docs": {"p1": [{"uuid": "d1", "file_name": "k.md", "content": "keep me",
                         "created_at": "2026-01-01"}],
                 "p2": [{"uuid": "d2", "file_name": "x.md", "content": "delete me",
                         "created_at": "2026-01-01"}]},
    }}
    client = _FakeClient(per_org)
    _wire(monkeypatch, client, [{"uuid": "org1", "name": "Org2"}])

    clync.run_sync("prof", None, full=False, download_files=False)
    con = clync.connect()
    assert {r[0] for r in con.execute("SELECT uuid FROM projects")} == {"p1", "p2"}
    assert {r[0] for r in con.execute("SELECT uuid FROM project_docs")} == {"d1", "d2"}
    con.close()

    # p2 vanishes upstream -> next sync no longer lists it
    per_org["org1"]["projects"] = [{"uuid": "p1", "name": "Keep"}]
    per_org["org1"]["docs"].pop("p2")
    clync.run_sync("prof", None, full=False, download_files=False)

    con = clync.connect()
    assert {r[0] for r in con.execute("SELECT uuid FROM projects")} == {"p1"}
    assert {r[0] for r in con.execute("SELECT uuid FROM project_docs")} == {"d1"}
    con.close()


def test_sync_incremental_skips_unchanged(sqlite_db, monkeypatch):
    per_org = {"org1": {"convs": [_conv("c1", "t1", "body")]}}
    client = _FakeClient(per_org)
    _wire(monkeypatch, client, [{"uuid": "org1", "name": "Org2"}])

    clync.run_sync("prof", None, full=False, download_files=False)      # first: fetches c1
    assert client.fetched == ["c1"]
    stats = clync.run_sync("prof", None, full=False, download_files=False)  # unchanged
    assert stats["skipped"] == 1 and stats["fetched"] == 0
    assert client.fetched == ["c1"]                     # get_conversation NOT called again
