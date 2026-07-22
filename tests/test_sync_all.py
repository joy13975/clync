"""The integrated daily-sync chain (`_sync_all`) — the exact orchestration the
launchd job runs via `cmd_scheduled`: claude.ai sync THEN local Claude Code ingest
THEN a single index pass, all in one process. These prove the chain end to end:
both sources land AND become searchable from one run, and a dead claude.ai cookie
never blocks local capture or indexing (the failure is returned, not swallowed)."""
from __future__ import annotations

import cc
import clync
from tests.test_cc import _cli_events, _write_session
from tests.test_sync import _FakeClient, _conv, _wire


def _wire_cc(tmp_path, monkeypatch):
    _write_session(tmp_path / "proj" / "sess-1.jsonl", _cli_events())
    monkeypatch.setattr(cc, "CC_ROOT", tmp_path)


def test_sync_all_indexes_both_sources_searchable(pg_test_db, tmp_path,
                                                  monkeypatch, mock_embed):
    """One `_sync_all(do_index=True)` run must leave BOTH a claude.ai chat and a
    local Claude Code session stored AND retrievable by hybrid_search — i.e. the
    daily job auto-embeds everything it synced, in one pass."""
    search = pg_test_db
    per_org = {"org1": {"convs": [_conv("c1", "2026-02-01T00:00:00+00:00", "hello from the app")]}}
    _wire(monkeypatch, _FakeClient(per_org), [{"uuid": "org1", "name": "Org2"}])
    _wire_cc(tmp_path, monkeypatch)

    err = clync._sync_all("prof", None, full=True, download_files=False,
                          do_index=True, notify_fail=False)
    assert err is None                                  # app leg succeeded -> nothing to re-raise

    con = clync.connect()
    try:
        kinds = {r["source"] for r in con.execute(
            "SELECT DISTINCT source FROM units").fetchall()}
        assert kinds == {"claude_ai", "claude_code"}    # both sources stored in one run
    finally:
        con.close()

    # both sources are in the freshly built index and reachable via search
    app_hits = {h["unit_id"] for h in search.hybrid_search("", source="claude_ai", topk=50)}
    cc_hits = {h["unit_id"] for h in search.hybrid_search("", source="claude_code", topk=50)}
    assert "c1" in app_hits                             # app chat auto-embedded + searchable
    assert "sess-1" in cc_hits                          # cc session auto-embedded + searchable


def test_sync_all_app_failure_still_ingests_and_indexes_cc(pg_test_db, tmp_path,
                                                           monkeypatch, mock_embed):
    """A dead/expired claude.ai cookie must NOT block local capture: the app error
    is returned (for the caller to re-raise, fail-loud) but cc ingest AND the index
    still run, so local sessions stay searchable even when the app leg is down."""
    search = pg_test_db

    class _BoomClient(_FakeClient):
        def list_conversations(self, org):
            raise RuntimeError("401 stale cookie")

    monkeypatch.setattr(clync, "read_auth_cookies", lambda profile: {})
    monkeypatch.setattr(clync, "ClaudeClient", lambda cookies, profile: _BoomClient({}))
    monkeypatch.setattr(clync, "resolve_orgs", lambda c, ref: [{"uuid": "org1", "name": "Org2"}])
    _wire_cc(tmp_path, monkeypatch)

    err = clync._sync_all("prof", None, full=True, download_files=False,
                          do_index=True, notify_fail=False)
    assert isinstance(err, RuntimeError)                # captured for loud re-raise by caller
    assert "stale cookie" in str(err)

    # cc ran despite the app failure, and the index picked it up
    cc_hits = {h["unit_id"] for h in search.hybrid_search("", source="claude_code", topk=50)}
    assert "sess-1" in cc_hits
