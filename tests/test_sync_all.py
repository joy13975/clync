"""The integrated daily-sync chain (`_sync_all`) — the exact orchestration the
launchd job runs via `cmd_scheduled`: claude.ai sync THEN local Claude Code ingest
THEN a single index pass, all in one process. These prove the chain end to end:
all raw sources land AND become searchable from one run, and a dead network
cookie never blocks local capture or indexing (the failure is returned, not swallowed)."""
from __future__ import annotations

import json
import types

import cc
import clync
from tests.test_cc import _cli_events, _write_session
from tests.test_chatgpt import _FakeCGPTClient, _pair, _wire_cgpt
from tests.test_opencode import _wire_opencode
from tests.test_sync import _conv, _FakeClient, _wire


def _wire_cc(tmp_path, monkeypatch):
    _write_session(tmp_path / "proj" / "sess-1.jsonl", _cli_events())
    monkeypatch.setattr(cc, "CC_ROOT", tmp_path)


def _wire_codex(tmp_path, monkeypatch):
    """Seed one real-shaped Codex rollout under a sandboxed root and point the
    Codex ingest there — so `_sync_all` ingests it without touching real ~/.codex."""
    import codex
    root = tmp_path / "codex"
    rollout = root / "sessions" / "2026" / "01" / "01" / "rollout-2026-01-01T00-00-00-cdx1.jsonl"
    rollout.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        {"timestamp": "2026-01-01T00:00:00Z", "type": "session_meta",
         "payload": {"id": "cdx1", "cwd": "/Users/example/code/clync",
                     "originator": "codex_cli_rs", "cli_version": "0.66.0"}},
        {"timestamp": "2026-01-01T00:00:01Z", "type": "turn_context",
         "payload": {"cwd": "/Users/example/code/clync", "model": "gpt-5.1-codex"}},
        {"timestamp": "2026-01-01T00:00:02Z", "type": "response_item",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "hello from codex cli"}]}},
        {"timestamp": "2026-01-01T00:00:03Z", "type": "response_item",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "the codex answer"}]}},
    ]
    rollout.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    monkeypatch.setattr(codex, "CODEX_ROOT", root)


def test_sync_all_indexes_all_sources_searchable(pg_test_db, tmp_path,
                                                 monkeypatch, mock_embed):
    search = pg_test_db
    per_org = {"org1": {"convs": [_conv("c1", "2026-02-01T00:00:00+00:00", "hello from the app")]}}
    _wire(monkeypatch, _FakeClient(per_org), [{"uuid": "org1", "name": "Example Org"}])
    _wire_cc(tmp_path, monkeypatch)
    _wire_codex(tmp_path, monkeypatch)
    _wire_opencode(tmp_path, monkeypatch)
    _wire_cgpt(monkeypatch, _FakeCGPTClient([
        _pair("cg1", "2026-02-01T00:00:00Z", "a chatgpt web question", "the chatgpt answer")]))

    err = clync._sync_all("prof", None, full=True, download_files=False,
                          do_index=True, notify_fail=False)
    assert err is None                                  # both network legs succeeded

    con = clync.connect()
    try:
        kinds = {r["source"] for r in con.execute(
            "SELECT DISTINCT source FROM units").fetchall()}
        assert kinds == {"claude_ai", "claude_code", "codex_cli", "chatgpt", "opencode"}
    finally:
        con.close()

    app_hits = {h["unit_id"] for h in search.hybrid_search("hello", source="claude_ai", topk=50)}
    cc_hits = {h["unit_id"] for h in search.hybrid_search("design", source="claude_code", topk=50)}
    codex_hits = {h["unit_id"] for h in search.hybrid_search("codex", source="codex_cli", topk=50)}
    cgpt_hits = {h["unit_id"] for h in search.hybrid_search("chatgpt", source="chatgpt", topk=50)}
    assert "c1" in app_hits                             # app chat auto-embedded + searchable
    assert "sess-1" in cc_hits                          # cc session auto-embedded + searchable
    assert "cdx1" in codex_hits                         # codex session auto-embedded + searchable
    assert "cg1" in cgpt_hits                           # chatgpt conv auto-embedded + searchable
    assert "opencode:ses_parent" in {h["unit_id"] for h in search.hybrid_search(
        "heliotrope", source="opencode", topk=50)}


def test_opencode_failure_does_not_block_other_sources_or_index(
        pg_test_db, tmp_path, monkeypatch, mock_embed):
    _wire(monkeypatch, _FakeClient({"org1": {"convs": []}}), [{"uuid": "org1", "name": "Example Org"}])
    _wire_cgpt(monkeypatch, _FakeCGPTClient([]))
    _wire_cc(tmp_path, monkeypatch)
    state, _ = _wire_opencode(tmp_path, monkeypatch)
    data = json.loads(state.read_text())
    data["fail"] = "transport"
    state.write_text(json.dumps(data))
    err = clync._sync_all("prof", None, full=False, download_files=False,
                          do_index=True, notify_fail=False)
    assert err is not None
    assert "sess-1" in {h["unit_id"] for h in pg_test_db.hybrid_search(
        "design", source="claude_code", topk=50)}


def test_scheduled_without_dream_still_syncs_and_indexes(
        pg_test_db, tmp_path, monkeypatch, mock_embed):
    import dream

    _wire(monkeypatch, _FakeClient({"org1": {"convs": [
        _conv("c1", "2026-02-01T00:00:00+00:00", "hello from the app")
    ]}}), [{"uuid": "org1", "name": "Example Org"}])
    _wire_cgpt(monkeypatch, _FakeCGPTClient([]))
    _wire_cc(tmp_path, monkeypatch)
    monkeypatch.setattr(clync, "notify", lambda *a: None)
    with clync.connect() as con:
        assert clync.get_meta(con, dream.WATERMARK_KEY) is None

    assert clync.cmd_scheduled(types.SimpleNamespace(
        claude_profile="prof", chatgpt_profile="prof", org=None, no_dream=True)) == 0
    assert "c1" in {h["unit_id"] for h in pg_test_db.hybrid_search(
        "hello from the app", source="claude_ai", topk=50)}
    assert "sess-1" in {h["unit_id"] for h in pg_test_db.hybrid_search(
        "design", source="claude_code", topk=50)}
    with clync.connect() as con:
        assert clync.get_meta(con, dream.WATERMARK_KEY) is None


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
    monkeypatch.setattr(clync, "resolve_orgs", lambda c, ref: [{"uuid": "org1", "name": "Example Org"}])
    _wire_cc(tmp_path, monkeypatch)
    _wire_cgpt(monkeypatch, _FakeCGPTClient([]))        # ChatGPT leg OK -> only the app leg fails

    err = clync._sync_all("prof", None, full=True, download_files=False,
                          do_index=True, notify_fail=False)
    assert isinstance(err, RuntimeError)                # captured for loud re-raise by caller
    assert "stale cookie" in str(err)

    # cc ran despite the app failure, and the index picked it up
    cc_hits = {h["unit_id"] for h in search.hybrid_search("design", source="claude_code", topk=50)}
    assert "sess-1" in cc_hits


def test_sync_all_routes_network_sources_to_distinct_profiles(
        pg_test_db, tmp_path, monkeypatch, mock_embed):
    """Claude.ai and ChatGPT must not share a Chrome profile: the former is the
    work account and the latter is personal. A shared profile silently mixes
    accounts in one run, which is the source-scope failure this guards."""
    import chatgpt

    seen: list[tuple[str, str]] = []
    claude_client = _FakeClient({"org1": {"convs": []}})
    cgpt_client = _FakeCGPTClient([])
    monkeypatch.setattr(clync, "read_auth_cookies", lambda profile: {})
    monkeypatch.setattr(clync, "ClaudeClient",
                        lambda cookies, profile: (seen.append(("claude", profile))
                                                   or claude_client))
    monkeypatch.setattr(clync, "resolve_orgs",
                        lambda client, ref: [{"uuid": "org1", "name": "Example Org"}])
    monkeypatch.setattr(clync, "read_chatgpt_cookies",
                        lambda profile: (seen.append(("chatgpt", profile))
                                         or {chatgpt.SESSION_COOKIE: "x"}))
    monkeypatch.setattr(chatgpt, "ChatGPTClient",
                        lambda cookies, profile: cgpt_client)
    _wire_cc(tmp_path, monkeypatch)

    err = clync._sync_all("ClaudeProfile", None, full=True, download_files=False,
                          do_index=True, notify_fail=False,
                          chatgpt_profile="ChatGPTProfile")
    assert err is None
    assert ("claude", "ClaudeProfile") in seen
    assert ("chatgpt", "ChatGPTProfile") in seen
