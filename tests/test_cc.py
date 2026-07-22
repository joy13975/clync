"""Claude Code ingest — pure parsing/cleaning/metadata (no DB) + an ingest
integration test against a throwaway Postgres db over a tmp transcript root."""
from __future__ import annotations

import json

import pytest

import clync
import cc


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #
def test_repo_and_worktree():
    assert cc._repo_and_worktree(
        "/Users/j/code/example-repo/.claude/worktrees/some-branch/sub/dir"
    ) == ("example-repo", "some-branch")
    assert cc._repo_and_worktree("/Users/j/code/clync") == ("clync", None)
    assert cc._repo_and_worktree(None) == (None, None)


def test_tool_use_line_primary_arg():
    assert cc._tool_use_line({"name": "Bash", "input": {"command": "ls -la"}}) == "[Bash] ls -la"
    assert cc._tool_use_line({"name": "Read", "input": {"file_path": "/a/b.py"}}) == "[Read] /a/b.py"
    # unknown tool -> compact key listing, never a full dump
    assert cc._tool_use_line({"name": "Weird", "input": {"z": 1, "a": 2}}) == "[Weird] a, z"


def test_tool_result_head_truncation_and_agent_whole():
    big = "x" * 5000
    head = cc._render_tool_result({"content": big}, "Bash")
    assert head.startswith("[result: 5000 bytes, head]") and "…" in head
    assert len(head) < 2000                                   # truncated
    # Agent AND Task results are subagent syntheses — both kept whole.
    for tool in ("Agent", "Task"):
        whole = cc._render_tool_result({"content": big}, tool)
        assert whole.startswith("[result: 5000 bytes]") and whole.endswith("x")
    err = cc._render_tool_result({"content": "boom", "is_error": True}, "Bash")
    assert "ERROR" in err


def _event(**kw):
    return kw


def _write_session(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


def _cli_events():
    """A realistic transcript: event shapes mirror the real producer. Title events
    are UPDATES emitted repeatedly (`{"type":"ai-title","aiTitle":...}` /
    `{"type":"custom-title","customTitle":...}` — last one is current), and a
    `frame-link` event carries a bare `title` key (a published artifact's page
    title) that must never be read as the session title."""
    base = {"cwd": "/Users/j/code/clync", "gitBranch": "main", "version": "2.1.9",
            "entrypoint": "cli", "sessionId": "sess-1"}
    return [
        _event(type="ai-title", aiTitle="Investigate design question", sessionId="sess-1"),
        _event(type="user", uuid="u1", timestamp="2026-01-01T00:00:00Z",
               promptSource="typed", message={"role": "user", "content": "how do I design X"}, **base),
        _event(type="user", uuid="m1", isMeta=True, timestamp="2026-01-01T00:00:01Z",
               message={"role": "user", "content": "<local-command-caveat>noise"}, **base),
        _event(type="assistant", uuid="a1", timestamp="2026-01-01T00:00:02Z",
               message={"role": "assistant", "content": [
                   {"type": "thinking", "thinking": "", "signature": "sig"},   # redacted -> drop
                   {"type": "text", "text": "here is the plan"},
                   {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]}, **base),
        _event(type="user", uuid="u2", timestamp="2026-01-01T00:00:03Z",
               message={"role": "user", "content": [
                   {"type": "tool_result", "tool_use_id": "t1", "content": "a\nb\nc"}]}, **base),
        _event(type="ai-title", aiTitle="Design X end to end", sessionId="sess-1"),
        _event(type="frame-link", sessionId="sess-1", path="/tmp/mock.html",
               frameUrl="https://claude.ai/code/artifact/abc",
               title="Artifact page title — not the session's",
               timestamp="2026-01-01T00:00:04Z"),
        _event(type="custom-title", customTitle="My Design Session", sessionId="sess-1"),
        _event(type="mode", mode="default", sessionId="sess-1"),               # harness noise
    ]


def test_title_reads_only_typed_title_events_last_wins():
    # ai-title is an update stream: the LAST non-empty one is current.
    evs = [{"type": "ai-title", "aiTitle": "first ai title"},
           {"type": "ai-title", "aiTitle": "current ai title"}]
    assert cc._title_from(evs) == "current ai title"
    # custom-title beats ai-title regardless of event order.
    assert cc._title_from(
        [{"type": "custom-title", "customTitle": "named by user"}] + evs
    ) == "named by user"
    # An empty/whitespace update never clobbers the last real one.
    assert cc._title_from(evs + [{"type": "ai-title", "aiTitle": "  "}]) == "current ai title"
    # A bare `title` key on an unrelated event (frame-link = published-artifact
    # page title) is NEVER a session title.
    assert cc._title_from(
        [{"type": "frame-link", "title": "Artifact page", "frameUrl": "https://x"}]
    ) is None
    # Keys are read only from the event type that owns them.
    assert cc._title_from([{"type": "frame-link", "aiTitle": "smuggled"}]) is None
    # Fallback: first typed user prompt.
    assert cc._title_from(
        [{"type": "user", "promptSource": "typed",
          "message": {"role": "user", "content": "hello\nworld"}}]) == "hello world"


def test_parse_session_cleaning_and_metadata(tmp_path):
    p = tmp_path / "proj" / "sess-1.jsonl"
    _write_session(p, _cli_events())
    s = cc.parse_session(p)
    assert s is not None
    assert s.repo == "clync" and s.git_branch == "main" and s.cc_version == "2.1.9"
    assert s.entrypoint == "cli" and s.title == "My Design Session"
    # meta / redacted-thinking / harness lines dropped; 3 real messages kept
    assert len(s.messages) == 3
    kinds = [m.role_detail for m in s.messages]
    assert kinds[0] == "text"                                 # typed prompt
    assert "text" in kinds[1] and "tool_use:Bash" in kinds[1]  # assistant text+tool one-liner
    assert "[Bash] ls" in s.messages[1].text
    assert kinds[2] == "tool_result"
    # two-tier: a pure tool_result contributes no embed signal
    assert s.messages[2].embed_text == ""


def test_parse_session_skips_non_cli(tmp_path):
    events = _cli_events()
    for e in events:
        if "entrypoint" in e:
            e["entrypoint"] = "sdk-cli"
    p = tmp_path / "proj" / "batch.jsonl"
    _write_session(p, events)
    assert cc.parse_session(p) is None


def test_iter_session_files_skips_subagents(tmp_path, monkeypatch):
    _write_session(tmp_path / "proj" / "s1.jsonl", [{"type": "mode"}])
    _write_session(tmp_path / "proj" / "s1" / "subagents" / "agent-x.jsonl", [{"type": "mode"}])
    monkeypatch.setattr(cc, "CC_ROOT", tmp_path)
    found = [p.name for p in cc.iter_session_files()]
    assert found == ["s1.jsonl"]                              # subagent file not walked


def test_iter_session_files_unavailable_root_raises(tmp_path):
    """An unavailable source must raise eagerly at call time — never enumerate
    as empty (consumers reconcile deletions from this enumeration, so an
    ambiguous empty would purge the whole stored corpus)."""
    with pytest.raises(FileNotFoundError):
        cc.iter_session_files(tmp_path / "does-not-exist")
    not_a_dir = tmp_path / "file.jsonl"
    not_a_dir.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        cc.iter_session_files(not_a_dir)
    # an existing-but-empty root stays a legitimate empty result
    empty = tmp_path / "empty"
    empty.mkdir()
    assert list(cc.iter_session_files(empty)) == []


# --------------------------------------------------------------------------- #
# Ingest integration (real Postgres, tmp transcript root)
# --------------------------------------------------------------------------- #
def test_ingest_cc_end_to_end(pg_test_db, tmp_path, monkeypatch):
    _write_session(tmp_path / "proj" / "sess-1.jsonl", _cli_events())
    batch = _cli_events()
    for e in batch:
        if "entrypoint" in e:
            e["entrypoint"] = "sdk-cli"
        if e.get("uuid"):
            e["uuid"] += "-b"
        if e.get("sessionId"):
            e["sessionId"] = "sess-batch"
    _write_session(tmp_path / "proj" / "batch.jsonl", batch)
    monkeypatch.setattr(cc, "CC_ROOT", tmp_path)

    stats = clync.ingest_cc(full=True)
    assert stats["sessions_ingested"] == 1                    # batch (sdk-cli) skipped
    assert stats["files_parsed"] == 2                         # both files examined+watermarked
    assert stats["sessions_removed"] == 0                     # nothing deleted on disk

    con = clync.connect()
    try:
        rows = con.execute(
            "SELECT unit_id, repo, git_branch, title, msg_count FROM units "
            "WHERE source='claude_code'").fetchall()
        assert len(rows) == 1
        assert rows[0]["repo"] == "clync" and rows[0]["git_branch"] == "main"
        assert rows[0]["title"] == "My Design Session"        # last custom-title wins
        stored = con.execute("SELECT COUNT(*) n FROM messages WHERE unit_id='sess-1'"
                             ).fetchone()["n"]
        assert stored == 3
        assert rows[0]["msg_count"] == stored                 # msg_count == retrievable rows
    finally:
        con.close()

    # incremental: nothing changed -> everything skipped
    stats2 = clync.ingest_cc(full=False)
    assert stats2["sessions_ingested"] == 0 and stats2["unchanged"] == 2


def _resume_pair():
    """A resumed session, as the real producer writes it: sess-B's file replays
    sess-A's event uuids under B's OWN sessionId, then appends new turns.
    (Measured on real transcripts: thousands of event uuids appear in >1 file,
    always under differing sessionIds.)"""
    def _base(sid):
        return {"cwd": "/Users/j/code/clync", "gitBranch": "main",
                "version": "2.1.9", "entrypoint": "cli", "sessionId": sid}

    def _turns(sid):
        return [
            _event(type="user", uuid="ra1", timestamp="2026-01-02T00:00:00Z",
                   promptSource="typed",
                   message={"role": "user", "content": "start the work"}, **_base(sid)),
            _event(type="assistant", uuid="ra2", timestamp="2026-01-02T00:00:01Z",
                   message={"role": "assistant", "content": [
                       {"type": "text", "text": "the shared prefix answer"}]}, **_base(sid)),
        ]

    a = _turns("sess-A")
    b = _turns("sess-B") + [
        _event(type="user", uuid="rb1", timestamp="2026-01-02T01:00:00Z",
               promptSource="typed",
               message={"role": "user", "content": "continue after resume"}, **_base("sess-B")),
        _event(type="assistant", uuid="rb2", timestamp="2026-01-02T01:00:01Z",
               message={"role": "assistant", "content": [
                   {"type": "text", "text": "the post-resume answer"}]}, **_base("sess-B")),
    ]
    return a, b


def test_ingest_cc_resume_pair_complete_transcripts(pg_test_db, tmp_path, monkeypatch):
    """A resumed session must be COMPLETE at the read boundary: its unit stores
    the whole replayed prefix + tail, msg_count equals the stored rows, and
    get_conversation renders the full transcript (never starting mid-stream)."""
    a, b = _resume_pair()
    _write_session(tmp_path / "proj" / "sess-A.jsonl", a)
    _write_session(tmp_path / "proj" / "sess-B.jsonl", b)
    monkeypatch.setattr(cc, "CC_ROOT", tmp_path)

    stats = clync.ingest_cc(full=True)
    assert stats["sessions_ingested"] == 2

    con = clync.connect()
    try:
        for unit_id, expect in (("sess-A", 2), ("sess-B", 4)):
            stored = con.execute(
                "SELECT COUNT(*) n FROM messages WHERE unit_id=%s",
                (unit_id,)).fetchone()["n"]
            msg_count = con.execute(
                "SELECT msg_count FROM units WHERE unit_id=%s",
                (unit_id,)).fetchone()["msg_count"]
            assert stored == expect                # complete transcript per unit
            assert msg_count == stored             # msg_count never lies
    finally:
        con.close()

    import mcp_server
    transcript = mcp_server.get_conversation("sess-B")
    # the shared prefix AND the tail — a resumed unit never starts mid-stream
    assert "start the work" in transcript
    assert "the shared prefix answer" in transcript
    assert "the post-resume answer" in transcript


def test_ingest_cc_reconciles_deleted_transcripts(pg_test_db, tmp_path, monkeypatch):
    """A transcript deleted from disk must purge its unit, messages, and
    watermark row on the next ingest — never silently persist as stale data."""
    p = tmp_path / "proj" / "sess-1.jsonl"
    _write_session(p, _cli_events())
    monkeypatch.setattr(cc, "CC_ROOT", tmp_path)
    clync.ingest_cc(full=True)

    p.unlink()
    stats = clync.ingest_cc()
    assert stats["sessions_removed"] == 1

    con = clync.connect()
    try:
        for q in ("SELECT COUNT(*) n FROM units WHERE source='claude_code'",
                  "SELECT COUNT(*) n FROM messages",
                  "SELECT COUNT(*) n FROM cc_sync_state"):
            assert con.execute(q).fetchone()["n"] == 0
    finally:
        con.close()


def test_ingest_cc_missing_root_raises_and_store_untouched(pg_test_db, tmp_path, monkeypatch):
    """Regression (review round 2): an unavailable transcript root (unmounted
    volume, moved ~/.claude/projects) must FAIL the ingest, not read as 'all
    transcripts deleted' and purge the entire cc corpus."""
    _write_session(tmp_path / "proj" / "sess-1.jsonl", _cli_events())
    monkeypatch.setattr(cc, "CC_ROOT", tmp_path)
    clync.ingest_cc(full=True)

    monkeypatch.setattr(cc, "CC_ROOT", tmp_path / "gone")
    with pytest.raises(FileNotFoundError):
        clync.ingest_cc()

    con = clync.connect()
    try:
        assert con.execute("SELECT COUNT(*) n FROM units WHERE source='claude_code'"
                           ).fetchone()["n"] == 1
        assert con.execute("SELECT COUNT(*) n FROM cc_sync_state").fetchone()["n"] == 1
        assert con.execute("SELECT COUNT(*) n FROM messages").fetchone()["n"] > 0
    finally:
        con.close()
