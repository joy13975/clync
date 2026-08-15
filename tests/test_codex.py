"""Codex CLI ingest — pure parsing/cleaning/metadata (no DB)."""
from __future__ import annotations

import json

import pytest

import cc
import codex


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #
def test_repo_and_worktree_reused_from_cc():
    # codex.py imports cc._repo_and_worktree rather than duplicating it (SSOT).
    assert codex._repo_and_worktree is cc._repo_and_worktree
    assert codex._repo_and_worktree(
        "/Users/j/code/example-repo/.claude/worktrees/some-branch/sub/dir"
    ) == ("example-repo", "some-branch")
    assert codex._repo_and_worktree("/Users/j/code/clync") == ("clync", None)
    assert codex._repo_and_worktree(None) == (None, None)


def test_concat_text_ignores_non_text_parts():
    parts = [{"type": "input_text", "text": "hello"}, {"type": "input_image", "image_url": "x"},
             {"type": "output_text", "text": "world"}]
    assert codex._concat_text(parts) == "hello\nworld"
    assert codex._concat_text([]) == ""


def test_concat_summary_reads_summary_text_only():
    parts = [{"type": "summary_text", "text": "assessing"}, {"type": "other", "text": "ignore"}]
    assert codex._concat_summary(parts) == "assessing"
    assert codex._concat_summary([]) == ""


def test_tool_call_line_truncates():
    short = codex._tool_call_line("apply_patch", "*** Begin Patch")
    assert short == "[apply_patch] *** Begin Patch"
    long_arg = "x" * 500
    long = codex._tool_call_line("shell", long_arg)
    assert long.startswith("[shell] ") and long.endswith("…") and len(long) < 300


def test_render_tool_output_head_truncation():
    big = "x" * 5000
    head = codex._render_tool_output(big)
    assert head.startswith("[result: 5000 bytes, head]") and "…" in head
    assert len(head) < 2000
    small = codex._render_tool_output("ok")
    assert small == "[result: 2 bytes]\nok"


def test_output_text_handles_string_and_image_list():
    # verified on real rollouts: an image-capturing tool's output is a list of
    # content parts (no text), not a plain string.
    assert codex._output_text("plain string") == "plain string"
    assert codex._output_text([{"type": "input_image", "image_url": "data:..."}]) == ""
    assert codex._output_text(
        [{"type": "input_text", "text": "caption"}, {"type": "input_image", "image_url": "x"}]
    ) == "caption"
    assert codex._output_text(None) == ""


def test_is_injection_prefixes():
    assert codex._is_injection("<environment_context>\n  <cwd>/x</cwd>")
    assert codex._is_injection("# AGENTS.md instructions for /repo")
    assert codex._is_injection("<user_instructions>do things")
    assert codex._is_injection("<INSTRUCTIONS>\nfoo")
    assert not codex._is_injection("please fix the bug")
    # leading whitespace before the marker still counts (lstrip)
    assert codex._is_injection("  <environment_context>noise")


# --------------------------------------------------------------------------- #
# Synthetic rollout construction
# --------------------------------------------------------------------------- #
def _line(**kw):
    return kw


def _write_rollout(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")


def _rollout_lines(session_id="019b08b8-bfea-7542-af4d-8e75278b2251", originator="codex_cli_rs"):
    return [
        _line(timestamp="2026-01-01T00:00:00.000Z", type="session_meta", payload={
            "id": session_id, "timestamp": "2026-01-01T00:00:00.000Z",
            "cwd": "/Users/j/code/clync", "originator": originator,
            "cli_version": "0.66.0", "instructions": "repo guidelines..."}),
        _line(timestamp="2026-01-01T00:00:01.000Z", type="response_item", payload={
            "type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "<environment_context>\n  <cwd>/x</cwd>\n"
                                                "</environment_context>"}]}),
        _line(timestamp="2026-01-01T00:00:02.000Z", type="turn_context", payload={
            "cwd": "/Users/j/code/clync", "model": "gpt-5.1-codex", "effort": "medium"}),
        _line(timestamp="2026-01-01T00:00:03.000Z", type="response_item", payload={
            "type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "how do I design X"}]}),
        _line(timestamp="2026-01-01T00:00:04.000Z", type="response_item", payload={
            "type": "reasoning", "summary": [
                {"type": "summary_text", "text": "assessing the design question"}],
            "content": None, "encrypted_content": "gAAAAA..."}),
        _line(timestamp="2026-01-01T00:00:05.000Z", type="response_item", payload={
            "type": "function_call", "name": "shell", "arguments": '{"command": "ls -la"}',
            "call_id": "call_1"}),
        _line(timestamp="2026-01-01T00:00:06.000Z", type="response_item", payload={
            "type": "function_call_output", "call_id": "call_1", "output": "a\nb\nc"}),
        _line(timestamp="2026-01-01T00:00:07.000Z", type="response_item", payload={
            "type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "here is the plan"}]}),
        _line(timestamp="2026-01-01T00:00:08.000Z", type="response_item", payload={
            "type": "ghost_snapshot", "ghost_commit": {"id": "abc"}}),
        _line(timestamp="2026-01-01T00:00:09.000Z", type="event_msg", payload={
            "type": "agent_message", "message": "duplicate echo of the plan"}),
    ]


def test_parse_session_cleaning_and_metadata(tmp_path):
    p = tmp_path / "sessions" / "2026" / "01" / "01" / "rollout-x.jsonl"
    _write_rollout(p, _rollout_lines())
    s = codex.parse_session(p)
    assert s is not None
    assert s.session_id == "019b08b8-bfea-7542-af4d-8e75278b2251"
    assert s.repo == "clync" and s.model == "gpt-5.1-codex" and s.cc_version == "0.66.0"
    assert s.git_branch is None and s.entrypoint is None and s.summary is None
    # environment_context injection dropped; event_msg stream never read
    assert not any("<environment_context>" in m.text for m in s.messages)
    assert not any("duplicate echo" in m.text for m in s.messages)

    kinds = [(m.sender, m.role_detail) for m in s.messages]
    assert ("user", "text") in kinds
    assert ("assistant", "reasoning") in kinds
    assert any(rd.startswith("tool_use:") for _, rd in kinds)
    assert ("user", "tool_result") in kinds
    assert ("assistant", "text") in kinds

    # title = first real user prompt, truncated
    assert s.title == "how do I design X"

    # msg_id synthesized positionally, unique within the session
    ids = [m.msg_id for m in s.messages]
    assert ids == [f"{s.session_id}-{i}" for i in range(len(s.messages))]

    # pure tool_result carries no embed signal
    tool_result_msg = next(m for m in s.messages if m.role_detail == "tool_result")
    assert tool_result_msg.embed_text == ""
    # everything else reuses `text` for embedding
    text_msg = next(m for m in s.messages if m.role_detail == "text")
    assert text_msg.embed_text is None


# The originator scope is one SSOT set (codex._SCOPE_ORIGINATORS); the gate and
# these tests read the SAME constant so they cannot drift. Out-of-scope values
# are the real ones observed in a ~/.codex corpus, not a fabricated placeholder.
_OUT_OF_SCOPE_ORIGINATORS = ("codex_exec", "Codex Desktop", "codex_work_desktop")


@pytest.mark.parametrize("originator", sorted(codex._SCOPE_ORIGINATORS))
def test_parse_session_ingests_every_in_scope_originator(tmp_path, originator):
    """Every originator in the SSOT scope set ingests — parametrized from the
    constant the gate reads, so a future edit to one but not the other is caught."""
    p = tmp_path / "sessions" / "2026" / "01" / "01" / "rollout-x.jsonl"
    _write_rollout(p, _rollout_lines(originator=originator))
    s = codex.parse_session(p)
    assert isinstance(s, codex.CodexSession)
    assert s.session_id == "019b08b8-bfea-7542-af4d-8e75278b2251"


def test_parse_session_ingests_current_codex_tui_originator(tmp_path):
    """Regression: `codex-tui` is the CURRENT interactive Codex CLI (the rename of
    `codex_cli_rs`), the exact analog of cc.py's `entrypoint == "cli"`. The
    pre-fix gate hardcoded `originator == "codex_cli_rs"` and silently dropped
    every current session — this must ingest, not return None/OUT_OF_SCOPE."""
    p = tmp_path / "sessions" / "2026" / "01" / "01" / "rollout-x.jsonl"
    _write_rollout(p, _rollout_lines(originator="codex-tui"))
    s = codex.parse_session(p)
    assert isinstance(s, codex.CodexSession)
    assert s.model == "gpt-5.1-codex"


@pytest.mark.parametrize("originator", _OUT_OF_SCOPE_ORIGINATORS)
def test_parse_session_out_of_scope_returns_sentinel_not_none(tmp_path, originator):
    """Out-of-scope producers (non-interactive exec, desktop GUI) return the
    DISTINCT OUT_OF_SCOPE sentinel — not None, not a session — so the ingest layer
    can count them separately and a whole-corpus scope drop can never read as a
    silent healthy run."""
    p = tmp_path / "rollout-x.jsonl"
    _write_rollout(p, _rollout_lines(originator=originator))
    assert codex.parse_session(p) is codex.OUT_OF_SCOPE


def test_parse_session_no_session_meta_returns_none(tmp_path):
    lines = [ln for ln in _rollout_lines() if ln["type"] != "session_meta"]
    p = tmp_path / "rollout-x.jsonl"
    _write_rollout(p, lines)
    assert codex.parse_session(p) is None


def test_parse_session_skips_corrupt_line_not_whole_file(tmp_path):
    lines = _rollout_lines()
    p = tmp_path / "rollout-x.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(ln) for ln in lines)
    text += "\nthis is not json\n"
    p.write_text(text, encoding="utf-8")
    s = codex.parse_session(p)
    assert s is not None
    assert len(s.messages) > 0


def test_parse_session_survives_truncated_utf8_tail(tmp_path):
    """A Codex session killed mid-write leaves a truncated multibyte UTF-8
    sequence on its final line. That partial-write tail must degrade like a
    corrupt JSON line (the line is skipped), NOT raise UnicodeDecodeError and
    abort the whole ingest (which would starve every subsequent sync until the
    user manually removed the file)."""
    lines = _rollout_lines()
    p = tmp_path / "rollout-x.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    good = "\n".join(json.dumps(ln) for ln in lines).encode("utf-8")
    # incomplete 3-byte UTF-8 sequence at EOF (first two bytes of "€", no third)
    p.write_bytes(good + b"\n\xe2\x82")
    s = codex.parse_session(p)          # must not raise
    assert s is not None
    assert len(s.messages) > 0          # the valid earlier lines still parsed


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_iter_session_files_unavailable_root_raises(tmp_path):
    """An unavailable source must raise eagerly at call time — never enumerate
    as empty (consumers reconcile deletions from this enumeration, so an
    ambiguous empty would purge the whole stored corpus)."""
    with pytest.raises(FileNotFoundError):
        codex.iter_session_files(tmp_path / "does-not-exist")
    not_a_dir = tmp_path / "file.jsonl"
    not_a_dir.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        codex.iter_session_files(not_a_dir)
    # an existing-but-empty root stays a legitimate empty result
    empty = tmp_path / "empty"
    empty.mkdir()
    assert list(codex.iter_session_files(empty)) == []


def test_iter_session_files_yields_both_sessions_and_archived(tmp_path, monkeypatch):
    live = tmp_path / "sessions" / "2026" / "01" / "01" / "rollout-x.jsonl"
    archived = tmp_path / "archived_sessions" / "rollout-y.jsonl"
    _write_rollout(live, [{"type": "session_meta"}])
    _write_rollout(archived, [{"type": "session_meta"}])
    monkeypatch.setattr(codex, "CODEX_ROOT", tmp_path)
    found = sorted(p.name for p in codex.iter_session_files())
    assert found == ["rollout-x.jsonl", "rollout-y.jsonl"]


def test_iter_session_files_missing_subdir_yields_nothing_for_it(tmp_path, monkeypatch):
    # only `sessions/` present -> archived_sessions absence must not raise
    live = tmp_path / "sessions" / "2026" / "01" / "01" / "rollout-x.jsonl"
    _write_rollout(live, [{"type": "session_meta"}])
    monkeypatch.setattr(codex, "CODEX_ROOT", tmp_path)
    found = [p.name for p in codex.iter_session_files()]
    assert found == ["rollout-x.jsonl"]


# --------------------------------------------------------------------------- #
# Ingest integration (real Postgres, tmp Codex root) — exercises clync.ingest_codex
# --------------------------------------------------------------------------- #
def _seed_codex(tmp_path, monkeypatch, session_id="cdx-1", originator="codex_cli_rs",
                name="rollout-x.jsonl"):
    p = tmp_path / "sessions" / "2026" / "01" / "01" / name
    _write_rollout(p, _rollout_lines(session_id=session_id, originator=originator))
    monkeypatch.setattr(codex, "CODEX_ROOT", tmp_path)
    return p


def test_ingest_codex_end_to_end(pg_test_db, tmp_path, monkeypatch):
    import clync
    _seed_codex(tmp_path, monkeypatch, session_id="cdx-1", name="rollout-a.jsonl")
    # a second rollout from an OUT-OF-SCOPE originator (real desktop-GUI value)
    # must be examined + watermarked but counted on the out_of_scope axis, never
    # as an ingested session.
    _write_rollout(tmp_path / "sessions" / "2026" / "01" / "01" / "rollout-b.jsonl",
                   _rollout_lines(session_id="cdx-2", originator="Codex Desktop"))

    stats = clync.ingest_codex(full=True)
    assert stats["sessions_ingested"] == 1        # out-of-scope originator skipped
    assert stats["files_parsed"] == 2             # both files examined + watermarked
    assert stats["out_of_scope"] == 1             # the desktop rollout, counted distinctly
    assert stats["sessions_removed"] == 0

    con = clync.connect()
    try:
        rows = con.execute(
            "SELECT unit_id, kind, source, repo, model, cc_version, git_branch, "
            "msg_count FROM units WHERE source='codex_cli'").fetchall()
        assert len(rows) == 1
        r = rows[0]
        assert r["unit_id"] == "cdx-1" and r["kind"] == "codex_session"
        assert r["repo"] == "clync" and r["model"] == "gpt-5.1-codex"
        assert r["cc_version"] == "0.66.0" and r["git_branch"] is None
        stored = con.execute("SELECT COUNT(*) n FROM messages WHERE unit_id='cdx-1'"
                             ).fetchone()["n"]
        assert stored > 0 and r["msg_count"] == stored   # msg_count == retrievable rows
    finally:
        con.close()

    stats2 = clync.ingest_codex(full=False)       # nothing changed -> all skipped
    assert stats2["sessions_ingested"] == 0 and stats2["unchanged"] == 2
    # the out-of-scope rollout was watermarked, so it is not re-parsed (proving
    # a mass of desktop files does not re-scan every sync)
    assert stats2["out_of_scope"] == 0


def test_ingest_codex_reconciles_deleted_rollout(pg_test_db, tmp_path, monkeypatch):
    import clync
    p = _seed_codex(tmp_path, monkeypatch, session_id="cdx-1")
    clync.ingest_codex(full=True)

    p.unlink()
    stats = clync.ingest_codex()
    assert stats["sessions_removed"] == 1

    con = clync.connect()
    try:
        for q in ("SELECT COUNT(*) n FROM units WHERE source='codex_cli'",
                  "SELECT COUNT(*) n FROM cc_sync_state WHERE source='codex_cli'"):
            assert con.execute(q).fetchone()["n"] == 0
    finally:
        con.close()


def test_ingest_codex_missing_root_raises_and_store_untouched(pg_test_db, tmp_path, monkeypatch):
    """An unavailable Codex root (moved/unmounted) must FAIL the ingest, not read
    as 'all rollouts deleted' and purge the entire codex corpus."""
    import clync
    _seed_codex(tmp_path, monkeypatch, session_id="cdx-1")
    clync.ingest_codex(full=True)

    monkeypatch.setattr(codex, "CODEX_ROOT", tmp_path / "gone")
    with pytest.raises(FileNotFoundError):
        clync.ingest_codex()

    con = clync.connect()
    try:
        assert con.execute("SELECT COUNT(*) n FROM units WHERE source='codex_cli'"
                           ).fetchone()["n"] == 1
        assert con.execute("SELECT COUNT(*) n FROM cc_sync_state WHERE source='codex_cli'"
                           ).fetchone()["n"] == 1
    finally:
        con.close()


def test_local_sources_reconcile_independently(pg_test_db, tmp_path, monkeypatch):
    """The `source` column partitions the shared cc_sync_state watermark: deleting
    a Claude Code transcript must NOT purge Codex units (and vice versa). Without
    the partition, one source's on-disk file set would mark the OTHER source's
    watermark rows 'deleted' and cascade-purge its units."""
    import cc
    import clync
    from tests.test_cc import _cli_events, _write_session

    cc_root = tmp_path / "cc"
    cc_file = cc_root / "proj" / "sess-1.jsonl"
    _write_session(cc_file, _cli_events())
    monkeypatch.setattr(cc, "CC_ROOT", cc_root)
    codex_root = tmp_path / "codex"
    _write_rollout(codex_root / "sessions" / "2026" / "01" / "01" / "rollout-a.jsonl",
                   _rollout_lines(session_id="cdx-1"))
    monkeypatch.setattr(codex, "CODEX_ROOT", codex_root)

    clync.ingest_cc(full=True)
    clync.ingest_codex(full=True)

    def _counts():
        con = clync.connect()
        try:
            return (con.execute("SELECT COUNT(*) n FROM units WHERE source='claude_code'"
                                ).fetchone()["n"],
                    con.execute("SELECT COUNT(*) n FROM units WHERE source='codex_cli'"
                                ).fetchone()["n"])
        finally:
            con.close()

    assert _counts() == (1, 1)

    # Delete the CC transcript and re-ingest BOTH: the CC unit is reconciled away,
    # the Codex unit is untouched — the two reconciliations never cross sources.
    cc_file.unlink()
    assert clync.ingest_cc()["sessions_removed"] == 1
    assert clync.ingest_codex()["sessions_removed"] == 0
    assert _counts() == (0, 1)


def test_cc_sync_state_source_migration_backfills_and_drops_default(pg_test_db):
    """The cc_sync_state.source migration (ADD COLUMN backfill + DROP DEFAULT) is
    only exercised for real on a production upgrade over a populated pre-source
    table; every other test provisions the current shape, so this is the one path
    that drives it. Simulate the old shape: drop the column, seed a legacy row,
    re-run the raw schema. Two invariants: (1) the legacy row backfills to
    'claude_code' (the pre-Codex era was all Claude Code); (2) the DEFAULT is gone
    afterward, so a later INSERT that omits `source` fails LOUD (NOT NULL) instead
    of silently mislabeling itself and slipping the source-partitioned reconcile."""
    import psycopg

    import clync
    con = clync.connect()
    try:
        # recreate the pre-source (old-shape) table with one legacy watermark row
        con.execute("ALTER TABLE cc_sync_state DROP COLUMN source")
        con.execute(
            "INSERT INTO cc_sync_state (file_path, mtime, size, session_id, synced_at) "
            "VALUES ('/legacy/sess.jsonl', 1.0, 10, 'legacy-1', now())")
        con.commit()
        # re-run the raw schema — the ADD COLUMN + backfill + DROP DEFAULT fires here
        for stmt in clync.sql_statements(clync.RAW_SCHEMA):
            con.execute(stmt)
        con.commit()
        # (1) legacy row stamped claude_code by the backfill DEFAULT
        assert con.execute(
            "SELECT source FROM cc_sync_state WHERE session_id='legacy-1'"
        ).fetchone()["source"] == "claude_code"
        # (2) DEFAULT dropped -> an omitted-source INSERT fails loud, never silently
        # mislabels itself claude_code
        with pytest.raises(psycopg.errors.NotNullViolation):
            con.execute(
                "INSERT INTO cc_sync_state (file_path, mtime, size, session_id, synced_at) "
                "VALUES ('/new/sess.jsonl', 2.0, 20, 'new-1', now())")
        con.rollback()
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Injection filtering — the full harness/tool-injected wrapper set (regression:
# `<recommended_plugins>` and ~10 other real-corpus wrappers leaked as titles)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("injected", [
    "<recommended_plugins>\nHere is a list of plugins...\n</recommended_plugins>",
    "<task-notification>\n<task-id>x</task-id>\n</task-notification>",
    "<command-message>recall</command-message>",
    "<command-name>/recap</command-name>",
    "<user_shell_command>\n<command>ls</command>\n</user_shell_command>",
    "<local-command-stdout>Login successful</local-command-stdout>",
    "<turn_aborted>\nThe user interrupted the previous turn.\n</turn_aborted>",
    '<codex_internal_context source="goal">\nContinue...\n</codex_internal_context>',
    '<in-app-browser-context source="ambient-ui-state">\n...\n</in-app-browser-context>',
    "<codex_delegation>\n<source_host_id>x</source_host_id>\n</codex_delegation>",
    '<image name=[Image #1] path="/tmp/x.png"></image>',
    "# Context from my IDE setup:\n## Active file: Untitled-1",
    "<environment_context>\n<cwd>/x</cwd>\n</environment_context>",
    "# AGENTS.md instructions for /Users/j/repo\n<INSTRUCTIONS>...",
])
def test_is_injection_covers_real_corpus_wrappers(injected):
    assert codex._is_injection(injected) is True


def test_is_injection_keeps_real_prompts_that_merely_contain_angle_brackets():
    # A genuine prompt whose leading tag is NOT a known injection wrapper is kept.
    assert codex._is_injection("fix the <div> rendering bug in the header") is False
    assert codex._is_injection("<div>paste</div> — why does this not center?") is False
    assert codex._is_injection("how do I design X") is False


def test_title_skips_injection_message_and_picks_real_prompt(tmp_path):
    # First user turn is a fused injection block (recommended_plugins + env);
    # the real prompt is the next user message. Title must be the real prompt,
    # and the injection text must not appear in any stored message.
    lines = [
        _line(timestamp="2026-01-01T00:00:00Z", type="session_meta", payload={
            "id": "cdx-inj", "cwd": "/Users/j/code/clync",
            "originator": "codex-tui", "cli_version": "0.66.0"}),
        _line(timestamp="2026-01-01T00:00:01Z", type="response_item", payload={
            "type": "message", "role": "user", "content": [{"type": "input_text",
            "text": "<recommended_plugins>\nplugins...\n</recommended_plugins>\n"
                    "# AGENTS.md instructions for /x\n<environment_context><cwd>/x</cwd>"
                    "</environment_context>"}]}),
        _line(timestamp="2026-01-01T00:00:02Z", type="response_item", payload={
            "type": "message", "role": "user", "content": [{"type": "input_text",
            "text": "actually fix the retry backoff in the client"}]}),
        _line(timestamp="2026-01-01T00:00:03Z", type="response_item", payload={
            "type": "message", "role": "assistant", "content": [{"type": "output_text",
            "text": "done"}]}),
    ]
    p = tmp_path / "sessions" / "2026" / "01" / "01" / "rollout-inj.jsonl"
    _write_rollout(p, lines)
    s = codex.parse_session(p)
    assert s is not None
    assert s.title == "actually fix the retry backoff in the client"
    assert not any("recommended_plugins" in m.text for m in s.messages)
    assert not any("environment_context" in m.text for m in s.messages)
