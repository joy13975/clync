"""Claude Code ingest — pure parsing/cleaning/metadata (no DB) + an ingest
integration test against a throwaway Postgres db over a tmp transcript root."""
from __future__ import annotations

import json

import pytest

import clync
import cc
import dream


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #
def test_repo_and_worktree():
    assert cc._repo_and_worktree(
        "/Users/example/code/example-repo/.claude/worktrees/fix-precision/sub/dir"
    ) == ("example-repo", "fix-precision")
    assert cc._repo_and_worktree("/Users/example/code/clync") == ("clync", None)
    assert cc._repo_and_worktree(None) == (None, None)


def test_tool_use_line_primary_arg():
    assert cc._tool_use_line({"name": "Bash", "input": {"command": "ls -la"}}) == "[Bash] ls -la"
    assert cc._tool_use_line({"name": "Read", "input": {"file_path": "/a/b.py"}}) == "[Read] /a/b.py"
    # unknown tool -> compact key listing, never a full dump
    assert cc._tool_use_line({"name": "Weird", "input": {"z": 1, "a": 2}}) == "[Weird] a, z"


def test_tool_result_head_truncation_and_agent_whole():
    big = "x" * 5000
    head = cc._render_tool_result({"content": big}, "Bash", {})
    assert head.startswith("[result: 5000 bytes, head]") and "…" in head
    assert len(head) < 2000                                   # truncated
    # Agent AND Task results are subagent syntheses — both kept whole.
    for tool in ("Agent", "Task"):
        whole = cc._render_tool_result({"content": big}, tool, {})
        assert whole.startswith("[result: 5000 bytes]") and whole.endswith("x")
    err = cc._render_tool_result({"content": "boom", "is_error": True}, "Bash", {})
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
    base = {"cwd": "/Users/example/code/clync", "gitBranch": "main", "version": "2.1.9",
            "entrypoint": "cli", "sessionId": "sess-1"}
    return [
        _event(type="ai-title", aiTitle="Investigate design question", sessionId="sess-1"),
        _event(type="user", uuid="u1", timestamp="2026-01-01T00:00:00Z",
               promptSource="typed", origin={"kind": "human"},
               message={"role": "user", "content": "how do I design X"}, **base),
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


def _channel_kind(uuid: str) -> str:
    """The `origin.kind` the fixture gave one event — the expected sender for a
    machine channel, read back from the fixture instead of retyped."""
    ev = next(e for e in _channel_events() if e.get("uuid") == uuid)
    return ev["origin"]["kind"]


def _channel_events():
    """One session carrying every channel the harness files under `type: "user"`.
    Shapes mirror the real producer: only the human's own turn has
    `origin.kind == "human"`; a subagent answer and a peer message carry their own
    kind; a compaction summary and a slash-command echo carry no `origin` at all."""
    base = {"cwd": "/Users/example/code/clync", "gitBranch": "main", "version": "2.1.9",
            "entrypoint": "cli", "sessionId": "sess-c"}
    return [
        _event(type="user", uuid="c1", timestamp="2026-02-01T00:00:00Z",
               promptSource="typed", origin={"kind": cc.HUMAN_ORIGIN_KIND},
               message={"role": "user", "content": "keep the retry, drop the clamp"},
               **base),
        _event(type="assistant", uuid="c2", timestamp="2026-02-01T00:00:01Z",
               message={"role": "assistant", "content": [
                   {"type": "text", "text": "dropping the clamp"}]}, **base),
        _event(type="user", uuid="c3", timestamp="2026-02-01T00:00:02Z",
               origin={"kind": "task-notification"},
               message={"role": "user", "content":
                        "Agent finished: I recommend clamping the value instead."},
               **base),
        _event(type="user", uuid="c4", timestamp="2026-02-01T00:00:03Z",
               origin={"kind": "peer"},
               message={"role": "user", "content":
                        "Peer session: the user wants the clamp kept."}, **base),
        _event(type="user", uuid="c5", timestamp="2026-02-01T00:00:04Z",
               isCompactSummary=True,
               message={"role": "user", "content":
                        "Summary: the user asked for a clamp."}, **base),
        _event(type="user", uuid="c6", timestamp="2026-02-01T00:00:05Z",
               message={"role": "user", "content":
                        "<command-name>/clear</command-name>"}, **base),
        _event(type="assistant", uuid="c7", timestamp="2026-02-01T00:00:06Z",
               message={"role": "assistant", "content": [
                   {"type": "tool_use", "id": "tc", "name": cc.ASK_TOOL,
                    "input": {"questions": [{"question": "Clamp or fix?",
                                             "options": [{"label": "Fix"}]}]}}]},
               **base),
        _event(type="user", uuid="c8", timestamp="2026-02-01T00:00:07Z",
               toolUseResult={"answers": {"Clamp or fix?": "fix it"}},
               message={"role": "user", "content": [
                   {"type": "tool_result", "tool_use_id": "tc",
                    "content": 'The user answered: "Clamp or fix?"="fix it"'}]},
               **base),
    ]


def test_only_the_human_channel_is_the_user_and_reaches_the_index(tmp_path):
    """Whose words a retrieved message is decides whether it can be quoted as the
    user's position, and the harness answers that in `origin.kind`. A subagent
    answer, a peer message, a compaction summary and a command echo must all fall
    OUTSIDE the senders every reader treats as the user (`dream.USER_SENDERS`) and
    must contribute no search chunk. Labelling them the user is what let a
    subagent's prose come back from search as something the user said. An answered
    question is the one exception the origin rule cannot see: it arrives as a
    tool_result with no origin, and it IS the user's own words."""
    p = tmp_path / "proj" / "sess-c.jsonl"
    _write_session(p, _channel_events())
    s = cc.parse_session(p)
    by_id = {m.msg_id: m for m in s.messages}
    # all kept for reading
    assert set(by_id) == {"c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"}

    # The user's records: the turn the harness marked human, and their answer.
    assert {i for i, m in by_id.items() if m.sender in dream.USER_SENDERS} == {"c1", "c8"}
    # A named machine channel keeps its own kind; an unmarked one is not guessed at.
    assert by_id["c3"].sender == _channel_kind("c3")
    assert by_id["c4"].sender == _channel_kind("c4")
    assert by_id["c5"].sender == cc.NO_ORIGIN_SENDER
    assert by_id["c6"].sender == cc.NO_ORIGIN_SENDER

    # search.py indexes a message only where COALESCE(embed_text, text) is non-empty.
    searchable = {i for i, m in by_id.items() if m.embed_text != ""}
    assert searchable == {"c1", "c2", "c7", "c8"}
    assert {by_id[i].sender for i in searchable} == set(cc.EMBEDDED_SENDERS)


_FORK_QUESTION = "Keep the clamp or drop it?"
_ANSWER_TAIL = "and the producer must never emit the bad value in the first place"
_ANSWER_WORDS = ("drop it — a clamp is compensation, " * 40) + _ANSWER_TAIL
_ANSWER_NOTE = "and rip out the retry while you are in there"


def _fork_events():
    """One `AskUserQuestion` round trip, twice: answered, then refused. Shapes
    mirror the real producer — the reply arrives as a `tool_result` on a user-slot
    record carrying NO `origin`, its body holds the question and the user's own
    words, and a refusal arrives the same way but with `is_error` and a body of
    harness boilerplate instead of anything the user wrote."""
    base = {"cwd": "/Users/example/code/clync", "gitBranch": "main", "version": "2.1.9",
            "entrypoint": "cli", "sessionId": "sess-f"}
    fork = {"questions": [{"question": _FORK_QUESTION,
                           "header": "Clamp",
                           "options": [{"label": "Drop the clamp",
                                        "description": "fix the producer instead"},
                                       {"label": "Keep the clamp",
                                        "description": "cheaper right now"}]}]}
    answer = f'The user answered: "{_FORK_QUESTION}"="{_ANSWER_WORDS}"'
    return [
        _event(type="assistant", uuid="f1", timestamp="2026-03-01T00:00:00Z",
               message={"role": "assistant", "content": [
                   {"type": "tool_use", "id": "t1", "name": "AskUserQuestion",
                    "input": fork}]}, **base),
        _event(type="user", uuid="f2", timestamp="2026-03-01T00:00:01Z",
               toolUseResult={"answers": {_FORK_QUESTION: _ANSWER_WORDS},
                              "annotations": {_FORK_QUESTION: {"notes": _ANSWER_NOTE}}},
               message={"role": "user", "content": [
                   {"type": "tool_result", "tool_use_id": "t1", "content": answer}]},
               **base),
        _event(type="assistant", uuid="f3", timestamp="2026-03-01T00:00:02Z",
               message={"role": "assistant", "content": [
                   {"type": "tool_use", "id": "t2", "name": "AskUserQuestion",
                    "input": fork}]}, **base),
        _event(type="user", uuid="f4", timestamp="2026-03-01T00:00:03Z",
               message={"role": "user", "content": [
                   {"type": "tool_result", "tool_use_id": "t2", "is_error": True,
                    "content": "The user doesn't want to proceed with this tool use."}]},
               **base),
    ]


def test_answered_question_is_the_users_own_turn_whole_and_indexed(tmp_path):
    """An answered `AskUserQuestion` is the densest decision record the harness
    produces: a fork, the option the user took, and their reasoning in their own
    words. It arrives as a tool_result with no `origin`, so the authorship rule
    alone filed it as a machine channel — unindexed, and head-truncated at
    `_TOOL_RESULT_HEAD`, which cut real answers mid-sentence. A refusal is NOT
    that: its body is harness boilerplate, so it stays a machine channel."""
    p = tmp_path / "proj" / "sess-f.jsonl"
    _write_session(p, _fork_events())
    by_id = {m.msg_id: m for m in cc.parse_session(p).messages}

    answered = by_id["f2"]
    assert answered.sender in dream.USER_SENDERS
    assert _ANSWER_TAIL in answered.text                  # whole, not head-truncated
    assert len(answered.text) > cc._TOOL_RESULT_HEAD

    # WHAT gets embedded under the user's name is only what the user wrote. The
    # rendered body also carries the assistant's question and the harness's own
    # framing; embedding those here is how the question comes back from a
    # sender="user" search as something the user said.
    assert answered.embed_text and _ANSWER_TAIL in answered.embed_text
    assert _ANSWER_NOTE in answered.embed_text            # a typed note is theirs too
    assert _FORK_QUESTION not in answered.embed_text
    assert _FORK_QUESTION in answered.text                # still readable in context

    refused = by_id["f4"]
    assert refused.sender not in dream.USER_SENDERS
    assert refused.embed_text == ""


def test_the_fork_the_answer_resolves_is_readable(tmp_path):
    """The question and the options live on the assistant's `tool_use`, and the
    generic one-liner rendered them as the bare input key — so the fork a retrieved
    answer resolves was unreadable, and the rejected alternatives were lost."""
    p = tmp_path / "proj" / "sess-f.jsonl"
    _write_session(p, _fork_events())
    asked = {m.msg_id: m for m in cc.parse_session(p).messages}["f1"]

    fork = _fork_events()[0]["message"]["content"][0]["input"]["questions"][0]
    assert fork["question"] in asked.text
    for option in fork["options"]:
        assert option["label"] in asked.text


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
        return {"cwd": "/Users/example/code/clync", "gitBranch": "main",
                "version": "2.1.9", "entrypoint": "cli", "sessionId": sid}

    def _turns(sid):
        return [
            _event(type="user", uuid="ra1", timestamp="2026-01-02T00:00:00Z",
                   promptSource="typed", origin={"kind": "human"},
                   message={"role": "user", "content": "start the work"}, **_base(sid)),
            _event(type="assistant", uuid="ra2", timestamp="2026-01-02T00:00:01Z",
                   message={"role": "assistant", "content": [
                       {"type": "text", "text": "the shared prefix answer"}]}, **_base(sid)),
        ]

    a = _turns("sess-A")
    b = _turns("sess-B") + [
        _event(type="user", uuid="rb1", timestamp="2026-01-02T01:00:00Z",
               promptSource="typed", origin={"kind": "human"},
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


def test_get_conversation_reads_a_bounded_window(pg_test_db, tmp_path, monkeypatch):
    """A retrieval hit names a position, and the only way to read around it was to
    fetch the whole unit, which a caller cannot afford per hit. Bounded by index,
    the read returns the window, labels each turn with its index so a further window
    can be asked for, and states what it left out rather than clipping in silence."""
    a, b = _resume_pair()
    _write_session(tmp_path / "proj" / "sess-B.jsonl", b)
    monkeypatch.setattr(cc, "CC_ROOT", tmp_path)
    clync.ingest_cc(full=True)

    import mcp_server
    lo, hi = 2, 3
    window = mcp_server.get_conversation("sess-B", from_idx=lo, to_idx=hi)
    assert "continue after resume" in window            # idx 2
    assert "the post-resume answer" in window           # idx 3
    assert "the shared prefix answer" not in window     # idx 1, outside the window
    # A bounded read states what it left out — the range against the unit's total,
    # both derived here rather than retyped, so a silent clip cannot pass.
    con = clync.connect()
    try:
        total = con.execute("SELECT msg_count FROM units WHERE unit_id='sess-B'"
                            ).fetchone()["msg_count"]
    finally:
        con.close()
    assert f"{lo}-{hi} of {total}" in window


def test_get_conversation_can_read_one_channel_of_a_window(pg_test_db, tmp_path,
                                                           monkeypatch):
    """An index range alone does not bound an episode read: an agent files many tool
    rows between the user's own turns. Restricting the read to one role is what makes
    the window readable, and the header names the role so a filtered read is not taken
    for the whole conversation."""
    _write_session(tmp_path / "proj" / "sess-c.jsonl", _channel_events())
    monkeypatch.setattr(cc, "CC_ROOT", tmp_path)
    clync.ingest_cc(full=True)

    import mcp_server
    role = next(r for r in cc.SENDER_ROLES if cc.HUMAN_SENDER in cc.SENDER_ROLES[r])
    whole = mcp_server.get_conversation("sess-c")
    his = mcp_server.get_conversation("sess-c", sender=role)

    assert "keep the retry, drop the clamp" in his          # c1, his own turn
    assert "fix it" in his                                  # c8, his answer
    assert "dropping the clamp" in whole                    # the agent's turn...
    assert "dropping the clamp" not in his                  # ...is not his
    assert "I recommend clamping" not in his                 # a subagent answer
    assert "Peer session" not in his                         # another session
    assert "Summary: the user asked" not in his              # a compaction summary
    # The header says which role was rendered, and how much of the unit that left out.
    con = clync.connect()
    try:
        total = con.execute("SELECT msg_count FROM units WHERE unit_id='sess-c'"
                            ).fetchone()["msg_count"]
    finally:
        con.close()
    assert role in his
    assert f"of {total}" in his
    assert len(his) < len(whole)
    # Under a role filter the rendered COUNT leads the header: a few turns far apart
    # must not read as a contiguous near-complete span of everything between their ends.
    con2 = clync.connect()
    try:
        n_his = con2.execute(
            "SELECT COUNT(*) c FROM messages WHERE unit_id='sess-c' AND sender = ANY(%s)",
            (cc.resolve_sender_role(role),)).fetchone()["c"]
    finally:
        con2.close()
    assert f"{n_his} {role}" in his
    assert n_his < total


def test_get_conversation_rejects_an_unregistered_role(pg_test_db, tmp_path, monkeypatch):
    """A read asks for a role by the same name a search does, and a STORED sender value
    is not one of those names — which is the confusion the registry exists to catch,
    since `claude_ai` spells the user's turn differently from every other source."""
    _write_session(tmp_path / "proj" / "sess-c.jsonl", _channel_events())
    monkeypatch.setattr(cc, "CC_ROOT", tmp_path)
    clync.ingest_cc(full=True)

    import mcp_server
    out = mcp_server.get_conversation("sess-c", sender=cc.CHAT_HUMAN_SENDER)
    assert cc.CHAT_HUMAN_SENDER not in cc.SENDER_ROLES     # the premise of the test
    for valid in cc.SENDER_ROLES:
        assert valid in out
    assert "keep the retry, drop the clamp" not in out     # nothing was rendered


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
