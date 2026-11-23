import json
import shutil
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pytest

import live_external
import opencode_source
import cc
from live_types import TOOL_SENDER, REASONING_KIND


def _grok(tmp_path, monkeypatch, sid="root", parent=None, directory=None):
    monkeypatch.setenv("CLYNC_GROK_ROOT", str(tmp_path))
    directory = directory or tmp_path / "sessions" / "encoded-cwd" / sid
    directory.mkdir(parents=True)
    summary = {"info": {"id": sid, "cwd": "/code/garden"},
               "generated_title": "Garden", "session_summary": "Notes",
               "created_at": "2026-11-01T00:00:00Z", "updated_at": "2026-11-02T00:00:00Z"}
    if parent:
        summary["parent_session_id"] = parent
    (directory / "summary.json").write_text(json.dumps(summary))
    return directory / "updates.jsonl"


def _update(kind, text=None, event="event1", **fields):
    update = {"sessionUpdate": kind, **fields}
    if text is not None:
        update["content"] = {"type": "text", "text": text}
    return {"timestamp": 1786871107, "method": "session/update",
            "params": {"sessionId": "root", "update": update, "_meta": {"eventId": event}}}


def test_grok_chunk_words_anchors_and_authorship(tmp_path, monkeypatch):
    path = _grok(tmp_path, monkeypatch)
    rows = [_update("user_message_chunk", "Find helio", event="user1"),
            _update("user_message_chunk", "trope"),
            _update("agent_thought_chunk", "private thought"),
            _update("tool_call", rawInput={"command": "machine input"}),
            _update("tool_call_update", content=[{"type": "content", "content": {
                "type": "text", "text": "machine output"}}]),
            _update("agent_message_chunk", "Out", event="assistant1"),
            _update("agent_message_chunk", "doors"), _update("turn_completed"),
            _update("user_message_chunk", "Next", event="user2")]
    rows[0]["method"] = "_x.ai/session/update"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path.with_name("chat_history.json").write_text(json.dumps({"text": "duplicate echo"}))
    _grok(tmp_path, monkeypatch, "child", "root")
    sessions = live_external.grok_sessions()
    assert [s.session_id for s in sessions] == ["root"]
    assert {s.session_id: s.parent_id for s in live_external.grok_sessions(
        include_subagents=True)} == {"root": None, "child": "root"}
    records = live_external.grok_records(sessions[0])
    assert [(r.sender, r.text) for r in records] == [
        ("user", "Find heliotrope"), ("assistant", "Outdoors"), ("user", "Next")]
    assert records[0].anchor == live_external.grok_records(sessions[0])[0].anchor
    assert "user1" in records[0].anchor
    assert datetime.fromisoformat(records[0].created_at).timestamp() == rows[0]["timestamp"]
    tools = live_external.grok_records(sessions[0], include_tools=True)
    assert [r.text for r in tools if r.sender == cc.HUMAN_SENDER] == ["Find heliotrope", "Next"]
    assert "machine output" in "\n".join(r.text for r in tools if r.sender == TOOL_SENDER)


@pytest.mark.parametrize("failure", ["missing_root", "missing_updates", "interior", "identity"])
def test_grok_incomplete_sources_fail(tmp_path, monkeypatch, failure):
    path = _grok(tmp_path, monkeypatch)
    if failure == "missing_root":
        monkeypatch.setenv("CLYNC_GROK_ROOT", str(tmp_path / "absent"))
        with pytest.raises(FileNotFoundError):
            live_external.grok_sessions()
        return
    session = live_external.grok_sessions()[0]
    row = _update("user_message_chunk", "hello")
    if failure == "identity":
        row["params"]["sessionId"] = "wrong-session"
    if failure != "missing_updates":
        text = json.dumps(row) + "\n"
        path.write_text(text + ('broken\n' + text if failure == "interior" else ""))
    with pytest.raises((FileNotFoundError, ValueError)):
        live_external.grok_records(session)


@pytest.mark.parametrize("valid_tail", [False, True])
def test_grok_pending_tail_retains_prefix_and_reports_partial_coverage(tmp_path, monkeypatch, valid_tail):
    path = _grok(tmp_path, monkeypatch)
    prefix = (json.dumps(_update("user_message_chunk", "Find heliotrope")) + "\n").encode()
    complete = json.dumps(_update("agent_message_chunk", "Outdoors")).encode()
    path.write_bytes(prefix + (complete if valid_tail else b'{"params":'))
    session, = live_external.grok_sessions()
    assert [r.text for r in live_external.grok_records(session)] == ["Find heliotrope"]
    warnings = session.metadata["warnings"]
    assert warnings and all(isinstance(warning, str) for warning in warnings)
    assert str(path) in warnings[0] and str(len(prefix)) in warnings[0]
    path.write_bytes(prefix + complete + b"\n")
    assert [r.text for r in live_external.grok_records(session)] == ["Find heliotrope", "Outdoors"]
    assert not session.metadata.get("warnings")


@pytest.mark.parametrize("counters", [(0, 0), (0, 1), (1, 0), (None, 0), (0, None), (False, 0)])
def test_grok_only_proven_empty_sessions_may_lack_logs(tmp_path, monkeypatch, counters):
    path = _grok(tmp_path, monkeypatch)
    summary_path = path.with_name("summary.json")
    summary = json.loads(summary_path.read_text())
    summary.update({key: value for key, value in zip(("num_messages", "num_chat_messages"), counters)
                    if value is not None})
    summary_path.write_text(json.dumps(summary))
    session, = live_external.grok_sessions()
    if all(type(value) is int and value == 0 for value in counters):
        assert session.metadata["num_messages"] == session.metadata["num_chat_messages"] == 0
        assert live_external.grok_records(session) == []
    else:
        with pytest.raises(FileNotFoundError):
            live_external.grok_records(session)


def test_grok_nested_children_retain_path_parent_and_readable_records(tmp_path, monkeypatch):
    root = _grok(tmp_path, monkeypatch)
    child = _grok(tmp_path, monkeypatch, "child", directory=root.parent / "subagents" / "child")
    grandchild = _grok(tmp_path, monkeypatch, "grandchild",
                       directory=child.parent / "subagents" / "grandchild")
    for sid, path in (("root", root), ("child", child), ("grandchild", grandchild)):
        row = _update("agent_message_chunk", "Native " + sid)
        row["params"]["sessionId"] = sid
        path.write_text(json.dumps(row) + "\n")
    assert [s.session_id for s in live_external.grok_sessions()] == ["root"]
    sessions = live_external.grok_sessions(include_subagents=True)
    assert {s.session_id: s.parent_id for s in sessions} == {
        "root": None, "child": "root", "grandchild": "child"}
    assert {s.session_id: [r.text for r in live_external.grok_records(s)] for s in sessions} == {
        sid: ["Native " + sid] for sid in ("root", "child", "grandchild")}
    summary_path = child.with_name("summary.json")
    summary = json.loads(summary_path.read_text())
    summary["parent_session_id"] = "unrelated"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        live_external.grok_sessions(include_subagents=True)


@pytest.mark.parametrize("link", ["project", "session", "summary", "updates", "subagents", "updates_after_discovery"])
def test_grok_symlinks_cannot_escape_source_root(tmp_path, monkeypatch, link):
    root = tmp_path / "configured"
    path = _grok(root, monkeypatch)
    path.write_text(json.dumps(_update("user_message_chunk", "Outside root")) + "\n")
    sessions = live_external.grok_sessions() if link == "updates_after_discovery" else None
    targets = {"project": path.parent.parent, "session": path.parent,
               "summary": path.with_name("summary.json"), "updates": path,
               "subagents": path.parent / "subagents", "updates_after_discovery": path}
    if link == live_external.CHILD_DIR:
        _grok(root, monkeypatch, "child", directory=targets[link] / "child")
    target = targets[link]
    external = tmp_path / "external"
    shutil.move(str(target), str(external))
    target.symlink_to(external, target_is_directory=external.is_dir())
    with pytest.raises(ValueError):
        for session in sessions or live_external.grok_sessions(include_subagents=True):
            live_external.grok_records(session)


def test_grok_summary_cannot_impersonate_another_directory(tmp_path, monkeypatch):
    path = _grok(tmp_path, monkeypatch).with_name("summary.json")
    summary = json.loads(path.read_text())
    summary["info"]["id"] = "another-session"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        live_external.grok_sessions()


def _opencode(monkeypatch, failure=None):
    info = {"id": "ses_root", "projectID": "project", "title": "Garden",
            "location": {"directory": "/code/garden/worktree"},
            "time": {"created": 1800000000000, "updated": 1800000001000}}
    child = {**info, "id": "ses_child", "parentID": "ses_root"}
    messages = [{"id": "msg_user", "type": "user", "time": info["time"], "text": "heliotrope"},
                {"id": "msg_machine", "type": "synthetic", "text": "machine instruction"},
                {"id": "msg_assistant", "type": "assistant", "time": info["time"], "content": [
                    {"type": "reasoning", "text": "private thought"},
                    {"type": "tool", "name": "read", "state": {"status": "completed",
                     "input": {"path": "/garden"}, "content": [{"type": "text", "text": "machine output"}]}},
                    {"type": "tool", "name": "question", "state": {"status": "completed",
                     "input": {"questions": []}, "metadata": {"answers": [["balcony"]]}, "content": []}},
                    {"type": "text", "text": "Keep it outdoors"}]}]
    def get(self, path):
        url = urlparse(path)
        query = parse_qs(url.query)
        cursor = query.get("cursor", [None])[0]
        if cursor:
            assert "order" not in query
        if url.path == live_external.INFO_PATH:
            return {"version": "1.0.0" if failure == "wrong-release" else "2.2.5"}
        if url.path == "/api/project":
            return [{"id": "project", "canonical": "/code/garden"}]
        if url.path == "/api/session":
            return {"data": [child if cursor else info], "cursor": {"next": None if cursor else "sessions2"}}
        if cursor and failure == "transport":
            raise RuntimeError("connection lost")
        data = messages[1:] if cursor else messages[:1]
        if failure == "identity":
            data = [{**messages[0], "id": "invalid/id"}]
        if failure == "wrong-kind":
            data = [{**messages[0], "type": "unrecognized"}]
        if cursor and failure == "duplicate":
            data = messages[:1]
        if failure == "schema":
            return {"data": data}
        return {"data": data, "cursor": {"next": "messages2" if not cursor or failure == "loop" else None}}

    monkeypatch.setattr(opencode_source.Client, "__init__", lambda self: None)
    monkeypatch.setattr(opencode_source.Client, "get", get)


def test_opencode_paginated_sources_and_authorship(monkeypatch):
    _opencode(monkeypatch)
    session, = live_external.opencode_sessions()
    assert session.session_id == "ses_root" and session.metadata["repo"] == "garden"
    assert {s.session_id: s.parent_id for s in live_external.opencode_sessions(
        include_subagents=True)} == {"ses_root": None, "ses_child": "ses_root"}
    records = live_external.opencode_records(session)
    assert [(r.sender, r.text) for r in records] == [
        ("user", "heliotrope"), ("user_answer", "balcony"), ("assistant", "Keep it outdoors")]
    tools = live_external.opencode_records(session, include_tools=True)
    assert [(r.sender, r.text) for r in tools if r.sender in ("user", "user_answer")] == [
        ("user", "heliotrope"), ("user_answer", "balcony")]
    assert "machine output" in "\n".join(r.text for r in tools if r.sender == TOOL_SENDER)
    assert "private thought" in "\n".join(r.text for r in tools if r.kind == REASONING_KIND)


@pytest.mark.parametrize("failure", ["wrong-release", "transport", "loop", "duplicate", "identity", "wrong-kind", "schema"])
def test_opencode_incomplete_sources_fail(monkeypatch, failure):
    _opencode(monkeypatch, failure)
    with pytest.raises((ValueError, RuntimeError, KeyError, TypeError)):
        session, = live_external.opencode_sessions()
        live_external.opencode_records(session)
