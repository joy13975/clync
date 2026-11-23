import copy
import json
import types

import pytest

import clync
import mcp_server
import opencode_source
from cc import ANSWER_SENDER


def _session(sid="ses_parent", **info):
    return {"info": {"id": sid, "projectID": "project", "title": "Heliotrope decision",
                     "location": {"directory": "/code/garden/src"},
                     "model": {"id": "botanist", "providerID": "local"},
                     "time": {"created": 1800000000000, "updated": 1800000001000}, **info},
            "messages": [
                {"id": "msg_prompt", "type": "user", "time": {"created": 1800000000000},
                 "text": "Keep the heliotrope outside"},
                {"id": "msg_reply", "type": "assistant",
                 "time": {"created": 1800000001000, "completed": 1800000001001},
                 "content": [{"type": "reasoning", "text": "It needs sunlight"},
                             {"type": "text", "text": "The heliotrope stays outdoors"}]}]}


def _wire_opencode(tmp_path, monkeypatch):
    state = tmp_path / "opencode-state.json"
    log = tmp_path / "requests.jsonl"
    command = tmp_path / "opencode-test"
    command.write_text(f'''#!/usr/bin/env python3
import json, sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs
state, log = Path({str(state)!r}), Path({str(log)!r})
data = json.loads(state.read_text())
path = sys.argv[-1]
with log.open("a") as f: f.write(json.dumps(path) + "\\n")
url = urlparse(path); q = parse_qs(url.query)
if data.get("fail") and ("cursor" in q or data["fail"] == "transport"):
    if data["fail"] == "malformed": print('{{'); sys.exit(0)
    if data["fail"] == "loop":
        print(json.dumps({{"data": [], "cursor": {{"next": q["cursor"][0]}}}})); sys.exit(0)
    print("unavailable", file=sys.stderr); sys.exit(1)
if url.path == "/api/info": result = {{"version": "2.2.5"}}
elif url.path == "/api/project": result = [{{"id": "project", "canonical": "/code/garden"}}]
elif url.path == "/api/session/active": result = {{"data": data.get("active", {{}})}}
elif url.path == "/api/session":
    visible = [s["info"] for s in data["sessions"].values() if s["info"]["id"] not in data.get("hidden", [])]
    offset = int(q.get("cursor", ["0"])[0]); page = visible[offset:offset+1]
    result = {{"data": page, "cursor": {{"next": str(offset+1) if page else None}}}}
    if data.get("fail") == "schema" and page: del page[0]["location"]
else:
    sid = url.path.split("/")[3 if "/experimental/" not in url.path else 4]
    if sid not in data["sessions"]:
        print(json.dumps({{"_tag": "SessionNotFoundError", "sessionID": sid}}))
        print("HTTP 404 Not Found", file=sys.stderr); sys.exit(1)
    session = data["sessions"][sid]
    result = {{"data": session if url.path.endswith("/export") else session["info"]}}
    if url.path.endswith("/export") and data.get("after_export"):
        data["sessions"][sid] = data.pop("after_export")
        state.write_text(json.dumps(data))
print(json.dumps(result))
''')
    command.chmod(0o700)
    config = tmp_path / "opencode.json"
    config.write_text(json.dumps({"command": str(command)}))
    monkeypatch.setenv("CLYNC_OPENCODE_CONFIG", str(config))
    state.write_text(json.dumps({"sessions": {"ses_parent": _session()}}))
    return state, log


def test_cleaning_never_stores_subagents_or_tool_results():
    exported = _session()
    forbidden = "forbidden-machine-payload"
    exported["info"]["metadata"] = {"output": forbidden}
    exported["messages"] += [
        {"id": "msg_injection", "type": "synthetic", "text": forbidden},
        {"id": "msg_system", "type": "system", "text": forbidden},
        {"id": "msg_tools", "type": "assistant", "time": {"created": 1800000002000},
         "content": [
             {"type": "tool", "id": "call_read", "name": "read",
              "state": {"status": "completed", "input": {"path": "/code/notes"},
                        "content": [{"type": "text", "text": forbidden}]}},
             {"type": "tool", "id": "call_child", "name": "subagent",
              "state": {"status": "completed", "input": {"prompt": forbidden},
                        "content": [{"type": "text", "text": forbidden}]}},
             {"type": "tool", "id": "call_answer", "name": "question",
              "state": {"status": "completed", "input": {"questions": [{"question": "Where?"}]},
                        "metadata": {"answers": [["On the balcony"]]},
                        "content": [{"type": "text", "text": forbidden}]}}]}]
    parsed = opencode_source.parse_session(exported, "garden")
    texts = "\n".join(m.text for m in parsed.messages)
    assert forbidden not in texts
    assert exported["messages"][0]["text"] in texts
    for part in exported["messages"][1]["content"]:
        assert part["text"] in texts
    answers = [m for m in parsed.messages if m.sender == ANSWER_SENDER]
    assert [m.embed_text for m in answers] == ["On the balcony"]
    child = copy.deepcopy(exported)
    child["info"]["parentID"] = "ses_other"
    assert opencode_source.parse_session(child, "garden") is None


def test_sync_search_retrieve_and_confirmed_deletion(pg_test_db, mock_embed, tmp_path, monkeypatch):
    state, log = _wire_opencode(tmp_path, monkeypatch)
    data = json.loads(state.read_text())
    child = _session("ses_child", parentID="ses_parent")
    child["messages"][0]["text"] = "child-only-secret"
    data["sessions"]["ses_child"] = child
    data["sessions"]["ses_archive"] = _session("ses_archive", fork={"sessionID": "ses_parent"})
    data["sessions"]["ses_archive"]["info"]["time"]["archived"] = 1800000003000
    data["sessions"]["ses_parent"]["messages"][1]["content"][1]["text"] += "x" * 300000
    forbidden = "tool-result-and-subagent-secret"
    data["sessions"]["ses_parent"]["info"]["metadata"] = {"result": forbidden}
    data["sessions"]["ses_parent"]["messages"] += [
        {"id": "msg_synthetic", "type": "synthetic", "text": forbidden},
        {"id": "msg_tool", "type": "assistant", "time": {"created": 1800000002000},
         "content": [{"type": "tool", "name": "read", "id": "call1",
                      "state": {"status": "completed", "input": {"path": "/notes"},
                                "content": [{"type": "text", "text": forbidden}]}}]}]
    state.write_text(json.dumps(data))
    assert clync.cmd_sync_opencode(types.SimpleNamespace(full=False, no_index=False)) == 0
    hits = pg_test_db.hybrid_search("heliotrope", source="opencode", repo="garden", sender="user")
    uid = "opencode:ses_parent"
    assert uid in {h["unit_id"] for h in hits}
    rendered = mcp_server.get_conversation(uid, sender="user")
    assert data["sessions"]["ses_parent"]["messages"][0]["text"] in rendered
    assert "source=claude_ai" not in rendered
    with clync.connect() as con:
        con.execute("INSERT INTO units(unit_id,kind,source,synced_at) VALUES(%s,%s,%s,%s)",
                    ("unrelated-history", "chat", "claude_ai", clync._now()))
        for table in ("units", "messages", "chunks"):
            rows = con.execute(f"SELECT row_to_json(t) AS row FROM {table} t").fetchall()
            assert forbidden not in json.dumps(rows, default=str)
            assert child["messages"][0]["text"] not in json.dumps(rows, default=str)
    assert not any("ses_child/export" in json.loads(p) for p in log.read_text().splitlines())
    assert opencode_source.ingest()["unchanged"] == 2
    data["hidden"] = ["ses_archive"]
    state.write_text(json.dumps(data))
    assert opencode_source.ingest()["sessions_removed"] == 0
    del data["sessions"]["ses_archive"]
    state.write_text(json.dumps(data))
    assert opencode_source.ingest()["sessions_removed"] == 1
    with clync.connect() as con:
        assert con.execute("SELECT unit_id FROM units WHERE source='opencode'").fetchall() == [{"unit_id": uid}]
        assert not con.execute("SELECT unit_id FROM chunks WHERE unit_id=%s", ("opencode:ses_archive",)).fetchall()
        assert con.execute("SELECT unit_id FROM units WHERE unit_id=%s", ("unrelated-history",)).fetchone()


@pytest.mark.parametrize("failure", ["malformed", "transport", "loop", "schema"])
def test_incomplete_inventory_never_purges(pg_test_db, tmp_path, monkeypatch, failure):
    state, _ = _wire_opencode(tmp_path, monkeypatch)
    opencode_source.ingest()
    data = json.loads(state.read_text())
    data["hidden"] = ["ses_parent"]
    data["fail"] = failure
    data["sessions"]["ses_new"] = _session("ses_new")
    state.write_text(json.dumps(data))
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        opencode_source.ingest()
    with clync.connect() as con:
        assert con.execute("SELECT unit_id FROM units WHERE unit_id=%s", ("opencode:ses_parent",)).fetchone()


def test_dirty_retry_updates_same_timestamp_and_revert(pg_test_db, tmp_path, monkeypatch):
    state, _ = _wire_opencode(tmp_path, monkeypatch)
    data = json.loads(state.read_text())
    data["active"] = {"ses_parent": {"type": "running"}}
    state.write_text(json.dumps(data))
    assert opencode_source.ingest()["dirty"] == 1
    data["active"] = {}
    data["sessions"]["ses_parent"]["messages"][0]["text"] = "Use the greenhouse instead"
    state.write_text(json.dumps(data))
    assert opencode_source.ingest()["sessions_ingested"] == 1
    assert opencode_source.ingest()["unchanged"] == 1
    moved = copy.deepcopy(data["sessions"]["ses_parent"])
    moved["info"]["title"] = "Moved greenhouse"
    moved["info"]["location"]["directory"] = "/code/garden/greenhouse"
    moved["messages"] = moved["messages"][:1]
    data["sessions"]["ses_parent"]["info"]["time"]["updated"] += 100
    data["after_export"] = moved
    state.write_text(json.dumps(data))
    assert opencode_source.ingest()["dirty"] == 1
    assert opencode_source.ingest()["sessions_ingested"] == 1
    with clync.connect() as con:
        unit = con.execute("SELECT title,cwd,msg_count FROM units WHERE source='opencode'").fetchone()
        assert unit == {"title": moved["info"]["title"], "cwd": moved["info"]["location"]["directory"], "msg_count": 1}
    data = json.loads(state.read_text())
    data["sessions"]["ses_parent"]["info"]["parentID"] = "ses_new_parent"
    state.write_text(json.dumps(data))
    assert opencode_source.ingest()["sessions_removed"] == 1
