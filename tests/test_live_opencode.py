import copy
import json
import socket
import sqlite3
import subprocess
from urllib.parse import urlparse

import pytest

import cc
import live_search
import opencode_source


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "opencode.db"
    con = sqlite3.connect(path, check_same_thread=False)
    con.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE session_v2(id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT,
            directory TEXT, title TEXT, time_created INTEGER, time_updated INTEGER);
        CREATE TABLE project(id TEXT PRIMARY KEY, worktree TEXT);
        CREATE TABLE session_message(id TEXT PRIMARY KEY, session_id TEXT, type TEXT,
            seq INTEGER, time_created INTEGER, time_updated INTEGER, data TEXT);
        CREATE TABLE session_part(message_id TEXT, session_id TEXT, ordinal INTEGER,
            type TEXT, data TEXT, PRIMARY KEY(message_id, ordinal));
        CREATE TABLE blob(hash TEXT PRIMARY KEY);
        INSERT INTO project VALUES ('project', '/code/garden');
    """)
    projected, requests = {}, []
    info = {"version": "2.2.6", "hostname": socket.gethostname(), "pid": 12345}

    def session(sid="ses_root", title="Garden", parent=None, directory="/code/garden/tree",
                updated=1800000001000):
        con.execute("INSERT INTO session_v2 VALUES (?, 'project', ?, ?, ?, ?, ?)",
                    (sid, parent, directory, title, 1800000000000, updated))
        con.commit()
        projected[sid] = []
        return sid

    def message(sid, mid, content, *, kind="assistant", parts=(), expected=None, ascii=False):
        data = {"time": {"created": 1800000000000}}
        data.update({"text": content} if kind == "user" else {"content": content})
        con.execute("INSERT INTO session_message VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (mid, sid, kind, len(projected[sid]), 1800000000000, 1800000000000,
                     json.dumps(data, ensure_ascii=ascii)))
        for ordinal, part in parts:
            con.execute("INSERT INTO session_part VALUES (?, ?, ?, ?, ?)",
                        (mid, sid, ordinal, part["type"], json.dumps(part, ensure_ascii=ascii)))
        con.commit()
        value = {**copy.deepcopy(data), "id": mid, "type": kind}
        if kind == "assistant":
            value["content"] = [copy.deepcopy(p) for _, p in sorted(parts)] + copy.deepcopy(content)
        projected[sid].append(expected or value)

    def get(self, target):
        route = urlparse(target).path
        if route == "/api/info":
            return dict(info)
        if route == "/api/project":
            return [{"id": "project", "canonical": "/code/garden"}]
        if route == "/api/session":
            values = [{"id": row[0], "projectID": row[1], "parentID": row[2],
                       "location": {"directory": row[3]}, "title": row[4],
                       "time": {"created": row[5], "updated": row[6]}}
                      for row in con.execute("SELECT * FROM session_v2 ORDER BY id")]
            return {"data": values, "cursor": {"next": None}}
        if route.endswith("/message"):
            sid = route.split("/")[-2]
            requests.append(sid)
            return {"data": copy.deepcopy(projected[sid]), "cursor": {"next": None}}
        raise AssertionError(target)

    def run(command, **kwargs):
        if command[1:] == ["debug", "paths", "db"]:
            return subprocess.CompletedProcess(command, 0, str(path) + "\n", "")
        if command[0].endswith("lsof"):
            return subprocess.CompletedProcess(command, 0, f"p12345\nn{path}\n", "")
        raise AssertionError(command)

    monkeypatch.setattr(opencode_source.Client, "__init__",
                        lambda self: setattr(self, "command", "/configured/opencode"))
    monkeypatch.setattr(opencode_source.Client, "get", get)
    monkeypatch.setattr(subprocess, "run", run)
    yield type("Store", (), {"path": path, "con": con, "info": info, "requests": requests,
                             "projected": projected,
                             "session": staticmethod(session), "message": staticmethod(message)})()
    con.close()


def text(value):
    return {"type": "text", "text": value}


def test_no_hit_uses_native_candidates_not_every_session_api(store):
    for i in range(12):
        sid = store.session(f"ses_noise{i}")
        store.message(sid, f"msg_noise{i}", [text("ordinary orchard notes")])
    result = live_search.search_sessions("xylophone", source="opencode", exact=True)
    assert result["hits"] == []
    assert result["coverage"]["opencode"]["status"] == live_search.COMPLETE
    assert store.requests == []
    assert result["coverage"]["opencode"]["decoded_records"] == 0


@pytest.mark.parametrize("query,value,options", [
    ("nectarine", "nectraine", {}),
    ("apricot", "apricot", {"exact": True}),
    ("nectarine", "nectXarine", {"exhaustive": True}),
    ("sso", "ＳＳＯ", {"exact": True}),
    ("2027", "予算 ２０２７ 年度", {"exact": True}),
    ("kelvin", "KELVIN", {}),
    ("楓盆栽", "楓盆栽", {"exact": True}),
    ("nectarine", "nectarine", {"exact": True, "ascii": True}),
])
def test_candidates_preserve_fuzzy_unicode_and_json_escapes(store, query, value, options):
    sid = store.session()
    store.message(sid, "msg_user", value, kind="user", ascii=options.pop("ascii", False))
    hit, = live_search.search_sessions(query, source="opencode", **options)["hits"]
    assert hit["session_id"] == sid and hit["text"] == value
    assert store.requests == []


def test_escape_inside_literal_is_still_a_candidate(store):
    sid = store.session()
    store.message(sid, "msg_escape", "nectarine", kind="user")
    store.con.execute("UPDATE session_message SET data=replace(data, 'nectarine', ?)",
                      (r"nec\u0074arine",))
    store.con.commit()
    assert live_search.search_sessions("nectarine", source="opencode", exact=True)["hits"]
    assert store.requests == []


def test_split_parts_inline_order_questions_and_api_read_anchors(store):
    sid = store.session()
    question = {"type": "tool", "name": "question", "state": {"status": "completed",
                "input": {"questions": []}, "metadata": {"answers": [["Heliotrope"]]}, "content": []}}
    store.message(sid, "msg_split", [text("inline orchid")],
                  parts=[(9, text("split dahlia")), (2, question)])
    answers = live_search.search_sessions("heliotrope", source="opencode", sender="user")
    answer, = answers["hits"]
    assert answer["sender"] == cc.ANSWER_SENDER and answer["anchor"].startswith("msg_split:0:answer@")
    split, = live_search.search_sessions("dahlia", source="opencode")["hits"]
    inline, = live_search.search_sessions("orchid", source="opencode")["hits"]
    assert split["anchor"].startswith("msg_split:1@")
    assert inline["anchor"].startswith("msg_split:2@")
    assert store.requests == []
    read = live_search.read_session(inline["reference"], inline["anchor"], context=0)
    assert read["records"][0]["text"] == "inline orchid"
    store.con.execute("UPDATE session_message SET data=replace(data,'inline orchid','changed text')")
    store.con.commit()
    store.projected[sid][0]["content"][-1]["text"] = "changed text"
    with pytest.raises(ValueError, match="Stale"):
        live_search.read_session(inline["reference"], inline["anchor"], context=0)


def test_tool_media_placeholders_and_legacy_fields_are_not_omitted(store):
    sid = store.session()
    missing = "media:" + "a" * 64
    tool = {"type": "tool", "name": "read", "providerState": {"ignored": "nativeonly"},
            "providerResultState": {"ignored": "nativeonly"}, "state": {"status": "completed",
            "input": {"path": "/garden"}, "content": [
                {"type": "file", "uri": missing, "mime": "image/png", "name": "diagram"}]}}
    store.message(sid, "msg_media", [tool, {"type": "reasoning", "state": {"ignored": "nativeonly"},
                                          "text": "private reasoning"}])
    store.projected[sid][0]["content"][0]["state"]["content"] = [
        text("[media unavailable: diagram (image/png)]")]
    assert not live_search.search_sessions("unavailable", source="opencode")["hits"]
    hit, = live_search.search_sessions("unavailable", source="opencode", include_tools=True)["hits"]
    assert "[media unavailable: diagram (image/png)]" in hit["text"]
    assert not live_search.search_sessions("nativeonly", source="opencode", include_tools=True)["hits"]
    assert live_search.search_sessions("reasoning", source="opencode", include_tools=True)["hits"]
    assert store.requests == []


def test_titles_scope_children_sender_and_ranking(store):
    root = store.session(title="Zinnia")
    child = store.session("ses_child", parent=root)
    other = store.session("ses_other", directory="/code/elsewhere", updated=1700000000000)
    store.message(root, "msg_root", "zinnia", kind="user")
    store.message(child, "msg_child", "zinnia", kind="user")
    store.message(other, "msg_other", "zinnia", kind="user")
    result = live_search.search_sessions("zinnia", source="opencode", repo="garden",
                                         since="2026-01-01", include_subagents=True)
    assert [h["session_id"] for h in result["hits"]] == [child, root]
    assert all(h["sender"] == cc.HUMAN_SENDER for h in result["hits"])
    assert live_search.search_sessions("zinnia", source="opencode", sender="assistant")["hits"] == []
    assert store.requests == []


@pytest.mark.parametrize("failure", ["version", "remote", "identity", "schema"])
def test_unverified_native_source_reports_unavailable_without_api_fallback(store, monkeypatch, failure):
    sid = store.session()
    store.message(sid, "msg_user", "zinnia", kind="user")
    if failure == "version":
        store.info["version"] = "2.3.0"
    elif failure == "remote":
        store.info["hostname"] = "different-host"
    elif failure == "identity":
        store.info["pid"] = 99999
    else:
        store.con.execute("ALTER TABLE session_part RENAME COLUMN ordinal TO changed")
        store.con.commit()
    result = live_search.search_sessions("zinnia", source="opencode")
    coverage = result["coverage"]["opencode"]
    assert coverage["status"] == live_search.UNAVAILABLE and coverage["errors"]
    assert result["hits"] == [] and store.requests == []


@pytest.mark.parametrize("table", ["session_message", "session_part"])
def test_scoped_search_does_not_validate_excluded_session_rows(store, table, monkeypatch):
    selected = store.session("ses_zzselected")
    for i in range(12):
        store.session(f"ses_selected{i}")
    connect = sqlite3.connect

    def limited_connect(*args, **kwargs):
        con = connect(*args, **kwargs)
        con.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 8)
        return con

    monkeypatch.setattr(sqlite3, "connect", limited_connect)
    store.message(selected, "msg_selected", "zinnia", kind="user")
    excluded = store.session("ses_excluded", directory="/code/elsewhere")
    store.message(excluded, "msg_excluded", [], parts=[(0, text("other notes"))])
    store.con.execute(f"UPDATE {table} SET type='future', data='{{' WHERE session_id=?", (excluded,))
    store.con.commit()
    result = live_search.search_sessions("zinnia", source="opencode", repo="tree", exact=True)
    assert [h["session_id"] for h in result["hits"]] == [selected]
    assert result["coverage"]["opencode"]["status"] == live_search.COMPLETE
    assert result["coverage"]["opencode"]["sessions"] == 13


def test_non_reference_media_uri_is_ordinary_ignored_file_content(store):
    sid = store.session()
    tool = {"type": "tool", "name": "read", "state": {"status": "completed", "input": {},
            "content": [{"type": "file", "uri": "media:broken", "mime": "image/png"}]}}
    store.message(sid, "msg_media", [tool, text("zinnia")])
    result = live_search.search_sessions("zinnia", source="opencode", include_tools=True, exact=True)
    assert result["coverage"]["opencode"]["status"] == live_search.COMPLETE
    hit, = result["hits"]
    assert hit["text"] == "zinnia"
    assert not live_search.search_sessions("unavailable", source="opencode", include_tools=True)["hits"]
    read = live_search.read_session(hit["reference"], hit["anchor"], include_tools=True, context=0)
    assert read["records"][0]["text"] == "zinnia"


def test_snapshot_does_not_mix_concurrent_commits_and_next_search_sees_them(store, monkeypatch):
    sid = store.session()
    store.message(sid, "msg_before", "zinnia", kind="user")
    original = live_search._match
    wrote = False

    def match(value, *args):
        nonlocal wrote
        if value == "zinnia" and not wrote:
            wrote = True
            store.message(sid, "msg_after", "magnolia", kind="user")
        return original(value, *args)

    monkeypatch.setattr(live_search, "_match", match)
    result = live_search.search_sessions("zinnia", source="opencode", exact=True)
    assert result["hits"][0]["text"] == "zinnia"
    assert result["coverage"]["opencode"]["status"] == live_search.PARTIAL
    assert live_search.search_sessions("magnolia", source="opencode", exact=True)["hits"]
    assert store.requests == []
