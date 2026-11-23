from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import live_search


@pytest.fixture
def native_roots(tmp_path, monkeypatch):
    import cc
    import codex
    cc_root, codex_root = tmp_path / "cc", tmp_path / "codex"
    (cc_root / "project").mkdir(parents=True)
    (codex_root / "sessions").mkdir(parents=True)
    monkeypatch.setattr(cc, "CC_ROOT", cc_root)
    monkeypatch.setattr(codex, "CODEX_ROOT", codex_root)
    return cc_root, codex_root


def write_cc(root, sid, texts):
    path = root / "project" / f"{sid}.jsonl"
    events = [{"type": "user", "uuid": f"{sid}-{i}", "sessionId": sid,
               "cwd": "/code/orchard", "entrypoint": "cli", "origin": {"kind": "human"},
               "message": {"content": text}} for i, text in enumerate(texts)]
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))
    return path


def test_live_fuzzy_evidence_read_and_append(native_roots):
    root, _ = native_roots
    path = write_cc(root, "fruit", ["Choose apricot orchards", "Irrigation timer reset"])
    write_cc(root, "noise", ["apricot facts", "orchards on another continent"])
    result = live_search.search_sessions("aprcot orchrads", source="claude_code")
    assert [h["session_id"] for h in result["hits"]] == ["fruit"]
    hit = result["hits"][0]
    assert "Choose apricot orchards" in hit["text"]
    read = live_search.read_session(hit["reference"], hit["anchor"], context=1)
    assert "Choose apricot orchards" in "\n".join(r["text"] for r in read["records"])
    with path.open("a") as out:
        out.write(json.dumps({"type": "assistant", "uuid": "added", "message": {
            "content": [{"type": "text", "text": "Trellis cable replaced"}]}}) + "\n")
    assert live_search.search_sessions("trellis cable", source="claude_code")["hits"]
    assert live_search.search_sessions("apricot", source="claude_code", repo="elsewhere")["hits"] == []


def test_noise_unicode_scope_and_unavailable(native_roots):
    root, codex_root = native_roots
    path = write_cc(root, "japanese", ["認証設定を変更した", "ＳＳＯを確認した"])
    with path.open("a") as out:
        out.write(json.dumps({"type": "user", "origin": {"kind": "human"},
                             "message": {"content": "楓盆栽の剪定"}}) + "\n")
        out.write(json.dumps({"type": "user", "isMeta": True, "message": {
            "content": "Forbidden papaya"}}) + "\n")
        out.write(json.dumps({"type": "user", "origin": {"kind": "task-notification"},
                              "message": {"content": "Forbidden papaya"}}) + "\n")
    assert live_search.search_sessions("認証設定", source="claude_code")["hits"]
    assert live_search.search_sessions("楓盆栽", source="claude_code")["hits"]
    assert live_search.search_sessions("sso", source="claude_code")["hits"]
    assert not live_search.search_sessions("papaya", source="claude_code")["hits"]
    for sid, source in (("desktop", "vscode"), ("child", {"subagent": {"other": "guardian"}})):
        events = [{"type": "session_meta", "payload": {"id": sid, "cwd": "/code/orchard",
                   "originator": "Codex Desktop", "source": source}},
                  {"type": "response_item", "payload": {"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": "Mango harvest"}]}},
                  {"type": "event_msg", "payload": {"type": "user_message", "message": "Mango harvest"}}]
        (codex_root / "sessions" / f"rollout-{sid}.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events))
    result = live_search.search_sessions("mango", source="codex_cli")
    assert [h["session_id"] for h in result["hits"]] == ["desktop"]
    assert {h["session_id"] for h in live_search.search_sessions(
        "mango", source="codex_cli", include_subagents=True)["hits"]} == {"desktop", "child"}
    import cc
    cc.CC_ROOT = root / "missing"
    unavailable = live_search.search_sessions("mango", source="claude_code")
    assert unavailable["coverage"]["claude_code"]["status"] != live_search.COMPLETE


def test_candidate_verification_and_stale_anchor(native_roots):
    root, _ = native_roots
    path = write_cc(root, "anchored", ["Nectarine bucket"])
    hit = live_search.search_sessions("nectarine", source="claude_code")["hits"][0]
    path.write_text(path.read_text().replace("Nectarine bucket", "Different record"))
    with pytest.raises(ValueError):
        live_search.read_session(hit["reference"], hit["anchor"])
    with pytest.raises(ValueError):
        live_search.read_session("claude_code:../../secrets", "0:invalid")
    with path.open("a") as out:
        out.write('{"type":"user","message":{"content":"unfinished apricot')
    result = live_search.search_sessions("apricot", source="claude_code")
    assert not result["hits"]
    assert result["coverage"]["claude_code"]["status"] != live_search.COMPLETE


def test_long_record_read_contains_the_actual_match(native_roots):
    root, _ = native_roots
    write_cc(root, "long", ["preamble " * 4000 + "Alstroemeria bracket repaired"])
    hit = live_search.search_sessions("alstroemeria bracket", source="claude_code")["hits"][0]
    result = live_search.read_session(hit["reference"], hit["anchor"], context=0)
    assert "Alstroemeria bracket repaired" in "\n".join(r["text"] for r in result["records"])


def test_pending_codex_header_does_not_hide_other_sessions(native_roots):
    _, root = native_roots
    events = [{"type": "session_meta", "payload": {"id": "healthy", "source": "cli", "cwd": "/code/garden"}},
              {"type": "response_item", "payload": {"type": "message", "role": "user",
               "content": [{"type": "input_text", "text": "Caladium bracket"}]}}]
    (root / "sessions" / "rollout-healthy.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events))
    (root / "sessions" / "rollout-pending.jsonl").write_text('{"type":"session_meta",')
    result = live_search.search_sessions("caladium bracket", source="codex_cli")
    assert [h["session_id"] for h in result["hits"]] == ["healthy"]
    assert result["coverage"]["codex_cli"]["errors"]


def test_live_cli_does_not_need_ml_or_database(tmp_path):
    root = tmp_path / "native"
    (root / "project").mkdir(parents=True)
    write_cc(root, "fresh", ["Dragonfruit spindle repaired"])
    import os
    env = {**os.environ, "CLYNC_CC_ROOT": str(root), "CLYNC_PG_BIN": "/missing/postgres"}
    command = [sys.executable, "-S", "clync.py", "sessions", "search", "dragonfrut spindle",
               "--source", "claude_code", "--json"]
    run = subprocess.run(command, cwd=Path(__file__).parents[1], env=env,
                         text=True, capture_output=True, check=True)
    assert json.loads(run.stdout)["hits"][0]["session_id"] == "fresh"


def test_staged_default_and_exhaustive_fuzzy_search(native_roots):
    root, _ = native_roots
    write_cc(root, "literal", ["Repair nectarine wiring"])
    write_cc(root, "typo", ["Weather notes", "Repair nectraine wiring"])
    quick = live_search.search_sessions("nectarine wiring", source="claude_code")
    assert {h["session_id"] for h in quick["hits"]} == {"literal", "typo"}
    quick = live_search.search_sessions("nectarine", source="claude_code")
    deep = live_search.search_sessions("nectarine", source="claude_code", exhaustive=True)
    assert {h["session_id"] for h in quick["hits"]} == {"literal"}
    assert {h["session_id"] for h in deep["hits"]} == {"literal", "typo"}


def test_tools_attribution_question_answers_and_same_record_read(native_roots):
    root, _ = native_roots
    path = write_cc(root, "question", ["Initial note"])
    events = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "fork", "name": "AskUserQuestion", "input": {}},
            {"type": "tool_use", "id": "output", "name": "Read", "input": {"path": "/garden"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "fork",
            "content": "assistant framing"}]}, "toolUseResult": {"answers": {"Which?": "Amaryllis"}}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "output", "content": "x" * 30000},
            {"type": "tool_result", "tool_use_id": "output", "content": "Begonia pot repaired"}]}},
    ]
    with path.open("a") as out:
        out.write("".join(json.dumps(e) + "\n" for e in events))
    assert live_search.search_sessions("amaryllis", source="claude_code", sender="user")["hits"]
    assert not live_search.search_sessions("begonia", source="claude_code")["hits"]
    hit = live_search.search_sessions("begonia", source="claude_code", include_tools=True)["hits"][0]
    read = live_search.read_session(hit["reference"], hit["anchor"], context=1, include_tools=True)
    assert "Begonia pot repaired" in "\n".join(r["text"] for r in read["records"])
    with pytest.raises(ValueError):
        live_search.read_session(hit["reference"], hit["anchor"])
