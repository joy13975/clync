from __future__ import annotations

import json

import pytest

import cc
import live_files
import live_search
import pi_source
from tests.test_pi_source import header, message, transcript


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(pi_source, "PI_ROOT", tmp_path)
    return tmp_path


def find(query, **kwargs):
    return live_search.search_sessions(query, source="pi", **kwargs)


def test_primary_children_forks_and_unverified_users(root):
    path = transcript(root, [header(), message("u", None, "user", "Orchid spindle")])
    transcript(root, [header("fork.with.dot", parentSession=str(path)),
                     message("u", None, "assistant", "Orchid spindle")], "fork.jsonl")
    transcript(root, [header("child"),
        {"type": "custom", "id": "m", "parentId": None, "customType": "agentstrap.child",
         "data": {"parentSessionID": "primary"}},
        message("u", "m", "user", "Orchid spindle")], "agentstrap-children/child.jsonl")
    result = find("orchid spindle")
    assert {h["session_id"] for h in result["hits"]} == {"primary", "fork.with.dot"}
    assert next(h for h in result["hits"] if h["session_id"] == "primary")["sender"] == pi_source.USER_SENDER
    child = next(h for h in find("orchid spindle", include_subagents=True)["hits"]
                 if h["session_id"] == "child")
    assert child["sender"] != cc.HUMAN_SENDER and child["parent_id"]
    filtered = find("orchid spindle", sender="user")
    assert not filtered["hits"] and filtered["coverage"]["pi"]["status"] == live_search.PARTIAL
    assert filtered["coverage"]["pi"]["errors"]
    assert {h["session_id"] for h in find("orchid spindle", sender="assistant")["hits"]} == {"fork.with.dot"}
    hit = next(h for h in result["hits"] if h["session_id"] == "fork.with.dot")
    assert live_search.read_session(hit["reference"], hit["anchor"])["records"]


def test_branch_read_never_joins_siblings_and_warns_on_choice(root):
    transcript(root, [header(), message("u", None, "user", "Shared ancestor"),
        message("left", "u", "assistant", "Left canary"),
        message("left2", "left", "user", "Left continuation"),
        {"type": "model_change", "id": "m", "parentId": "u", "modelId": "other"},
        message("right", "m", "assistant", "Right canary")])
    hit = find("right canary")["hits"][0]
    read = live_search.read_session(hit["reference"], hit["anchor"], context=3)
    assert [r["text"] for r in read["records"]] == ["Shared ancestor", "Right canary"]
    warnings = read["warnings"]
    hit = find("shared ancestor")["hits"][0]
    read = live_search.read_session(hit["reference"], hit["anchor"], context=3)
    assert len(read["warnings"]) > len(warnings)
    texts = [r["text"] for r in read["records"]]
    assert not ("Left canary" in texts and "Right canary" in texts)


def test_compaction_and_context_edits_do_not_hide_raw_history(root):
    transcript(root, [header(), message("u", None, "user", "Precompaction celosia"),
        {"type": "compaction", "id": "c", "parentId": "u", "summary": "Summary noise",
         "firstKeptEntryId": "c"},
        {"type": "context_edit", "id": "e", "parentId": "c", "targetId": "u", "replacement": None},
        message("a", "e", "assistant", "After compaction")])
    assert find("precompaction celosia")["hits"]
    assert not find("summary noise")["hits"]


def test_tool_reasoning_and_machine_noise_are_opt_in(root):
    assistant = message("a", "u", "assistant", "Visible reply")
    assistant["message"]["content"] += [{"type": "thinking", "thinking": "Reasoning foxglove"},
        {"type": "toolCall", "id": "call", "name": "bash", "arguments": {"command": "Toolintent gaillardia"}}]
    transcript(root, [header(), message("sys", None, "system", "System chrysanthemum"),
        message("u", "sys", "user", "Prompt"), assistant,
        message("t", "a", "toolResult", "Output narcissus"),
        {"type": "custom_message", "id": "c", "parentId": "t", "content": "Injected freesia"}])
    for word in ("foxglove", "gaillardia", "narcissus", "chrysanthemum", "freesia"):
        assert not find(word)["hits"]
    for word, sender in (("foxglove", "assistant"), ("gaillardia", "tool"), ("narcissus", "tool")):
        hit = find(word, include_tools=True)["hits"][0]
        assert hit["sender"] == sender
        assert live_search.read_session(hit["reference"], hit["anchor"], include_tools=True)["records"]
        with pytest.raises(ValueError):
            live_search.read_session(hit["reference"], hit["anchor"])


def test_unicode_escaped_fuzzy_exact_and_exhaustive(root):
    transcript(root, [header(), message("u", None, "user", "認証設定 ＳＳＯ nectarine")])
    path = transcript(root, [header("escaped"), message("u", None, "user", "楓盆栽")], "escaped.jsonl")
    path.write_text(path.read_text().replace("楓盆栽", "\\u6953\\u76c6\\u683d"))
    transcript(root, [header("typo"), message("u", None, "assistant", "nectraine")], "typo.jsonl")
    for word in ("認証設定", "sso", "楓盆栽", "nectarne"):
        assert find(word)["hits"]
    assert not find("nectarne", exact=True)["hits"]
    assert {h["session_id"] for h in find("nectarine", exhaustive=True)["hits"]} == {"primary", "typo"}


def test_metadata_after_large_system_record_and_repo_filter(root):
    transcript(root, [header(), message("s", None, "system", "x" * 70000),
        message("u", "s", "user", "Delayed title lobelia"),
        {"type": "session_info", "id": "name", "parentId": "u", "name": "Named salvia"}])
    result = find("lobelia", repo="actual-repo")
    assert result["hits"][0]["title"] == "Named salvia"
    assert find("salvia")["hits"][0]["text"] == result["hits"][0]["title"]
    assert not find("lobelia", repo="not-the-project")["hits"]


def test_pending_malformed_mutation_append_and_stale_anchors(root, monkeypatch):
    path = transcript(root, [header(), message("a", None, "assistant", "Tuberose bracket")])
    hit = find("tuberose")["hits"][0]
    with path.open("a") as stream:
        stream.write('{"type":"message","message":{"role":"user","content":"unfinished tuberose')
    result = find("tuberose")
    assert result["hits"] and result["coverage"]["pi"]["status"] == live_search.PARTIAL
    path.write_text(path.read_text().splitlines()[0] + '\n{"type":"message","message":{"role":"user","content":"tuberose",broken}\n')
    result = find("tuberose")
    assert not result["hits"] and result["coverage"]["pi"]["status"] == live_search.PARTIAL
    with pytest.raises(ValueError):
        live_search.read_session(hit["reference"], hit["anchor"])
    transcript(root, [header(), message("a", None, "assistant", "Tuberose bracket")])
    original = live_files.candidates
    def mutate(*args, **kwargs):
        yield from original(*args, **kwargs)
        with path.open("a") as stream:
            stream.write(json.dumps(message("b", "a", "assistant", "Added delphinium")) + "\n")
    monkeypatch.setattr(live_files, "candidates", mutate)
    result = find("tuberose")
    assert result["coverage"]["pi"]["status"] == live_search.PARTIAL
    monkeypatch.setattr(live_files, "candidates", original)
    assert find("delphinium")["hits"]


def test_duplicate_ids_root_missing_and_path_safety(root, monkeypatch):
    transcript(root, [header(), message("a", None, "assistant", "Zinnia")])
    transcript(root, [header(), message("a", None, "assistant", "Zinnia duplicate")], "export.jsonl")
    result = find("zinnia")
    assert not result["hits"] and result["coverage"]["pi"]["status"] != live_search.COMPLETE
    assert any("duplicate" in e.lower() for e in result["coverage"]["pi"]["errors"])
    with pytest.raises(ValueError, match="[Dd]uplicate|[Aa]mbiguous"):
        live_search.read_session("pi:primary", "metadata")
    monkeypatch.setattr(pi_source, "PI_ROOT", root / "missing")
    assert find("zinnia")["coverage"]["pi"]["status"] == live_search.UNAVAILABLE
    for ref in ("pi:../../escape", "pi:bad/id", "pi:.."):
        with pytest.raises(ValueError):
            live_search.read_session(ref, "metadata")
