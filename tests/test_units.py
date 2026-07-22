"""Pure-function units — no Postgres, no model, no network."""
from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest

import clync

search = pytest.importorskip("search")


def _fake_client(orgs):
    return types.SimpleNamespace(list_orgs=lambda: orgs)


def test_resolve_orgs_defaults_to_all_chat_capable():
    client = _fake_client([
        {"uuid": "o1", "name": "Work", "capabilities": ["chat"]},
        {"uuid": "o2", "name": "Org2", "capabilities": ["raven", "chat"]},
        {"uuid": "o3", "name": "Org3", "capabilities": ["api"]},
    ])
    assert [o["name"] for o in clync.resolve_orgs(client, None)] == ["Work", "Org2"]


def test_resolve_orgs_includes_orgs_with_unknown_capabilities():
    client = _fake_client([
        {"uuid": "o1", "name": "Chatful", "capabilities": ["chat"]},
        {"uuid": "o2", "name": "NoCapField"},
        {"uuid": "o3", "name": "EmptyCaps", "capabilities": []},
        {"uuid": "o4", "name": "ApiOnly", "capabilities": ["api"]},
    ])
    assert ([o["name"] for o in clync.resolve_orgs(client, None)]
            == ["Chatful", "NoCapField", "EmptyCaps"])


def test_resolve_orgs_never_syncs_nothing():
    client = _fake_client([
        {"uuid": "o1", "name": "A", "capabilities": ["api"]},
        {"uuid": "o2", "name": "B", "capabilities": ["raven"]},
    ])
    assert [o["name"] for o in clync.resolve_orgs(client, None)] == ["A", "B"]


def test_resolve_orgs_explicit_selects_one():
    client = _fake_client([
        {"uuid": "o1", "name": "Work", "capabilities": ["chat"]},
        {"uuid": "o2", "name": "Org2", "capabilities": ["chat"]},
    ])
    assert [o["uuid"] for o in clync.resolve_orgs(client, "Org2")] == ["o2"]
    assert [o["uuid"] for o in clync.resolve_orgs(client, "o1")] == ["o1"]


def test_message_text_content_blocks():
    msg = {"content": [{"type": "text", "text": "hello"},
                       {"type": "tool_use"},                     # non-text ignored
                       {"type": "text", "text": "world"}]}
    assert clync._message_text(msg) == "hello\nworld"


def test_message_text_fallback_and_attachments():
    msg = {"text": "top", "attachments": [{"file_name": "a.txt",
                                            "extracted_content": "doc body"}]}
    out = clync._message_text(msg)
    assert "top" in out and "[attachment: a.txt]" in out and "doc body" in out


def test_message_text_hollow_is_empty():
    assert clync._message_text({"content": [{"type": "tool_use"}]}) == ""
    assert clync._message_text({}) == ""


def test_scrub_nul_recursive():
    # Postgres rejects 0x00 in text AND the u0000 escape in jsonb; one
    # recursive scrub backs both chokepoints (connect_pg str dumpers + _json).
    assert clync._scrub_nul("a\x00b") == "ab"
    assert (clync._scrub_nul({"k\x00": ["v\x00", 1, None, {"x": "y\x00"}]})
            == {"k": ["v", 1, None, {"x": "y"}]})
    assert clync._scrub_nul(None) is None


def test_sparse_literal_is_1_based():
    lit = search._sparse_literal({5: 0.2, 0: 0.5})
    assert lit == f"{{1:0.5,6:0.2}}/{search.SPARSE_DIM}"


def test_sparse_literal_empty_raises():
    with pytest.raises(ValueError):
        search._sparse_literal({})
    with pytest.raises(ValueError):
        search._sparse_literal({3: 0.0})


def test_lang_detection():
    assert search._lang("hello world") == "en"
    assert search._lang("こんにちは") == "ja"
    assert search._lang("你好世界") == "zh"


def test_validate_facets_rejects_source_mismatch():
    with pytest.raises(ValueError):
        search._validate_facets("claude_ai", project=None, model=None,
                                repo="clync", worktree=None, branch=None)
    with pytest.raises(ValueError):
        search._validate_facets("claude_code", project="Work", model=None,
                                repo=None, worktree=None, branch=None)
    with pytest.raises(ValueError):
        search._validate_facets("bogus", project=None, model=None,
                                repo=None, worktree=None, branch=None)
    # a valid combination does not raise
    search._validate_facets("claude_code", project=None, model=None,
                            repo="clync", worktree=None, branch="main")


def test_pg_db_name_rejects_injection_at_import():
    # $CLYNC_PG_DB is interpolated into SQL identifiers -> must be validated at load
    # (the validation lives in clync, which search imports).
    env = {**os.environ, "CLYNC_PG_DB": "clync; DROP DATABASE postgres"}
    r = subprocess.run([sys.executable, "-c", "import clync"],
                       cwd=os.path.dirname(os.path.dirname(__file__)),
                       env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert "invalid $CLYNC_PG_DB" in (r.stderr + r.stdout)
