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
        {"uuid": "o1", "name": "Workspace One", "capabilities": ["chat"]},
        {"uuid": "o2", "name": "Workspace Two", "capabilities": ["raven", "chat"]},
        {"uuid": "o3", "name": "API Workspace", "capabilities": ["api"]},
    ])
    assert [o["name"] for o in clync.resolve_orgs(client, None)] == ["Workspace One", "Workspace Two"]


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
        {"uuid": "o1", "name": "Workspace One", "capabilities": ["chat"]},
        {"uuid": "o2", "name": "Workspace Two", "capabilities": ["chat"]},
    ])
    assert [o["uuid"] for o in clync.resolve_orgs(client, "Workspace Two")] == ["o2"]
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


def _facets(**over):
    f = dict.fromkeys(search.FACET_SOURCES)
    f.update(over)
    return f


def test_validate_facets_rejects_source_mismatch():
    with pytest.raises(ValueError):
        search._validate_facets("claude_ai", _facets(repo="clync"))
    with pytest.raises(ValueError):
        search._validate_facets("claude_code", _facets(project="Work"))
    with pytest.raises(ValueError):
        search._validate_facets("bogus", _facets())
    # a valid combination does not raise
    search._validate_facets("claude_code", _facets(repo="clync", branch="main"))


def test_validate_facets_covers_every_facet_the_where_builder_knows():
    """The regression: `session` reached the WHERE builder but not the
    hand-enumerated guard, so source='dream' + session=... (and claude_ai +
    session=...) validated fine and then ANDed an unsatisfiable predicate —
    a silent, confident empty result. The guard and the builder now read ONE
    registry, so every registered facet is guarded for every source."""
    with pytest.raises(ValueError, match="session"):
        search._validate_facets("dream", _facets(session="clync-main"))
    with pytest.raises(ValueError, match="session"):
        search._validate_facets("claude_ai", _facets(session="clync-main"))
    search._validate_facets("claude_code", _facets(session="clync-main"))


def test_an_unregistered_facet_fails_loud_in_guard_and_builder():
    """A facet added to the WHERE builder without registering it in FACET_SOURCES
    must fail on the first call, never silently skip validation."""
    with pytest.raises(ValueError, match="not registered"):
        search._validate_facets("all", _facets(sprocket="x"))
    with pytest.raises(ValueError, match="not registered"):
        search._facet_where("all", _facets(sprocket="x"), None, None, {})


def test_pg_db_name_rejects_injection_at_import():
    # $CLYNC_PG_DB is interpolated into SQL identifiers -> must be validated at load
    # (the validation lives in clync, which search imports).
    env = {**os.environ, "CLYNC_PG_DB": "clync; DROP DATABASE postgres"}
    r = subprocess.run([sys.executable, "-c", "import clync"],
                       cwd=os.path.dirname(os.path.dirname(__file__)),
                       env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert "invalid $CLYNC_PG_DB" in (r.stderr + r.stdout)


def test_resource_fraction_caps_every_pool_the_embedder_reads(monkeypatch):
    """A sync must leave the machine usable. The embedder's stack sizes each of its
    thread pools from the environment at load time and otherwise takes every core, so
    the fraction has to reach ALL of them, not just torch's."""
    import clync
    for var in clync._THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(clync, "RESOURCE_FRACTION", 0.5)

    applied = clync.limit_resources()
    want = max(1, (os.cpu_count() or 1) // 2)
    assert {applied[v] for v in clync._THREAD_VARS} == {str(want)}
    # applied to the environment itself, which is where the pools read it
    assert os.environ["OMP_NUM_THREADS"] == str(want)


def test_an_exported_thread_count_wins_over_the_fraction(monkeypatch):
    # The fraction is a default, not an override: a user who set a pool size meant it.
    import clync
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setattr(clync, "RESOURCE_FRACTION", 1.0)
    assert clync.limit_resources()["OMP_NUM_THREADS"] == "3"


def test_a_fraction_below_one_core_still_leaves_one(monkeypatch):
    import clync
    for var in clync._THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(clync, "RESOURCE_FRACTION", 0.001)
    assert clync.limit_resources()["OMP_NUM_THREADS"] == "1"


@pytest.mark.parametrize('host,reader,auth_name', [
    ('claude.ai', clync.read_auth_cookies, 'sessionKey'),
    ('chatgpt.com', clync.read_chatgpt_cookies, clync.chatgpt_mod_session_cookie()),
])
def test_cookie_lookup_excludes_lookalike_domains(tmp_path, monkeypatch, host, reader, auth_name):
    import sqlite3

    with sqlite3.connect(tmp_path / 'Cookies') as con:
        con.execute('CREATE TABLE cookies (host_key TEXT, name TEXT, encrypted_value BLOB)')
        con.executemany('INSERT INTO cookies VALUES (?,?,?)', [
            ('.' + host, auth_name, b'sk-ant-sid-legitimate'),
            (host, 'cf_clearance', b'legitimate-clearance'),
            (host + '.attacker.invalid', auth_name, b'sk-ant-sid-attacker'),
            ('not' + host, 'cf_clearance', b'attacker-clearance'),
        ])
    monkeypatch.setattr(clync, 'resolve_profile_dir', lambda profile: tmp_path)
    monkeypatch.setattr(clync, '_safe_storage_key', lambda: b'unused')
    monkeypatch.setattr(clync, '_decrypt_cookie', lambda value, key: value.decode())
    cookies = reader('Test profile')
    assert cookies[auth_name] == 'sk-ant-sid-legitimate'
    assert cookies['cf_clearance'] == 'legitimate-clearance'
