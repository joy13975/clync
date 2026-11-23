import json
import os
import re
import sqlite3
import sys
import types
from pathlib import Path

import pytest

import clync
from cc import CCMessage, CCSession


def test_pi_archive_tests_use_an_empty_isolated_default_root(tmp_path):
    root = Path(os.environ["CLYNC_PI_ROOT"])
    assert root.is_relative_to(tmp_path)
    assert root.is_dir() and list(root.iterdir()) == []
    assert "PI_CODING_AGENT_SESSION_DIR" not in os.environ
    if "pi_source" in sys.modules:
        assert sys.modules["pi_source"].PI_ROOT == root


@pytest.fixture
def archive(tmp_path, monkeypatch):
    db = tmp_path / "archive.sqlite"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE units (
            unit_id TEXT PRIMARY KEY, kind TEXT, source TEXT, title TEXT, summary TEXT,
            model TEXT, created_at TEXT, updated_at TEXT, repo TEXT, cwd TEXT, worktree TEXT,
            git_branch TEXT, cc_version TEXT, entrypoint TEXT, raw TEXT, synced_at TEXT,
            msg_count INTEGER DEFAULT 0);
        CREATE TABLE messages (
            unit_id TEXT, msg_id TEXT, idx INTEGER, sender TEXT, role_detail TEXT,
            text TEXT, embed_text TEXT, created_at TEXT, raw TEXT,
            PRIMARY KEY (unit_id, msg_id));
        CREATE TABLE cc_sync_state (
            file_path TEXT PRIMARY KEY, mtime REAL, size INTEGER, session_id TEXT,
            source TEXT, synced_at TEXT);
    """)
    con.close()

    class Connection:
        def __init__(self):
            self.db = sqlite3.connect(db)
            self.db.row_factory = sqlite3.Row

        def execute(self, sql, params=()):
            sql = re.sub(r"=\s*ANY\(%s\)", "IN (SELECT value FROM json_each(%s))", sql)
            sql = sql.replace("DELETE FROM units u", "DELETE FROM units AS u")
            sql = sql.replace("RETURNING u.unit_id", "RETURNING unit_id")
            sql = sql.replace("SELECT COUNT(*) FROM", "SELECT COUNT(*) AS count FROM")
            return self.db.execute(sql.replace("%s", "?"), tuple(
                json.dumps(p) if isinstance(p, list) else p for p in params))

        def commit(self):
            self.db.commit()

        def close(self):
            self.db.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            if exc[0] is None:
                self.commit()
            self.close()

    monkeypatch.setattr(clync, "connect", Connection)
    monkeypatch.setattr(clync, "_json", json.dumps)
    monkeypatch.setattr(clync, "ensure_cluster", lambda: None)
    return Connection


@pytest.fixture
def pi(tmp_path, monkeypatch):
    root = tmp_path / "pi-sessions"
    root.mkdir()
    parsed = []

    def iter_files():
        if not root.is_dir():
            raise FileNotFoundError(f"Pi source unavailable: {root}")
        return root.glob("*.jsonl")

    def parse(path):
        parsed.append(path)
        return CCSession(
            session_id=f"pi:{path.stem}", title=path.stem, summary=None,
            created_at="2026-01-01T00:00:00Z", updated_at="2026-01-01T00:00:01Z",
            repo="garden", cwd="/code/garden", worktree=None, git_branch=None,
            cc_version=None, entrypoint="pi", model="local",
            messages=[CCMessage("question", 0, "pi_user", "text", path.read_text(), None)])

    module = types.SimpleNamespace(PI_ROOT=root, iter_session_files=iter_files,
                                   parse_session=parse, parsed=parsed)
    monkeypatch.setitem(sys.modules, "pi_source", module)
    monkeypatch.delenv("CLYNC_PI_ROOT", raising=False)
    return module


def test_pi_ingest_is_incremental_and_reconciles_only_pi(archive, pi):
    transcript = pi.PI_ROOT / "one.jsonl"
    transcript.write_text("Keep the heliotrope outdoors")
    with archive() as con:
        con.execute("INSERT INTO units(unit_id,kind,source) VALUES(%s,%s,%s)",
                    ("other", "cc_session", "claude_code"))
    assert clync.ingest_pi()["sessions_ingested"] == 1
    assert clync.ingest_pi()["unchanged"] == 1
    assert pi.parsed == [transcript]
    transcript.write_text("Move the heliotrope into the greenhouse instead")
    assert clync.ingest_pi()["sessions_ingested"] == 1
    with archive() as con:
        row = con.execute("SELECT kind,source,msg_count FROM units WHERE unit_id=%s",
                          ("pi:one",)).fetchone()
        assert tuple(row) == ("pi_session", "pi", 1)
        message = con.execute("SELECT sender,text FROM messages WHERE unit_id=%s",
                              ("pi:one",)).fetchone()
        assert tuple(message) == ("pi_user", transcript.read_text())
    assert clync.ingest_pi(full=True)["files_parsed"] == 1
    transcript.unlink()
    assert clync.ingest_pi()["sessions_removed"] == 1
    with archive() as con:
        assert [r["unit_id"] for r in con.execute("SELECT unit_id FROM units").fetchall()] == ["other"]


def test_unconfigured_missing_default_is_reported_without_ingesting(archive, pi, capsys):
    pi.PI_ROOT.rmdir()
    result = clync.ingest_pi()
    assert result["skipped"] is True
    assert "[pi]" in capsys.readouterr().out
    assert pi.parsed == []


@pytest.mark.parametrize("configured", ["missing", "file", "empty"])
def test_explicit_invalid_root_fails_loud(archive, pi, monkeypatch, configured):
    pi.PI_ROOT.rmdir()
    if configured == "file":
        pi.PI_ROOT.write_text("not a directory")
    monkeypatch.setenv("CLYNC_PI_ROOT", "" if configured == "empty" else str(pi.PI_ROOT))
    with pytest.raises((OSError, RuntimeError, ValueError), match="Pi|pi|CLYNC_PI_ROOT"):
        clync.ingest_pi()


@pytest.mark.parametrize("stored", ["unit", "watermark"])
def test_missing_previously_indexed_source_fails_without_purging(archive, pi, stored):
    with archive() as con:
        if stored == "unit":
            con.execute("INSERT INTO units(unit_id,kind,source) VALUES(%s,%s,%s)",
                        ("pi:old", "pi_session", "pi"))
        else:
            con.execute("INSERT INTO cc_sync_state(file_path,source) VALUES(%s,%s)",
                        ("/old/session.jsonl", "pi"))
    pi.PI_ROOT.rmdir()
    with pytest.raises((OSError, RuntimeError), match="Pi|pi"):
        clync.ingest_pi()
    with archive() as con:
        table = "units" if stored == "unit" else "cc_sync_state"
        assert con.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"] == 1


def test_existing_non_directory_default_fails_loud(archive, pi):
    pi.PI_ROOT.rmdir()
    pi.PI_ROOT.write_text("not a directory")
    with pytest.raises((OSError, RuntimeError), match="Pi|pi"):
        clync.ingest_pi()


def test_existing_unreadable_source_never_becomes_optional(archive, pi):
    def unreadable():
        raise PermissionError(f"Pi source inaccessible: {pi.PI_ROOT}")
    pi.iter_session_files = unreadable
    with pytest.raises(PermissionError, match="Pi"):
        clync.ingest_pi()


@pytest.mark.parametrize("full,no_index", [(False, False), (True, False), (True, True)])
def test_sync_pi_cli_ingests_before_optional_index(archive, pi, monkeypatch, full, no_index):
    (pi.PI_ROOT / "cli.jsonl").write_text("heliotrope decision")
    indexed = []
    monkeypatch.setattr(clync, "_index_after_sync", lambda value: indexed.append(
        (value, len(pi.parsed))))
    monkeypatch.setattr(sys, "argv", ["clync", "sync-pi"]
                        + (["--full"] if full else []) + (["--no-index"] if no_index else []))
    assert clync.main() == 0
    assert len(pi.parsed) == 1
    assert indexed == ([] if no_index else [(full, 1)])


@pytest.mark.parametrize("full", [False, True])
def test_all_source_sync_ingests_pi_before_one_index(archive, pi, monkeypatch, full):
    (pi.PI_ROOT / "scheduled.jsonl").write_text("heliotrope decision")
    for name in ("run_sync", "ingest_chatgpt", "ingest_cc", "ingest_codex"):
        monkeypatch.setattr(clync, name, lambda *a, **kw: None)
    indexed = []
    monkeypatch.setattr(clync, "_index_after_sync", lambda value: indexed.append(
        (value, len(pi.parsed))))
    assert clync._sync_all("unused", None, full, False, True, False) is None
    assert indexed == [(full, 1)]


@pytest.mark.parametrize("facet,value", [
    ("repo", "garden"), ("session", "pi:facet"), ("model", "local"),
])
def test_pi_archive_metadata_facets_are_searchable(facet, value):
    import search

    facets = {facet: value}
    search._validate_facets("pi", facets)
    params = {}
    search._facet_where("pi", facets, None, None, params)
    assert params["sources"] == ["pi"]
    assert params[facet] == value


def test_archive_pi_source_reaches_list(archive, pi, monkeypatch, capsys):
    (pi.PI_ROOT / "facet.jsonl").write_text("heliotrope decision")
    clync.ingest_pi()
    monkeypatch.setattr(sys, "argv", ["clync", "list", "--source", "pi"])
    assert clync.main() == 0
    assert "[pi]" in capsys.readouterr().out
