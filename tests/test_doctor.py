from __future__ import annotations

import json
import subprocess
import sys
import types
from urllib.parse import parse_qs, urlsplit

import pytest

import clync
import search


class _FakeResult:
    def __init__(self, n: int = 0):
        self.n = n

    def fetchone(self):
        return {"n": self.n}


class _FakeConnection:
    def execute(self, sql, *args):
        return _FakeResult()

    def close(self):
        pass


@pytest.fixture
def healthy_doctor(monkeypatch, tmp_path):
    monkeypatch.setattr(clync.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(search, "index_status", lambda: {
        "available": True, "cluster_running": True, "error": None, "chunks": 1,
    })
    monkeypatch.setattr(clync, "connect", lambda: _FakeConnection())
    monkeypatch.setattr(clync, "get_meta", lambda con, key: None)
    monkeypatch.setitem(sys.modules, "mcp_server", types.ModuleType("mcp_server"))

    import dream
    monkeypatch.setattr(dream, "status", lambda con: {
        "topics": [], "usage": {"failed": 0}, "pending": [], "watermark": None,
    })


def test_doctor_reports_pi_archive_count(healthy_doctor, monkeypatch, capsys):
    monkeypatch.setattr(_FakeConnection, "execute", lambda self, sql, *args:
                        _FakeResult(5 if "source='pi'" in sql else 0))
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0, stdout="", stderr=""))
    assert clync.cmd_doctor(types.SimpleNamespace(mcp_client="claude")) == 0
    assert "5 pi sessions" in capsys.readouterr().out


def test_doctor_fails_loud_on_scheduled_failure_and_failed_dream_queue(
        healthy_doctor, monkeypatch, capsys):
    import dream
    monkeypatch.setattr(dream, "status", lambda con: {
        "topics": [], "usage": {"failed": 1}, "pending": [], "watermark": None,
    })
    def _run(argv, **kwargs):
        if argv[0] == "launchctl":
            return subprocess.CompletedProcess(argv, 0, stdout="last exit code = 1\n",
                                               stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)

    assert clync.cmd_doctor(types.SimpleNamespace(mcp_client="claude")) == 1
    output = capsys.readouterr().out
    assert "last scheduled run exited 1" in output
    assert "dream has 1 failed queue item" in output
    assert "All green." not in output


@pytest.mark.parametrize("store_loaded", [True, False])
def test_doctor_requires_database_supervision(healthy_doctor, monkeypatch, store_loaded):
    def _run(argv, **kwargs):
        is_store = argv[0] == "launchctl" and argv[-1].endswith("/" + clync.PG_LAUNCHD_LABEL)
        return subprocess.CompletedProcess(
            argv, 113 if is_store and not store_loaded else 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    assert clync.cmd_doctor(types.SimpleNamespace(mcp_client="claude")) == (0 if store_loaded else 1)


@pytest.mark.parametrize("client", ["claude", "codex"])
def test_doctor_rejects_missing_mcp_executable(healthy_doctor, monkeypatch, tmp_path, client):
    executable = str(tmp_path / "removed-python-version/bin/uv")
    if client == "claude":
        (tmp_path / ".claude.json").write_text(json.dumps({
            "mcpServers": {"clync": {"command": executable, "args": []}}}))
    else:
        config = tmp_path / ".codex/config.toml"
        config.parent.mkdir()
        config.write_text(f'[mcp_servers.clync]\ncommand = "{executable}"\nargs = []\n')
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0, stdout="", stderr=""))
    assert clync.cmd_doctor(types.SimpleNamespace(mcp_client=client)) == 1


@pytest.mark.parametrize("state,expected", [
    ("connected", 0), ("failed", 1), ("disabled", 1), ("pending", 1), (None, 1),
])
def test_doctor_checks_selected_opencode_connection(healthy_doctor, monkeypatch, state, expected):
    def run(argv, **kwargs):
        assert argv[0] != "claude"
        if argv[-1].startswith("/api/"):
            assert parse_qs(urlsplit(argv[-1]).query)["location[directory]"] == [str(clync.Path.cwd())]
        servers = [] if state is None else [{"name": clync.MCP_NAME, "status": {"status": state}}]
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({"data": servers}), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert clync.cmd_doctor(types.SimpleNamespace(mcp_client="opencode")) == expected
