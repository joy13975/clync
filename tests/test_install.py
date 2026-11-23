"""Launchd installation emits a valid property-list with both source profiles."""
from __future__ import annotations

import plistlib
import socket
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from uuid import uuid4

import clync
import pytest


@pytest.fixture
def isolated_install(tmp_path, monkeypatch):
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]
    cluster = tempfile.TemporaryDirectory(prefix="clync-")
    pg_dir = Path(cluster.name)
    label = f"io.clync.test.{uuid4().hex}"
    for name, value in {
        "PG_DIR": pg_dir,
        "PG_DATA": pg_dir / "data",
        "PG_LOG": pg_dir / "postmaster.log",
        "PG_PORT": port,
        "PG_DB": "clync_lifecycle_test",
        "LAUNCHD_LABEL": label + ".sync",
        "PLIST_PATH": tmp_path / "sync.plist",
        "PG_LAUNCHD_LABEL": label + ".postgres",
        "PG_PLIST_PATH": tmp_path / "postgres.plist",
        "SCHED_LOG": tmp_path / "scheduled.log",
    }.items():
        monkeypatch.setattr(clync, name, value, raising=False)
    try:
        yield types.SimpleNamespace(
            claude_profile="ClaudeProfile", chatgpt_profile="ChatGPTProfile", at="23:59", no_dream=True)
    finally:
        cluster.cleanup()


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS launchd")
def test_installed_store_starts_at_login_and_restarts_after_exit(isolated_install, capfd):
    if not (clync.PG_BIN / "pg_ctl").exists():
        pytest.skip("requires PG17")
    domain = f"gui/{clync.os.getuid()}"
    service = f"{domain}/{clync.PG_LAUNCHD_LABEL}"
    try:
        clync.ensure_cluster()
        assert capfd.readouterr().out == ""
        assert clync.cmd_install(isolated_install) == 0
        managed = subprocess.run(["launchctl", "print", service], capture_output=True, text=True)
        assert managed.returncode == 0, managed.stderr
        with clync.connect_pg() as con:
            con.execute("CREATE TABLE lifecycle_marker (value text)")
            con.execute("INSERT INTO lifecycle_marker VALUES ('retained')")

        def wait_for_store(previous_pid):
            import psycopg
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    with clync.connect_pg() as con:
                        assert con.execute("SELECT value FROM lifecycle_marker").fetchone() == {
                            "value": "retained"}
                        pid = (clync.PG_DATA / "postmaster.pid").read_text().splitlines()[0]
                        if pid != previous_pid:
                            return pid
                except (psycopg.OperationalError, FileNotFoundError):
                    pass
                time.sleep(0.1)
            pytest.fail(clync.PG_LOG.read_text()[-4000:])

        def unload_store():
            subprocess.run(["launchctl", "bootout", service], check=True)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                loaded = subprocess.run(
                    ["launchctl", "print", service], capture_output=True).returncode == 0
                if not loaded and not clync.cluster_running():
                    return
                time.sleep(0.1)
            pytest.fail("launchd did not unload the disposable database")

        pid = (clync.PG_DATA / "postmaster.pid").read_text().splitlines()[0]
        assert clync.cmd_install(isolated_install) == 0
        assert (clync.PG_DATA / "postmaster.pid").read_text().splitlines()[0] == pid
        unload_store()
        subprocess.run(["launchctl", "bootstrap", domain, str(clync.PG_PLIST_PATH)], check=True)
        pid = wait_for_store(pid)
        subprocess.run(["launchctl", "kill", "SIGINT", service], check=True)
        wait_for_store(pid)

        unload_store()
        assert clync.cmd_install(isolated_install) == 0
        with clync.connect_pg() as con:
            assert con.execute("SELECT value FROM lifecycle_marker").fetchone() == {
                "value": "retained"}

        assert clync.cmd_uninstall(isolated_install) == 0
        assert not clync.cluster_running()
        assert not clync.PG_PLIST_PATH.exists()
        assert (clync.PG_DATA / "PG_VERSION").exists()
    finally:
        for label in (clync.LAUNCHD_LABEL, clync.PG_LAUNCHD_LABEL):
            subprocess.run(["launchctl", "bootout", f"{domain}/{label}"], capture_output=True)
        clync.stop_cluster()


def test_cmd_install_writes_parseable_split_profile_plist(tmp_path, monkeypatch):
    plist_path = tmp_path / "io.clync.sync.plist"
    log_path = tmp_path / "scheduled.log"
    monkeypatch.setattr(clync, "PLIST_PATH", plist_path)
    monkeypatch.setattr(clync, "SCHED_LOG", log_path)
    monkeypatch.setattr(clync, "install_cluster_service", lambda: None, raising=False)

    def fake_run(*args, **kwargs):
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(clync.subprocess, "run", fake_run)

    rc = clync.cmd_install(types.SimpleNamespace(
        claude_profile="ClaudeProfile", chatgpt_profile="ChatGPTProfile", at="09:00", no_dream=False))

    assert rc == 0
    document = plistlib.loads(plist_path.read_bytes())
    assert document["EnvironmentVariables"] == {
        "CLYNC_CLAUDE_PROFILE": "ClaudeProfile",
        "CLYNC_CHATGPT_PROFILE": "ChatGPTProfile",
    }
    assert document["ProgramArguments"][-1] == (
        f"cd {clync.REPO_DIR} && exec uv run python clync.py scheduled")


def test_cli_launcher_uses_its_own_checkout_through_a_symlink(tmp_path):
    import json
    import os

    checkout = tmp_path / 'checkout with spaces'
    checkout.mkdir()
    launcher = checkout / 'clync'
    launcher.write_bytes((clync.REPO_DIR / 'clync').read_bytes())
    launcher.chmod(0o755)
    commands = tmp_path / 'bin'
    commands.mkdir()
    link = commands / 'clync'
    link.symlink_to(launcher)
    uv = commands / 'uv'
    uv.write_text(f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    uv.chmod(0o755)
    env = {**os.environ, 'PATH': str(commands) + os.pathsep + os.environ['PATH']}
    result = subprocess.run([str(link), 'status'], capture_output=True, text=True,
                            check=True, env=env)
    args = json.loads(result.stdout)
    assert Path(args[args.index('--project') + 1]) == checkout
    assert Path(args[-2]) == checkout / 'clync.py'
    assert args[-1] == 'status'
