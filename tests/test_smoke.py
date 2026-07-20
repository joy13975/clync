"""Smoke: the entry points import and the CLI runs without the search extra."""
from __future__ import annotations

import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(__file__))


def test_entry_points_import_without_pulling_heavy_deps():
    # Core-only safety: the entry points must import WITHOUT eagerly loading the
    # search extra's heavy deps — that laziness is what lets a core-only install
    # run sync/MCP. Checked in a subprocess so it holds regardless of whether the
    # extra is installed here or what other tests loaded into this process.
    probe = (
        "import clync, mcp_server, search, sys;"
        "heavy = [m for m in ('FlagEmbedding', 'torch', 'psycopg') if m in sys.modules];"
        "assert not heavy, f'eagerly imported: {heavy}'"
    )
    r = subprocess.run([sys.executable, "-c", probe], cwd=_REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_cli_help_runs():
    r = subprocess.run([sys.executable, "clync.py", "--help"],
                       cwd=_REPO, capture_output=True, text=True)
    assert r.returncode == 0
    assert "search" in r.stdout and "sync" in r.stdout
