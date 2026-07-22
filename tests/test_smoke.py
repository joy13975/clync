"""Smoke: the entry points import (with heavy deps staying lazy) and the CLI runs."""
from __future__ import annotations

import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(__file__))


def test_entry_points_import_without_pulling_heavy_deps():
    # The entry points must import WITHOUT eagerly loading the ML / DB heavy deps —
    # that laziness is what keeps MCP-server startup fast (it only pays for the
    # embedder on the first search). Checked in a subprocess so it holds regardless
    # of what other tests loaded into this process.
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
    assert all(s in r.stdout for s in ("search", "sync", "sync-app", "sync-cc"))
