"""Shared fixtures. Tests never touch the user's real DB or the real `clync`
Postgres database: SQLite is redirected to a tmp file, and Postgres work runs in
a throwaway database inside the (already-running) contained cluster."""
from __future__ import annotations

import pytest

import clync


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="run tests marked `slow` (real model / heavy integration)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def sqlite_db(tmp_path, monkeypatch):
    """Redirect clync's SQLite store to a fresh temp file (schema auto-created)."""
    db = tmp_path / "history.db"
    monkeypatch.setattr(clync, "DB_PATH", db)
    return db


@pytest.fixture
def pg_test_db(monkeypatch):
    """A throwaway Postgres database in the contained cluster. Skips if the search
    extra / cluster isn't available. Reuses the running cluster; drops the db after."""
    search = pytest.importorskip("search")
    if not search.available():
        pytest.skip("search extra not installed")
    monkeypatch.setattr(search, "PG_DB", "clync_pytest")
    search.ensure_cluster()          # creates clync_pytest + schema in the live cluster
    yield search
    # teardown: drop the temp db (FORCE terminates any lingering backends, PG13+)
    import psycopg
    with psycopg.connect(host="localhost", port=search.PG_PORT,
                         dbname="postgres", user=search.PG_USER, autocommit=True) as c:
        c.execute("DROP DATABASE IF EXISTS clync_pytest WITH (FORCE)")


@pytest.fixture
def mock_embed(monkeypatch):
    """Replace BGE-M3 with a deterministic stub: one fixed dense vector + one
    positive sparse weight per text. Enough to exercise the SQL/fusion/watermark
    logic without loading the real model. Returns nothing; just patches search._embed."""
    search = pytest.importorskip("search")

    def _fake(texts, max_length):
        dense = [[1.0] + [0.0] * (search.DENSE_DIM - 1) for _ in texts]
        sparse = [{0: 1.0} for _ in texts]   # token 0 -> sparsevec index 1
        return dense, sparse

    monkeypatch.setattr(search, "_embed", _fake)
