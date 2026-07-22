"""Shared fixtures. Tests never touch the user's real store: Postgres work runs in
a throwaway database (`clync_pytest`) inside the already-running contained cluster,
and the Claude Code root is redirected to a tmp dir."""
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


def _drop_pytest_db():
    import psycopg
    with psycopg.connect(host="localhost", port=clync.PG_PORT,
                         dbname="postgres", user=clync.PG_USER, autocommit=True) as c:
        c.execute("DROP DATABASE IF EXISTS clync_pytest WITH (FORCE)")


@pytest.fixture
def pg_test_db(monkeypatch):
    """A throwaway Postgres database in the contained cluster. Skips if the PG17 +
    pgvector toolchain isn't present. Fresh per test; dropped after. Yields the
    `search` module (its heavy deps load lazily on first embed)."""
    import search
    if not search.available():
        pytest.skip("PG17 + pgvector toolchain not available")
    monkeypatch.setattr(clync, "PG_DB", "clync_pytest")
    _drop_pytest_db()                # fresh slate even if a prior run left one behind
    clync.ensure_cluster()           # creates clync_pytest + full schema in the cluster
    yield search
    _drop_pytest_db()


@pytest.fixture
def mock_embed(monkeypatch):
    """Replace BGE-M3 with a deterministic stub: one fixed dense vector + one
    positive sparse weight per text. Exercises the SQL/fusion/watermark logic
    without loading the real model."""
    import search

    def _fake(texts, max_length):
        dense = [[1.0] + [0.0] * (search.DENSE_DIM - 1) for _ in texts]
        sparse = [{0: 1.0} for _ in texts]   # token 0 -> sparsevec index 1
        return dense, sparse

    monkeypatch.setattr(search, "_embed", _fake)
