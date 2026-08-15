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
    """Replace BGE-M3 with a deterministic hashed bag-of-words embedder.

    It must DISCRIMINATE, or every relevance assertion built on it is vacuous. The
    previous stub returned one fixed dense vector and one fixed sparse weight for
    every text, so all similarities were equal and result *order* was whatever the
    plan happened to produce — a test asserting "the right unit ranked first" proved
    nothing. Here each word contributes to a hashed dimension, so texts sharing
    words are genuinely nearer each other and a query ranks matching chunks first,
    while staying model-free and fast (no BGE-M3 load)."""
    import hashlib
    import math
    import re

    import search

    def _fake(texts, max_length):
        dense, sparse = [], []
        for t in texts:
            vec = [0.0] * search.DENSE_DIM
            weights: dict[int, float] = {}
            for tok in re.findall(r"\w+", (t or "").lower()):
                h = int(hashlib.sha1(tok.encode()).hexdigest(), 16)
                vec[h % search.DENSE_DIM] += 1.0
                key = h % search.SPARSE_DIM
                weights[key] = weights.get(key, 0.0) + 1.0
            norm = math.sqrt(sum(x * x for x in vec))
            if not norm:            # wordless text: a fixed non-zero unit vector,
                vec[0], norm = 1.0, 1.0   # never the zero vector (cosine is undefined)
            dense.append([x / norm for x in vec])
            sparse.append(weights or {0: 1.0})
        return dense, sparse

    monkeypatch.setattr(search, "_embed", _fake)


@pytest.fixture(autouse=True)
def _sandbox_codex_root(tmp_path_factory, monkeypatch):
    """Tests must NEVER read the user's real `~/.codex`. Point the Codex ingest
    root at an isolated empty dir by default (an existing-empty root is a valid
    empty corpus -> ingest_codex is a no-op); a test that exercises Codex ingest
    overrides `codex.CODEX_ROOT` itself. (Claude Code ingest tests already redirect
    `cc.CC_ROOT` per-test by the same convention.)"""
    import codex
    monkeypatch.setattr(codex, "CODEX_ROOT", tmp_path_factory.mktemp("codex_root_sandbox"))
