"""HNSW behavior of hybrid search (ADR 0005): which legs scan an index, when a small
selection is served exactly instead, that filtered legs still fill their shortlist, and
the explicit rules (ef_search range, sparse nnz cap, pgvector floor) the index relies on.

Real Postgres (throwaway db), embedder mocked: the routing and SQL are under test, not
BGE-M3 or recall on the real corpus (that is `experiments/hnsw_bench`)."""
from __future__ import annotations

import clync
import pytest

from tests.test_search import _seed, _seed_cc


def _hnsw_scans() -> dict[str, int]:
    """idx_scan of every HNSW index on chunks. The scanning backend flushes its counters
    when its connection closes, which `hybrid_search` does on return."""
    with clync.connect_pg() as con:
        return {r["indexrelname"]: r["idx_scan"] for r in con.execute(
            "SELECT s.indexrelname, s.idx_scan FROM pg_stat_user_indexes s "
            "JOIN pg_indexes i ON i.indexname = s.indexrelname AND i.tablename = s.relname "
            "WHERE s.relname = 'chunks' AND i.indexdef ILIKE '%%USING hnsw%%'").fetchall()}


def _drop_hnsw_indexes() -> None:
    with clync.connect_pg() as con:
        for name in _hnsw_scans():
            con.execute(f"DROP INDEX {name}")
        con.commit()


def _corpus(n_repos: int = 3, per_repo: int = 4) -> None:
    for r in range(n_repos):
        for i in range(per_repo):
            _seed_cc(f"s{r}-{i}", f"session {r} {i}", f"repo{r}",
                     [f"shared topic words plus repo{r} detail {i}", f"unrelated filler {i}"])


def test_the_dense_and_sparse_legs_each_have_an_hnsw_index(pg_test_db):
    assert len(_hnsw_scans()) == 2


def test_a_large_selection_scans_both_hnsw_indexes(pg_test_db, mock_embed, monkeypatch):
    search = pg_test_db
    _corpus()
    search.build_index()
    monkeypatch.setattr(search, "EXACT_SELECTION_MAX", 0)    # every selection counts as large
    before = _hnsw_scans()
    assert search.hybrid_search("shared topic words", topk=5)
    after = _hnsw_scans()
    assert all(after[name] > before[name] for name in before)


def test_a_small_selection_is_exact_and_leaves_the_indexes_alone(pg_test_db, mock_embed):
    search = pg_test_db
    _corpus()
    search.build_index()
    before = _hnsw_scans()
    with_index = search.hybrid_search("shared topic words", topk=20, repo="repo1")
    assert _hnsw_scans() == before
    _drop_hnsw_indexes()
    without_index = search.hybrid_search("shared topic words", topk=20, repo="repo1")
    assert [(r["unit_id"], r["msg_idx"], r["chunk_idx"]) for r in with_index] == [
        (r["unit_id"], r["msg_idx"], r["chunk_idx"]) for r in without_index
    ]
    assert [r["score"] for r in with_index] == pytest.approx(
        [r["score"] for r in without_index], rel=1e-9
    )


def test_a_filtered_leg_still_fills_its_shortlist(pg_test_db, mock_embed, monkeypatch):
    """An HNSW scan returns about ef_search rows then the filter discards most: a
    selection holding enough rows must still yield a full shortlist per leg via HNSW."""
    search = pg_test_db
    for i in range(300):
        _seed_cc(f"u{i}", f"session {i}", "wanted" if i % 5 == 0 else "other",
                 [f"common words here number {i}"])
    search.build_index()
    monkeypatch.setattr(search, "EXACT_SELECTION_MAX", 0)
    monkeypatch.setattr(search, "HNSW_EF_SEARCH", search.PRE_TOPK)   # the narrowest scan allowed
    before = _hnsw_scans()
    hits = search.hybrid_search("common words here", topk=100, repo="wanted")
    after = _hnsw_scans()
    assert all(after[name] > before[name] for name in before)
    assert len(hits) >= search.PRE_TOPK
    assert {h["unit_id"] for h in hits} <= {f"u{i}" for i in range(0, 300, 5)}


@pytest.mark.parametrize("side", ["below", "above"])
def test_an_ef_search_outside_the_range_pgvector_honours_is_rejected(
        pg_test_db, mock_embed, monkeypatch, side):
    search = pg_test_db
    bad = (search.PRE_TOPK - 1 if side == "below"
           else search.PGVECTOR_EF_SEARCH_MAX + 1)
    with pytest.raises(ValueError, match="ef_search"):
        search.check_ef_search(bad)
    _corpus()
    search.build_index()
    monkeypatch.setattr(search, "HNSW_EF_SEARCH", bad)
    with pytest.raises(ValueError, match="ef_search"):
        search.hybrid_search("shared topic words")


def _literal_indices(literal: str) -> set[int]:
    body = literal.split("}", 1)[0].lstrip("{")
    return {int(item.split(":")[0]) for item in body.split(",")}


def test_a_sparse_vector_over_the_index_limit_keeps_its_heaviest_weights(pg_test_db, capsys):
    search = pg_test_db
    extra = 50
    n = search.SPARSE_MAX_NNZ + extra
    weights = {token: float(token + 1) for token in range(n)}    # heavier = higher token id
    kept = _literal_indices(search._sparse_literal(weights))     # 1-based in the literal
    assert kept == {token + 1 for token in range(extra, n)}
    assert str(extra) in capsys.readouterr().err                  # dropped count is reported


def test_a_sparse_vector_at_the_limit_is_stored_whole(pg_test_db, capsys):
    search = pg_test_db
    weights = {token: 1.0 for token in range(search.SPARSE_MAX_NNZ)}
    assert len(_literal_indices(search._sparse_literal(weights))) == search.SPARSE_MAX_NNZ
    assert capsys.readouterr().err == ""


def test_indexing_a_chunk_wider_than_the_index_limit_succeeds_capped(
        pg_test_db, mock_embed, monkeypatch):
    search = pg_test_db
    plain = search._embed
    wide = {token: 1.0 + token / 1e4 for token in range(search.SPARSE_MAX_NNZ + 200)}

    def embed(texts, max_length):
        dense, sparse = plain(texts, max_length)
        return dense, [wide if "wide" in t else s for t, s in zip(texts, sparse)]

    monkeypatch.setattr(search, "_embed", embed)
    _seed("W", "Wide", "2026-01-01T00:00:00+00:00", ["a wide chunk"])
    search.build_index()
    with clync.connect_pg() as con:
        stored = con.execute("SELECT sparse::text AS s FROM chunks").fetchone()["s"]
    assert len(_literal_indices(stored)) == search.SPARSE_MAX_NNZ


def test_an_outdated_pgvector_is_refused_loudly(pg_test_db, monkeypatch):
    """Older pgvector lacks iterative HNSW scans and must be rejected."""
    search = pg_test_db
    monkeypatch.setattr(search, "PGVECTOR_MIN_VERSION", (99, 0, 0))
    with pytest.raises(RuntimeError, match="pgvector"):
        search.ensure_index_schema()


def test_pgvector_below_vacuum_fix_warns_on_build_and_status(pg_test_db, monkeypatch, capsys):
    search = pg_test_db
    monkeypatch.setattr(search, "PGVECTOR_SAFE_VACUUM_VERSION", (99, 0, 0))
    # Idempotent check stays quiet when HNSW indexes already exist.
    search.ensure_index_schema()
    assert capsys.readouterr().err == ""
    # Creating a missing HNSW index emits the vacuum warning once, and doctor sees it.
    _drop_hnsw_indexes()
    search.ensure_index_schema()
    assert "99.0.0" in capsys.readouterr().err
    assert "99.0.0" in (search.index_status()["pgvector_warning"] or "")
