"""One real end-to-end retrieval with the actual BGE-M3 model (no mocking).
Slow (~model load) and opt-in:  uv run --extra search pytest -m slow"""
from __future__ import annotations

import pytest

from tests.test_search import T1, _seed


@pytest.mark.slow
def test_real_model_retrieves_the_right_conversation(sqlite_db, pg_test_db):
    search = pg_test_db     # NOTE: no mock_embed -> real embeddings
    _seed("boto3", "Boto3 pagination", T1,
          ["give me a function that paginates boto3 results generically for all endpoints"])
    _seed("greet", "Japanese greeting", T1, ["おはようございます、今日は良い天気ですね"])
    _seed("memo", "Role change memo", T1,
          ["please proof-read this memo about my transition off the team lead role"])
    search.build_index(full=True)

    hits = search.hybrid_search("aws sdk pagination helper", topk=3)
    assert hits[0]["conv_uuid"] == "boto3"          # correct anchor ranked first
