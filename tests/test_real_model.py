"""One real end-to-end retrieval with the actual BGE-M3 model (no mocking).
Slow (~model load) and opt-in:  uv run pytest -m slow --run-slow"""
from __future__ import annotations

import pytest

from tests.test_search import T1, _seed


@pytest.mark.slow
def test_real_model_retrieves_the_right_conversation(pg_test_db):
    search = pg_test_db     # NOTE: no mock_embed -> real embeddings
    _seed("boto3", "Boto3 pagination", T1,
          ["give me a function that paginates boto3 results generically for all endpoints"])
    _seed("greet", "Japanese greeting", T1, ["おはようございます、今日は良い天気ですね"])
    _seed("memo", "Role change memo", T1,
          ["please proof-read this memo about my transition off the team lead role"])
    search.build_index(full=True)

    hits = search.hybrid_search("aws sdk pagination helper", topk=3)
    assert hits[0]["unit_id"] == "boto3"            # correct anchor ranked first


@pytest.mark.slow
def test_real_model_daily_chain_embeds_both_sources(pg_test_db, tmp_path, monkeypatch):
    """The integrated daily chain (`_sync_all`) with the REAL model: one run syncs
    a claude.ai chat + a local Claude Code session and auto-embeds both, and each
    is retrievable by MEANING (not just presence) from the freshly built index."""
    import cc
    import clync
    from tests.test_cc import _cli_events, _write_session
    from tests.test_sync import _FakeClient, _conv, _wire

    search = pg_test_db     # no mock_embed -> real embeddings end to end
    per_org = {"org1": {"convs": [_conv(
        "c1", "2026-02-01T00:00:00+00:00",
        "how do I paginate boto3 results generically across all aws endpoints")]}}
    _wire(monkeypatch, _FakeClient(per_org), [{"uuid": "org1", "name": "Org2"}])
    _write_session(tmp_path / "proj" / "sess-1.jsonl", _cli_events())
    monkeypatch.setattr(cc, "CC_ROOT", tmp_path)

    assert clync._sync_all("prof", None, full=True, download_files=False,
                           do_index=True, notify_fail=False) is None

    # semantic retrieval into each source from the one auto-embedded run
    app = search.hybrid_search("aws sdk pagination helper", source="claude_ai", topk=3)
    assert app and app[0]["unit_id"] == "c1"
    cc_hits = {h["unit_id"] for h in search.hybrid_search("design session", source="claude_code", topk=5)}
    assert "sess-1" in cc_hits
