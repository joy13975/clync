"""Behavior tests for Phase 2 clync-share: lite cipher, client relay, and tamper alerting."""
from __future__ import annotations

import asyncio
import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from cryptography.exceptions import InvalidTag
from mcp.server.fastmcp import FastMCP

from clync_cipher import ClyncCipher, make_citation_handle

ROOT = Path(__file__).resolve().parents[1]
SHARE_CLIENT_DIR = str(ROOT / "share-client")
CLOUD_DIR = str(ROOT / "cloud")
for _d in (SHARE_CLIENT_DIR, CLOUD_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import clync_share_client  # noqa: E402
import tamper_alert  # noqa: E402


def _key(seed: int = 7) -> bytes:
    return bytes((seed + i * 13) % 256 for i in range(32))


def test_aes_gcm_roundtrip_random_nonces_and_aad_binding():
    cipher = ClyncCipher(_key(1), dense_dim=16, sparse_dim=64)
    secret = "private transcript line with secrets"
    ct1 = cipher.encrypt_text(secret, aad="chunks:c1")
    ct2 = cipher.encrypt_text(secret, aad="chunks:c1")
    assert ct1 != ct2
    assert secret not in ct1
    assert cipher.decrypt_text(ct1, aad="chunks:c1") == secret
    with pytest.raises(InvalidTag):
        cipher.decrypt_text(ct1, aad="chunks:c2")


def test_dense_rotation_preserves_cosine_similarity_while_hiding_coordinates():
    cipher = ClyncCipher(_key(2), dense_dim=64, sparse_dim=128)
    rng = np.random.default_rng(123)
    a = rng.standard_normal(64).astype(np.float32)
    b = rng.standard_normal(64).astype(np.float32)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    ra = np.asarray(cipher.rotate_dense(a), dtype=np.float32)
    rb = np.asarray(cipher.rotate_dense(b), dtype=np.float32)
    assert not np.allclose(a, ra)
    assert float(ra @ rb) == pytest.approx(float(a @ b), abs=1e-5)
    assert float(np.linalg.norm(ra)) == pytest.approx(1.0, abs=1e-5)


def test_sparse_permutation_preserves_inner_product_and_hides_token_ids():
    cipher = ClyncCipher(_key(3), dense_dim=16, sparse_dim=1000)
    q = {10: 0.7, 42: 1.5, 512: 0.4}
    d = {10: 0.9, 42: 0.5, 999: 2.0}
    pq = cipher.permute_sparse(q)
    pd = cipher.permute_sparse(d)
    assert set(pq) != set(q)
    plain_ip = sum(q[k] * d[k] for k in set(q) & set(d))
    perm_ip = sum(pq[k] * pd[k] for k in set(pq) & set(pd))
    assert perm_ip == pytest.approx(plain_ip, abs=1e-6)


def test_hmac_facet_is_deterministic_domain_separated_and_opaque():
    cipher = ClyncCipher(_key(4), dense_dim=16, sparse_dim=64)
    val = "my-secret-repo"
    h1 = cipher.hmac_facet("repo", val)
    h2 = cipher.hmac_facet("repo", val)
    h_other = cipher.hmac_facet("project", val)
    assert h1 == h2
    assert h1 != h_other
    assert val not in h1


def test_citation_handles_are_random_and_distinct():
    handles = {make_citation_handle("alice") for _ in range(20)}
    assert len(handles) == 20
    assert all(h.startswith("clync-alice-") for h in handles)


def test_connect_and_disconnect_persist_endpoint_and_enforce_owner_default(tmp_path, monkeypatch):
    monkeypatch.setenv("CLYNC_DATA_HOME", str(tmp_path))
    arn = "arn:aws:lambda:us-east-1:111122223333:function:clync-share-alice"

    def fake_invoke(fn_arn, action, params, *, profile=None, region=None):
        assert fn_arn == arn
        return {
            "owner": "alice",
            "modes": ["ask"],
            "tools": ["ask_alice"],
            "caller_arn": "arn:aws:sts::111122223333:assumed-role/TeamRole/bob",
        }

    monkeypatch.setattr(clync_share_client, "invoke_share_lambda", fake_invoke)

    with pytest.raises(RuntimeError):
        clync_share_client.connect_endpoint(arn, set_default=True)

    entry = clync_share_client.connect_endpoint(arn, profile="team-prof")
    assert entry["alias"] == "alice"
    assert clync_share_client.load_shares()["endpoints"]["alice"]["profile"] == "team-prof"
    assert clync_share_client.disconnect_endpoint("alice") is True
    assert clync_share_client.load_shares()["endpoints"] == {}


def test_register_share_tools_relays_ask_and_gates_owner_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("CLYNC_DATA_HOME", str(tmp_path))
    seen_questions: list[str] = []

    def fake_invoke(fn_arn, action, params, *, profile=None, region=None):
        seen_questions.extend(v for v in params.values() if isinstance(v, str))
        return {
            "answer": "Synthesized answer [clync-alice-deadbeef0001].",
            "citations": [{"handle": "clync-alice-deadbeef0001", "source_type": "claude_code"}],
        }

    monkeypatch.setattr(clync_share_client, "invoke_share_lambda", fake_invoke)
    ask_only = FastMCP("test-ask")
    names = clync_share_client.register_share_tools(
        ask_only,
        endpoints={
            "alice": {
                "alias": "alice",
                "arn": "arn:aws:lambda:us-east-1:111122223333:function:clync-share-alice",
                "region": "us-east-1",
                "owner": "alice",
                "modes": ["ask"],
            }
        },
    )
    assert names == ["ask_alice"]
    out = asyncio.run(ask_only.call_tool("ask_alice", {"question": "What caching rule was picked?"}))
    rendered = "".join(c.text for c in out[0] if hasattr(c, "text")) if isinstance(out, tuple) else "".join(c.text for c in out if hasattr(c, "text"))
    assert "clync-alice-deadbeef0001" in rendered
    assert seen_questions == ["What caching rule was picked?"]


def test_tamper_alert_ignores_expected_decrypt_and_flags_unauthorized_events(monkeypatch):
    monkeypatch.setattr(tamper_alert, "OWNER", "alice")
    monkeypatch.setattr(tamper_alert, "WATCHED_KMS_KEY_ID", "11111111-2222-3333-4444-555555555555")
    monkeypatch.setattr(tamper_alert, "ALLOWED_KMS_ROLE_ARN", "arn:aws:iam::111122223333:role/clync-share-alice-role")
    monkeypatch.setattr(tamper_alert, "ALLOWED_OWNER_USER_IDS", {"AROAAA:alice"})
    monkeypatch.setattr(tamper_alert, "WATCHED_LAMBDA_NAME", "clync-share-alice")

    expected_decrypt = {
        "detail": {
            "eventName": "Decrypt",
            "requestParameters": {"keyId": "11111111-2222-3333-4444-555555555555"},
            "userIdentity": {
                "principalId": "AROALAMBDA:clync-share-alice",
                "sessionContext": {"sessionIssuer": {"arn": "arn:aws:iam::111122223333:role/clync-share-alice-role"}},
            },
        }
    }
    assert tamper_alert.lambda_handler(expected_decrypt, None)["alerted"] is False

    rogue_decrypt = {
        "detail": {
            "eventName": "Decrypt",
            "requestParameters": {"keyId": "11111111-2222-3333-4444-555555555555"},
            "userIdentity": {"principalId": "AROAROGUE:mallory", "arn": "arn:aws:sts::111122223333:assumed-role/Admin/mallory"},
        }
    }
    res = tamper_alert.lambda_handler(rogue_decrypt, None)
    assert res["alerted"] is True
    assert res["record"]["actor_arn"].endswith("/mallory")

    code_swap = {
        "detail": {
            "eventName": "UpdateFunctionCode20150331v2",
            "requestParameters": {"functionName": "clync-share-alice"},
            "userIdentity": {"principalId": "AROAROGUE:mallory", "arn": "arn:aws:sts::111122223333:assumed-role/PU/mallory"},
        }
    }
    assert tamper_alert.lambda_handler(code_swap, None)["alerted"] is True

    other_lambda = {
        "detail": {
            "eventName": "UpdateFunctionCode20150331v2",
            "requestParameters": {"functionName": "unrelated-service"},
            "userIdentity": {"arn": "arn:aws:sts::111122223333:assumed-role/PU/mallory"},
        }
    }
    assert tamper_alert.lambda_handler(other_lambda, None)["alerted"] is False


def test_share_client_imports_without_heavy_dependencies():
    mod = importlib.reload(clync_share_client)
    assert mod is not None
    assert "torch" not in sys.modules
    assert "FlagEmbedding" not in sys.modules


def test_lambda_handler_gates_owner_actions_and_rejects_bad_sts_host(monkeypatch):
    monkeypatch.setenv("CLYNC_OWNER", "alice")
    monkeypatch.setenv("CLYNC_DB_SCHEMA", "clync_alice")
    monkeypatch.setenv("CLYNC_DB_HOST", "localhost")
    monkeypatch.setenv("CLYNC_DB_USER", "clync_alice")
    monkeypatch.setenv("CLYNC_ALLOWED_ACCOUNT_ID", "111122223333")
    monkeypatch.setenv("CLYNC_OWNER_USER_IDS", "AROAOWNER:alice")
    monkeypatch.setenv("CLYNC_ALLOWED_ROLE_PREFIXES", "arn:aws:sts::111122223333:assumed-role/TeamPU/")
    monkeypatch.setenv("CLYNC_DATA_KEY_CT_B64", "AAAA")
    monkeypatch.setenv("CLYNC_KMS_KEY_ID", "arn:aws:kms:us-east-1:111122223333:key/abc")

    import handler

    importlib.reload(handler)

    # Rejects SSRF-style STS URL targeting a non-AWS host or non-default port before making any network request.
    for bad_url in (
        "https://evil.example/?Action=GetCallerIdentity&Version=2011-06-15",
        "https://sts.us-east-1.amazonaws.com:8443/?Action=GetCallerIdentity&Version=2011-06-15",
    ):
        bad_res = handler.lambda_handler({"action": "handshake", "auth": {"url": bad_url, "headers": {}}}, None)
        assert bad_res.get(handler.ERROR_KEY)

    # Non-owner PowerUser caller gets ask mode only; every owner action is refused.
    monkeypatch.setattr(
        handler,
        "_verify_caller",
        lambda _auth: {
            "account": "111122223333",
            "arn": "arn:aws:sts::111122223333:assumed-role/TeamPU/bob",
            "user_id": "AROAPU:bob",
            "is_owner": False,
        },
    )
    hs = handler.lambda_handler({"action": "handshake", "auth": {}}, None)
    assert hs["modes"] == ["ask"]
    assert hs["tools"] == ["ask_alice"]
    for act in ("search_history", "search_transcripts", "get_conversation", "resolve_citation", "ingest"):
        denied = handler.lambda_handler({"action": act, "auth": {}}, None)
        assert denied.get(handler.ERROR_KEY)


def test_push_delta_watermark_skips_unchanged_and_honors_full():
    import deploy

    t1 = datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 11, 21, 10, 0, tzinfo=timezone.utc)
    units = [
        {"unit_id": "u-same", "updated_at": t1},
        {"unit_id": "u-updated", "updated_at": t2},
        {"unit_id": "u-new", "updated_at": t2},
    ]
    watermark = {"u-same": t1.isoformat(), "u-updated": t1.isoformat()}
    delta = deploy._filter_changed_units(units, watermark, full=False)
    assert [u["unit_id"] for u in delta] == ["u-updated", "u-new"]
    assert [u["unit_id"] for u in deploy._filter_changed_units(units, watermark, full=True)] == [
        "u-same",
        "u-updated",
        "u-new",
    ]


