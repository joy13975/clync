"""ChatGPT ingest — pure linearization of the mapping tree (no DB) + the network
ingest reconcile against a fake client (real Postgres, no network)."""
from __future__ import annotations

import chatgpt
import clync
import pytest


# --------------------------------------------------------------------------- #
# Synthetic mapping-tree builders
# --------------------------------------------------------------------------- #
def _node(msg):
    return {"message": msg}


def _mapping(messages: list[dict | None]) -> dict:
    """Chain the given messages into a linear mapping tree (root -> ... -> leaf)
    and return {mapping, current_node}. A None entry is a message-less node (a
    real ChatGPT root/placeholder), which linearize must skip."""
    mapping, prev = {}, None
    ids = [f"n{i}" for i in range(len(messages))]
    for i, (nid, msg) in enumerate(zip(ids, messages)):
        mapping[nid] = {
            "id": nid, "message": ({"id": f"m{i}", **msg} if msg else None),
            "parent": prev, "children": [ids[i + 1]] if i + 1 < len(ids) else []}
        prev = nid
    return {"mapping": mapping, "current_node": ids[-1]}


def _user(text, ct="text"):
    parts = text if isinstance(text, list) else [text]
    return {"author": {"role": "user"}, "content": {"content_type": ct, "parts": parts},
            "create_time": 1000.0}


def _asst_text(text, model=None):
    m = {"author": {"role": "assistant"},
         "content": {"content_type": "text", "parts": [text]}, "create_time": 1001.0}
    if model:
        m["metadata"] = {"model_slug": model}
    return m


def _asst_thoughts(steps):
    return {"author": {"role": "assistant"},
            "content": {"content_type": "thoughts",
                        "thoughts": [{"summary": "", "content": s} for s in steps]},
            "create_time": 1002.0}


def _asst_recap(text):
    return {"author": {"role": "assistant"},
            "content": {"content_type": "reasoning_recap", "content": text},
            "create_time": 1002.5}


def _asst_code(code):
    return {"author": {"role": "assistant"},
            "content": {"content_type": "code", "text": code}, "create_time": 1003.0}


def _tool(text):
    return {"author": {"role": "tool"},
            "content": {"content_type": "execution_output", "text": text},
            "create_time": 1004.0}


# --------------------------------------------------------------------------- #
# Timestamp normalization (the two API surfaces disagree on format)
# --------------------------------------------------------------------------- #
def test_iso_accepts_both_list_iso_string_and_detail_epoch():
    assert chatgpt._iso("2026-08-15T10:23:55.109384Z") == \
        "2026-08-15T10:23:55.109384+00:00"                 # list surface: ISO string
    assert chatgpt._iso(1755253200.0) == "2025-08-15T10:20:00+00:00"  # detail: epoch
    assert chatgpt._iso(None) is None
    assert chatgpt._iso("   ") is None
    assert chatgpt._iso("not-a-timestamp") is None         # never fabricate a time


# --------------------------------------------------------------------------- #
# Pure content extraction
# --------------------------------------------------------------------------- #
def test_text_parts_drops_non_string_parts():
    content = {"parts": ["look at this", {"content_type": "image_asset_pointer"}, "and this"]}
    assert chatgpt._text_parts(content) == "look at this\nand this"
    assert chatgpt._text_parts({"parts": None}) == ""


def test_thoughts_text_joins_steps_and_reads_recap():
    assert chatgpt._thoughts_text(
        {"thoughts": [{"content": "step one"}, {"summary": "", "content": "step two"}]}
    ) == "step one\n\nstep two"
    assert chatgpt._thoughts_text({"content": "the recap"}) == "the recap"


def test_reuses_cc_truncation_constants():
    # SSOT: chatgpt.py imports cc.py's truncation limits, never redefines them.
    import cc
    assert chatgpt._TITLE_MAX is cc._TITLE_MAX
    assert chatgpt._TOOL_ARG_MAX is cc._TOOL_ARG_MAX
    assert chatgpt._TOOL_RESULT_HEAD is cc._TOOL_RESULT_HEAD


# --------------------------------------------------------------------------- #
# linearize — the whole tree -> a cleaned session
# --------------------------------------------------------------------------- #
def test_linearize_classifies_roles_and_content_types():
    detail = _mapping([
        None,                                  # root placeholder -> skipped
        _user("hello there"),
        _asst_thoughts(["reasoning a", "reasoning b"]),
        _asst_text("the answer", model="gpt-5-6-thinking"),
        _asst_code('{"q":"a search query"}'),
        _tool("tool stdout payload"),
        _asst_text("final reply", model="gpt-5-6-thinking"),
    ])
    detail["default_model_slug"] = "gpt-5-6-instant"
    meta = {"id": "conv-1", "title": "A Real Title",
            "create_time": "2026-08-15T04:00:00Z", "update_time": "2026-08-15T05:00:00Z"}

    conv = chatgpt.linearize(detail, meta)

    assert conv.session_id == "conv-1"
    assert conv.title == "A Real Title"
    assert conv.model == "gpt-5-6-thinking"           # last assistant model_slug wins
    assert conv.created_at == "2026-08-15T04:00:00+00:00"
    assert conv.updated_at == "2026-08-15T05:00:00+00:00"
    assert conv.repo is None and conv.cwd is None and conv.git_branch is None

    rendered = [(m.sender, m.role_detail, m.embed_text) for m in conv.messages]
    assert rendered == [
        ("user", "text", None),
        ("assistant", "reasoning", None),             # thoughts -> reasoning, embedded
        ("assistant", "text", None),
        ("assistant", "tool_use", ""),                # code -> tool_use, NOT embedded
        ("tool", "tool_result", ""),                  # tool output -> NOT embedded
        ("assistant", "text", None),
    ]
    assert conv.messages[1].text == "reasoning a\n\nreasoning b"
    assert [m.idx for m in conv.messages] == [0, 1, 2, 3, 4, 5]   # contiguous, in order


def test_linearize_skips_system_hidden_and_empty():
    detail = _mapping([
        {"author": {"role": "system"}, "content": {"content_type": "text", "parts": ["sys"]}},
        _user("kept prompt"),
        {"author": {"role": "assistant"},                          # hidden from user
         "content": {"content_type": "text", "parts": ["hidden"]},
         "metadata": {"is_visually_hidden_from_conversation": True}},
        {"author": {"role": "assistant"},                          # empty text -> skip
         "content": {"content_type": "text", "parts": [""]}},
        _asst_text("kept answer"),
    ])
    conv = chatgpt.linearize(detail, {"id": "c", "title": "t", "update_time": None})
    assert [(m.sender, m.text) for m in conv.messages] == [
        ("user", "kept prompt"), ("assistant", "kept answer")]


def test_linearize_truncates_tool_use_and_tool_result():
    big_code = "x" * 5000
    big_out = "y" * 5000
    detail = _mapping([_user("q"), _asst_code(big_code), _tool(big_out)])
    conv = chatgpt.linearize(detail, {"id": "c", "title": "t", "update_time": None})
    tool_use = next(m for m in conv.messages if m.role_detail == "tool_use")
    tool_res = next(m for m in conv.messages if m.role_detail == "tool_result")
    assert tool_use.text.endswith("…") and len(tool_use.text) <= chatgpt._TOOL_ARG_MAX + 1
    assert tool_res.text.endswith("…") and len(tool_res.text) <= chatgpt._TOOL_RESULT_HEAD + 1


def test_linearize_title_falls_back_to_first_user_prompt():
    detail = _mapping([_user("what is the airspeed velocity"), _asst_text("depends")])
    conv = chatgpt.linearize(detail, {"id": "c", "title": "", "update_time": None})
    assert conv.title == "what is the airspeed velocity"


def test_linear_chain_is_cycle_safe():
    # A malformed parent cycle must terminate the walk, not hang.
    mapping = {
        "a": {"id": "a", "message": _msg("A"), "parent": "b"},
        "b": {"id": "b", "message": _msg("B"), "parent": "a"},
    }
    chain = chatgpt._linear_chain({"mapping": mapping, "current_node": "a"})
    assert len(chain) <= 2      # terminates; never loops forever


def _msg(text):
    return {"id": "x", "author": {"role": "user"},
            "content": {"content_type": "text", "parts": [text]}}


# --------------------------------------------------------------------------- #
# Network ingest against a fake client (real Postgres, no network)
# --------------------------------------------------------------------------- #
class _FakeCGPTClient:
    def __init__(self, pairs):        # pairs: list[(meta, detail)]
        self._pairs = pairs

    def iter_conversations(self):
        for meta, _ in self._pairs:
            yield meta

    def get_conversation(self, cid):
        for meta, detail in self._pairs:
            if meta["id"] == cid:
                return detail
        raise KeyError(cid)


def _pair(cid, update_time, user_text, answer, title=None, model="gpt-5-6-thinking"):
    detail = _mapping([_user(user_text), _asst_text(answer, model=model)])
    detail["default_model_slug"] = model
    meta = {"id": cid, "title": title or answer[:20],
            "create_time": update_time, "update_time": update_time}
    return (meta, detail)


def test_read_chatgpt_cookies_accepts_chunked_session_token(monkeypatch):
    """next-auth splits an oversized session token into <name>.0/.1 (each Chrome
    cookie capped at ~4 KB); the whole login is present as chunks, not as the
    bare cookie. read_chatgpt_cookies must treat the .0 chunk as proof of login,
    not raise a false 'no login found'."""
    base = chatgpt.SESSION_COOKIE
    chunked = {f"{base}.0": "part0", f"{base}.1": "part1", "cf_clearance": "c"}
    monkeypatch.setattr(clync, "_read_profile_cookies",
                        lambda *a, **k: dict(chunked))
    assert clync.read_chatgpt_cookies("prof") == chunked

    monkeypatch.setattr(clync, "_read_profile_cookies",
                        lambda *a, **k: {"cf_clearance": "c"})
    with pytest.raises(RuntimeError, match="No ChatGPT login"):
        clync.read_chatgpt_cookies("prof")


class _FakeResp:
    def __init__(self, status_code, text='{"items": []}'):
        self.status_code = status_code
        self.text = text
        self.headers = {}

    def json(self):
        import json
        return json.loads(self.text)


class _ScriptedSession:
    """Serves a scripted list of responses to ChatGPTClient._request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.cookies = type("_C", (), {"set": lambda *a, **k: None})()

    def get(self, url, **kw):
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _scripted_client(monkeypatch, responses):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)   # no real backoff waits
    c = chatgpt.ChatGPTClient({}, "prof")
    c._s = _ScriptedSession(responses)
    return c


def test_request_browser_identity_matches_impersonation():
    import json
    import re
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    class CaptureHeaders(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(dict(self.headers)).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with HTTPServer(('127.0.0.1', 0), CaptureHeaders) as server:
        server.timeout = 1
        client = chatgpt.ChatGPTClient({})
        thread = Thread(target=server.handle_request, daemon=True)
        try:
            thread.start()
            headers = client._request(
                f'http://127.0.0.1:{server.server_port}/', bearer=None).json()
        finally:
            client._s.close()
            thread.join(timeout=5)
        assert not thread.is_alive()
        headers = {k.lower(): v for k, v in headers.items()}
        chrome = re.search(r'Chrome/(\d+)', headers['user-agent']).group(1)
        chromium = re.search(r'"Chromium";v="(\d+)"', headers['sec-ch-ua']).group(1)
        assert chrome == chromium


@pytest.mark.parametrize('challenge', [False, True])
def test_forbidden_is_not_diagnosed_as_expired_login(monkeypatch, challenge):
    response = _FakeResp(403, '<html>Just a moment...</html>' if challenge else
                         '{"detail":"Forbidden"}')
    response.headers['Content-Type'] = 'text/html' if challenge else 'application/json'
    if challenge:
        response.headers['cf-mitigated'] = 'challenge'
    client = _scripted_client(monkeypatch, [response] * chatgpt._MAX_ATTEMPTS)
    url = f'{chatgpt.CHATGPT_HOST}/backend-api/conversations'
    with pytest.raises(RuntimeError) as exc:
        client._request(url, bearer='secret-token')
    error = str(exc.value)
    assert url in error
    assert '403' in error
    assert 'expired' not in error
    assert 'LOG IN' not in error
    assert 'secret-token' not in error
    assert client._s.calls == chatgpt._MAX_ATTEMPTS
    if challenge:
        assert 'Cloudflare challenge' in error
    else:
        assert 'Forbidden' in error


def test_request_survives_observed_cloudflare_retry_exhaustion(monkeypatch):
    import time
    response = _FakeResp(403, '<html>Cloudflare challenge</html>')
    response.headers.update({'Content-Type': 'text/html; charset=UTF-8',
                             'cf-mitigated': 'challenge'})
    client = _scripted_client(monkeypatch, [response] * 8 + [
        _FakeResp(200, '{"items": [{"id": "recovered"}], "total": 1}')])
    sleeps = []
    monkeypatch.setattr(time, 'sleep', sleeps.append)
    monkeypatch.setattr(chatgpt, 'REQUEST_PACING_S', 0)
    page = client._request(
        f'{chatgpt.CHATGPT_HOST}/backend-api/conversations?offset=0&limit=100&order=updated',
        bearer='secret-token').json()
    assert page['items'][0]['id'] == 'recovered'
    assert client._s.calls == 9
    assert sleeps == [4, 8, 16, 32, 60, 60, 60, 60]


_JWKS_503 = '{"detail":"Unable to fetch authentication verification keys."}'


def _timeout():
    from curl_cffi.const import CurlECode
    from curl_cffi.requests.exceptions import Timeout
    return Timeout(
        "Failed to perform, curl: (28) Operation timed out after 30002 milliseconds "
        "with 0 bytes received.", CurlECode.OPERATION_TIMEDOUT)


def test_request_retries_timeout_and_http_failure_then_succeeds(monkeypatch, caplog):
    import time
    unavailable = _FakeResp(503, _JWKS_503)
    unavailable.headers["Retry-After"] = "7"
    c = _scripted_client(monkeypatch, [
        _timeout(), unavailable, _timeout(), _FakeResp(200, '{"items": [1]}')])
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(chatgpt, "REQUEST_PACING_S", 0)
    assert c._request("https://x/backend-api/conversations", bearer="t").json() == {
        "items": [1]}
    assert c._s.calls == 4
    assert sleeps == [4.0, 7.0, 8.0]
    assert "timed out" in caplog.text


def test_request_persistent_timeout_fails_with_cause_and_endpoint(monkeypatch):
    failure = _timeout()
    c = _scripted_client(monkeypatch, [failure] * chatgpt._MAX_ATTEMPTS)
    url = "https://x/backend-api/conversations"
    with pytest.raises(RuntimeError, match="timed out") as exc:
        c._request(url, bearer="secret-token")
    assert c._s.calls == chatgpt._MAX_ATTEMPTS
    assert exc.value.__cause__ is failure
    assert url in str(exc.value)
    assert "secret-token" not in str(exc.value)


@pytest.mark.parametrize("timeout_first", [True, False])
def test_timeout_and_http_errors_share_attempt_budget(monkeypatch, timeout_first):
    errors = [_timeout(), _FakeResp(503, _JWKS_503)]
    responses = (errors if timeout_first else errors[::-1]) * chatgpt._MAX_ATTEMPTS
    c = _scripted_client(monkeypatch, responses)
    with pytest.raises(RuntimeError):
        c._request("https://x/backend-api/conversations", bearer="t")
    assert c._s.calls == chatgpt._MAX_ATTEMPTS


def test_request_does_not_retry_certificate_errors(monkeypatch):
    from curl_cffi.requests.exceptions import CertificateVerifyError
    failure = CertificateVerifyError("certificate invalid")
    c = _scripted_client(monkeypatch, [failure])
    with pytest.raises(CertificateVerifyError) as exc:
        c._request("https://x/backend-api/conversations", bearer="t")
    assert exc.value is failure
    assert c._s.calls == 1


@pytest.mark.parametrize("status", [400, 401])
def test_timeout_does_not_make_permanent_http_errors_retryable(monkeypatch, status):
    c = _scripted_client(monkeypatch, [_timeout(), _FakeResp(status), _FakeResp(200)])
    with pytest.raises(RuntimeError, match=str(status)):
        c._request("https://x/backend-api/conversations", bearer="t")
    assert c._s.calls == 2


def test_request_retries_a_backend_5xx_then_succeeds(monkeypatch):
    """A transient ChatGPT 503 (observed: OpenAI's backend failing to fetch its
    own JWKS to verify a perfectly valid Bearer JWT) must NOT abort the sync —
    it used to kill the whole leg on the first conversations page."""
    c = _scripted_client(monkeypatch, [_FakeResp(503, _JWKS_503),
                                       _FakeResp(502, "bad gateway"),
                                       _FakeResp(200, '{"items": [], "total": 0}')])
    assert c._request("https://x/backend-api/conversations", bearer="t").json() == {
        "items": [], "total": 0}
    assert c._s.calls == 3


def test_request_raises_loudly_on_a_persistent_5xx(monkeypatch):
    """Exhausted retries still fail loud — and name OpenAI's side, not the login,
    so the user does not go re-authenticate a session that is fine."""
    c = _scripted_client(monkeypatch,
                         [_FakeResp(503, _JWKS_503)] * chatgpt._MAX_ATTEMPTS)
    with pytest.raises(RuntimeError, match="fault on OpenAI's side"):
        c._request("https://x/backend-api/conversations", bearer="t")
    assert c._s.calls == chatgpt._MAX_ATTEMPTS


def _wire_cgpt(monkeypatch, client):
    monkeypatch.setattr(clync, "read_chatgpt_cookies",
                        lambda profile: {chatgpt.SESSION_COOKIE: "x"})
    monkeypatch.setattr(chatgpt, "ChatGPTClient",
                        lambda cookies, profile: client)


def test_ingest_chatgpt_end_to_end_then_incremental(pg_test_db, monkeypatch):
    client = _FakeCGPTClient([
        _pair("cg-1", "2026-08-15T05:00:00Z", "plum seed question", "the plum answer",
              title="Plum Seed Germination Tips"),
        _pair("cg-2", "2026-08-14T05:00:00Z", "mushroom nutrition", "the mushroom answer"),
    ])
    _wire_cgpt(monkeypatch, client)

    stats = clync.ingest_chatgpt("prof", full=True)
    assert stats["fetched"] == 2 and stats["conversations_in_db"] == 2

    con = clync.connect()
    try:
        rows = con.execute(
            "SELECT unit_id, kind, source, title, model, created_at, updated_at "
            "FROM units WHERE source='chatgpt' ORDER BY updated_at DESC").fetchall()
        assert [r["unit_id"] for r in rows] == ["cg-1", "cg-2"]
        assert {r["kind"] for r in rows} == {"chatgpt"}
        assert rows[0]["title"] == "Plum Seed Germination Tips"
        assert rows[0]["model"] == "gpt-5-6-thinking"
        assert rows[0]["updated_at"] is not None       # incremental watermark populated
        # messages landed
        nmsg = con.execute(
            "SELECT COUNT(*) n FROM messages WHERE unit_id='cg-1'").fetchone()["n"]
        assert nmsg == 2
    finally:
        con.close()

    # Nothing changed upstream -> a second (incremental) run skips everything.
    stats2 = clync.ingest_chatgpt("prof", full=False)
    assert stats2["fetched"] == 0 and stats2["unchanged"] == 2
    assert stats2["conversations_in_db"] == 2


def test_ingest_chatgpt_reconciles_deleted_conversation(pg_test_db, monkeypatch):
    full = _FakeCGPTClient([
        _pair("cg-1", "2026-08-15T05:00:00Z", "q1", "a1"),
        _pair("cg-2", "2026-08-14T05:00:00Z", "q2", "a2"),
    ])
    _wire_cgpt(monkeypatch, full)
    clync.ingest_chatgpt("prof", full=True)

    # cg-2 deleted upstream -> the live listing no longer returns it.
    _wire_cgpt(monkeypatch, _FakeCGPTClient([
        _pair("cg-1", "2026-08-15T05:00:00Z", "q1", "a1")]))
    stats = clync.ingest_chatgpt("prof", full=False)
    assert stats["conversations_removed"] == 1
    assert stats["conversations_in_db"] == 1

    con = clync.connect()
    try:
        remaining = [r["unit_id"] for r in con.execute(
            "SELECT unit_id FROM units WHERE source='chatgpt'").fetchall()]
        assert remaining == ["cg-1"]
        # messages of the deleted conversation cascaded away
        assert con.execute(
            "SELECT COUNT(*) n FROM messages WHERE unit_id='cg-2'").fetchone()["n"] == 0
    finally:
        con.close()


def test_ingest_chatgpt_reconcile_scoped_to_source(pg_test_db, monkeypatch):
    """A ChatGPT deletion-reconcile must touch ONLY source=chatgpt / kind=chatgpt,
    never another source's units (the source-partition invariant)."""
    # Seed a foreign (claude_ai) chat unit directly.
    con = clync.connect()
    try:
        con.execute(
            "INSERT INTO units (unit_id, kind, source, title, synced_at) "
            "VALUES ('other-1','chat','claude_ai','a claude.ai chat', now())")
        con.commit()
    finally:
        con.close()

    _wire_cgpt(monkeypatch, _FakeCGPTClient([
        _pair("cg-1", "2026-08-15T05:00:00Z", "q1", "a1")]))
    clync.ingest_chatgpt("prof", full=True)

    # Now ALL ChatGPT conversations vanish upstream -> reconcile empties chatgpt...
    _wire_cgpt(monkeypatch, _FakeCGPTClient([]))
    stats = clync.ingest_chatgpt("prof", full=False)
    assert stats["conversations_removed"] == 1
    assert stats["conversations_in_db"] == 0

    con = clync.connect()
    try:
        # ...but the claude_ai unit is untouched.
        assert con.execute(
            "SELECT COUNT(*) n FROM units WHERE source='claude_ai'").fetchone()["n"] == 1
    finally:
        con.close()


def _project_client(monkeypatch, pages, pairs):
    import json
    from urllib.parse import parse_qs, urlsplit

    client = chatgpt.ChatGPTClient({}, 'prof')
    monkeypatch.setattr(client, '_access_token', lambda: 'test-token')

    def request(url, *, bearer):
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        page = query.get('cursor', query.get('offset', [None]))[0]
        key = (parsed.path.removeprefix('/backend-api'), None if page == '0' else page)
        response = pages[key]
        if isinstance(response, Exception):
            raise response
        return _FakeResp(200, json.dumps(response))

    monkeypatch.setattr(client, '_request', request)
    monkeypatch.setattr(client, 'get_conversation', _FakeCGPTClient(pairs).get_conversation)
    _wire_cgpt(monkeypatch, client)
    return client


def _project(pid, name):
    return {'gizmo': {'gizmo': {'id': pid, 'display': {'name': name}}},
            'conversations': {'items': [], 'cursor': None}}


def test_project_threads_paginate_merge_and_remain_searchable(pg_test_db, mock_embed, monkeypatch):
    search = pg_test_db
    pairs = [_pair(f'project-{i}', '2026-08-15T05:00:00Z', 'project budget', f'answer {i}')
             for i in range(3)]
    ordinary = _pair('ordinary', '2026-08-15T05:00:00Z', 'loose question', 'loose answer')
    # Shapes from the live sidebar and per-project endpoints; identities are anonymized.
    pages = {
        ('/conversations', None): {'items': [ordinary[0], pairs[0][0]], 'total': 2},
        ('/gizmos/snorlax/sidebar', None): {
            'items': [_project('g-p-one', 'Finance')], 'cursor': 'projects+next='},
        ('/gizmos/snorlax/sidebar', 'projects+next='): {
            'items': [_project('g-p-two', 'House')], 'cursor': None},
        ('/gizmos/g-p-one/conversations', None): {
            'items': [pairs[0][0]], 'cursor': 'threads+next='},
        ('/gizmos/g-p-one/conversations', 'threads+next='): {
            'items': [pairs[1][0]], 'cursor': None},
        ('/gizmos/g-p-two/conversations', None): {'items': [pairs[2][0]], 'cursor': None},
    }
    client = _project_client(monkeypatch, pages, [ordinary, *pairs])
    stats = clync.ingest_chatgpt('prof')
    assert stats['conversations_in_db'] == 4
    assert stats['fetched'] == 4
    search.build_index()
    hits = search.hybrid_search('budget', source='chatgpt', project='Finance', topk=10)
    assert {h['unit_id'] for h in hits} == {'project-0', 'project-1'}
    with clync.connect() as con:
        assert con.execute("SELECT count(*) AS n FROM messages WHERE unit_id='project-1'").fetchone()['n'] == 2

    # Renaming/moving a project does not necessarily change a conversation's update_time.
    pages[('/gizmos/snorlax/sidebar', None)]['items'][0] = _project('g-p-one', 'Renamed')
    pages[('/gizmos/g-p-one/conversations', None)]['items'] = []
    pages[('/conversations', None)]['items'] = [ordinary[0], pairs[0][0]]
    monkeypatch.setattr(client, 'get_conversation', lambda cid: pytest.fail('unchanged detail fetched'))
    stats = clync.ingest_chatgpt('prof')
    assert stats['unchanged'] == 4 and stats['conversations_removed'] == 0
    search.build_index()
    assert {h['unit_id'] for h in search.hybrid_search(
        'budget', source='chatgpt', project='Renamed', topk=10)} == {'project-1'}
    assert {h['unit_id'] for h in search.hybrid_search(
        '', source='chatgpt', project='none', topk=10)} == {'ordinary', 'project-0'}


@pytest.mark.parametrize('broken', [
    RuntimeError('project endpoint unavailable'),
    {'cursor': None},
    {'items': [], 'cursor': 'repeated'},
    'truncated general listing',
])
def test_incomplete_listing_cannot_delete_stored_threads(pg_test_db, monkeypatch, broken):
    pair = _pair('preserved', '2026-08-15T05:00:00Z', 'important prompt', 'important answer')
    with monkeypatch.context() as seed_patch:
        _wire_cgpt(seed_patch, _FakeCGPTClient([pair]))
        clync.ingest_chatgpt('prof')
    pages = {
        ('/conversations', None): {'items': [], 'total': 0},
        ('/gizmos/snorlax/sidebar', None): {
            'items': [_project('g-p-one', 'Finance')], 'cursor': None},
        ('/gizmos/g-p-one/conversations', None): broken,
        ('/gizmos/g-p-one/conversations', 'repeated'): broken,
    }
    if broken == 'truncated general listing':
        pages[('/conversations', None)] = {'items': [], 'total': 1}
        pages[('/gizmos/snorlax/sidebar', None)] = {'items': [], 'cursor': None}
    _project_client(monkeypatch, pages, [])
    with pytest.raises((RuntimeError, KeyError, ValueError)):
        clync.ingest_chatgpt('prof')
    with clync.connect() as con:
        assert con.execute("SELECT count(*) AS n FROM messages WHERE unit_id=%s",
                           (pair[0]['id'],)).fetchone()['n'] == 2


def test_general_pagination_uses_total_when_server_returns_short_pages(monkeypatch):
    pairs = [_pair(str(i), '2026-08-15T05:00:00Z', 'question', 'answer') for i in range(2)]
    pages = {
        ('/conversations', None): {'items': [pairs[0][0]], 'total': 2},
        ('/conversations', '1'): {'items': [pairs[1][0]], 'total': 2},
        ('/gizmos/snorlax/sidebar', None): {'items': [], 'cursor': None},
    }
    client = _project_client(monkeypatch, pages, pairs)
    assert [c['id'] for c in client.iter_conversations()] == [p[0]['id'] for p in pairs]


def test_general_pagination_refreshes_the_reported_total(monkeypatch):
    pairs = [_pair(str(i), '2026-08-15T05:00:00Z', 'question', 'answer') for i in range(201)]
    pages = {
        ('/conversations', None): {'items': [p[0] for p in pairs[:100]], 'total': 101},
        ('/conversations', '100'): {'items': [p[0] for p in pairs[100:200]], 'total': 201},
        ('/conversations', '200'): {'items': [pairs[200][0]], 'total': 201},
        ('/gizmos/snorlax/sidebar', None): {'items': [], 'cursor': None},
    }
    client = _project_client(monkeypatch, pages, pairs)
    assert [c['id'] for c in client.iter_conversations()] == [p[0]['id'] for p in pairs]


def test_projects_only_import_preserves_ordinary_history_and_reconciles_moves(pg_test_db, monkeypatch):
    ordinary = _pair('ordinary', '2026-08-15T05:00:00Z', 'ordinary prompt', 'ordinary answer')
    moved = _pair('moved', '2026-08-15T05:00:00Z', 'moved prompt', 'moved answer')
    deleted = _pair('deleted', '2026-08-15T05:00:00Z', 'deleted prompt', 'deleted answer')
    for meta, _ in (moved, deleted):
        meta.update(project_uuid='g-p-one', project_name='Finance')
    _wire_cgpt(monkeypatch, _FakeCGPTClient([ordinary, moved, deleted]))
    clync.ingest_chatgpt('prof')

    moved[0].update(project_uuid=None, project_name=None)
    new_project = _pair('new-project', '2026-08-16T05:00:00Z', 'new budget', 'new answer')
    new_project[0].update(project_uuid='g-p-one', project_name='Finance')
    new_ordinary = _pair('new-ordinary', '2026-08-16T05:00:00Z', 'outside scope', 'not imported')
    _wire_cgpt(monkeypatch, _FakeCGPTClient([moved, new_project, new_ordinary]))
    stats = clync.ingest_chatgpt('prof', projects_only=True)
    assert stats['fetched'] == 1 and stats['unchanged'] == 1
    assert stats['conversations_removed'] == 1
    with clync.connect() as con:
        rows = con.execute("SELECT unit_id, project_uuid FROM units WHERE source='chatgpt'").fetchall()
        assert {r['unit_id']: r['project_uuid'] for r in rows} == {
            'ordinary': None, 'moved': None, 'new-project': new_project[0]['project_uuid']}
        assert con.execute("SELECT text FROM messages WHERE unit_id=%s AND sender='user'",
                           (ordinary[0]['id'],)).fetchone()['text'] == 'ordinary prompt'
