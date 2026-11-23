from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

import pytest

import clync
import mcp_server


def test_archive_tools_only_provision_once_on_first_use(
        pg_test_db, mock_embed, monkeypatch):
    def unexpected_provisioning():
        raise AssertionError('a retrieval request must not provision the store')

    async def exercise():
        server = mcp_server.mcp._mcp_server
        async with server.lifespan(server):
            monkeypatch.setattr(mcp_server, '_archive_ready', False)
            mcp_server.search_transcripts()
            monkeypatch.setattr(mcp_server, 'ensure_cluster', unexpected_provisioning)
            mcp_server.search_transcripts()
            mcp_server.search_history()
            mcp_server.get_conversation('missing')

    asyncio.run(exercise())


def test_local_source_header_preserves_client_version(pg_test_db):
    version = "client-release-47"
    with clync.connect() as con:
        con.execute(
            "INSERT INTO units(unit_id,kind,source,cwd,cc_version,synced_at) "
            "VALUES(%s,%s,%s,%s,%s,%s)",
            ("legacy-local", "cc_session", "claude_code", "/code/demo", version, clync._now()))
    assert version in mcp_server.get_conversation("legacy-local")


@pytest.mark.parametrize(('name', 'arguments', 'module', 'attribute', 'result'), [
    ('search_sessions', {'query': 'iris'}, 'live_search', 'search_sessions', {}),
    ('read_session', {'reference': 'ref', 'anchor': 'anchor'}, 'live_search', 'read_session', {}),
    ('search_transcripts', {}, 'search', 'hybrid_search', []),
    ('search_history', {}, 'mcp_server', '_recall', 'recall'),
    ('search_insights', {}, 'mcp_server', '_recall', 'recall'),
    ('get_conversation', {'uuid': 'unit'}, 'mcp_server', 'read_conversation', 'transcript'),
])
def test_blocking_tools_leave_mcp_dispatch_responsive(
        monkeypatch, name, arguments, module, attribute, result):
    import importlib

    entered = threading.Event()
    release = threading.Event()

    def blocking(*args, **kwargs):
        entered.set()
        assert release.wait(2), 'test failed to release blocking handler'
        return result

    monkeypatch.setattr(mcp_server, '_prepare_archive', lambda: None)
    monkeypatch.setattr(importlib.import_module(module), attribute, blocking)

    async def exercise():
        request = asyncio.create_task(mcp_server.mcp.call_tool(name, arguments))
        try:
            async with asyncio.timeout(1):
                while not entered.is_set():
                    await asyncio.sleep(0.001)
                tools = await mcp_server.mcp.list_tools()
                assert any(tool.name == name for tool in tools)
                assert not request.done(), 'handler blocked unrelated MCP/event-loop work'
        finally:
            release.set()
            results = await asyncio.gather(request, return_exceptions=True)
        assert not isinstance(results[0], BaseException)

    asyncio.run(exercise())


def test_live_search_requests_overlap(monkeypatch):
    import live_search

    both_entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    entered = 0

    def blocking(query, **kwargs):
        nonlocal entered
        with lock:
            entered += 1
            if entered == 2:
                both_entered.set()
        assert release.wait(2), 'test failed to release live searches'
        return {'query': query}

    monkeypatch.setattr(live_search, 'search_sessions', blocking)

    async def exercise():
        requests = [asyncio.create_task(mcp_server.mcp.call_tool(
            'search_sessions', {'query': query})) for query in ('iris', 'freesia')]
        try:
            async with asyncio.timeout(1):
                while not both_entered.is_set():
                    await asyncio.sleep(0.001)
        finally:
            release.set()
            results = await asyncio.gather(*requests, return_exceptions=True)
        assert not any(isinstance(result, BaseException) for result in results)

    asyncio.run(exercise())


def test_cancelled_live_requests_retain_capacity_until_work_finishes(monkeypatch):
    import live_search

    release = threading.Event()
    lock = threading.Lock()
    entered = []

    def blocking(query, **kwargs):
        with lock:
            entered.append(query)
        assert release.wait(2), 'test failed to release cancelled work'
        return {}

    monkeypatch.setattr(live_search, 'search_sessions', blocking)

    async def exercise():
        requests = [asyncio.create_task(mcp_server.mcp.call_tool(
            'search_sessions', {'query': str(i)})) for i in range(16)]
        try:
            async with asyncio.timeout(1):
                while len(entered) < 4:
                    await asyncio.sleep(0.001)
            assert len(entered) == 4
            for request in requests[:4]:
                request.cancel()
            await asyncio.gather(*requests[:4], return_exceptions=True)
            for request in requests[4:12]:
                request.cancel()
            await asyncio.gather(*requests[4:12], return_exceptions=True)
            await asyncio.sleep(0.03)
            assert len(entered) == 4, 'cancellation released capacity before blocking work ended'
        finally:
            release.set()
            await asyncio.gather(*requests, return_exceptions=True)
        assert set(entered[:4]) == {'0', '1', '2', '3'}
        assert set(entered[4:]) == {'12', '13', '14', '15'}

    asyncio.run(exercise())


def test_cancelled_request_logs_later_worker_failure(monkeypatch, caplog):
    import live_search

    entered = threading.Event()
    release = threading.Event()
    failure = RuntimeError('worker failed after caller cancelled')

    def failing(query, **kwargs):
        entered.set()
        assert release.wait(2), 'test failed to release cancelled handler'
        raise failure

    monkeypatch.setattr(live_search, 'search_sessions', failing)

    async def exercise():
        request = asyncio.create_task(mcp_server.mcp.call_tool('search_sessions', {'query': 'iris'}))
        try:
            async with asyncio.timeout(1):
                while not entered.is_set():
                    await asyncio.sleep(0.001)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            release.set()
            async with asyncio.timeout(1):
                while not any(record.exc_info and record.exc_info[1] is failure
                              for record in caplog.records):
                    await asyncio.sleep(0.001)
        finally:
            release.set()
            await asyncio.gather(request, return_exceptions=True)

    asyncio.run(exercise())


def test_cancelled_request_before_worker_start_does_not_log_callback_error(monkeypatch, caplog):
    from concurrent.futures import Future

    work = Future()
    submitted = threading.Event()

    def submit(*args, **kwargs):
        submitted.set()
        return work

    monkeypatch.setattr(mcp_server._live_workers.pool, 'submit', submit)

    async def exercise():
        request = asyncio.create_task(mcp_server.mcp.call_tool('search_sessions', {'query': 'iris'}))
        async with asyncio.timeout(1):
            while not submitted.is_set():
                await asyncio.sleep(0.001)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await asyncio.sleep(0)
        assert work.cancelled()
        assert not caplog.records

    asyncio.run(exercise())


def test_archive_requests_serialize_without_consuming_live_capacity(monkeypatch):
    import live_search
    import search

    release = threading.Event()
    entered = []

    def blocking(name, result):
        def handler(*args, **kwargs):
            entered.append(name)
            assert release.wait(2), 'test failed to release archive request'
            return result
        return handler

    monkeypatch.setattr(mcp_server, '_prepare_archive', lambda: None)
    monkeypatch.setattr(search, 'hybrid_search', blocking('transcripts', []))
    monkeypatch.setattr(mcp_server, '_recall', blocking('recall', 'recall'))
    monkeypatch.setattr(mcp_server, 'read_conversation', blocking('conversation', 'transcript'))
    monkeypatch.setattr(live_search, 'search_sessions', lambda *args, **kwargs: {'hits': []})

    async def exercise():
        requests = [asyncio.create_task(mcp_server.mcp.call_tool(name, arguments))
                    for name, arguments in (
                        ('search_transcripts', {}), ('search_history', {}),
                        ('search_insights', {}), ('get_conversation', {'uuid': 'unit'}))]
        try:
            async with asyncio.timeout(1):
                while not entered:
                    await asyncio.sleep(0.001)
                await mcp_server.mcp.call_tool('search_sessions', {'query': 'iris'})
                await asyncio.sleep(0.03)
                assert len(entered) == 1, 'shared archive/model work ran concurrently'
        finally:
            release.set()
            results = await asyncio.gather(*requests, return_exceptions=True)
        assert len(entered) == 4
        assert not any(isinstance(result, BaseException) for result in results)

    asyncio.run(exercise())


def test_live_stdio_search_read_and_new_activity_without_archive(tmp_path):
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client
    from mcp.shared.exceptions import McpError

    root = tmp_path / "cc" / "project"
    root.mkdir(parents=True)
    transcript = root / "fresh.jsonl"

    def append(text):
        with transcript.open("a") as stream:
            stream.write(json.dumps({"type": "user", "sessionId": "fresh", "cwd": "/code/garden",
                "entrypoint": "cli", "origin": {"kind": "human"}, "message": {"content": text}}) + "\n")

    append("Hyacinth support replaced")
    failure_message = 'MCP E2E cancelled worker failed'
    (tmp_path / "sitecustomize.py").write_text(
        "import builtins\noriginal = builtins.__import__\n"
        "def guard(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in ('torch', 'FlagEmbedding', 'psycopg', 'search'):\n"
        "        raise ImportError('Archive/ML import forbidden in live E2E')\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = guard\n"
        "import live_search, threading\nfrom pathlib import Path\n"
        "original_search = live_search.search_sessions\n"
        "pair = threading.Barrier(2, timeout=2)\n"
        "def paired_search(query, **kwargs):\n"
        "    if query.endswith(' '):\n"
        f"        Path({str(tmp_path / 'started')!r}).touch()\n"
        "        pair.wait()\n"
        "    result = original_search(query, **kwargs)\n"
        "    if query.endswith('   '):\n"
        f"        raise RuntimeError({failure_message!r})\n"
        "    return result\n"
        "live_search.search_sessions = paired_search\n")
    parameters = StdioServerParameters(
        command=sys.executable, args=[str(Path(mcp_server.__file__))],
        cwd=str(Path(mcp_server.__file__).parent), env={**os.environ,
            "CLYNC_CC_ROOT": str(root.parent), "CLYNC_PG_BIN": "/absent/postgres",
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(mcp_server.__file__).parent)))})

    async def exercise(server_log):
        async with stdio_client(parameters, errlog=server_log) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                found = await client.call_tool("search_sessions", {
                    "query": "hyacnth support", "source": "claude_code"})
                assert not found.isError
                hit = json.loads(found.content[0].text)["hits"][0]
                read = await client.call_tool("read_session", {
                    "reference": hit["reference"], "anchor": hit["anchor"]})
                assert any("Hyacinth support replaced" in r["text"]
                           for r in json.loads(read.content[0].text)["records"])
                async def wait_for_slow_search():
                    async with asyncio.timeout(1):
                        while not (tmp_path / 'started').exists():
                            await asyncio.sleep(0.001)
                        await client.send_ping()
                        assert (await client.list_tools()).tools
                        window = await client.call_tool('read_session', {
                            'reference': hit['reference'], 'anchor': hit['anchor']})
                        assert not window.isError

                async def paired_search(spaces):
                    return await client.call_tool('search_sessions', {
                        'query': 'hyacinth support' + ' ' * spaces, 'source': 'claude_code'})

                first = asyncio.create_task(paired_search(1))
                await wait_for_slow_search()
                async with asyncio.timeout(1):
                    parallel = await asyncio.gather(first, paired_search(2))
                assert all(json.loads(result.content[0].text)['hits'] for result in parallel)

                (tmp_path / 'started').unlink()
                cancelled_id = client._request_id
                cancelled = asyncio.create_task(paired_search(3))
                await wait_for_slow_search()
                await client.send_notification(types.ClientNotification(types.CancelledNotification(
                    params=types.CancelledNotificationParams(requestId=cancelled_id))))
                async with asyncio.timeout(1):
                    with pytest.raises(McpError, match='Request cancelled'):
                        await cancelled
                    after_cancel = await paired_search(2)
                assert json.loads(after_cancel.content[0].text)['hits']
                async with asyncio.timeout(1):
                    while failure_message.replace(' ', '') not in ''.join(
                            (tmp_path / 'server.log').read_text().split()):
                        await asyncio.sleep(0.001)

                append("Freesia cable repaired")
                added = await client.call_tool("search_sessions", {
                    "query": "freesia cable", "source": "claude_code"})
                assert json.loads(added.content[0].text)["hits"][0]["session_id"] == hit["session_id"]
                bad = await client.call_tool("read_session", {
                    "reference": "claude_code:../../secret", "anchor": hit["anchor"]})
                assert bad.isError

    with (tmp_path / 'server.log').open('w') as server_log:
        asyncio.run(exercise(server_log))
