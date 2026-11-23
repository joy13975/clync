"""Manual CLI/stdio E2E against native files and a temporary real OpenCode session."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("claude_code", "codex_cli", "opencode", "grok")
sys.path.insert(0, str(ROOT))


def api(method, path, body=None):
    command = ["opencode", "api", method, path]
    if body is not None:
        command.extend(("--data", json.dumps(body)))
    with tempfile.TemporaryFile() as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr.decode())
        if method == "delete":
            return None
        output.seek(0)
        return json.load(output)


def run():
    import live_search

    temp_root = Path(tempfile.gettempdir()) / "opencode"
    temp_root.mkdir(exist_ok=True)
    receipt = {}
    with tempfile.TemporaryDirectory(prefix="clync-e2e-", dir=temp_root) as temporary:
        root = Path(temporary)
        repo = root.name
        cc_root, codex_root, grok_root = (root / name for name in ("cc", "codex", "grok"))
        cc_path = cc_root / "project" / "e2e.jsonl"
        codex_path = codex_root / "sessions" / "rollout-e2e.jsonl"
        grok_path = grok_root / "sessions" / "project" / "e2e" / "updates.jsonl"
        for path in (cc_path, codex_path, grok_path):
            path.parent.mkdir(parents=True)

        def append(path, event):
            with path.open("a") as stream:
                stream.write(json.dumps(event) + "\n")

        def cc_event(text):
            return {"type": "user", "sessionId": "e2e", "cwd": str(root), "entrypoint": "cli",
                    "origin": {"kind": "human"}, "message": {"content": text}}

        text = "Heliotrope spindle repaired"
        append(cc_path, cc_event(text))
        append(codex_path, {"type": "session_meta", "payload": {
            "id": "e2e", "cwd": str(root), "originator": "Codex Desktop", "source": "vscode"}})
        append(codex_path, {"type": "response_item", "payload": {"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": text}]}})
        summary = {"info": {"id": "e2e", "cwd": str(root)}, "generated_title": "E2E",
                   "created_at": "2026-11-01T00:00:00Z", "updated_at": "2026-11-01T00:00:01Z"}
        grok_path.with_name("summary.json").write_text(json.dumps(summary))
        for i, chunk in enumerate(("Helio", "trope spindle repaired")):
            append(grok_path, {"timestamp": time.time(), "method": "session/update", "params": {
                "sessionId": "e2e", "update": {"sessionUpdate": "user_message_chunk",
                "content": {"type": "text", "text": chunk}}, "_meta": {"eventId": f"event{i}"}}})
        env = {**os.environ, "CLYNC_CC_ROOT": str(cc_root), "CLYNC_CODEX_ROOT": str(codex_root),
               "CLYNC_GROK_ROOT": str(grok_root), "CLYNC_PG_BIN": "/absent/postgres"}
        query = "helitrope spindle"
        info = api("post", "/api/session", {"title": repo, "location": {"directory": str(root)}})["data"]
        sid = info["id"]
        exists = True
        try:
            api("delete", f"/api/session/{sid}")
            exists = False
            message = {"id": "msg_" + uuid.uuid4().hex, "type": "user", "text": text,
                       "time": {"created": int(time.time() * 1000)}}
            api("post", "/api/experimental/session/import", {
                "info": info, "messages": [message], "location": {"directory": str(root)}})
            exists = True
            command = [sys.executable, "-S", str(ROOT / "clync.py"), "sessions", "search", query,
                       "--repo", repo, "--sender", "user", "--source", ",".join(SOURCES), "--json"]
            cli = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30, check=True)
            result = json.loads(cli.stdout)
            assert {h["source"] for h in result["hits"]} == set(SOURCES), {
                "hits": [(h["source"], h["session_id"]) for h in result["hits"]],
                "coverage": result["coverage"]}
            receipt["cli_ms"] = result["elapsed_ms"]

            (root / "sitecustomize.py").write_text(
                "import builtins\noriginal = builtins.__import__\n"
                "def guard(name, *a, **kw):\n"
                "    if name.split('.')[0] in ('torch','FlagEmbedding','psycopg','search'):\n"
                "        raise ImportError('Archive/ML unavailable during live E2E')\n"
                "    return original(name, *a, **kw)\n"
                "builtins.__import__ = guard\n")
            params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")],
                                           cwd=str(ROOT), env={**env, "PYTHONPATH": str(root)})

            async def exercise():
                async with stdio_client(params) as (reader, writer):
                    async with ClientSession(reader, writer) as client:
                        await client.initialize()
                        arguments = {"query": query, "repo": repo, "sender": "user",
                                     "source": ",".join(SOURCES)}
                        found = await client.call_tool("search_sessions", arguments)
                        assert not found.isError
                        result = json.loads(found.content[0].text)
                        assert {h["source"] for h in result["hits"]} == set(SOURCES)
                        receipt["mcp_ms"] = result["elapsed_ms"]
                        for hit in result["hits"]:
                            read = await client.call_tool("read_session", {
                                "reference": hit["reference"], "anchor": hit["anchor"], "context": 0})
                            assert not read.isError
                            assert any(text in r["text"] for r in json.loads(read.content[0].text)["records"])
                        append(cc_path, cc_event("Cyclamen latch repaired"))
                        fresh = await client.call_tool("search_sessions", {
                            "query": "cyclamen latch", "source": "claude_code"})
                        assert json.loads(fresh.content[0].text)["hits"]
                        with grok_path.open("a") as stream:
                            stream.write('{"params":')
                        partial = await client.call_tool("search_sessions", arguments)
                        payload = json.loads(partial.content[0].text)
                        assert {h["source"] for h in payload["hits"]} == set(SOURCES)
                        assert payload["coverage"]["grok"]["status"] == live_search.PARTIAL
                        hit = next(h for h in payload["hits"] if h["reference"] == "claude_code:e2e")
                        cc_path.write_text(cc_path.read_text().replace("spindle repaired", "spindle changed"))
                        stale = await client.call_tool("read_session", {
                            "reference": hit["reference"], "anchor": hit["anchor"]})
                        assert stale.isError
                        unsafe = await client.call_tool("read_session", {
                            "reference": "claude_code:../../secret", "anchor": hit["anchor"]})
                        assert unsafe.isError
                        receipt.update(sources=sorted(SOURCES), reads=len(SOURCES),
                                       append_visible=True, pending_tail_retains_hits=True,
                                       stale_rejected=True, traversal_rejected=True,
                                       postgres_and_ml_blocked=True)

            asyncio.run(exercise())
            unavailable = subprocess.run(
                [sys.executable, "-S", str(ROOT / "clync.py"), "sessions", "search", "cyclamen",
                 "--source", "grok", "--json"], env={**env, "CLYNC_GROK_ROOT": str(root / "missing")},
                capture_output=True, text=True, timeout=30)
            assert unavailable.returncode == 2
            assert json.loads(unavailable.stdout)["coverage"]["grok"]["status"] == live_search.UNAVAILABLE
            receipt["unavailable_exit"] = unavailable.returncode
        finally:
            if exists:
                api("delete", f"/api/session/{sid}")
    return receipt


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
