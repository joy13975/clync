"""OpenCode's projected v2 transcripts, without machine messages or tool outputs."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import clync
from cc import _TITLE_MAX, _TOOL_ARG_MAX, ANSWER_SENDER, CCMessage, CCSession, _clip, _fork_lines

SOURCE = "opencode"
KIND = "opencode_session"
_PARSER_VERSION = 1
_SUBAGENT_TOOLS = frozenset({"subagent", "task", "agent"})
_MACHINE_TYPES = frozenset({"synthetic", "system", "skill", "shell", "compaction", "idle",
                            "agent-switched", "model-switched", "location-switched"})


def config_path() -> Path:
    return Path(os.environ.get("CLYNC_OPENCODE_CONFIG", clync.DATA_HOME / "opencode.json"))


class SessionNotFound(RuntimeError):
    pass


class Client:
    def __init__(self):
        command = (json.loads(config_path().read_text())["command"]
                   if config_path().exists() else shutil.which("opencode"))
        if not isinstance(command, str) or not Path(command).is_absolute():
            raise ValueError("OpenCode requires an absolute executable path in its source config")
        self.command = command

    def get(self, path: str):
        # OpenCode can exit before piped stdout drains; regular-file writes are synchronous.
        with tempfile.TemporaryFile() as output:
            result = subprocess.run([self.command, "api", "get", path], check=False, stdout=output,
                                    stderr=subprocess.PIPE, text=True, timeout=120)
            output.seek(0)
            if result.returncode:
                try:
                    error = json.load(output)
                except json.JSONDecodeError:
                    error = None
                if (isinstance(error, dict) and error.get("_tag") == "SessionNotFoundError"
                        and error.get("sessionID") in path.split("/")
                        and "HTTP 404" in result.stderr):
                    raise SessionNotFound(error["sessionID"])
                raise RuntimeError(f"OpenCode request {path} failed: {result.stderr.strip()}")
            return json.load(output)

    def inventory(self) -> dict[str, dict]:
        sessions, seen = {}, set()
        query = {"limit": "100", "order": "asc"}
        while True:
            page = self.get("/api/session?" + urlencode(query))
            if not isinstance(page["data"], list) or not isinstance(page["cursor"], dict):
                raise TypeError("Invalid OpenCode session inventory")
            for info in page["data"]:
                _validate_info(info)
                sessions[info["id"]] = info
            cursor = page["cursor"].get("next")
            if cursor is None:
                return sessions
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise ValueError("OpenCode session pagination did not advance")
            seen.add(cursor)
            query = {"limit": "100", "cursor": cursor}


def _validate_info(info: dict) -> None:
    if not isinstance(info["id"], str) or not re.fullmatch(r"ses_[A-Za-z0-9_]+", info["id"]):
        raise ValueError("Invalid OpenCode session identity")
    for key in ("created", "updated"):
        _iso(info["time"][key])
    if not isinstance(info["location"]["directory"], str):
        raise TypeError("Invalid OpenCode session directory")
    if not isinstance(info["projectID"], str):
        raise TypeError("Invalid OpenCode project identity")
    if info.get("parentID") is not None and not isinstance(info["parentID"], str):
        raise ValueError("Invalid OpenCode parent identity")


def _iso(milliseconds: float) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, UTC).isoformat()


def _signature(info: dict) -> str:
    metadata = {k: info.get(k) for k in
                ("title", "projectID", "location", "model", "parentID", "fork")}
    metadata["time"] = {k: info["time"].get(k) for k in
                        ("created", "updated", "busy", "idle", "archived")}
    metadata["parser_version"] = _PARSER_VERSION
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def parse_session(exported: dict, repo: str | None) -> CCSession | None:
    info = exported["info"]
    _validate_info(info)
    if info.get("parentID"):
        return None
    if not isinstance(exported["messages"], list):
        raise TypeError("Invalid OpenCode transcript")
    messages = []

    def add(mid, sender, detail, text, timestamp, embed=None):
        if text.strip():
            messages.append(CCMessage(mid, len(messages), sender, detail, text.strip(),
                                      _iso(timestamp), embed))

    for message in exported["messages"]:
        mtype = message["type"]
        if mtype in _MACHINE_TYPES:
            continue
        mid, timestamp = message["id"], message["time"]["created"]
        if mtype == "user":
            add(mid, "user", "text", message["text"], timestamp)
        elif mtype == "assistant":
            for ordinal, part in enumerate(message["content"]):
                pid = f"{mid}:{ordinal}"
                ptype = part["type"]
                if ptype in ("text", "reasoning"):
                    add(pid, "assistant", ptype, part["text"], timestamp)
                elif ptype == "tool":
                    name = part["name"]
                    if name.lower() in _SUBAGENT_TOOLS:
                        continue
                    state = part["state"]
                    inp = state["input"]
                    if not isinstance(inp, dict):
                        if state["status"] == "streaming":
                            continue
                        raise ValueError("Invalid OpenCode tool input")
                    if name == "question":
                        line = f"[{name}]\n{_fork_lines(inp)}"
                    else:
                        arg = next((inp[k] for k in
                                    ("command", "path", "pattern", "url", "query", "description", "id")
                                    if isinstance(inp.get(k), str)), ", ".join(sorted(inp)))
                        line = f"[{name}] {_clip(arg.replace(chr(10), ' '), _TOOL_ARG_MAX)}".rstrip()
                    add(pid, "assistant", f"tool_use:{name}", line, timestamp)
                    if name == "question" and state["status"] == "completed":
                        answers = state.get("metadata", {}).get("answers")
                        if answers is not None:
                            if (not isinstance(answers, list) or any(
                                    not isinstance(a, list) or any(not isinstance(v, str) for v in a)
                                    for a in answers)):
                                raise ValueError("Invalid OpenCode question answers")
                            text = "\n".join(v for a in answers for v in a)
                            add(f"{pid}:answer", ANSWER_SENDER, "answer", text, timestamp, text)
                else:
                    raise ValueError(f"Unknown OpenCode content type: {ptype}")
        else:
            raise ValueError(f"Unknown OpenCode message type: {mtype}")
    title = info.get("title") or next((m.text.replace("\n", " ")[:_TITLE_MAX]
                                       for m in messages if m.sender == "user"), None)
    model = info.get("model")
    return CCSession(f"{SOURCE}:{info['id']}", title, None, _iso(info["time"]["created"]),
                     _iso(info["time"]["updated"]), repo, info["location"]["directory"],
                     None, None, None, None, model["id"] if model else None, messages)


def ingest(full: bool = False) -> dict:
    client = Client()
    version = client.get("/api/info")["version"]
    if version.split(".")[0] != "2":
        raise ValueError(f"OpenCode v2 required, got {version}")
    inventory = client.inventory()
    projects = {p["id"]: Path(p["canonical"]).name
                for p in client.get("/api/project?archived=true")}
    active = client.get("/api/session/active")["data"]
    if not isinstance(active, dict):
        raise TypeError("Invalid OpenCode active-session inventory")
    stats = {"sessions_ingested": 0, "unchanged": 0, "out_of_scope": 0,
             "dirty": 0, "sessions_removed": 0}
    with clync.connect() as con:
        stored = {r["unit_id"]: r["raw"] for r in con.execute(
            "SELECT unit_id,raw FROM units WHERE source=%s", (SOURCE,)).fetchall()}
        live = set()
        for sid, listed in inventory.items():
            uid = f"{SOURCE}:{sid}"
            if listed.get("parentID"):
                stats["out_of_scope"] += 1
                continue
            live.add(uid)
            previous = stored.get(uid)
            if (not full and previous and not previous["dirty"] and sid not in active
                    and previous["sync_signature"] == _signature(listed)):
                stats["unchanged"] += 1
                continue
            exported = client.get(f"/api/experimental/session/{sid}/export")["data"]
            info = exported["info"]
            if info["id"] != sid:
                raise ValueError("OpenCode export identity changed")
            session = parse_session(exported, projects.get(info["projectID"]) or None)
            if session is None:
                stats["out_of_scope"] += 1
                live.discard(uid)
                continue
            after = client.get(f"/api/session/{sid}")["data"]
            _validate_info(after)
            if after["id"] != sid:
                raise ValueError("OpenCode session identity changed")
            if after.get("parentID"):
                stats["out_of_scope"] += 1
                live.discard(uid)
                continue
            active_after = client.get("/api/session/active")["data"]
            if not isinstance(active_after, dict):
                raise TypeError("Invalid OpenCode active-session inventory")
            dirty = (sid in active or sid in active_after
                     or _signature(info) != _signature(after))
            model = info.get("model") or {}
            raw = {"native_session_id": sid, "project_id": info["projectID"],
                   "provider_id": model.get("providerID"), "variant": model.get("variant"),
                   "archived": info["time"].get("archived"),
                   "fork_session_id": info["fork"]["sessionID"] if info.get("fork") else None,
                   "sync_signature": _signature(info), "dirty": dirty}
            clync._upsert_local_session(con, session, SOURCE, KIND, raw=raw)
            con.commit()
            stats["sessions_ingested"] += 1
            stats["dirty"] += int(dirty)
        for uid, raw in stored.items():
            if uid in live:
                continue
            try:
                confirmed = client.get(f"/api/session/{raw['native_session_id']}")["data"]
                _validate_info(confirmed)
                if confirmed["id"] != raw["native_session_id"]:
                    raise ValueError("OpenCode deletion lookup identity changed")
                if not confirmed.get("parentID"):
                    continue
            except SessionNotFound:
                pass
            con.execute("DELETE FROM units WHERE unit_id=%s AND source=%s", (uid, SOURCE))
            stats["sessions_removed"] += 1
        con.commit()
    print(f"[{SOURCE}] " + json.dumps(stats))
    return stats
