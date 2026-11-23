"""Read native Grok ACP logs and OpenCode's projected v2 messages."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

from live_types import Record, Session, TOOL_SENDER, REASONING_KIND

INFO_PATH = "/api/info"
CHILD_DIR = "subagents"


def _string(value, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Invalid {label}")
    return value


def _grok_path(path: Path, root: Path) -> Path:
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Grok path escapes configured root: {path}")
    return path


def grok_sessions(*, include_subagents: bool = False) -> list[Session]:
    root = Path(os.environ.get("CLYNC_GROK_ROOT", Path.home() / ".grok")).expanduser().resolve()
    sessions, visited = [], set()

    def visit(directory, path_parent=None):
        _grok_path(directory, root)
        resolved = directory.resolve()
        if resolved in visited:
            raise ValueError(f"Repeated Grok session directory: {directory}")
        visited.add(resolved)
        summary = json.loads(_grok_path(directory / "summary.json", root).read_text())
        info = summary["info"]
        sid = _string(info["id"], "Grok session identity")
        if not sid or sid != directory.name:
            raise ValueError("Grok session identity does not match its directory")
        parent = summary.get("parent_session_id", info.get("parentId"))
        if parent is not None:
            _string(parent, "Grok parent identity")
        if path_parent is not None and parent is not None and parent != path_parent:
            raise ValueError("Grok parent identity does not match its directory")
        parent = path_parent if path_parent is not None else parent
        if parent and not include_subagents:
            return
        metadata = {"created_at": _string(summary["created_at"], "Grok creation time"),
                    "_root": str(root)}
        metadata.update({key: summary[key] for key in ("num_messages", "num_chat_messages")
                         if key in summary})
        if summary.get("git_root_dir"):
            metadata["repo"] = Path(_string(summary["git_root_dir"], "Grok git root")).name
        sessions.append(Session(
            "grok", sid, _string(summary.get("generated_title") or summary["session_summary"],
                                 "Grok title"),
            _string(info["cwd"], "Grok directory"),
            _string(summary["updated_at"], "Grok update time"),
            _grok_path(directory / "updates.jsonl", root), parent, metadata))
        if include_subagents:
            children = _grok_path(directory / CHILD_DIR, root)
            if children.exists():
                for child in sorted(children.iterdir()):
                    if _grok_path(child, root).is_dir():
                        visit(child, sid)

    for project in sorted(_grok_path(root / "sessions", root).iterdir()):
        if not _grok_path(project, root).is_dir():
            continue
        for directory in sorted(project.iterdir()):
            if _grok_path(directory, root).is_dir():
                visit(directory)
    return sessions


def _content_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(filter(None, (_content_text(part) for part in content)))
    if not isinstance(content, dict):
        raise TypeError("Invalid source content")
    if content["type"] == "text":
        return _string(content["text"], "source text")
    if content["type"] == "content":
        return _content_text(content["content"])
    return ""


def grok_records(session: Session, *, include_tools: bool = False) -> list[Record]:
    if session.source != "grok" or session.path is None:
        raise ValueError("Expected a Grok session with an updates path")
    session.metadata.pop("warnings", None)
    root = Path(session.metadata["_root"])
    path = _grok_path(session.path, root)
    try:
        stream = path.open("rb")
    except FileNotFoundError:
        if all(type(session.metadata.get(key)) is int and session.metadata[key] == 0
               for key in ("num_messages", "num_chat_messages")):
            return []
        raise
    records, chunks = [], []
    pending = None
    group = None

    def flush():
        nonlocal pending, group
        if pending is not None:
            pending.text = "".join(chunks)
            if pending.text.strip():
                records.append(pending)
        chunks.clear()
        pending = group = None

    with stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                session.metadata["warnings"] = [
                    f"Pending Grok JSONL tail at {path}:{offset} (byte offset); complete prefix only"]
                break
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f"Invalid Grok JSONL at {session.path}:{offset} (byte offset)") from error
            if row["method"] not in ("session/update", "_x.ai/session/update"):
                flush()
                continue
            params = row["params"]
            if params["sessionId"] != session.session_id:
                raise ValueError("Grok update session identity changed")
            update = params["update"]
            kind = _string(update["sessionUpdate"], "Grok update type")
            meta = params.get("_meta", {})
            event = meta.get("eventId")
            if event is not None and not _string(event, "Grok event identity"):
                raise ValueError("Empty Grok event identity")
            anchor = (f"event:{event}" if event is not None else
                      f"byte:{offset}:{hashlib.sha256(line).hexdigest()[:16]}")
            value = row["timestamp"]
            timestamp = (datetime.fromtimestamp(value, UTC).isoformat()
                         if isinstance(value, (int, float)) and not isinstance(value, bool)
                         else _string(value, "Grok event timestamp"))
            sender = {"user_message_chunk": "user", "agent_message_chunk": "assistant",
                      "agent_thought_chunk": "assistant"}.get(kind)
            if sender is not None and (kind != "agent_thought_chunk" or include_tools):
                content = update["content"]
                if content["type"] != "text":
                    flush()
                    continue
                detail = REASONING_KIND if kind == "agent_thought_chunk" else "text"
                current = (kind, meta.get("promptId"), update.get("messageId"))
                if group != current:
                    flush()
                    group = current
                    pending = Record(anchor, sender, "", detail, timestamp)
                chunks.append(_string(content["text"], "Grok chunk text"))
                continue
            flush()
            if include_tools and kind in ("tool_call", "tool_call_update"):
                parts = []
                for key in ("rawInput", "rawOutput"):
                    if key in update:
                        parts.append(json.dumps(update[key], ensure_ascii=False))
                if "content" in update:
                    parts.append(_content_text(update["content"]))
                text = "\n".join(filter(None, parts))
                if text.strip():
                    records.append(Record(anchor, TOOL_SENDER, text, kind, timestamp))
    flush()
    return records


def _opencode_client():
    from opencode_source import Client

    client = Client()
    version = _string(client.get(INFO_PATH)["version"], "OpenCode version")
    if version.split(".")[0] != "2":
        raise ValueError(f"OpenCode v2 required, got {version}")
    return client


def opencode_sessions(*, include_subagents: bool = False) -> list[Session]:
    from opencode_source import _iso

    client = _opencode_client()
    inventory = client.inventory()
    projects = {project["id"]: Path(_string(project["canonical"], "OpenCode project root")).name
                for project in client.get("/api/project?archived=true")}
    return [Session("opencode", info["id"], _string(info.get("title") or "", "OpenCode title"),
                    info["location"]["directory"], _iso(info["time"]["updated"]),
                    parent_id=info.get("parentID"),
                    metadata={"repo": projects.get(info["projectID"]),
                              "created_at": _iso(info["time"]["created"])})
            for info in inventory.values() if include_subagents or not info.get("parentID")]


def opencode_records(session: Session, *, include_tools: bool = False) -> list[Record]:
    from opencode_source import Client

    if session.source != "opencode" or not re.fullmatch(r"ses_[A-Za-z0-9_]+", session.session_id):
        raise ValueError("Invalid OpenCode session identity")
    client = Client()
    records, cursors, identities = [], set(), set()
    query = {"limit": "100", "order": "asc"}
    while True:
        page = client.get(f"/api/session/{session.session_id}/message?" + urlencode(query))
        if not isinstance(page["data"], list) or not isinstance(page["cursor"], dict):
            raise TypeError("Invalid OpenCode message page")
        for message in page["data"]:
            if message["id"] in identities:
                raise ValueError("Repeated OpenCode message identity")
            identities.add(message["id"])
            records.extend(opencode_message_records(message, include_tools=include_tools))
        cursor = page["cursor"]["next"]
        if cursor is None:
            return records
        if not isinstance(cursor, str) or not cursor or cursor in cursors or not page["data"]:
            raise ValueError("OpenCode message pagination did not advance")
        cursors.add(cursor)
        query = {"limit": "100", "cursor": cursor}


def opencode_message_records(message: dict, *, include_tools: bool = False) -> list[Record]:
    from opencode_source import _MACHINE_TYPES, _iso

    records = []

    def add(anchor, sender, text, kind, timestamp):
        text = _string(text, "OpenCode record text")
        if text.strip():
            records.append(Record(anchor, sender, text, kind, timestamp))

    mid = _string(message["id"], "OpenCode message identity")
    if not re.fullmatch(r"msg_[A-Za-z0-9_]+", mid):
        raise ValueError("Invalid OpenCode message identity")
    mtype = _string(message["type"], "OpenCode message type")
    if mtype in _MACHINE_TYPES:
        return records
    timestamp = _iso(message["time"]["created"])
    if mtype == "user":
        add(mid, "user", message["text"], "text", timestamp)
    elif mtype == "assistant":
        if not isinstance(message["content"], list):
            raise TypeError("Invalid OpenCode assistant content")
        for ordinal, part in enumerate(message["content"]):
            anchor = f"{mid}:{ordinal}"
            kind = part["type"]
            if kind in ("text", REASONING_KIND):
                if kind == "text" or include_tools:
                    add(anchor, "assistant", part["text"], kind, timestamp)
            elif kind == "tool":
                name = _string(part["name"], "OpenCode tool name")
                state = part["state"]
                if include_tools:
                    inp = state["input"]
                    if not isinstance(inp, dict) and state["status"] != "streaming":
                        raise TypeError("Invalid OpenCode tool input")
                    text = json.dumps(inp, ensure_ascii=False)
                    if "content" in state:
                        text += "\n" + _content_text(state["content"])
                    add(anchor, TOOL_SENDER, text, f"tool:{name}", timestamp)
                if name == "question" and state["status"] == "completed":
                    answers = state.get("metadata", {}).get("answers")
                    if answers is not None:
                        if not isinstance(answers, list) or any(
                            not isinstance(answer, list) or any(not isinstance(v, str) for v in answer)
                            for answer in answers
                        ):
                            raise ValueError("Invalid OpenCode question answers")
                        add(f"{anchor}:answer", "user_answer",
                            "\n".join(v for answer in answers for v in answer),
                            "answer", timestamp)
            else:
                raise ValueError(f"Unknown OpenCode content type: {kind}")
    else:
        raise ValueError(f"Unknown OpenCode message type: {mtype}")
    return records
