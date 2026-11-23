"""Read-only native Pi JSONL history, not Pi's compacted model projection.

Archive messages are append-order tree history, not a linear conversation. Their
role_detail identifies the entry and its nearest archived-message ancestor.
Pi user-role text has no authoritative human provenance and stays `pi_user`.
"""
from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from cc import CCMessage, CCSession, _repo_and_worktree
from live_types import Record, Session

_DEFAULT_ROOT = Path.home() / ".pi/agent/sessions"
PI_ROOT = Path(os.environ.get("CLYNC_PI_ROOT") or os.environ.get("PI_CODING_AGENT_SESSION_DIR")
               or str(Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi/agent")) / "sessions")).expanduser()
USER_SENDER = "pi_user"
HISTORY = "Pi append-order tree history; entry/ancestor labels describe branches, not adjacent turns."
_USER_ROLE = re.compile(rb'"role"\s*:\s*"user"')


def _header(raw: bytes) -> dict:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Malformed or pending Pi session header") from error
    if (not isinstance(value, dict) or value.get("type") != "session"
            or not isinstance(value.get("id"), str) or not value["id"]
            or value.get("version", 1) not in (1, 2, 3)
            or not isinstance(value.get("cwd"), str)):
        raise ValueError("Invalid Pi session header, cwd or unsupported version")
    return value


def _paths(root: Path) -> list[Path]:
    if not root.is_dir():
        if root == _DEFAULT_ROOT and not any(os.environ.get(k) for k in
                ("CLYNC_PI_ROOT", "PI_CODING_AGENT_DIR", "PI_CODING_AGENT_SESSION_DIR")):
            return []
        raise FileNotFoundError(f"Pi transcript root unavailable: {root}; restore it or set CLYNC_PI_ROOT")
    boundary = root.resolve()
    found = []
    for folder, directories, files in os.walk(root):
        for name in [*directories, *files]:
            path = Path(folder) / name
            if not path.resolve().is_relative_to(boundary):
                raise ValueError(f"Transcript escapes its Pi source root: {path}")
        found.extend(Path(folder) / name for name in files if name.endswith(".jsonl"))
    return sorted(found)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Invalid Pi message content")
    return "\n".join(block["text"] for block in content if isinstance(block, dict)
                     and block.get("type") == "text" and isinstance(block.get("text"), str))


def records(event: dict, at: str, include_tools: bool = False) -> list[Record]:
    if event.get("type") != "message":
        return []
    message = event["message"]
    role, content = message["role"], message.get("content", [])
    timestamp = event.get("timestamp")
    result = []
    if role in ("user", "assistant", "toolResult"):
        text = _text(content)
        if text and (role != "toolResult" or include_tools):
            sender = USER_SENDER if role == "user" else "tool" if role == "toolResult" else "assistant"
            result.append(Record(at, sender, text, "tool_result" if role == "toolResult" else "text", timestamp))
    if include_tools and role == "assistant" and isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("Invalid Pi assistant block")
            if block.get("type") == "thinking" and block.get("thinking"):
                result.append(Record(at, "assistant", block["thinking"], "reasoning", timestamp))
            elif block.get("type") == "toolCall":
                result.append(Record(at, "tool", json.dumps(block.get("arguments", {}), ensure_ascii=False),
                                     f"tool_use:{block.get('name')}", timestamp))
    elif include_tools and role == "bashExecution":
        result.append(Record(at, "tool", message.get("command", "") + "\n" + message.get("output", ""),
                             "bash_execution", timestamp))
    return result


def sessions(include_subagents: bool = False, root: Path | None = None,
             *, titles: bool = True) -> list[Session]:
    import live_files

    root = Path(root) if root is not None else PI_ROOT
    found = []
    for path in _paths(root):
        if not include_subagents and "agentstrap-children" in path.relative_to(root).parts:
            continue
        stat = path.stat()
        info = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        session = Session("pi", path.stem, "", "", datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                          path, metadata=info)
        found.append(session)
        try:
            with path.open("rb") as stream:
                raw = stream.readline()
                if not raw.endswith(b"\n"):
                    raise ValueError("Incomplete Pi session header")
                meta = _header(raw)
                session.session_id, session.cwd = meta["id"], meta["cwd"]
                info.update(version=meta.get("version", 1), header=meta,
                            repo=_repo_and_worktree(session.cwd)[0])
                if "agentstrap-children" in path.relative_to(root).parts:
                    session.parent_id = "agentstrap.child"
                if titles:
                    for raw in stream:
                        if _USER_ROLE.search(raw) and raw.endswith(b"\n"):
                            event = json.loads(raw)
                            if event.get("type") == "message" and event.get("message", {}).get("role") == "user":
                                session.title = _text(event["message"].get("content", []))[:120]
                                break
                stream.seek(max(0, stat.st_size - 1))
                info["pending_tail"] = bool(stat.st_size and stream.read(1) != b"\n")
        except (ValueError, TypeError, KeyError) as error:
            info["error"] = f"Invalid Pi source {path}: {error}"
    healthy = [s for s in found if not s.metadata.get("error")]
    for session, _, raw in live_files.candidates(healthy, ['"session_info"', '"agentstrap.child"'], False, True):
        try:
            if not raw.endswith(b"\n"):
                raise ValueError("Incomplete Pi metadata")
            entry = json.loads(raw)
            if entry.get("type") == "session_info" and titles:
                name = entry.get("name")
                if not isinstance(name, str):
                    raise ValueError("Invalid Pi session name")
                session.title = name[:120]
            elif entry.get("type") == "custom" and entry.get("customType") == "agentstrap.child":
                data = entry.get("data", {})
                session.parent_id = data.get("parentSessionID") or data.get("parentSessionId") or "agentstrap.child"
        except (ValueError, TypeError, KeyError) as error:
            session.metadata["error"] = f"Invalid Pi metadata in {session.path}: {error}"
    found = [s for s in found if include_subagents or not s.parent_id]
    ids = set()
    for session in found:
        if not session.metadata.get("error"):
            if session.session_id in ids:
                raise ValueError(f"Duplicate Pi session header id {session.session_id}; remove or relocate exports, or narrow CLYNC_PI_ROOT")
            ids.add(session.session_id)
    return found


def iter_session_files(root: Path | None = None):
    inventory = sessions(root=root, titles=False)
    errors = [s.metadata["error"] for s in inventory if s.metadata.get("error")]
    if errors:
        raise ValueError("; ".join(errors))
    return iter(s.path for s in inventory)


def load_entries(path: Path) -> tuple[dict, list[tuple[int, bytes, dict]]]:
    before = path.stat()
    entries, ids, previous = [], set(), None
    with path.open("rb") as stream:
        raw = stream.readline()
        if not raw.endswith(b"\n"):
            raise ValueError("Incomplete Pi session header")
        meta = _header(raw)
        while raw := stream.readline():
            offset = stream.tell() - len(raw)
            if not raw.endswith(b"\n"):
                raise ValueError("Incomplete trailing Pi record; retry after the writer finishes")
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f"Malformed Pi record at byte {offset}") from error
            if not isinstance(event, dict) or "type" not in event or event["type"] == "session":
                raise ValueError(f"Malformed Pi entry at byte {offset}")
            if meta.get("version", 1) == 1:
                event["id"], event["parentId"] = f"v1-{offset}", previous
            eid, parent = event.get("id"), event.get("parentId")
            if not isinstance(eid, str) or not eid or eid in ids:
                raise ValueError(f"Missing or duplicate Pi entry id at byte {offset}")
            if "parentId" not in event or (parent is not None and parent not in ids):
                raise ValueError(f"Invalid Pi ancestor for entry {eid}")
            ids.add(eid)
            previous = eid
            entries.append((offset, raw, event))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Pi source changed while reading; retry")
    return meta, entries


def parse_session(path: Path) -> CCSession | None:
    path = Path(path)
    meta, entries = load_entries(path)
    if "agentstrap-children" in path.parts or any(e.get("type") == "custom" and
            e.get("customType") == "agentstrap.child" for _, _, e in entries):
        return None
    repo, worktree = _repo_and_worktree(meta["cwd"])
    session = CCSession("pi:" + meta["id"], None, HISTORY, meta.get("timestamp"), meta.get("timestamp"),
                        repo, meta["cwd"], worktree, None, str(meta.get("version", 1)), "pi", None)
    ancestors = {}
    for _, _, event in entries:
        eid = event["id"]
        ancestor = ancestors.get(event["parentId"])
        if event.get("timestamp"):
            session.updated_at = event["timestamp"]
        if event.get("type") == "model_change":
            session.model = event.get("modelId") or session.model
        elif event.get("type") == "session_info":
            session.title = event.get("name") or session.title
        elif event.get("type") == "message":
            session.model = event.get("message", {}).get("model") or session.model
        items = records(event, "")
        for record in items:
            if not session.title and record.sender == USER_SENDER:
                session.title = record.text[:120]
            session.messages.append(CCMessage(eid, len(session.messages), record.sender,
                f"text;entry={eid};ancestor={ancestor or 'root'}", record.text, record.created_at))
        ancestors[eid] = eid if items else ancestor
    return session
