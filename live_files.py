"""Candidate-record reads from native Claude Code and Codex JSONL."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from collections import deque
from datetime import UTC, datetime

import cc
import codex
from live_types import Record, Session

HEADER_BYTES = 32768
WINDOW_BYTES = 262144
_CONVERSATION_TYPE = re.compile(rb'"type"\s*:\s*"(?:user|assistant)"')
_ASSISTANT_TYPE = re.compile(rb'"type"\s*:\s*"assistant"')
_TEXT_TYPE = re.compile(rb'"type"\s*:\s*"text"')
_HUMAN_ORIGIN = re.compile(rb'"origin"\s*:\s*\{[^{}]*"kind"\s*:\s*"'
                           + re.escape(cc.HUMAN_ORIGIN_KIND.encode()) + b'"')
_RESPONSE_TYPE = re.compile(rb'"type"\s*:\s*"response_item"')
_MESSAGE_TYPE = re.compile(rb'"type"\s*:\s*"message"')


def searchable_record(source: str, raw: bytes, include_tools: bool) -> bool:
    if source == "pi":
        # Decode every Pi candidate, including malformed JSON, to report partial coverage.
        return True
    if source == "claude_code":
        if include_tools:
            return _CONVERSATION_TYPE.search(raw) is not None
        return ((_ASSISTANT_TYPE.search(raw) is not None and _TEXT_TYPE.search(raw) is not None)
                or _HUMAN_ORIGIN.search(raw) is not None or b'"toolUseResult"' in raw)
    return (_RESPONSE_TYPE.search(raw) is not None and
            (include_tools or _MESSAGE_TYPE.search(raw) is not None))


def _json_lines(data: bytes):
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # A bounded metadata probe may begin or end inside a record.
        if isinstance(value, dict):
            yield value


def sessions(source: str, include_subagents: bool) -> list[Session]:
    root = (cc.CC_ROOT if source == "claude_code" else codex.CODEX_ROOT).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{source} root unavailable: {root}")
    if source == "claude_code":
        paths = list(root.glob("*/*.jsonl"))
        if include_subagents:
            paths.extend(root.glob("*/*/subagents/*.jsonl"))
    else:
        paths = [p for folder in ("sessions", "archived_sessions")
                 for p in (root / folder).glob("**/rollout-*.jsonl")]
    found = {}
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Transcript escapes its source root: {path}")
        stat = path.stat()
        with path.open("rb") as stream:
            head = list(_json_lines(stream.read(HEADER_BYTES)))
            stream.seek(max(0, stat.st_size - HEADER_BYTES))
            tail_bytes = stream.read()
        tail = list(_json_lines(tail_bytes))
        parent = None
        if source == "codex_cli":
            meta = next((e["payload"] for e in head if e.get("type") == "session_meta"), None)
            if meta is None:
                found[str(path)] = Session(source, path.stem, "", "",
                    datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(), path,
                    metadata={"error": f"Missing or pending Codex session metadata: {path}"})
                continue
            origin = meta.get("source")
            if isinstance(origin, dict) and "subagent" in origin:
                child = origin["subagent"]
                parent = (child.get("thread_spawn", {}).get("parent_thread_id")
                          if isinstance(child, dict) else None) or "subagent"
            sid, cwd = meta["id"], meta.get("cwd", "")
            title = ""
            for event in head:
                if event.get("type") == "response_item":
                    item = event.get("payload", {})
                    if item.get("type") == "message" and item.get("role") == "user":
                        text = codex._concat_text(item.get("content", []))
                        if text and not codex._is_injection(text):
                            title = text[:cc._TITLE_MAX]
                            break
        else:
            meta = {key: next((e[key] for e in head if e.get(key)), None)
                    for key in ("sessionId", "cwd", "entrypoint")}
            if meta["entrypoint"] in ("sdk-cli", "sdk-py") and path.parent.name != "subagents":
                continue
            if path.parent.name == "subagents":
                parent = meta["sessionId"] or path.parent.parent.name
            sid = path.stem if parent else meta["sessionId"] or path.stem
            cwd = meta["cwd"] or ""
            title = cc._title_from(head + tail) or ""
        if parent and not include_subagents:
            continue
        timestamp = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
        info = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "pending_tail": bool(tail_bytes and not tail_bytes.endswith(b"\n"))}
        session = Session(source, sid, title, cwd, timestamp, path, parent, info)
        previous = found.get(sid)
        if previous is None or previous.updated_at < timestamp:
            found[sid] = session
    return list(found.values())


def anchor(offset: int, raw: bytes) -> str:
    return f"{offset}:{hashlib.sha256(raw).hexdigest()[:16]}"


def _answer_tools(session: Session, event: dict) -> dict[str, str]:
    if not cc._answers(event):
        return {}
    ids = [b["tool_use_id"] for b in event.get("message", {}).get("content", [])
           if isinstance(b, dict) and b.get("type") == "tool_result"]
    names = {}
    for tool_id in ids:
        result = subprocess.run(["rg", "--no-heading", "--no-filename", "-F", "--",
                                 tool_id, str(session.path)], capture_output=True, check=False)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode(errors="replace"))
        for candidate in _json_lines(result.stdout):
            if candidate.get("type") == "assistant":
                for block in candidate.get("message", {}).get("content", []):
                    if isinstance(block, dict) and block.get("id") == tool_id:
                        names[tool_id] = block["name"]
    return names


def records(session: Session, event: dict, at: str, include_tools: bool) -> list[Record]:
    if session.source == "pi":
        import pi_source
        return pi_source.records(event, at, include_tools)
    timestamp = event.get("timestamp")
    if session.source == "codex_cli":
        if event.get("type") != "response_item":
            return []
        payload = event.get("payload", {})
        kind = payload.get("type")
        if kind == "message":
            cleaned = codex._clean_response_item(payload)
            if cleaned:
                sender, detail, text = cleaned
                return [Record(at, sender, text, detail, timestamp)]
        elif include_tools:
            if kind in ("function_call_output", "custom_tool_call_output"):
                return [Record(at, "tool", codex._output_text(payload.get("output")),
                               "tool_result", timestamp)]
            cleaned = codex._clean_response_item(payload)
            if cleaned:
                sender, detail, text = cleaned
                return [Record(at, sender, text, detail, timestamp)]
        return []
    if event.get("isMeta"):
        return []
    kind = event.get("type")
    content = event.get("message", {}).get("content")
    result = []
    if kind == "user":
        sender = cc._user_sender(event)
        if isinstance(content, str) and sender == cc.HUMAN_SENDER:
            return [Record(at, "user", content, created_at=timestamp)]
        if isinstance(content, list):
            names = _answer_tools(session, event)
            cleaned = cc._clean_event(event, names)
            if cleaned and cleaned[0] == cc.ANSWER_SENDER:
                return [Record(at, cc.ANSWER_SENDER, cleaned[3], created_at=timestamp)]
            for block in content:
                if block.get("type") == "text" and sender == cc.HUMAN_SENDER:
                    result.append(Record(at, "user", block["text"], created_at=timestamp))
                elif include_tools and block.get("type") == "tool_result":
                    result.append(Record(at, "tool", cc._tool_result_body(block),
                                         "tool_result", timestamp))
    elif kind == "assistant" and isinstance(content, list):
        texts = [b["text"] for b in content if b.get("type") == "text" and b.get("text")]
        if texts:
            result.append(Record(at, "assistant", "\n".join(texts), created_at=timestamp))
        if include_tools:
            for block in content:
                if block.get("type") == "thinking" and block.get("thinking"):
                    result.append(Record(at, "assistant", block["thinking"], "reasoning", timestamp))
                elif block.get("type") == "tool_use":
                    result.append(Record(at, "tool", json.dumps(block.get("input", {}),
                                         ensure_ascii=False), f"tool_use:{block.get('name')}", timestamp))
    return result


def candidates(sessions_: list[Session], needles: list[str], exhaustive: bool, ascii_case: bool):
    if exhaustive:
        for session in sessions_:
            with session.path.open("rb") as stream:
                while raw := stream.readline():
                    yield session, stream.tell() - len(raw), raw
        return
    by_path = {str(s.path): s for s in sessions_}
    paths = list(by_path)
    while paths:
        batch, size = [], 0
        while paths and size < 100000:
            name = paths.pop()
            batch.append(name)
            size += len(name.encode()) + 1
        # Mixing ASCII anchors with Unicode/escape needles disables rg's fast literal path.
        groups = ([n for n in needles if n.isascii() and not n.startswith(r"\u")],
                  [n for n in needles if n.startswith(r"\u")],
                  [n for n in needles if not n.isascii()])
        seen = set()
        for group in groups:
            if not group:
                continue
            # Emit locations only, rather than piping multi-megabyte JSON records.
            command = ["rg", "--threads", "2", "--text", "--no-heading", "--with-filename",
                       "--null", "--byte-offset", "--no-line-number", "--max-columns", "1", "-i", "-F"]
            if ascii_case:
                command.append("--no-unicode")
            for needle in group:
                command.extend(("-e", needle))
            command.extend(("--", *batch))
            with tempfile.TemporaryFile() as errors, subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=errors) as process:
                current_path, stream = None, None
                try:
                    for output in process.stdout:
                        name, found = output.split(b"\0", 1)
                        offset = found.split(b":", 1)[0]
                        key = (name, int(offset))
                        if key not in seen:
                            seen.add(key)
                            if current_path != name:
                                if stream is not None:
                                    stream.close()
                                stream = by_path[name.decode()].path.open("rb")
                                current_path = name
                            stream.seek(key[1])
                            raw = stream.readline()
                            yield by_path[name.decode()], key[1], raw
                    code = process.wait()
                    if code not in (0, 1):
                        errors.seek(0)
                        raise RuntimeError(errors.read().decode(errors="replace"))
                finally:
                    if stream is not None:
                        stream.close()
                    if process.poll() is None:
                        process.terminate()
                        process.wait()


def read(session: Session, at: str, context: int, include_tools: bool,
         *, expected_hash: str | None) -> dict:
    match = re.fullmatch(r"(\d+:[0-9a-f]{16})(?::(\d+))?", at)
    if match is None:
        raise ValueError("Invalid file anchor")
    offset = int(at.split(":", 1)[0])
    ordinal = int(match[2]) if match[2] is not None else None
    with session.path.open("rb") as stream:
        stream.seek(offset)
        target = stream.readline()
        if anchor(offset, target) != match[1] or not target.endswith(b"\n"):
            raise ValueError("Stale source anchor; search again")
        stream.seek(max(0, offset - WINDOW_BYTES))
        if stream.tell():
            stream.readline()
        before, after, matched = deque(maxlen=context), [], []
        end = offset + len(target) + WINDOW_BYTES
        while stream.tell() < end and (raw := stream.readline()):
            position = stream.tell() - len(raw)
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError("Malformed record in source window") from error
            items = records(session, event, anchor(position, raw), include_tools)
            for index, record in enumerate(items):
                record.anchor += f":{index}"
            if position < offset:
                before.extend(items)
            elif position == offset:
                if ordinal is None:
                    matched = items
                else:
                    if ordinal >= len(items):
                        raise ValueError("Anchor content is not available with these content filters")
                    if expected_hash != hashlib.sha256(items[ordinal].text.encode()).hexdigest()[:16]:
                        raise ValueError("Anchor content changed or has different content filters; search again")
                    before.extend(items[:ordinal])
                    matched = items[ordinal:ordinal + 1]
                    after.extend(items[ordinal + 1:])
                if len(after) >= context:
                    break
            else:
                after.extend(items)
                if len(after) >= context:
                    break
        return {"records": [*before, *matched, *after[:context]],
                "window_bounded": True}
