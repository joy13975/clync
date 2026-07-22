"""Claude Code session ingest — pure parsing, cleaning, metadata (no DB).

Reads local Claude Code transcripts (`~/.claude/projects/<enc-cwd>/<sessionId>.jsonl`)
and turns each into a cleaned, retrievable `CCSession` (a `units` row + `messages`
rows). The DB write lives in clync.py; this module is deliberately side-effect-free
and unit-testable against real transcript bytes.

Design (ADR 0003):
- **Metadata from authoritative event fields** (`cwd`, `gitBranch`, `version`,
  `sessionId`), never the lossy directory-name encoding.
- **Scope:** only `entrypoint == "cli"` (interactive) sessions; subagent files
  (`<sessionId>/subagents/*.jsonl`, two levels deep) are never walked.
- **Cleaning:** drop harness/meta noise; keep human prompts + assistant text +
  non-empty thinking verbatim; compress tool I/O to a one-liner + head-truncated
  result (the `Agent`/`Task` tools' results are kept whole — subagent syntheses).
- **Two-tier text:** `text` is the richer display rendering; `embed_text` (set
  only when it differs) is a tighter string with tool-result bodies dropped, so
  the vector index is not dominated by stdout.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

CC_ROOT = Path(os.environ.get("CLYNC_CC_ROOT", Path.home() / ".claude/projects"))

_TOOL_RESULT_HEAD = 800        # chars of a tool_result body kept (except Agent/Task)
_TOOL_ARG_MAX = 240            # chars of a tool_use primary-arg one-liner kept
_TITLE_MAX = 120               # chars of a derived title
# Tool -> the input field that best summarizes the call as a one-liner.
_TOOL_PRIMARY_ARG = {
    "Bash": "command", "Read": "file_path", "Edit": "file_path", "Write": "file_path",
    "NotebookEdit": "notebook_path", "Glob": "pattern", "Grep": "pattern",
    "WebFetch": "url", "WebSearch": "query", "Task": "description", "Agent": "description",
}


@dataclass
class CCMessage:
    msg_id: str                # source event uuid (unique within its unit)
    idx: int
    sender: str                # user | assistant
    role_detail: str           # e.g. "text", "thinking", "tool_use:Bash", "tool_result"
    text: str                  # display rendering
    created_at: str | None
    embed_text: str | None = None   # tighter embed string; None => reuse `text`


@dataclass
class CCSession:
    session_id: str
    title: str | None
    summary: str | None
    created_at: str | None
    updated_at: str | None
    repo: str | None
    cwd: str | None
    worktree: str | None
    git_branch: str | None
    cc_version: str | None
    entrypoint: str | None
    model: str | None
    messages: list[CCMessage] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def iter_session_files(root: Path | None = None) -> Iterator[Path]:
    """Return top-level session transcripts only: `<project>/<sessionId>.jsonl`.

    Deliberately one level deep so subagent transcripts
    (`<project>/<sessionId>/subagents/*.jsonl`) and other nested files (memory,
    file-history) are never walked — subagents are out of scope (ADR 0003).
    Reads the module `CC_ROOT` at call time when no root is given.

    Fails loud (raises, eagerly at call time) when the root is missing or not a
    directory: an unavailable source (unmounted volume, moved/renamed dir) must
    NEVER read as "zero transcripts" — consumers reconcile deletions from this
    enumeration, so an ambiguous empty would purge the whole stored corpus. An
    existing-but-empty root is a legitimate empty result (a genuine
    delete-everything)."""
    root = root or CC_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"Claude Code transcript root unavailable: {root} "
            f"({'not a directory' if root.exists() else 'does not exist'}). "
            "Refusing to treat a missing source as an empty corpus — deletion "
            "reconciliation would purge every stored session. Restore the "
            "mount/path or set CLYNC_CC_ROOT; an existing empty directory is "
            "a valid empty corpus.")

    def _iter() -> Iterator[Path]:
        for project_dir in sorted(root.iterdir()):
            if not project_dir.is_dir():
                continue
            yield from sorted(project_dir.glob("*.jsonl"))

    return _iter()


# --------------------------------------------------------------------------- #
# Metadata extraction (from authoritative event fields)
# --------------------------------------------------------------------------- #
_WORKTREE_MARKER = "/.claude/worktrees/"


def _repo_and_worktree(cwd: str | None) -> tuple[str | None, str | None]:
    """Derive (repo, worktree) from an authoritative `cwd`.

    Worktree layout is `<repo-root>/.claude/worktrees/<worktree>/<subpath...>`:
    repo = basename of the part before the marker; worktree = the first segment
    after it. Otherwise not a worktree: repo = basename(cwd), worktree = None."""
    if not cwd:
        return None, None
    if _WORKTREE_MARKER in cwd:
        before, after = cwd.split(_WORKTREE_MARKER, 1)
        repo = os.path.basename(before.rstrip("/")) or None
        worktree = after.split("/", 1)[0] or None
        return repo, worktree
    return os.path.basename(cwd.rstrip("/")) or None, None


# --------------------------------------------------------------------------- #
# Cleaning / rendering
# --------------------------------------------------------------------------- #
def _tool_use_line(block: dict) -> str:
    """One-liner for a tool_use block: `[Name] <primary arg>` (arg truncated)."""
    name = block.get("name") or "tool"
    inp = block.get("input") or {}
    arg = ""
    key = _TOOL_PRIMARY_ARG.get(name)
    if key and isinstance(inp.get(key), str):
        arg = inp[key]
    elif isinstance(inp, dict) and inp:
        # No known primary arg: a compact key listing beats dumping the whole input.
        arg = ", ".join(sorted(inp.keys()))
    arg = arg.replace("\n", " ").strip()
    if len(arg) > _TOOL_ARG_MAX:
        arg = arg[:_TOOL_ARG_MAX] + "…"
    return f"[{name}] {arg}".rstrip()


def _tool_result_body(block: dict) -> str:
    """Flatten a tool_result block's `content` (string or list of text blocks)."""
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content
                 if isinstance(c, dict) and c.get("type") == "text"]
        return "\n".join(p for p in parts if p)
    return ""


def _render_tool_result(block: dict, tool_name: str | None) -> str:
    """`[result: N bytes] <head>` — full body for the Agent/Task tools (both are
    subagent syntheses), else head-truncated. Marks truncation and errors
    explicitly."""
    body = _tool_result_body(block)
    n = len(body.encode("utf-8"))
    err = " ERROR" if block.get("is_error") else ""
    whole = tool_name in ("Agent", "Task")
    if whole or len(body) <= _TOOL_RESULT_HEAD:
        shown = body
        marker = f"[result: {n} bytes{err}]"
    else:
        shown = body[:_TOOL_RESULT_HEAD] + "…"
        marker = f"[result: {n} bytes{err}, head]"
    return f"{marker}\n{shown}".rstrip() if shown else marker


def _clean_event(ev: dict, tool_names: dict[str, str]) -> tuple[str, str, str] | None:
    """Render one user/assistant event to (sender, role_detail, display_text), or
    None if it carries no signal. `tool_names` maps tool_use_id -> tool name for
    the whole session (so a tool_result knows which tool produced it)."""
    if ev.get("isMeta"):
        return None
    etype = ev.get("type")
    msg = ev.get("message") or {}
    content = msg.get("content")

    if etype == "user":
        # A genuine typed/queued prompt is a plain string; an array is tool_result(s).
        if isinstance(content, str):
            txt = content.strip()
            return ("user", "text", txt) if txt else None
        if isinstance(content, list):
            parts, details = [], []
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    name = tool_names.get(b.get("tool_use_id"))
                    parts.append(_render_tool_result(b, name))
                    details.append("tool_result")
                elif b.get("type") == "text" and b.get("text"):
                    parts.append(b["text"])
                    details.append("text")
            body = "\n".join(p for p in parts if p).strip()
            return ("user", "+".join(dict.fromkeys(details)) or "tool_result", body) if body else None
        return None

    if etype == "assistant":
        if not isinstance(content, list):
            return None
        parts, details = [], []
        for b in content:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text" and b.get("text"):
                parts.append(b["text"])
                details.append("text")
            elif bt == "thinking" and (b.get("thinking") or "").strip():
                parts.append(b["thinking"])
                details.append("thinking")
            elif bt == "tool_use":
                parts.append(_tool_use_line(b))
                details.append(f"tool_use:{b.get('name')}")
        body = "\n".join(p for p in parts if p).strip()
        return ("assistant", "+".join(dict.fromkeys(details)), body) if body else None

    return None


def _embed_text_for(sender: str, role_detail: str) -> str | None:
    """Tighter embed string: for a user turn made purely of tool_result(s), drop
    the result bodies entirely — the assistant's tool_use one-liner already carries
    the intent, and stdout is noise in the vector index. Returns None when the embed
    text equals the display text (the common case), so storage stays lean."""
    if sender == "user" and "tool_result" in role_detail and "text" not in role_detail:
        return ""
    return None


# --------------------------------------------------------------------------- #
# Session parse
# --------------------------------------------------------------------------- #
# Each typed title event owns exactly one key. Title events are UPDATES emitted
# repeatedly as a session is (re)titled, so the LAST one is current. Never read a
# bare `title` key — unrelated event types carry one too (e.g. `frame-link`
# holds a published artifact's page title, not the session's).
_TITLE_EVENT_KEY = {"custom-title": "customTitle", "ai-title": "aiTitle"}


def _title_from(events: list[dict]) -> str | None:
    """LAST custom-title -> LAST ai-title -> first typed user prompt (truncated)."""
    latest: dict[str, str] = {}
    for ev in events:
        key = _TITLE_EVENT_KEY.get(ev.get("type"))
        if key:
            v = ev.get(key)
            if isinstance(v, str) and v.strip():
                latest[key] = v.strip()
    for key in ("customTitle", "aiTitle"):
        if key in latest:
            return latest[key][:_TITLE_MAX]
    for ev in events:
        if ev.get("type") == "user" and ev.get("promptSource") == "typed":
            c = (ev.get("message") or {}).get("content")
            if isinstance(c, str) and c.strip():
                return c.strip().replace("\n", " ")[:_TITLE_MAX]
    return None


def _first(events: list[dict], key: str) -> str | None:
    for ev in events:
        v = ev.get(key)
        if v:
            return v
    return None


def parse_session(path: Path) -> CCSession | None:
    """Parse one transcript into a cleaned CCSession, or None if it is skipped
    (non-`cli` entrypoint, or no content survives cleaning). Malformed JSON lines
    are skipped individually; a file that is entirely unreadable raises."""
    events: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a single corrupt line must not lose the session
    if not events:
        return None

    entrypoint = _first(events, "entrypoint")
    if entrypoint != "cli":       # scope: interactive sessions only (ADR 0003)
        return None

    # Session-wide tool_use_id -> tool name map (a tool_result references it).
    tool_names: dict[str, str] = {}
    for ev in events:
        if ev.get("type") == "assistant":
            for b in (ev.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id"):
                    tool_names[b["id"]] = b.get("name")

    messages: list[CCMessage] = []
    timestamps: list[str] = []
    for ev in events:
        if ev.get("type") not in ("user", "assistant"):
            continue
        cleaned = _clean_event(ev, tool_names)
        if not cleaned:
            continue
        uid = ev.get("uuid")
        if not uid:
            continue
        sender, role_detail, text = cleaned
        ts = ev.get("timestamp")
        if ts:
            timestamps.append(ts)
        messages.append(CCMessage(
            msg_id=uid, idx=len(messages), sender=sender, role_detail=role_detail,
            text=text, created_at=ts,
            embed_text=_embed_text_for(sender, role_detail)))

    if not messages:
        return None

    cwd = _first(events, "cwd")
    repo, worktree = _repo_and_worktree(cwd)
    # model: first assistant event carrying one.
    model = None
    for ev in events:
        if ev.get("type") == "assistant":
            model = (ev.get("message") or {}).get("model")
            if model:
                break

    return CCSession(
        session_id=_first(events, "sessionId") or path.stem,
        title=_title_from(events),
        summary=None,
        created_at=min(timestamps) if timestamps else None,
        updated_at=max(timestamps) if timestamps else None,
        repo=repo, cwd=cwd, worktree=worktree,
        git_branch=_first(events, "gitBranch") or None,
        cc_version=_first(events, "version"),
        entrypoint=entrypoint, model=model, messages=messages,
    )
