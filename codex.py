"""OpenAI Codex CLI session ingest — pure parsing, cleaning, metadata (no DB).

Reads local Codex CLI rollout files (`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`
and `~/.codex/archived_sessions/rollout-*.jsonl`) and turns each into a cleaned,
retrievable `CodexSession` (a `units` row + `messages` rows). The DB write lives
in clync.py; this module is deliberately side-effect-free and unit-testable
against real rollout bytes. Mirrors `cc.py` in structure and contracts.

Design:
- **Metadata from authoritative event fields** (`cwd`, `cli_version`, session
  `id`), never a lossy filename encoding.
- **Scope:** interactive terminal sessions only — the originator dimension is
  decided by the `_SCOPE_ORIGINATORS` SSOT set (the analog of cc.py's
  `entrypoint == "cli"` gate). Every observed originator value has a documented
  decision at that constant; out-of-scope rollouts are skipped *visibly* (a
  distinct sentinel + a distinct ingest-stats counter), never silently dropped.
- **Message SSOT:** the ordered `response_item` stream (`message`, `reasoning`,
  `function_call`/`function_call_output`, `custom_tool_call`/`_output`). The
  `event_msg` stream is a display echo of the same content and is never read —
  reading both would double-count every turn.
- **Cleaning:** drop synthetic first-turn instruction injections (environment
  context, AGENTS.md, user instructions); keep real user/assistant text,
  reasoning summaries verbatim, and tool calls/results compressed to a
  one-liner + head-truncated body.
- **Two-tier text:** `text` is the richer display rendering; `embed_text` is set
  to `""` for a pure tool result (stdout is noise in the vector index), else
  `None` (reuse `text`).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from cc import _TITLE_MAX, _TOOL_ARG_MAX, _TOOL_RESULT_HEAD, _repo_and_worktree

CODEX_ROOT = Path(os.environ.get("CLYNC_CODEX_ROOT", Path.home() / ".codex"))

# --------------------------------------------------------------------------- #
# Originator scope (SSOT) — the ONLY dimension deciding whether a rollout is an
# interactive terminal session we ingest. Codex writes `originator` into
# session_meta and the value has EVOLVED, so a single-value literal here is a
# latent silent whole-corpus drop: the day Codex renamed `codex_cli_rs` ->
# `codex-tui`, every current session parsed to None with no log. The full set
# observed in a real ~/.codex corpus (2026-08, 286 rollouts), with the decision
# for EACH value so a future producer change is a visible, deliberate edit:
#   codex_cli_rs       INGEST  — legacy interactive terminal CLI (pre-rename).
#   codex-tui          INGEST  — current interactive terminal CLI (the rename of
#                                codex_cli_rs); the exact analog of cc.py's
#                                `entrypoint == "cli"` gate.
#   codex_exec         EXCLUDE — `codex exec` non-interactive one-shot/scripted
#                                runs; mirrors cc.py ingesting only interactive
#                                sessions, not headless/batch (cc's "sdk-cli").
#   "Codex Desktop"    EXCLUDE — desktop GUI app (not a terminal session).
#   codex_work_desktop EXCLUDE — desktop GUI app variant (not a terminal session).
# Any originator not in this set is skipped via the OUT_OF_SCOPE sentinel and
# counted distinctly at ingest, so a further rename can never again read as a
# healthy zero-drop run.
_SCOPE_ORIGINATORS = frozenset({"codex_cli_rs", "codex-tui"})


class _OutOfScope:
    """Singleton sentinel: a rollout that WAS examined and deliberately skipped
    because its originator is outside `_SCOPE_ORIGINATORS`. Distinct from `None`
    (no session_meta / no surviving content) so the ingest layer can count
    out-of-scope skips on their own axis — a mass of these with zero ingests is
    the tell that the producer enum moved again."""
    __slots__ = ()

    def __repr__(self) -> str:                # pragma: no cover - debug aid
        return "codex.OUT_OF_SCOPE"


OUT_OF_SCOPE = _OutOfScope()

# Harness/tool-injected context that Codex prepends to (or fuses into) a
# user-role turn — never something the human typed. Two shapes, both enumerated
# from the real ~/.codex corpus: XML-ish wrappers keyed by TAG NAME (matched
# attribute-tolerantly, so `<codex_internal_context source=...>` and
# `<image name=...>` also match), plus a couple of markdown context headers. A
# user message that BEGINS with one of these is dropped whole — the analog of
# cc.py's isMeta drop. (A real prompt is always its OWN message, never fused
# behind an injection wrapper; verified across the corpus — the wrappers that
# fuse are all injection, e.g. recommended_plugins + AGENTS.md + environment.)
# To exclude a wrapper a future Codex build introduces, add its tag/marker here.
_INJECTION_TAGS = frozenset({
    "environment_context", "user_instructions", "INSTRUCTIONS",
    "recommended_plugins", "task-notification", "command-message", "command-name",
    "user_shell_command", "local-command-stdout", "turn_aborted",
    "codex_internal_context", "in-app-browser-context", "codex_delegation", "image",
})
_INJECTION_TEXT_PREFIXES = ("# AGENTS.md instructions", "# Context from my IDE setup:")
_LEADING_TAG = re.compile(r"<([A-Za-z0-9_-]+)")


@dataclass
class CodexMessage:
    msg_id: str                # synthesized: f"{session_id}-{running_index}"
    idx: int
    sender: str                # user | assistant
    role_detail: str           # "text" | "reasoning" | "tool_use:<name>" | "tool_result"
    text: str                  # display rendering
    created_at: str | None
    embed_text: str | None = None   # "" for a pure tool_result; else None


@dataclass
class CodexSession:
    session_id: str
    title: str | None
    summary: str | None            # always None
    created_at: str | None
    updated_at: str | None
    repo: str | None
    cwd: str | None
    worktree: str | None
    git_branch: str | None         # always None (Codex rollouts carry no git branch)
    cc_version: str | None         # the Codex cli_version (reuses cc.py's column name)
    entrypoint: str | None         # always None (Claude-Code-only concept)
    model: str | None
    messages: list[CodexMessage] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def iter_session_files(root: Path | None = None) -> Iterator[Path]:
    """Yield every rollout file under `root/sessions/**/` and
    `root/archived_sessions/`, sorted within each subtree.

    Reads the module `CODEX_ROOT` at call time when no root is given.

    Fails loud (raises, eagerly at call time) when the root is missing or not a
    directory: an unavailable source (unmounted volume, moved/renamed dir) must
    NEVER read as "zero sessions" — consumers reconcile deletions from this
    enumeration, so an ambiguous empty would purge the whole stored corpus. An
    existing-but-empty root (or a root missing one of the two subdirs) is a
    legitimate empty/partial result."""
    root = root or CODEX_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"Codex CLI session root unavailable: {root} "
            f"({'not a directory' if root.exists() else 'does not exist'}). "
            "Refusing to treat a missing source as an empty corpus — deletion "
            "reconciliation would purge every stored session. Restore the "
            "mount/path or set CLYNC_CODEX_ROOT; an existing empty directory is "
            "a valid empty corpus.")

    def _iter() -> Iterator[Path]:
        sessions_dir = root / "sessions"
        if sessions_dir.is_dir():
            yield from sorted(sessions_dir.glob("**/rollout-*.jsonl"))
        archived_dir = root / "archived_sessions"
        if archived_dir.is_dir():
            yield from sorted(archived_dir.glob("rollout-*.jsonl"))

    return _iter()


# --------------------------------------------------------------------------- #
# Cleaning / rendering
# --------------------------------------------------------------------------- #
def _concat_text(parts: list) -> str:
    """Concatenate a `content` list's text-bearing parts (`input_text` /
    `output_text` / `text`); non-text parts (e.g. `input_image`) are ignored."""
    texts = [p.get("text", "") for p in parts
              if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text")]
    return "\n".join(t for t in texts if t)


def _concat_summary(parts: list) -> str:
    """Concatenate a reasoning payload's `summary` list (`{type:"summary_text",
    text:...}` items) — a distinct shape from a message's `content` list."""
    texts = [p.get("text", "") for p in parts
              if isinstance(p, dict) and p.get("type") == "summary_text"]
    return "\n".join(t for t in texts if t)


def _tool_call_line(name: str | None, arg: str) -> str:
    """One-liner for a function_call/custom_tool_call: `[name] <arg>` (truncated)."""
    name = name or "tool"
    arg = arg.replace("\n", " ").strip()
    if len(arg) > _TOOL_ARG_MAX:
        arg = arg[:_TOOL_ARG_MAX] + "…"
    return f"[{name}] {arg}".rstrip()


def _output_text(output) -> str:
    """A function_call_output/custom_tool_call_output `output` is usually a plain
    string, but a tool that returns an image (e.g. a screenshot capture) instead
    carries a list of content parts shaped like a message's `content` (verified
    on real rollouts: `[{"type": "input_image", ...}]`, no text). Reuse the same
    text extraction, which drops non-text parts."""
    if isinstance(output, list):
        return _concat_text(output)
    return output or ""


def _render_tool_output(output: str) -> str:
    """`[result: N bytes] <head>` — head-truncated, byte-counted, matching cc.py's
    `_render_tool_result` shape (Codex has no per-call error flag to surface)."""
    output = output or ""
    n = len(output.encode("utf-8"))
    if len(output) <= _TOOL_RESULT_HEAD:
        shown, marker = output, f"[result: {n} bytes]"
    else:
        shown, marker = output[:_TOOL_RESULT_HEAD] + "…", f"[result: {n} bytes, head]"
    return f"{marker}\n{shown}".rstrip() if shown else marker


def _is_injection(text: str) -> bool:
    """True if a user-role message is harness/tool-injected context, not a typed
    prompt (dropped whole). See _INJECTION_TAGS."""
    stripped = text.lstrip()
    if any(stripped.startswith(p) for p in _INJECTION_TEXT_PREFIXES):
        return True
    m = _LEADING_TAG.match(stripped)
    return bool(m and m.group(1) in _INJECTION_TAGS)


def _clean_response_item(payload: dict) -> tuple[str, str, str] | None:
    """Render one response_item payload to (sender, role_detail, display_text),
    or None if it carries no signal / is out of scope (e.g. `ghost_snapshot`)."""
    ptype = payload.get("type")

    if ptype == "message":
        role = payload.get("role")
        if role not in ("user", "assistant"):     # e.g. "developer" harness turns
            return None
        text = _concat_text(payload.get("content") or []).strip()
        if not text:
            return None
        if role == "user" and _is_injection(text):
            return None
        return (role, "text", text)

    if ptype == "reasoning":
        text = _concat_summary(payload.get("summary") or []).strip()
        return ("assistant", "reasoning", text) if text else None

    if ptype == "function_call":
        return ("assistant", f"tool_use:{payload.get('name')}",
                _tool_call_line(payload.get("name"), payload.get("arguments") or ""))

    if ptype == "custom_tool_call":
        return ("assistant", f"tool_use:{payload.get('name')}",
                _tool_call_line(payload.get("name"), payload.get("input") or ""))

    if ptype in ("function_call_output", "custom_tool_call_output"):
        return ("user", "tool_result", _render_tool_output(_output_text(payload.get("output"))))

    return None    # e.g. ghost_snapshot, agent_message, web_search_call: out of scope


def _embed_text_for(role_detail: str) -> str | None:
    """Tighter embed string: a pure tool_result contributes no embed signal — the
    paired tool_use one-liner already carries the intent, and stdout is noise in
    the vector index."""
    return "" if role_detail == "tool_result" else None


# --------------------------------------------------------------------------- #
# Session parse
# --------------------------------------------------------------------------- #
def parse_session(path: Path) -> CodexSession | _OutOfScope | None:
    """Parse one rollout file into a cleaned CodexSession.

    Three-way result so the ingest layer can distinguish a real skip from a
    deliberate scope exclusion:
      - `CodexSession`  — an in-scope session with surviving content.
      - `OUT_OF_SCOPE`  — examined, but its originator is outside
                          `_SCOPE_ORIGINATORS` (desktop GUI, non-interactive
                          exec, …); counted distinctly at ingest, never silent.
      - `None`          — no session_meta, or no content survives cleaning.
    Malformed lines are skipped individually — both a JSON syntax error and a
    truncated/invalid UTF-8 tail (a session killed mid-write leaves one): the
    file is decoded with `errors="replace"`, so a partial multibyte byte becomes
    a replacement char and degrades to at worst a corrupt JSON line rather than
    aborting the whole ingest. A file that is entirely unreadable (e.g. an I/O
    error) still raises."""
    lines: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a single corrupt line must not lose the session
    if not lines:
        return None

    meta = next((ln for ln in lines if ln.get("type") == "session_meta"), None)
    if meta is None:
        return None
    meta_payload = meta.get("payload") or {}
    if meta_payload.get("originator") not in _SCOPE_ORIGINATORS:   # scope gate — see _SCOPE_ORIGINATORS
        return OUT_OF_SCOPE

    # Single derivation of the session identity: it keys BOTH the msg_id prefix
    # below and the returned CodexSession.session_id (== units.unit_id, the
    # deletion-reconciliation join key). Defining it once keeps those in lockstep.
    session_id = meta_payload.get("id") or path.stem

    messages: list[CodexMessage] = []
    timestamps: list[str] = []
    for ln in lines:
        if ln.get("type") != "response_item":
            continue
        cleaned = _clean_response_item(ln.get("payload") or {})
        if not cleaned:
            continue
        sender, role_detail, text = cleaned
        ts = ln.get("timestamp")
        if ts:
            timestamps.append(ts)
        messages.append(CodexMessage(
            msg_id=f"{session_id}-{len(messages)}", idx=len(messages), sender=sender,
            role_detail=role_detail, text=text, created_at=ts,
            embed_text=_embed_text_for(role_detail)))

    if not messages:
        return None

    model = None
    turn_cwd = None
    for ln in lines:
        if ln.get("type") == "turn_context":
            tc = ln.get("payload") or {}
            if turn_cwd is None:
                turn_cwd = tc.get("cwd")
            if model is None:
                model = tc.get("model")
            if model and turn_cwd:
                break

    cwd = meta_payload.get("cwd") or turn_cwd
    repo, worktree = _repo_and_worktree(cwd)

    # Title = the first typed user prompt. A Codex session launched WITHOUT one
    # (a slash-command / automated trigger whose only user turn was injected
    # context, now filtered out) falls back to its first real action — the first
    # message carrying any text (assistant reply, tool call, or reasoning) — so
    # every session has a usable retrieval handle instead of a blank "(untitled)".
    def _as_title(m) -> str:
        return m.text.strip().replace("\n", " ")[:_TITLE_MAX]

    title = None
    for m in messages:
        if m.sender == "user" and m.role_detail == "text" and m.text.strip():
            title = _as_title(m)
            break
    if not title:
        for m in messages:
            if m.text.strip():
                title = _as_title(m)
                break

    return CodexSession(
        session_id=session_id,
        title=title,
        summary=None,
        created_at=min(timestamps) if timestamps else None,
        updated_at=max(timestamps) if timestamps else None,
        repo=repo, cwd=cwd, worktree=worktree,
        git_branch=None,
        cc_version=meta_payload.get("cli_version"),
        entrypoint=None, model=model, messages=messages,
    )
