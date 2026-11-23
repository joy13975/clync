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
- **Sender is the channel, read from `origin.kind`** (see `_user_sender`): the
  human's own turn is `sender="user"`, and every other thing the harness files
  under `type: "user"` keeps its own channel name.
- **Two-tier text:** `text` is the richer display rendering; `embed_text` (set
  only when it differs) is a tighter string. `""` means the message is stored but
  contributes no search chunk, which is how every machine channel stays out of
  the index while an episode still reads whole.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

CC_ROOT = Path(os.environ.get("CLYNC_CC_ROOT", Path.home() / ".claude/projects"))

_TOOL_RESULT_HEAD = 800        # chars of a tool_result body kept (except the whole-body tools)
_TOOL_ARG_MAX = 240            # chars of a tool_use primary-arg one-liner kept
_TITLE_MAX = 120               # chars of a derived title
# Tool -> the input field that best summarizes the call as a one-liner.
_TOOL_PRIMARY_ARG = {
    "Bash": "command", "Read": "file_path", "Edit": "file_path", "Write": "file_path",
    "NotebookEdit": "notebook_path", "Glob": "pattern", "Grep": "pattern",
    "WebFetch": "url", "WebSearch": "query", "Task": "description", "Agent": "description",
}
ASK_TOOL = "AskUserQuestion"
# Results kept WHOLE: `Agent`/`Task` are subagent syntheses, and an answered
# `ASK_TOOL` is the user's own decision (see `_is_user_answer`).
_WHOLE_RESULT_TOOLS = ("Agent", "Task")


# --------------------------------------------------------------------------- #
# Whose turn is it?
# --------------------------------------------------------------------------- #
# The harness files far more than typed prompts under `type: "user"`: tool
# results, a subagent's whole answer (`task-notification`), another agent
# session's message (`peer`), auto-continuations, compaction summaries,
# slash-command echoes and local-command output. Measured on the real corpus,
# those outnumber the human's own turns 25:1 — and calling them `user` is what
# let a subagent's prose come back from search as something the user said.
#
# `origin.kind` is the harness's own answer to the question, so read that field
# and never infer from wording. The human's turn is `origin.kind == "human"` and
# nothing else; anything else keeps its kind as its sender, so a kind nobody has
# seen yet gets its own sender and stays out of the index without a list here to
# extend.
HUMAN_ORIGIN_KIND = "human"
HUMAN_SENDER = "user"
ANSWER_SENDER = "user_answer"       # the user's reply to `ASK_TOOL`
CHAT_HUMAN_SENDER = "human"         # claude.ai's own word for the user, stored as sent
ASSISTANT_SENDER = "assistant"
NO_ORIGIN_SENDER = "harness"        # a user-slot record with no `origin` at all
# The only senders that reach the search index.
EMBEDDED_SENDERS = (HUMAN_SENDER, ANSWER_SENDER, ASSISTANT_SENDER)
# WHOSE words a message is, as one role per stored sender. Every reader that asks
# that question — `dream`'s attribution gate, `search`'s sender facet — resolves it
# here, so a new channel is admitted in one place instead of in each of their lists.
SENDER_ROLES: dict[str, tuple[str, ...]] = {
    "user": (HUMAN_SENDER, ANSWER_SENDER, CHAT_HUMAN_SENDER),
    "assistant": (ASSISTANT_SENDER,),
}


def resolve_sender_role(role: str) -> list[str]:
    """The stored senders one role covers, or a loud failure. A caller asks for a
    ROLE because the stored spelling differs per source, so accepting a raw sender
    would quietly search one source's corpus and silently skip another's."""
    if role not in SENDER_ROLES:
        raise ValueError(f"sender role must be one of {sorted(SENDER_ROLES)} — "
                         f"got {role!r}, which is not a role")
    return list(SENDER_ROLES[role])


def _user_sender(ev: dict) -> str:
    origin = ev.get("origin")
    kind = origin.get("kind") if isinstance(origin, dict) else None
    if kind == HUMAN_ORIGIN_KIND:
        return HUMAN_SENDER
    return kind or NO_ORIGIN_SENDER


@dataclass
class CCMessage:
    msg_id: str                # source event uuid (unique within its unit)
    idx: int
    sender: str                # user | assistant | a machine channel (_user_sender)
    role_detail: str           # e.g. "text", "thinking", "tool_use:Bash", "tool_result"
    text: str                  # display rendering
    created_at: str | None
    embed_text: str | None = None   # "" => no search chunk; None => reuse `text`


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
def _clip(text: str, limit: int) -> str:
    return text[:limit] + "…" if len(text) > limit else text


def _fork_lines(inp: dict) -> str:
    """The fork `ASK_TOOL` put to the user: each question, then the options offered.
    The generic one-liner below renders this input as the bare key `questions`, which
    leaves a retrieved answer with no readable question and loses the alternatives
    the user turned down — the other half of the decision."""
    out = []
    for q in inp.get("questions") or []:
        out.append(_clip(f"Q: {q.get('question', '')}".replace("\n", " "), _TOOL_ARG_MAX))
        labels = " | ".join(o.get("label", "") for o in q.get("options") or [])
        if labels:
            out.append(_clip(f"options: {labels}".replace("\n", " "), _TOOL_ARG_MAX))
    return "\n".join(out)


def _tool_use_line(block: dict) -> str:
    """One-liner for a tool_use block: `[Name] <primary arg>` (arg truncated).
    `ASK_TOOL` renders as its questions and options instead — it has no one primary
    argument, and what it asked is what makes the user's answer readable."""
    name = block.get("name") or "tool"
    inp = block.get("input") or {}
    if name == ASK_TOOL:
        return f"[{name}]\n{_fork_lines(inp)}".rstrip()
    arg = ""
    key = _TOOL_PRIMARY_ARG.get(name)
    if key and isinstance(inp.get(key), str):
        arg = inp[key]
    elif isinstance(inp, dict) and inp:
        # No known primary arg: a compact key listing beats dumping the whole input.
        arg = ", ".join(sorted(inp.keys()))
    arg = _clip(arg.replace("\n", " ").strip(), _TOOL_ARG_MAX)
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


def _answers(ev: dict) -> dict:
    """The user's own answers, as the harness recorded them structurally:
    `{question: answer}`. The KEYS are the assistant's questions."""
    result = ev.get("toolUseResult")
    answers = result.get("answers") if isinstance(result, dict) else None
    return answers if isinstance(answers, dict) and answers else {}


def _is_user_answer(block: dict, tool_name: str | None, ev: dict) -> bool:
    """Whether this result is an `ASK_TOOL` question the user ANSWERED, in a form
    whose authorship can be separated. A refusal or an interrupt arrives on the same
    tool with `is_error` set and a body of harness boilerplate, which is nothing the
    user wrote; and without the structural answers there is no way to tell the
    user's words from the question they were asked, so it is not attributed to
    them."""
    return tool_name == ASK_TOOL and not block.get("is_error") and bool(_answers(ev))


def _answer_embed_text(ev: dict) -> str:
    """ONLY the words the user contributed: the answers they chose and any note they
    typed. The rendered body also carries the question and the harness's own framing
    around it, and embedding that under the user's sender would let the assistant's
    question come back from search as something the user said."""
    parts = list(_answers(ev).values())
    annotations = (ev.get("toolUseResult") or {}).get("annotations")
    if isinstance(annotations, dict):
        parts += [a["notes"] for a in annotations.values()
                  if isinstance(a, dict) and a.get("notes")]
    return "\n".join(str(p) for p in parts if p)


def _render_tool_result(block: dict, tool_name: str | None, ev: dict) -> str:
    """`[result: N bytes] <head>` — full body for `_WHOLE_RESULT_TOOLS` and for an
    answered `ASK_TOOL`, else head-truncated. Marks truncation and errors
    explicitly."""
    body = _tool_result_body(block)
    n = len(body.encode("utf-8"))
    err = " ERROR" if block.get("is_error") else ""
    whole = tool_name in _WHOLE_RESULT_TOOLS or _is_user_answer(block, tool_name, ev)
    if whole or len(body) <= _TOOL_RESULT_HEAD:
        shown = body
        marker = f"[result: {n} bytes{err}]"
    else:
        shown = _clip(body, _TOOL_RESULT_HEAD)
        marker = f"[result: {n} bytes{err}, head]"
    return f"{marker}\n{shown}".rstrip() if shown else marker


def _clean_event(ev: dict, tool_names: dict[str, str]
                 ) -> tuple[str, str, str, str | None] | None:
    """Render one user/assistant event to
    (sender, role_detail, display_text, embed_override), or None if it carries no
    signal. `tool_names` maps tool_use_id -> tool name for the whole session (so a
    tool_result knows which tool produced it). `embed_override` is None unless the
    record's searchable text is NARROWER than what it displays — an answered
    question shows the question it answers but must embed only the user's reply."""
    if ev.get("isMeta"):
        return None
    etype = ev.get("type")
    msg = ev.get("message") or {}
    content = msg.get("content")

    if etype == "user":
        # A genuine typed/queued prompt is a plain string; an array is tool_result(s).
        sender = _user_sender(ev)
        if isinstance(content, str):
            txt = content.strip()
            return (sender, "text", txt, None) if txt else None
        if isinstance(content, list):
            parts, details, answers = [], [], []
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    name = tool_names.get(b.get("tool_use_id"))
                    answers.append(_is_user_answer(b, name, ev))
                    parts.append(_render_tool_result(b, name, ev))
                    details.append("tool_result")
                elif b.get("type") == "text" and b.get("text"):
                    parts.append(b["text"])
                    details.append("text")
                    answers.append(False)   # unattributed prose, not an answer
            # An answer carries no `origin`, so the rule above files it as a machine
            # channel — which hid the user's own decisions from search entirely. It is
            # theirs only when EVERY block in the record is one: anything else sharing
            # the record would be stored under the user's name too.
            embed = None
            if answers and all(answers):
                sender, embed = ANSWER_SENDER, _answer_embed_text(ev)
            body = "\n".join(p for p in parts if p).strip()
            detail = "+".join(dict.fromkeys(details)) or "tool_result"
            return (sender, detail, body, embed) if body else None
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
        return ((ASSISTANT_SENDER, "+".join(dict.fromkeys(details)), body, None)
                if body else None)

    return None


def _embed_text_for(sender: str) -> str | None:
    """`""` => the message is stored but contributes no search chunk (search.py's
    `_load_chunks` is the gate); None => embed the display `text` unchanged.

    Only the human's own turns and the assistant's are searchable. Every machine
    channel is kept for reading — an episode with holes in it cannot be read —
    and excluded from retrieval, so no search hit can carry a subagent's or the
    harness's words under the user's name. Tool output leaves the index by the
    same rule: its `sender` is a channel, and the paired tool_use one-liner
    already carries the intent."""
    return None if sender in EMBEDDED_SENDERS else ""


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
        sender, role_detail, text, embed_override = cleaned
        ts = ev.get("timestamp")
        if ts:
            timestamps.append(ts)
        messages.append(CCMessage(
            msg_id=uid, idx=len(messages), sender=sender, role_detail=role_detail,
            text=text, created_at=ts,
            embed_text=(_embed_text_for(sender) if embed_override is None
                        else embed_override)))

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
