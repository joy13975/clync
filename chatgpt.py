"""ChatGPT web-history client + conversation linearizer (clync's fourth source).

Reads the user's ChatGPT history over the same private web API the browser uses:
the Chrome cookie ``__Secure-next-auth.session-token`` (+ Cloudflare's
``cf_clearance``) is exchanged at ``/api/auth/session`` for a short-lived
``accessToken`` (a JWT), which then authorizes Bearer calls to
``/backend-api/conversations`` (the list) and ``/backend-api/conversation/{id}``
(one conversation's message tree). TLS-impersonated via curl_cffi so Cloudflare
passes, exactly like clync's claude.ai leg (``ClaudeClient``). The ChatGPT
*desktop* app stores history encrypted at rest, so this web API is the only
readable route.

Parsing is pure and side-effect-free: a conversation-detail dict + its list
metadata -> a ``ChatGPTConversation`` whose field names are IDENTICAL to
``CCSession``/``CodexSession`` so ``clync._upsert_local_session`` duck-types it
(the ChatGPT-only fields repo/cwd/worktree/git_branch/cc_version/entrypoint are
None — ChatGPT carries no local-workspace facets). Timestamps are ISO strings
(same as cc.py), so ``clync._same_instant`` compares them for incremental sync.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from urllib.parse import urlencode

from curl_cffi import requests as creq
from curl_cffi.requests.exceptions import Timeout

# Reuse cc.py's truncation SSOT — one definition of "how long is a kept title /
# tool one-liner / tool-output head", shared across every source (cc, codex, this).
from cc import _TITLE_MAX, _TOOL_ARG_MAX, _TOOL_RESULT_HEAD

CHATGPT_HOST = "https://chatgpt.com"
IMPERSONATE = "chrome"
# ChatGPT's /backend-api/conversation/{id} endpoint allows a short burst (~15-20
# requests from idle) then throttles hard to roughly one request per ~25-30s, and
# sends NO Retry-After header. So a full history backfill is inherently slow
# (order ~an hour for a few hundred conversations) — the design leans on the
# caller's per-conversation commit + incremental update_time skip to make it
# resumable across runs, rather than trying to outrun a hard server limit.
REQUEST_PACING_S = 1.0          # steady inter-request gap (uses the burst, paces the rest)
_MAX_ATTEMPTS = 20
_MAX_RETRY_WAIT_S = 60.0
# Statuses that are external transients, not a verdict about our request: 403
# (Cloudflare JS challenge), 429 (rate limit), and 5xx from ChatGPT's own backend.
# The 5xx entries are load-bearing: a single 503 {"detail":"Unable to fetch
# authentication verification keys."} — OpenAI's backend failing to fetch its own
# JWKS to verify our (valid) Bearer JWT — used to abort the whole ChatGPT leg on
# the FIRST list page, losing a day of history for a fault entirely on their side.
_RETRY_STATUS = (403, 429, 500, 502, 503, 504)
_LIST_PAGE = 100                # conversations per list page (max the API honours)
SESSION_COOKIE = "__Secure-next-auth.session-token"


# --------------------------------------------------------------------------- #
# Parsed shape — field names mirror CCSession/CCMessage (see module docstring)
# --------------------------------------------------------------------------- #
@dataclass
class ChatGPTMessage:
    msg_id: str                # conversation-node message uuid (unique within unit)
    idx: int
    sender: str                # user | assistant | tool
    role_detail: str           # "text" | "reasoning" | "tool_use" | "tool_result"
    text: str
    created_at: str | None
    embed_text: str | None = None   # "" => not embedded (tool noise); None => reuse text


@dataclass
class ChatGPTConversation:
    session_id: str            # the ChatGPT conversation id (== unit_id)
    title: str | None
    summary: str | None
    created_at: str | None
    updated_at: str | None
    repo: str | None           # always None for ChatGPT (no local workspace)
    cwd: str | None
    worktree: str | None
    git_branch: str | None
    cc_version: str | None
    entrypoint: str | None
    model: str | None
    messages: list[ChatGPTMessage] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Web API client
# --------------------------------------------------------------------------- #
class ChatGPTClient:
    """Authenticated read-only client for the ChatGPT private web API.

    Mirrors ``clync.ClaudeClient``: request pacing + a Cloudflare-403 backoff
    retry, with unauthorized logins and forbidden requests reported separately. The
    accessToken is fetched once (lazily) from the session-token cookie and
    cached for the client's lifetime."""

    def __init__(self, cookies: dict[str, str], profile: str | None = None):
        self._s = creq.Session()
        for name, value in cookies.items():
            self._s.cookies.set(name, value, domain=".chatgpt.com")
        self._profile = profile
        self._token: str | None = None
        self._last_request = 0.0

    def _request(self, url: str, *, bearer: str | None):
        # These read-only GETs can be repeated after an external transport timeout.
        import time
        backoff = 4.0
        headers = {"Accept": "application/json"}
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        for attempt in range(_MAX_ATTEMPTS):
            gap = REQUEST_PACING_S - (time.monotonic() - self._last_request)
            if gap > 0:
                time.sleep(gap)
            try:
                r = self._s.get(url, headers=headers, impersonate=IMPERSONATE, timeout=30)
            except Timeout as e:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"ChatGPT GET {url} timed out on attempt {attempt + 1}/"
                        f"{_MAX_ATTEMPTS}; retry budget exhausted: {e}") from e
                logging.getLogger(__name__).warning(
                    "ChatGPT GET %s timed out (attempt %s/%s); retrying in %ss",
                    url, attempt + 1, _MAX_ATTEMPTS, backoff)
                ra = None
            else:
                if r.status_code == 200:
                    return r
                if r.status_code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS - 1:
                    ra = r.headers.get("Retry-After")
                    logging.getLogger(__name__).warning(
                        "ChatGPT GET %s returned %s (attempt %s/%s; cf-mitigated=%s)",
                        url, r.status_code, attempt + 1, _MAX_ATTEMPTS,
                        r.headers.get("cf-mitigated"))
                elif r.status_code == 401:
                    where = (f"the '{self._profile}' Chrome profile" if self._profile
                             else "Chrome")
                    raise RuntimeError(
                        f"ChatGPT returned 401 — authentication was rejected for GET {url}. "
                        f"Open {CHATGPT_HOST} in {where} and "
                        "LOG IN (a Chrome profile existing does NOT mean it is logged "
                        "into ChatGPT), then re-run.")
                elif r.status_code == 403:
                    reason = ("Cloudflare challenge blocked the request"
                              if r.headers.get("cf-mitigated") == "challenge"
                              else "ChatGPT refused the request")
                    raise RuntimeError(
                        f"ChatGPT returned 403 — {reason} after {_MAX_ATTEMPTS} attempts: "
                        f"GET {url} (content-type={r.headers.get('Content-Type')}). "
                        f"Your Chrome login may still be valid. "
                        f"Response: {r.text[:200]}")
                elif r.status_code == 429:
                    raise RuntimeError(
                        f"ChatGPT rate-limited this sync (429) even after "
                        f"{_MAX_ATTEMPTS} backoff attempts. Wait a few minutes and "
                        f"re-run `clync sync-chatgpt` — per-conversation commits make "
                        f"it resume where it stopped.")
                elif r.status_code >= 500:
                    raise RuntimeError(
                        f"ChatGPT's backend returned {r.status_code} even after "
                        f"{_MAX_ATTEMPTS} backoff attempts — this is a fault on "
                        f"OpenAI's side, not your login: GET {url} -> {r.text[:160]}. "
                        f"Re-run `clync sync-chatgpt` when it recovers; "
                        f"per-conversation commits make it resume where it stopped.")
                else:
                    raise RuntimeError(f"GET {url} -> {r.status_code}: {r.text[:200]}")
            finally:
                self._last_request = time.monotonic()
            if ra and ra.strip().isdigit():
                wait = min(float(ra), _MAX_RETRY_WAIT_S)
            else:
                wait = backoff
                backoff = min(backoff * 2, _MAX_RETRY_WAIT_S)
            time.sleep(wait)

    def _access_token(self) -> str:
        if self._token:
            return self._token
        j = self._request(f"{CHATGPT_HOST}/api/auth/session", bearer=None).json()
        tok = j.get("accessToken")
        if not tok:
            where = (f"the '{self._profile}' Chrome profile" if self._profile
                     else "Chrome")
            raise RuntimeError(
                f"No ChatGPT accessToken from /api/auth/session — {where} is not "
                f"logged in (session response keys: {sorted(j.keys())}). Open "
                f"{CHATGPT_HOST} in that profile and LOG IN, then re-run.")
        self._token = tok
        return tok

    def account_email(self) -> str | None:
        j = self._request(f"{CHATGPT_HOST}/api/auth/session", bearer=None).json()
        return (j.get("user") or {}).get("email")

    def iter_conversations(self):
        conversations = {c["id"]: c for c in self._iter_general_conversations()}
        for item in self._iter_cursor_pages(
                "/gizmos/snorlax/sidebar",
                conversations_per_gizmo=0, owned_only="false", limit=20):
            project = item["gizmo"]["gizmo"]
            pid, name = project["id"], project["display"]["name"]
            for conv in self._iter_cursor_pages(f"/gizmos/{pid}/conversations"):
                conversations[conv["id"]] = {
                    **conv, "project_uuid": pid, "project_name": name}
        yield from conversations.values()

    def _iter_cursor_pages(self, path: str, **params):
        seen = set()
        while True:
            page = self._request(
                f"{CHATGPT_HOST}/backend-api{path}?{urlencode(params)}",
                bearer=self._access_token()).json()
            items, cursor = page["items"], page["cursor"]
            if not isinstance(items, list):
                raise ValueError(f"ChatGPT {path}: items is not a list")
            yield from items
            if cursor is None:
                return
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise ValueError(f"ChatGPT {path}: invalid or repeated pagination cursor")
            seen.add(cursor)
            params["cursor"] = cursor

    def _iter_general_conversations(self):
        tok = self._access_token()
        offset = 0
        while True:
            j = self._request(
                f"{CHATGPT_HOST}/backend-api/conversations"
                f"?offset={offset}&limit={_LIST_PAGE}&order=updated",
                bearer=tok).json()
            items = j["items"]
            if not isinstance(items, list):
                raise ValueError("ChatGPT conversations: items is not a list")
            total = j["total"]
            if not items:
                if offset < total:
                    raise ValueError(
                        f"ChatGPT conversations ended after {offset} of {total} items")
                return
            yield from items
            offset += len(items)
            if offset >= total:
                return

    def get_conversation(self, conv_id: str) -> dict:
        """One conversation's full detail (title, create_time, current_node,
        mapping tree, default_model_slug)."""
        return self._request(
            f"{CHATGPT_HOST}/backend-api/conversation/{conv_id}",
            bearer=self._access_token()).json()


# --------------------------------------------------------------------------- #
# Pure linearizer: conversation-detail dict -> ChatGPTConversation
# --------------------------------------------------------------------------- #
def _iso(t) -> str | None:
    """Normalize a ChatGPT timestamp to an ISO-8601 string (matching cc.py's
    stored shape, so clync._same_instant compares it for incremental sync).

    The two API surfaces disagree on format: the conversation LIST sends ISO-8601
    strings (e.g. '2026-08-15T10:23:55.109384Z'), while the conversation DETAIL
    and per-message create_time send epoch-second floats. Accept both; a value
    that is neither returns None (never a fabricated timestamp)."""
    if t is None:
        return None
    if isinstance(t, str):
        s = t.strip()
        if not s:
            return None
        try:                        # 'Z' suffix: fromisoformat needs +00:00 pre-3.11
            return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return None
    try:
        return datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _text_parts(content: dict) -> str:
    """Join the string parts of a text / multimodal_text content block; image
    (and other non-string) parts are dropped — they carry no searchable text."""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    return "\n".join(p for p in parts if isinstance(p, str)).strip()


def _thoughts_text(content: dict) -> str:
    """A `thoughts` block is a list of {summary, content} reasoning steps; a
    `reasoning_recap` block is a single `content` string."""
    thoughts = content.get("thoughts")
    if isinstance(thoughts, list):
        steps = []
        for t in thoughts:
            if not isinstance(t, dict):
                continue
            piece = (t.get("content") or t.get("summary") or "").strip()
            if piece:
                steps.append(piece)
        return "\n\n".join(steps)
    recap = content.get("content")
    return recap.strip() if isinstance(recap, str) else ""


def _render_message(m: dict) -> ChatGPTMessage | None:
    """One mapping-tree message -> a cleaned ChatGPTMessage, or None to skip it
    (system/hidden turns, and empty tool/placeholder nodes).

    Kept, by role and content_type:
      user   text|multimodal_text          -> sender=user,      role_detail=text
      asst   text|multimodal_text          -> sender=assistant, role_detail=text
      asst   thoughts|reasoning_recap      -> sender=assistant, role_detail=reasoning
      asst   code (a tool call)            -> sender=assistant, role_detail=tool_use  (not embedded)
      tool   any                           -> sender=tool,       role_detail=tool_result (not embedded)
    Tool I/O is truncated and excluded from the embed index (embed_text="") — the
    same policy cc.py/codex.py apply to stdout noise."""
    author = (m.get("author") or {}).get("role")
    content = m.get("content") or {}
    ctype = content.get("content_type")
    meta = m.get("metadata") or {}
    if author in (None, "system"):
        return None
    if meta.get("is_visually_hidden_from_conversation"):
        return None

    mid = m.get("id")
    created = _iso(m.get("create_time"))

    if author == "user":
        if ctype in ("text", "multimodal_text"):
            text = _text_parts(content)
            if not text:
                return None
            return ChatGPTMessage(mid, 0, "user", "text", text, created)
        return None

    if author == "assistant":
        if ctype in ("text", "multimodal_text"):
            text = _text_parts(content)
            if not text:
                return None
            return ChatGPTMessage(mid, 0, "assistant", "text", text, created)
        if ctype in ("thoughts", "reasoning_recap"):
            text = _thoughts_text(content)
            if not text:
                return None
            return ChatGPTMessage(mid, 0, "assistant", "reasoning", text, created)
        if ctype == "code":
            code = (content.get("text") or "").strip()
            if not code:
                return None
            if len(code) > _TOOL_ARG_MAX:
                code = code[:_TOOL_ARG_MAX] + "…"
            return ChatGPTMessage(mid, 0, "assistant", "tool_use", code, created,
                                  embed_text="")
        return None

    if author == "tool":
        body = _text_parts(content) or (content.get("text") or "").strip()
        if not body:
            return None
        if len(body) > _TOOL_RESULT_HEAD:
            body = body[:_TOOL_RESULT_HEAD] + "…"
        return ChatGPTMessage(mid, 0, "tool", "tool_result", body, created,
                              embed_text="")
    return None


def _linear_chain(detail: dict) -> list[dict]:
    """The visible turn order: walk current_node -> parent to the root, reverse.
    A malformed mapping (missing node / a parent cycle) stops the walk rather
    than looping forever."""
    mapping = detail.get("mapping") or {}
    node = detail.get("current_node")
    chain, seen = [], set()
    while node and node not in seen:
        seen.add(node)
        n = mapping.get(node)
        if not n:
            break
        chain.append(n)
        node = n.get("parent")
    chain.reverse()
    return chain


def linearize(detail: dict, meta: dict) -> ChatGPTConversation:
    """Conversation detail (the mapping tree) + its list metadata -> a parsed
    session. `meta` (the list item: id/title/create_time/update_time) is the
    authority for identity + timestamps, because the list's update_time is the
    exact value incremental sync compares against; `detail` supplies the messages
    and model."""
    conv_id = meta["id"]
    messages: list[ChatGPTMessage] = []
    model = detail.get("default_model_slug")
    idx = 0
    for n in _linear_chain(detail):
        m = n.get("message")
        if not m:
            continue
        msg = _render_message(m)
        if msg is None:
            continue
        slug = (m.get("metadata") or {}).get("model_slug")
        if slug:
            model = slug          # last assistant turn's model = the session's model
        msg.msg_id = msg.msg_id or f"{conv_id}#{idx}"
        msg.idx = idx
        messages.append(msg)
        idx += 1

    title = (meta.get("title") or detail.get("title") or "").strip()
    if not title:
        first_user = next((x.text for x in messages if x.sender == "user"), None)
        title = first_user.replace("\n", " ") if first_user else ""
    title = title[:_TITLE_MAX] or None

    return ChatGPTConversation(
        session_id=conv_id, title=title, summary=None,
        created_at=_iso(meta.get("create_time") or detail.get("create_time")),
        updated_at=_iso(meta.get("update_time")),
        repo=None, cwd=None, worktree=None, git_branch=None,
        cc_version=None, entrypoint=None, model=model, messages=messages)
