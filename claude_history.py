"""Headless claude.ai history sync.

Reads the current `sessionKey` cookie live from a named Chrome profile, clears
Cloudflare via curl_cffi TLS impersonation, and upserts conversations + messages
into a local SQLite/FTS5 database. Incremental by `updated_at`.

Fail-loud by contract: any auth/HTTP/schema problem raises and the process exits
non-zero, which the /schedule launchd dispatcher surfaces as a modal+sound alert.
Never degrades silently to stale data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from curl_cffi import requests as creq

PROFILE_ENV = "CLAUDE_HISTORY_PROFILE"  # deployment sets this (e.g. in the launchd plist)
CHROME_DIR = Path.home() / "Library/Application Support/Google/Chrome"
DB_PATH = Path(os.environ.get("CLAUDE_HISTORY_DB",
                              Path.home() / ".local/share/claude-history/history.db"))
BASE = "https://claude.ai/api"
IMPERSONATE = "chrome"
# Cookies the browser sends that matter for auth + Cloudflare clearance. sessionKey
# is the durable account credential; cf_clearance is Cloudflare's per-browser token
# — without it, claude.ai intermittently serves a JS challenge (HTTP 403) under load.
AUTH_COOKIE_NAMES = ("sessionKey", "cf_clearance")
REQUEST_PACING_S = 0.4  # polite gap between requests so a large sync isn't rate-limited


# --------------------------------------------------------------------------- #
# Chrome cookie extraction
# --------------------------------------------------------------------------- #
def _safe_storage_key() -> bytes:
    """Chrome's 'Safe Storage' key from the macOS login keychain."""
    import subprocess

    out = subprocess.run(
        ["security", "find-generic-password", "-ws", "Chrome Safe Storage"],
        capture_output=True, text=True,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(
            "Could not read 'Chrome Safe Storage' from the login keychain "
            f"(rc={out.returncode}). Unlock the keychain and retry."
        )
    password = out.stdout.strip().encode()
    return hashlib.pbkdf2_hmac("sha1", password, b"saltysalt", 1003, 16)


def _decrypt_cookie(enc: bytes, key: bytes) -> str:
    if enc[:3] != b"v10":
        raise RuntimeError(
            f"Cookie encryption prefix {enc[:3]!r} is not the decryptable 'v10' "
            "scheme (app-bound 'v20' is not supported)."
        )
    dec = Cipher(algorithms.AES(key), modes.CBC(b" " * 16)).decryptor()
    plain = dec.update(enc[3:]) + dec.finalize()
    plain = plain[: -plain[-1]]  # strip PKCS7 padding
    for candidate in (plain, plain[32:]):  # newer Chrome prepends a 32-byte host hash
        try:
            s = candidate.decode("utf-8")
            if s.isprintable():
                return s
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Decrypted cookie is not valid UTF-8 text.")


def resolve_profile_dir(display_name: str) -> Path:
    """Map a Chrome profile *display name* (e.g. 'AWS Work') to its directory."""
    local_state = json.loads((CHROME_DIR / "Local State").read_text())
    cache = local_state.get("profile", {}).get("info_cache", {})
    matches = [d for d, info in cache.items() if info.get("name") == display_name]
    if not matches:
        names = sorted(info.get("name") for info in cache.values())
        raise RuntimeError(
            f"No Chrome profile named {display_name!r}. Available: {names}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple profiles named {display_name!r}: {matches}")
    return CHROME_DIR / matches[0]


def read_auth_cookies(profile_display_name: str) -> dict[str, str]:
    """Live-read + decrypt claude.ai auth cookies from a Chrome profile.

    Returns a dict with at least 'sessionKey'; includes 'cf_clearance' when the
    browser currently holds one. Read fresh every run so both are as current as
    the browser's last claude.ai activity.
    """
    profile_dir = resolve_profile_dir(profile_display_name)
    cookies_db = profile_dir / "Cookies"
    if not cookies_db.exists():
        raise RuntimeError(f"No Cookies DB at {cookies_db}")
    key = _safe_storage_key()
    placeholders = ",".join("?" * len(AUTH_COOKIE_NAMES))
    with tempfile.TemporaryDirectory() as tmp:
        # Copy to dodge Chrome's live lock.
        snapshot = Path(tmp) / "Cookies"
        shutil.copy2(cookies_db, snapshot)
        con = sqlite3.connect(snapshot)
        try:
            rows = con.execute(
                "SELECT name, encrypted_value FROM cookies "
                f"WHERE host_key LIKE '%claude.ai%' AND name IN ({placeholders})",
                AUTH_COOKIE_NAMES,
            ).fetchall()
        finally:
            con.close()
    cookies = {name: _decrypt_cookie(val, key) for name, val in rows}
    if "sessionKey" not in cookies:
        raise RuntimeError(
            f"Profile {profile_display_name!r} has no claude.ai sessionKey — "
            "log into claude.ai in that Chrome profile."
        )
    if not cookies["sessionKey"].startswith("sk-ant-sid"):
        raise RuntimeError("Decrypted sessionKey has an unexpected format.")
    return cookies


# --------------------------------------------------------------------------- #
# claude.ai private API client (Cloudflare-passing via TLS impersonation)
# --------------------------------------------------------------------------- #
class ClaudeClient:
    def __init__(self, cookies: dict[str, str]):
        self._s = creq.Session()
        for name, value in cookies.items():
            self._s.cookies.set(name, value, domain=".claude.ai")
        self._last_request = 0.0

    def _get(self, path: str) -> object:
        # Cloudflare (external, not ours) intermittently 403s a JS challenge under
        # load. Pace requests and retry a 403 with backoff; a persistent 403 after
        # retries is a real auth/clearance failure and is raised loudly.
        backoff = 2.0
        for attempt in range(3):
            gap = REQUEST_PACING_S - (time.monotonic() - self._last_request)
            if gap > 0:
                time.sleep(gap)
            r = self._s.get(f"{BASE}{path}", impersonate=IMPERSONATE, timeout=30)
            self._last_request = time.monotonic()
            if r.status_code == 200:
                return r.json()
            if r.status_code == 403 and attempt < 2:
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.status_code in (401, 403):
                raise RuntimeError(
                    f"claude.ai returned {r.status_code} for {path} after retries — "
                    "the sessionKey/cf_clearance is expired or Cloudflare is "
                    "blocking. Open claude.ai in the Chrome profile to refresh, "
                    "then re-run."
                )
            raise RuntimeError(f"GET {path} -> {r.status_code}: {r.text[:200]}")

    def list_orgs(self) -> list[dict]:
        return self._get("/organizations")

    def list_conversations(self, org_uuid: str) -> list[dict]:
        return self._get(f"/organizations/{org_uuid}/chat_conversations")

    def get_conversation(self, org_uuid: str, conv_uuid: str) -> dict:
        return self._get(
            f"/organizations/{org_uuid}/chat_conversations/{conv_uuid}"
            "?tree=True&rendering_mode=messages&render_all_tools=true"
        )


def active_org(client: ClaudeClient) -> dict:
    """The 'first active' org, per the profile's org ordering."""
    orgs = client.list_orgs()
    if not orgs:
        raise RuntimeError("Account has no organizations.")
    return orgs[0]


# --------------------------------------------------------------------------- #
# SQLite / FTS5 storage (single source of truth)
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    uuid           TEXT PRIMARY KEY,
    org_uuid       TEXT NOT NULL,
    name           TEXT,
    summary        TEXT,
    model          TEXT,
    project_uuid   TEXT,
    created_at     TEXT,
    updated_at     TEXT,
    message_count  INTEGER,
    raw            TEXT NOT NULL,
    synced_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    uuid               TEXT PRIMARY KEY,
    conversation_uuid  TEXT NOT NULL REFERENCES conversations(uuid),
    idx                INTEGER,
    sender             TEXT,
    text               TEXT,
    created_at         TEXT,
    raw                TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_uuid);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text,
    conversation_uuid UNINDEXED,
    message_uuid UNINDEXED,
    tokenize='porter unicode61'
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def _message_text(msg: dict) -> str:
    """Concatenate the human-readable text from a message's content blocks."""
    parts: list[str] = []
    for block in msg.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
    if parts:
        return "\n".join(parts)
    return msg.get("text") or ""  # fallback to the rendered top-level text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_conversation(con: sqlite3.Connection, org_uuid: str, full: dict) -> None:
    msgs = full.get("chat_messages") or []
    con.execute(
        """INSERT INTO conversations
           (uuid, org_uuid, name, summary, model, project_uuid,
            created_at, updated_at, message_count, raw, synced_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(uuid) DO UPDATE SET
             name=excluded.name, summary=excluded.summary, model=excluded.model,
             project_uuid=excluded.project_uuid, created_at=excluded.created_at,
             updated_at=excluded.updated_at, message_count=excluded.message_count,
             raw=excluded.raw, synced_at=excluded.synced_at""",
        (
            full["uuid"], org_uuid, full.get("name"), full.get("summary"),
            full.get("model"), full.get("project_uuid"),
            full.get("created_at"), full.get("updated_at"), len(msgs),
            json.dumps(full, ensure_ascii=False), _now(),
        ),
    )
    # Replace this conversation's messages wholesale (handles edits/branch changes).
    con.execute("DELETE FROM messages WHERE conversation_uuid=?", (full["uuid"],))
    con.execute("DELETE FROM messages_fts WHERE conversation_uuid=?", (full["uuid"],))
    for i, m in enumerate(msgs):
        text = _message_text(m)
        con.execute(
            "INSERT INTO messages (uuid, conversation_uuid, idx, sender, text, "
            "created_at, raw) VALUES (?,?,?,?,?,?,?)",
            (m["uuid"], full["uuid"], i, m.get("sender"), text,
             m.get("created_at"), json.dumps(m, ensure_ascii=False)),
        )
        if text:
            con.execute(
                "INSERT INTO messages_fts (text, conversation_uuid, message_uuid) "
                "VALUES (?,?,?)",
                (text, full["uuid"], m["uuid"]),
            )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def _require_profile(args) -> str:
    if not args.profile:
        raise RuntimeError(
            f"No Chrome profile given. Pass --profile NAME or set ${PROFILE_ENV}."
        )
    return args.profile


def cmd_whoami(args) -> int:
    profile = _require_profile(args)
    org = active_org(ClaudeClient(read_auth_cookies(profile)))
    print(f"Chrome profile : {profile}")
    print(f"Active org     : {org.get('name')!r}")
    print(f"Org uuid       : {org['uuid']}")
    print(f"Billing type   : {org.get('billing_type')}")
    return 0


def cmd_sync(args) -> int:
    profile = _require_profile(args)
    client = ClaudeClient(read_auth_cookies(profile))
    org = active_org(client) if not args.org_uuid else {"uuid": args.org_uuid, "name": "(explicit)"}
    org_uuid = org["uuid"]
    print(f"[{_now()}] syncing profile={profile!r} org={org.get('name')!r} ({org_uuid})")

    con = connect()
    stored = dict(con.execute(
        "SELECT uuid, updated_at FROM conversations WHERE org_uuid=?", (org_uuid,)
    ).fetchall())

    remote = client.list_conversations(org_uuid)
    print(f"  remote conversations: {len(remote)} | already stored: {len(stored)}")

    fetched = skipped = 0
    for c in remote:
        cid = c["uuid"]
        if not args.full and stored.get(cid) == c.get("updated_at"):
            skipped += 1
            continue
        full = client.get_conversation(org_uuid, cid)
        upsert_conversation(con, org_uuid, full)
        con.commit()
        fetched += 1
        print(f"  [{fetched}] {c.get('name') or '(untitled)'} "
              f"({len(full.get('chat_messages') or [])} msgs)")

    total_msgs = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    con.close()
    print(f"[done] fetched/updated={fetched} unchanged={skipped} "
          f"total_messages_in_db={total_msgs} db={DB_PATH}")
    return 0


def cmd_search(args) -> int:
    con = connect()
    rows = con.execute(
        """SELECT c.name AS conv, c.uuid AS cuuid,
                  snippet(messages_fts, 0, '[', ']', ' … ', 14) AS snip
           FROM messages_fts
           JOIN conversations c ON c.uuid = messages_fts.conversation_uuid
           WHERE messages_fts MATCH ?
           ORDER BY rank LIMIT ?""",
        (args.query, args.limit),
    ).fetchall()
    if not rows:
        print("(no matches)")
    for r in rows:
        print(f"\n• {r['conv'] or '(untitled)'}  [{r['cuuid']}]\n  {r['snip']}")
    con.close()
    return 0


def cmd_list(args) -> int:
    con = connect()
    rows = con.execute(
        "SELECT name, uuid, updated_at, message_count FROM conversations "
        "ORDER BY updated_at DESC LIMIT ?", (args.limit,)
    ).fetchall()
    for r in rows:
        print(f"{r['updated_at']}  {r['message_count']:>3}msg  "
              f"{r['name'] or '(untitled)'}  [{r['uuid']}]")
    con.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=os.environ.get(PROFILE_ENV),
                   help=f"Chrome profile display name (default: ${PROFILE_ENV})")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("whoami", help="print the resolved account + active org")
    sp.set_defaults(func=cmd_whoami)

    sp = sub.add_parser("sync", help="incremental sync into the local DB")
    sp.add_argument("--org-uuid", help="override the active org")
    sp.add_argument("--full", action="store_true", help="re-fetch every conversation")
    sp.set_defaults(func=cmd_sync)

    sp = sub.add_parser("search", help="FTS5 search over message text")
    sp.add_argument("query")
    sp.add_argument("--limit", type=int, default=10)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("list", help="most recently updated conversations")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_list)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
