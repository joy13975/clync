"""clync — headless claude.ai history sync.

Reads the current auth cookies live from a named Chrome profile, clears
Cloudflare via curl_cffi TLS impersonation, and upserts conversations + messages
(including text-attachment content) into a local SQLite/FTS5 database.
Incremental by `updated_at`.

Fail-loud by contract: any auth/HTTP/schema problem raises and the process exits
non-zero. The scheduled path additionally surfaces failures as a macOS
modal+sound notification. Never degrades silently to stale data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from curl_cffi import requests as creq

PROFILE_ENV = "CLYNC_PROFILE"  # deployment sets this (e.g. in the launchd plist)
CHROME_DIR = Path.home() / "Library/Application Support/Google/Chrome"
DB_PATH = Path(os.environ.get("CLYNC_DB",
                              Path.home() / ".local/share/clync/history.db"))
FILES_DIR = DB_PATH.parent / "files"
HOST = "https://claude.ai"
BASE = f"{HOST}/api"
IMPERSONATE = "chrome"
# content-type -> extension for downloaded image files (claude.ai serves webp)
IMAGE_EXT = {"image/webp": ".webp", "image/png": ".png", "image/jpeg": ".jpg",
             "image/gif": ".gif"}
# Cookies the browser sends that matter for auth + Cloudflare clearance. sessionKey
# is the durable account credential; cf_clearance is Cloudflare's per-browser token
# — without it, claude.ai intermittently serves a JS challenge (HTTP 403) under load.
AUTH_COOKIE_NAMES = ("sessionKey", "cf_clearance")
REQUEST_PACING_S = 0.4  # polite gap between requests so a large sync isn't rate-limited

REPO_DIR = Path(__file__).resolve().parent
LAUNCHD_LABEL = "io.clync.sync"
PLIST_PATH = Path.home() / "Library/LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
SCHED_LOG = DB_PATH.parent / "scheduled.log"
LATE_THRESHOLD_H = 25  # a >25h gap since last success means a daily run was missed


# --------------------------------------------------------------------------- #
# Chrome cookie extraction (auth cookies are per-Chrome-profile, hence --profile)
# --------------------------------------------------------------------------- #
def _safe_storage_key() -> bytes:
    out = subprocess.run(
        ["security", "find-generic-password", "-ws", "Chrome Safe Storage"],
        capture_output=True, text=True,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(
            "Could not read 'Chrome Safe Storage' from the login keychain "
            f"(rc={out.returncode}). Unlock the keychain and retry."
        )
    return hashlib.pbkdf2_hmac("sha1", out.stdout.strip().encode(),
                               b"saltysalt", 1003, 16)


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
    """Map a Chrome profile *display name* (e.g. 'Work') to its directory."""
    local_state = json.loads((CHROME_DIR / "Local State").read_text())
    cache = local_state.get("profile", {}).get("info_cache", {})
    matches = [d for d, info in cache.items() if info.get("name") == display_name]
    if not matches:
        names = sorted(info.get("name") for info in cache.values())
        raise RuntimeError(f"No Chrome profile named {display_name!r}. Available: {names}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple profiles named {display_name!r}: {matches}")
    return CHROME_DIR / matches[0]


def read_auth_cookies(profile_display_name: str) -> dict[str, str]:
    """Live-read + decrypt claude.ai auth cookies from a Chrome profile.

    Read fresh every run, so sessionKey + cf_clearance are always as current as
    the browser's last claude.ai activity — that is the freshness mechanism.
    """
    profile_dir = resolve_profile_dir(profile_display_name)
    cookies_db = profile_dir / "Cookies"
    if not cookies_db.exists():
        raise RuntimeError(f"No Cookies DB at {cookies_db}")
    key = _safe_storage_key()
    placeholders = ",".join("?" * len(AUTH_COOKIE_NAMES))
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / "Cookies"  # copy to dodge Chrome's live lock
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
            f"No claude.ai login found in the {profile_display_name!r} Chrome "
            f"profile. Open {HOST} in that profile and LOG IN, then re-run."
        )
    if not cookies["sessionKey"].startswith("sk-ant-sid"):
        raise RuntimeError("Decrypted sessionKey has an unexpected format.")
    return cookies


# --------------------------------------------------------------------------- #
# claude.ai private API client (Cloudflare-passing via TLS impersonation)
# --------------------------------------------------------------------------- #
class ClaudeClient:
    def __init__(self, cookies: dict[str, str], profile: str | None = None):
        self._s = creq.Session()
        for name, value in cookies.items():
            self._s.cookies.set(name, value, domain=".claude.ai")
        self._last_request = 0.0
        self._profile = profile

    def _request(self, url: str):
        # Cloudflare (external, not ours) intermittently 403s a JS challenge under
        # load. Pace requests and retry a 403 with backoff; a persistent 401/403
        # after retries is a real auth/clearance failure and is raised loudly.
        backoff = 2.0
        for attempt in range(3):
            gap = REQUEST_PACING_S - (time.monotonic() - self._last_request)
            if gap > 0:
                time.sleep(gap)
            r = self._s.get(url, impersonate=IMPERSONATE, timeout=30)
            self._last_request = time.monotonic()
            if r.status_code == 200:
                return r
            if r.status_code == 403 and attempt < 2:
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.status_code in (401, 403):
                where = f"the '{self._profile}' Chrome profile" if self._profile else "Chrome"
                raise RuntimeError(
                    f"claude.ai returned {r.status_code} — your login is expired or "
                    f"Cloudflare is blocking. Open {HOST} in {where} and LOG IN "
                    "(a Chrome profile existing does NOT mean it is logged into "
                    "Claude), then re-run."
                )
            raise RuntimeError(f"GET {url} -> {r.status_code}: {r.text[:200]}")

    def _get(self, path: str) -> object:
        return self._request(f"{BASE}{path}").json()

    def get_file(self, rel_url: str) -> tuple[str, bytes]:
        """Download a file by its relative API url (e.g. .../files/{uuid}/preview)."""
        r = self._request(f"{HOST}{rel_url}")
        return r.headers.get("content-type", "").split(";")[0], r.content

    def list_orgs(self) -> list[dict]:
        return self._get("/organizations")

    def list_conversations(self, org_uuid: str) -> list[dict]:
        return self._get(f"/organizations/{org_uuid}/chat_conversations")

    def get_conversation(self, org_uuid: str, conv_uuid: str) -> dict:
        return self._get(
            f"/organizations/{org_uuid}/chat_conversations/{conv_uuid}"
            "?tree=True&rendering_mode=messages&render_all_tools=true"
        )


def resolve_org(client: ClaudeClient, org_ref: str | None) -> dict:
    """Resolve the target org: an explicit name/uuid, else the first (active) org."""
    orgs = client.list_orgs()
    if not orgs:
        raise RuntimeError("Account has no organizations.")
    if not org_ref:
        return orgs[0]
    for o in orgs:
        if org_ref in (o.get("uuid"), o.get("name")):
            return o
    raise RuntimeError(
        f"No org matching {org_ref!r} in this account. "
        f"Available: {[o.get('name') for o in orgs]}"
    )


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
    text, conversation_uuid UNINDEXED, message_uuid UNINDEXED,
    tokenize='porter unicode61'
);
CREATE TABLE IF NOT EXISTS files (
    file_uuid          TEXT PRIMARY KEY,
    conversation_uuid  TEXT NOT NULL,
    message_uuid       TEXT,
    file_kind          TEXT,
    file_name          TEXT,
    size_bytes         INTEGER,
    local_path         TEXT,          -- set once downloaded (images only)
    created_at         TEXT,
    raw                TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_conv ON files(conversation_uuid);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def get_meta(con: sqlite3.Connection, key: str) -> str | None:
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _message_text(msg: dict) -> str:
    """Human-readable text: message content blocks + any text-attachment content."""
    parts: list[str] = []
    for block in msg.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
    if not parts and msg.get("text"):
        parts.append(msg["text"])  # fallback to rendered top-level text
    for att in msg.get("attachments") or []:
        extracted = att.get("extracted_content")
        if extracted:
            name = att.get("file_name") or "attachment"
            parts.append(f"\n[attachment: {name}]\n{extracted}")
    return "\n".join(parts)


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
        (full["uuid"], org_uuid, full.get("name"), full.get("summary"),
         full.get("model"), full.get("project_uuid"), full.get("created_at"),
         full.get("updated_at"), len(msgs),
         json.dumps(full, ensure_ascii=False), _now()),
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
                "VALUES (?,?,?)", (text, full["uuid"], m["uuid"]),
            )


def sync_files(con: sqlite3.Connection, client: ClaudeClient, conv: dict,
               download: bool) -> int:
    """Record file metadata for a conversation; download image files (not yet
    local) to FILES_DIR. Returns how many files were newly downloaded.

    Binary blobs (file_kind='blob') expose only a server sandbox `path`, no
    download URL, so they are recorded as metadata only — not a silent skip.
    """
    downloaded = 0
    for m in conv.get("chat_messages") or []:
        for f in m.get("files") or []:
            fid = f.get("file_uuid") or f.get("uuid")
            if not fid:
                continue
            row = con.execute("SELECT local_path FROM files WHERE file_uuid=?",
                              (fid,)).fetchone()
            local = row[0] if row else None
            if (download and not local and f.get("file_kind") == "image"
                    and f.get("preview_url")):
                ctype, data = client.get_file(f["preview_url"])  # full-res (webp)
                FILES_DIR.mkdir(parents=True, exist_ok=True)
                path = FILES_DIR / f"{fid}{IMAGE_EXT.get(ctype, '.bin')}"
                path.write_bytes(data)
                local = str(path)
                downloaded += 1
            con.execute(
                """INSERT INTO files (file_uuid, conversation_uuid, message_uuid,
                     file_kind, file_name, size_bytes, local_path, created_at, raw)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(file_uuid) DO UPDATE SET
                     file_kind=excluded.file_kind, file_name=excluded.file_name,
                     size_bytes=excluded.size_bytes, local_path=excluded.local_path,
                     raw=excluded.raw""",
                (fid, conv["uuid"], m.get("uuid"), f.get("file_kind"),
                 f.get("file_name"), f.get("size_bytes"), local,
                 f.get("created_at"), json.dumps(f, ensure_ascii=False)),
            )
    return downloaded


def run_sync(profile: str, org_ref: str | None, full: bool,
             download_files: bool = True) -> dict:
    """Core incremental sync. Returns stats. Raises loudly on any failure."""
    client = ClaudeClient(read_auth_cookies(profile), profile)
    org = resolve_org(client, org_ref)
    org_uuid = org["uuid"]
    print(f"[{_now()}] syncing profile={profile!r} org={org.get('name')!r} ({org_uuid})")

    con = connect()
    stored = dict(con.execute(
        "SELECT uuid, updated_at FROM conversations WHERE org_uuid=?", (org_uuid,)
    ).fetchall())
    remote = client.list_conversations(org_uuid)
    print(f"  remote conversations: {len(remote)} | already stored: {len(stored)}")

    fetched = skipped = files_dl = 0
    for c in remote:
        cid = c["uuid"]
        if not full and stored.get(cid) == c.get("updated_at"):
            skipped += 1
            continue
        full_conv = client.get_conversation(org_uuid, cid)
        upsert_conversation(con, org_uuid, full_conv)
        files_dl += sync_files(con, client, full_conv, download_files)
        con.commit()
        fetched += 1
        print(f"  [{fetched}] {c.get('name') or '(untitled)'} "
              f"({len(full_conv.get('chat_messages') or [])} msgs)")

    set_meta(con, "last_success", _now())
    total_msgs = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    con.commit()
    con.close()
    print(f"[done] fetched/updated={fetched} unchanged={skipped} "
          f"images_downloaded={files_dl} total_messages_in_db={total_msgs} db={DB_PATH}")
    return {"fetched": fetched, "skipped": skipped, "images_downloaded": files_dl,
            "org": org.get("name")}


# --------------------------------------------------------------------------- #
# macOS notifications — technique replicated from the user's validated sched.py
# (detached osascript in a NEW SESSION so a launchd-fired notification survives
# to render; a plain detached Popen is reaped before NotificationCenter shows it).
# --------------------------------------------------------------------------- #
def _osa(script: str) -> None:
    try:
        subprocess.Popen(["osascript", "-e", script], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except Exception as e:  # a notification failure must not mask the real outcome
        print(f"WARN osascript notify failed: {e}", file=sys.stderr)


def _q(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def notify(level: str, title: str, message: str) -> None:
    """level: ok | warn | fail. Banner (with sound) + an OK dialog."""
    sound = "Glass" if level == "ok" else "Sosumi"
    icon = {"fail": "stop", "warn": "caution"}.get(level, "note")
    _osa(f'display notification "{_q(message)}" with title "{_q(title)}" '
         f'sound name "{sound}"')
    _osa(f'display dialog "{_q(message)}" with title "{_q(title)}" '
         f'buttons {{"OK"}} default button "OK" with icon {icon}')


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
    org = resolve_org(ClaudeClient(read_auth_cookies(profile), profile), args.org)
    print(f"Chrome profile : {profile}")
    print(f"Org            : {org.get('name')!r}")
    print(f"Org uuid       : {org['uuid']}")
    print(f"Billing type   : {org.get('billing_type')}")
    return 0


def cmd_sync(args) -> int:
    run_sync(_require_profile(args), args.org, args.full,
             download_files=not args.no_files)
    return 0


def cmd_scheduled(args) -> int:
    """Entry point for the launchd job: sync, notify LOUDLY on failure, and warn
    (but do not error) when a prior daily run was missed because the Mac was down."""
    profile = args.profile or os.environ.get(PROFILE_ENV)
    if not profile:
        notify("fail", "clync sync failed", f"${PROFILE_ENV} is not set")
        raise RuntimeError(f"${PROFILE_ENV} is not set")
    # Detect a missed prior run before this one updates last_success.
    con = connect()
    last = get_meta(con, "last_success")
    con.close()
    late_gap_h = None
    if last:
        gap = datetime.now(timezone.utc) - datetime.fromisoformat(last)
        if gap > timedelta(hours=LATE_THRESHOLD_H):
            late_gap_h = gap.total_seconds() / 3600
    try:
        stats = run_sync(profile, args.org, full=False)
    except Exception as e:
        notify("fail", "clync sync failed", str(e))
        raise
    if late_gap_h is not None:
        notify("warn", "clync ran late",
               f"previous scheduled sync was missed (~{late_gap_h:.0f}h gap) — "
               "Mac was likely asleep/off. Synced now.")
    print(f"[scheduled] ok: {stats}")
    return 0


def cmd_search(args) -> int:
    con = connect()
    rows = con.execute(
        """SELECT c.name AS conv, c.uuid AS cuuid,
                  snippet(messages_fts, 0, '[', ']', ' … ', 14) AS snip
           FROM messages_fts
           JOIN conversations c ON c.uuid = messages_fts.conversation_uuid
           WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?""",
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


def cmd_doctor(args) -> int:
    """Health check: DB, deps/MCP-server import, launchd job, MCP registration."""
    problems: list[str] = []

    if not DB_PATH.exists():
        problems.append(f"DB missing at {DB_PATH} — run `clync sync`.")
    else:
        con = connect()
        nconv = con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        nmsg = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        nfile = con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        ndl = con.execute("SELECT COUNT(*) FROM files WHERE local_path IS NOT NULL"
                          ).fetchone()[0]
        last = get_meta(con, "last_success")
        con.close()
        print(f"DB          : {DB_PATH}\n              {nconv} conversations, "
              f"{nmsg} messages, {nfile} files ({ndl} images downloaded), "
              f"last_success={last}")
        if last:
            gap = datetime.now(timezone.utc) - datetime.fromisoformat(last)
            if gap > timedelta(hours=LATE_THRESHOLD_H):
                problems.append(f"last sync was {gap.total_seconds()/3600:.0f}h ago "
                                f"(> {LATE_THRESHOLD_H}h) — scheduler may not be running.")

    try:
        import mcp_server  # noqa: F401 — self-test that deps resolve + server builds
        print("MCP server  : import OK (deps resolve, FastMCP builds)")
    except Exception as e:
        problems.append(f"MCP server import failed: {e}")

    loaded = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"],
        capture_output=True, text=True,
    ).returncode == 0
    print(f"launchd job : {'loaded' if loaded else 'NOT loaded'} ({LAUNCHD_LABEL})")
    if not loaded:
        problems.append("scheduler not installed — run `clync install`.")

    reg = subprocess.run(["claude", "mcp", "get", "clync"],
                         capture_output=True, text=True)
    if reg.returncode == 0:
        print("MCP register: registered with Claude Code")
    else:
        problems.append("MCP server not registered — see README for `claude mcp add`.")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    print("\nAll green.")
    return 0


def cmd_install(args) -> int:
    """Write + load a self-contained launchd LaunchAgent for the daily sync."""
    profile = _require_profile(args)
    hour, minute = (int(x) for x in args.at.split(":"))
    SCHED_LOG.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    # A login shell resolves `uv` on PATH (pyenv shims aren't reliable under launchd).
    cmd = f"cd {REPO_DIR} && exec uv run python clync.py scheduled"
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key>
    <array><string>/bin/zsh</string><string>-lc</string><string>{cmd}</string></array>
  <key>EnvironmentVariables</key><dict>
    <key>{PROFILE_ENV}</key><string>{profile}</string>
  </dict>
  <key>StartCalendarInterval</key><dict>
    <key>Hour</key><integer>{hour}</integer>
    <key>Minute</key><integer>{minute}</integer>
  </dict>
  <key>StandardOutPath</key><string>{SCHED_LOG}</string>
  <key>StandardErrorPath</key><string>{SCHED_LOG}</string>
</dict></plist>
"""
    PLIST_PATH.write_text(plist)
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LAUNCHD_LABEL}"],
                   capture_output=True, text=True)  # idempotent: ignore "not loaded"
    r = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(PLIST_PATH)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"launchctl bootstrap failed: {r.stderr.strip()}")
    print(f"Installed {LAUNCHD_LABEL}: daily at {hour:02d}:{minute:02d}, "
          f"profile={profile!r}\n  plist: {PLIST_PATH}\n  log:   {SCHED_LOG}")
    print("Missed runs (Mac asleep/off) run once on wake and fire a 'ran late' "
          "warning; failures fire a modal+sound alert.")
    return 0


def cmd_uninstall(args) -> int:
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LAUNCHD_LABEL}"],
                   capture_output=True, text=True)
    if PLIST_PATH.exists():
        PLIST_PATH.unlink()
    print(f"Uninstalled {LAUNCHD_LABEL}.")
    return 0


def cmd_status(args) -> int:
    """Scheduled-sync status: last success, launchd state, recent run log."""
    con = connect()
    last = get_meta(con, "last_success")
    con.close()
    print(f"last successful sync : {last or '(never)'}")
    loaded = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"],
        capture_output=True, text=True,
    ).returncode == 0
    print(f"launchd job          : {'loaded (daily)' if loaded else 'NOT loaded'}")
    if SCHED_LOG.exists():
        lines = SCHED_LOG.read_text(errors="replace").splitlines()
        print(f"\nlast {min(args.tail, len(lines))} scheduled-log lines ({SCHED_LOG}):")
        for ln in lines[-args.tail:]:
            print(f"  {ln}")
    else:
        print(f"\nno scheduled runs logged yet ({SCHED_LOG} absent).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=os.environ.get(PROFILE_ENV),
                   help=f"Chrome profile display name (default: ${PROFILE_ENV})")
    p.add_argument("--org", help="target org name or uuid (default: first/active org)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("whoami", help="print the resolved account + org").set_defaults(func=cmd_whoami)

    sp = sub.add_parser("sync", help="incremental sync into the local DB")
    sp.add_argument("--full", action="store_true", help="re-fetch every conversation")
    sp.add_argument("--no-files", action="store_true", help="skip image downloads")
    sp.set_defaults(func=cmd_sync)

    sub.add_parser("scheduled", help="launchd entry point (sync + loud fail/late notify)"
                   ).set_defaults(func=cmd_scheduled)

    sp = sub.add_parser("search", help="FTS5 search over message + attachment text")
    sp.add_argument("query")
    sp.add_argument("--limit", type=int, default=10)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("list", help="most recently updated conversations")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_list)

    sub.add_parser("doctor", help="health check DB / deps / launchd / MCP").set_defaults(func=cmd_doctor)

    sp = sub.add_parser("install", help="install the daily launchd sync")
    sp.add_argument("--at", default="09:00", help="daily time HH:MM (default 09:00)")
    sp.set_defaults(func=cmd_install)

    sub.add_parser("uninstall", help="remove the launchd sync").set_defaults(func=cmd_uninstall)

    sp = sub.add_parser("status", help="scheduled-sync status + recent run log")
    sp.add_argument("--tail", type=int, default=15, help="log lines to show")
    sp.set_defaults(func=cmd_status)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
