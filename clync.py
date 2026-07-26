"""clync — headless history sync + hybrid search.

Two sources, one contained store (ADR 0003):
  * claude.ai chats — reads auth cookies live from a named Chrome profile, clears
    Cloudflare via curl_cffi TLS impersonation, upserts conversations + project
    docs (incremental by `updated_at`).
  * local Claude Code sessions — cleaned + metadata-extracted from
    `~/.claude/projects/**/*.jsonl` (see cc.py); no network / cookies.

Both land in a contained Postgres 17 + pgvector cluster (single source of truth
AND search index) as unified `units` + `messages`, embedded by search.py.

Fail-loud by contract: any auth/HTTP/schema problem raises and the process exits
non-zero. The scheduled path additionally surfaces failures as a macOS
modal+sound notification. Never degrades silently to stale data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3          # Chrome's cookie store is SQLite; clync's own store is Postgres
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

# Runtime state root (DB cluster, downloaded images, scheduled log). Everything
# lives under one data home; the store itself is the contained Postgres cluster.
DATA_HOME = Path(os.environ.get("CLYNC_DATA_HOME",
                                Path.home() / ".local/share/clync"))
FILES_DIR = DATA_HOME / "files"

# Contained Postgres 17 + pgvector cluster — clync's single source of truth AND
# search index (ADR 0003). A private initdb cluster on a non-default port, fully
# isolated from any system Postgres. pgvector is installed against PG17's share
# dir, so the cluster MUST use those binaries (a PATH `initdb` may lack pgvector).
PG_DIR = DATA_HOME / "pg"
PG_DATA = PG_DIR / "data"
PG_LOG = PG_DIR / "postmaster.log"
PG_BIN = Path(os.environ.get("CLYNC_PG_BIN", "/opt/homebrew/opt/postgresql@17/bin"))
PG_PORT = int(os.environ.get("CLYNC_PG_PORT", "54329"))
PG_DB = os.environ.get("CLYNC_PG_DB", "clync")
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", PG_DB):
    raise SystemExit(f"invalid $CLYNC_PG_DB {PG_DB!r} — must match [A-Za-z_][A-Za-z0-9_]*")
PG_USER = os.environ.get("USER", "postgres")

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
DEFAULT_TOPK = 10       # SSOT for the default result count (CLI, MCP tool, search.TOPK)
# SSOT for the `source` facet values (CLI argparse choices + search validation).
# `all` deliberately means RAW ONLY (claude_ai + claude_code). The Dream layer
# (ADR 0004) is reachable via source='dream' or the dream-first `recall_knowledge`
# surface, never by accident: that keeps every pre-existing query's meaning intact
# AND makes non-circularity the default — the dig cannot retrieve its own output.
VALID_SOURCES = ("all", "claude_ai", "claude_code", "dream")
RAW_SOURCES = ("claude_ai", "claude_code")


def resolve_sources(source: str) -> list[str]:
    """The ONE resolution of a `--source` value to concrete source names.
    `all` means RAW ONLY (see VALID_SOURCES above) — every consumer of the facet
    (`search.py`'s WHERE builder, `cmd_list`) resolves through here, so the
    raw-only rule cannot be true at one surface and false at another."""
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")
    return list(RAW_SOURCES) if source == "all" else [source]

REPO_DIR = Path(__file__).resolve().parent
LAUNCHD_LABEL = "io.clync.sync"
PLIST_PATH = Path.home() / "Library/LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
SCHED_LOG = DATA_HOME / "scheduled.log"
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

    def list_projects(self, org_uuid: str) -> list[dict]:
        return self._get(f"/organizations/{org_uuid}/projects")

    def list_project_docs(self, org_uuid: str, project_uuid: str) -> list[dict]:
        return self._get(f"/organizations/{org_uuid}/projects/{project_uuid}/docs")


CHAT_CAPABILITY = "chat"       # capability token that marks an org as chat-bearing


def resolve_orgs(client: ClaudeClient, org_ref: str | None) -> list[dict]:
    """Which orgs to sync. Explicit --org name/uuid -> just that one. Otherwise ALL
    chat-bearing orgs on the login (a user's history spans every org they chat in;
    syncing only the first silently drops the rest).

    An org is EXCLUDED only when it positively advertises capabilities that do NOT
    include the chat token (e.g. an api-only org, which 403s on chat endpoints). An
    org with an absent/empty capabilities field is INCLUDED, not silently dropped —
    we can't prove it holds no chats, and a spurious chat endpoint just yields zero
    conversations. Every exclusion is printed, so a wrong capability-token
    assumption fails loud instead of silently swallowing a real chat org."""
    orgs = client.list_orgs()
    if not orgs:
        raise RuntimeError("Account has no organizations.")
    if org_ref:
        match = [o for o in orgs if org_ref in (o.get("uuid"), o.get("name"))]
        if not match:
            raise RuntimeError(
                f"No org matching {org_ref!r} in this account. "
                f"Available: {[o.get('name') for o in orgs]}")
        return match
    selected, excluded = [], []
    for o in orgs:
        caps = o.get("capabilities")
        # Include when chat is advertised OR when capabilities are unknown (absent/
        # empty); exclude only a positively-non-chat org.
        if not caps or CHAT_CAPABILITY in caps:
            selected.append(o)
        else:
            excluded.append(o)
    if excluded:
        print(f"[resolve_orgs] excluding {len(excluded)} non-chat org(s) "
              f"(no {CHAT_CAPABILITY!r} capability): {[o.get('name') for o in excluded]}")
    if not selected:
        # Every org positively advertised non-chat capabilities -> don't sync nothing.
        print("[resolve_orgs] no chat-bearing org found; syncing all orgs")
        return orgs
    return selected


# --------------------------------------------------------------------------- #
# Postgres store — single source of truth AND search index (ADR 0003)
# --------------------------------------------------------------------------- #
# The unified retrieval model: `units` (one row per chat / cc session / project
# doc) + `messages` (cleaned per-turn content, msg_id = source event uuid) +
# `chunks` (embeddings — owned by search.py, which pins the vector dims). Facets
# are typed, indexed columns on `units` (and denormalized onto `chunks` for
# no-join filtering). The vector `chunks`/`indexed_units` tables are created by
# search.ensure_index_schema (it owns DENSE_DIM/SPARSE_DIM); this module owns the
# raw store + cluster lifecycle.
RAW_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS units (
    unit_id      text PRIMARY KEY,
    kind         text NOT NULL,          -- chat | project_doc | cc_session
    source       text NOT NULL,          -- claude_ai | claude_code
    title        text,
    summary      text,
    model        text,
    lang         text,
    created_at   timestamptz,
    updated_at   timestamptz,
    org_uuid     text,                   -- claude.ai facets
    project_uuid text,
    project_name text,
    repo         text,                   -- claude_code facets
    cwd          text,
    worktree     text,
    git_branch   text,
    cc_version   text,
    entrypoint   text,
    msg_count    int,
    raw          jsonb,
    synced_at    timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_units_source    ON units(source);
CREATE INDEX IF NOT EXISTS idx_units_project   ON units(project_uuid);
CREATE INDEX IF NOT EXISTS idx_units_pname     ON units(project_name);
CREATE INDEX IF NOT EXISTS idx_units_repo      ON units(repo);
CREATE INDEX IF NOT EXISTS idx_units_worktree  ON units(worktree);
CREATE INDEX IF NOT EXISTS idx_units_branch    ON units(git_branch);
CREATE INDEX IF NOT EXISTS idx_units_updated   ON units(updated_at);
CREATE INDEX IF NOT EXISTS idx_units_title_trgm ON units USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_units_repo_trgm  ON units USING gin (repo gin_trgm_ops);

-- Message identity is per-unit: (unit_id, msg_id). A cc resume replays the
-- original session's event uuids into the resumed session's own unit, and each
-- unit stores its COMPLETE transcript (get_conversation must never start
-- mid-stream). The PK doubles as the unit_id lookup index.
CREATE TABLE IF NOT EXISTS messages (
    unit_id     text NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
    msg_id      text NOT NULL,           -- source event uuid
    idx         int,
    sender      text,
    role_detail text,
    text        text,
    embed_text  text,                    -- tighter embed string (NULL => reuse text)
    created_at  timestamptz,
    raw         jsonb,
    PRIMARY KEY (unit_id, msg_id)
);

CREATE TABLE IF NOT EXISTS projects (
    uuid        text PRIMARY KEY,
    org_uuid    text NOT NULL,
    name        text,
    description text,
    created_at  timestamptz,
    updated_at  timestamptz,
    raw         jsonb,
    synced_at   timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    file_uuid    text PRIMARY KEY,
    unit_id      text NOT NULL,
    message_uuid text,
    file_kind    text,
    file_name    text,
    size_bytes   bigint,
    local_path   text,                   -- set once downloaded (images only)
    created_at   timestamptz,
    raw          jsonb
);
CREATE INDEX IF NOT EXISTS idx_files_unit ON files(unit_id);

-- Per-file incremental watermark for the local Claude Code ingest: a transcript
-- is re-parsed only when its mtime/size changes.
CREATE TABLE IF NOT EXISTS cc_sync_state (
    file_path  text PRIMARY KEY,
    mtime      double precision,
    size       bigint,
    session_id text,
    synced_at  timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (key text PRIMARY KEY, value text);
"""


# --- Cluster lifecycle (contained; never touches a system cluster) ---------- #
def _pg(binary: str) -> str:
    path = PG_BIN / binary
    if not path.exists():
        raise SystemExit(
            f"Postgres 17 binary not found: {path}\n"
            f"clync runs its own PG17+pgvector cluster. Install with:\n"
            f"  brew install postgresql@17 pgvector\n"
            f"or set $CLYNC_PG_BIN to the PG17 bin dir.")
    return str(path)


def _vector_control_present() -> bool:
    share = PG_BIN.parent / "share" / "postgresql@17" / "extension" / "vector.control"
    alt = Path("/opt/homebrew/share/postgresql@17/extension/vector.control")
    return share.exists() or alt.exists()


def cluster_running() -> bool:
    r = subprocess.run([_pg("pg_ctl"), "-D", str(PG_DATA), "status"],
                       capture_output=True, text=True)
    return r.returncode == 0


def start_cluster() -> None:
    if cluster_running():
        return
    PG_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [_pg("pg_ctl"), "-D", str(PG_DATA), "-l", str(PG_LOG),
         "-o", f"-p {PG_PORT} -c listen_addresses=localhost "
               f"-c unix_socket_directories={PG_DIR}",
         "-w", "start"],
        check=True)


def stop_cluster() -> bool:
    """Stop the contained cluster if running. True iff a running cluster was
    stopped; False (no-op) if there's no cluster data or the PG17 binary is
    absent — not an error."""
    if not (PG_DATA / "PG_VERSION").exists() or not (PG_BIN / "pg_ctl").exists():
        return False
    if not cluster_running():
        return False
    subprocess.run([_pg("pg_ctl"), "-D", str(PG_DATA), "-m", "fast", "stop"],
                   check=True)
    return True


def connect_pg():
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.string import StrBinaryDumper, StrDumperUnknown

    # Postgres cannot store NUL (0x00) in `text`, and NUL is never meaningful
    # content (it turns up in raw tool stdout / pasted binary). Registering
    # NUL-stripping str dumpers on every clync connection makes EVERY text
    # parameter share one sanitization chokepoint by construction — no call
    # site can forget. (jsonb has the same rule; `_json` scrubs its payloads.)
    # The bases mirror psycopg's own defaults for str — StrDumperUnknown keeps
    # oid 0 so the server still infers param types (timestamptz etc.).
    class _NulFreeStr(StrDumperUnknown):
        def dump(self, obj):
            return super().dump(obj.replace("\x00", ""))

    class _NulFreeStrBinary(StrBinaryDumper):
        def dump(self, obj):
            return super().dump(obj.replace("\x00", ""))

    con = psycopg.connect(host="localhost", port=PG_PORT, dbname=PG_DB,
                          user=PG_USER, row_factory=dict_row)
    # Binary first, text second: each registration also claims the AUTO slot,
    # and AUTO must stay on the unknown-oid text dumper (psycopg's default).
    con.adapters.register_dumper(str, _NulFreeStrBinary)
    con.adapters.register_dumper(str, _NulFreeStr)
    return con


def sql_statements(script: str):
    """Split a DDL script into individual statements (psycopg runs one per execute).

    Line comments are stripped FIRST: a `;` inside a `-- comment` is prose, not a
    statement terminator, and splitting naively on `;` turns the comment's tail
    into a bare statement and a syntax error. SSOT for both schema scripts
    (`RAW_SCHEMA` here, `search.INDEX_SCHEMA`)."""
    stripped = "\n".join(line.split("--", 1)[0] for line in script.splitlines())
    return [s for s in (stmt.strip() for stmt in stripped.split(";")) if s]


def ensure_cluster() -> None:
    """Idempotent: initdb (if absent) -> start -> create db + extensions + raw
    schema. Also ensures the search index schema so a single call fully provisions
    the store. Fails loud on a missing PG17/pgvector toolchain."""
    if not _vector_control_present():
        raise SystemExit(
            "pgvector not found for PG17. Install with:  brew install pgvector")
    if not (PG_DATA / "PG_VERSION").exists():
        PG_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [_pg("initdb"), "-D", str(PG_DATA), "-U", PG_USER,
             "--encoding=UTF8", "--locale=en_US.UTF-8", "-A", "trust"],
            check=True, capture_output=True)
    start_cluster()
    exists = subprocess.run(
        [_pg("psql"), "-h", "localhost", "-p", str(PG_PORT), "-d", "postgres",
         "-tAc", f"SELECT 1 FROM pg_database WHERE datname='{PG_DB}'"],
        capture_output=True, text=True, check=True).stdout.strip()
    if exists != "1":
        subprocess.run([_pg("psql"), "-h", "localhost", "-p", str(PG_PORT),
                        "-d", "postgres", "-c", f'CREATE DATABASE "{PG_DB}"'],
                       check=True, capture_output=True)
    with connect_pg() as con:
        for stmt in sql_statements(RAW_SCHEMA):
            con.execute(stmt)
        con.commit()
    # Each derived layer owns its own tables + their stale-shape guard.
    import search
    search.ensure_index_schema()        # chunks/indexed_units (search owns the dims)
    import dream
    dream.ensure_dream_schema()         # dream_* (dream owns the stance vocabularies)


def connect():
    """A live connection to the contained store (assumes the cluster is up —
    callers provision it once via ensure_cluster at command entry)."""
    return connect_pg()


def get_meta(con, key: str) -> str | None:
    row = con.execute("SELECT value FROM meta WHERE key=%s", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(con, key: str, value: str) -> None:
    con.execute("INSERT INTO meta(key,value) VALUES(%s,%s) "
                "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key, value))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scrub_nul(obj):
    """Recursively strip NUL (0x00) from every string in a JSON-shaped value.
    Postgres jsonb rejects \\u0000 exactly as `text` rejects 0x00, and claude.ai
    payloads can carry it (pasted binary, attachment extracted_content)."""
    if isinstance(obj, str):
        return obj.replace("\x00", "")
    if isinstance(obj, list):
        return [_scrub_nul(v) for v in obj]
    if isinstance(obj, dict):
        return {_scrub_nul(k): _scrub_nul(v) for k, v in obj.items()}
    return obj


def _json(obj):
    """The jsonb parameter chokepoint: every stored jsonb value goes through here,
    NUL-scrubbed (text parameters are scrubbed by the connection's str dumpers —
    see connect_pg — so ALL stored strings share the same sanitization rule)."""
    from psycopg.types.json import Json
    return Json(_scrub_nul(obj))


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


def _replace_unit_messages(con, unit_id: str, rows: list[tuple]) -> None:
    """Wholesale-replace one unit's messages (handles edits / branch changes).
    Each row is (msg_id, idx, sender, role_detail, text, embed_text, created_at, raw).
    Message identity is per-unit — a cc resume replaying another session's event
    uuids stores them again under its own unit, so every unit's transcript is
    complete. ON CONFLICT guards only intra-unit duplicate uuids (first kept).
    Also the sole writer of units.msg_count: set to the rows actually stored, so
    it always equals what a reader can retrieve."""
    con.execute("DELETE FROM messages WHERE unit_id=%s", (unit_id,))
    for r in rows:
        con.execute(
            "INSERT INTO messages (msg_id, unit_id, idx, sender, role_detail, text, "
            "embed_text, created_at, raw) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (unit_id, msg_id) DO NOTHING",
            (r[0], unit_id, r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
    con.execute(
        "UPDATE units SET msg_count = "
        "(SELECT COUNT(*) FROM messages WHERE unit_id=%s) WHERE unit_id=%s",
        (unit_id, unit_id))


def upsert_conversation(con, org_uuid: str, full: dict) -> None:
    """A claude.ai chat -> one `units` row (kind=chat) + its `messages`."""
    msgs = full.get("chat_messages") or []
    con.execute(
        """INSERT INTO units
             (unit_id, kind, source, title, summary, model, project_uuid,
              created_at, updated_at, org_uuid, raw, synced_at)
           VALUES (%s,'chat','claude_ai',%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT(unit_id) DO UPDATE SET
             title=EXCLUDED.title, summary=EXCLUDED.summary, model=EXCLUDED.model,
             project_uuid=EXCLUDED.project_uuid, created_at=EXCLUDED.created_at,
             updated_at=EXCLUDED.updated_at, org_uuid=EXCLUDED.org_uuid,
             raw=EXCLUDED.raw, synced_at=EXCLUDED.synced_at""",
        (full["uuid"], full.get("name"), full.get("summary"), full.get("model"),
         full.get("project_uuid"), full.get("created_at"), full.get("updated_at"),
         org_uuid, _json(full), _now()),
    )
    rows = [(m["uuid"], i, m.get("sender"), None, _message_text(m), None,
             m.get("created_at"), _json(m)) for i, m in enumerate(msgs)]
    _replace_unit_messages(con, full["uuid"], rows)


def sync_files(con, client: ClaudeClient, conv: dict, download: bool) -> int:
    """Record file metadata for a conversation's unit; download image files (not
    yet local) to FILES_DIR. Returns how many files were newly downloaded.

    Binary blobs (file_kind='blob') expose only a server sandbox `path`, no
    download URL, so they are recorded as metadata only — not a silent skip.
    """
    downloaded = 0
    for m in conv.get("chat_messages") or []:
        for f in m.get("files") or []:
            fid = f.get("file_uuid") or f.get("uuid")
            if not fid:
                continue
            row = con.execute("SELECT local_path FROM files WHERE file_uuid=%s",
                              (fid,)).fetchone()
            local = row["local_path"] if row else None
            if (download and not local and f.get("file_kind") == "image"
                    and f.get("preview_url")):
                ctype, data = client.get_file(f["preview_url"])  # full-res (webp)
                FILES_DIR.mkdir(parents=True, exist_ok=True)
                path = FILES_DIR / f"{fid}{IMAGE_EXT.get(ctype, '.bin')}"
                path.write_bytes(data)
                local = str(path)
                downloaded += 1
            con.execute(
                """INSERT INTO files (file_uuid, unit_id, message_uuid,
                     file_kind, file_name, size_bytes, local_path, created_at, raw)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT(file_uuid) DO UPDATE SET
                     file_kind=EXCLUDED.file_kind, file_name=EXCLUDED.file_name,
                     size_bytes=EXCLUDED.size_bytes, local_path=EXCLUDED.local_path,
                     raw=EXCLUDED.raw""",
                (fid, conv["uuid"], m.get("uuid"), f.get("file_kind"),
                 f.get("file_name"), f.get("size_bytes"), local,
                 f.get("created_at"), _json(f)),
            )
    return downloaded


def upsert_project(con, org_uuid: str, p: dict) -> None:
    con.execute(
        """INSERT INTO projects
             (uuid, org_uuid, name, description, created_at, updated_at, raw, synced_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT(uuid) DO UPDATE SET
             name=EXCLUDED.name, description=EXCLUDED.description,
             created_at=EXCLUDED.created_at, updated_at=EXCLUDED.updated_at,
             raw=EXCLUDED.raw, synced_at=EXCLUDED.synced_at""",
        (p["uuid"], org_uuid, p.get("name"), p.get("description"),
         p.get("created_at"), p.get("updated_at"), _json(p), _now()),
    )


def upsert_project_docs(con, org_uuid: str, project: dict, docs: list[dict]) -> None:
    """Replace a surviving project's knowledge docs wholesale, each as a
    `units` row (kind=project_doc) + one `messages` row holding its content — so
    docs are retrievable alongside chats with no query-side special-casing. A
    doc deleted upstream disappears (wholesale replace); a whole deleted project
    is purged in _sync_org."""
    pid, pname = project["uuid"], project.get("name")
    existing = [r["unit_id"] for r in con.execute(
        "SELECT unit_id FROM units WHERE kind='project_doc' AND project_uuid=%s",
        (pid,)).fetchall()]
    live = {d["uuid"] for d in docs}
    for gone in (u for u in existing if u not in live):
        con.execute("DELETE FROM units WHERE unit_id=%s", (gone,))   # cascades messages
    for d in docs:
        con.execute(
            """INSERT INTO units
                 (unit_id, kind, source, title, created_at, org_uuid,
                  project_uuid, project_name, raw, synced_at)
               VALUES (%s,'project_doc','claude_ai',%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT(unit_id) DO UPDATE SET
                 title=EXCLUDED.title, created_at=EXCLUDED.created_at,
                 project_name=EXCLUDED.project_name, raw=EXCLUDED.raw,
                 synced_at=EXCLUDED.synced_at""",
            (d["uuid"], d.get("file_name"), d.get("created_at"), org_uuid,
             pid, pname, _json(d), _now()),
        )
        _replace_unit_messages(con, d["uuid"], [(
            f"{d['uuid']}#0", 0, "project_doc", "doc",
            d.get("content") or "", None, d.get("created_at"), _json(d))])


def run_sync(profile: str, org_ref: str | None, full: bool,
             download_files: bool = True) -> dict:
    """Core incremental claude.ai sync across ALL target orgs (see resolve_orgs).
    Returns aggregate stats. Raises loudly on any failure. (Local Claude Code
    ingest is a separate, cookie-independent step — see ingest_cc.)"""
    client = ClaudeClient(read_auth_cookies(profile), profile)
    orgs = resolve_orgs(client, org_ref)
    print(f"[{_now()}] syncing profile={profile!r} "
          f"orgs={[o.get('name') for o in orgs]}")

    con = connect()
    totals = {"fetched": 0, "skipped": 0, "images_downloaded": 0,
              "projects": 0, "project_docs": 0, "orgs": []}
    try:
        for org in orgs:
            _sync_org(con, client, org, full, download_files, totals)
        set_meta(con, "last_success", _now())
        con.commit()
        total_msgs = con.execute(
            "SELECT COUNT(*) FROM messages m JOIN units u ON u.unit_id=m.unit_id "
            "WHERE u.source='claude_ai'").fetchone()["count"]
    finally:
        con.close()
    print(f"[done] fetched/updated={totals['fetched']} unchanged={totals['skipped']} "
          f"images_downloaded={totals['images_downloaded']} "
          f"projects={totals['projects']} project_docs={totals['project_docs']} "
          f"claude_ai_messages_in_db={total_msgs} db={PG_DB}@localhost:{PG_PORT}")
    return totals


def _sync_org(con, client: ClaudeClient, org: dict,
              full: bool, download_files: bool, totals: dict) -> None:
    """Sync one org: its conversations (incl. project-associated chats, which the
    chat_conversations listing already returns) and its projects + knowledge docs."""
    org_uuid = org["uuid"]
    print(f"  org {org.get('name')!r} ({org_uuid}):")
    stored = {r["unit_id"]: r["updated_at"] for r in con.execute(
        "SELECT unit_id, updated_at FROM units "
        "WHERE kind='chat' AND org_uuid=%s", (org_uuid,)).fetchall()}
    remote = client.list_conversations(org_uuid)
    print(f"    conversations: {len(remote)} remote | {len(stored)} already stored")
    fetched = skipped = files_dl = 0
    for c in remote:
        cid = c["uuid"]
        # stored updated_at is a timestamptz; compare on the instant, not the string.
        if not full and cid in stored and _same_instant(stored[cid], c.get("updated_at")):
            skipped += 1
            continue
        full_conv = client.get_conversation(org_uuid, cid)
        upsert_conversation(con, org_uuid, full_conv)
        files_dl += sync_files(con, client, full_conv, download_files)
        con.commit()
        fetched += 1

    projects = client.list_projects(org_uuid)
    ndocs = 0
    for p in projects:
        upsert_project(con, org_uuid, p)
        docs = client.list_project_docs(org_uuid, p["uuid"])
        upsert_project_docs(con, org_uuid, p, docs)
        ndocs += len(docs)
    # Backfill project_name onto this org's chat units (a chat's project may have
    # been synced only just now), so the "in project X" facet is queryable.
    con.execute(
        "UPDATE units u SET project_name = p.name FROM projects p "
        "WHERE u.project_uuid = p.uuid AND u.org_uuid = %s AND u.kind='chat'",
        (org_uuid,))
    # Purge projects (and their doc units) deleted upstream.
    live = {p["uuid"] for p in projects}
    stored_projects = [r["uuid"] for r in con.execute(
        "SELECT uuid FROM projects WHERE org_uuid=%s", (org_uuid,)).fetchall()]
    for pid in (x for x in stored_projects if x not in live):
        con.execute("DELETE FROM units WHERE kind='project_doc' AND project_uuid=%s",
                    (pid,))    # cascades their messages
        con.execute("DELETE FROM projects WHERE uuid=%s", (pid,))
    con.commit()
    print(f"    fetched/updated={fetched} unchanged={skipped} images={files_dl} "
          f"projects={len(projects)} project_docs={ndocs}")
    for k, v in (("fetched", fetched), ("skipped", skipped),
                 ("images_downloaded", files_dl), ("projects", len(projects)),
                 ("project_docs", ndocs)):
        totals[k] += v
    totals["orgs"].append(org.get("name"))


def _same_instant(stored, remote_iso: str | None) -> bool:
    """True iff a stored timestamptz equals a remote ISO timestamp (same instant).
    `stored` is a datetime from Postgres; comparing instants avoids string-format
    drift that would make every conversation look changed."""
    if not remote_iso or stored is None:
        return False
    try:
        return stored == datetime.fromisoformat(remote_iso)
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Local Claude Code session ingest (cookie-independent; see cc.py + ADR 0003)
# --------------------------------------------------------------------------- #
def _upsert_cc_session(con, s) -> None:
    """Write one parsed CCSession -> a `units` row (kind=cc_session) + its
    cleaned `messages` (wholesale-replaced; per-unit identity)."""
    con.execute(
        """INSERT INTO units
             (unit_id, kind, source, title, summary, model, created_at, updated_at,
              repo, cwd, worktree, git_branch, cc_version, entrypoint,
              raw, synced_at)
           VALUES (%s,'cc_session','claude_code',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT(unit_id) DO UPDATE SET
             title=EXCLUDED.title, summary=EXCLUDED.summary, model=EXCLUDED.model,
             created_at=EXCLUDED.created_at, updated_at=EXCLUDED.updated_at,
             repo=EXCLUDED.repo, cwd=EXCLUDED.cwd, worktree=EXCLUDED.worktree,
             git_branch=EXCLUDED.git_branch, cc_version=EXCLUDED.cc_version,
             entrypoint=EXCLUDED.entrypoint,
             raw=EXCLUDED.raw, synced_at=EXCLUDED.synced_at""",
        (s.session_id, s.title, s.summary, s.model,
         s.created_at, s.updated_at, s.repo, s.cwd, s.worktree, s.git_branch,
         s.cc_version, s.entrypoint,
         _json({"session_id": s.session_id, "cwd": s.cwd, "repo": s.repo,
                "worktree": s.worktree, "git_branch": s.git_branch,
                "cc_version": s.cc_version, "entrypoint": s.entrypoint}), _now()),
    )
    rows = [(m.msg_id, m.idx, m.sender, m.role_detail, m.text, m.embed_text,
             m.created_at, None) for m in s.messages]
    _replace_unit_messages(con, s.session_id, rows)


def ingest_cc(full: bool = False) -> dict:
    """Ingest local Claude Code transcripts into the store. Incremental by file
    mtime/size (a `cc_sync_state` watermark); `full` re-parses every file. Also
    reconciles deletions: a transcript removed from disk loses its unit/messages
    and its watermark row (the mirror of the claude.ai upstream-deletion purge —
    never degrade silently to stale data). An unavailable transcript root raises
    (in `cc.iter_session_files`, before any store mutation) — a missing source
    is an error, never evidence of deletion. No network / cookies. Returns
    stats. Fails loud on a store error."""
    import cc
    files = list(cc.iter_session_files())  # raises on a missing root, before the store is touched
    con = connect()
    state = {r["file_path"]: (r["mtime"], r["size"]) for r in con.execute(
        "SELECT file_path, mtime, size FROM cc_sync_state").fetchall()}
    ingested = skipped = sessions = 0
    try:
        for path in files:
            st = path.stat()
            key = str(path)
            if not full and state.get(key) == (st.st_mtime, st.st_size):
                skipped += 1
                continue
            parsed = cc.parse_session(path)
            if parsed is not None:
                _upsert_cc_session(con, parsed)
                sessions += 1
            con.execute(
                "INSERT INTO cc_sync_state (file_path, mtime, size, session_id, synced_at) "
                "VALUES (%s,%s,%s,%s,%s) ON CONFLICT(file_path) DO UPDATE SET "
                "mtime=EXCLUDED.mtime, size=EXCLUDED.size, "
                "session_id=EXCLUDED.session_id, synced_at=EXCLUDED.synced_at",
                (key, st.st_mtime, st.st_size,
                 parsed.session_id if parsed else None, _now()))
            con.commit()
            ingested += 1
        # Reconcile deletions: drop watermark rows for files no longer on disk,
        # then every cc unit left without ANY backing file (FK cascades its
        # messages; build_index drops its chunks via the `gone` path).
        on_disk = {str(p) for p in files}
        dead_paths = [fp for fp in state if fp not in on_disk]
        if dead_paths:
            con.execute("DELETE FROM cc_sync_state WHERE file_path = ANY(%s)",
                        (dead_paths,))
        removed = len(con.execute(
            "DELETE FROM units u WHERE u.kind='cc_session' AND NOT EXISTS "
            "(SELECT 1 FROM cc_sync_state s WHERE s.session_id = u.unit_id) "
            "RETURNING u.unit_id").fetchall())
        con.commit()
        total = con.execute(
            "SELECT COUNT(*) FROM units WHERE source='claude_code'").fetchone()["count"]
    finally:
        con.close()
    print(f"[cc] files_parsed={ingested} unchanged={skipped} "
          f"cli_sessions_ingested={sessions} sessions_removed={removed} "
          f"cc_sessions_in_db={total}")
    return {"files_parsed": ingested, "unchanged": skipped,
            "sessions_ingested": sessions, "sessions_removed": removed,
            "cc_sessions_in_db": total}


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
    orgs = resolve_orgs(ClaudeClient(read_auth_cookies(profile), profile), args.org)
    print(f"Chrome profile : {profile}")
    print(f"Orgs to sync   : {len(orgs)}")
    for org in orgs:
        print(f"  - {org.get('name')!r}  uuid={org['uuid']}  "
              f"billing={org.get('billing_type')}  caps={org.get('capabilities')}")
    return 0


def _index_after_sync(full: bool) -> None:
    """Refresh the hybrid-search index after a sync. Search is core (ADR 0003):
    a broken index fails loud (never a silent skip)."""
    import search
    try:
        stats = search.build_index(full=full)
    except Exception as e:
        notify("fail", "clync index failed", str(e))
        raise
    print(f"[index] reindexed_units={stats['reindexed_units']} "
          f"chunks={stats['chunks']} removed_units={stats['removed_units']}")


def _sync_all(profile: str, org_ref: str | None, full: bool, download_files: bool,
              do_index: bool, notify_fail: bool) -> Exception | None:
    """Run claude.ai sync (cookie/network — may fail) then ALWAYS the local
    Claude Code ingest (no network), then index. claude.ai failure is captured and
    RETURNED for the caller to re-raise (fail-loud) — but only after cc + index run,
    so a stale/expired cookie never blocks local-session capture. A cc-ingest or
    index failure is not captured — it raises immediately (notifying first on the
    scheduled path, so an unattended failure is never silent)."""
    ensure_cluster()
    err: Exception | None = None
    try:
        run_sync(profile, org_ref, full, download_files=download_files)
    except Exception as e:                    # captured, re-raised by caller (loud)
        if notify_fail:
            notify("fail", "clync sync failed", str(e))
        print(f"claude.ai sync FAILED: {e}\n  -> continuing with local Claude Code "
              f"ingest (cookie-independent)", file=sys.stderr)
        err = e
    try:
        ingest_cc(full=full)
    except Exception as e:
        if notify_fail:
            notify("fail", "clync cc ingest failed", str(e))
        raise
    if do_index:
        _index_after_sync(full)
    return err


def cmd_sync(args) -> int:
    err = _sync_all(_require_profile(args), args.org, args.full,
                    download_files=not args.no_files, do_index=not args.no_index,
                    notify_fail=False)
    if err:
        raise err
    return 0


def cmd_sync_app(args) -> int:
    """Sync claude.ai (the app) ONLY — network/cookies, fails loud — then index.
    No local Claude Code ingest."""
    ensure_cluster()
    run_sync(_require_profile(args), args.org, args.full,
             download_files=not args.no_files)
    if not args.no_index:
        _index_after_sync(args.full)
    return 0


def cmd_sync_cc(args) -> int:
    """Ingest local Claude Code sessions ONLY (no network/cookies), then index."""
    ensure_cluster()
    ingest_cc(full=args.full)
    if not args.no_index:
        _index_after_sync(args.full)
    return 0


def cmd_index(args) -> int:
    import search
    try:
        stats = search.build_index(full=args.full)
    except Exception as e:
        notify("fail", "clync index failed", str(e))
        raise
    print(f"reindexed_units={stats['reindexed_units']} chunks={stats['chunks']} "
          f"removed_units={stats['removed_units']}")
    return 0


def cmd_scheduled(args) -> int:
    """Entry point for the launchd job: sync, notify LOUDLY on failure, and warn
    (but do not error) when a prior daily run was missed because the Mac was down."""
    profile = args.profile or os.environ.get(PROFILE_ENV)
    if not profile:
        notify("fail", "clync sync failed", f"${PROFILE_ENV} is not set")
        raise RuntimeError(f"${PROFILE_ENV} is not set")
    ensure_cluster()
    # Detect a missed prior run before this one updates last_success.
    con = connect()
    try:
        last = get_meta(con, "last_success")
    finally:
        con.close()
    late_gap_h = None
    if last:
        gap = datetime.now(timezone.utc) - datetime.fromisoformat(last)
        if gap > timedelta(hours=LATE_THRESHOLD_H):
            late_gap_h = gap.total_seconds() / 3600
    err = _sync_all(profile, args.org, full=False, download_files=True,
                    do_index=True, notify_fail=True)
    if late_gap_h is not None:
        notify("warn", "clync ran late",
               f"previous scheduled sync was missed (~{late_gap_h:.0f}h gap) — "
               "Mac was likely asleep/off. Synced now.")
    if err:
        raise err
    # The nightly dream pass runs INSIDE this job, immediately after sync+index,
    # rather than as a second launchd job at a later hour: the dig is only as good
    # as the index it queries, and sequencing it here makes that ordering true by
    # construction instead of depending on the sync having finished by some
    # guessed-at second wake time. A sync failure above already raised, so the
    # index is current whenever this line is reached.
    import dream
    con = connect()
    try:
        report = dream.run_incremental(con)
    except dream.DreamThrottled as e:
        # Expected and resumable: the subscription's limit is a fact of the
        # substrate, not a defect. State is in dream_queue; the next run continues.
        notify("warn", "clync dream throttled", str(e))
        print(f"[scheduled] sync ok; dream throttled: {e}")
        return 0
    except Exception as e:
        notify("fail", "clync dream failed", str(e))
        raise
    finally:
        con.close()
    print(f"[scheduled] ok; dream calls={report['calls']} "
          f"changed={report['changed']} topics={report['topics_touched']}")
    return 0


def cmd_search(args) -> int:
    import search
    results = search.hybrid_search(
        args.query, topk=args.limit, source=args.source, project=args.project,
        model=args.model, repo=args.repo, worktree=args.worktree, branch=args.branch,
        session=args.session, since=args.since, until=args.until, sort=args.sort,
        lang=args.lang)
    if not results:
        print("(no matches)")
    for r in results:
        snippet = r["text"].replace("\n", " ")[:200]
        print(f"\n• [{r['source']}] {r['unit_name'] or '(untitled)'}  "
              f"[{r['unit_id']}]  score={r['score']:.4f}\n  {snippet}")
    return 0


def cmd_list(args) -> int:
    ensure_cluster()
    con = connect()
    try:
        rows = con.execute(
            "SELECT title, unit_id, source, updated_at, msg_count FROM units "
            "WHERE source = ANY(%s) "
            "ORDER BY updated_at DESC NULLS LAST LIMIT %s",
            (resolve_sources(args.source), args.limit)).fetchall()
    finally:
        con.close()
    for r in rows:
        n = r["msg_count"] or 0
        print(f"{r['updated_at']}  {n:>3}msg  [{r['source']}]  "
              f"{r['title'] or '(untitled)'}  [{r['unit_id']}]")
    return 0


def cmd_dream_topics(args) -> int:
    """List dream topics: id, status, active insight count, digest presence, last dig."""
    import dream
    ensure_cluster()
    con = connect()
    try:
        if args.seed:
            n = dream.seed_topics(con)
            print(f"seeded {n} new topic(s)")
        st = dream.status(con)
    finally:
        con.close()
    rows = st["topics"] if args.all else [r for r in st["topics"] if r["status"] == "active"]
    if not rows:
        print("(no topics)")
    for r in rows:
        print(f"{r['topic_id']:<28} {r['status']:<10} active={r['active']:<4} "
              f"digest={'yes' if r['digest'] else 'no':<3} last_dig_at={r['last_dig_at']}")
    return 0


def cmd_dream_run(args) -> int:
    """Run the incremental (nightly-shaped) dream pass and print the report."""
    import dream
    ensure_cluster()
    con = connect()
    try:
        report = dream.run_incremental(con, max_calls=args.max_calls)
    finally:
        con.close()
    _print_dream_report(report)
    return 0


def cmd_dream_backfill(args) -> int:
    """Run the explicit bulk backfill pass and print the report."""
    import dream
    ensure_cluster()
    con = connect()
    try:
        report = dream.run_backfill(con, topic_id=args.topic, max_calls=args.max_calls)
    finally:
        con.close()
    _print_dream_report(report)
    return 0


def _print_dream_report(report: dict) -> int:
    print(f"mode        : {report['mode']}")
    if "changed" in report:
        print(f"changed     : {report['changed']} unit(s) since watermark")
    if report.get("skipped_empty"):
        # Skipped-as-empty is a decision, not an error — but it must be VISIBLE,
        # or an abandoned chat is indistinguishable from a triaged one.
        print(f"skipped     : {len(report['skipped_empty'])} empty unit(s) "
              f"(nothing renderable): {', '.join(report['skipped_empty'])}")
    if report.get("topics_touched"):
        print(f"topics      : {', '.join(report['topics_touched'])}")
    if "queued" in report:
        print(f"queued      : {report['queued']} (topic, unit) assignment(s) for digging")
    for tid, digs in report.get("digs", {}).items():
        for d in (digs if isinstance(digs, list) else [digs]):
            written = ", ".join(f"{k}={v}" for k, v in d["written"].items()) or "(none)"
            print(f"  dig[{tid}] units={d['units']} candidates={d['candidates']} "
                  f"rejected(grounding={d['rejected_grounding']}, "
                  f"falsify={d['rejected_falsify']}, substance={d['rejected_substance']}) "
                  f"written={{{written}}}")
            # The REASONS, not just the count: a high reject rate is information about
            # charter and prompt quality, and it is useless as a bare number.
            for r in d.get("rejections", []):
                print(f"      - {r}")
    if report.get("consolidated"):
        print(f"consolidated: {', '.join(report['consolidated'])}")
    print(f"calls       : {report['calls']}")
    if report.get("stopped_early"):
        print(f"stopped_early: {report['stopped_early']}")
    # A night that digs nothing is normal (topics batch up across nights) — but it
    # must SAY what is waiting, so "cheap" is never mistaken for "stuck".
    for d in report.get("deferred", []):
        print(f"  deferred[{d['topic_id']}] pending={d['pending']} "
              f"oldest={d['oldest']} (digs at {dream_thresholds()})")
    if report.get("note"):
        print(f"note        : {report['note']}")
    return 0


def dream_thresholds() -> str:
    import dream
    return (f">={dream.DIG_MIN_UNITS} units or >{dream.DIG_MAX_DEFER_DAYS}d old")


def cmd_dream_recall(args) -> int:
    """Dream-first recall: tiered coverage / digest / insights / raw output."""
    import dream
    ensure_cluster()
    con = connect()
    try:
        result = dream.recall(con, args.query, topic_id=args.topic, limit=args.limit,
                              as_of=args.as_of, include_evidence=args.evidence)
    finally:
        con.close()
    print("\n".join(dream.format_recall(result, requested_topic=args.topic,
                                        include_evidence=args.evidence)))
    return 0


def cmd_dream_status(args) -> int:
    """Watermark, topic table, and usage totals for the dream layer."""
    import dream
    ensure_cluster()
    con = connect()
    try:
        st = dream.status(con)
    finally:
        con.close()
    print(f"watermark : {st['watermark']}")
    print("\ntopics:")
    for r in st["topics"]:
        print(f"  {r['topic_id']:<28} {r['status']:<10} active={r['active']:<4} "
              f"superseded={r['superseded']:<4} contested={r['contested']:<4} "
              f"digest={'yes' if r['digest'] else 'no':<3} last_dig_at={r['last_dig_at']}")
    print("\nusage by kind:")
    for r in st["usage"]["by_kind"]:
        print(f"  {r['kind']:<10} calls={r['calls']:<4} input={r['input_tokens']:<8} "
              f"output={r['output_tokens']:<8} cache_read={r['cache_read']:<8} "
              f"cache_create={r['cache_create']}")
    print("\npending (triaged, awaiting a dig):")
    for r in st["pending"]:
        print(f"  {r['topic_id']:<28} pending={r['n']:<4} oldest={r['oldest']}")
    if not st["pending"]:
        print("  (none)")
    print(f"\nfailed queue items: {st['usage']['failed']}")
    return 0


def cmd_doctor(args) -> int:
    """Health check: store cluster + counts, search index, MCP, launchd."""
    problems: list[str] = []

    import search
    st = search.index_status()          # structured, never raises, no schema leak
    if not st["available"]:
        print("store       : FAIL (PG17 + pgvector toolchain missing)")
        problems.append("PG17/pgvector not found — brew install postgresql@17 "
                        "pgvector (or set $CLYNC_PG_BIN).")
    elif not st["cluster_running"]:
        print(f"store       : FAIL (Postgres cluster not running at {PG_DATA})")
        problems.append("store cluster not running — run `clync sync` or "
                        "`clync index` (provisions + starts it).")
    elif st["error"]:
        print(f"store       : FAIL ({st['error']})")
        problems.append(f"store probe failed: {st['error']}")
    else:
        con = connect()
        try:
            def _n(sql, *a):
                return con.execute(sql, a).fetchone()["n"]
            nchat = _n("SELECT COUNT(*) n FROM units WHERE kind='chat'")
            ndoc = _n("SELECT COUNT(*) n FROM units WHERE kind='project_doc'")
            ncc = _n("SELECT COUNT(*) n FROM units WHERE kind='cc_session'")
            nmsg = _n("SELECT COUNT(*) n FROM messages")
            nfile = _n("SELECT COUNT(*) n FROM files")
            ndl = _n("SELECT COUNT(*) n FROM files WHERE local_path IS NOT NULL")
            last = get_meta(con, "last_success")
            import dream
            dst = dream.status(con)
        finally:
            con.close()
        print(f"store       : {PG_DB}@localhost:{PG_PORT}\n"
              f"              {nchat} chats, {ndoc} project docs, {ncc} cc sessions, "
              f"{nmsg} messages, {nfile} files ({ndl} images), last_success={last}")
        print(f"search index: ok ({st['chunks']} chunks indexed)")
        if last:
            gap = datetime.now(timezone.utc) - datetime.fromisoformat(last)
            if gap > timedelta(hours=LATE_THRESHOLD_H):
                problems.append(f"last claude.ai sync was {gap.total_seconds()/3600:.0f}h "
                                f"ago (> {LATE_THRESHOLD_H}h) — scheduler may not be running.")

        import dream
        con = connect()
        try:
            dst = dream.status(con)
        finally:
            con.close()
        n_active_topics = sum(1 for r in dst["topics"] if r["status"] == "active")
        n_active_insights = sum(r["active"] for r in dst["topics"])
        n_digests = sum(1 for r in dst["topics"] if r["digest"])
        print(f"dream       : {n_active_topics} active topic(s), {n_active_insights} "
              f"active insight(s), {n_digests} digest(s), "
              f"{dst['usage']['failed']} failed queue item(s)")

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


BIN_LINK = Path.home() / ".local/bin/clync"
SKILL_LINK = Path.home() / ".claude/skills/clync"          # history-search (auto-fires)
OPS_SKILL_LINK = Path.home() / ".claude/skills/clync-ops"  # operate/troubleshoot (invoked)
MCP_NAME = "clync"


def _relink(link: Path, target: Path) -> None:
    """Idempotently point a symlink at target; refuse to clobber a real file."""
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"{link} exists and is not a symlink — refusing to overwrite.")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def cmd_setup(args) -> int:
    """One command to wire everything up from the repo: deps, CLI, MCP, scheduler,
    skill. Every artifact is defined in this repo and symlinked/registered out;
    `clync unsetup` reverses it."""
    _require_profile(args)

    subprocess.run(["uv", "sync", "--quiet"], cwd=REPO_DIR, check=True)
    print("✓ deps synced (uv)")

    _relink(BIN_LINK, REPO_DIR / "clync")
    on_path = str(BIN_LINK.parent) in os.environ.get("PATH", "").split(":")
    print(f"✓ CLI: {BIN_LINK} -> clync"
          + ("" if on_path else f"   ⚠ {BIN_LINK.parent} is NOT on your PATH"))

    uv = shutil.which("uv") or "uv"
    subprocess.run(["claude", "mcp", "remove", MCP_NAME], capture_output=True, text=True)
    r = subprocess.run(
        ["claude", "mcp", "add", "--scope", "user", MCP_NAME, "--",
         uv, "run", "--project", str(REPO_DIR), "python", str(REPO_DIR / "mcp_server.py")],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"`claude mcp add` failed: {r.stderr.strip()}")
    print("✓ MCP server registered with Claude Code (user scope)")

    cmd_install(args)  # launchd daily job (prints its own lines)

    _relink(SKILL_LINK, REPO_DIR / "skill")
    print(f"✓ skill: {SKILL_LINK} -> skill/  (resolves to {SKILL_LINK.resolve()})")
    _relink(OPS_SKILL_LINK, REPO_DIR / "skill-ops")
    print(f"✓ ops skill: {OPS_SKILL_LINK} -> skill-ops/  "
          f"(resolves to {OPS_SKILL_LINK.resolve()})")

    # Search is core (ADR 0003): provision the contained cluster + build the initial
    # index unconditionally. Real provisioning/embedding failures fail loud — a
    # broken install must NOT be reported as a successful setup.
    import search
    ensure_cluster()
    stats = search.build_index(full=True)
    print(f"✓ store + search: contained PG17+pgvector cluster provisioned, "
          f"indexed reindexed_units={stats['reindexed_units']} "
          f"chunks={stats['chunks']}")

    # The Dream layer distils only what `dream_topics` says to distil, so an unseeded
    # install has a layer that silently does nothing. Seeding is free (no model calls)
    # and idempotent; mining history is the explicit, costly step the user opts into.
    import dream
    con = connect()
    try:
        seeded = dream.seed_topics(con)
    finally:
        con.close()
    print(f"✓ dream layer: schema ready, {seeded} topic(s) seeded "
          f"(nightly pass runs inside the daily job; run `clync dream backfill` "
          f"to mine existing history — that one spends real quota)")

    print("\nSetup complete. Open a NEW Claude Code session to load the MCP tools "
          "+ skill. Reverse anytime with `clync unsetup`.")
    return 0


def cmd_unsetup(args) -> int:
    """Remove every external artifact setup created (repo stays intact)."""
    cmd_uninstall(args)  # launchd
    subprocess.run(["claude", "mcp", "remove", MCP_NAME], capture_output=True, text=True)
    print(f"✓ MCP server '{MCP_NAME}' unregistered")
    for link in (BIN_LINK, SKILL_LINK, OPS_SKILL_LINK):
        if link.is_symlink():
            link.unlink()
            print(f"✓ removed {link}")
    # stop_cluster is defensive (no-op if the cluster/binaries are absent) and
    # only shells out to pg_ctl, so call it unconditionally.
    if stop_cluster():
        print("✓ store cluster stopped (its data was left intact)")
    print("Unset complete. The store in ~/.local/share/clync was left intact.")
    return 0


def cmd_status(args) -> int:
    """Scheduled-sync status: last success, launchd state, recent run log."""
    import search
    if search.index_status()["cluster_running"]:
        con = connect()
        try:
            last = get_meta(con, "last_success")
        finally:
            con.close()
    else:
        last = None
    print(f"last successful sync : {last or '(never / store not running)'}")
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
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared account flags, attached to the subcommands that authenticate — so
    # `clync sync --profile X` works (a top-level flag would have to precede the
    # subcommand, which is a footgun).
    cred = argparse.ArgumentParser(add_help=False)
    cred.add_argument("--profile", default=os.environ.get(PROFILE_ENV),
                      help=f"Chrome profile display name (default: ${PROFILE_ENV})")
    cred.add_argument("--org", help="restrict to one org name or uuid "
                                     "(default: all chat-capable orgs)")

    sub.add_parser("whoami", parents=[cred], help="print the resolved account + org"
                   ).set_defaults(func=cmd_whoami)

    sp = sub.add_parser("sync", parents=[cred],
                        help="sync BOTH sources (claude.ai + local Claude Code) + index")
    sp.add_argument("--full", action="store_true", help="re-fetch/re-parse everything")
    sp.add_argument("--no-files", action="store_true", help="skip image downloads")
    sp.add_argument("--no-index", action="store_true",
                     help="skip hybrid-search indexing after sync")
    sp.set_defaults(func=cmd_sync)

    sp = sub.add_parser("sync-app", parents=[cred],
                        help="sync claude.ai chats ONLY (network/cookies) + index")
    sp.add_argument("--full", action="store_true", help="re-fetch every conversation")
    sp.add_argument("--no-files", action="store_true", help="skip image downloads")
    sp.add_argument("--no-index", action="store_true", help="skip indexing after sync")
    sp.set_defaults(func=cmd_sync_app)

    sp = sub.add_parser("sync-cc", help="ingest local Claude Code sessions ONLY "
                                        "(no network / cookies) + index")
    sp.add_argument("--full", action="store_true", help="re-parse every transcript")
    sp.add_argument("--no-index", action="store_true", help="skip indexing after ingest")
    sp.set_defaults(func=cmd_sync_cc)

    sp = sub.add_parser("index", help="(re)build the hybrid-search index")
    sp.add_argument("--full", action="store_true", help="reindex every unit")
    sp.set_defaults(func=cmd_index)

    sub.add_parser("scheduled", parents=[cred],
                   help="launchd entry point (sync + loud fail/late notify)"
                   ).set_defaults(func=cmd_scheduled)

    sp = sub.add_parser("search", help="faceted hybrid (dense + sparse) semantic "
                                       "search across claude.ai + Claude Code")
    sp.add_argument("query", nargs="?", default="",
                    help="natural-language query (empty => recency browse)")
    sp.add_argument("--limit", type=int, default=DEFAULT_TOPK)
    sp.add_argument("--source", choices=VALID_SOURCES,
                    default="all", help="restrict to one source (default: all)")
    sp.add_argument("--project", help="claude.ai project name/uuid; "
                    "'any' = in some project, 'none' = not in a project")
    sp.add_argument("--model", help="claude.ai model facet")
    sp.add_argument("--repo", help="Claude Code repo name")
    sp.add_argument("--worktree", help="Claude Code worktree name")
    sp.add_argument("--branch", help="Claude Code git branch")
    sp.add_argument("--session", help="Claude Code session name or id")
    sp.add_argument("--since", help="only units updated on/after this ISO date")
    sp.add_argument("--until", help="only units updated on/before this ISO date")
    sp.add_argument("--sort", choices=["relevance", "recency"], default="relevance")
    sp.add_argument("--lang", choices=["en", "ja", "zh"], default=None,
                     help="restrict to one language (default: all)")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("list", help="most recently updated units")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--source", choices=VALID_SOURCES, default="all")
    sp.set_defaults(func=cmd_list)

    sub.add_parser("doctor", help="health check store / index / launchd / MCP"
                   ).set_defaults(func=cmd_doctor)

    dp = sub.add_parser("dream", help="knowledge-distillation layer (ADR 0004)")
    dsub = dp.add_subparsers(dest="dream_action", required=True)

    sp = dsub.add_parser("topics", help="list dream topics")
    sp.add_argument("--seed", action="store_true", help="insert the seed topics first")
    sp.add_argument("--all", action="store_true", help="include non-active topics")
    sp.set_defaults(func=cmd_dream_topics)

    sp = dsub.add_parser("run", help="incremental (nightly-shaped) dream pass")
    sp.add_argument("--max-calls", type=int, default=None,
                    dest="max_calls", help="cap model calls this run (default: no cap)")
    sp.set_defaults(func=cmd_dream_run)

    sp = dsub.add_parser("backfill", help="explicit bulk backfill pass (spends real quota)")
    sp.add_argument("--topic", help="restrict to one topic id (default: all topics)")
    sp.add_argument("--max-calls", type=int, default=30, dest="max_calls",
                    help="cap model calls this run (default: 30)")
    sp.set_defaults(func=cmd_dream_backfill)

    sp = dsub.add_parser("recall", help="dream-first recall: digest -> insights -> raw")
    sp.add_argument("query", nargs="?", default="", help="natural-language query")
    sp.add_argument("--topic", help="restrict to one topic id")
    sp.add_argument("--limit", type=int, default=DEFAULT_TOPK)
    sp.add_argument("--as-of", dest="as_of", help="ISO date — positions held at that date")
    sp.add_argument("--evidence", action="store_true", help="also print cited quotes")
    sp.set_defaults(func=cmd_dream_recall)

    dsub.add_parser("status", help="watermark, topic table, usage totals"
                    ).set_defaults(func=cmd_dream_status)

    sp = sub.add_parser("setup", parents=[cred],
                        help="one-shot install: deps, CLI, MCP, scheduler, skill")
    sp.add_argument("--at", default="09:00", help="daily sync time HH:MM (default 09:00)")
    sp.set_defaults(func=cmd_setup)

    sub.add_parser("unsetup", help="remove everything setup installed (keeps the DB)"
                   ).set_defaults(func=cmd_unsetup)

    sp = sub.add_parser("install", parents=[cred], help="install just the daily launchd sync")
    sp.add_argument("--at", default="09:00", help="daily time HH:MM (default 09:00)")
    sp.set_defaults(func=cmd_install)

    sub.add_parser("uninstall", help="remove just the launchd sync").set_defaults(func=cmd_uninstall)

    sp = sub.add_parser("status", help="scheduled-sync status + recent run log")
    sp.add_argument("--tail", type=int, default=15, help="log lines to show")
    sp.set_defaults(func=cmd_status)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
