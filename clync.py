"""clync — history sync, semantic archive, and live lexical session finding.

Raw sources, one contained store (ADR 0003):
  * claude.ai chats — reads auth cookies live from the selected Chrome profile,
    clears Cloudflare via curl_cffi TLS impersonation, and upserts conversations
    + project docs (incremental by `updated_at`).
  * ChatGPT web history — reads auth cookies live from the selected Chrome
    profile and upserts conversations (see chatgpt.py).
  * local Claude Code sessions — cleaned + metadata-extracted from
    `~/.claude/projects/**/*.jsonl` (see cc.py); no network / cookies.
  * local Codex CLI sessions — parsed from `~/.codex/**/*.jsonl` (see codex.py);
    no network / cookies.
  * local Pi sessions — cleaned by pi_source.py; no network / cookies.
  * configured OpenCode v2 root transcripts (opencode_source.py) and Artiport
    artifacts (artiport_source.py); see README for setup and scope.

Archive sources land in a contained Postgres 17 + pgvector cluster (single source of truth
AND search index) as unified `units` + `messages`, embedded by search.py.
The independent `sessions` commands read native sources without that archive.

Fail-loud by contract: any auth/HTTP/schema problem raises and the process exits
non-zero. The scheduled path additionally surfaces failures as a macOS
modal+sound notification. Never degrades silently to stale data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import shlex
import shutil
import sqlite3          # Chrome's cookie store is SQLite; clync's own store is Postgres
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

# Network sources use separate Chrome identities: Claude.ai is the work account,
# while ChatGPT is the personal account. Keeping these as distinct settings makes
# an accidental cross-account sync impossible at the scheduler boundary.
CLAUDE_PROFILE_ENV = "CLYNC_CLAUDE_PROFILE"
CHATGPT_PROFILE_ENV = "CLYNC_CHATGPT_PROFILE"
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
PG17_BIN_CANDIDATES = (
    "/opt/homebrew/opt/postgresql@17/bin",
    "/usr/local/opt/postgresql@17/bin",
)


def _default_pg_bin() -> str:
    for candidate in PG17_BIN_CANDIDATES:
        if (Path(candidate) / "initdb").exists():
            return candidate
    return PG17_BIN_CANDIDATES[0]


PG_BIN = Path(os.environ.get("CLYNC_PG_BIN") or _default_pg_bin())
PG_PORT = int(os.environ.get("CLYNC_PG_PORT", "54329"))
PG_DB = os.environ.get("CLYNC_PG_DB", "clync")
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", PG_DB):
    raise SystemExit(f"invalid $CLYNC_PG_DB {PG_DB!r} — must match [A-Za-z_][A-Za-z0-9_]*")
PG_USER = os.environ.get("USER", "postgres")
PG_LOCALE = "en_US.UTF-8"

# Share of the machine's CPUs a sync may take. The embedder is the heavy consumer
# and sizes itself to the whole machine unless told otherwise.
RESOURCE_FRACTION = float(os.environ.get("CLYNC_RESOURCE_FRACTION", "0.5"))
# Every thread-pool knob the embedder's stack reads at load time: torch/OpenMP,
# Apple's vecLib, numexpr, and the Rust tokenizer's rayon pool.
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")


def limit_resources() -> dict[str, str]:
    """Cap this process's CPU share to `RESOURCE_FRACTION`, and return the counts in
    force. Call it BEFORE importing torch: the thread pools are sized at load time
    from these variables and ignore a later change. An explicitly exported value
    always wins — this sets a default, it does not override config.

    CPU only. Metal exposes no utilisation fraction, so the GPU is bounded by
    running the embedder on the CPU instead (`$CLYNC_EMBED_DEVICE`); its watermark
    ratios bound memory, not utilisation."""
    threads = max(1, int((os.cpu_count() or 1) * RESOURCE_FRACTION))
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(threads))
    return {v: os.environ[v] for v in _THREAD_VARS}

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
# The Dream layer (ADR 0004) is reachable via source='dream' or the combined
# `search_history` surface, never by accident: that keeps every pre-existing query's
# meaning intact AND makes non-circularity the default — the dig cannot retrieve its
# own output.
RAW_SOURCES = ("claude_ai", "claude_code", "codex_cli", "chatgpt", "artiport", "opencode", "pi")
VALID_SOURCES = ("all", *RAW_SOURCES, "dream")


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
NO_DREAM_FLAG = "--no-dream"
PLIST_PATH = Path.home() / "Library/LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
PG_LAUNCHD_LABEL = "io.clync.postgres"
PG_PLIST_PATH = PLIST_PATH.with_name(f"{PG_LAUNCHD_LABEL}.plist")
SCHED_LOG = DATA_HOME / "scheduled.log"
LATE_THRESHOLD_H = 25  # a >25h gap since last success means a daily run was missed


# --------------------------------------------------------------------------- #
# Chrome cookie extraction (auth cookies are per-Chrome-profile; each network
# source has its own explicit profile flag)
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
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
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
    """Map a Chrome profile *display name* (e.g. 'Research') to its directory."""
    local_state = json.loads((CHROME_DIR / "Local State").read_text())
    cache = local_state.get("profile", {}).get("info_cache", {})
    matches = [d for d, info in cache.items() if info.get("name") == display_name]
    if not matches:
        names = sorted(info.get("name") for info in cache.values())
        raise RuntimeError(f"No Chrome profile named {display_name!r}. Available: {names}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple profiles named {display_name!r}: {matches}")
    return CHROME_DIR / matches[0]


def _read_profile_cookies(profile_display_name: str, host: str,
                          names: tuple[str, ...] | None = None,
                          tolerant: bool = False) -> dict[str, str]:
    """Live-read + decrypt a Chrome profile's cookies for a host. `names` limits
    to specific cookie names; None reads every cookie for the host. The DB is
    copied first to dodge Chrome's live lock; decryption uses the login-keychain
    Safe Storage key. `tolerant` skips a cookie that will not decrypt (e.g. an
    app-bound v20 cookie irrelevant to us) instead of raising — each caller then
    validates that the cookie IT needs is present and well-formed (fail-loud on
    the load-bearing one, not on incidental ones)."""
    profile_dir = resolve_profile_dir(profile_display_name)
    cookies_db = profile_dir / "Cookies"
    if not cookies_db.exists():
        raise RuntimeError(f"No Cookies DB at {cookies_db}")
    key = _safe_storage_key()
    # CAST(... AS BLOB) is load-bearing: some profiles store encrypted_value with
    # TEXT storage class, and sqlite3's default text_factory then tries to UTF-8
    # decode the ciphertext and raises OperationalError before we ever decrypt it
    # The CAST keeps `name` a str while forcing bytes here.
    if names:
        placeholders = ",".join("?" * len(names))
        sql = ("SELECT name, CAST(encrypted_value AS BLOB) FROM cookies "
               f"WHERE host_key IN (?, ?) AND name IN ({placeholders})")
        params: tuple = (host, "." + host, *names)
    else:
        sql = ("SELECT name, CAST(encrypted_value AS BLOB) FROM cookies "
               "WHERE host_key IN (?, ?)")
        params = (host, "." + host)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / "Cookies"  # copy to dodge Chrome's live lock
        shutil.copy2(cookies_db, snapshot)
        con = sqlite3.connect(snapshot)
        try:
            rows = con.execute(sql, params).fetchall()
        finally:
            con.close()
    cookies: dict[str, str] = {}
    for name, val in rows:
        try:
            cookies[name] = _decrypt_cookie(val, key)
        except RuntimeError:
            if not tolerant:
                raise
    return cookies


def read_auth_cookies(profile_display_name: str) -> dict[str, str]:
    """Live-read + decrypt claude.ai auth cookies from a Chrome profile.

    Read fresh every run, so sessionKey + cf_clearance are always as current as
    the browser's last claude.ai activity — that is the freshness mechanism.
    """
    cookies = _read_profile_cookies(profile_display_name, "claude.ai",
                                    AUTH_COOKIE_NAMES)
    if "sessionKey" not in cookies:
        raise RuntimeError(
            f"No claude.ai login found in the {profile_display_name!r} Chrome "
            f"profile. Open {HOST} in that profile and LOG IN, then re-run."
        )
    if not cookies["sessionKey"].startswith("sk-ant-sid"):
        raise RuntimeError("Decrypted sessionKey has an unexpected format.")
    return cookies


def read_chatgpt_cookies(profile_display_name: str) -> dict[str, str]:
    """Live-read + decrypt every chatgpt.com cookie from a Chrome profile. All of
    them are set on the request (as the browser does) so Cloudflare's clearance
    cookies ride along; the session-token cookie is the load-bearing credential
    and its absence fails loud. Read fresh every run for freshness (mirror of
    read_auth_cookies)."""
    cookies = _read_profile_cookies(profile_display_name, "chatgpt.com",
                                    names=None, tolerant=True)
    base = chatgpt_mod_session_cookie()
    # next-auth splits an oversized session token into <name>.0, <name>.1, …
    # (each Chrome cookie capped at ~4 KB) and the server reassembles the
    # chunks; the browser sends every chunk, so all of them ride along here.
    # Either the whole cookie or its first chunk is proof of login.
    if base not in cookies and f"{base}.0" not in cookies:
        raise RuntimeError(
            f"No ChatGPT login found in the {profile_display_name!r} Chrome "
            f"profile (no {base} cookie). Open "
            f"https://chatgpt.com in that profile and LOG IN, then re-run."
        )
    return cookies


def chatgpt_mod_session_cookie() -> str:
    """The ChatGPT session-token cookie name (SSOT lives in chatgpt.py)."""
    import chatgpt
    return chatgpt.SESSION_COOKIE


# --------------------------------------------------------------------------- #
# claude.ai private API client (Cloudflare-passing via TLS impersonation)
# --------------------------------------------------------------------------- #
class ClaudeClient:
    def __init__(self, cookies: dict[str, str], profile: str | None = None):
        from curl_cffi import requests as creq
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
# Shared predicate for messages that can be rendered/embedded (SSOT for partial
# index below and Dream's renderability checks).
NONEMPTY_MSG_SQL = "btrim(coalesce(embed_text, text, '')) <> ''"

RAW_SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS units (
    unit_id      text PRIMARY KEY,
    kind         text NOT NULL,
    source       text NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_messages_nonempty
    ON messages(unit_id, sender) WHERE {NONEMPTY_MSG_SQL};

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

-- Per-file incremental watermarks are partitioned by `source`; deletion
-- reconciliation must never cross sources.
CREATE TABLE IF NOT EXISTS cc_sync_state (
    file_path  text PRIMARY KEY,
    mtime      double precision,
    size       bigint,
    session_id text,
    source     text NOT NULL,
    synced_at  timestamptz NOT NULL
);
-- Backfill the source column onto a pre-existing table (rows written before the
-- Codex ingest existed are all Claude Code). The DEFAULT exists ONLY to stamp
-- those legacy rows during this ADD COLUMN.
ALTER TABLE IF EXISTS cc_sync_state
    ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'claude_code';
-- Drop the DEFAULT once the backfill has run: its sole purpose was the migration
-- above. Keeping it live would let a future watermark INSERT that omits `source`
-- silently label itself 'claude_code' (NOT NULL would still pass) and slip the
-- source-partitioned deletion reconciliation. Every live INSERT binds source
-- explicitly, so with no default an omitted bind now fails loud (NOT NULL).
ALTER TABLE IF EXISTS cc_sync_state ALTER COLUMN source DROP DEFAULT;

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
    roots = [PG_BIN.parent]
    if len(PG_BIN.parents) > 2:
        roots.append(PG_BIN.parents[2])
    return any((r / "share" / "postgresql@17" / "extension" / "vector.control").exists()
               for r in roots)


def cluster_running() -> bool:
    r = subprocess.run([_pg("pg_ctl"), "-D", str(PG_DATA), "status"],
                       capture_output=True, text=True)
    return r.returncode == 0


def postgres_command() -> list[str]:
    return [_pg("postgres"), "-D", str(PG_DATA), "-p", str(PG_PORT),
            "-c", "listen_addresses=localhost", "-c", f"unix_socket_directories={PG_DIR}"]


def start_cluster() -> None:
    if (PG_PLIST_PATH.exists()
            and plistlib.loads(PG_PLIST_PATH.read_bytes())["ProgramArguments"] == postgres_command()):
        subprocess.run(["launchctl", "kickstart", f"gui/{os.getuid()}/{PG_LAUNCHD_LABEL}"],
                       check=True, capture_output=True, text=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            ready = subprocess.run(
                [_pg("pg_isready"), "-h", "localhost", "-p", str(PG_PORT), "-d", "postgres"],
                capture_output=True, text=True)
            if ready.returncode == 0 and cluster_running():
                return
            time.sleep(0.1)
        raise RuntimeError(f"{PG_LAUNCHD_LABEL} did not become ready; inspect {PG_LOG}")
    if cluster_running():
        return
    PG_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [_pg("pg_ctl"), "-D", str(PG_DATA), "-l", str(PG_LOG),
         "-o", shlex.join(postgres_command()[3:]),
         "-w", "start"],
        check=True, stdout=sys.stderr)


def install_cluster_service() -> None:
    init_cluster()
    service = f"gui/{os.getuid()}/{PG_LAUNCHD_LABEL}"
    document = {
        "Label": PG_LAUNCHD_LABEL,
        "ProgramArguments": postgres_command(),
        "EnvironmentVariables": {"LC_ALL": PG_LOCALE},
        "KeepAlive": True,
        "StandardOutPath": str(PG_LOG),
        "StandardErrorPath": str(PG_LOG),
    }
    content = plistlib.dumps(document, fmt=plistlib.FMT_XML)
    loaded = subprocess.run(["launchctl", "print", service], capture_output=True).returncode == 0
    if loaded and PG_PLIST_PATH.exists() and PG_PLIST_PATH.read_bytes() == content:
        ensure_cluster()
        return
    if loaded:
        unload_cluster_service()
    else:
        stop_cluster()
    PG_PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PG_PLIST_PATH.write_bytes(content)
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(PG_PLIST_PATH)],
                   check=True, capture_output=True, text=True)
    ensure_cluster()


def unload_cluster_service() -> None:
    service = f"gui/{os.getuid()}/{PG_LAUNCHD_LABEL}"
    subprocess.run(["launchctl", "bootout", service], check=True, capture_output=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        loaded = subprocess.run(["launchctl", "print", service], capture_output=True).returncode == 0
        if not loaded and not cluster_running():
            return
        time.sleep(0.1)
    raise RuntimeError(f"{PG_LAUNCHD_LABEL} did not stop; inspect {PG_LOG}")


def stop_cluster() -> bool:
    """Stop the contained cluster if running. True iff a running cluster was
    stopped; False (no-op) if there's no cluster data or the PG17 binary is
    absent — not an error."""
    if not (PG_DATA / "PG_VERSION").exists() or not (PG_BIN / "pg_ctl").exists():
        return False
    if not cluster_running():
        return False
    subprocess.run([_pg("pg_ctl"), "-D", str(PG_DATA), "-m", "fast", "stop"],
                   check=True, stdout=sys.stderr)
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


class StaleDerivedSchema(RuntimeError):
    """A derived layer's tables no longer match the shipped schema, and rebuilding
    them would destroy derived data. Raised instead of rebuilding."""


def require_rebuild(rebuild: bool, *, layer: str, cost: str) -> None:
    """Gate every destructive rebuild of a DERIVED layer behind an explicit opt-in.

    A stale shape used to be dropped and recreated inline by whichever call
    happened to provision the store first. Two ways that bites, both observed:
    an ordinary READ then takes DDL locks (and deadlocks against its own caller's
    still-open read transaction — no timeout, no error, forever), and when it does
    NOT deadlock it destroys the layer silently, mid-run, with nobody asking for it.
    Rebuilding is derived-data loss, so it is an operator decision, not a side
    effect of an arbitrary call path. SSOT for that refusal message."""
    if not rebuild:
        raise StaleDerivedSchema(
            f"{layer} schema is STALE: the shipped layout has moved on. Rebuilding "
            f"it drops and re-derives {cost}. Nothing has been changed. To rebuild, "
            f"run:  clync migrate")


def init_cluster() -> None:
    if not _vector_control_present():
        raise SystemExit(
            "pgvector not found for PG17. Install with:  brew install pgvector")
    if not (PG_DATA / "PG_VERSION").exists():
        PG_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [_pg("initdb"), "-D", str(PG_DATA), "-U", PG_USER,
             "--encoding=UTF8", f"--locale={PG_LOCALE}", "-A", "trust"],
            check=True, capture_output=True)


def ensure_cluster(*, rebuild_derived: bool = False) -> None:
    """Provision the store; rebuilding stale derived schemas requires explicit opt-in."""
    init_cluster()
    start_cluster()
    exists = subprocess.run(
        [_pg("psql"), "-h", "localhost", "-p", str(PG_PORT), "-U", PG_USER, "-d", "postgres",
         "-tAc", f"SELECT 1 FROM pg_database WHERE datname='{PG_DB}'"],
        capture_output=True, text=True, check=True).stdout.strip()
    if exists != "1":
        subprocess.run([_pg("psql"), "-h", "localhost", "-p", str(PG_PORT), "-U", PG_USER,
                        "-d", "postgres", "-c", f'CREATE DATABASE "{PG_DB}"'],
                       check=True, capture_output=True)
    with connect_pg() as con:
        for stmt in sql_statements(RAW_SCHEMA):
            con.execute(stmt)
        con.commit()
    # Each derived layer owns its own tables + their stale-shape guard.
    import search
    search.ensure_index_schema(rebuild=rebuild_derived)   # chunks/indexed_units
    import dream
    dream.ensure_dream_schema(rebuild=rebuild_derived)    # dream_* (stance vocab)


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
            if Path(fid).name != fid or fid in {".", ".."}:
                raise ValueError("Image/file identifier must be a filename component")
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
# Local file-based session ingest (cookie-independent), partitioned by the
# `source` column of cc_sync_state. Parsers own discovery and cleaning.
# --------------------------------------------------------------------------- #
def _upsert_local_session(con, s, source: str, kind: str, *, raw: dict | None = None) -> None:
    """Write one parsed session (a CCSession / CodexSession / ChatGPTConversation
    — duck-typed on identical field names; the network-sourced ChatGPT object
    leaves the local-workspace fields None) -> a `units` row + its cleaned
    `messages` (wholesale-replaced; per-unit identity)."""
    con.execute(
        """INSERT INTO units
             (unit_id, kind, source, title, summary, model, created_at, updated_at,
              repo, cwd, worktree, git_branch, cc_version, entrypoint,
              raw, synced_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT(unit_id) DO UPDATE SET
             kind=EXCLUDED.kind, source=EXCLUDED.source,
             title=EXCLUDED.title, summary=EXCLUDED.summary, model=EXCLUDED.model,
             created_at=EXCLUDED.created_at, updated_at=EXCLUDED.updated_at,
             repo=EXCLUDED.repo, cwd=EXCLUDED.cwd, worktree=EXCLUDED.worktree,
             git_branch=EXCLUDED.git_branch, cc_version=EXCLUDED.cc_version,
             entrypoint=EXCLUDED.entrypoint,
             raw=EXCLUDED.raw, synced_at=EXCLUDED.synced_at""",
        (s.session_id, kind, source, s.title, s.summary, s.model,
         s.created_at, s.updated_at, s.repo, s.cwd, s.worktree, s.git_branch,
         s.cc_version, s.entrypoint,
         _json(raw if raw is not None else {"session_id": s.session_id, "cwd": s.cwd, "repo": s.repo,
                "worktree": s.worktree, "git_branch": s.git_branch,
                "cc_version": s.cc_version, "entrypoint": s.entrypoint}), _now()),
    )
    rows = [(m.msg_id, m.idx, m.sender, m.role_detail, m.text, m.embed_text,
             m.created_at, None) for m in s.messages]
    _replace_unit_messages(con, s.session_id, rows)


_UNSET = object()   # default out_of_scope_marker: a value no parser ever returns


def _ingest_local(source: str, kind: str, iter_files, parse_session, full: bool,
                  out_of_scope_marker: object = _UNSET) -> dict:
    """Shared incremental ingest for a local file-based source. Incremental by
    file mtime/size (the `cc_sync_state` watermark, partitioned by `source`);
    `full` re-parses every file. Also reconciles deletions WITHIN this source: a
    file removed from disk loses its unit/messages and its watermark row (mirror
    of the claude.ai upstream-deletion purge — never degrade silently to stale
    data). An unavailable root raises (in `iter_files`, before any store
    mutation) — a missing source is an error, never evidence of deletion. No
    network / cookies. Returns stats. Fails loud on a store error.

    `out_of_scope_marker` is the sentinel a parser returns for a file it examined
    but deliberately skipped by scope (e.g. codex.OUT_OF_SCOPE for a non-terminal
    originator). Those are watermarked (so they are not re-parsed every sync) but
    counted on their OWN `out_of_scope` axis, NOT as ingested sessions — so a
    producer rename that pushes the whole corpus out of scope shows up as
    out_of_scope==files_parsed with sessions_ingested==0, never a silent healthy
    run. A source with no scope sentinel leaves the default (matches nothing)."""
    files = list(iter_files())  # raises on a missing root, before the store is touched
    con = connect()
    # Only THIS source's watermarks — so a cross-source path never reads as
    # "deleted" and gets its unit purged during reconciliation.
    state = {r["file_path"]: (r["mtime"], r["size"]) for r in con.execute(
        "SELECT file_path, mtime, size FROM cc_sync_state WHERE source=%s",
        (source,)).fetchall()}
    ingested = skipped = sessions = out_of_scope = 0
    try:
        for path in files:
            st = path.stat()
            key = str(path)
            if not full and state.get(key) == (st.st_mtime, st.st_size):
                skipped += 1
                continue
            parsed = parse_session(path)
            if parsed is out_of_scope_marker:      # examined, deliberately skipped by scope
                out_of_scope += 1
                session = None
            else:
                session = parsed
            if session is not None:
                _upsert_local_session(con, session, source, kind)
                sessions += 1
            con.execute(
                "INSERT INTO cc_sync_state (file_path, mtime, size, session_id, source, "
                "synced_at) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(file_path) DO UPDATE SET "
                "mtime=EXCLUDED.mtime, size=EXCLUDED.size, "
                "session_id=EXCLUDED.session_id, source=EXCLUDED.source, "
                "synced_at=EXCLUDED.synced_at",
                (key, st.st_mtime, st.st_size,
                 session.session_id if session else None, source, _now()))
            con.commit()
            ingested += 1
        # Reconcile deletions WITHIN this source: drop watermark rows for files no
        # longer on disk, then every unit of this kind left without ANY backing
        # file (FK cascades its messages; build_index drops its chunks via `gone`).
        on_disk = {str(p) for p in files}
        dead_paths = [fp for fp in state if fp not in on_disk]
        if dead_paths:
            con.execute("DELETE FROM cc_sync_state WHERE file_path = ANY(%s)",
                        (dead_paths,))
        removed = len(con.execute(
            "DELETE FROM units u WHERE u.kind=%s AND NOT EXISTS "
            "(SELECT 1 FROM cc_sync_state s WHERE s.session_id = u.unit_id "
            " AND s.source=%s) RETURNING u.unit_id", (kind, source)).fetchall())
        con.commit()
        total = con.execute(
            "SELECT COUNT(*) FROM units WHERE source=%s", (source,)).fetchone()["count"]
    finally:
        con.close()
    print(f"[{source}] files_parsed={ingested} unchanged={skipped} "
          f"sessions_ingested={sessions} out_of_scope={out_of_scope} "
          f"sessions_removed={removed} sessions_in_db={total}")
    return {"files_parsed": ingested, "unchanged": skipped,
            "sessions_ingested": sessions, "out_of_scope": out_of_scope,
            "sessions_removed": removed, "sessions_in_db": total}


def ingest_cc(full: bool = False) -> dict:
    """Ingest local Claude Code transcripts (`~/.claude/projects`). See _ingest_local."""
    import cc
    return _ingest_local("claude_code", "cc_session",
                         cc.iter_session_files, cc.parse_session, full)


def ingest_codex(full: bool = False) -> dict:
    """Ingest local Codex CLI rollouts (`~/.codex/sessions` + archived). See _ingest_local."""
    import codex
    return _ingest_local("codex_cli", "codex_session",
                         codex.iter_session_files, codex.parse_session, full,
                         out_of_scope_marker=codex.OUT_OF_SCOPE)


def ingest_pi(full: bool = False) -> dict:
    import pi_source
    if "CLYNC_PI_ROOT" in os.environ and not os.environ["CLYNC_PI_ROOT"].strip():
        raise ValueError("$CLYNC_PI_ROOT must name a Pi sessions directory, not be empty")
    try:
        pi_source.PI_ROOT.stat()
    except FileNotFoundError as e:
        if "CLYNC_PI_ROOT" in os.environ:
            raise RuntimeError(f"Configured Pi source unavailable: {pi_source.PI_ROOT}") from e
        with connect() as con:
            known = con.execute(
                "SELECT EXISTS (SELECT 1 FROM units WHERE source=%s) OR "
                "EXISTS (SELECT 1 FROM cc_sync_state WHERE source=%s) AS known",
                ("pi", "pi")).fetchone()["known"]
        if known:
            raise RuntimeError(
                f"Previously archived Pi source unavailable: {pi_source.PI_ROOT}") from e
        reason = f"unconfigured default root missing: {pi_source.PI_ROOT}"
        print(f"[pi] skipped ({reason}); archive unchanged")
        return {"skipped": True, "reason": reason}
    return _ingest_local("pi", "pi_session", pi_source.iter_session_files,
                         pi_source.parse_session, full)


def ingest_chatgpt(profile: str, full: bool = False, projects_only: bool = False) -> dict:
    """Ingest ChatGPT web history (source=chatgpt, kind=chatgpt) via the private
    web API (cookies -> accessToken -> /backend-api). NETWORK/cookies — fails loud
    on an expired login (mirror of the claude.ai leg, NOT the cookie-independent
    local ingests). Incremental by each conversation's `update_time` (skip when the
    stored instant matches); `full` re-fetches every conversation. Deletion is
    reconciled against the live listing — a conversation gone upstream loses its
    unit (FK-cascades its messages), scoped to kind='chatgpt' so no other source is
    ever touched. Returns stats."""
    import chatgpt
    client = chatgpt.ChatGPTClient(read_chatgpt_cookies(profile), profile)
    con = connect()
    fetched = skipped = removed = 0
    try:
        stored = {r["unit_id"]: r["updated_at"] for r in con.execute(
            "SELECT unit_id, updated_at FROM units WHERE kind='chatgpt' "
            "AND (NOT %s OR project_uuid IS NOT NULL)", (projects_only,)).fetchall()}
        remote = list(client.iter_conversations())
        remote_ids = {c["id"] for c in remote}
        if projects_only:
            remote = [c for c in remote if c.get("project_uuid") or c["id"] in stored]
        print(f"[chatgpt] conversations: {len(remote)} remote | "
              f"{len(stored)} already stored")
        for c in remote:
            cid = c["id"]
            if (not full and cid in stored
                    and _same_instant(stored[cid], chatgpt._iso(c.get("update_time")))):
                skipped += 1
            else:
                conv = chatgpt.linearize(client.get_conversation(cid), c)
                _upsert_local_session(con, conv, "chatgpt", "chatgpt")
                fetched += 1
            project = (c.get("project_uuid"), c.get("project_name"))
            con.execute(
                "UPDATE units SET project_uuid=%s, project_name=%s, synced_at=%s "
                "WHERE unit_id=%s AND (project_uuid, project_name) IS DISTINCT FROM (%s,%s)",
                (*project, _now(), cid, *project))
            con.commit()
        # Reconcile deletions WITHIN source=chatgpt only (kind guards the scope).
        dead = [u for u in stored if u not in remote_ids]
        if dead:
            removed = len(con.execute(
                "DELETE FROM units WHERE kind='chatgpt' AND unit_id = ANY(%s) "
                "RETURNING unit_id", (dead,)).fetchall())
        con.commit()
        total = con.execute(
            "SELECT COUNT(*) AS n FROM units WHERE source='chatgpt'").fetchone()["n"]
    finally:
        con.close()
    print(f"[chatgpt] fetched/updated={fetched} unchanged={skipped} "
          f"conversations_removed={removed} conversations_in_db={total}")
    return {"fetched": fetched, "unchanged": skipped,
            "conversations_removed": removed, "conversations_in_db": total}


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
def _require_profile(args, *, attr: str, env: str) -> str:
    profile = getattr(args, attr, None)
    if not profile:
        raise RuntimeError(
            f"No Chrome profile given. Pass --{attr.replace('_', '-')} NAME "
            f"or set ${env}."
        )
    return profile


def _require_claude_profile(args) -> str:
    return _require_profile(args, attr="claude_profile", env=CLAUDE_PROFILE_ENV)


def _require_chatgpt_profile(args) -> str:
    return _require_profile(args, attr="chatgpt_profile", env=CHATGPT_PROFILE_ENV)


def cmd_whoami(args) -> int:
    profile = _require_claude_profile(args)
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
          f"chunks={stats['chunks']} removed_units={stats['removed_units']} "
          f"dropped_chunks={stats['dropped_chunks']} "
          f"repositioned_chunks={stats['repositioned_chunks']}")


def _sync_all(claude_profile: str, org_ref: str | None, full: bool,
              download_files: bool, do_index: bool, notify_fail: bool,
              *, chatgpt_profile: str | None = None) -> Exception | None:
    """Run both network legs with their source-specific profiles, then ALWAYS the
    local ingests (Claude Code + Codex CLI + Pi, no network), then index. Each network
    leg's failure is captured INDEPENDENTLY and RETURNED for the caller to re-raise
    (fail-loud) — but only after the local ingests + index run, so one stale/expired
    cookie never blocks the other network leg OR local-session capture. A
    local-ingest or index failure is not captured — it raises immediately
    (notifying first on the scheduled path, so an unattended failure is never
    silent). The optional ChatGPT profile preserves the internal test helper's
    single-profile call shape; all CLI and scheduled callers pass it explicitly."""
    if chatgpt_profile is None:
        chatgpt_profile = claude_profile
    ensure_cluster()
    net_errs: list[Exception] = []
    sources = [("claude.ai", lambda: run_sync(
                            claude_profile, org_ref, full,
                            download_files=download_files)),
                       ("ChatGPT", lambda: ingest_chatgpt(chatgpt_profile,
                                                           full=full))]
    import artiport_source
    if artiport_source.config_path().exists():
        sources.append(("Artiport", lambda: artiport_source.ingest(full=full)))
    import opencode_source
    if opencode_source.config_path().exists():
        sources.append(("OpenCode", lambda: opencode_source.ingest(full=full)))
    for label, run in sources:
        try:
            run()
        except Exception as e:                # captured, re-raised by caller (loud)
            if notify_fail:
                notify("fail", f"clync {label} sync failed", str(e))
            print(f"{label} sync FAILED: {e}\n  -> continuing with the remaining "
                  f"network + local ingests", file=sys.stderr)
            net_errs.append(e)
    try:
        ingest_cc(full=full)
        ingest_codex(full=full)
        ingest_pi(full=full)
    except Exception as e:
        if notify_fail:
            notify("fail", "clync local ingest failed", str(e))
        raise
    if do_index:
        _index_after_sync(full)
    if not net_errs:
        return None
    if len(net_errs) == 1:
        return net_errs[0]
    return RuntimeError("; ".join(f"{type(e).__name__}: {e}" for e in net_errs))


def cmd_sync(args) -> int:
    err = _sync_all(_require_claude_profile(args), args.org, args.full,
                    download_files=not args.no_files, do_index=not args.no_index,
                    notify_fail=False,
                    chatgpt_profile=_require_chatgpt_profile(args))
    if err:
        raise err
    return 0


def cmd_sync_app(args) -> int:
    """Sync claude.ai (the app) ONLY — network/cookies, fails loud — then index.
    No local Claude Code ingest."""
    ensure_cluster()
    run_sync(_require_claude_profile(args), args.org, args.full,
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


def cmd_sync_codex(args) -> int:
    """Ingest local Codex CLI sessions ONLY (no network/cookies), then index."""
    ensure_cluster()
    ingest_codex(full=args.full)
    if not args.no_index:
        _index_after_sync(args.full)
    return 0


def cmd_sync_pi(args) -> int:
    """Ingest local Pi sessions ONLY (no network/cookies), then index."""
    ensure_cluster()
    ingest_pi(full=args.full)
    if not args.no_index:
        _index_after_sync(args.full)
    return 0


def cmd_sync_artiport(args) -> int:
    import artiport_source
    ensure_cluster()
    artiport_source.ingest(full=args.full)
    if not args.no_index:
        _index_after_sync(args.full)
    return 0


def cmd_sync_opencode(args) -> int:
    import opencode_source
    ensure_cluster()
    opencode_source.ingest(full=args.full)
    if not args.no_index:
        _index_after_sync(args.full)
    return 0


def cmd_sync_chatgpt(args) -> int:
    """Sync ChatGPT web history ONLY — network/cookies, fails loud — then index."""
    ensure_cluster()
    ingest_chatgpt(_require_chatgpt_profile(args), full=args.full,
                   projects_only=args.projects_only)
    if not args.no_index:
        _index_after_sync(args.full)
    return 0


def cmd_index(args) -> int:
    import search
    ensure_cluster()                  # command entry provisions; build_index does not
    try:
        stats = search.build_index(full=args.full)
    except Exception as e:
        notify("fail", "clync index failed", str(e))
        raise
    print(f"reindexed_units={stats['reindexed_units']} chunks={stats['chunks']} "
          f"removed_units={stats['removed_units']} "
          f"dropped_chunks={stats['dropped_chunks']} "
          f"repositioned_chunks={stats['repositioned_chunks']}")
    return 0


def cmd_migrate(args) -> int:
    """Apply a destructive rebuild of the DERIVED layers whose shape went stale.

    The one door for that rebuild. Everything else refuses (`StaleDerivedSchema`)
    so no read, sync or nightly run can destroy derived data as a side effect."""
    before = {}
    if cluster_running():
        with connect_pg() as con:
            for t in ("chunks", "dream_insights"):
                exists = con.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema='public' AND table_name=%s", (t,)).fetchone()
                before[t] = con.execute(
                    f"SELECT count(*) AS n FROM {t}").fetchone()["n"] if exists else 0
    ensure_cluster(rebuild_derived=True)
    with connect_pg() as con:
        after = {t: con.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"]
                 for t in ("chunks", "dream_insights")}
    for t in ("chunks", "dream_insights"):
        lost = before.get(t, 0) - after[t]
        print(f"{t}: {before.get(t, 0)} -> {after[t]}"
              + (f"  ({lost} row(s) dropped; re-derive with "
                 f"{'clync index' if t == 'chunks' else 'clync dream backfill'})"
                 if lost > 0 else "  (unchanged)"))
    # Dropped insights cost model calls to re-derive, so the rebuild copies them
    # first. Print what survived: a backup the operator cannot see is a backup they
    # cannot rely on.
    import dream
    with connect_pg() as con:
        snaps = dream.list_snapshots(con)
    if snaps:
        print(f"\nsnapshots (inert copies, GC'd after {dream.SNAPSHOT_KEEP_DAYS} days):")
        for name, n in snaps:
            print(f"  {name}: {n} row(s)")
    print("schema migrated. Search and dream schemas now match the shipped layout.")
    return 0


def cmd_scheduled(args) -> int:
    """Entry point for the launchd job: sync, notify LOUDLY on failure, and warn
    (but do not error) when a prior daily run was missed because the Mac was down."""
    try:
        claude_profile = _require_claude_profile(args)
        chatgpt_profile = _require_chatgpt_profile(args)
    except RuntimeError as e:
        notify("fail", "clync sync failed", str(e))
        raise
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
    err = _sync_all(claude_profile, args.org, full=False, download_files=True,
                    do_index=True, notify_fail=True,
                    chatgpt_profile=chatgpt_profile)
    if late_gap_h is not None:
        notify("warn", "clync ran late",
               f"previous scheduled sync was missed (~{late_gap_h:.0f}h gap) — "
               "Mac was likely asleep/off. Synced now.")
    if err:
        raise err
    if args.no_dream:
        print("[scheduled] sync and index ok; dream disabled")
        return 0
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
    # A topic whose dig failed is isolated inside run_incremental so the other topics still
    # complete and persist — but an isolated failure is still a failure: surface it loudly
    # (notify + non-zero exit) rather than letting it hide behind an "ok" line.
    failed = report.get("failed_topics") or []
    if failed:
        summary = "; ".join(f"{f['topic_id']}: {f['error'][:150]}" for f in failed)
        notify("fail", f"clync dream: {len(failed)} topic(s) failed",
               f"other topics completed; these did not: {summary}")
        print(f"[scheduled] dream ran (calls={report['calls']} "
              f"changed={report['changed']} topics={report['topics_touched']}) but "
              f"{len(failed)} topic(s) FAILED: {summary}", file=sys.stderr)
        return 1
    print(f"[scheduled] ok; dream calls={report['calls']} "
          f"changed={report['changed']} topics={report['topics_touched']}")
    return 0


def cmd_search(args) -> int:
    import search
    ensure_cluster()                  # command entry provisions; hybrid_search does not
    results = search.hybrid_search(
        args.query, topk=args.limit, source=args.source, project=args.project,
        model=args.model, repo=args.repo, worktree=args.worktree, branch=args.branch,
        session=args.session, sender=args.sender, since=args.since, until=args.until,
        sort=args.sort, lang=args.lang, per_unit=args.per_unit)
    if not results:
        print("(no matches)")
    for r in results:
        snippet = r["text"].replace("\n", " ")[:200]
        at = "" if r["msg_idx"] is None else f"  msg_idx={r['msg_idx']}"
        who = "" if r["sender"] is None else f"  sender={r['sender']}"
        print(f"\n• [{r['source']}] {r['unit_name'] or '(untitled)'}  "
              f"[{r['unit_id']}]{who}{at}  score={r['score']:.4f}\n  {snippet}")
    return 0


class ConversationNotFoundError(LookupError):
    pass


def read_conversation(uuid: str, from_idx: int | None = None,
                      to_idx: int | None = None, sender: str | None = None) -> str:
    import cc
    senders = cc.resolve_sender_role(sender) if sender else None
    con = connect()
    try:
        u = con.execute(
            "SELECT kind, source, title, summary, model, updated_at, repo, cwd, "
            "worktree, git_branch, cc_version, project_name, msg_count "
            "FROM units WHERE unit_id=%s", (uuid,)).fetchone()
        if not u:
            raise ConversationNotFoundError(f"No unit with uuid {uuid!r} in the local store.")
        window = "" if from_idx is None else " AND idx >= %(lo)s"
        window += "" if to_idx is None else " AND idx <= %(hi)s"
        window += "" if senders is None else " AND sender = ANY(%(senders)s)"
        msgs = con.execute(
            "SELECT idx, sender, role_detail, text, created_at FROM messages "
            "WHERE unit_id=%(uuid)s" + window + " ORDER BY idx",
            {"uuid": uuid, "lo": from_idx, "hi": to_idx, "senders": senders}).fetchall()
        files = con.execute(
            "SELECT file_kind, file_name, local_path FROM files WHERE unit_id=%s",
            (uuid,)).fetchall()
    finally:
        con.close()

    if u["cwd"] is not None:
        meta = (f"source={u['source']} repo={u['repo']} model={u['model']} "
                f"worktree={u['worktree']} branch={u['git_branch']}\n"
                f"cwd={u['cwd']}\n")
        if u["cc_version"] is not None:
            meta += f"cc_version={u['cc_version']}\n"
    else:
        meta = (f"source={u['source']} model={u['model']} "
                f"project={u['project_name'] or '(none)'}\n"
                f"summary: {u['summary'] or '(none)'}\n")
    header = f"# {u['title'] or '(untitled)'}\nupdated={u['updated_at']}\n{meta}"
    if files:
        header += "attached files:\n" + "\n".join(
            f"  - [{f['file_kind']}] {f['file_name']}"
            + (f" -> {f['local_path']}" if f['local_path'] else " (not downloaded)")
            for f in files
        ) + "\n"
    if not msgs:
        header += f"messages none of {u['msg_count']}\n"
    elif sender:
        header += (f"messages: {len(msgs)} {sender} turn(s) spanning "
                   f"{msgs[0]['idx']}-{msgs[-1]['idx']} of {u['msg_count']}\n")
    else:
        header += f"messages {msgs[0]['idx']}-{msgs[-1]['idx']} of {u['msg_count']}\n"
    body = "\n\n".join(
        f"## [{m['idx']}] {m['sender']}"
        + (f" [{m['role_detail']}]" if m["role_detail"] else "")
        + f" ({m['created_at']})\n{m['text']}"
        for m in msgs
    )
    return f"{header}\n{body}"


def cmd_read(args) -> int:
    ensure_cluster()
    print(read_conversation(args.uuid, args.from_idx, args.to_idx, args.sender))
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
    """List dream topics: id, status, active insight count, last dig time."""
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
              f"last_dig_at={r['last_dig_at']}")
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
    # An isolated topic failure is still a failure: exit non-zero so a caller (or the
    # operator watching $?) sees it, not just the printed "! FAILED" line.
    return 1 if report.get("failed_topics") else 0


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
    return 1 if report.get("failed_topics") else 0


def _print_dream_report(report: dict) -> None:
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
    # What a real reader asked and this layer could not answer. Those words are the
    # best probe queries available — better than a charter guess — so say when they
    # were fed back, and name the ones that matched no topic at all: those are new
    # topic candidates, which is the operator's call, not a silent insert.
    for tid, qs in (report.get("questions_fed") or {}).items():
        print(f"questions   : fed {len(qs)} unanswered question(s) into '{tid}' "
              f"as probe queries: {'; '.join(q[:60] for q in qs)}")
    if report.get("topic_candidates"):
        print(f"! {len(report['topic_candidates'])} unanswered question(s) match NO "
              f"topic — candidates for a new one (`clync dream topics`): "
              f"{'; '.join(q[:60] for q in report['topic_candidates'])}")
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
            # Whatever the dig CAPPED or could not serve — an unserved read request,
            # a unit it could not position. Cheap-looking output must never hide
            # work that was silently skipped.
            for n in d.get("notes", []):
                print(f"      ! {n}")
    print(f"calls       : {report['calls']}")
    if report.get("stopped_early"):
        print(f"stopped_early: {report['stopped_early']}")
    # An isolated topic failure (run_incremental kept the other topics going) must be
    # visible here too, or a manual run hides it behind an otherwise-clean report.
    for f in report.get("failed_topics", []):
        print(f"  ! FAILED[{f['topic_id']}] {f['error']}")
    # A night that digs nothing is normal (topics batch up across nights) — but it
    # must SAY what is waiting, so "cheap" is never mistaken for "stuck".
    for d in report.get("deferred", []):
        print(f"  deferred[{d['topic_id']}] pending={d['pending']} "
              f"oldest={d['oldest']} (digs at {dream_thresholds()})")
    if report.get("note"):
        print(f"note        : {report['note']}")


def dream_thresholds() -> str:
    import dream
    return (f">={dream.DIG_MIN_UNITS} units or >={dream.DIG_MAX_DEFER_DAYS}d old")


def cmd_dream_recall(args) -> int:
    """Recall: coverage / insights / raw transcript sections."""
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
              f"last_dig_at={r['last_dig_at']}")
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
            ncodex = _n("SELECT COUNT(*) n FROM units WHERE kind='codex_session'")
            nopencode = _n("SELECT COUNT(*) n FROM units WHERE source='opencode'")
            npi = _n("SELECT COUNT(*) n FROM units WHERE source='pi'")
            nchatgpt = _n("SELECT COUNT(*) n FROM units WHERE kind='chatgpt'")
            nmsg = _n("SELECT COUNT(*) n FROM messages")
            nfile = _n("SELECT COUNT(*) n FROM files")
            ndl = _n("SELECT COUNT(*) n FROM files WHERE local_path IS NOT NULL")
            last = get_meta(con, "last_success")
        finally:
            con.close()
        print(f"store       : {PG_DB}@localhost:{PG_PORT}\n"
              f"              {nchat} chats, {ndoc} project docs, {ncc} cc sessions, "
              f"{ncodex} codex sessions, {nopencode} opencode sessions, {npi} pi sessions, "
              f"{nchatgpt} chatgpt sessions, "
              f"{nmsg} messages, {nfile} files ({ndl} images), last_success={last}")
        print(f"search index: ok ({st['chunks']} chunks indexed)")
        if st.get("pgvector_warning"):
            print(f"              note: {st['pgvector_warning']}")
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
        failed_queue = dst["usage"]["failed"]
        print(f"dream       : {n_active_topics} active topic(s), {n_active_insights} "
              f"active insight(s), {failed_queue} failed queue item(s)")
        if failed_queue:
            problems.append(f"dream has {failed_queue} failed queue item(s) — "
                            "run `clync dream status` and retry the failed work.")

    try:
        import mcp_server  # noqa: F401 — self-test that deps resolve + server builds
        print("MCP server  : import OK (deps resolve, FastMCP builds)")
    except Exception as e:
        problems.append(f"MCP server import failed: {e}")

    store_job = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{PG_LAUNCHD_LABEL}"],
        capture_output=True, text=True)
    print(f"store job   : {'loaded' if store_job.returncode == 0 else 'NOT loaded'} "
          f"({PG_LAUNCHD_LABEL})")
    if store_job.returncode != 0:
        problems.append("store has no launchd supervision — run `clync install`.")

    launchd = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"],
        capture_output=True, text=True,
    )
    loaded = launchd.returncode == 0
    last_exit_match = re.search(r"^\s*last exit code = (-?\d+)\s*$",
                                launchd.stdout, re.MULTILINE)
    last_exit = int(last_exit_match.group(1)) if last_exit_match else None
    suffix = f", last_exit={last_exit}" if last_exit is not None else ""
    print(f"launchd job : {'loaded' if loaded else 'NOT loaded'} ({LAUNCHD_LABEL}{suffix})")
    if not loaded:
        problems.append("scheduler not installed — run `clync install`.")
    elif last_exit not in (None, 0):
        problems.append(f"last scheduled run exited {last_exit} — "
                        "run `clync status` and inspect scheduled.log.")

    if args.mcp_client == "opencode":
        route = "/api/mcp?" + urlencode({"location[directory]": str(Path.cwd())})
        reg = subprocess.run(["opencode", "api", "get", route],
                             capture_output=True, text=True, check=True, timeout=30)
        servers = json.loads(reg.stdout)["data"]
        server = next((s for s in servers if s["name"] == MCP_NAME), None)
        state = server["status"]["status"] if server else "not registered"
        print(f"MCP OpenCode: {state}")
        if state != "connected":
            problems.append(f"OpenCode MCP {state} — see README 'Connect an assistant'.")
    else:
        client = args.mcp_client
        if client == "claude":
            reg = subprocess.run([client, "mcp", "get", MCP_NAME],
                                 capture_output=True, text=True, check=False)
            if reg.returncode != 0:
                problems.append("Claude Code MCP not registered — see README.")
            path, parse, key = Path.home() / ".claude.json", json.loads, "mcpServers"
        else:
            path, parse, key = Path.home() / ".codex/config.toml", tomllib.loads, "mcp_servers"
        registration = None
        if path.exists():
            registration = parse(path.read_text()).get(key, {}).get(MCP_NAME)
        if registration:
            print(f"MCP register: registered with {client}")
            if not shutil.which(registration["command"]):
                problems.append(f"{client} MCP executable missing: {registration['command']} "
                                "— re-register clync (see README).")
        elif client == "codex":
            problems.append("Codex MCP not registered — see README.")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    print("\nAll green.")
    return 0


def cmd_install(args) -> int:
    """Install the supervised database and daily sync LaunchAgents."""
    claude_profile = _require_claude_profile(args)
    chatgpt_profile = _require_chatgpt_profile(args)
    hour, minute = (int(x) for x in args.at.split(":"))
    install_cluster_service()
    SCHED_LOG.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    # A login shell resolves `uv` on PATH (pyenv shims aren't reliable under launchd).
    cmd = f"cd {shlex.quote(str(REPO_DIR))} && exec uv run python clync.py scheduled"
    if args.no_dream:
        cmd += f" {NO_DREAM_FLAG}"
    # Serialize through plistlib rather than interpolating XML: the shell
    # command contains `&&`, and arbitrary paths/profiles may contain XML
    # metacharacters. A malformed plist makes launchd state misleading.
    plist = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": ["/bin/zsh", "-lc", cmd],
        "EnvironmentVariables": {
            CLAUDE_PROFILE_ENV: claude_profile,
            CHATGPT_PROFILE_ENV: chatgpt_profile,
        },
        "StartCalendarInterval": {"Hour": hour, "Minute": minute},
        "StandardOutPath": str(SCHED_LOG),
        "StandardErrorPath": str(SCHED_LOG),
    }
    PLIST_PATH.write_bytes(plistlib.dumps(plist, fmt=plistlib.FMT_XML))
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LAUNCHD_LABEL}"],
                   capture_output=True, text=True)  # idempotent: ignore "not loaded"
    r = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(PLIST_PATH)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"launchctl bootstrap failed: {r.stderr.strip()}")
    print(f"Installed {LAUNCHD_LABEL}: daily at {hour:02d}:{minute:02d}, "
          f"claude_profile={claude_profile!r}, chatgpt_profile={chatgpt_profile!r}\n"
          f"  plist: {PLIST_PATH}\n  log:   {SCHED_LOG}")
    print("Missed runs (Mac asleep/off) run once on wake and fire a 'ran late' "
          "warning; failures fire a modal+sound alert.")
    return 0


def cmd_uninstall(args) -> int:
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LAUNCHD_LABEL}"],
                   capture_output=True, text=True)
    if PLIST_PATH.exists():
        PLIST_PATH.unlink()
    if PG_PLIST_PATH.exists():
        service = f"gui/{os.getuid()}/{PG_LAUNCHD_LABEL}"
        if subprocess.run(["launchctl", "print", service], capture_output=True).returncode == 0:
            unload_cluster_service()
        else:
            stop_cluster()
        PG_PLIST_PATH.unlink()
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
    _require_claude_profile(args)
    _require_chatgpt_profile(args)

    subprocess.run(["uv", "sync", "--quiet"], cwd=REPO_DIR, check=True)
    print("✓ deps synced (uv)")

    _relink(BIN_LINK, REPO_DIR / "clync")
    on_path = str(BIN_LINK.parent) in os.environ.get("PATH", "").split(":")
    print(f"✓ CLI: {BIN_LINK} -> clync"
          + ("" if on_path else f"   ⚠ {BIN_LINK.parent} is NOT on your PATH"))

    subprocess.run(["claude", "mcp", "remove", MCP_NAME], capture_output=True, text=True)
    r = subprocess.run(
        ["claude", "mcp", "add", "--scope", "user", MCP_NAME, "--",
         "/bin/zsh", "-lc", 'exec uv run --project "$1" python "$1/mcp_server.py"',
         MCP_NAME, str(REPO_DIR)],
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
          f"(nightly pass {'disabled' if args.no_dream else 'runs inside the daily job'}; "
          f"run `clync dream backfill` "
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


def cmd_sessions(args) -> int:
    import live_search
    if args.session_action == "search":
        result = live_search.search_sessions(
            args.query, source=args.source, repo=args.repo, sender=args.sender,
            since=args.since, limit=args.limit, include_tools=args.include_tools,
            include_subagents=args.include_subagents, exact=args.exact,
            exhaustive=args.exhaustive)
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json
              else live_search.format_search(result))
        return 0 if all(c["status"] == live_search.COMPLETE for c in result["coverage"].values()) else 2
    result = live_search.read_session(args.reference, args.around,
                                     context=args.context, include_tools=args.include_tools)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"{result['reference']} — {result['title']} (bounded source window)")
        for record in result["records"]:
            print(f"\n[{record['sender']}] {record['anchor']}\n{record['text']}")
        if result["truncated"]:
            print("\n[output truncated]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    dream_schedule = argparse.ArgumentParser(add_help=False)
    dream_schedule.add_argument(NO_DREAM_FLAG, action="store_true",
                                help="disable scheduled Dream inference; keep sync and indexing")

    # Network sources authenticate against different accounts, so their profile
    # flags are separate. A shared `--profile` would make cross-account sync easy
    # to trigger accidentally and is intentionally not offered.
    claude_cred = argparse.ArgumentParser(add_help=False)
    claude_cred.add_argument("--claude-profile", dest="claude_profile",
                             default=os.environ.get(CLAUDE_PROFILE_ENV),
                             help=f"Claude.ai Chrome profile (default: ${CLAUDE_PROFILE_ENV})")
    claude_cred.add_argument("--org", help="restrict to one org name or uuid "
                                      "(default: all chat-capable orgs)")

    chatgpt_cred = argparse.ArgumentParser(add_help=False)
    chatgpt_cred.add_argument("--chatgpt-profile", dest="chatgpt_profile",
                              default=os.environ.get(CHATGPT_PROFILE_ENV),
                              help=f"ChatGPT Chrome profile (default: ${CHATGPT_PROFILE_ENV})")

    split_cred = argparse.ArgumentParser(add_help=False)
    split_cred.add_argument("--claude-profile", dest="claude_profile",
                            default=os.environ.get(CLAUDE_PROFILE_ENV),
                            help=f"Claude.ai Chrome profile (default: ${CLAUDE_PROFILE_ENV})")
    split_cred.add_argument("--chatgpt-profile", dest="chatgpt_profile",
                            default=os.environ.get(CHATGPT_PROFILE_ENV),
                            help=f"ChatGPT Chrome profile (default: ${CHATGPT_PROFILE_ENV})")
    split_cred.add_argument("--org", help="restrict Claude.ai to one org name or uuid "
                                      "(default: all chat-capable orgs)")

    sub.add_parser("whoami", parents=[claude_cred], help="print the resolved Claude.ai account + org"
                   ).set_defaults(func=cmd_whoami)

    sp = sub.add_parser(
        "sync", parents=[split_cred],
        help="sync web and local sessions + configured sources, then index",
    )
    sp.add_argument("--full", action="store_true", help="re-fetch/re-parse everything")
    sp.add_argument("--no-files", action="store_true", help="skip image downloads")
    sp.add_argument("--no-index", action="store_true",
                     help="skip hybrid-search indexing after sync")
    sp.set_defaults(func=cmd_sync)

    sp = sub.add_parser("sync-app", parents=[claude_cred],
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

    sp = sub.add_parser("sync-chatgpt", parents=[chatgpt_cred],
                        help="sync ChatGPT web history ONLY (network/cookies)")
    sp.add_argument("--full", action="store_true",
                    help="re-fetch every conversation (ignore incremental watermarks)")
    sp.add_argument("--no-index", action="store_true", help="skip reindex after sync")
    sp.add_argument("--projects-only", action="store_true",
                     help="sync project threads and reconcile their membership; "
                          "leave unrelated ordinary history unchanged")
    sp.set_defaults(func=cmd_sync_chatgpt)

    sp = sub.add_parser("sync-codex", help="ingest local Codex CLI sessions ONLY "
                                           "(no network / cookies) + index")
    sp.add_argument("--full", action="store_true", help="re-parse every rollout")
    sp.add_argument("--no-index", action="store_true", help="skip indexing after ingest")
    sp.set_defaults(func=cmd_sync_codex)

    sp = sub.add_parser("sync-pi", help="ingest local Pi sessions ONLY "
                                        "(no network / cookies) + index")
    sp.add_argument("--full", action="store_true", help="re-parse every transcript")
    sp.add_argument("--no-index", action="store_true", help="skip indexing after ingest")
    sp.set_defaults(func=cmd_sync_pi)

    sp = sub.add_parser("sync-artiport", help="sync configured Artiport static artifacts + index")
    sp.add_argument("--full", action="store_true", help="replay the source change feed")
    sp.add_argument("--no-index", action="store_true", help="skip reindex after ingest")
    sp.set_defaults(func=cmd_sync_artiport)

    sp = sub.add_parser("sync-opencode", help="sync OpenCode v2 root transcripts + index")
    sp.add_argument("--full", action="store_true", help="re-export every root session")
    sp.add_argument("--no-index", action="store_true", help="skip indexing after sync")
    sp.set_defaults(func=cmd_sync_opencode)

    sp = sub.add_parser("index", help="(re)build the hybrid-search index")
    sp.add_argument("--full", action="store_true", help="reindex every unit")
    sp.set_defaults(func=cmd_index)

    sub.add_parser(
        "scheduled", parents=[split_cred, dream_schedule],
        help="launchd entry point (all-source sync + loud fail/late notify)",
    ).set_defaults(func=cmd_scheduled)

    sub.add_parser("migrate", help="rebuild DERIVED schemas whose shape went stale "
                                   "(DESTRUCTIVE: drops embeddings/insights)"
                   ).set_defaults(func=cmd_migrate)

    sp = sub.add_parser("search", help="faceted hybrid (dense + sparse) semantic "
                                       "search across all raw sources")
    sp.add_argument("query", nargs="?", default="",
                    help="natural-language query (empty => recency browse)")
    sp.add_argument("--limit", type=int, default=DEFAULT_TOPK)
    sp.add_argument("--source", choices=VALID_SOURCES,
                    default="all", help="restrict to one source (default: all)")
    sp.add_argument("--project", help="claude.ai or ChatGPT project name/uuid; "
                    "'any' = in some project, 'none' = not in a project")
    sp.add_argument("--model", help="claude.ai model facet")
    sp.add_argument("--repo", help="local session repo name")
    sp.add_argument("--worktree", help="Claude Code worktree name")
    sp.add_argument("--branch", help="Claude Code git branch")
    sp.add_argument("--session", help="local session name or clync unit id")
    sp.add_argument("--sender", help="whose words the hit must be: user | assistant")
    sp.add_argument("--per-unit", type=int, default=1, dest="per_unit",
                    help="hits one unit may contribute (default 1)")
    sp.add_argument("--since", help="only units updated on/after this ISO date")
    sp.add_argument("--until", help="only units updated on/before this ISO date")
    sp.add_argument("--sort", choices=["relevance", "recency"], default="relevance")
    sp.add_argument("--lang", choices=["en", "ja", "zh"], default=None,
                     help="restrict to one language (default: all)")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("read", help="read an archived conversation, optionally bounded by index/role")
    sp.add_argument("uuid", help="unit ID returned by archived search")
    sp.add_argument("--from-idx", type=int)
    sp.add_argument("--to-idx", type=int)
    sp.add_argument("--sender", choices=["user", "assistant"])
    sp.set_defaults(func=cmd_read)

    sessions = sub.add_parser("sessions", help="live lexical search; no sync, database or embeddings")
    actions = sessions.add_subparsers(dest="session_action", required=True)
    sp = actions.add_parser("search", help="find current native sessions with bounded typo tolerance")
    sp.add_argument("query", help="keywords matched within one passage, not a semantic question")
    sp.add_argument("--source", default="all", help="all or comma-separated claude_code,codex_cli,opencode,grok,pi")
    sp.add_argument("--repo", help="repository name or exact directory component")
    sp.add_argument("--sender", choices=["user", "assistant", "tool"])
    sp.add_argument("--since", help="ISO date; source modification / update time")
    sp.add_argument("--limit", type=int, default=DEFAULT_TOPK)
    sp.add_argument("--exact", action="store_true", help="disable one-edit ASCII word matching")
    sp.add_argument("--exhaustive", action="store_true", help="include fuzzy-only passages alongside anchored hits")
    sp.add_argument("--include-subagents", action="store_true")
    sp.add_argument("--include-tools", action="store_true", help="include reasoning and tool records with attribution")
    sp.add_argument("--json", action="store_true", help="structured hits and per-source coverage")
    sp.set_defaults(func=cmd_sessions)
    sp = actions.add_parser("read", help="read a bounded live source window around a returned anchor")
    sp.add_argument("reference")
    sp.add_argument("--around", required=True, help="anchor returned by sessions search")
    sp.add_argument("--context", type=int, default=3, help="neighboring records per side (0-10)")
    sp.add_argument("--include-tools", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_sessions)

    sp = sub.add_parser("list", help="most recently updated units")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--source", choices=VALID_SOURCES, default="all")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("doctor", help="health check store / index / launchd / MCP")
    sp.add_argument("--mcp-client", choices=["claude", "codex", "opencode"], default="claude",
                    help="client whose Clync registration to check (default: claude)")
    sp.set_defaults(func=cmd_doctor)

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

    sp = dsub.add_parser("recall", help="recall: insights + raw transcripts, in labelled sections")
    sp.add_argument("query", nargs="?", default="", help="natural-language query")
    sp.add_argument("--topic", help="restrict to one topic id")
    sp.add_argument("--limit", type=int, default=DEFAULT_TOPK)
    sp.add_argument("--as-of", dest="as_of", help="ISO date — positions held at that date")
    sp.add_argument("--evidence", action="store_true", help="also print cited quotes")
    sp.set_defaults(func=cmd_dream_recall)

    dsub.add_parser("status", help="watermark, topic table, usage totals"
                    ).set_defaults(func=cmd_dream_status)

    sp = sub.add_parser("setup", parents=[split_cred, dream_schedule],
                        help="one-shot install: deps, CLI, MCP, scheduler, skill")
    sp.add_argument("--at", default="09:00", help="daily sync time HH:MM (default 09:00)")
    sp.set_defaults(func=cmd_setup)

    sub.add_parser("unsetup", help="remove everything setup installed (keeps the DB)"
                   ).set_defaults(func=cmd_unsetup)

    sp = sub.add_parser("install", parents=[split_cred, dream_schedule],
                        help="install the supervised database and daily launchd sync")
    sp.add_argument("--at", default="09:00", help="daily time HH:MM (default 09:00)")
    sp.set_defaults(func=cmd_install)

    sub.add_parser("uninstall", help="remove launchd jobs and stop the database (keeps its data)"
                   ).set_defaults(func=cmd_uninstall)

    sp = sub.add_parser("status", help="scheduled-sync status + recent run log")
    sp.add_argument("--tail", type=int, default=15, help="log lines to show")
    sp.set_defaults(func=cmd_status)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
