# clync

Headless sync of **claude.ai** conversation history into a local SQLite/FTS5
database, exposed to Claude Code over MCP. macOS + Chrome only.

## How it works

1. **Auth** — reads your claude.ai cookies (`sessionKey` + `cf_clearance`) *live*
   from a named Chrome profile's cookie store each run (decrypted with the
   Chrome Safe Storage key from your login keychain). Nothing is stored; the
   cookies are as fresh as your browser's last claude.ai activity.
2. **Fetch** — calls claude.ai's private API through `curl_cffi` with Chrome TLS
   impersonation, which is what clears Cloudflare (a plain HTTP client gets a 403
   challenge). Requests are paced; transient 403s are retried with backoff.
3. **Store** — upserts conversations + messages (including **text-attachment
   content**) into `~/.local/share/clync/history.db`. Incremental by `updated_at`.
4. **Query** — `mcp_server.py` exposes `search_history`, `get_conversation`,
   `list_conversations` to Claude Code over stdio.

**Fail-loud:** any auth/HTTP/schema error raises and exits non-zero. It never
silently serves stale data. The scheduled run additionally fires a macOS
modal+sound alert on failure.

## Setup

One command wires up everything (deps, `clync` CLI on your PATH, MCP
registration, daily launchd sync, and the Claude Code skill) — all defined in
this repo, symlinked/registered out:

```sh
uv run python clync.py setup --profile "Work"   # or export CLYNC_PROFILE; --at HH:MM for time
uv run python clync.py doctor                      # verify: DB / deps / launchd / MCP
```

`clync unsetup` removes every external artifact (symlinks, launchd job, MCP
registration) and leaves the local DB intact. Then open a **new** Claude Code
session to load the MCP tools + skill.

## Commands

| Command | Purpose |
|---|---|
| `sync [--full] [--no-files]` | incremental sync (`--full` re-fetches all; `--no-files` skips image downloads) |
| `scheduled` | launchd entry point: sync + loud fail/late notification |
| `search <q> [--limit N]` | FTS5 search over message + attachment text |
| `list [--limit N]` | most recently updated conversations |
| `whoami` | resolved account + org |
| `doctor` | health check: DB, deps, launchd job, MCP registration |
| `status [--tail N]` | last successful sync, launchd state, recent scheduled-run log |
| `install [--at HH:MM]` / `uninstall` | manage the daily launchd job |

A `clync` wrapper script sits in the repo; symlink it onto your PATH for global use:

```sh
ln -s ~/code/clync/clync ~/.local/bin/clync   # then: clync search "…", clync sync, clync status
```

## Config

| Setting | Source | Default |
|---|---|---|
| Chrome profile | `--profile` / `$CLYNC_PROFILE` | *(required — fails loud if unset)* |
| Target org | `--org <name-or-uuid>` | first (active) org in the account |
| DB path | `$CLYNC_DB` | `~/.local/share/clync/history.db` |

## Scheduling & missed runs

launchd (`io.clync.sync`) runs daily. If the Mac is asleep/off at the scheduled
time, launchd runs the job **once** on wake; clync detects the >25h gap and fires
a "ran late" warning (it does not attempt catch-up storms). Notifications reach
this Mac only.

## Limitations

- **Cookie staleness:** if `sessionKey`/`cf_clearance` expire (you logged out, or
  the browser hasn't touched claude.ai in a long time), sync fails loud with a
  message to open claude.ai in the Chrome profile. There is no silent recovery —
  refreshing requires a real browser session (Cloudflare's JS challenge can't be
  solved headlessly).
- **File uploads:**
  - *Text attachments* (`.md`/`.txt`/`.docx` …) — fully indexed via their
    `extracted_content` (searchable).
  - *Images* — downloaded to `~/.local/share/clync/files/<uuid>.<ext>` and
    recorded in the `files` table. Note claude.ai serves them re-encoded as
    **webp**, so this is the full-resolution image, not the byte-identical
    original upload.
  - *Binary blobs* (non-image, non-text uploads) — recorded as metadata only.
    claude.ai exposes no download URL for them (the file object gives only a
    server-sandbox path), so their bytes are not retrievable.
- **Auth cookies are per Chrome profile** — a browser-level cookie jar, one
  claude.ai login per profile. The tool is pointed at a profile because that is
  where the login cookie physically lives; the org is selected after
  authenticating (`--org` to pick a non-default one).
