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

```sh
# one-time: install the daily background sync (reads $CLYNC_PROFILE)
CLYNC_PROFILE="Work" uv run python clync.py install        # daily 09:00; --at HH:MM to change

# one-time: register the MCP server with Claude Code (all projects)
claude mcp add --scope user clync -- \
  "$(command -v uv)" run --project ~/code/clync python ~/code/clync/mcp_server.py

# check everything
uv run python clync.py doctor
```

## Commands

| Command | Purpose |
|---|---|
| `sync [--full]` | incremental sync (`--full` re-fetches everything) |
| `scheduled` | launchd entry point: sync + loud fail/late notification |
| `search <q> [--limit N]` | FTS5 search over message + attachment text |
| `list [--limit N]` | most recently updated conversations |
| `whoami` | resolved account + org |
| `doctor` | health check: DB, deps, launchd job, MCP registration |
| `install [--at HH:MM]` / `uninstall` | manage the daily launchd job |

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
- **Binary file uploads** (images, non-text files in the `files` field) are
  recorded as metadata in the stored raw JSON but **not downloaded**. Text
  attachments (`.md`/`.txt`/`.docx` etc.) are fully indexed via their
  `extracted_content`.
- **Auth cookies are per Chrome profile**, so the tool is pointed at a profile,
  not directly at an org — the org is selected after authenticating.
