# clync — guide for Claude Code

clync syncs the user's **claude.ai** conversation history into a local
SQLite/FTS5 database and exposes it to Claude Code over MCP. macOS + Chrome only.

## Docs index

This file is an index — the actual documentation lives in these:

| Doc | What's in it |
|---|---|
| [README.md](README.md) | **Start here.** What it is, one-command setup, all CLI commands, config, architecture, and limitations. |
| [skill/SKILL.md](skill/SKILL.md) | The Claude Code skill — when/how to query the user's history. |
| `clync.py` (module docstring + `--help`) | Sync engine, cookie/Cloudflare handling, storage, CLI. |
| `mcp_server.py` (docstring) | The three MCP tools (`search_history`, `get_conversation`, `list_conversations`). |

## One-command setup

```sh
clync setup --profile <ChromeProfile>     # or export CLYNC_PROFILE
```

Installs deps, the `clync` CLI wrapper, the MCP registration, the daily launchd
sync, and the skill — all defined in this repo, symlinked/registered out.
`clync unsetup` reverses it (keeps the DB).

## Orientation for editing here

- **SSOT:** `clync.py` owns cookies + API client + DB; `mcp_server.py` imports
  from it. Don't duplicate DB/query logic across the two.
- **Config, not hardcoded:** `$CLYNC_PROFILE` / `--profile`, `$CLYNC_DB`, `--org`.
  Nothing personal is baked into source.
- **Fail loud:** auth/HTTP/schema errors raise and exit non-zero; the scheduled
  run also fires a macOS alert. Never degrade silently to stale data. Exception
  handlers re-raise (see the repo's coding rules).
- **Runtime state** (DB, downloaded images, scheduled log) lives in
  `~/.local/share/clync/` — outside the repo, never committed.
