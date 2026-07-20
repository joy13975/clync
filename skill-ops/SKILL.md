---
name: clync-ops
description: Operate and troubleshoot the clync sync + hybrid-search tool — run or debug a sync, (re)build the search index, run a hybrid search from the shell, check health, or diagnose why search returns nothing/stale/errors. Invoke when the user says things like "sync my claude history", "reindex clync", "clync search for …", "is clync healthy / why is search broken", "clync doctor". This is the OPERATIONS skill (deliberate CLI actions); the separate `clync` skill is the one that auto-answers "what did I discuss before" via the MCP tools.
allowed-tools: Bash
---

clync syncs the user's claude.ai history into a local SQLite DB and serves a
hybrid semantic+lexical search over a **contained** Postgres/pgvector cluster.
This skill runs and troubleshoots those operations from the shell.

## The one gotcha that breaks everything

Search + indexing live in the optional **`search` extra** (BGE-M3, psycopg).
**Every index/search command MUST run through it:**

```sh
uv run --extra search --project ~/code/clync python ~/code/clync/clync.py <cmd>
```

If the installed `clync` CLI wrapper (from `clync setup`) is on PATH, `clync <cmd>`
works too — it already targets the project. Plain `python clync.py search` WITHOUT
`--extra search` will fail loud with "search extra not installed". That is correct
behavior, not a bug — add the extra.

The search backend is clync's own PG17 cluster on port **54329** (data under
`~/.local/share/clync/pg`), isolated from any system Postgres. It is provisioned
by `clync setup` / first `clync index`. It uses the PG17 binaries at
`/opt/homebrew/opt/postgresql@17/bin` (override with `$CLYNC_PG_BIN`).

## Operations

| Goal | Command | Notes |
|---|---|---|
| Sync new/updated conversations | `clync sync --profile <ChromeProfile>` | Auto-reindexes after. Needs the Chrome profile (or `$CLYNC_PROFILE`). `--no-index` to skip, `--full` to refetch all. |
| Rebuild the index only | `uv run --extra search … clync.py index [--full]` | No network. `--full` re-embeds everything; default is incremental by per-unit content signature. |
| Search from the shell | `uv run --extra search … clync.py search "<natural query>" [--limit N] [--lang en\|ja\|zh]` | Natural language, not FTS. Returns best chunk per conversation with a fused score. |
| Health check | `uv run --extra search … clync.py doctor` | Reports DB, search deps, cluster running, indexed chunk count. Exit 1 + a PROBLEMS list if anything's off. |
| Scheduler status / last sync | `clync status` | Shows last successful sync + launchd state + recent scheduled-run log. |
| First-time setup / teardown | `clync setup --profile <P>` · `clync unsetup` | setup wires CLI+MCP+scheduler+both skills+cluster; unsetup reverses it, keeping the DB + cluster data. |

## Troubleshooting (diagnose before acting)

1. **Always start with `clync doctor`** — it distinguishes the failure modes below as data.
2. **"search extra not installed"** → run with `uv run --extra search` (see gotcha), or `uv sync --extra search` in the repo.
3. **Cluster not running / "PG17 not found"** → `clync index` provisions+starts it; if PG17 is missing, `brew install postgresql@17 pgvector`.
4. **Search returns nothing for a topic the user knows they discussed** → it may be a **hollow conversation**: some old (pre-2024) threads have message rows but empty bodies because claude.ai's API no longer returns their text. clync mirrors the API faithfully; there is nothing to index, so content search cannot reach them. Confirm with:
   ```sh
   sqlite3 ~/.local/share/clync/history.db \
     "SELECT c.name, COALESCE(SUM(LENGTH(m.text)),0) chars FROM conversations c \
      LEFT JOIN messages m ON m.conversation_uuid=c.uuid GROUP BY c.uuid HAVING chars=0;"
   ```
   Rows returned = unsearchable-by-content conversations. This is expected, not a defect.
5. **MCP `search_history` in Claude Code returns old FTS-style results / stale tool description** → the MCP server is a long-lived subprocess that loaded the code at session start. **Restart the Claude Code session** to pick up the current code (this also reaps stale `mcp_server.py` processes). The CLI always runs current code.
6. **Stale results after new chats** → the daily scheduled sync may not have run (Mac asleep). `clync sync --profile <P>` now, or check `clync status`.

Fail-loud everywhere: auth/HTTP/cluster/schema errors raise and exit non-zero — never assume a silent success. Cite the conversation name when reporting a search hit; if search is empty, say so rather than inventing history.
