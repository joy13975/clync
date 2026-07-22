# clync

Headless sync of **claude.ai** conversation history *and* local **Claude Code**
sessions into a contained Postgres/pgvector store, hybrid-searchable over MCP.
macOS + Chrome only.

## How it works

1. **Auth** — reads your claude.ai cookies (`sessionKey` + `cf_clearance`) *live*
   from a named Chrome profile's cookie store each run (decrypted with the
   Chrome Safe Storage key from your login keychain). Nothing is stored; the
   cookies are as fresh as your browser's last claude.ai activity.
2. **Fetch** — calls claude.ai's private API through `curl_cffi` with Chrome TLS
   impersonation, which is what clears Cloudflare (a plain HTTP client gets a 403
   challenge). Requests are paced; transient 403s are retried with backoff.
3. **Ingest local Claude Code sessions** — cleans + extracts metadata from
   `~/.claude/projects/**/*.jsonl` (no network; runs even if claude.ai cookies
   are dead). See [Claude Code sessions](#claude-code-sessions) below.
4. **Store** — upserts both sources as unified `units` + `messages` (including
   **text-attachment content**) into clync's own contained Postgres 17 +
   pgvector cluster — the single source of truth *and* the search index.
   Incremental by `updated_at` (claude.ai) / file mtime (Claude Code).
5. **Query** — `mcp_server.py` exposes `search_history` (one faceted hybrid
   search across both sources) and `get_conversation` to Claude Code over stdio.

**Fail-loud:** any auth/HTTP/schema error raises and exits non-zero. It never
silently serves stale data. The scheduled run additionally fires a macOS
modal+sound alert on failure. (claude.ai auth failures don't block the local
Claude Code ingest — see the `sync` command below.)

Design rationale for the store + Claude Code ingest:
[docs/adr/0003](docs/adr/0003-postgres-only-and-cc-ingest.md).

## Setup

One command wires up everything (deps, `clync` CLI on your PATH, MCP
registration, daily launchd sync, and both Claude Code skills — `clync`
history-search + `clync-ops` operations) — all defined in this repo,
symlinked/registered out:

```sh
uv run python clync.py setup --profile "Work"   # or export CLYNC_PROFILE; --at HH:MM for time
uv run python clync.py doctor                      # verify: store / deps / launchd / MCP
```

Search is core, not an optional extra — `uv sync` (which `setup` runs for you)
pulls in the BGE-M3 embedder + Postgres client along with everything else.

`clync unsetup` removes every external artifact (symlinks, launchd job, MCP
registration) and leaves the local store intact. Then open a **new** Claude Code
session to load the MCP tools + skills.

## Commands

| Command | Purpose |
|---|---|
| `sync [--full] [--no-files] [--no-index]` | sync **both** sources: claude.ai (network/cookies, fails loud), **then** local Claude Code ingest (no network — runs even if cookies are dead), **then** index. `--full` re-fetches/re-parses everything; `--no-files` skips image downloads; `--no-index` skips the post-sync hybrid-search index |
| `sync-app [--full] [--no-files] [--no-index]` | sync **only** claude.ai chats (network/cookies), then index |
| `sync-cc [--full] [--no-index]` | ingest **only** local Claude Code sessions (no network/cookies), then index |
| `index [--full]` | (re)build the hybrid-search index from the store |
| `scheduled` | launchd entry point: sync (both sources) + loud fail/late notification |
| `search [<q>] [--source all\|claude_ai\|claude_code] [--project P] [--model M] [--repo R] [--worktree W] [--branch B] [--session S] [--since DATE] [--until DATE] [--sort relevance\|recency] [--limit N] [--lang en\|ja\|zh]` | faceted hybrid semantic + lexical search over both sources; empty query browses by recency |
| `list [--limit N] [--source all\|claude_ai\|claude_code]` | most recently updated units |
| `whoami` | resolved claude.ai account + org |
| `doctor` | health check: store, deps, launchd job, MCP registration, search index |
| `status [--tail N]` | last successful sync, launchd state, recent scheduled-run log |
| `install [--at HH:MM]` / `uninstall` | manage the daily launchd job |

## Claude Code sessions

The second content source: local Claude Code transcripts under
`~/.claude/projects/**/*.jsonl` (root overridable via `$CLYNC_CC_ROOT`).

- **Ingested:** only **interactive** sessions — those whose event `entrypoint`
  field is `"cli"`. Programmatic batch runs (`sdk-cli`, `sdk-py`) are excluded.
- **Excluded:** subagent transcripts (`<sessionId>/subagents/agent-*.jsonl`) —
  the parent session's `Agent` tool call already captures the task + its
  un-truncated result.
- **Metadata** comes from authoritative event fields, never the (lossy)
  project-directory name: `cwd` (raw), `repo` / `worktree` (derived from `cwd`),
  `git_branch`, `cc_version`, `entrypoint`, and a `title` (custom title → AI
  title → first prompt).
- **Cleaning:** harness/meta noise (mode, permission-mode, file-history
  snapshots, etc.) is dropped; user prompts, assistant text, and non-empty
  thinking are kept verbatim; tool calls are compressed to one-liners and tool
  results are head-truncated (the `Agent` tool's result is kept in full, since
  it's a synthesis, not stdout).

Full design: [docs/adr/0003](docs/adr/0003-postgres-only-and-cc-ingest.md).

## Hybrid search

clync runs its **own** contained Postgres 17 + pgvector cluster — a private
`initdb` cluster under `~/.local/share/clync/pg`, listening on `localhost:54329`,
fully isolated from any system/shared Postgres. It is the single source of
truth *and* the search index for both sources. It embeds synced messages with
**BGE-M3** (dense `vector(1024)` + learned-sparse `sparsevec`) and searches with
dense cosine + sparse dot-product, fused by **RRF** (k=60), plus a small typed-
metadata boost (recency half-life, query/chunk language agreement) and facet
filters (source, project, model, repo, worktree, branch, session, since/until).
There is no cross-encoder reranker — see the design rationale below.

```sh
clync setup --profile "..."   # provisions the cluster + builds the initial index
# — or, if already set up —
clync index [--full]          # (re)build the index incrementally (or fully)
clync search "boto3 pagination" --repo clync --lang en
```

`clync sync` automatically reindexes changed units afterward (pass
`--no-index` to skip). `clync doctor` reports cluster + index health.
`clync unsetup` stops the cluster but leaves its data on disk.

Design rationale: [docs/adr/0001](docs/adr/0001-drop-cross-encoder-reranker.md)
(why there's no reranker), [docs/adr/0002](docs/adr/0002-embedder-choice.md)
(why BGE-M3 over Qwen3-Embedding), and
[docs/adr/0003](docs/adr/0003-postgres-only-and-cc-ingest.md) (why Postgres-only
+ the unified schema across both sources).

A `clync` wrapper script sits in the repo; symlink it onto your PATH for global use:

```sh
ln -s ~/code/clync/clync ~/.local/bin/clync   # then: clync search "…", clync sync, clync status
```

## Config

| Setting | Source | Default |
|---|---|---|
| Chrome profile | `--profile` / `$CLYNC_PROFILE` | *(required — fails loud if unset)* |
| Target org | `--org <name-or-uuid>` | first (active) org in the account |
| Runtime data home (store cluster, files, log) | `$CLYNC_DATA_HOME` | `~/.local/share/clync` |
| Claude Code sessions root | `$CLYNC_CC_ROOT` | `~/.claude/projects` |
| Postgres binaries | `$CLYNC_PG_BIN` | `/opt/homebrew/opt/postgresql@17/bin` |
| Postgres port | `$CLYNC_PG_PORT` | `54329` |
| Postgres database name | `$CLYNC_PG_DB` | `clync` |

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
- **Claude Code sessions:**
  - Only sessions with event `entrypoint == "cli"` are ingested — programmatic
    `sdk-cli`/`sdk-py` batch runs are excluded, not just deprioritized.
  - **Subagent transcripts are out of scope** — `subagents/agent-*.jsonl` files
    are never read. Only the parent session's `Agent` tool call (task) and its
    result are indexed.
- **Auth cookies are per Chrome profile** — a browser-level cookie jar, one
  claude.ai login per profile. The tool is pointed at a profile because that is
  where the login cookie physically lives; the org is selected after
  authenticating (`--org` to pick a non-default one).
