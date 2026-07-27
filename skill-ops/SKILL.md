---
name: clync-ops
description: Operate and troubleshoot the clync sync + hybrid-search tool — run or debug a sync, (re)build the search index, run a hybrid search from the shell, check health, or diagnose why search returns nothing/stale/errors. Invoke when the user says things like "sync my claude history", "reindex clync", "clync search for …", "is clync healthy / why is search broken", "clync doctor". This is the OPERATIONS skill (deliberate CLI actions); the separate `clync` skill is the one that auto-answers "what did I discuss before" via the MCP tools.
allowed-tools: Bash
---

clync syncs the user's claude.ai chats **and** local Claude Code sessions into
one contained Postgres/pgvector cluster — the single source of truth *and* the
hybrid semantic+lexical search index. This skill runs and troubleshoots those
operations from the shell.

## Two sources, one sync

`clync sync` runs claude.ai (network/cookies — fails loud if the profile's
cookies are stale) **then** local Claude Code ingest (`~/.claude/projects/**/*.jsonl`,
no network — runs even if the claude.ai leg failed), **then** indexes once. To
ingest Claude Code sessions only, without touching claude.ai/network at all,
use `clync sync-cc [--full]`; to sync claude.ai only, use `clync sync-app`.

Search is **core** — the BGE-M3 embedder + Postgres client ship with the base
install (`uv sync`, no extras). There is no `--extra search` step anymore; a
plain `clync <cmd>` or `uv run --project ~/code/clync python ~/code/clync/clync.py <cmd>`
just works.

The search backend is clync's own PG17 cluster on port **54329** (data under
`~/.local/share/clync/pg`, overridable via `$CLYNC_DATA_HOME`), isolated from
any system Postgres. It is provisioned by `clync setup` / first `clync index`.
It uses the PG17 binaries at `/opt/homebrew/opt/postgresql@17/bin` (override
with `$CLYNC_PG_BIN`).

## Operations

| Goal | Command | Notes |
|---|---|---|
| Sync both sources | `clync sync --profile <ChromeProfile>` | claude.ai then local Claude Code ingest then index. Needs the Chrome profile (or `$CLYNC_PROFILE`) for the claude.ai leg only. `--no-index` to skip indexing, `--full` to refetch/re-parse all. |
| Sync claude.ai chats only | `clync sync-app --profile <ChromeProfile> [--full] [--no-index]` | claude.ai leg + index; no local Claude Code ingest. Fails loud on stale cookies. |
| Ingest local Claude Code sessions only | `clync sync-cc [--full] [--no-index]` | No network/cookies needed. Only `entrypoint=="cli"` (interactive) sessions are ingested; subagent transcripts are never read. |
| Rebuild the index only | `clync index [--full]` | No network. `--full` re-embeds everything; default is incremental by per-unit content signature. |
| Search from the shell | `clync search ["<natural query>"] [--source all\|claude_ai\|claude_code] [--repo R] [--worktree W] [--branch B] [--project P] [--model M] [--since/--until DATE] [--sort relevance\|recency] [--limit N] [--lang en\|ja\|zh]` | Natural language, not FTS; empty query browses by recency. Returns best chunk per unit with a fused score. A facet that doesn't apply to `--source` fails loud. |
| List recent units | `clync list [--limit N] [--source all\|claude_ai\|claude_code]` | Most recently updated units, either source. |
| Health check | `clync doctor` | Reports store (chat/doc/cc counts), search deps, cluster running, indexed chunk count. Exit 1 + a PROBLEMS list if anything's off. |
| Scheduler status / last sync | `clync status` | Shows last successful sync + launchd state + recent scheduled-run log. |
| First-time setup / teardown | `clync setup --profile <P>` · `clync unsetup` | setup wires CLI+MCP+scheduler+both skills+cluster; unsetup reverses it, keeping the store + cluster data. |
| List dream topics | `clync dream topics [--seed] [--all]` | `--seed` inserts the built-in starter topics if not already present; default shows only `active` topics, `--all` includes `proposed`/`retired` too. |
| Run the nightly (incremental) dream pass | `clync dream run [--max-calls N]` | Triage-gated: only units changed since the watermark are looked at. Digs batch ACROSS nights - a topic is dug once 3 units are pending or the oldest has waited 7 days - so a run that digs nothing is normal, not broken; it prints the deferred backlog. Runs automatically inside `clync scheduled`, right after sync+index — you normally don't invoke it by hand except to test/inspect. The very first run does nothing but initialize the watermark by design (see Troubleshooting). |
| Mine existing history into a topic | `clync dream backfill [--topic T] [--max-calls N]` | Explicit only, never scheduled — this is the expensive pass (order 100-200 calls for a full first pass across all topics). Resumable: a capped run picks up where it stopped. Also the right tool after adding a topic or rewriting a charter. |
| Dream-first recall from the shell | `clync dream recall "<query>" [--topic T] [--limit N] [--as-of ISO] [--evidence]` | Tiered output: TOPIC / COVERAGE / DIGEST / INSIGHTS / RAW TRANSCRIPTS. `--as-of` reads bi-temporal history (positions held then, not now); `--evidence` prints the quoted source text behind each insight. |
| Dream layer health | `clync dream status` | Watermark, per-topic table (active/superseded/contested insight counts, last dig time), model-usage totals by stage, failed-queue count. |

## Dream layer operations

Inference for the dream layer runs through the local `claude -p` CLI under the
existing subscription OAuth — no API key, no API billing. It consumes the
subscription's rate limits, same as any interactive session.

- **A `dream run` that digs nothing is the expected common case.** Cost scales
  with topics dug, so topics accumulate a backlog first. Both `dream run` and
  `dream status` print the pending count and the age of the oldest pending unit
  per topic - that backlog is how you tell "cheap tonight" from "stuck".
- **Reading `dream status`:** the `watermark` line is the last time
  `dream run` completed; a topic with `active=0` has produced nothing yet (its
  digest is rendered from those rows, so it is empty too); `contested` counts insights the falsification
  gate flagged as unresolved. The usage totals and failed-queue count come
  straight from `dream_queue`, so they reflect real spend, not an estimate.
- **`clync doctor` includes a `dream` line** — active topic count, active
  insight count, and failed queue item count. Treat it the same
  as the rest of doctor's output: a nonzero failed count is a signal to look
  closer, not necessarily a hard failure.

## Troubleshooting (diagnose before acting)

1. **Always start with `clync doctor`** — it distinguishes the failure modes below as data.
2. **Cluster not running / "PG17/pgvector not found"** → `clync sync` or `clync index` provisions+starts it; if PG17 is missing, `brew install postgresql@17 pgvector`.
3. **Search returns nothing for a topic the user knows they discussed** → for claude.ai, it may be a **hollow conversation**: some old (pre-2024) threads have message rows but empty bodies because claude.ai's API no longer returns their text. clync mirrors the API faithfully; there is nothing to index, so content search cannot reach them. Confirm with:
   ```sh
   psql -h localhost -p 54329 -U "$USER" -d clync -c \
     "SELECT title, msg_count FROM units WHERE source='claude_ai' AND msg_count > 0 \
      AND unit_id NOT IN (SELECT DISTINCT unit_id FROM messages WHERE length(text) > 0);"
   ```
   Rows returned = unsearchable-by-content conversations. This is expected, not a defect.
   For Claude Code, check `entrypoint` — only `"cli"` sessions are ingested; a
   session run via `sdk-cli`/`sdk-py`, or a subagent transcript, is out of
   scope by design (ADR 0003), not a bug.
4. **MCP `search_history` in Claude Code returns stale results / stale tool description** → the MCP server is a long-lived subprocess that loaded the code at session start. **Restart the Claude Code session** to pick up the current code (this also reaps stale `mcp_server.py` processes). The CLI always runs current code.
5. **Stale results after new chats or coding sessions** → the daily scheduled sync may not have run (Mac asleep). `clync sync --profile <P>` now, or check `clync status`.
6. **Dream run reports it hit a throttle / was cut short** → this is expected, not a defect — the subscription's rate limit was hit, the run stopped cleanly and recorded its state. Just re-run `clync dream run` (or `clync dream backfill`) later; it resumes from where it stopped rather than re-doing work.
7. **`clync dream status` shows failed queue items (or `clync doctor`'s `dream` line has a nonzero failed count)** → inspect with `clync dream status`; a failed item keeps its raw error, so read that before re-running. Re-running the same `dream run`/`backfill` retries the outstanding work.
8. **A topic has `active=0` in `clync dream topics`/`status`** → it has never been distilled. Run `clync dream backfill --topic <id>` to mine it — the nightly `dream run` only digs topics that triage flags as touched by *new* activity, it does not backfill history.
9. **First `clync dream run` appears to do nothing** → by design. On the very first run there's no watermark yet, so it only initializes the watermark and reports that history mining is `dream backfill`'s job, not the nightly pass's. Use `clync dream backfill` to mine existing history.

Fail-loud everywhere: auth/HTTP/cluster/schema errors raise and exit non-zero — never assume a silent success. Cite the conversation/session name when reporting a search hit; if search is empty, say so rather than inventing history.
