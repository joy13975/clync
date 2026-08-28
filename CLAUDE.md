# clync — guide for Claude Code

clync syncs the user's **claude.ai** conversation history *and* local **Claude
Code** sessions into a contained Postgres/pgvector store and exposes it to
Claude Code over MCP. macOS + Chrome only.

## Docs index

This file is an index — the actual documentation lives in these:

| Doc | What's in it |
|---|---|
| [README.md](README.md) | **Start here.** What it is, one-command setup, all CLI commands, config, architecture, and limitations. |
| [skill/SKILL.md](skill/SKILL.md) | The auto-firing Claude Code skill — when/how to query the user's history (both sources) via MCP. |
| [skill-ops/SKILL.md](skill-ops/SKILL.md) | The `clync-ops` skill — invoked to run/troubleshoot sync, index, search, doctor from the shell. |
| `clync.py` (module docstring + `--help`) | Sync engine, cookie/Cloudflare handling, the contained Postgres store, CLI. |
| `cc.py` (module docstring) | Local Claude Code session ingest: JSONL parsing, cleaning, metadata extraction. |
| `mcp_server.py` (docstring) | The MCP tools: `search_history` (default — distilled positions AND raw transcripts in one call), the `search_insights` / `search_transcripts` drill-downs, and `get_conversation`. |
| `search.py` (module docstring) | Hybrid search: clync's own contained PG17+pgvector cluster, BGE-M3 indexing, RRF-fused dense+sparse query. |
| `dream.py` (module docstring) | **The Dream layer:** distills grounded, stance-tagged insights + per-topic digests out of the raw transcripts using headless `claude -p` (no API billing). Two separate run modes: incremental nightly, explicit bulk backfill. |
| [LICENSE](LICENSE) | MIT. Nothing in this repo may contain or derive from real conversation content — see the rule in `experiments/reranker_ablation/.gitignore`. |
| [docs/adr/](docs/adr/) | Architecture decisions: why no cross-encoder reranker (0001), why BGE-M3 (0002), why Postgres-only + Claude Code ingest (0003), the Dream knowledge-distillation layer (0004). |

## One-command setup

```sh
clync setup --profile <ChromeProfile>     # or export CLYNC_PROFILE
clync setup                               # local-only: Claude Code + Codex, no claude.ai
```

Installs deps, the `clync` CLI wrapper, the MCP registration, the daily launchd
sync, and both skills (`clync` history-search + `clync-ops` operations) — all
defined in this repo, symlinked/registered out. `clync unsetup` reverses it
(keeps the store's cluster data).

## Orientation for editing here

- **SSOT:** `clync.py` owns cookies + API client + the contained Postgres
  store (single source of truth AND search index, ADR 0003); `cc.py` owns
  local Claude Code session parsing/cleaning (no DB); `search.py` owns the
  vector index + hybrid query; `dream.py` owns the derived knowledge layer
  (its own tables, prompts, gates, and the `format_recall` renderer both
  surfaces print); `mcp_server.py` imports from `clync.py`/`search.py`/`dream.py`.
  Don't duplicate store/query logic across these. Each derived layer ensures its
  OWN schema (`search.ensure_index_schema`, `dream.ensure_dream_schema`) — there
  is no migration mechanism, so a changed shape is dropped and rebuilt.
- **The Dream layer must never read its own output.** `source='all'` means raw
  only; the dig rejects any citation to a non-raw unit; the worker runs with
  `--no-session-persistence` so it writes no transcript for `sync-cc` to ingest.
  Breaking any of those three makes the layer feed on itself.
- **Config, not hardcoded:** `$CLYNC_PROFILE` / `--profile`, `$CLYNC_DATA_HOME`,
  `$CLYNC_CC_ROOT`, `$CLYNC_PG_BIN`/`$CLYNC_PG_PORT`/`$CLYNC_PG_DB`, `--org`.
  Nothing personal is baked into source.
- **Fail loud:** auth/HTTP/schema errors raise and exit non-zero; the scheduled
  run also fires a macOS alert. Never degrade silently to stale data. Exception
  handlers re-raise (see the repo's coding rules).
- **Runtime state** (Postgres cluster, downloaded images, scheduled log) lives
  in `~/.local/share/clync/` — outside the repo, never committed.
