# clync — guide for Claude Code

clync syncs AI conversation history into a contained Postgres/pgvector store and
exposes it over MCP. See README's source table for coverage. macOS + Chrome only.

## Docs index

This file is an index — the actual documentation lives in these:

| Doc | What's in it |
|---|---|
| [README.md](README.md) | **Start here.** Imported sources, quick start, MCP clients, usage, and resource requirements. |
| [skill/SKILL.md](skill/SKILL.md) | The auto-firing Claude Code skill — when/how to query the user's history (all raw sources and the derived dream layer) via MCP. |
| [skill-ops/SKILL.md](skill-ops/SKILL.md) | The `clync-ops` skill — invoked to run/troubleshoot sync, index, search, doctor from the shell. |
| `clync.py` (module docstring + `--help`) | Sync engine, cookie/Cloudflare handling, the contained Postgres store, CLI. |
| `cc.py` (module docstring) | Local Claude Code session ingest: JSONL parsing, cleaning, metadata extraction. |
| `opencode_source.py` | OpenCode v2 API ingest: root-only scope, no machine messages or tool-result bodies, dirty-session retry. |
| `mcp_server.py` (docstring) | The MCP tools: `search_history` (default — distilled positions AND raw transcripts in one call), the `search_insights` / `search_transcripts` drill-downs, and `get_conversation`. |
| `search.py` (module docstring) | Hybrid search: clync's own contained PG17+pgvector cluster, BGE-M3 indexing, HNSW-indexed dense+sparse RRF query (ADR 0005). |
| `clync_cipher.py`, `share-client/`, `cloud/` | Opt-in serverless `clync-share` on AWS: lite cipher, lightweight IAM-authenticated MCP relay (`clync connect` / `clync-share`), per-owner ECR container Lambda, and CloudTrail tamper alerting (ADR 0006). |
| `live_search.py`, `live_files.py`, `live_external.py` | Independent live finder, candidate-record reads, and native Grok/OpenCode readers; shared contracts in `live_types.py`. |
| `dream.py` (module docstring) | **The Dream layer:** distills grounded, stance-tagged insights + per-topic digests out of the raw transcripts using headless `claude -p`. Two separate run modes: incremental nightly, explicit bulk backfill. |
| [docs/adr/](docs/adr/) | Architecture decisions: why no cross-encoder reranker (0001), why BGE-M3 (0002), why Postgres-only + Claude Code ingest (0003), the Dream knowledge-distillation layer (0004), HNSW hybrid search (0005), serverless `clync-share` (0006). |

## One-command setup

```sh
clync setup --claude-profile <ClaudeProfile> --chatgpt-profile <ChatGPTProfile>
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
- **Live search is independent of the archive:** no database provisioning or model
  imports in startup or live requests; archive setup is lazy. Preserve both source
  and matching coverage. See README's live-search contract.
- **The Dream layer must never read its own output.** `source='all'` means raw
  only; the dig rejects any citation to a non-raw unit; the worker runs with
  `--no-session-persistence` so it writes no transcript for `sync-cc` to ingest.
  Breaking any of those three makes the layer feed on itself.
- **Config, not hardcoded:** `$CLYNC_CLAUDE_PROFILE` / `--claude-profile`,
  `$CLYNC_CHATGPT_PROFILE` / `--chatgpt-profile`, `$CLYNC_DATA_HOME`,
  `$CLYNC_CC_ROOT`, `$CLYNC_OPENCODE_CONFIG`, `$CLYNC_PG_BIN`/`$CLYNC_PG_PORT`/`$CLYNC_PG_DB`,
  `$CLYNC_RESOURCE_FRACTION`, `$CLYNC_EMBED_DEVICE`, `--org`.
  Nothing personal is baked into source.
- **A sync shares the machine.** `clync.limit_resources()` caps every thread pool
  the embedder's stack reads, and it must be called BEFORE torch is imported — the
  pools are sized at load time. README "Staying out of the way" has the knobs and
  why the GPU has no equivalent.
- **Fail loud:** auth/HTTP/schema errors raise and exit non-zero; the scheduled
  run also fires a macOS alert. Never degrade silently to stale data. Exception
  handlers re-raise (see the repo's coding rules).
- **Runtime state** (Postgres cluster, downloaded images, scheduled log) lives
  in `~/.local/share/clync/` — outside the repo, never committed.
