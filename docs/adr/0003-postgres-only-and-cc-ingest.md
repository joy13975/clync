# ADR 0003 — Postgres-only store + Claude Code session ingest

- **Status:** Accepted
- **Date:** 2026-07-22
- **Scope:** clync's storage substrate and its second content source. Supersedes
  the "search is an optional extra over a SQLite source of truth" stance implicit
  in [ADR 0001](0001-drop-cross-encoder-reranker.md) / [ADR 0002](0002-embedder-choice.md)
  and the original README.

## Context

clync began as: **SQLite is the single source of truth**; a *contained* Postgres
17 + pgvector cluster holds a *derived, rebuildable* BGE-M3 index; the whole
search stack (torch/FlagEmbedding/psycopg) is an **optional `search` extra** so
plain sync + MCP reads stay dependency-light. Two things changed:

1. **Search is now mandatory core, not an optional extra.** The value of clync
   *is* hybrid search over your history; a clync that can't search is not a
   product we ship. The two-store split (SQLite raw + Postgres index) existed to
   preserve a clusterless, ML-dep-free base mode. That mode is no longer a goal,
   so the split now costs more than it buys: a SQLite→Postgres load step, two
   schemas to keep aligned, and a "SSOT is SQLite but search happens in Postgres"
   split-brain that confuses every reader.

2. **A second content source: local Claude Code (`cc`) sessions.** The
   `~/.claude/projects/**/*.jsonl` transcripts are the other half of the user's
   working context. Unifying them with claude.ai chat history under one searchable
   surface is the goal. This forces a metadata model that spans two sources with
   different, source-specific facets — which is far cleaner to express once, in
   one store, than to bolt onto a two-store design.

## Decision

### 1. Postgres-only. Drop SQLite and its FTS5 table.

The contained PG17+pgvector cluster becomes **both** the source of truth **and**
the index. The `search` deps (`FlagEmbedding`, `psycopg`, `transformers<5`) fold
into core `dependencies`; the optional-extra and the `search.available()` gating
premise are removed. `messages_fts` is deleted — it was populated but **never
queried** (BGE-M3's learned-sparse vector already is the lexical signal; exact
facet matching is served by typed btree/trigram indexes, not full-text).

No data migration is needed: the claude.ai raw store is re-derivable from the API
(`clync sync --full` repopulates), and the cc store is re-derivable from the local
JSONL. The switch is a schema replacement + a full re-sync, not an ETL.

### 2. One unified, faceted schema (all in Postgres).

A single retrieval SSOT — **`units` + `messages` + `chunks`** — spans all sources.

- **`units`** — one row per retrievable unit. `kind ∈ {chat, project_doc,
  cc_session}`, `source ∈ {claude_ai, claude_code}`. Common columns (title,
  summary, created/updated, lang) + **facets as typed, indexed columns** (not
  buried in `raw`): claude.ai — `org_uuid, project_uuid, project_name, model`;
  cc — `repo, cwd, worktree, git_branch, cc_version, entrypoint`.
- **`messages`** — per-turn cleaned content, `msg_id` = source event uuid;
  identity is **per-unit** (`PRIMARY KEY (unit_id, msg_id)`). A cc resume replays
  the original session's uuids into the resumed session's file — each unit stores
  its complete transcript (duplicate storage accepted), so `get_conversation`
  never starts mid-stream and `units.msg_count` always equals the rows actually
  stored (it is written from a post-insert `COUNT(*)`, never the parse length).
- **`chunks`** — BGE-M3 dense+sparse embeddings, with the filterable facets
  **denormalized onto the chunk row** so faceted filtering happens *inside* the
  vector CTE with no join (exactly as `conv_name`/`lang`/`updated_at` already were).
- `projects` (claude.ai project dimension) and `files` (attachment/image metadata)
  remain as auxiliary tables; `cc_sync_state` (file → mtime/size) drives cc
  incrementality **and deletion reconciliation** (a transcript deleted from disk
  purges its unit/messages/watermark, mirroring the claude.ai upstream-deletion
  purge); `indexed_units` carries the per-unit content-signature watermark.

### 3. Claude Code metadata comes from authoritative event fields, never the folder name.

The project directory name is a **lossy** encoding (`/`, `.`, literal `-` all
collapse to `-`). Every `user`/`assistant` event instead carries the truth
directly — verified **0 null `cwd` across 108,922 events**:

| Facet | Rule | Example |
|---|---|---|
| `cwd` | authoritative, stored raw | `…/example-repo/.claude/worktrees/some-branch/sub/dir` |
| `repo` | if `/.claude/worktrees/` in cwd → basename *before* it; else git-toplevel basename | `example-repo` |
| `worktree` | segment *after* `/.claude/worktrees/`; else null | `some-branch` |
| `git_branch`, `cc_version`, `entrypoint` | authoritative from event | `worktree/…`, `2.1.205`, `cli` |
| `title` | **last** `custom-title` → **last** `ai-title` → first typed prompt (truncated). Title events are updates emitted repeatedly (last wins); each key is read only from its own typed event — a bare `title` key on other events (e.g. `frame-link`, an artifact page title) is never a session title | — |
| `created/updated` | min/max event `timestamp` | — |

### 4. Cleaning: a per-turn classifier (tool noise is ~80% of bytes).

Measured on a large session: tool_result + tool_use inputs ≈ 80% of content bytes;
human + assistant text ≈ 20%. So:

- **Drop:** harness/meta event types (`ai-title`, `custom-title`, `mode`,
  `permission-mode`, `file-history-snapshot`, `worktree-state`, `queue-operation`,
  `pr-link`, `bridge-session`, `agent-name`, `last-prompt`), `system`, `isMeta`,
  `attachment`, empty/redacted `thinking`.
- **Keep verbatim:** typed user prompts, assistant `text`, non-empty `thinking`.
- **Compress tool I/O:** `tool_use` → `[Bash] <command>` / `[Edit] <file_path>`
  (name + primary arg); `tool_result` → `[result: N bytes, exit C]` + head-truncated
  (~800 chars). **Exception:** the `Agent` tool's result is kept un-truncated (it
  is a synthesis, not stdout).
- **Two-tier text:** `messages.text` stores the richer *display* rendering; the
  chunk loader builds a *tighter* embed string (tool bodies dropped) so the vector
  index is not dominated by stdout.

### 5. Scope: interactive sessions only; subagents excluded.

- **`entrypoint == "cli"` only.** `sdk-cli` / `sdk-py` are programmatic batch runs
  (a single batch harness accounted for 2,383 such files), not conversations. `entrypoint`
  is a typed event field — a principled filter with **no hardcoded personal paths**.
  It is stored as a facet, so batch runs are filterable-*in* later, not discarded.
- **Subagents out of scope.** `subagents/agent-*.jsonl` are not ingested; the
  parent session's `Agent` tool_use (the task) + its un-truncated tool_result (the
  end result) already capture what is needed.

### 6. One faceted MCP endpoint; `list_conversations` dropped.

`search_history(query="", source, project, model, repo, worktree, branch, session,
since, until, sort, limit)` — filters layer on top of the hybrid dense+sparse RRF
ranking. An **empty query degrades to a metadata browse** sorted by recency (this
absorbs `list_conversations`, which is removed). A **facet/source mismatch fails
loud** (e.g. `repo=` with `source="claude_ai"`). `get_conversation` becomes
source-aware (reads `messages` by `unit_id`).

### 7. One `clync sync`; no new skill.

`clync sync` runs claude.ai (network/cookies, fails loud) **then** cc (local, no
network — runs even if cookies are dead), then indexes once. One launchd job. The
existing `clync` and `clync-ops` skills are updated in place to describe the cc
source + new filters; **no third skill**.

## Consequences

- **Simpler mental model, one store.** No SQLite↔Postgres load step, no
  split-brain. The cost is that any clync use now requires the cluster + ML deps —
  accepted, because search is core.
- **Unified search with zero query-side special-casing across sources** — a cc
  session, a claude.ai chat, and a project doc are all `units` feeding `chunks`.
- **The cluster is now durability-critical**, not disposable. Mitigated by
  full-re-derivability from both sources (`sync --full`), so a lost cluster is a
  re-sync, not data loss.
- **Initial index cost is larger** (the cc corpus adds many chunks); incremental
  (content-signature watermark + `cc_sync_state` mtime gate) keeps steady-state
  runs cheap.
