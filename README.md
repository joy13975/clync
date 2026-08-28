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
5. **Dream** — a third layer, derived rather than synced: `dream.py` distills
   topic-scoped knowledge (grounded, stance-tagged insights + structured
   digests) out of the raw transcripts. See [Dream layer](#dream-layer) below.
6. **Query** — `mcp_server.py` exposes `search_history` (the default: one hybrid
   search over BOTH the distilled layer and the raw sources, returned as labelled
   sections), the two drill-downs `search_insights` / `search_transcripts`, and
   `get_conversation` to Claude Code over stdio.

**Fail-loud:** any auth/HTTP/schema error raises and exits non-zero. It never
silently serves stale data. The scheduled run additionally fires a macOS
modal+sound alert on failure. (claude.ai auth failures don't block the local
Claude Code ingest — see the `sync` command below.)

Design rationale for the store + Claude Code ingest:
[docs/adr/0003](docs/adr/0003-postgres-only-and-cc-ingest.md).

## Setup

**Prerequisites** (macOS + Chrome only — see [Limitations](#limitations)):

```sh
brew install uv postgresql@17 pgvector
```

clync provisions its **own** contained PG17 cluster from those binaries under
`~/.local/share/clync/`; it never touches an existing Postgres install, its data,
or its port. Set `$CLYNC_PG_BIN` if your PG17 lives outside Homebrew.

**Install.** One command wires up everything (deps, `clync` CLI on your PATH, MCP
registration, daily launchd sync, and both Claude Code skills — `clync`
history-search + `clync-ops` operations) — all defined in this repo,
symlinked/registered out:

```sh
git clone https://github.com/joy13975/clync.git ~/code/clync && cd ~/code/clync
uv run python clync.py setup --profile "Person 1"   # or export CLYNC_PROFILE; --at HH:MM for time
uv run python clync.py doctor                      # verify: store / deps / launchd / MCP
```

`--profile` names the **Chrome profile** logged in to claude.ai, by its display
name; a wrong name fails loud and lists the available ones.

**Don't use claude.ai in Chrome?** Omit `--profile`. Setup then wires a
**local-only** install — local Claude Code + Codex sessions, fully searchable —
and the nightly job skips the network legs rather than failing them. A *lost*
`$CLYNC_PROFILE` is still a loud error; local-only is recorded explicitly as
`$CLYNC_LOCAL_ONLY` in the launchd job, so the two cases never blur.

**Or let Claude Code do it.** Drop [`skill/SKILL.md`](skill/SKILL.md) into
`~/.claude/skills/clync/SKILL.md` and say `/clync install` — the skill walks the
prerequisites, clone, and setup, then supersedes its own bootstrap copy with a
symlink to the repo. One file to share; no other bootstrap needed.

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
| `search [<q>] [--source all\|claude_ai\|claude_code\|dream] [--project P] [--model M] [--repo R] [--worktree W] [--branch B] [--session S] [--since DATE] [--until DATE] [--sort relevance\|recency] [--limit N] [--lang en\|ja\|zh]` | faceted hybrid semantic + lexical search; `all` means the two RAW sources only — distilled dream units are reachable only via `--source dream` |
| `list [--limit N] [--source all\|claude_ai\|claude_code\|dream]` | most recently updated units (`all` = raw sources only, same rule as `search`) |
| `whoami` | resolved claude.ai account + org |
| `doctor` | health check: store, deps, launchd job, MCP registration, search index, dream layer |
| `status [--tail N]` | last successful sync, launchd state, recent scheduled-run log |
| `install [--at HH:MM]` / `uninstall` | manage the daily launchd job |
| `dream topics [--seed] [--all]` | list dream topics (id, status, active insight count, last dig time); `--seed` inserts the built-in starter topics if absent, `--all` includes non-active topics |
| `dream run [--max-calls N]` | the incremental (nightly-shaped) dream pass — see [Dream layer](#dream-layer) |
| `dream backfill [--topic T] [--max-calls N]` | the explicit bulk dream pass — see [Dream layer](#dream-layer) |
| `dream recall "<query>" [--topic T] [--limit N] [--as-of ISO] [--evidence]` | dream-first tiered recall from the shell: TOPIC / COVERAGE / DIGEST / INSIGHTS / RAW TRANSCRIPTS |
| `dream status` | watermark, per-topic table, model-usage totals by stage, failed queue count |

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

## Dream layer

A third content layer (design: [docs/adr/0004](docs/adr/0004-dream-layer.md)),
built on top of the raw `units`/`messages` store. Where `search`/`search_history`
answer *"where did I discuss X?"*, the Dream layer answers *"what do I actually
think about X?"* by distilling topic-scoped, cited **insight** atoms out of the
raw transcripts, and rendering a per-topic **digest** from them on read.

- **Inference has no API billing.** All model calls go through the local
  headless `claude -p` CLI under the existing subscription OAuth — no
  `ANTHROPIC_API_KEY`, no API quota. This consumes the subscription's rate
  limits; if a run gets throttled it stops cleanly, keeps its state, and warns
  (not a failure) rather than retrying — the next `dream run`/`backfill`
  resumes where it left off.
- **Every insight is grounded.** It must cite real `(unit_id, msg_id)` pairs
  whose quoted text is found verbatim in the stored message, or it is
  rejected — a mechanical DB check, not the model's self-report.
- **Every insight carries a stance** — who held the position:
  `user_asserted` / `user_endorsed` / `user_rejected` / `co_derived` /
  `claude_proposed`. A position the user rejected stays retrievable as
  rejected; a Claude suggestion is never silently promoted into "what the user
  thinks".
- **Nothing is deleted.** A changed position closes the old insight's validity
  window and links the supersession, which is what `dream recall --as-of`
  reads — positions held at a given date, not just now.
- **Questions are signal.** Every query put to the recall surface is logged
  (`dream_queries`). One that came back with no insight becomes a probe query on its
  topic, so the next dig hunts it in the raw transcripts; one that matched no topic at
  all is reported as a new-topic candidate. The composed answer is never stored as an
  insight — that would make the layer read its own output.
- **`clync migrate` snapshots before it drops.** Insights cost model calls, so a
  rebuild copies the dream tables to dated, constraint-free `*_bakYYYYMMDD` tables
  (garbage-collected after 183 days) and prints what it preserved.
- **Non-circular by construction.** `source=all`/`search_transcripts` still means
  raw-only (`claude_ai` + `claude_code`); dreams are reachable only via
  `source=dream` or the recall surfaces. The dig cannot retrieve its own output,
  and the worker writes no session transcript, so the layer can never feed on
  itself.
- **Coverage gaps are stated loudly, never hidden** — an un-dug or stale topic
  says so explicitly in its output rather than presenting thin coverage as
  complete.

Two separate passes, with cost profiles two orders of magnitude apart:

- **`dream run [--max-calls N]`** — the incremental, nightly-shaped pass, gated
  by triage: only units changed since the watermark are looked at (batched 15
  per triage call). Digs are **batched across nights**: a triaged unit is queued against
  its topic, and that topic is dug once 3 units have accumulated or its oldest
  has waited a week. This matters because nightly cost scales with *topics dug*,
  not units changed — measured, digging every flagged topic immediately cost 14
  calls for one ordinary day. Nothing is lost by waiting (the queue is a table,
  and `dream run`/`dream status` both report the backlog). So: no changed units
  costs 0 calls, changed units with nothing ripe costs 1, and a ripe topic costs
  4-5. This runs automatically inside the daily `clync scheduled` launchd job,
  immediately after sync+index. The very first `dream run` deliberately does
  nothing but initialize the watermark — it tells you to use `dream backfill` to
  mine existing history.
- **`dream backfill [--topic T] [--max-calls N]`** — the bulk pass. Explicit
  only, never scheduled. This is what mines the existing history and is where
  the real cost lives: on the order of 100-200 calls for a full first pass
  across all topics, against a **`--max-calls` default of 30**, so the plain
  command deliberately stops early and prints `stopped_early: max-calls
  reached`. That is not a failure — repeat it (or pass a larger `--max-calls`)
  until it stops reporting `stopped_early`. Resumable by construction: progress
  lives in the evidence table, so a capped run picks up where it stopped.

`dream recall "<query>" [--topic T] [--limit N] [--as-of ISO] [--evidence]`
does dream-first tiered retrieval from the shell: TOPIC / COVERAGE / DIGEST /
INSIGHTS / RAW TRANSCRIPTS. `--as-of` reads the bi-temporal history (positions
held at that date, not now); `--evidence` prints the quoted source text behind
each insight. The same retrieval is exposed to Claude Code as the `search_history`
MCP tool (see below), which searches the distilled layer AND the raw transcripts in
one call so the caller never has to classify the question first; `search_insights`
and `search_transcripts` are the per-layer drill-downs.

`dream status` reports the watermark, a per-topic table, model-usage totals by
stage, and the failed-queue count; `clync doctor` includes a one-line dream
health summary.

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
- **Dream layer:** distillation quality depends on the topic charters — a
  charter that's too broad or too narrow shapes what gets triaged into a
  topic, same as any retrieval-driven system. `dream backfill` (mining
  existing history) is the expensive pass, on the order of 100-200 model
  calls for a full first pass across all topics; `dream run` (nightly) stays
  cheap by construction, but a topic added or a charter materially rewritten
  needs a fresh `backfill` to catch it up.

## License

MIT — see [LICENSE](LICENSE).

Your synced history never leaves your machine: it lives in clync's own contained
Postgres cluster under `~/.local/share/clync/`, outside this repo. Nothing in
this repository contains or is derived from any real conversation content — the
ablation experiments ship aggregate numbers only, and their query sets stay
local by the rule in `experiments/reranker_ablation/.gitignore`.
