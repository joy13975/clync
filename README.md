# clync

**Your AI conversations, remembered across tools.**

Search conversations and coding sessions live or from a local archive. Recover a decision,
find the session behind a fix, or revisit your reasoning with supporting transcripts.

[Sources](#sources) · [Live session search](#live-session-search) · [Quick start](#quick-start) · [Connect](#connect-an-assistant) · [Usage](#usage) · [Dream](#dream)

## Sources

| Imported from | What is included |
| --- | --- |
| **claude.ai** | Conversations and project documents |
| **ChatGPT** | Conversations, including project threads |
| **Claude Code** | Local interactive CLI sessions |
| **Codex CLI** | Local interactive terminal sessions |
| **Pi** | Local primary-session tree history, including forks and pre-compaction messages |
| **OpenCode v2** | Local root sessions, including archives and forks (when configured) |
| **Artiport** | Static artifact text, canonical links, revisions, and copy provenance (when configured) |

> [!NOTE]
> **Grok Build history is not imported, but is live-searchable below.** Codex Desktop, headless runs, and
> separate subagent transcripts are also excluded. Connecting an MCP client
> does not add its history to clync.

## Live session search

Find current **Claude Code, Codex (CLI/Desktop), OpenCode v2, Pi, and Grok Build** sessions
without a sync, Postgres, embeddings, a content index, or a background watcher.
The live CLI needs Python 3.11+; file-backed search also needs `rg` on PATH:

```sh
python3 clync.py sessions search "authentcation cookie" --repo my-project
python3 clync.py sessions search "scheduler" --source claude_code,codex_cli --json
python3 clync.py sessions read '<reference>' --around '<anchor>'
```

Queries are lexical keywords, not semantic questions. Words are case-insensitive;
ASCII words of length five or more tolerate one insertion, deletion, substitution,
or adjacent transposition. Identifier separators/camel case and full-width ASCII
are folded; Japanese keywords and JSON-escaped text are supported.

For Claude Code, Codex and Pi, the fast default verifies candidate records with literal query
anchors and independently matches session titles. If it finds nothing, it broadens to
fuzzy candidates. **It does not enumerate every fuzzy-only match when anchored hits already
exist.** Use `--exhaustive` for that. OpenCode and Grok verify all potentially matching
passages, including fuzzy-only matches. `--exact` disables typo tolerance for every source.
Matching coverage is reported separately from source availability. Reads are bounded and
content-hashed; search again after a stale-anchor error.

Default content is primary-session conversation text, including structured answers
to questions. Claude Code SDK runs are excluded; Codex interface origin does not
exclude Desktop sessions. `--include-subagents` includes child transcripts;
`--include-tools` also searches reasoning and tool records with their own attribution.
Use `--sender user` for the user's own words, `--repo` for a directory component or
native project root name, and `--since` for file-modification/native session-update time.

`CLYNC_CC_ROOT`, `CLYNC_CODEX_ROOT`, and `CLYNC_GROK_ROOT` override native roots
(Grok's root contains `sessions/`). OpenCode uses the executable configuration described
below and requires the local service and `lsof`. Search verifies that the service PID
holds the configured native SQLite database open, then filters raw message/part candidates
in a read-only snapshot before decoding them. It creates no content index or transcript copy.
Configured executable wrappers must forward both `api get` and `debug paths db` commands.
The supported release/schema contract is pinned in [`live_opencode.py`](live_opencode.py).
An unsupported release/schema, remote service, or unverifiable database reports unavailable;
there is no full-history API fallback. Hit reads still use OpenCode's projected API.
Concurrent database commits are reported as partial coverage with snapshot results.
Missing sources, changing files, and pending JSONL tails are explicit coverage states;
the CLI exits **2** for partial/unavailable coverage while still returning verified hits.

MCP exposes the same engine as **`search_sessions`** and **`read_session`**. Server
startup and live requests do not initialize the archive or warm a model. Archived
tools initialize the store on first use. Reload existing MCP clients after updating.
Pi reads native JSONL directly using candidate-first `rg` scans, without invoking a Pi CLI.
The default root follows `PI_CODING_AGENT_DIR` (normally `~/.pi/agent`);
`CLYNC_PI_ROOT` overrides its sessions directory. Both grouped and flat session roots work.
Live reads follow tree edges, not append-order siblings, and report ambiguous branches.
Pi user-role messages are labelled `pi_user`: human authorship is not encoded in the
source, so `--sender user` excludes them. Omit that filter to search them.

`clync sync-pi` ingests primary Pi sessions and indexes them; `--no-index` skips
embedding and `--full` reparses all sessions. Regular sync also includes Pi.
Archive history retains entry/ancestor labels and excludes tool-output bodies and
delegated children. Unchanged files are skipped. Duplicate session IDs or incomplete
files fail explicitly instead of being silently indexed as complete.

## Quick start

> [!WARNING]
> Archive indexing and semantic relevance search use local embeddings and can consume substantial
> CPU/GPU resources. There is no full embedding opt-out: `sync --no-index` skips
> indexing for that run only; scheduled sync still indexes.

Requires **macOS**, Chrome signed into both web accounts, `uv`, Python 3.11+,
Homebrew, and Claude Code. From the cloned repository:

```sh
brew install postgresql@17 pgvector

# Chrome profile display names; both can use the same profile.
export CLYNC_CLAUDE_PROFILE="Your Claude profile"
export CLYNC_CHATGPT_PROFILE="Your ChatGPT profile"

uv run python clync.py setup
uv run python clync.py sync
uv run python clync.py doctor
```

Setup prepares the database, registers Claude Code tools and skills, and schedules
daily sync. The separate `sync` command imports and indexes your history.

<details>
<summary>Paths and authentication</summary>

- Full sync requires `~/.claude/projects` and `~/.codex`; override with
  `CLYNC_CC_ROOT` and `CLYNC_CODEX_ROOT`.
- PostgreSQL defaults to `/opt/homebrew/opt/postgresql@17/bin`. Use
  `CLYNC_PG_BIN` for another installation with pgvector `>= 0.8.0` for PostgreSQL 17
  (`0.8.4+` recommended for VACUUM HNSW fixes; earlier `0.8.x` runs with a warning).
- Cookie access uses the macOS login keychain. Refresh expired web logins
  in the selected Chrome profile.

</details>

## Connect an assistant

**Claude Code:** open a new session after setup.

**Codex:** register from the repository directory:

```sh
codex mcp add clync -- /bin/zsh -lc \
  'exec uv run --project "$1" python "$1/mcp_server.py"' clync "$PWD"
```

**OpenCode v2:** register from the repository directory:

```sh
opencode mcp add clync --global -- /bin/zsh -lc \
  'exec uv run --project "$1" python "$1/mcp_server.py"' clync "$PWD"
opencode mcp list
clync doctor --mcp-client opencode
```

OpenCode discovers compatible skills separately from MCP servers. Loading the Clync
skill does not connect its tools, and importing OpenCode history below does not
register an MCP client. `/mcps` manages configured connections; default Code Mode
exposes connected tools through catalog discovery. Configuration reloads automatically.

**Grok Build:** [Claude compatibility](https://docs.x.ai/build/features/mcp-servers#compatibility),
enabled by default, discovers the registration in `~/.claude.json`.
Other local stdio MCP clients can launch the same server command.
The login shell resolves `uv` from the current PATH, so the registration does
not depend on the Python environment that originally installed `uv`. `doctor`
checks Claude Code by default; `--mcp-client codex` checks Codex registration and its
executable, while the OpenCode check queries the current location's live connection state.

Ask “Why did I choose Postgres?” or “Find the session where we fixed ChatGPT sync.”

Use `search_sessions` / `read_session` for live coding-session finding.
`search_history` returns archived insights and transcript matches; `get_conversation`
retrieves an archived transcript.

## Usage

Run from the repository directory:

```sh
# Search original conversations.
uv run python clync.py search "authentication" --repo my-project

# Read an archived hit without fetching the whole conversation.
uv run python clync.py read '<unit-id>' --from-idx 10 --to-idx 20 --sender user

# Recall insights and supporting conversations.
uv run python clync.py dream recall "local model requirements" --evidence

# Check scheduled sync or browse commands.
uv run python clync.py status
uv run python clync.py --help
```

Use `sync-app`, `sync-chatgpt`, `sync-cc`, `sync-codex`, or `sync-opencode` for one source.
Each command has `--help`.

### OpenCode

`clync sync-opencode` reads the local OpenCode v2 service through the installed
`opencode` CLI, then indexes the imported transcripts. It uses OpenCode's existing
authentication and service discovery; no cookies or API token are needed.

To include it in ordinary and daily sync, save
`{"command":"/absolute/path/to/opencode"}` in `$CLYNC_DATA_HOME/opencode.json`
(default `~/.local/share/clync/opencode.json`). Use `command -v opencode` to find
the executable. `CLYNC_OPENCODE_CONFIG` selects another configuration file.
A configured source that becomes unavailable fails loudly without preventing
the other sources from syncing. The standalone command can also resolve
`opencode` from PATH when no configuration exists.

Only root sessions are imported; archived sessions and forks remain included.
Child/subagent transcripts are never exported. Synthetic/system messages,
subagent invocations, and **all tool-result bodies are excluded from storage**,
including raw metadata. User text, assistant prose/reasoning, compact tool intent,
and structured user answers to questions are retained. Root sessions started
headlessly are included: OpenCode does not expose a standard interactive-origin
field. Binary attachments are not imported.

Search with `--source opencode`; supported facets are `repo`, `session`, `model`,
and `sender`. Session IDs in clync are prefixed with `opencode:`. In-flight exports
are marked dirty and retried on the next sync. `--full` re-exports every root;
`--no-index` skips indexing for that run. Imported conversations are eligible for
Dream processing like the other transcript sources.

For Artiport, save `{"url":"https://YOUR-ARTIPORT","token":"READ_ONLY_TOKEN"}` in
`$CLYNC_DATA_HOME/artiport.json` (default `~/.local/share/clync/artiport.json`), with
file permissions `600`. `CLYNC_ARTIPORT_CONFIG` can select another configuration
file. One Artiport server is configured at a time; switching its instance removes
records imported from the previous instance after verifying the new inventory.
Run `clync sync-artiport` once; the configured server also joins the daily
sync. Indexing does not renew artifact retention. Archived artifacts remain
searchable; payload deletion removes their imported text and embeddings, even
with `--no-index`. Search with `--source artiport` or the normal all-source tools.
Artifact text is excluded from Dream distillation so expiration cannot leave a
permanent distilled copy.
After updating clync, reload existing MCP clients so their running server process
loads the new source; the CLI and newly started MCP clients load it immediately.

## Dream

Dream distills history into cited insights, distinguishing your positions from
assistant suggestions and recording revisions. Backfill existing history with:

```sh
uv run python clync.py dream backfill --max-calls 30
uv run python clync.py dream status
```

Backfills are resumable. Scheduled Dream runs process new activity after an
initial watermark-only run. Dream sends transcript content through Claude Code;
usage and billing depend on its authentication. The isolated worker uses the
Claude user settings' `env` and `apiKeyHelper`, including a configured provider.
It excludes other settings and retains inherited credentials when no user
settings file exists.

Use `--no-dream` with `install` or `setup` to disable scheduled Dream inference
while keeping history sync and indexing. Explicit `dream run` and `dream backfill`
commands remain available.

`install` and `setup` also register a separate Postgres LaunchAgent. It starts
the database at login and restarts it after an exit, independently of the daily
sync. Run `clync install` once to add supervision to an existing installation;
pass the same profiles, schedule, and Dream preference as before. `doctor`
reports missing database supervision. `uninstall` removes both launchd jobs
and stops the database while retaining its data.

## Cloud `clync-share` (opt-in serverless AWS)

`clync-share` deploys an encrypted per-owner Postgres schema + arm64 ECR Container Lambda
(direct `lambda:InvokeFunction` with per-call SigV4 STS verification — no ALB/API Gateway)
exposing two optional interfaces: owner-only full recall (`search_history`, `resolve_citation`)
and read-only, live-redacted teammate Q&A (`ask_<owner>`) with opaque citation handles
(`clync-<owner>-<rand>`). See [ADR 0006](docs/adr/0006-serverless-clync-share.md).

**Connect from an existing `clync` install (auto-detects the matching AWS profile from `~/.aws/config`):**

```sh
clync connect arn:aws:lambda:<region>:<account>:function:clync-share-<owner>
clync shares
clync ask <owner> "What did we decide about HNSW parameters?"
clync resolve-citation clync-<owner>-<id>   # owner only
```

Or invoke `/clync-connect <lambda-arn>` in Claude Code (`clync setup` installs the `clync-connect` skill alongside `clync` and `clync-ops`).

**Connect without installing `clync` (teammates, zero PyTorch/Postgres):**

```sh
uvx --from "git+https://github.com/joy13975/clync#subdirectory=share-client" \
  clync-share connect arn:aws:lambda:<region>:<account>:function:clync-share-<owner>
```

## Data and resources

History and the index live under `~/.local/share/clync` by default
(`CLYNC_DATA_HOME` overrides it). Embeddings run locally. Retrieved content
is shared with the assistant that queries it.

`uv run python clync.py unsetup` removes the scheduler and Claude Code integration,
stops the database, and retains its data. Remove manually added MCP registrations
separately.

<details>
<summary>Staying out of the way</summary>

`CLYNC_RESOURCE_FRACTION` controls embedding thread counts relative to logical
CPUs; explicit thread-pool variables take precedence.
`CLYNC_EMBED_DEVICE=cpu` keeps embedding off the GPU.

</details>

---

[Search guide](skill/SKILL.md) · [Architecture](docs/adr/) · [Dream design](docs/adr/0004-dream-layer.md)
