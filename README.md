# clync

**Your AI conversations, remembered across tools.**

Sync conversation history into a local archive, search it from your assistant,
and recall cited insights without hunting through old chats.

[Quick start](#quick-start) · [Connect an assistant](#connect-an-assistant) · [Dream](#dream) · [Design](docs/adr/)

## What it imports

| History source | Imported content |
|---|---|
| **claude.ai** | Conversations and project documents |
| **ChatGPT** | Conversations exposed by its history-listing API |
| **Claude Code** | Local interactive CLI sessions |
| **Codex CLI** | Local interactive terminal sessions |

> [!NOTE]
> **Grok Build can use clync's MCP tools, but its history is not imported.**
> MCP access and history ingestion are separate capabilities. Codex Desktop,
> headless runs, and separate subagent transcripts are not supported history sources.
> ChatGPT project-history discovery is not implemented.

## Before you start

macOS, Python 3.11+, and Homebrew are required for the instructions below.
Chrome is needed only for web-history sync.

> [!WARNING]
> Indexing and relevance search run **BGE-M3 locally** and can consume substantial
> CPU/GPU resources and memory, especially on the first import. There is currently
> **no full embedding opt-out**. `--no-index` skips one sync's indexing;
> it does not disable search embeddings or scheduled indexing.

## Quick start

Install dependencies and keep the clone at a stable path:

```sh
brew install uv postgresql@17 pgvector
git clone https://github.com/joy13975/clync.git
cd clync
uv sync
```

Import whichever local history you have; each command also indexes it:

```sh
uv run python clync.py sync-cc
uv run python clync.py sync-codex
uv run python clync.py search "why we changed the authentication flow"
```

Run only the import commands whose source directories exist. These manual steps
do not install a background schedule.

<details>
<summary>Import claude.ai or ChatGPT history</summary>

Sign in to the relevant website in Chrome, then substitute your Chrome profile's
display name:

```sh
uv run python clync.py sync-app --profile "Person 1"
uv run python clync.py sync-chatgpt --profile "Person 1"
```

Web sync reads that profile's existing login cookies. Expired sessions may require
opening the website again. Both commands accept `CLYNC_PROFILE` as the default
profile. Web APIs are unofficial and may change.

</details>

## Connect an assistant

From the repository directory, register the server with your chosen client:

```sh
# Claude Code
claude mcp add --scope user clync -- uv run --project "$PWD" python "$PWD/mcp_server.py"

# Codex
codex mcp add clync -- uv run --project "$PWD" python "$PWD/mcp_server.py"
```

Start a new client session, then ask it to search your history.
**Grok Build** can import the Claude MCP registration through its
[Claude compatibility support](https://docs.x.ai/build/features/mcp-servers#compatibility).

The MCP tools provide combined insight/transcript search, separate searches of
each layer, and full conversation retrieval. The shell `search` command searches
raw history by default. Filter by source, repository, project, session, or dates;
see `uv run python clync.py search --help`.

## Dream

Dream uses Claude Code to distill **stance-tagged insights with source citations**.
It tracks revisions and can return the evidence behind a recalled insight.

```sh
uv run python clync.py dream topics --seed
uv run python clync.py dream backfill --max-calls 30
uv run python clync.py dream recall "authentication decisions" --evidence
```

Backfill mines existing history and saves progress for later runs. The first
incremental `dream run` initializes its watermark rather than mining the archive.
Dream consumes Claude usage; billing depends on your Claude Code authentication.

## Data and operation

The archive lives in a dedicated local PostgreSQL/pgvector cluster under
`~/.local/share/clync`; override this with `CLYNC_DATA_HOME`.
Embeddings run locally, but retrieved text goes to your chosen assistant, and
Dream sends selected conversation content to Claude.

Use `--help` for commands and options. [CLAUDE.md](CLAUDE.md) indexes the code,
skills, and operational guidance; [architecture decisions](docs/adr/) explain
the retrieval and storage design.

[MIT license](LICENSE).
