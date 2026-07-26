---
name: clync
description: Search, read, or recall the user's OWN past history and distilled positions across BOTH their claude.ai conversations (chat history from the claude.ai web/desktop app) AND their local Claude Code sessions (~/.claude/projects transcripts), synced locally by clync. Use when the user refers to something they discussed with Claude before — "what did I say about X", "find my chat about Y", "the conversation where we designed Z", "pull up my thread on …", "search my claude history", "did I already ask Claude about …", "what did I discuss in the clync repo", "my chat about X in project Y", "find that Claude Code session where I fixed…" — and ALSO when they ask what they think or have concluded: "what's my position on X", "what are my principles about Y", "how do I usually approach Z", "have I settled on X", "did I reject that approach before", "what do I know about X". NOT for web search.
allowed-tools: mcp__clync__search_history, mcp__clync__recall_knowledge, mcp__clync__get_conversation, Bash
---

clync keeps a local Postgres/pgvector mirror of the user's history from **two
sources**: claude.ai chats (+ project docs) and local Claude Code sessions. On
top of those sits a **derived layer** of distilled positions (ADR 0004).
Prefer the **MCP tools** (already registered); fall back to the `clync` CLI
only if the tools are unavailable in this session.

## Which tool — episode or position?

Two questions that sound alike need different tools. Get this right first:

| The user is asking | Tool | Because |
|---|---|---|
| **Where/when did I discuss X?** — find a conversation, a snippet, a pasted doc, a session | `search_history` | They want the *episode*. Raw transcripts, cited by name. |
| **What do I think about X?** — my position, principle, convention, "how do I usually…", "did I already decide/reject…" | `recall_knowledge` | They want the *conclusion*. Distilled claims, each grounded in real quotes and tagged with who held the position. |

When both readings are live, run `recall_knowledge` first: it returns the raw tier
too, so it answers the episode question as a side effect, whereas
`search_history` cannot answer the position question at all.

Do NOT try to answer a "what do I think" question by reading raw transcripts
yourself and summarizing. That is what the distilled layer already did, with a
grounding gate you cannot reproduce in-context, and your summary would silently
mix Claude's past suggestions in with the user's own positions.

## Reach for this when
The user points at prior conversations or coding sessions — recalling a
decision, a snippet, a document they pasted, or a whole thread — from either
source. Their words won't say "clync"; they'll say "that conversation where…",
"what did I tell Claude about…", "find my chat on…", "what did I discuss in the
`clync` repo", "that Claude Code session on the `feat-x` branch". Attachments
count too: pasted-file text is indexed.

## How to answer

1. **Find it** — `mcp__clync__search_history(query, ...)`. `query` is a
   natural-language question, not a keyword expression — the tool runs hybrid
   semantic + lexical search (BGE-M3 dense + learned-sparse, RRF-fused) over
   the indexed corpus, so phrase it the way you'd ask a person. An **empty
   query** degrades to a recency browse of whatever facets you pass — use this
   for "what did I last work on" style asks. Returns the best-matching unit per
   hit (name + uuid + a text snippet).

   Facets narrow the search to one source or dimension — pass only what the
   user's phrasing implies:
   - `source`: `"all"` (default) | `"claude_ai"` | `"claude_code"`.
   - claude.ai only: `project` (name/uuid, or `"any"`/`"none"`), `model`.
   - Claude Code only: `repo`, `worktree`, `branch`, `session` (name or id).
   - Both: `since` / `until` (ISO dates), `sort` (`"relevance"` default |
     `"recency"`), `limit`.
   - A facet that doesn't apply to the chosen `source` fails loud — e.g. don't
     pass `repo=` with `source="claude_ai"`.

   Example mappings: "what did I discuss in the clync repo" →
   `search_history(query="", source="claude_code", repo="clync")`; "my chat
   about X in project Y" → `search_history(query="X", source="claude_ai",
   project="Y")`; "that session on the feat-cc-sessions branch" →
   `search_history(query="", source="claude_code", branch="feat-cc-sessions")`.

2. **Recall a position** — `mcp__clync__recall_knowledge(query, topic=…,
   limit=…, as_of=…, include_evidence=…)`. Returns explicit tiers, and you must
   respect the distinction between them:
   - `TOPIC` — which topic answered. If it says `[derived …, not requested]`, the
     topic was inferred from what came back; if it looks wrong, the answer is
     about something else than what was asked — say so rather than reporting it.
   - `COVERAGE` — gaps, stated loudly. **Never suppress these.** "That topic has
     never been dug" or "12 units changed since it was last distilled" is the
     honest answer to a confident-sounding question; presenting a thin digest as
     the user's settled view is the failure mode this layer exists to avoid.
   - `DIGEST` — the synthesized current position, in fixed sections (Settled /
     Rejected approaches / Changed positions / Open).
   - `INSIGHTS` — atomic claims, each with a **stance**. Attribute correctly:
     `user_asserted`/`user_endorsed` is the user's own position;
     `user_rejected` is something they turned down (report it AS rejected — it is
     knowledge, not an error); `co_derived` was worked out together;
     `claude_proposed` was never adjudicated by the user, so never present it as
     what they think. Pass `include_evidence=True` when the user wants proof.
   - `RAW` — drill-down transcript hits, not distilled. Use
     `get_conversation` on these when they want the whole thread.

   `as_of="YYYY-MM-DD"` reads the bi-temporal history — the position held *then*,
   which is how to answer "have I changed my mind about X" or "what did I think
   before". Positions are never deleted, only superseded.

3. **Read it** — `mcp__clync__get_conversation(uuid)` for the full transcript
   (oldest-first) of either a claude.ai chat or a Claude Code session,
   including source-specific metadata (model/project, or repo/worktree/branch),
   a list of attached files, and any downloaded image paths.

Cite the conversation/session name when you use a result. If search returns
nothing, say so — do not invent history. The data is only as fresh as the last
sync.

## CLI fallback / maintenance (Bash)
- `clync search ["<query>"] [--source all|claude_ai|claude_code] [--repo R] [--worktree W] [--branch B] [--project P] [--model M] [--since/--until DATE] [--sort relevance|recency] [--lang en|ja|zh]` · `clync list [--source all|claude_ai|claude_code]` — same queries from the shell.
- `clync sync` — pull new/updated claude.ai conversations, THEN ingest local Claude Code sessions, then reindex (needs `$CLYNC_PROFILE` for the claude.ai leg; the Claude Code leg runs regardless).
- `clync sync-cc [--full]` — ingest only local Claude Code sessions (no network); `clync sync-app` syncs only claude.ai.
- `clync index [--full]` — rebuild the hybrid-search index without a sync.
- `clync dream recall "<query>" [--topic T] [--as-of ISO] [--evidence]` — the same tiered recall from the shell; `clync dream topics` lists what is being distilled.
- `clync status` — last successful sync + scheduled-run log.

If a query returns nothing and the user expects recent chats, suggest
`clync sync` (the scheduled sync runs daily; it may just be stale).
