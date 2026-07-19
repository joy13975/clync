---
name: clync
description: Search or read the user's OWN past claude.ai conversations (their chat history from the claude.ai web/desktop app), synced locally by clync. Use when the user refers to something they discussed with Claude before — "what did I say about X", "find my chat about Y", "the conversation where we designed Z", "pull up my thread on …", "search my claude history", "did I already ask Claude about …". NOT for the current Claude Code session's own history, and NOT for web search.
allowed-tools: mcp__clync__search_history, mcp__clync__get_conversation, mcp__clync__list_conversations, Bash
---

clync keeps a local SQLite/FTS5 mirror of the user's claude.ai conversation
history. Prefer the **MCP tools** (already registered); fall back to the `clync`
CLI only if the tools are unavailable in this session.

## Reach for this when
The user points at prior claude.ai chats — recalling a decision, a snippet, a
document they pasted, or a whole thread. Their words won't say "clync"; they'll
say "that conversation where…", "what did I tell Claude about…", "find my chat
on…". Attachments count too: pasted-file text is indexed.

## How to answer

1. **Find it** — `mcp__clync__search_history(query, limit)`. `query` is an FTS5
   MATCH expression: bare words are ANDed, `"quoted"` is a phrase, `OR`/`NEAR`
   work. Returns conversation name + uuid + a highlighted snippet per hit.
2. **Read it** — `mcp__clync__get_conversation(uuid)` for the full transcript
   (oldest-first), including a list of attached files and any downloaded image
   paths.
3. **Browse** — `mcp__clync__list_conversations(limit, project_uuid?)` for the
   most recently updated conversations.

Cite the conversation name when you use a result. If search returns nothing,
say so — do not invent history. The data is only as fresh as the last sync.

## CLI fallback / maintenance (Bash)
- `clync search "<fts query>"` · `clync list` — same queries from the shell.
- `clync sync` — pull new/updated conversations now (needs `$CLYNC_PROFILE`).
- `clync status` — last successful sync + scheduled-run log.

If a query returns nothing and the user expects recent chats, suggest
`clync sync` (the scheduled sync runs daily; it may just be stale).
