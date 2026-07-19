"""MCP server exposing the local claude.ai history DB to Claude Code (stdio).

Read-only query layer over the SQLite/FTS5 database that `clync sync` populates.
Registered with:

    claude mcp add --scope user clync -- \
        uv run --project ~/code/clync python ~/code/clync/mcp_server.py
"""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from clync import connect

mcp = FastMCP("clync")


@mcp.tool()
def search_history(query: str, limit: int = 10) -> str:
    """Full-text search across all synced claude.ai conversation messages.

    `query` is an SQLite FTS5 MATCH expression (supports AND/OR/NEAR/"phrases").
    Returns matching conversations with a highlighted snippet and the message uuid.
    """
    con = connect()
    try:
        rows = con.execute(
            """SELECT c.name AS conv, c.uuid AS cuuid, c.updated_at AS updated,
                      m.sender AS sender,
                      snippet(messages_fts, 0, '**', '**', ' … ', 18) AS snip
               FROM messages_fts
               JOIN conversations c ON c.uuid = messages_fts.conversation_uuid
               JOIN messages m ON m.uuid = messages_fts.message_uuid
               WHERE messages_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (query, limit),
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return f"No matches for {query!r}."
    out = [f"{len(rows)} match(es) for {query!r}:\n"]
    for r in rows:
        out.append(
            f"- conversation: {r['conv'] or '(untitled)'} "
            f"(uuid={r['cuuid']}, updated={r['updated']})\n"
            f"  [{r['sender']}] {r['snip']}"
        )
    return "\n".join(out)


@mcp.tool()
def get_conversation(uuid: str) -> str:
    """Return the full transcript of one conversation by its uuid, oldest-first."""
    con = connect()
    try:
        conv = con.execute(
            "SELECT name, summary, model, updated_at FROM conversations WHERE uuid=?",
            (uuid,),
        ).fetchone()
        if not conv:
            return f"No conversation with uuid {uuid!r} in the local DB."
        msgs = con.execute(
            "SELECT sender, text, created_at FROM messages "
            "WHERE conversation_uuid=? ORDER BY idx",
            (uuid,),
        ).fetchall()
        files = con.execute(
            "SELECT file_kind, file_name, local_path FROM files "
            "WHERE conversation_uuid=?", (uuid,),
        ).fetchall()
    finally:
        con.close()
    header = (f"# {conv['name'] or '(untitled)'}\n"
              f"model={conv['model']} updated={conv['updated_at']}\n"
              f"summary: {conv['summary'] or '(none)'}\n")
    if files:
        header += "attached files:\n" + "\n".join(
            f"  - [{f['file_kind']}] {f['file_name']}"
            + (f" -> {f['local_path']}" if f['local_path'] else " (not downloaded)")
            for f in files
        ) + "\n"
    body = "\n\n".join(
        f"## {m['sender']} ({m['created_at']})\n{m['text']}" for m in msgs
    )
    return f"{header}\n{body}"


@mcp.tool()
def list_conversations(limit: int = 20, project_uuid: str | None = None) -> str:
    """List the most recently updated conversations, optionally filtered by project."""
    con = connect()
    try:
        if project_uuid:
            rows = con.execute(
                "SELECT name, uuid, updated_at, message_count FROM conversations "
                "WHERE project_uuid=? ORDER BY updated_at DESC LIMIT ?",
                (project_uuid, limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT name, uuid, updated_at, message_count FROM conversations "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    finally:
        con.close()
    if not rows:
        return "No conversations synced yet. Run `claude_history.py sync`."
    return json.dumps(
        [
            {"name": r["name"], "uuid": r["uuid"],
             "updated_at": r["updated_at"], "messages": r["message_count"]}
            for r in rows
        ],
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()
