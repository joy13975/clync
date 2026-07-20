"""MCP server exposing the local claude.ai history DB to Claude Code (stdio).

Read-only query layer over the DB that `clync sync` populates: `search_history`
runs clync's hybrid BGE-M3 index (contained Postgres/pgvector — see search.py);
`get_conversation` / `list_conversations` read the SQLite store directly.
Registered with:

    claude mcp add --scope user clync -- \
        uv run --project ~/code/clync python ~/code/clync/mcp_server.py
"""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from clync import DEFAULT_TOPK, connect

mcp = FastMCP("clync")


@mcp.tool()
def search_history(query: str, limit: int = DEFAULT_TOPK) -> str:
    """Hybrid semantic + lexical search across all synced claude.ai conversations.

    Takes a natural-language query (not an FTS5 expression) and returns the best-
    matching conversation per hit, ranked by a fused dense + learned-sparse score.
    """
    import search
    if not search.available():
        return ("Hybrid search is not installed. Run `uv sync --extra search` "
                "in the clync repo, then `clync index`.")
    results = search.hybrid_search(query, topk=limit)
    if not results:
        return f"No matches for {query!r}."
    out = [f"{len(results)} match(es) for {query!r}:\n"]
    for r in results:
        snippet = r["text"].replace("\n", " ")[:300]
        out.append(
            f"- conversation: {r['conv_name'] or '(untitled)'} "
            f"(uuid={r['conv_uuid']}, lang={r['lang']}, score={r['score']:.4f})\n"
            f"  {snippet}"
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
