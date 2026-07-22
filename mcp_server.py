"""MCP server exposing the local clync store to Claude Code (stdio).

Read-only query layer over the contained Postgres store that `clync sync`
populates (claude.ai chats + local Claude Code sessions + project docs — ADR
0003). `search_history` is one faceted hybrid-search endpoint across both sources;
`get_conversation` returns a full transcript by unit id. Registered with:

    claude mcp add --scope user clync -- \
        uv run --project ~/code/clync python ~/code/clync/mcp_server.py
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from clync import DEFAULT_TOPK, connect, ensure_cluster

mcp = FastMCP("clync")


@mcp.tool()
def search_history(
    query: str = "",
    source: str = "all",
    project: str | None = None,
    model: str | None = None,
    repo: str | None = None,
    worktree: str | None = None,
    branch: str | None = None,
    session: str | None = None,
    since: str | None = None,
    until: str | None = None,
    sort: str = "relevance",
    limit: int = DEFAULT_TOPK,
) -> str:
    """Faceted hybrid semantic + lexical search across the user's own history —
    both claude.ai chats and local Claude Code sessions.

    query: natural-language query (NOT an FTS expression). Empty => a recency
      browse of the units matching the facets.
    source: "all" (default) | "claude_ai" | "claude_code".
    claude.ai facets: project (a project name/uuid, or "any" = in some project,
      "none" = not in a project), model.
    Claude Code facets: repo (e.g. "clync"), worktree, branch, session (name or id).
    Temporal: since / until (ISO dates, on updated_at); sort "relevance" | "recency".
    A facet that contradicts `source` (e.g. repo= with source="claude_ai") is
    rejected loudly.
    """
    import search
    try:
        results = search.hybrid_search(
            query, topk=limit, source=source, project=project, model=model,
            repo=repo, worktree=worktree, branch=branch, session=session,
            since=since, until=until, sort=sort)
    except ValueError as e:                 # facet/source contradiction — surface it
        return f"Invalid query: {e}"
    if not results:
        return f"No matches for {query!r} (source={source})."
    out = [f"{len(results)} match(es) for {query!r} (source={source}):\n"]
    for r in results:
        snippet = r["text"].replace("\n", " ")[:300]
        out.append(
            f"- [{r['source']}] {r['unit_name'] or '(untitled)'} "
            f"(uuid={r['unit_id']}, lang={r['lang']}, score={r['score']:.4f})\n"
            f"  {snippet}"
        )
    return "\n".join(out)


@mcp.tool()
def get_conversation(uuid: str) -> str:
    """Return the full transcript of one unit (claude.ai chat or Claude Code
    session) by its uuid/id, oldest-first."""
    ensure_cluster()
    con = connect()
    try:
        u = con.execute(
            "SELECT kind, source, title, summary, model, updated_at, repo, cwd, "
            "worktree, git_branch, cc_version, project_name "
            "FROM units WHERE unit_id=%s", (uuid,)).fetchone()
        if not u:
            return f"No unit with uuid {uuid!r} in the local store."
        msgs = con.execute(
            "SELECT sender, role_detail, text, created_at FROM messages "
            "WHERE unit_id=%s ORDER BY idx", (uuid,)).fetchall()
        files = con.execute(
            "SELECT file_kind, file_name, local_path FROM files WHERE unit_id=%s",
            (uuid,)).fetchall()
    finally:
        con.close()

    if u["source"] == "claude_code":
        meta = (f"source=claude_code repo={u['repo']} worktree={u['worktree']} "
                f"branch={u['git_branch']} cc_version={u['cc_version']}\n"
                f"cwd={u['cwd']}\n")
    else:
        meta = (f"source=claude_ai model={u['model']} "
                f"project={u['project_name'] or '(none)'}\n"
                f"summary: {u['summary'] or '(none)'}\n")
    header = f"# {u['title'] or '(untitled)'}\nupdated={u['updated_at']}\n{meta}"
    if files:
        header += "attached files:\n" + "\n".join(
            f"  - [{f['file_kind']}] {f['file_name']}"
            + (f" -> {f['local_path']}" if f['local_path'] else " (not downloaded)")
            for f in files
        ) + "\n"
    body = "\n\n".join(
        f"## {m['sender']}"
        + (f" [{m['role_detail']}]" if m["role_detail"] else "")
        + f" ({m['created_at']})\n{m['text']}"
        for m in msgs
    )
    return f"{header}\n{body}"


if __name__ == "__main__":
    mcp.run()
