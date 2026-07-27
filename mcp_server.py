"""MCP server exposing the local clync store to Claude Code (stdio).

Read-only query layer over the contained Postgres store that `clync sync`
populates (claude.ai chats + local Claude Code sessions + project docs — ADR
0003). `search_history` is one faceted hybrid-search endpoint across both sources;
`recall_knowledge` is the dream-first surface over the DISTILLED layer (ADR 0004);
`get_conversation` returns a full transcript by unit id. Registered with:

    claude mcp add --scope user clync -- \
        uv run --project ~/code/clync python ~/code/clync/mcp_server.py
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from clync import DEFAULT_TOPK, connect, ensure_cluster

mcp = FastMCP("clync")


@mcp.tool()
def search_transcripts(
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
    """RAW TRANSCRIPTS ONLY, with facets — the drill-down for episode questions
    ("which session", "in that repo", "back in June"). For a normal question use
    `search_history` instead: it searches transcripts AND the user's distilled
    positions in one call, so it cannot miss the half you did not ask for.

    query: natural-language query (NOT an FTS expression). Empty => a recency
      browse of the units matching the facets.
    source: "all" (default) | "claude_ai" | "claude_code" | "dream". "all" means
      the two RAW sources only; distilled dream units are returned only when
      asked for by name (prefer `search_insights` for those).
    claude.ai facets: project (a project name/uuid, or "any" = in some project,
      "none" = not in a project), model.
    Claude Code facets: repo (e.g. "clync"), worktree, branch, session (name or id).
    Temporal: since / until (ISO dates, on updated_at); sort "relevance" | "recency".
    A facet that contradicts `source` (e.g. repo= with source="claude_ai") is
    rejected loudly.
    """
    import search
    ensure_cluster()      # tool entry provisions (hybrid_search deliberately doesn't)
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
def search_history(query: str = "", topic: str | None = None,
                   limit: int = DEFAULT_TOPK, as_of: str | None = None,
                   include_evidence: bool = True) -> str:
    """THE DEFAULT for any question about the user's own past — searches BOTH layers
    in one call: their DISTILLED positions (what they concluded) and their RAW
    transcripts (what was actually said, in claude.ai chats and Claude Code
    sessions). Use it for "what do I think about X", "what did I decide about Y",
    "where did I discuss Z", and anything in between — you do not have to classify
    the question first, which is the point.

    Output is labelled sections (COVERAGE / INSIGHTS / RAW TRANSCRIPTS), never one
    blended ranking, because a distilled claim and a transcript chunk are not ranked
    on comparable scores. Read the labels: COVERAGE states what is NOT distilled and
    must be relayed rather than silently omitted, and RAW hits are undistilled
    transcript, NOT the user's settled position. Insights carry a stance tag — the
    user's own assertion is not the same as something Claude once proposed.

    Drill down only when this is not enough: `search_insights` for positions alone,
    `search_transcripts` for faceted raw search (by repo, project, branch, date).

    topic: restrict to one topic id (see `clync dream topics`); omitted => best-effort
      match from the query.
    as_of: ISO date (e.g. "2026-07-01") — the position(s) held AT that date rather
      than the current ones.
    """
    return _recall(query, topic, limit, as_of, include_evidence, include_raw=True)


@mcp.tool()
def search_insights(query: str = "", topic: str | None = None,
                    limit: int = DEFAULT_TOPK, as_of: str | None = None,
                    include_evidence: bool = True) -> str:
    """DISTILLED POSITIONS ONLY — the drill-down for when `search_history` surfaced a
    position and you want more of them without the transcript noise. Same claims,
    same stance tags and citations, no RAW section.

    Prefer `search_history` for a first query: skipping the transcripts means a
    question the distilled layer does not cover yet returns nothing at all, and
    COVERAGE is then the only thing telling you the gap is a gap.
    """
    return _recall(query, topic, limit, as_of, include_evidence, include_raw=False)


def _recall(query, topic, limit, as_of, include_evidence, *, include_raw: bool) -> str:
    """One body for both recall surfaces — they differ ONLY by include_raw, and two
    copies of the error handling is where the two would drift."""
    import dream
    ensure_cluster()
    con = connect()
    try:
        result = dream.recall(con, query, topic_id=topic, limit=limit, as_of=as_of,
                              include_evidence=include_evidence,
                              include_raw=include_raw)
    # A bad topic id or an unparseable as_of is CORRECTABLE input, and the caller is a
    # model: it must get the same "here is what is wrong" string the sibling tool
    # returns for a bad facet, not an opaque exception. `ValueError` alone caught
    # neither of the two errors actually reachable here.
    except (ValueError, dream.DreamError) as e:
        return f"Invalid query: {e}"
    finally:
        con.close()

    return "\n".join(dream.format_recall(
        result, requested_topic=topic, include_evidence=include_evidence))


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
