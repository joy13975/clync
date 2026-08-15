"""ChatGPT ingest — pure linearization of the mapping tree (no DB) + the network
ingest reconcile against a fake client (real Postgres, no network)."""
from __future__ import annotations

import chatgpt
import clync


# --------------------------------------------------------------------------- #
# Synthetic mapping-tree builders
# --------------------------------------------------------------------------- #
def _node(msg):
    return {"message": msg}


def _mapping(messages: list[dict | None]) -> dict:
    """Chain the given messages into a linear mapping tree (root -> ... -> leaf)
    and return {mapping, current_node}. A None entry is a message-less node (a
    real ChatGPT root/placeholder), which linearize must skip."""
    mapping, prev = {}, None
    ids = [f"n{i}" for i in range(len(messages))]
    for i, (nid, msg) in enumerate(zip(ids, messages)):
        mapping[nid] = {
            "id": nid, "message": ({"id": f"m{i}", **msg} if msg else None),
            "parent": prev, "children": [ids[i + 1]] if i + 1 < len(ids) else []}
        prev = nid
    return {"mapping": mapping, "current_node": ids[-1]}


def _user(text, ct="text"):
    parts = text if isinstance(text, list) else [text]
    return {"author": {"role": "user"}, "content": {"content_type": ct, "parts": parts},
            "create_time": 1000.0}


def _asst_text(text, model=None):
    m = {"author": {"role": "assistant"},
         "content": {"content_type": "text", "parts": [text]}, "create_time": 1001.0}
    if model:
        m["metadata"] = {"model_slug": model}
    return m


def _asst_thoughts(steps):
    return {"author": {"role": "assistant"},
            "content": {"content_type": "thoughts",
                        "thoughts": [{"summary": "", "content": s} for s in steps]},
            "create_time": 1002.0}


def _asst_recap(text):
    return {"author": {"role": "assistant"},
            "content": {"content_type": "reasoning_recap", "content": text},
            "create_time": 1002.5}


def _asst_code(code):
    return {"author": {"role": "assistant"},
            "content": {"content_type": "code", "text": code}, "create_time": 1003.0}


def _tool(text):
    return {"author": {"role": "tool"},
            "content": {"content_type": "execution_output", "text": text},
            "create_time": 1004.0}


# --------------------------------------------------------------------------- #
# Timestamp normalization (the two API surfaces disagree on format)
# --------------------------------------------------------------------------- #
def test_iso_accepts_both_list_iso_string_and_detail_epoch():
    assert chatgpt._iso("2026-08-15T10:23:55.109384Z") == \
        "2026-08-15T10:23:55.109384+00:00"                 # list surface: ISO string
    assert chatgpt._iso(1755253200.0) == "2025-08-15T10:20:00+00:00"  # detail: epoch
    assert chatgpt._iso(None) is None
    assert chatgpt._iso("   ") is None
    assert chatgpt._iso("not-a-timestamp") is None         # never fabricate a time


# --------------------------------------------------------------------------- #
# Pure content extraction
# --------------------------------------------------------------------------- #
def test_text_parts_drops_non_string_parts():
    content = {"parts": ["look at this", {"content_type": "image_asset_pointer"}, "and this"]}
    assert chatgpt._text_parts(content) == "look at this\nand this"
    assert chatgpt._text_parts({"parts": None}) == ""


def test_thoughts_text_joins_steps_and_reads_recap():
    assert chatgpt._thoughts_text(
        {"thoughts": [{"content": "step one"}, {"summary": "", "content": "step two"}]}
    ) == "step one\n\nstep two"
    assert chatgpt._thoughts_text({"content": "the recap"}) == "the recap"


def test_reuses_cc_truncation_constants():
    # SSOT: chatgpt.py imports cc.py's truncation limits, never redefines them.
    import cc
    assert chatgpt._TITLE_MAX is cc._TITLE_MAX
    assert chatgpt._TOOL_ARG_MAX is cc._TOOL_ARG_MAX
    assert chatgpt._TOOL_RESULT_HEAD is cc._TOOL_RESULT_HEAD


# --------------------------------------------------------------------------- #
# linearize — the whole tree -> a cleaned session
# --------------------------------------------------------------------------- #
def test_linearize_classifies_roles_and_content_types():
    detail = _mapping([
        None,                                  # root placeholder -> skipped
        _user("hello there"),
        _asst_thoughts(["reasoning a", "reasoning b"]),
        _asst_text("the answer", model="gpt-5-6-thinking"),
        _asst_code('{"q":"a search query"}'),
        _tool("tool stdout payload"),
        _asst_text("final reply", model="gpt-5-6-thinking"),
    ])
    detail["default_model_slug"] = "gpt-5-6-instant"
    meta = {"id": "conv-1", "title": "A Real Title",
            "create_time": "2026-08-15T04:00:00Z", "update_time": "2026-08-15T05:00:00Z"}

    conv = chatgpt.linearize(detail, meta)

    assert conv.session_id == "conv-1"
    assert conv.title == "A Real Title"
    assert conv.model == "gpt-5-6-thinking"           # last assistant model_slug wins
    assert conv.created_at == "2026-08-15T04:00:00+00:00"
    assert conv.updated_at == "2026-08-15T05:00:00+00:00"
    assert conv.repo is None and conv.cwd is None and conv.git_branch is None

    rendered = [(m.sender, m.role_detail, m.embed_text) for m in conv.messages]
    assert rendered == [
        ("user", "text", None),
        ("assistant", "reasoning", None),             # thoughts -> reasoning, embedded
        ("assistant", "text", None),
        ("assistant", "tool_use", ""),                # code -> tool_use, NOT embedded
        ("tool", "tool_result", ""),                  # tool output -> NOT embedded
        ("assistant", "text", None),
    ]
    assert conv.messages[1].text == "reasoning a\n\nreasoning b"
    assert [m.idx for m in conv.messages] == [0, 1, 2, 3, 4, 5]   # contiguous, in order


def test_linearize_skips_system_hidden_and_empty():
    detail = _mapping([
        {"author": {"role": "system"}, "content": {"content_type": "text", "parts": ["sys"]}},
        _user("kept prompt"),
        {"author": {"role": "assistant"},                          # hidden from user
         "content": {"content_type": "text", "parts": ["hidden"]},
         "metadata": {"is_visually_hidden_from_conversation": True}},
        {"author": {"role": "assistant"},                          # empty text -> skip
         "content": {"content_type": "text", "parts": [""]}},
        _asst_text("kept answer"),
    ])
    conv = chatgpt.linearize(detail, {"id": "c", "title": "t", "update_time": None})
    assert [(m.sender, m.text) for m in conv.messages] == [
        ("user", "kept prompt"), ("assistant", "kept answer")]


def test_linearize_truncates_tool_use_and_tool_result():
    big_code = "x" * 5000
    big_out = "y" * 5000
    detail = _mapping([_user("q"), _asst_code(big_code), _tool(big_out)])
    conv = chatgpt.linearize(detail, {"id": "c", "title": "t", "update_time": None})
    tool_use = next(m for m in conv.messages if m.role_detail == "tool_use")
    tool_res = next(m for m in conv.messages if m.role_detail == "tool_result")
    assert tool_use.text.endswith("…") and len(tool_use.text) <= chatgpt._TOOL_ARG_MAX + 1
    assert tool_res.text.endswith("…") and len(tool_res.text) <= chatgpt._TOOL_RESULT_HEAD + 1


def test_linearize_title_falls_back_to_first_user_prompt():
    detail = _mapping([_user("what is the airspeed velocity"), _asst_text("depends")])
    conv = chatgpt.linearize(detail, {"id": "c", "title": "", "update_time": None})
    assert conv.title == "what is the airspeed velocity"


def test_linear_chain_is_cycle_safe():
    # A malformed parent cycle must terminate the walk, not hang.
    mapping = {
        "a": {"id": "a", "message": _msg("A"), "parent": "b"},
        "b": {"id": "b", "message": _msg("B"), "parent": "a"},
    }
    chain = chatgpt._linear_chain({"mapping": mapping, "current_node": "a"})
    assert len(chain) <= 2      # terminates; never loops forever


def _msg(text):
    return {"id": "x", "author": {"role": "user"},
            "content": {"content_type": "text", "parts": [text]}}


# --------------------------------------------------------------------------- #
# Network ingest against a fake client (real Postgres, no network)
# --------------------------------------------------------------------------- #
class _FakeCGPTClient:
    def __init__(self, pairs):        # pairs: list[(meta, detail)]
        self._pairs = pairs

    def iter_conversations(self):
        for meta, _ in self._pairs:
            yield meta

    def get_conversation(self, cid):
        for meta, detail in self._pairs:
            if meta["id"] == cid:
                return detail
        raise KeyError(cid)


def _pair(cid, update_time, user_text, answer, title=None, model="gpt-5-6-thinking"):
    detail = _mapping([_user(user_text), _asst_text(answer, model=model)])
    detail["default_model_slug"] = model
    meta = {"id": cid, "title": title or answer[:20],
            "create_time": update_time, "update_time": update_time}
    return (meta, detail)


def _wire_cgpt(monkeypatch, client):
    monkeypatch.setattr(clync, "read_chatgpt_cookies",
                        lambda profile: {chatgpt.SESSION_COOKIE: "x"})
    monkeypatch.setattr(chatgpt, "ChatGPTClient",
                        lambda cookies, profile: client)


def test_ingest_chatgpt_end_to_end_then_incremental(pg_test_db, monkeypatch):
    client = _FakeCGPTClient([
        _pair("cg-1", "2026-08-15T05:00:00Z", "plum seed question", "the plum answer",
              title="Plum Seed Germination Tips"),
        _pair("cg-2", "2026-08-14T05:00:00Z", "mushroom nutrition", "the mushroom answer"),
    ])
    _wire_cgpt(monkeypatch, client)

    stats = clync.ingest_chatgpt("prof", full=True)
    assert stats["fetched"] == 2 and stats["conversations_in_db"] == 2

    con = clync.connect()
    try:
        rows = con.execute(
            "SELECT unit_id, kind, source, title, model, created_at, updated_at "
            "FROM units WHERE source='chatgpt' ORDER BY updated_at DESC").fetchall()
        assert [r["unit_id"] for r in rows] == ["cg-1", "cg-2"]
        assert {r["kind"] for r in rows} == {"chatgpt"}
        assert rows[0]["title"] == "Plum Seed Germination Tips"
        assert rows[0]["model"] == "gpt-5-6-thinking"
        assert rows[0]["updated_at"] is not None       # incremental watermark populated
        # messages landed
        nmsg = con.execute(
            "SELECT COUNT(*) n FROM messages WHERE unit_id='cg-1'").fetchone()["n"]
        assert nmsg == 2
    finally:
        con.close()

    # Nothing changed upstream -> a second (incremental) run skips everything.
    stats2 = clync.ingest_chatgpt("prof", full=False)
    assert stats2["fetched"] == 0 and stats2["unchanged"] == 2
    assert stats2["conversations_in_db"] == 2


def test_ingest_chatgpt_reconciles_deleted_conversation(pg_test_db, monkeypatch):
    full = _FakeCGPTClient([
        _pair("cg-1", "2026-08-15T05:00:00Z", "q1", "a1"),
        _pair("cg-2", "2026-08-14T05:00:00Z", "q2", "a2"),
    ])
    _wire_cgpt(monkeypatch, full)
    clync.ingest_chatgpt("prof", full=True)

    # cg-2 deleted upstream -> the live listing no longer returns it.
    _wire_cgpt(monkeypatch, _FakeCGPTClient([
        _pair("cg-1", "2026-08-15T05:00:00Z", "q1", "a1")]))
    stats = clync.ingest_chatgpt("prof", full=False)
    assert stats["conversations_removed"] == 1
    assert stats["conversations_in_db"] == 1

    con = clync.connect()
    try:
        remaining = [r["unit_id"] for r in con.execute(
            "SELECT unit_id FROM units WHERE source='chatgpt'").fetchall()]
        assert remaining == ["cg-1"]
        # messages of the deleted conversation cascaded away
        assert con.execute(
            "SELECT COUNT(*) n FROM messages WHERE unit_id='cg-2'").fetchone()["n"] == 0
    finally:
        con.close()


def test_ingest_chatgpt_reconcile_scoped_to_source(pg_test_db, monkeypatch):
    """A ChatGPT deletion-reconcile must touch ONLY source=chatgpt / kind=chatgpt,
    never another source's units (the source-partition invariant)."""
    # Seed a foreign (claude_ai) chat unit directly.
    con = clync.connect()
    try:
        con.execute(
            "INSERT INTO units (unit_id, kind, source, title, synced_at) "
            "VALUES ('other-1','chat','claude_ai','a claude.ai chat', now())")
        con.commit()
    finally:
        con.close()

    _wire_cgpt(monkeypatch, _FakeCGPTClient([
        _pair("cg-1", "2026-08-15T05:00:00Z", "q1", "a1")]))
    clync.ingest_chatgpt("prof", full=True)

    # Now ALL ChatGPT conversations vanish upstream -> reconcile empties chatgpt...
    _wire_cgpt(monkeypatch, _FakeCGPTClient([]))
    stats = clync.ingest_chatgpt("prof", full=False)
    assert stats["conversations_removed"] == 1
    assert stats["conversations_in_db"] == 0

    con = clync.connect()
    try:
        # ...but the claude_ai unit is untouched.
        assert con.execute(
            "SELECT COUNT(*) n FROM units WHERE source='claude_ai'").fetchone()["n"] == 1
    finally:
        con.close()
