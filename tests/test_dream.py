"""Dream layer tests (ADR 0004).

The model calls are stubbed (`stub_worker`) so the gates, lifecycle, schema guard
and retrieval tiers are exercised deterministically without spending quota. The
one real-Opus test is marked `slow`.

These target the invariants that actually protect the knowledge base — grounding,
attribution, non-circularity, bi-temporal supersession, fail-loud — not accessors.
"""
from __future__ import annotations

import json
import subprocess

import pytest

import clync
import dream


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def store(pg_test_db, mock_embed, monkeypatch):
    """Throwaway store with the dream schema + one seeded topic and a raw unit
    whose messages can be legitimately cited.

    The content is INDEXED, because a dig is now positioned by retrieval: it reads
    the window around the matched message, so `seeds_for` needs real chunks. The
    embedder is the deterministic stub, so this stays fast."""
    con = clync.connect()
    con.execute("INSERT INTO units (unit_id,kind,source,title,created_at,updated_at,"
                "msg_count,synced_at) VALUES ('u1','cc_session','claude_code','sess',"
                "'2026-01-01','2026-01-02',2,now())")
    for i, (mid, sender, text) in enumerate([
            ("m1", "user", "Fix bugs at the layer that should have prevented them."),
            ("m2", "assistant", "Agreed - that is the root-cause rule.")]):
        con.execute("INSERT INTO messages (unit_id,msg_id,idx,sender,text,created_at) "
                    "VALUES ('u1',%s,%s,%s,%s,'2026-01-01')", (mid, i, sender, text))
    con.execute("INSERT INTO dream_topics (topic_id,name,charter,probe_queries,status,"
                "created_at) VALUES ('t1','Topic One','Charter text.',%s,'active',now())",
                (clync._json(["root cause"]),))
    con.commit()
    _reindex()
    yield con
    con.close()


def _reindex():
    """Refresh the vector index after seeding raw content. A dig positions itself
    from `chunks`, so unindexed content is invisible to it."""
    import search
    search.build_index(full=False)


@pytest.fixture
def stub_worker(monkeypatch):
    """Replace the model call with a queue of canned structured outputs."""
    outputs: list[dict] = []

    def _fake(prompt, schema, *, system, effort=None, model=dream.WORKER_MODEL,
              timeout=900):
        if not outputs:
            raise AssertionError("stub_worker ran out of queued outputs")
        return outputs.pop(0), {"input_tokens": 1, "output_tokens": 2}

    monkeypatch.setattr(dream, "_run_worker", _fake)
    return outputs


# The distiller cites SHORT slot#index labels, never UUIDs (see render_windows) —
# so a test supplies both the candidate and the refmap its labels resolve through.
REFMAP = {"a#0": ("u1", "m1"), "a#1": ("u1", "m2")}


def _candidate(**over):
    c = {"statement": "Fix bugs at the layer that should have prevented them.",
         "elaboration": "Root-cause rule.", "stance": "user_asserted",
         "evidence": [{"ref": "a#0", "quote": "Fix bugs at the layer"}]}
    c.update(over)
    return c


# --------------------------------------------------------------------------- #
# The DDL splitter — a `;` inside a comment is prose, not a terminator
# --------------------------------------------------------------------------- #
def test_sql_statements_ignores_semicolon_inside_a_comment():
    script = "CREATE TABLE a (x int);\n-- a note; with a semicolon\nCREATE TABLE b (y int);"
    assert clync.sql_statements(script) == ["CREATE TABLE a (x int)",
                                            "CREATE TABLE b (y int)"]


def test_real_schemas_split_into_valid_looking_statements():
    for script in (clync.RAW_SCHEMA, dream.DREAM_SCHEMA):
        for stmt in clync.sql_statements(script):
            assert stmt.upper().startswith(("CREATE", "ALTER", "DROP", "INSERT")), stmt


# --------------------------------------------------------------------------- #
# Schema guard: derived data is rebuildable, so a stale shape is recreated
# --------------------------------------------------------------------------- #
def test_stale_insight_shape_is_recreated_topics_survive_and_units_are_purged(store):
    clync.set_meta(store, dream.WATERMARK_KEY, "2026-01-01T00:00:00+00:00")
    store.execute("ALTER TABLE dream_insights DROP COLUMN contested")
    store.execute("INSERT INTO units (unit_id,kind,source,title,synced_at) "
                  "VALUES ('dream-x',%s,'dream','stale insight',now())",
                  (dream.KIND_INSIGHT,))
    store.commit()

    # A stale shape must NOT be rebuilt implicitly: the rebuild destroys every
    # distilled insight, and it used to happen as a side effect of whatever call
    # provisioned the store first — including a read, which then took DDL locks and
    # deadlocked against its caller's open transaction. Refuse, and say how.
    with pytest.raises(clync.StaleDerivedSchema, match="clync migrate"):
        dream.ensure_dream_schema()
    assert "contested" not in {r["column_name"] for r in store.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='dream_insights'").fetchall()}, "refusal must not mutate"
    assert clync.get_meta(store, dream.WATERMARK_KEY) is not None

    dream.ensure_dream_schema(rebuild=True)          # the explicit opt-in

    cols = {r["column_name"] for r in store.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='dream_insights'").fetchall()}
    assert cols == dream._INSIGHT_COLS
    # hand-edited config survives; the derived units do NOT linger orphaned
    assert [t["topic_id"] for t in dream.topics(store)] == ["t1"]
    assert store.execute("SELECT count(*) n FROM units WHERE source='dream'"
                         ).fetchone()["n"] == 0
    # ...and the watermark is cleared, or the nightly pass would only look at units
    # changed since a rebuild that discarded everything — silently never re-mining
    # the wiped history while reporting success.
    assert clync.get_meta(store, dream.WATERMARK_KEY) is None


# --------------------------------------------------------------------------- #
# Condensation — the sender vocabulary differs per source
# --------------------------------------------------------------------------- #
def test_skeleton_reads_both_sender_vocabularies(store):
    """claude.ai says 'human', Claude Code says 'user'. Hardcoding one silently
    empties the other source's entire corpus."""
    store.execute("INSERT INTO units (unit_id,kind,source,title,synced_at) "
                  "VALUES ('u2','chat','claude_ai','chat',now())")
    store.execute("INSERT INTO messages (unit_id,msg_id,idx,sender,text) "
                  "VALUES ('u2','m1',0,'human','a claude.ai user turn')")
    store.commit()
    assert "user turn" in dream.skeleton(store, "u2")          # 'human'
    assert "prevented them" in dream.skeleton(store, "u1")     # 'user'


def test_skeleton_uses_body_for_project_docs(store):
    store.execute("INSERT INTO units (unit_id,kind,source,title,synced_at) "
                  "VALUES ('d1','project_doc','claude_ai','doc',now())")
    store.execute("INSERT INTO messages (unit_id,msg_id,idx,sender,text) "
                  "VALUES ('d1','m1',0,'project_doc','the document body')")
    store.commit()
    assert "document body" in dream.skeleton(store, "d1")


def test_skeleton_refuses_to_triage_on_metadata_alone(store):
    store.execute("INSERT INTO units (unit_id,kind,source,title,synced_at) "
                  "VALUES ('u3','chat','claude_ai','empty',now())")
    store.commit()
    with pytest.raises(dream.DreamError, match="empty triage skeleton"):
        dream.skeleton(store, "u3")


def _long_unit(con, unit_id="long", n=200, hit_at=140,
               hit_text="the marmoset calibration needs a torque wrench"):
    """A unit long enough that its opening and its matched passage cannot overlap."""
    con.execute("INSERT INTO units (unit_id,kind,source,title,updated_at,msg_count,"
                "synced_at) VALUES (%s,'cc_session','claude_code','long',now(),%s,now())",
                (unit_id, n))
    for i in range(n):
        con.execute(
            "INSERT INTO messages (unit_id,msg_id,idx,sender,text,created_at) "
            "VALUES (%s,%s,%s,%s,%s,'2026-01-01')",
            (unit_id, f"m{i}", i, "user" if i % 2 == 0 else "assistant",
             hit_text if i == hit_at else f"routine turn {i} about other things"))
    con.commit()
    _reindex()


def test_windows_are_centred_on_the_retrieval_hit_not_the_units_opening(store):
    """THE point of the rendering. Measured on the real corpus before this: rendering
    from idx 0 up to a char budget put 44/45 = 98% of retrieval matches OUTSIDE the
    window the dreamer was shown (median match at idx 1039, median 23 messages
    shown). The layer was distilling conversation preambles."""
    _long_unit(store, hit_at=140)
    body, refmap, slots = dream.render_windows(
        store, [{"unit_id": "long", "msg_idx": 140}])

    idxs = sorted(int(r.split("#")[1]) for r in refmap)
    assert 140 in idxs, "the matched message itself must be rendered"
    assert idxs == list(range(140 - dream.WINDOW_BEFORE, 140 + dream.WINDOW_AFTER + 1))
    assert 0 not in idxs, "rendering the opening is the bug this replaced"
    assert "marmoset calibration" in body                 # the matched text is present
    assert "<-- retrieval match" in body                  # ...and marked as the match
    assert "200 messages" in body                         # honest about what was NOT shown
    assert slots == {"a": "long"}


def test_refs_are_slot_plus_index_never_uuids(store):
    """Measured: shown a correct 36-char UUID, the model wrote back one transposed
    digit, and the grounding gate discarded three genuinely-grounded insights over
    the typo. So refs are short and code owns the mapping. Slot+index also makes the
    ref a RULE rather than a table, which is what lets navigation address messages
    that were not in the first rendering."""
    body, refmap, _ = dream.render_windows(store, [{"unit_id": "u1", "msg_idx": 0}])
    assert refmap == {"a#0": ("u1", "m1"), "a#1": ("u1", "m2")}
    assert "[a#0 user]" in body and "[a#1 assistant]" in body
    # No per-message id anywhere in the prompt: there is nothing to mistype. (The
    # unit header still names its unit as context — it is not citable, since the
    # schema only accepts a ref.)
    assert not any(msg_id in body for _, msg_id in refmap.values())


def test_refs_are_unique_across_a_multi_unit_batch(store):
    """One label space per prompt: a duplicated label would silently resolve a
    citation to the wrong transcript."""
    _add_units(store, 2)
    body, refmap, slots = dream.render_windows(
        store, [{"unit_id": u, "msg_idx": 0} for u in ("u1", "u2", "u3")])
    assert len(refmap) == 4                       # u1 has two messages, u2/u3 one each
    assert {u for u, _ in refmap.values()} == {"u1", "u2", "u3"}
    assert slots == {"a": "u1", "b": "u2", "c": "u3"}
    assert sorted(refmap) == ["a#0", "a#1", "b#0", "c#0"]


def test_a_seed_without_a_matched_position_is_refused(store):
    """A browse (no query) yields no match position. Centring on idx 0 in that case
    would quietly reinstate the render-the-opening bug, so it fails instead."""
    with pytest.raises(dream.DreamError, match="no msg_idx"):
        dream.render_windows(store, [{"unit_id": "u1", "msg_idx": None}])


def test_requested_expansions_render_and_merge_with_the_seed_window(store):
    """Navigation: the dreamer asks for a range it was not shown, and gets it. An
    expansion abutting the seed window renders as ONE continuous passage — a repeated
    overlapping block would read as two separate exchanges of the same turns."""
    _long_unit(store, hit_at=140)
    seed = [{"unit_id": "long", "msg_idx": 140}]
    _, before, _ = dream.render_windows(store, seed)
    body, after, _ = dream.render_windows(
        store, seed, [{"slot": "a", "from_idx": 100, "to_idx": 125, "why": "context"}])

    assert set(before) < set(after), "the expansion must add refs, not replace them"
    assert "a#100" in after and "a#120" in after
    assert body.count("-- a: messages") == 2      # a gap at 126-133 keeps them apart
    # ...and an expansion that touches the seed window merges into one range
    body2, _, _ = dream.render_windows(
        store, seed, [{"slot": "a", "from_idx": 120, "to_idx": 135, "why": "lead-in"}])
    assert body2.count("-- a: messages") == 1


def test_a_wide_expansion_can_never_evict_the_matched_message(store):
    """Found while building this: an expansion merged with the seed window and the
    size cap then spent itself from the START of the merged range, dropping the
    matched message and its neighbours — reinstating the render-the-wrong-part bug
    through the very feature meant to fix it. The budget is spent OUTWARD FROM THE
    MATCH, so the hit and its closest context always survive."""
    _long_unit(store, n=200, hit_at=140)
    seed = [{"unit_id": "long", "msg_idx": 140}]
    _, refs, _ = dream.render_windows(
        store, seed,
        [{"slot": "a", "from_idx": 0, "to_idx": 139, "why": "the entire lead-in"}])
    assert "a#140" in refs, "the matched message was evicted by an expansion"
    for near in ("a#139", "a#141"):
        assert near in refs, f"{near} (adjacent to the match) was evicted"


def test_an_expansion_naming_an_unknown_transcript_fails_loud(store):
    _long_unit(store, hit_at=140)
    with pytest.raises(dream.DreamError, match="unknown transcript"):
        dream.render_windows(store, [{"unit_id": "long", "msg_idx": 140}],
                             [{"slot": "z", "from_idx": 0, "to_idx": 5, "why": "x"}])


def test_an_oversized_range_says_what_it_omitted(store):
    """Silent truncation is how the dreamer comes to believe it read a passage it
    did not. The omission is stated in the text the model sees."""
    _long_unit(store, n=200, hit_at=140)
    body, _, _ = dream.render_windows(
        store, [{"unit_id": "long", "msg_idx": 140}],
        [{"slot": "a", "from_idx": 0, "to_idx": 199, "why": "everything"}])
    assert "message(s) in this range omitted for size" in body


def test_distill_serves_read_requests_then_stops(store, stub_worker):
    """One navigation round: the dreamer asks, code fulfils, the dreamer distills from
    the wider rendering. Insights from BOTH rounds are kept — a round is an addition,
    not a do-over, so nothing already paid for is discarded."""
    _long_unit(store, hit_at=140)
    first = _distill_one(ref="a#140", statement="First-round claim.")
    first["read_requests"] = [{"slot": "a", "from_idx": 100, "to_idx": 120,
                               "why": "the thread starts earlier"}]
    second = _distill_one(ref="a#105", statement="Second-round claim.")
    stub_worker += [first, second]

    notes = []
    got, refmap = dream.distill(store, dream.get_topic(store, "t1"),
                                [{"unit_id": "long", "msg_idx": 140}], notes)
    assert [c["statement"] for c in got] == ["First-round claim.", "Second-round claim."]
    assert "a#105" in refmap, "the requested range must be citable in round two"
    assert not stub_worker, "exactly two calls: one round of navigation, then stop"
    assert notes == []


def test_unserved_read_requests_are_reported_never_silently_dropped(store, stub_worker):
    """A request arriving in the final round cannot be served. Saying so is the only
    signal that MAX_NAV_ROUNDS is set too low; dropping it reads as 'the windows
    sufficed' when they did not."""
    _long_unit(store, hit_at=140)
    for statement in ("R1", "R2"):
        out = _distill_one(ref="a#140", statement=statement)
        out["read_requests"] = [{"slot": "a", "from_idx": 0, "to_idx": 9,
                                 "why": "still need the opening"}]
        stub_worker.append(out)

    notes = []
    dream.distill(store, dream.get_topic(store, "t1"),
                  [{"unit_id": "long", "msg_idx": 140}], notes)
    assert any("unserved" in n for n in notes), notes
    assert any("still need the opening" in n for n in notes)


def test_too_many_read_requests_are_capped_and_the_cap_is_reported(store, stub_worker):
    _long_unit(store, hit_at=140)
    out = _distill_one(ref="a#140")
    out["read_requests"] = [{"slot": "a", "from_idx": i, "to_idx": i + 1, "why": "w"}
                            for i in range(dream.MAX_EXPANSIONS + 3)]
    stub_worker += [out, _distill_one(ref="a#140", statement="second")]
    notes = []
    dream.distill(store, dream.get_topic(store, "t1"),
                  [{"unit_id": "long", "msg_idx": 140}], notes)
    assert any(f"capped {dream.MAX_EXPANSIONS + 3} read requests" in n for n in notes)


def test_seeds_for_positions_a_pending_unit_by_retrieval(store):
    """`dream_pending` stores unit assignments; WHERE the topic lives in a unit is a
    retrieval fact computed at dig time. It must find the matching passage, not idx 0."""
    _long_unit(store, hit_at=140, hit_text="root cause rather than the symptom site")
    seeds, unpositioned = dream.seeds_for(
        store, dream.get_topic(store, "t1"), ["long"])
    assert unpositioned == []
    assert [s["unit_id"] for s in seeds] == ["long"]
    assert seeds[0]["msg_idx"] == 140


def test_an_unindexed_unit_is_reported_not_positioned_at_zero(store):
    """A unit with no chunk cannot be positioned. Centring it on idx 0 would silently
    reinstate the render-the-opening bug; dropping it would lose the assignment."""
    store.execute("INSERT INTO units (unit_id,kind,source,title,updated_at,synced_at) "
                  "VALUES ('ghost','cc_session','claude_code','g',now(),now())")
    store.commit()                      # deliberately NOT indexed
    seeds, unpositioned = dream.seeds_for(
        store, dream.get_topic(store, "t1"), ["ghost"])
    assert seeds == [] and unpositioned == ["ghost"]


def test_render_refuses_a_batch_with_nothing_quotable(store):
    store.execute("INSERT INTO units (unit_id,kind,source,title,synced_at) "
                  "VALUES ('empty','chat','claude_ai','e',now())")
    store.commit()
    with pytest.raises(dream.DreamError, match="nothing quotable"):
        dream.render_windows(store, [{"unit_id": "empty", "msg_idx": 0}])


# --------------------------------------------------------------------------- #
# Gate 1 — grounding is a DB check and cannot be argued with
# --------------------------------------------------------------------------- #
def test_ground_accepts_a_verbatim_quote(store):
    assert len(dream.ground(store, _candidate(), REFMAP)) == 1


def test_ground_accepts_whitespace_differences_only(store):
    c = _candidate(evidence=[{"ref": "a#0", "quote": "  Fix   bugs\nat the layer  "}])
    assert len(dream.ground(store, c, REFMAP)) == 1


@pytest.mark.parametrize("evidence,match", [
    ([], "cites no evidence"),
    ([{"ref": "e99", "quote": "Fix bugs"}], "never shown to the distiller"),
    ([{"ref": "", "quote": "Fix bugs"}], "never shown to the distiller"),
    ([{"ref": "a#0", "quote": "text that is absent"}], "not found verbatim"),
])
def test_ground_rejects_ungrounded_candidates(store, evidence, match):
    with pytest.raises(dream.DreamError, match=match):
        dream.ground(store, _candidate(evidence=evidence), REFMAP)


def test_ground_rejects_a_candidate_with_one_fabricated_reference(store):
    """Partly grounded is not grounded — it is untrustworthy."""
    c = _candidate(evidence=[{"ref": "a#0", "quote": "Fix bugs at the layer"},
                             {"ref": "ghost", "quote": "Fix bugs at the layer"}])
    with pytest.raises(dream.DreamError, match="never shown to the distiller"):
        dream.ground(store, c, REFMAP)


def test_ground_refuses_to_cite_derived_text(store):
    """Citing a dream unit would make the layer circular."""
    store.execute("INSERT INTO units (unit_id,kind,source,title,synced_at) "
                  "VALUES ('dr1',%s,'dream','an insight',now())", (dream.KIND_INSIGHT,))
    store.execute("INSERT INTO messages (unit_id,msg_id,idx,sender,text) "
                  "VALUES ('dr1','m1',0,'dream','Fix bugs at the layer')")
    store.commit()
    c = _candidate(evidence=[{"ref": "d1", "quote": "Fix bugs at the layer"}])
    with pytest.raises(dream.DreamError, match="non-raw unit"):
        dream.ground(store, c, {**REFMAP, "d1": ("dr1", "m1")})


def test_ground_dedupes_two_quotes_from_one_message(store):
    """(src_unit, src_msg) is the citation key AND the evidence PK."""
    c = _candidate(evidence=[{"ref": "a#0", "quote": "Fix bugs at the layer"},
                             {"ref": "a#0", "quote": "should have prevented"}])
    assert len(dream.ground(store, c, REFMAP)) == 1


def test_a_user_stance_must_cite_a_user_authored_turn(store):
    """The mechanical half of attribution: without it, the only things between an
    assistant turn and a 'user_asserted' insight are two LLM opinions — and
    passes_substance waives the two-source bar on exactly that stance."""
    c = _candidate(evidence=[{"ref": "a#1", "quote": "root-cause rule"}])
    with pytest.raises(dream.DreamError, match="no user-authored turn"):
        dream.ground(store, c, REFMAP)                # e2 -> m2, an assistant turn
    ok = _candidate(stance="claude_proposed",
                    evidence=[{"ref": "a#1", "quote": "root-cause rule"}])
    assert len(dream.ground(store, ok, REFMAP)) == 1  # non-user stance: fine


def test_a_falsify_corrected_stance_is_rechecked_mechanically():
    """falsify may UPGRADE a stance to user_asserted; the authorship gate must
    re-run on the corrected value, not only on the distiller's original."""
    with pytest.raises(dream.DreamError, match="no user-authored turn"):
        dream._require_user_evidence(
            "user_endorsed", [{"sender": "assistant"}], "some claim")
    assert dream._require_user_evidence(
        "user_endorsed", [{"sender": "human"}], "some claim") is None


# --------------------------------------------------------------------------- #
# Gate 3 — substance
# --------------------------------------------------------------------------- #
def test_substance_gate_admits_a_single_source_only_when_user_asserted():
    one = [{"src_unit_id": "u1", "src_msg_id": "m1"}]
    two = [{"src_unit_id": "u1", "src_msg_id": "m1"},
           {"src_unit_id": "u2", "src_msg_id": "m9"}]
    assert dream.passes_substance({"stance": "user_asserted"}, one)
    assert not dream.passes_substance({"stance": "claude_proposed"}, one)
    assert dream.passes_substance({"stance": "claude_proposed"}, two)


# --------------------------------------------------------------------------- #
# Worker — every failure mode raises; none returns "empty"
# --------------------------------------------------------------------------- #
def _proc(stdout, rc=0):
    return subprocess.CompletedProcess([], rc, stdout=stdout, stderr="")


@pytest.mark.parametrize("stdout,rc,match", [
    ("not json", 0, "non-JSON"),
    (json.dumps({"is_error": True, "subtype": "error_during_execution"}), 0, "failed"),
    (json.dumps({"subtype": "error_max_turns"}), 0, "failed"),
    (json.dumps({"subtype": "success", "result": "text only"}), 0,
     "no validated structured_output"),
    ("", 1, "exited 1"),
])
def test_worker_fails_loud_on_every_bad_result(monkeypatch, stdout, rc, match):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _proc(stdout, rc))
    with pytest.raises(dream.DreamError, match=match):
        dream._run_worker("p", {}, system="s")


def test_worker_never_persists_a_session(monkeypatch):
    """A worker transcript under ~/.claude/projects would be re-ingested by
    sync-cc, making the layer circular."""
    seen = {}

    def _capture(argv, **kw):
        seen["argv"] = argv
        return _proc(json.dumps({"subtype": "success", "structured_output": {"a": 1}}))

    monkeypatch.setattr(subprocess, "run", _capture)
    dream._run_worker("p", {}, system="s")
    assert "--no-session-persistence" in seen["argv"]
    assert "--strict-mcp-config" in seen["argv"]      # no MCP tool schemas in prompt
    assert seen["argv"][seen["argv"].index("--tools") + 1] == ""


def test_failed_call_is_recorded_then_reraised(store, monkeypatch):
    def _boom(*a, **k):
        raise dream.DreamError("kaboom")

    monkeypatch.setattr(dream, "_run_worker", _boom)
    with pytest.raises(dream.DreamError, match="kaboom"):
        dream._call(store, "triage", {}, None, "p", {}, system="s")
    row = store.execute("SELECT state, error FROM dream_queue").fetchone()
    assert row["state"] == "failed" and "kaboom" in row["error"]


def test_throttling_is_distinguishable_from_other_failures():
    for msg in ("hit the rate limit", "usage limit reached", "429 Too Many Requests"):
        with pytest.raises(dream.DreamThrottled):
            dream._reraise_throttle(dream.DreamError(msg))
    assert dream._reraise_throttle(dream.DreamError("syntax error")) is None


# --------------------------------------------------------------------------- #
# Triage — hallucinated ids must not be silently dropped
# --------------------------------------------------------------------------- #
def test_triage_routes_and_omits_unassigned(store, stub_worker):
    stub_worker.append({"assignments": [
        {"unit_id": "u1", "topic_ids": ["t1"], "reason": "r"}]})
    assert dream.triage(store, ["u1"]) == {"u1": ["t1"]}


def test_triage_rejects_an_invented_topic(store, stub_worker):
    stub_worker.append({"assignments": [
        {"unit_id": "u1", "topic_ids": ["ghost-topic"], "reason": "r"}]})
    with pytest.raises(dream.DreamError, match="invented topic_id"):
        dream.triage(store, ["u1"])


def test_triage_rejects_an_unknown_unit(store, stub_worker):
    stub_worker.append({"assignments": [
        {"unit_id": "not-in-batch", "topic_ids": [], "reason": "r"}]})
    with pytest.raises(dream.DreamError, match="unknown unit_id"):
        dream.triage(store, ["u1"])


def test_triage_fails_loud_on_an_omitted_unit(store, stub_worker):
    """A low-effort call that drops a list item must not look like 'assigned no
    topics' — the watermark would advance past the omission permanently."""
    _add_units(store, 1)                                   # u2
    stub_worker.append({"assignments": [
        {"unit_id": "u1", "topic_ids": [], "reason": "r"}]})
    with pytest.raises(dream.DreamError, match="no verdict for offered"):
        dream.triage(store, ["u1", "u2"])


# --------------------------------------------------------------------------- #
# Gate 2 — an unjudged candidate must never be promoted by default
# --------------------------------------------------------------------------- #
def test_falsify_drops_rejected_and_keeps_contested(store, stub_worker):
    cands = [{**_candidate(), "_evidence": []},
             {**_candidate(statement="B"), "_evidence": []}]
    stub_worker.append({"verdicts": [
        {"index": 0, "verdict": "rejected", "corrected_stance": "user_asserted",
         "reason": "generic"},
        {"index": 1, "verdict": "contested", "corrected_stance": "co_derived",
         "reason": "two readings"}]})
    rejections: list[str] = []
    out = dream.falsify(store, dream.get_topic(store, "t1"), cands, rejections)
    assert [c["statement"] for c in out] == ["B"]
    assert out[0]["stance"] == "co_derived" and out[0]["verdict"] == "contested"
    # the REASON survives, not just the count — it is what tunes charters/prompts
    assert len(rejections) == 1 and "generic" in rejections[0]


def test_falsify_fails_loud_if_a_candidate_went_unjudged(store, stub_worker):
    cands = [{**_candidate(), "_evidence": []}, {**_candidate(statement="B"),
                                                 "_evidence": []}]
    stub_worker.append({"verdicts": [
        {"index": 0, "verdict": "upheld", "corrected_stance": "user_asserted",
         "reason": "ok"}]})
    with pytest.raises(dream.DreamError, match="skipped candidate index"):
        dream.falsify(store, dream.get_topic(store, "t1"), cands)


# --------------------------------------------------------------------------- #
# Lifecycle — append-only, bi-temporal
# --------------------------------------------------------------------------- #
def _persist_one(store, **over):
    t = dream.get_topic(store, "t1")
    c = _candidate(**over)
    c["_evidence"] = dream.ground(store, c, REFMAP)
    c.update({"action": "new", "target_id": ""})
    dream.persist(store, t, [c])
    return store.execute("SELECT * FROM dream_insights ORDER BY distilled_at DESC "
                         "LIMIT 1").fetchone()


def test_persist_writes_an_indexable_unit_and_its_citations(store):
    row = _persist_one(store)
    assert row["status"] == dream.STATUS_ACTIVE and row["support_count"] == 1
    u = store.execute("SELECT kind, source FROM units WHERE unit_id=%s",
                      (row["unit_id"],)).fetchone()
    assert (u["kind"], u["source"]) == (dream.KIND_INSIGHT, dream.DREAM_SOURCE)
    # a unit with no message row is stored but never indexed -> silently unfindable
    assert store.execute("SELECT count(*) n FROM messages WHERE unit_id=%s",
                         (row["unit_id"],)).fetchone()["n"] == 1
    assert store.execute("SELECT count(*) n FROM dream_evidence WHERE unit_id=%s",
                         (row["unit_id"],)).fetchone()["n"] == 1


def test_reinforce_support_count_is_derived_from_distinct_cited_units(store):
    """'N source(s)' must be true by construction: re-mining the SAME unit (a
    re-touched conversation re-dug nightly) reinforces without inflating the
    counter; only evidence from a genuinely new source unit raises it."""
    first = _persist_one(store)
    t = dream.get_topic(store, "t1")
    c = _candidate()
    c["_evidence"] = dream.ground(store, c, REFMAP)
    c.update({"action": "reinforce", "target_id": first["unit_id"]})
    dream.persist(store, t, [c])
    dream.persist(store, t, [c])              # third night, byte-identical evidence
    rows = store.execute("SELECT * FROM dream_insights").fetchall()
    assert len(rows) == 1 and rows[0]["support_count"] == 1   # ONE distinct source

    _add_units(store, 1, start=90)            # u90 — a genuinely new source unit
    c2 = _candidate(evidence=[{"ref": "e9", "quote": "Fix bugs at the layer"}])
    c2["_evidence"] = dream.ground(store, c2, {**REFMAP, "e9": ("u90", "m1")})
    c2.update({"action": "reinforce", "target_id": first["unit_id"]})
    dream.persist(store, t, [c2])
    assert store.execute("SELECT support_count FROM dream_insights"
                         ).fetchone()["support_count"] == 2


def test_contradict_closes_the_old_window_and_keeps_history(store):
    old = _persist_one(store)
    t = dream.get_topic(store, "t1")
    c = _candidate(statement="Actually, fix it where it surfaces.")
    c["_evidence"] = dream.ground(store, c, REFMAP)
    c.update({"action": "contradict", "target_id": old["unit_id"]})
    dream.persist(store, t, [c])

    prev = store.execute("SELECT * FROM dream_insights WHERE unit_id=%s",
                         (old["unit_id"],)).fetchone()
    assert prev["status"] == dream.STATUS_SUPERSEDED   # kept, not deleted
    assert prev["valid_until"] is not None             # window CLOSED
    assert prev["superseded_by"] is not None
    assert prev["superseded_kind"] == "contradict"     # WHY is recorded, not inferred
    assert store.execute("SELECT count(*) n FROM dream_insights").fetchone()["n"] == 2


def test_refine_closes_the_window_and_records_the_kind(store):
    """Every supersession closes the window — a NULL valid_until would satisfy
    every as_of read forever, returning the retired wording beside its
    replacement. `refine` is distinguished by the RECORDED kind, never by a
    NULL timestamp."""
    old = _persist_one(store)
    t = dream.get_topic(store, "t1")
    c = _candidate(statement="Fix bugs at the preventing layer (sharper wording).")
    c["_evidence"] = dream.ground(store, c, REFMAP)
    c.update({"action": "refine", "target_id": old["unit_id"]})
    dream.persist(store, t, [c])
    prev = store.execute("SELECT * FROM dream_insights WHERE unit_id=%s",
                         (old["unit_id"],)).fetchone()
    assert prev["status"] == dream.STATUS_SUPERSEDED
    assert prev["valid_until"] is not None             # window ALWAYS closed
    assert prev["superseded_kind"] == "refine"


def test_as_of_after_a_refine_returns_exactly_one_row(store):
    """A historical read must never fabricate a second simultaneous position out
    of a rewording."""
    old = _persist_one(store)
    t = dream.get_topic(store, "t1")
    c = _candidate(statement="Sharper wording of the same position.")
    c["_evidence"] = dream.ground(store, c, REFMAP)
    c.update({"action": "refine", "target_id": old["unit_id"]})
    dream.persist(store, t, [c])
    got = dream.recall(store, "", topic_id="t1", as_of="2026-06-01")["insights"]
    assert [i["statement"] for i in got] == ["Sharper wording of the same position."]


def test_older_contradicting_evidence_never_inverts_the_timeline(store):
    """Backfill retrieval is relevance-ranked, so mining a 2024 chat after a 2026
    one is routine. Superseding backwards would give the current position an
    EMPTY validity window and report the stale claim as current — instead the
    old claim lands as a contested NEW row and the target stays untouched."""
    cur = _persist_one(store)                          # valid_from = 2026-01-01
    _add_units(store, 1, start=80)                     # u80
    store.execute("UPDATE messages SET created_at='2024-01-01' WHERE unit_id='u80'")
    store.commit()
    t = dream.get_topic(store, "t1")
    c = _candidate(statement="The opposite, argued back in 2024.")
    c["_evidence"] = dream.ground(
        store, c, {"a#0": ("u80", "m1")})
    c.update({"action": "contradict", "target_id": cur["unit_id"]})
    stats = dream.persist(store, t, [c])
    assert stats == {"new": 1, "reinforce": 0, "refine": 0, "contradict": 0,
                     "out_of_order": 1}
    kept = store.execute("SELECT * FROM dream_insights WHERE unit_id=%s",
                         (cur["unit_id"],)).fetchone()
    assert kept["status"] == dream.STATUS_ACTIVE and kept["valid_until"] is None
    added = store.execute("SELECT * FROM dream_insights WHERE unit_id<>%s",
                          (cur["unit_id"],)).fetchone()
    assert added["contested"] is True and added["superseded_by"] is None


# --------------------------------------------------------------------------- #
# Non-circularity + facets
# --------------------------------------------------------------------------- #
def test_unqualified_search_excludes_derived_units(store, pg_test_db, mock_embed):
    _persist_one(store)
    pg_test_db.build_index()
    assert {h["source"] for h in pg_test_db.hybrid_search("bugs layer", source="all")
            } <= set(clync.RAW_SOURCES)
    assert {h["source"] for h in pg_test_db.hybrid_search("bugs layer", source="dream")
            } == {dream.DREAM_SOURCE}


def test_raw_facets_are_rejected_for_the_dream_source(pg_test_db):
    with pytest.raises(ValueError, match="do not apply to source='dream'"):
        pg_test_db.hybrid_search("x", source="dream", repo="clync")


# --------------------------------------------------------------------------- #
# Recall — tiers, loud gaps, bi-temporal
# --------------------------------------------------------------------------- #
def test_recall_states_coverage_gaps_loudly(store):
    r = dream.recall(store, "anything", topic_id="t1")
    assert any("no digest" in c for c in r["coverage"])
    assert any("never been dug" in c for c in r["coverage"])
    assert r["digest"] is None and r["insights"] == []


def test_recall_warns_when_the_dig_is_stale(store):
    _persist_one(store)
    store.execute("UPDATE dream_topics SET last_dig_at='2026-01-01' WHERE topic_id='t1'")
    store.execute("UPDATE units SET updated_at=now() WHERE unit_id='u1'")
    store.commit()
    assert any("not yet distilled" in c
               for c in dream.recall(store, "", topic_id="t1")["coverage"])


def test_recall_derives_the_topic_from_what_came_back(store, pg_test_db, mock_embed):
    _persist_one(store)
    pg_test_db.build_index()
    r = dream.recall(store, "bugs layer prevented")
    assert r["topic"] is not None and r["topic"]["topic_id"] == "t1"


def test_recall_as_of_reads_the_historical_position(store):
    _persist_one(store)
    assert dream.recall(store, "", topic_id="t1", as_of="2020-01-01")["insights"] == []
    assert len(dream.recall(store, "", topic_id="t1", as_of="2026-06-01")["insights"]) == 1


@pytest.fixture
def keyword_embed(monkeypatch):
    """A DISCRIMINATING embedding stub: the dense direction depends on which
    keywords the text contains, so an off-topic query gets a low cosine to every
    indexed chunk. The uniform mock_embed stub cannot express irrelevance, which
    is exactly what the recall relevance floor needs to be tested against."""
    import search
    words = ["bugs", "layer", "zucchini", "sourdough"]

    def _fake(texts, max_length):
        dense, sparse = [], []
        for t in texts:
            v = [0.0] * search.DENSE_DIM
            for i, w in enumerate(words):
                if w in t.lower():
                    v[i] = 1.0
            if not any(v):
                v[len(words)] = 1.0
            norm = sum(x * x for x in v) ** 0.5
            dense.append([x / norm for x in v])
            sparse.append({i: 1.0 for i, w in enumerate(words)
                           if w in t.lower()} or {9: 1.0})
        return dense, sparse

    monkeypatch.setattr(search, "_embed", _fake)


def test_recall_returns_an_empty_tier_for_an_off_topic_query(store, pg_test_db,
                                                             keyword_embed):
    """RRF always returns topk rows, so without a floor an off-topic query gets a
    full confident insight list and 'no gaps' — attribution pollution asserted
    as complete. Below the dense-similarity floor the tier must be EMPTY, the
    gap stated, and the query must never fall through to an unfiltered scan."""
    _persist_one(store)
    pg_test_db.build_index()
    r = dream.recall(store, "zucchini sourdough hydration")
    assert r["insights"] == [] and r["topic"] is None
    assert any("no distilled insight matches" in c for c in r["coverage"])
    on_topic = dream.recall(store, "bugs layer")
    assert len(on_topic["insights"]) == 1     # the floor does not eat real matches


def test_recall_ranks_trusted_stances_above_unadjudicated(store):
    _persist_one(store, statement="asserted one", stance="user_asserted")
    _persist_one(store, statement="proposed one", stance="claude_proposed")
    got = [i["stance"] for i in dream.recall(store, "", topic_id="t1")["insights"]]
    assert got.index("user_asserted") < got.index("claude_proposed")


# --------------------------------------------------------------------------- #
# Digest — fixed sections, bullets never invented
# --------------------------------------------------------------------------- #
def test_digest_buckets_route_by_stance_and_contest(store):
    _persist_one(store, statement="settled one", stance="user_asserted")
    _persist_one(store, statement="rejected one", stance="user_rejected")
    _persist_one(store, statement="unadjudicated one", stance="claude_proposed")
    b = dream._digest_buckets(store, "t1")
    assert [r["statement"] for r in b["settled"]] == ["settled one"]
    assert [r["statement"] for r in b["rejected"]] == ["rejected one"]
    assert [r["statement"] for r in b["open"]] == ["unadjudicated one"]


def test_digest_changed_bucket_keys_on_contradict_never_refine(store):
    """A refine is the SAME position in better words — reporting it under
    'Changed positions' with a date asserts a change of mind that never
    happened. Only a recorded contradict-supersession lands there."""
    t = dream.get_topic(store, "t1")
    old = _persist_one(store)

    refined = _candidate(statement="Same position, sharper wording.")
    refined["_evidence"] = dream.ground(store, refined, REFMAP)
    refined.update({"action": "refine", "target_id": old["unit_id"]})
    dream.persist(store, t, [refined])
    assert dream._digest_buckets(store, "t1")["changed"] == []

    mid = store.execute("SELECT unit_id FROM dream_insights WHERE status=%s",
                        (dream.STATUS_ACTIVE,)).fetchone()
    flipped = _candidate(statement="Actually, the opposite position.")
    flipped["_evidence"] = dream.ground(store, flipped, REFMAP)
    flipped.update({"action": "contradict", "target_id": mid["unit_id"]})
    dream.persist(store, t, [flipped])
    changed = dream._digest_buckets(store, "t1")["changed"]
    assert [r["statement"] for r in changed] == ["Actually, the opposite position."]
    assert changed[0]["superseded_statement"] == "Same position, sharper wording."


def test_digest_prints_only_db_statements_even_if_the_model_rambles(store, stub_worker):
    _persist_one(store, statement="the only real claim")
    stub_worker.append({s: "INVENTED CLAIM: everything is fine" if s == "settled" else ""
                        for s in dream.DIGEST_SECTIONS})
    text = dream.consolidate(store, dream.get_topic(store, "t1"))
    assert "the only real claim" in text
    assert text.count("- ") == 1          # exactly one bullet, from the DB


def test_digest_is_none_for_a_topic_with_no_insights(store):
    assert dream.consolidate(store, dream.get_topic(store, "t1")) is None


# --------------------------------------------------------------------------- #
# Run modes — the nightly path must never sweep history
# --------------------------------------------------------------------------- #
def test_first_incremental_run_initialises_the_watermark_without_digging(store):
    r = dream.run_incremental(store)
    assert r["changed"] == 0 and r["calls"] == 0
    assert "backfill" in r["note"]
    assert clync.get_meta(store, dream.WATERMARK_KEY) is not None


def test_incremental_run_with_nothing_new_costs_zero_calls(store):
    clync.set_meta(store, dream.WATERMARK_KEY, "2030-01-01T00:00:00+00:00")
    store.commit()
    r = dream.run_incremental(store)
    assert r["calls"] == 0 and r["digs"] == {} and "no new" in r["note"]


def test_untouched_topics_cost_nothing(store, stub_worker):
    clync.set_meta(store, dream.WATERMARK_KEY, "2020-01-01T00:00:00+00:00")
    store.commit()
    stub_worker.append({"assignments": [{"unit_id": "u1", "topic_ids": [],
                                         "reason": "mechanical"}]})
    r = dream.run_incremental(store)
    assert r["changed"] == 1 and r["topics_touched"] == []
    # `calls` is what the audit log says happened, NOT what the budget reserved: a
    # dig reserves 3 up front but skips reconcile on a topic with no prior insights,
    # and reporting the reservation would overstate spend.
    assert r["calls"] == store.execute(
        "SELECT count(*) n FROM dream_queue").fetchone()["n"] == 1


def _add_units(con, n, start=2):
    """Extra citable raw units, so a topic's backlog can reach the dig threshold.
    Indexed on the way out — a dig can only position itself on indexed content."""
    ids = []
    for i in range(start, start + n):
        uid = f"u{i}"
        con.execute("INSERT INTO units (unit_id,kind,source,title,updated_at,msg_count,"
                    "synced_at) VALUES (%s,'cc_session','claude_code',%s,now(),1,now())",
                    (uid, f"sess {i}"))
        con.execute("INSERT INTO messages (unit_id,msg_id,idx,sender,text,created_at) "
                    "VALUES (%s,'m1',0,'user',%s,'2026-01-01')",
                    (uid, "Fix bugs at the layer that should have prevented them."))
        ids.append(uid)
    con.commit()
    _reindex()
    return ids


def _triage_all(unit_ids, topic="t1"):
    return {"assignments": [{"unit_id": u, "topic_ids": [topic], "reason": "r"}
                            for u in unit_ids]}


def _distill_one(ref="a#0", statement="Fix bugs at the preventing layer."):
    """`ref` is a label render_windows() emitted: slot letter + message index."""
    return {"insights": [{"statement": statement, "elaboration": "e",
                          "stance": "user_asserted",
                          "evidence": [{"ref": ref, "quote": "Fix bugs at the layer"}]}],
            "probe_queries_to_add": []}


_UPHELD = {"verdicts": [{"index": 0, "verdict": "upheld",
                         "corrected_stance": "user_asserted", "reason": "ok"}]}


def test_a_thin_night_queues_instead_of_digging(store, stub_worker):
    """The nightly cost lever: cost scales with TOPICS DUG, not units changed, so a
    topic waits until its backlog is worth a dig. Measured on the real corpus, digging
    every flagged topic immediately cost 14 calls for one ordinary day."""
    clync.set_meta(store, dream.WATERMARK_KEY, "2020-01-01T00:00:00+00:00")
    store.commit()
    stub_worker.append(_triage_all(["u1"]))
    r = dream.run_incremental(store)
    assert r["calls"] == 1                      # triage only
    assert r["queued"] == 1 and r["digs"] == {}
    # and it is VISIBLE, not silently dropped
    assert r["deferred"] == [{"topic_id": "t1", "pending": 1,
                              "oldest": r["deferred"][0]["oldest"]}]
    assert store.execute("SELECT count(*) n FROM dream_pending").fetchone()["n"] == 1


def test_a_ripe_backlog_digs_and_reports_actual_calls(store, stub_worker,
                                                      pg_test_db, mock_embed):
    """Once DIG_MIN_UNITS accumulate the dig fires. A dig RESERVES 3 calls so it never
    starts a gate chain it can't finish, but a topic with no prior insights skips
    reconcile — so reported calls must come from the audit log, not the reservation."""
    ids = ["u1"] + _add_units(store, dream.DIG_MIN_UNITS - 1)
    clync.set_meta(store, dream.WATERMARK_KEY, "2020-01-01T00:00:00+00:00")
    store.commit()
    stub_worker += [_triage_all(ids), _distill_one(), _UPHELD,
                    {s: "" for s in dream.DIGEST_SECTIONS}]
    r = dream.run_incremental(store)
    # triage + distill + falsify + consolidate = 4; reconcile was skipped.
    # The budget reserved 5 (1 triage + 3 dig + 1 consolidate).
    assert r["calls"] == 4
    assert r["calls"] == store.execute(
        "SELECT count(*) n FROM dream_queue").fetchone()["n"]
    assert r["consolidated"] == ["t1"]
    assert not stub_worker, f"{len(stub_worker)} stub output(s) went unused"


def test_units_beyond_the_batch_cap_stay_pending_and_are_not_lost(store, stub_worker,
                                                                  pg_test_db, mock_embed):
    """The regression that motivated the queue: the dig takes at most
    DIG_BATCH_UNITS units, and the watermark advances regardless. Truncating the list
    without keeping the remainder meant those units were NEVER distilled."""
    ids = ["u1"] + _add_units(store, dream.DIG_BATCH_UNITS + 2)
    clync.set_meta(store, dream.WATERMARK_KEY, "2020-01-01T00:00:00+00:00")
    store.commit()
    stub_worker += [_triage_all(ids), _distill_one(), _UPHELD,
                    {s: "" for s in dream.DIGEST_SECTIONS}]
    dream.run_incremental(store)

    left = {r["unit_id"] for r in store.execute(
        "SELECT unit_id FROM dream_pending").fetchall()}
    assert len(left) == len(ids) - dream.DIG_BATCH_UNITS
    assert left == set(ids[dream.DIG_BATCH_UNITS:])   # the oldest were mined first


def test_a_capped_run_does_not_advance_the_watermark(store, stub_worker):
    """Otherwise the units the budget skipped would never be triaged again."""
    clync.set_meta(store, dream.WATERMARK_KEY, "2020-01-01T00:00:00+00:00")
    store.commit()
    r = dream.run_incremental(store, max_calls=0)
    assert r["stopped_early"] and clync.get_meta(store, dream.WATERMARK_KEY) == \
        "2020-01-01T00:00:00+00:00"


def test_an_empty_unit_is_skipped_visibly_never_wedging_the_run(store, stub_worker):
    """One abandoned claude.ai chat (a unit with no non-empty user turn) must not
    abort triage every night forever: emptiness is decided at SELECTION, the
    skip is REPORTED, and skeleton()'s raise stays reserved for a systemic
    mis-render of units that passed the predicate."""
    store.execute("INSERT INTO units (unit_id,kind,source,title,updated_at,"
                  "msg_count,synced_at) VALUES ('eu','chat','claude_ai',"
                  "'abandoned',now(),1,now())")
    store.execute("INSERT INTO messages (unit_id,msg_id,idx,sender,text) "
                  "VALUES ('eu','m1',0,'human','   ')")
    clync.set_meta(store, dream.WATERMARK_KEY, "2020-01-01T00:00:00+00:00")
    store.commit()
    stub_worker.append({"assignments": [{"unit_id": "u1", "topic_ids": [],
                                         "reason": "mechanical"}]})
    r = dream.run_incremental(store)          # raises if 'eu' had been offered
    assert r["skipped_empty"] == ["eu"]
    assert r["changed"] == 1                  # only u1 was eligible
    assert clync.get_meta(store, dream.WATERMARK_KEY) != "2020-01-01T00:00:00+00:00"


def test_selection_keys_on_when_clync_learned_of_a_unit(store):
    """`updated_at` is SOURCE-authored: NULL for project docs, months old for
    late-ingested material. Both are new TO CLYNC (synced_at=now) and must be
    eligible — comparing updated_at alone to the watermark hides them forever."""
    store.execute("INSERT INTO units (unit_id,kind,source,title,msg_count,synced_at)"
                  " VALUES ('doc1','project_doc','claude_ai','doc',1,now())")
    store.execute("INSERT INTO messages (unit_id,msg_id,idx,sender,text) "
                  "VALUES ('doc1','m1',0,'project_doc','the document body')")
    store.execute("INSERT INTO units (unit_id,kind,source,title,updated_at,"
                  "msg_count,synced_at) VALUES ('old1','chat','claude_ai',"
                  "'late ingest','2019-06-01',1,now())")
    store.execute("INSERT INTO messages (unit_id,msg_id,idx,sender,text) "
                  "VALUES ('old1','m1',0,'human','an old but newly synced turn')")
    store.commit()
    eligible, skipped = dream._changed_units(store, "2020-01-01T00:00:00+00:00")
    assert {"doc1", "old1", "u1"} <= set(eligible)
    assert skipped == []


def test_watermark_advances_to_the_pre_selection_snapshot(store, stub_worker,
                                                          monkeypatch):
    """The pass takes minutes of model calls; a unit synced in between was not
    selected, so the watermark must be the snapshot taken BEFORE selection —
    never end-of-run now() — or that unit is skipped silently forever."""
    clync.set_meta(store, dream.WATERMARK_KEY, "2020-01-01T00:00:00+00:00")
    store.commit()
    ticks = iter(f"2026-07-01T00:00:{i:02d}+00:00" for i in range(60))
    monkeypatch.setattr(dream, "_now", lambda: next(ticks))
    stub_worker.append({"assignments": [{"unit_id": "u1", "topic_ids": [],
                                         "reason": "mechanical"}]})
    dream.run_incremental(store)
    assert clync.get_meta(store, dream.WATERMARK_KEY) == \
        "2026-07-01T00:00:00+00:00"           # the FIRST tick, not a later one


def test_a_run_indexes_what_it_wrote(store, pg_test_db, mock_embed):
    """The scheduled job runs build_index BEFORE the dream pass, so a run that does
    not index its own output leaves tonight's insights unreachable through the
    insight tier until tomorrow — and QUIETLY, because the digest tier still
    answers. Retrieval must not lag writing by a day."""
    row = _persist_one(store)
    report = {"digs": {"t1": {"written": {"new": 1}}}, "consolidated": []}
    dream._index_written(store, report)
    assert report["indexed"]["reindexed_units"] >= 1
    # no manual build_index between writing and recalling
    found = dream.recall(store, "bugs layer prevented", topic_id="t1")["insights"]
    assert [i["unit_id"] for i in found] == [row["unit_id"]]


def test_a_run_that_wrote_nothing_does_not_index(store, pg_test_db, mock_embed):
    report = {"digs": {"t1": {"written": {"new": 0, "reinforce": 0}}}, "consolidated": []}
    dream._index_written(store, report)
    assert "indexed" not in report


# --------------------------------------------------------------------------- #
# Real model — the one test that spends quota
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_real_worker_returns_validated_structured_output():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}},
              "required": ["answer"], "additionalProperties": False}
    out, usage = dream._run_worker("Reply with the single word: pong.", schema,
                                   system="You answer tersely.")
    assert isinstance(out.get("answer"), str) and out["answer"]
    assert usage.get("output_tokens", 0) > 0
