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
    """Replace the model call with a queue of canned structured outputs. A queued
    EXCEPTION is raised instead of returned — how a test stages a throttled call."""
    outputs: list[dict] = []

    def _fake(prompt, schema, *, system, effort=None, model=dream.WORKER_MODEL,
              timeout=900):
        if not outputs:
            raise AssertionError("stub_worker ran out of queued outputs")
        out = outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out, {"input_tokens": 1, "output_tokens": 2}

    monkeypatch.setattr(dream, "_run_worker", _fake)
    return outputs


# The distiller cites SHORT slot#index labels, never UUIDs (see render_windows) —
# so a test supplies both the candidate and the refmap its labels resolve through.
REFMAP = {"a#0": ("u1", "m1"), "a#1": ("u1", "m2")}


def _candidate(**over):
    c = {"statement": "Fix bugs at the layer that should have prevented them.",
         "elaboration": "Root-cause rule.", "stance": "user_asserted",
         "evidence": [{"ref": "a#0", "quote": "Fix bugs at the layer that should have prevented"}]}
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
    assert cols == dream._declared_cols("dream_insights")
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
    from idx 0 up to a char budget put 42/45 = 93% of retrieval matches OUTSIDE the
    window the dreamer was shown (median match at idx 1284, median 23 messages
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


def test_unserved_read_requests_are_reported_never_silently_dropped(
        store, stub_worker, monkeypatch):
    """A request arriving in the final round cannot be served. Saying so is the only
    signal that MAX_NAV_ROUNDS is set too low; dropping it reads as 'the windows
    sufficed' when they did not. The cap is pinned locally: this pins the behaviour at
    the final-round boundary, not whatever the shipped budget currently happens to be.
    """
    monkeypatch.setattr(dream, "MAX_NAV_ROUNDS", 1)
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
    c = _candidate(evidence=[{"ref": "a#0", "quote": "  Fix   bugs\nat the layer that should  have prevented "}])
    assert len(dream.ground(store, c, REFMAP)) == 1


@pytest.mark.parametrize("evidence,match", [
    ([], "cites no evidence"),
    ([{"ref": "e99", "quote": "Fix bugs at the layer that should"}], "never shown to the distiller"),
    ([{"ref": "", "quote": "Fix bugs at the layer that should"}], "never shown to the distiller"),
    ([{"ref": "a#0", "quote": "a sentence that appears in no transcript"}], "not found verbatim"),
])
def test_ground_rejects_ungrounded_candidates(store, evidence, match):
    with pytest.raises(dream.DreamError, match=match):
        dream.ground(store, _candidate(evidence=evidence), REFMAP)


def test_ground_rejects_a_candidate_with_one_fabricated_reference(store):
    """Partly grounded is not grounded — it is untrustworthy."""
    c = _candidate(evidence=[{"ref": "a#0", "quote": "Fix bugs at the layer that should have prevented"},
                             {"ref": "ghost", "quote": "Fix bugs at the layer that should have prevented"}])
    with pytest.raises(dream.DreamError, match="never shown to the distiller"):
        dream.ground(store, c, REFMAP)


def test_ground_refuses_to_cite_derived_text(store):
    """Citing a dream unit would make the layer circular."""
    store.execute("INSERT INTO units (unit_id,kind,source,title,synced_at) "
                  "VALUES ('dr1',%s,'dream','an insight',now())", (dream.KIND_INSIGHT,))
    store.execute("INSERT INTO messages (unit_id,msg_id,idx,sender,text) "
                  "VALUES ('dr1','m1',0,'dream','Fix bugs at the layer')")
    store.commit()
    c = _candidate(evidence=[{"ref": "d1", "quote": "Fix bugs at the layer that should have prevented"}])
    with pytest.raises(dream.DreamError, match="non-raw unit"):
        dream.ground(store, c, {**REFMAP, "d1": ("dr1", "m1")})


def test_ground_dedupes_two_quotes_from_one_message(store):
    """(src_unit, src_msg) is the citation key AND the evidence PK."""
    c = _candidate(evidence=[{"ref": "a#0", "quote": "Fix bugs at the layer that should have prevented"},
                             {"ref": "a#0", "quote": "that should have prevented them"}])
    assert len(dream.ground(store, c, REFMAP)) == 1


def test_a_user_stance_must_cite_a_user_authored_turn(store):
    """The mechanical half of attribution: without it, the only things between an
    assistant turn and a 'user_asserted' insight are two LLM opinions — and
    passes_substance waives the two-source bar on exactly that stance."""
    c = _candidate(evidence=[{"ref": "a#1", "quote": "that is the root-cause rule"}])
    with pytest.raises(dream.DreamError, match="no user-authored turn"):
        dream.ground(store, c, REFMAP)                # e2 -> m2, an assistant turn
    ok = _candidate(stance="claude_proposed",
                    evidence=[{"ref": "a#1", "quote": "that is the root-cause rule"}])
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


def test_model_text_is_stripped_of_control_characters(store, stub_worker):
    """Postgres rejects NUL in a `text` column outright, so an unstripped one aborts
    the dig inside `persist` — after the quota is already spent. Other C0 controls
    survive into stored statements, corrupting the rendered digest and skewing the
    whitespace-normalised quote match. Tabs and newlines are real content, kept."""
    stub_worker.append({"insights": [{"statement": "A\x00 claim\x07 here",
                                      "elaboration": "tab\tand\nnewline kept",
                                      "stance": "user_asserted", "evidence": []}],
                        "probe_queries_to_add": [], "read_requests": []})
    got, _ = dream.distill(store, dream.get_topic(store, "t1"),
                           [{"unit_id": "u1", "msg_idx": 0}])
    assert got[0]["statement"] == "A claim here"
    assert got[0]["elaboration"] == "tab\tand\nnewline kept"


def test_a_quote_too_short_to_identify_a_message_is_rejected(store):
    """The gate had a prefix CAP but no minimum, so a two-word "quote" grounded an
    arbitrary claim: any real message contains "Postgres" or "cache", so the citation
    resolved, the substance gate was waived for user_asserted, and `--evidence`
    printed the fragment as though it proved the claim."""
    c = _candidate(statement="The user prefers Postgres over SQLite.",
                   evidence=[{"ref": "a#0", "quote": "Fix bugs"}])
    with pytest.raises(dream.DreamError, match="too short to ground"):
        dream.ground(store, c, REFMAP)


# --------------------------------------------------------------------------- #
# Gate 2 — an unjudged candidate must never be promoted by default
# --------------------------------------------------------------------------- #
def test_falsify_drops_rejected_and_keeps_contested(store, stub_worker):
    cands = [{**_candidate(), "_evidence": []},
             {**_candidate(statement="B"), "_evidence": []}]
    stub_worker.append({"verdicts": [
        {"index": 0, "verdict": "rejected", "corrected_stance": "user_asserted",
         "action": "new", "target": "", "reason": "generic"},
        {"index": 1, "verdict": "contested", "corrected_stance": "co_derived",
         "action": "new", "target": "", "reason": "two readings"}]})
    rejections: list[str] = []
    out = dream.falsify(store, dream.get_topic(store, "t1"), cands, rejections)
    assert [c["statement"] for c in out] == ["B"]
    assert out[0]["stance"] == "co_derived" and out[0]["verdict"] == "contested"
    # the REASON survives, not just the count — it is what tunes charters/prompts
    assert len(rejections) == 1 and "generic" in rejections[0]


def _held(con, statement="An existing claim.", unit_id="dream-held1"):
    """One active insight for this topic, so falsify has something to reconcile to."""
    con.execute("INSERT INTO units (unit_id,kind,source,title,synced_at) "
                "VALUES (%s,%s,'dream',%s,now())", (unit_id, dream.KIND_INSIGHT, statement))
    con.execute(
        "INSERT INTO dream_insights (unit_id,topic_id,statement,stance,status,"
        "support_count,contested,valid_from,first_seen_at,last_seen_at,distilled_at,"
        "model,evidence_sig) VALUES (%s,'t1',%s,'user_asserted','active',1,false,"
        "'2026-01-01','2026-01-01','2026-01-01','2026-01-01','opus','sig')",
        (unit_id, statement))
    con.commit()
    return unit_id


def test_falsify_resolves_its_reconcile_target_from_a_short_label(store, stub_worker):
    """The reconcile decision now rides on falsify's verdict — one call, since falsify
    already had to be shown the held insights. The target is a SHORT [hN] label and
    code owns the mapping: reconcile used to make the model copy a 36-char uuid
    verbatim, the same hazard that cost three insights in the distiller."""
    unit_id = _held(store)
    cands = [{**_candidate(), "_evidence": []}]
    stub_worker.append({"verdicts": [
        {"index": 0, "verdict": "upheld", "corrected_stance": "user_asserted",
         "action": "refine", "target": "h0", "reason": "sharper wording"}]})
    out = dream.falsify(store, dream.get_topic(store, "t1"), cands)
    assert out[0]["action"] == "refine"
    assert out[0]["target_id"] == unit_id, "the label must resolve to the real id"


def test_falsify_rejects_a_target_label_it_was_not_shown(store, stub_worker):
    _held(store)
    stub_worker.append({"verdicts": [
        {"index": 0, "verdict": "upheld", "corrected_stance": "user_asserted",
         "action": "contradict", "target": "h9", "reason": "changed position"}]})
    with pytest.raises(dream.DreamError, match="unknown target"):
        dream.falsify(store, dream.get_topic(store, "t1"),
                      [{**_candidate(), "_evidence": []}])


def test_with_nothing_held_the_action_is_forced_to_new(store, stub_worker):
    """No held insights means `new` is the only truthful action whatever the model
    says — mechanically, so a stray 'reinforce' can never reach persist and name a
    target that does not exist."""
    stub_worker.append({"verdicts": [
        {"index": 0, "verdict": "upheld", "corrected_stance": "user_asserted",
         "action": "reinforce", "target": "h0", "reason": "confused"}]})
    out = dream.falsify(store, dream.get_topic(store, "t1"),
                        [{**_candidate(), "_evidence": []}])
    assert out[0]["action"] == "new" and out[0]["target_id"] == ""


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
    c2 = _candidate(evidence=[{"ref": "e9", "quote": "Fix bugs at the layer that should have prevented"}])
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


def test_two_survivors_cannot_supersede_the_same_held_insight(store):
    """A dig that splits one held claim into two sharper atoms used to send both as
    `refine` on the same target: the second overwrote `superseded_by`, orphaning the
    first replacement, and the digest read "was A, now N2" while N1 sat in Settled as
    if independently established. With two `contradict`s, both stayed active and the
    digest presented contradictory claims as simultaneously current. A target can be
    superseded once; later claimants land as independent new rows."""
    target = _held(store, statement="One broad claim.")
    ev = [{"src_unit_id": "u1", "src_msg_id": "m1", "quote": "q", "sender": "user"}]
    decided = [{**_candidate(statement="Sharper atom one."), "_evidence": ev,
                "verdict": "upheld", "action": "refine", "target_id": target},
               {**_candidate(statement="Sharper atom two."), "_evidence": ev,
                "verdict": "upheld", "action": "refine", "target_id": target}]
    stats = dream.persist(store, dream.get_topic(store, "t1"), decided)

    assert stats["refine"] == 1 and stats["duplicate_target"] == 1
    assert stats["new"] == 1
    # exactly one row points at the target, and nothing is orphaned
    n = store.execute("SELECT count(*) AS n FROM dream_insights WHERE superseded_by="
                      "(SELECT superseded_by FROM dream_insights WHERE unit_id=%s)",
                      (target,)).fetchone()["n"]
    assert n == 1
    assert store.execute("SELECT status FROM dream_insights WHERE unit_id=%s",
                         (target,)).fetchone()["status"] == dream.STATUS_SUPERSEDED


def test_deleting_a_source_conversation_takes_its_evidence_with_it(store):
    """The user deletes a claude.ai conversation; the next sync removes its unit. The
    quotes drawn from it used to survive, so `dream recall --evidence` went on
    reprinting verbatim text from the deleted conversation, and the claim could no
    longer be re-checked by the gate that produced it."""
    _persist_one(store)
    assert store.execute("SELECT count(*) AS n FROM dream_evidence").fetchone()["n"] > 0
    store.execute("DELETE FROM units WHERE unit_id='u1'")     # the source, as sync does
    store.commit()
    assert store.execute("SELECT count(*) AS n FROM dream_evidence "
                         "WHERE src_unit_id='u1'").fetchone()["n"] == 0


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
                     "duplicate_target": 0,
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
def test_recall_rejects_uninterpretable_as_of_and_limit_with_advice(store):
    """The caller is usually a model, and a docstring saying "ISO date" invites
    `as_of='last week'`. Unvalidated it reached the bind and surfaced as psycopg's
    InvalidDatetimeFormat, which the MCP tool's handler cannot turn into a
    correction. It now fails with the correction."""
    for bad in ("last week", "2026-13-45", ""):
        with pytest.raises(dream.DreamError, match="ISO date"):
            dream.recall(store, "x", topic_id="t1", as_of=bad)
    for bad in (0, -1, "8"):
        with pytest.raises(dream.DreamError, match="positive integer"):
            dream.recall(store, "x", topic_id="t1", limit=bad)


def test_recall_states_coverage_gaps_loudly(store):
    r = dream.recall(store, "anything", topic_id="t1")
    assert any("no insights" in c for c in r["coverage"])
    assert any("never been dug" in c for c in r["coverage"])
    assert r["digest"] == [] and r["insights"] == []


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


def test_digest_is_rendered_from_rows_with_no_model_call_at_all(store):
    """The digest was a model call that wrote connective prose around bullets code had
    already assembled — the only ungrounded text in the store, precomputed for a
    question nobody had asked. It is now rendered from the rows at read time, so
    there is nothing to invent and nothing to go stale. `stub_worker` is deliberately
    absent: a model call here would raise for want of a stub."""
    _persist_one(store, statement="the only real claim")
    lines = dream.render_digest(store, "t1")
    assert any("the only real claim" in ln for ln in lines)
    assert sum(ln.count("    - ") for ln in lines) == 1   # one bullet, from the DB
    assert any("Settled" in ln for ln in lines)


def test_digest_is_empty_for_a_topic_with_no_insights(store):
    assert dream.render_digest(store, "t1") == []


def test_a_retired_stored_digest_unit_is_cleaned_up(store):
    """Digests used to be stored `units` rows written by a model call. Now they are
    rendered from the insight rows at read time, so a leftover row is an orphan
    nothing refreshes — and it would go on answering `search_history` with stale
    synthesis prose that no longer matches the insights."""
    store.execute("INSERT INTO units (unit_id,kind,source,title,summary,synced_at) "
                  "VALUES ('dream-digest-t1','dream_digest','dream','d','stale',now())")
    store.commit()
    dream.ensure_dream_schema()
    assert store.execute("SELECT count(*) n FROM units WHERE kind='dream_digest'"
                         ).fetchone()["n"] == 0


def test_probe_queries_are_capped_and_never_evict_the_authored_seeds(store):
    """`candidates` runs one hybrid search PER probe query, and every dig could add
    more while nothing removed any — a live topic reached 49 from 5 seeds. Past the
    cap the newest LEARNED queries displace the oldest, and the hand-written seeds
    are never displaced: they are the charter expressed as retrieval."""
    seeds = list(dream.SEED_TOPICS[0][3])
    tid = dream.SEED_TOPICS[0][0]
    store.execute("INSERT INTO dream_topics (topic_id,name,charter,probe_queries,"
                  "status,created_at) VALUES (%s,'n','c',%s,'active',now())",
                  (tid, clync._json(seeds)))
    store.commit()

    merged = dream.add_probe_queries(
        store, tid, [f"learned query {i}" for i in range(40)])
    assert len(merged) == dream.PROBE_QUERIES_MAX
    assert set(seeds) <= set(merged), "an authored seed query was evicted"
    # the survivors are the NEWEST learned ones, not the oldest
    assert "learned query 39" in merged and "learned query 0" not in merged


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
    # dig reserves 3 up front but only navigates when the dreamer asks,
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
                          "evidence": [{"ref": ref, "quote": "Fix bugs at the layer that should have prevented"}]}],
            "probe_queries_to_add": [], "read_requests": []}


def _attempted_ranges(con, topic_id) -> dict[str, list[tuple[int, int]]]:
    """The attempt record, per unit: the merged transcript ranges already rendered to
    the distiller for this topic — independent of whether any insight survived the
    gates. Production reads this table two ways, neither of them per-unit: retrieval
    excludes ranges in SQL (`hybrid_search(exclude_attempted=...)`) and `recall`
    aggregates coverage. This per-unit view exists only to assert on here."""
    out: dict[str, list[tuple[int, int]]] = {}
    for r in con.execute(
            "SELECT unit_id, from_idx, to_idx FROM dream_attempted "
            "WHERE topic_id=%s ORDER BY unit_id, from_idx", (topic_id,)).fetchall():
        out.setdefault(r["unit_id"], []).append((r["from_idx"], r["to_idx"]))
    return out


_UPHELD = {"verdicts": [{"index": 0, "verdict": "upheld",
                         "corrected_stance": "user_asserted",
                         "action": "new", "target": "", "reason": "ok"}]}


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
    starts a gate chain it can't finish, but it only spends the third when the dreamer
    asks to read further passages — so reported calls must come from the audit log,
    not the reservation.

    Also pins the nightly cost after the redesign: a ripe topic is triage + distill +
    falsify = 3 calls. It was 5 (reconcile was a separate call, and a digest was
    written per changed topic); reconcile was redundant with falsify's own input and
    the digest is now rendered at read time."""
    ids = ["u1"] + _add_units(store, dream.DIG_MIN_UNITS - 1)
    clync.set_meta(store, dream.WATERMARK_KEY, "2020-01-01T00:00:00+00:00")
    store.commit()
    stub_worker += [_triage_all(ids), _distill_one(), _UPHELD]
    r = dream.run_incremental(store)
    assert r["calls"] == 3
    assert r["calls"] == store.execute(
        "SELECT count(*) n FROM dream_queue").fetchone()["n"]
    assert r["digs"]["t1"]["written"] == {"new": 1, "reinforce": 0, "refine": 0,
                                          "contradict": 0, "out_of_order": 0,
                                          "duplicate_target": 0}
    assert not stub_worker, f"{len(stub_worker)} stub output(s) went unused"


def test_units_beyond_the_batch_cap_stay_pending_and_are_not_lost(store, stub_worker,
                                                                  pg_test_db, mock_embed):
    """The regression that motivated the queue: the dig takes at most
    DIG_BATCH_UNITS units, and the watermark advances regardless. Truncating the list
    without keeping the remainder meant those units were NEVER distilled."""
    ids = ["u1"] + _add_units(store, dream.DIG_BATCH_UNITS + 2)
    clync.set_meta(store, dream.WATERMARK_KEY, "2020-01-01T00:00:00+00:00")
    store.commit()
    stub_worker += [_triage_all(ids), _distill_one(), _UPHELD]
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
# Progress derives from the ATTEMPT, never from surviving gate output
# --------------------------------------------------------------------------- #
_EMPTY_DISTILL = {"insights": [], "probe_queries_to_add": [], "read_requests": []}


def test_a_fully_rejected_dig_still_records_the_attempt(store, stub_worker,
                                                        pg_test_db, mock_embed):
    """An empty insight list is an explicitly valid worker answer, and the gates
    reject whole batches by design — yet a dig that yielded nothing used to leave
    no trace: no last_dig_at (so coverage said "never been dug") and no exclusion
    record (so backfill re-bought the same batch forever)."""
    t = dream.get_topic(store, "t1")
    stub_worker.append(_EMPTY_DISTILL)
    dream.dig(store, t, [{"unit_id": "u1", "msg_idx": 0}])
    # ...and the record is the RANGE actually rendered, not a bare unit mark
    assert _attempted_ranges(store, "t1") == {"u1": [(0, 1)]}
    assert dream.get_topic(store, "t1")["last_dig_at"] is not None
    cov = dream.recall(store, "", topic_id="t1")["coverage"]
    assert not any("never been dug" in c for c in cov)
    assert any("no insights" in c for c in cov)   # honest, and distinct from undug


def test_backfill_excludes_attempted_ranges_so_a_rejected_batch_is_never_rebought(
        store, stub_worker, pg_test_db, mock_embed):
    """The re-dig loop: `candidates` is deterministic for a fixed index, so
    excluding only CITED units re-selected a fully-rejected batch at the same rank
    and re-dug it with the identical prompt until the budget expired."""
    stub_worker.append(_EMPTY_DISTILL)
    r = dream.run_backfill(store, topic_id="t1", max_calls=30)
    assert not stub_worker, "the same batch was dug more than once"
    assert len(r["digs"]["t1"]) == 1
    assert r["stopped_early"] is None             # pool exhausted, not budget
    assert _attempted_ranges(store, "t1") == {"u1": [(0, 1)]}
    # a later invocation resumes past the attempted ranges at zero model calls
    r2 = dream.run_backfill(store, topic_id="t1", max_calls=30)
    assert r2["digs"] == {} and r2["calls"] == 0


def test_a_throttled_call_leaves_no_phantom_attempt(store, stub_worker,
                                                    pg_test_db, mock_embed):
    """The attempt row is written only after the model call RETURNS. Written
    before it (as it once was), a throttled first call — the layer's one
    documented resumable failure — permanently excluded its whole batch from
    backfill without a single message ever having been distilled, while
    coverage showed the topic freshly dug with no gap."""
    stub_worker.append(RuntimeError("upstream says: rate limit exceeded"))
    with pytest.raises(dream.DreamThrottled):
        dream.run_backfill(store, topic_id="t1", max_calls=30)
    assert _attempted_ranges(store, "t1") == {}, "attempt recorded for a call that never ran"
    assert dream.get_topic(store, "t1")["last_dig_at"] is None
    # resuming re-selects the SAME batch and actually distills it this time
    stub_worker.append(_EMPTY_DISTILL)
    r = dream.run_backfill(store, topic_id="t1", max_calls=30)
    assert len(r["digs"]["t1"]) == 1
    assert _attempted_ranges(store, "t1") == {"u1": [(0, 1)]}


def _two_hit_unit(con, unit_id="wide", n=60, hits=(10, 50)):
    """A long unit that is on-topic at TWO distant positions — the shape a
    per-unit attempt record cannot represent. The two hit texts are distinct
    (identical texts embed identically, and the topk cut then breaks the tie
    arbitrarily) but both carry the probe's vocabulary."""
    # Texts stay ABOVE dream.MIN_DENSE_SIM under the hashed-BOW stub (short,
    # query-dense); the routine turns share no query vocabulary and fall below
    # it — which is what lets the unit retire once its genuine hits are covered.
    hit_text = {hits[0]: "root cause of failure at producer",
                hits[1]: "root cause lives where contract is owned"}
    con.execute("INSERT INTO units (unit_id,kind,source,title,updated_at,msg_count,"
                "synced_at) VALUES (%s,'cc_session','claude_code','wide',now(),%s,now())",
                (unit_id, n))
    for i in range(n):
        con.execute(
            "INSERT INTO messages (unit_id,msg_id,idx,sender,text,created_at) "
            "VALUES (%s,%s,%s,%s,%s,'2026-01-01')",
            (unit_id, f"m{i}", i, "user" if i % 2 == 0 else "assistant",
             hit_text.get(i, f"routine turn {i} regarding neutral matters")))
    con.commit()
    _reindex()


def test_one_window_does_not_retire_a_units_other_on_topic_positions(
        store, pg_test_db, mock_embed):
    """A dig reads a ~15-message window around one retrieval hit, so a unit-level
    attempt record marked a whole conversation mined on the strength of that
    window and nothing could ever revisit the rest. Exclusion is per HIT: the
    unit re-enters the pool at its next unread position, and it retires only
    when the recorded ranges cover its hits."""
    _two_hit_unit(store, hits=(10, 50))
    t = dream.get_topic(store, "t1")

    # per_query=2 keeps wide's pool to exactly its two genuine hits
    s0 = dream.candidates(store, t, per_query=2)
    assert s0[0]["unit_id"] == "wide", "the two-hit unit must rank first"
    first = s0[0]["msg_idx"]
    assert first in (10, 50)

    # the first dig's window covers only that hit -> the OTHER hit stays minable
    dream.record_attempt(store, "t1",
                         [("wide", first - dream.WINDOW_BEFORE,
                           first + dream.WINDOW_AFTER)])
    s1 = dream.candidates(store, t, per_query=2)
    other = 50 if first == 10 else 10
    assert [(s["unit_id"], s["msg_idx"]) for s in s1
            if s["unit_id"] == "wide"] == [("wide", other)]

    # once the ranges cover both hits, the unit finally retires
    dream.record_attempt(store, "t1",
                         [("wide", other - dream.WINDOW_BEFORE,
                           other + dream.WINDOW_AFTER)])
    s2 = dream.candidates(store, t, per_query=2)
    assert all(s["unit_id"] != "wide" for s in s2), s2

    # ...and recall's coverage states the unrendered remainder from the record
    cov = dream.recall(store, "", topic_id="t1")["coverage"]
    assert any("retrieval windows only" in c for c in cov), cov


def test_a_verbose_unit_cannot_monopolize_the_candidate_pool(
        store, pg_test_db, mock_embed):
    """Regression (review round 3): retrieval trimmed to topk CHUNKS before the
    attempted-range exclusion ran in Python, so one long conversation that is
    on-topic throughout occupied every slot — every other on-topic unit was
    structurally unreachable, and once the top chunks all fell inside recorded
    ranges the backfill misread its saturated window as an exhausted pool.
    Exclusion now lives in retrieval's SQL WHERE and rows are collapsed to one
    best UNREAD hit per unit BEFORE the cut, so topk spans distinct units and
    an empty pool genuinely means no unread hit anywhere."""
    con = store
    n = 40
    con.execute("INSERT INTO units (unit_id,kind,source,title,updated_at,msg_count,"
                "synced_at) VALUES ('verbose','cc_session','claude_code','verbose',"
                "now(),%s,now())", (n,))
    for i in range(n):    # on-topic at EVERY message
        con.execute(
            "INSERT INTO messages (unit_id,msg_id,idx,sender,text,created_at) "
            "VALUES ('verbose',%s,%s,%s,%s,'2026-01-01')",
            (f"m{i}", i, "user" if i % 2 == 0 else "assistant",
             f"root cause discussion continues at turn {i}"))
    shorts = {f"short{k}" for k in range(5)}
    for k, uid in enumerate(sorted(shorts)):
        con.execute("INSERT INTO units (unit_id,kind,source,title,updated_at,"
                    "msg_count,synced_at) VALUES (%s,'cc_session','claude_code',%s,"
                    "now(),1,now())", (uid, uid))
        con.execute("INSERT INTO messages (unit_id,msg_id,idx,sender,text,created_at) "
                    "VALUES (%s,'m0',0,'user',%s,'2026-01-01')",
                    (uid, f"root cause angle number {k}"))
    con.commit()
    _reindex()
    t = dream.get_topic(store, "t1")

    # one probe query's budget spans DISTINCT units, not one unit's chunks
    s0 = dream.candidates(store, t, per_query=10, limit=10)
    assert shorts | {"verbose", "u1"} <= {s["unit_id"] for s in s0}, s0

    # hand-walk the backfill loop: the pool must not report exhausted while any
    # unit still has an unread hit — every unit gets visited before it empties
    seen: set[str] = set()
    for _ in range(30):
        batch = dream.candidates(store, t, per_query=10, limit=10)
        if not batch:
            break
        for s in batch:
            seen.add(s["unit_id"])
            dream.record_attempt(store, "t1", [
                (s["unit_id"], s["msg_idx"] - dream.WINDOW_BEFORE,
                 s["msg_idx"] + dream.WINDOW_AFTER)])
    else:
        pytest.fail("candidate pool never drained")
    assert shorts | {"verbose", "u1"} <= seen, seen


def test_an_empty_rendering_records_no_attempt(store):
    dream.record_attempt(store, "t1", [])
    assert _attempted_ranges(store, "t1") == {}
    assert dream.get_topic(store, "t1")["last_dig_at"] is None


def test_a_derived_schema_rebuild_erases_the_progress_record_too(store):
    """last_dig_at is derived progress state (written with the attempt record);
    surviving a rebuild would make coverage claim digs that were just erased."""
    dream.record_attempt(store, "t1", [("u1", 0, 1)])
    store.execute("ALTER TABLE dream_insights ADD COLUMN legacy int")
    store.commit()
    dream.ensure_dream_schema(rebuild=True)
    assert _attempted_ranges(store, "t1") == {}
    assert dream.get_topic(store, "t1")["last_dig_at"] is None


# --------------------------------------------------------------------------- #
# Evidence loss — support and liveness re-derived from the surviving rows
# --------------------------------------------------------------------------- #
def test_evidence_loss_recounts_support_and_removes_ungrounded_insights(store):
    """Deleting a source unit cascades its evidence rows, but the insight, its
    units row and its index entry knew nothing of it: the claim stayed active with
    an overstated 'N source(s)'. reconcile_evidence is the one owner of the
    re-derivation, and recall runs it, so the stale state is unobservable there."""
    first = _persist_one(store)
    _add_units(store, 1, start=90)                       # u90, a second source
    c2 = _candidate(evidence=[{"ref": "e9",
                               "quote": "Fix bugs at the layer that should have prevented"}])
    c2["_evidence"] = dream.ground(store, c2, {**REFMAP, "e9": ("u90", "m1")})
    c2.update({"action": "reinforce", "target_id": first["unit_id"]})
    dream.persist(store, dream.get_topic(store, "t1"), [c2])
    assert store.execute("SELECT support_count FROM dream_insights"
                         ).fetchone()["support_count"] == 2

    store.execute("DELETE FROM units WHERE unit_id='u90'")   # as sync does
    store.commit()
    assert dream.reconcile_evidence(store) == {"removed": 0, "recounted": 1}
    assert store.execute("SELECT support_count FROM dream_insights"
                         ).fetchone()["support_count"] == 1

    store.execute("DELETE FROM units WHERE unit_id='u1'")    # the last grounding
    store.commit()
    r = dream.recall(store, "", topic_id="t1")               # recall reconciles
    assert r["insights"] == [] and r["digest"] == []
    assert store.execute("SELECT count(*) n FROM units WHERE unit_id=%s",
                         (first["unit_id"],)).fetchone()["n"] == 0


# --------------------------------------------------------------------------- #
# Quote verification — every stored/displayed evidence byte was matched
# --------------------------------------------------------------------------- #
def test_ground_rejects_a_genuine_prefix_with_an_invented_continuation(store):
    """Only the first 60 normalized chars used to be verified while the FULL quote
    was stored, shown to the falsifier and printed as verbatim evidence — so a
    genuine opening could carry an invented continuation through every gate."""
    c = _candidate(evidence=[{
        "ref": "a#0",
        "quote": "Fix bugs at the layer that should have prevented them. "
                 "Always add a compensating fallback afterwards."}])
    with pytest.raises(dream.DreamError, match="not found verbatim"):
        dream.ground(store, c, REFMAP)


# --------------------------------------------------------------------------- #
# Evidence authoring time — explicit fallback to the unit, never wall-clock now
# --------------------------------------------------------------------------- #
def test_evidence_authoring_time_falls_back_to_the_unit_never_the_clock(store):
    """Project docs are written with NULL message timestamps on real sync paths.
    Substituting now() fed the chronology gate its MOST PERMISSIVE value, letting
    a timestamp-less source supersede the genuinely current position."""
    store.execute("INSERT INTO units (unit_id,kind,source,title,created_at,"
                  "msg_count,synced_at) VALUES ('ud','project_doc','claude_ai',"
                  "'doc','2026-02-01',1,now())")
    store.execute("INSERT INTO messages (unit_id,msg_id,idx,sender,text,created_at) "
                  "VALUES ('ud','m1',0,'project_doc',"
                  "'A durable documented position on layered bug fixing.',NULL)")
    store.commit()
    c = _candidate(stance="co_derived", evidence=[{
        "ref": "d#0",
        "quote": "A durable documented position on layered bug fixing."}])
    ev = dream.ground(store, c, {"d#0": ("ud", "m1")})
    lo, hi = dream._evidence_span(store, ev)
    assert lo.date().isoformat() == "2026-02-01" and hi == lo


def test_evidence_with_no_authoring_time_at_all_is_rejected_not_defaulted(store):
    store.execute("INSERT INTO units (unit_id,kind,source,title,msg_count,synced_at)"
                  " VALUES ('un','chat','claude_ai','untimed',1,now())")
    store.execute("INSERT INTO messages (unit_id,msg_id,idx,sender,text,created_at) "
                  "VALUES ('un','m1',0,'human',"
                  "'An untimestamped but quotable position statement.',NULL)")
    store.commit()
    c = _candidate(stance="co_derived", evidence=[{
        "ref": "n#0", "quote": "An untimestamped but quotable position statement."}])
    with pytest.raises(dream.DreamError, match="no authoring time"):
        dream.ground(store, c, {"n#0": ("un", "m1")})
    # and the span computation refuses too (backstop for any other caller)
    with pytest.raises(dream.DreamError, match="bi-temporal"):
        dream._evidence_span(store, [{"src_unit_id": "un", "src_msg_id": "m1",
                                      "quote": "x"}])


# --------------------------------------------------------------------------- #
# Navigation — the refmap is a union across rounds, like the insights
# --------------------------------------------------------------------------- #
def test_distill_unions_refmaps_across_rounds(store, stub_worker, monkeypatch):
    """Round 1's rendering is NOT a superset of round 0's (the per-range budget is
    spent nearest-first over the MERGED range), so replacing the refmap each round
    made ground() reject already-paid round-0 insights as 'fabricated citation'
    when their message fell out of the final rendering."""
    maps = [{"a#0": ("u1", "m1")}, {"a#5": ("u1", "m2")}]   # round 1 lost a#0
    calls: list[int] = []

    def fake_render(con, seeds, expansions=None):
        m = maps[len(calls)]
        calls.append(1)
        return "body", dict(m), {"a": "u1"}

    monkeypatch.setattr(dream, "render_windows", fake_render)
    round0 = _candidate()                                    # cites a#0
    stub_worker += [
        {"insights": [round0], "probe_queries_to_add": [],
         "read_requests": [{"slot": "a", "from_idx": 0, "to_idx": 3, "why": "w"}]},
        _EMPTY_DISTILL,
    ]
    t = dream.get_topic(store, "t1")
    found, refmap = dream.distill(store, t, [{"unit_id": "u1", "msg_idx": 0}])
    assert set(refmap) == {"a#0", "a#5"}
    assert len(dream.ground(store, found[0], refmap)) == 1   # round-0 pay kept


# --------------------------------------------------------------------------- #
# Prompts — the citation-ref format is stated once and matches the renderer
# --------------------------------------------------------------------------- #
def test_distill_system_prompt_teaches_the_refs_the_renderer_emits(store):
    """The system prompt once taught [e<N>]/'e7' while the renderer emitted
    a#1042 — two contradicting instructions in one call, the wrong one carrying
    the DROP-the-insight consequence. The worked example must be a ref ground()
    could actually resolve, and the format must appear ONCE (system prompt only)."""
    import re
    _, refmap, _ = dream.render_windows(store, [{"unit_id": "u1", "msg_idx": 0}])
    ref_shape = re.compile(r"^[a-z]#\d+$")
    assert all(ref_shape.match(r) for r in refmap)
    example = re.search(r'e\.g\.\s+"([^"]+)"', dream._DISTILL_SYSTEM)
    assert example and ref_shape.match(example.group(1)), (
        "the system prompt's worked example is not a ref the renderer emits")
    assert "[<slot>#<index> sender]" in dream._DISTILL_SYSTEM
    user_prompt = dream._distill_prompt({"name": "n", "charter": "c"}, "b", 1, held=[])
    assert "#<index>" not in user_prompt and "e<N>" not in user_prompt


# --------------------------------------------------------------------------- #
# falsify — the held-insight listing is bounded (nearest first), never the world
# --------------------------------------------------------------------------- #
def _insert_insight(store, uid, statement, when="2026-01-01"):
    from datetime import datetime, timezone
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    dream._write_unit(store, uid, dream.KIND_INSIGHT, statement, statement, ts, ts)
    store.execute(
        "INSERT INTO dream_insights (unit_id,topic_id,statement,stance,status,"
        "support_count,contested,valid_from,first_seen_at,last_seen_at,"
        "distilled_at,model,evidence_sig) VALUES "
        "(%s,'t1',%s,'user_asserted','active',1,false,%s,%s,%s,%s,'opus','sig')",
        (uid, statement, when, when, when, when))
    store.commit()


def test_falsify_held_listing_is_bounded_to_the_insights_nearest_the_candidates(
        store, monkeypatch, pg_test_db, mock_embed):
    """Unbounded, the listing grew monotonically with the topic's insight count
    until the context limit made the topic permanently un-diggable. Bounded to
    the held insights nearest the candidates, the restate/refine/contradict
    decision keeps exactly the rows it can be about."""
    monkeypatch.setattr(dream, "FALSIFY_HELD_MAX", 5)
    for i in range(7):
        _insert_insight(store, f"dream-far{i}",
                        f"Knitting sweaters requires wool tension number {i}.")
    _insert_insight(store, "dream-near", "Fix bugs at the preventing layer.")
    _reindex()
    t = dream.get_topic(store, "t1")
    held = dream._held_for_falsify(store, t, [_candidate()])
    assert len(held) == 5
    assert "dream-near" in {h["unit_id"] for h in held}
    # under the cap, everything is shown — the bound only trims, never reorders
    monkeypatch.setattr(dream, "FALSIFY_HELD_MAX", 50)
    assert len(dream._held_for_falsify(store, t, [_candidate()])) == 8


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
