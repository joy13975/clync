"""The Dream layer: topic-driven knowledge distillation over clync's store (ADR 0004).

Turns raw transcript (claude.ai chats + Claude Code sessions) into *derived
knowledge*: topic-scoped **insight atoms** (one atomic claim, attributed and
cited) and a **structured digest** per topic. Both are stored as `units` rows
(`source='dream'`), so `search.py` indexes them with no second retrieval path.

Two levels, never recursive: insights derive only from raw transcript, digests
only from insight rows. Derived text is never input to another derivation — that
caps hallucination amplification at one hop.

Inference runs through the **local `claude` CLI under subscription OAuth** — no
Anthropic API key, no API billing. Every call is a pure function of its prompt:
no tools, no MCP, no settings, no session persistence (so the worker's own
transcripts can never be re-ingested by `sync-cc`).

Fail loud: a worker call that errors, or returns no validated `structured_output`,
raises. A candidate insight whose citations don't resolve to real text in
`messages` is REJECTED — that gate is a DB check, not an LLM judgement.

Two run modes, deliberately separate code paths (not one loop with a ratio knob):
`run_incremental` is the nightly pass — triage-gated, batched, only units changed
since the watermark, so a quiet day costs zero calls and it NEVER re-sweeps
history. `run_backfill` is the explicit bulk pass that mines history, and is where
the cost lives. Each indexes what it wrote before returning, or tonight's insights
would be unreachable through `recall`'s insight tier until the next `build_index`.

Module boundary (ADR 0003): this module owns topics, the dig loop, the worker and
the insight lifecycle. It *calls* `search.hybrid_search` and `clync.connect_pg`;
it never reimplements retrieval or the store.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timedelta, timezone

from clync import (RAW_SOURCES, _json, _now, connect_pg, get_meta, set_meta,
                   sql_statements)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DREAM_SOURCE = "dream"
KIND_INSIGHT = "dream_insight"
KIND_DIGEST = "dream_digest"

WORKER_MODEL = "opus"           # every knowledge judgement runs on Opus
TRIAGE_EFFORT = "low"           # triage is routing, not judgement

# Stance — the dialogue-attribution axis (ADR 0004 D5). Without it the layer
# cannot honestly answer "what do *I* think" and would attribute to the user
# things they argued against.
STANCES = ("user_asserted", "user_endorsed", "user_rejected",
           "co_derived", "claude_proposed")
TRUSTED_STANCES = ("user_asserted", "user_endorsed")

STATUS_ACTIVE, STATUS_SUPERSEDED, STATUS_RETRACTED = "active", "superseded", "retracted"

# `RAW_SOURCES` (imported from clync — one definition, ADR 0003) is what the dig
# is allowed to read. Enforced at the query, not by convention: that is what makes
# the layer non-circular.
#
# The two sources label the user's turns DIFFERENTLY — claude.ai says 'human',
# Claude Code says 'user' (verified against the live store). Hardcoding either one
# silently empties the skeleton for the other source's entire corpus, so this set
# is the SSOT for "a turn the user wrote".
USER_SENDERS = ("human", "user")

TRIAGE_SKELETON_CHARS = 1200     # per unit, for the batched triage call
CONDENSE_CHARS = 6000            # per unit, for a distill call
DIG_BATCH_UNITS = 6              # candidate units per distill call
# Nightly digs are batched ACROSS nights. Measured on the real corpus: six changed
# units triaged into four topics, and digging every flagged topic immediately cost
# ~16 calls for one ordinary day — several times the intended nightly budget, since
# cost scales with TOPICS TOUCHED, not units changed. So a topic waits until it has
# accumulated enough pending units to be worth a dig, or until it has waited too
# long. Nothing is lost by waiting: `dream_pending` holds the assignments.
WATERMARK_KEY = "dream_last_run_at"   # meta key: high-water mark of the nightly pass
DIG_MIN_UNITS = 3                # dig a topic once this many units are pending...
DIG_MAX_DEFER_DAYS = 7           # ...or this long since its oldest pending unit
QUOTE_MATCH_CHARS = 60           # prefix of a quote that must appear verbatim
# Relevance floor for the recall insight tier, on BGE-M3 dense cosine similarity
# (RRF scores are rank-only and always return topk rows, so they cannot express
# "nothing matches"). Calibrated 2026-07-26 on the live store's dream units:
# on-topic queries put their best hits at 0.51-0.63; off-topic probes top out at
# 0.41 ("best way to knit a sweater") and 0.31 ("zucchini sourdough hydration").
# Below the floor the tier is EMPTY and coverage says so — never a confident
# list of unrelated claims.
RECALL_MIN_SIM = 0.45


class DreamError(RuntimeError):
    """A Dream worker or gate failed. Always raised, never swallowed."""


# --------------------------------------------------------------------------- #
# Schema — this module owns its own tables, exactly as search.py owns the index
# --------------------------------------------------------------------------- #
# clync.py owns `units`/`messages` (the store); search.py owns `chunks` (it pins
# the vector dims); dream.py owns the tables below (it pins the stance/status
# vocabularies). Insights/digests themselves ARE `units` rows, so search indexes
# them with no second retrieval path.
DREAM_SCHEMA = """
CREATE TABLE IF NOT EXISTS dream_topics (
    topic_id      text PRIMARY KEY,
    name          text NOT NULL,
    charter       text NOT NULL,
    probe_queries jsonb NOT NULL,
    status        text NOT NULL,
    last_dig_at   timestamptz,
    created_at    timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS dream_insights (
    unit_id       text PRIMARY KEY REFERENCES units(unit_id) ON DELETE CASCADE,
    topic_id      text NOT NULL REFERENCES dream_topics(topic_id) ON DELETE CASCADE,
    statement     text NOT NULL,
    stance        text NOT NULL,
    status        text NOT NULL,
    superseded_by text REFERENCES units(unit_id) ON DELETE SET NULL,
    superseded_kind text,
    support_count int NOT NULL,
    contested     boolean NOT NULL,
    valid_from    timestamptz NOT NULL,
    valid_until   timestamptz,
    first_seen_at timestamptz NOT NULL,
    last_seen_at  timestamptz NOT NULL,
    distilled_at  timestamptz NOT NULL,
    model         text NOT NULL,
    evidence_sig  text NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dream_ins_topic  ON dream_insights(topic_id, status);
CREATE INDEX IF NOT EXISTS idx_dream_ins_stance ON dream_insights(stance);

CREATE TABLE IF NOT EXISTS dream_evidence (
    unit_id     text NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
    src_unit_id text NOT NULL,
    src_msg_id  text NOT NULL,
    quote       text NOT NULL,
    PRIMARY KEY (unit_id, src_unit_id, src_msg_id)
);
CREATE INDEX IF NOT EXISTS idx_dream_ev_src ON dream_evidence(src_unit_id);

CREATE TABLE IF NOT EXISTS dream_queue (
    item_id    text PRIMARY KEY,
    kind       text NOT NULL,
    topic_id   text,
    payload    jsonb NOT NULL,
    state      text NOT NULL,
    attempts   int NOT NULL,
    error      text,
    usage      jsonb,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dream_queue_state ON dream_queue(state, kind);

-- Triage assignments awaiting a dig. This table is why the nightly pass can be
-- both CHEAP and LOSSLESS: a unit triaged into a topic stays here until that topic
-- is actually mined, so a night may defer digging without dropping the unit, and
-- the watermark can advance safely. Without it, deferral (or the per-dig unit cap)
-- silently discards units the watermark has already moved past.
CREATE TABLE IF NOT EXISTS dream_pending (
    topic_id  text NOT NULL REFERENCES dream_topics(topic_id) ON DELETE CASCADE,
    unit_id   text NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
    queued_at timestamptz NOT NULL,
    PRIMARY KEY (topic_id, unit_id)
);
"""

# The column set `dream_insights` must have. A mismatch means the shipped schema
# moved on, and `CREATE TABLE IF NOT EXISTS` would silently keep the old shape
# until the first INSERT/SELECT blew up somewhere unrelated.
_INSIGHT_COLS = {
    "unit_id", "topic_id", "statement", "stance", "status", "superseded_by",
    "superseded_kind", "support_count", "contested", "valid_from", "valid_until",
    "first_seen_at", "last_seen_at", "distilled_at", "model", "evidence_sig",
}


def ensure_dream_schema() -> None:
    """Create the Dream tables (idempotent). Assumes `units`/`messages` exist —
    `clync.ensure_cluster` applies RAW_SCHEMA first, and the FKs need it.

    Like the search index (ADR 0003), Dream data is DERIVED and rebuildable — the
    insights re-derive from raw transcript by re-digging. So a stale `dream_insights`
    layout is dropped and recreated rather than migrated. `dream_topics` is spared:
    charters and probe queries are hand-edited config, not derived, and re-digging
    is what costs quota, not re-seeding topics."""
    with connect_pg() as con:
        have = {r["column_name"] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='dream_insights'").fetchall()}
        if have and have != _INSIGHT_COLS:
            con.execute("DROP TABLE IF EXISTS dream_evidence")
            con.execute("DROP TABLE IF EXISTS dream_insights")
            con.execute("DROP TABLE IF EXISTS dream_pending")
            # The watermark goes too. Keeping it would leave the nightly pass looking
            # at only units changed since a rebuild that threw everything away — so
            # the wiped history would never be re-mined, and the run would report
            # success while doing nothing. Clearing it makes the next `dream run`
            # behave as a first run: initialise, and say that backfill mines history.
            con.execute("DELETE FROM meta WHERE key=%s", (WATERMARK_KEY,))
            # The insight/digest `units` rows are part of the same derived layer.
            # Leaving them behind would orphan them — indexed and searchable, with
            # no insight row behind them (no stance, no citations, no lifecycle).
            con.execute("DELETE FROM units WHERE source=%s", (DREAM_SOURCE,))
        for stmt in sql_statements(DREAM_SCHEMA):
            con.execute(stmt)
        con.commit()


# --------------------------------------------------------------------------- #
# The worker: headless `claude -p` under subscription auth
# --------------------------------------------------------------------------- #
# Measured 2026-07-26 (claude 2.1.220), as total prefill (input + cache_creation +
# cache_read): these flags take per-call overhead from 76,068 tokens (bare
# `claude -p`) to 9,209, and replacing the system prompt takes it to a 515 floor.
# With this layer's real system prompts + schemas a call prefills ~1.6-2.1k.
# `--strict-mcp-config` +
# `--setting-sources ""` + `--disable-slash-commands` strip MCP tool schemas,
# user settings and skills; `--system-prompt` replaces Claude Code's own prompt.
# `--no-session-persistence` writes NO transcript under ~/.claude/projects, which
# is what stops `sync-cc` re-ingesting the worker's output. `--bare` is NOT usable
# here: its auth is strictly ANTHROPIC_API_KEY, never OAuth.
_WORKER_FLAGS = (
    "--output-format", "json",
    "--tools", "",
    "--strict-mcp-config",
    "--setting-sources", "",
    "--disable-slash-commands",
    "--no-session-persistence",
)


def _run_worker(prompt: str, schema: dict, *, system: str,
                effort: str | None = None, model: str = WORKER_MODEL,
                timeout: int = 900) -> tuple[dict, dict]:
    """One schema-constrained Opus call. Returns (structured_output, usage).

    Fails loud on every failure mode the CLI reports: non-zero exit, unparseable
    result JSON, `is_error`, a non-success `subtype`, or a missing/invalid
    `structured_output`. There is deliberately no partial-success path — a caller
    can never mistake a failed derivation for an empty one."""
    argv = ["claude", "-p", prompt, "--model", model,
            "--system-prompt", system, "--json-schema", json.dumps(schema),
            *_WORKER_FLAGS]
    if effort:
        argv += ["--effort", effort]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise DreamError(
            f"claude CLI exited {proc.returncode}: {(proc.stderr or proc.stdout)[:2000]}")
    try:
        res = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise DreamError(f"claude CLI returned non-JSON: {proc.stdout[:2000]}") from e
    if res.get("is_error") or res.get("subtype") != "success":
        raise DreamError(
            f"worker call failed: subtype={res.get('subtype')!r} "
            f"api_error_status={res.get('api_error_status')!r} "
            f"terminal_reason={res.get('terminal_reason')!r} "
            f"result={str(res.get('result'))[:1000]}")
    out = res.get("structured_output")
    if not isinstance(out, dict):
        raise DreamError(
            "worker returned no validated structured_output "
            f"(stop_reason={res.get('stop_reason')!r}); got {type(out).__name__}")
    return out, res.get("usage") or {}


# --------------------------------------------------------------------------- #
# Topics
# --------------------------------------------------------------------------- #
# Seeds, not hardcoded config: these are written to `dream_topics` by
# `clync dream topics --seed` and are editable rows thereafter. The charter is
# addressed to the distiller, so it reads as instructions, not description.
SEED_TOPICS = [
    ("development-principles", "Development principles",
     "How software should be built: root-cause vs symptom fixes, scope discipline, "
     "SSOT/DRY, fail-loud vs silent fallback, when to defer vs converge, testing "
     "philosophy, code review standards. IN: transferable rules and the reasoning "
     "behind them. OUT: one-off bug details, project-specific trivia.",
     ["root cause versus bandaid fix", "single source of truth duplication drift",
      "fail loud silent fallback exception swallowing",
      "scope discipline minimum code speculative abstraction",
      "when to defer work versus fix now", "what makes a test worth writing"]),
    ("writing-manners", "Writing manners",
     "How prose and technical writing should read: tone, concision, structure, "
     "what to lead with, jargon, formatting conventions, how to write for a reader "
     "with no context. IN: stated preferences and corrections about writing. "
     "OUT: the content of any particular document.",
     ["writing style tone concise verbose preference",
      "lead with the outcome bottom line first",
      "avoid jargon abbreviations arrow chains readability",
      "how to write a summary for someone with no context",
      "formatting markdown tables headings preference"]),
    ("architecture-thinking", "Ways to think about architecture",
     "How to reason about system design: choosing boundaries, what belongs in one "
     "place, when to split, coupling and cohesion, deriving config vs hardcoding, "
     "how to evaluate a design fork. IN: reusable reasoning patterns and design "
     "heuristics. OUT: the specifics of one system's layout.",
     ["architecture boundary module ownership decision",
      "design fork tradeoff simplest thing that works",
      "coupling cohesion abstraction premature",
      "derive from config versus hardcode", "schema design single store versus split"]),
    ("ai-frontier-research", "AI frontier research",
     "Frontier model capability, evaluation, agent architectures, scaling and "
     "training trends, published research and its implications. IN: findings, "
     "positions and open questions about where AI is going. OUT: routine tool usage.",
     ["frontier model capability benchmark evaluation",
      "agent architecture memory context research",
      "model scaling training trend implication",
      "reasoning effort thinking budget research finding"]),
    ("local-ai-research", "Local AI research",
     "Running models locally: embedders, quantization, inference runtimes, Apple "
     "Silicon/MPS, VRAM and throughput constraints, local-vs-hosted tradeoffs. "
     "IN: measured findings and hardware-shaped conclusions. OUT: cloud API usage.",
     ["local model inference quantization VRAM memory limit",
      "embedding model choice BGE local benchmark",
      "Apple Silicon MPS metal GPU throughput",
      "local versus hosted model tradeoff cost latency"]),
    ("claude-code-optimization", "Claude Code token economy & output quality",
     "Getting more out of Claude Code: token/cache economics, context hygiene, "
     "delegation and subagent patterns, prompt and skill design, what raises output "
     "quality. IN: measured effects and durable technique. OUT: one-off session chatter.",
     ["token economy cache read cost context hygiene",
      "delegate subagent model tier cheaper",
      "skill design prompt instruction following",
      "compaction context window management",
      "what improves Claude Code output quality"]),
]


def seed_topics(con) -> int:
    """Insert the seed topics that don't exist yet. Idempotent; never overwrites a
    charter or probe list the user has since edited."""
    n = 0
    for topic_id, name, charter, probes in SEED_TOPICS:
        r = con.execute(
            "INSERT INTO dream_topics "
            "(topic_id,name,charter,probe_queries,status,created_at) "
            "VALUES (%s,%s,%s,%s,'active',%s) ON CONFLICT (topic_id) DO NOTHING",
            (topic_id, name, charter, _json(probes), _now()))
        n += r.rowcount
    con.commit()
    return n


def topics(con, status: str | None = "active") -> list[dict]:
    q = "SELECT * FROM dream_topics"
    params: tuple = ()
    if status:
        q += " WHERE status=%s"
        params = (status,)
    return con.execute(q + " ORDER BY topic_id", params).fetchall()


def get_topic(con, topic_id: str) -> dict:
    row = con.execute("SELECT * FROM dream_topics WHERE topic_id=%s",
                      (topic_id,)).fetchone()
    if not row:
        raise DreamError(f"no such topic: {topic_id!r} "
                         f"(known: {[t['topic_id'] for t in topics(con, None)]})")
    return row


def add_probe_queries(con, topic_id: str, new: list[str]) -> list[str]:
    """Extend a topic's probe queries (the distiller may propose vocabulary the
    seeds missed — this is what makes each dig sharpen the next one's retrieval)."""
    cur = list(get_topic(con, topic_id)["probe_queries"])
    merged = cur + [q for q in new if q and q not in cur]
    if merged != cur:
        con.execute("UPDATE dream_topics SET probe_queries=%s WHERE topic_id=%s",
                    (_json(merged), topic_id))
        con.commit()
    return merged


# --------------------------------------------------------------------------- #
# Condensation — what the model actually reads
# --------------------------------------------------------------------------- #
# Tool stdout is ~80% of transcript bytes (ADR 0003) and carries no durable
# knowledge, so both renderings read `embed_text` (the tighter tier that already
# drops tool bodies) and fall back to `text` only when it is absent.
_TEXT = "COALESCE(m.embed_text, m.text)"


def _unit_header(u: dict) -> str:
    bits = [f"unit_id={u['unit_id']}", f"title={u['title'] or '(untitled)'}"]
    if u.get("repo"):
        bits.append(f"repo={u['repo']}")
    if u.get("project_name"):
        bits.append(f"project={u['project_name']}")
    when = u.get("updated_at") or u.get("created_at")
    if when:
        bits.append(f"when={when.date().isoformat()}")
    return " | ".join(bits)


def skeleton(con, unit_id: str) -> str:
    """Triage rendering: the unit's identity plus its intent-bearing turns. For a
    conversation that is the USER's turns (topic relevance is visible from what the
    user asked for, at a fraction of the tokens); for a `project_doc` there are no
    user turns and the document body *is* the signal.

    Fails loud on a unit that renders to nothing: routing a unit on its title alone
    is silent degradation, and it is how a whole source's corpus can go
    mis-triaged without a single error (the 'human' vs 'user' sender split did
    exactly that during development)."""
    u = con.execute("SELECT * FROM units WHERE unit_id=%s", (unit_id,)).fetchone()
    if not u:
        raise DreamError(f"no such unit: {unit_id!r}")
    if u["kind"] == "project_doc":
        where, params = "m.unit_id=%s", (unit_id,)
    else:
        where, params = "m.unit_id=%s AND m.sender = ANY(%s)", (unit_id, list(USER_SENDERS))
    rows = con.execute(
        f"SELECT {_TEXT} AS t FROM messages m WHERE {where} ORDER BY m.idx",
        params).fetchall()
    body = "\n".join(f"- {(r['t'] or '').strip()[:400]}"
                     for r in rows if (r["t"] or "").strip())
    if not body.strip():
        raise DreamError(
            f"unit {unit_id!r} (kind={u['kind']}, source={u['source']}) rendered an "
            f"empty triage skeleton from {len(rows)} row(s) — refusing to triage on "
            f"metadata alone")
    return f"{_unit_header(u)}\n{body[:TRIAGE_SKELETON_CHARS]}"


def condense(con, unit_ids: list[str]) -> tuple[str, dict[str, tuple[str, str]]]:
    """Render a batch of transcripts for the distiller. Returns (text, refmap).

    Every quotable message gets a SHORT label — `[e7 user]` — and `refmap` maps that
    label back to the real `(unit_id, msg_id)`. The model never sees or copies a UUID.

    This is deliberate, and it replaced copying `unit_id=… msg_id=…` pairs: measured
    on a real dig, the model was shown the correct 36-char UUID and wrote back
    `…ad4d227fceff` for `…ad4d277fceff` — one transposed digit. The grounding gate
    correctly rejected it, which meant THREE genuinely-grounded insights (20% of that
    dig) were thrown away over a copy error. Fuzzy-matching the id would have been the
    wrong fix: the nearest id is not necessarily the cited one, and a mis-resolved
    citation is worse than a rejected one. So the transcription burden is removed
    instead — a 2-3 character token is copyable, and an unknown label still fails
    loud."""
    refmap: dict[str, tuple[str, str]] = {}
    parts, n = [], 0
    for unit_id in unit_ids:
        u = con.execute("SELECT * FROM units WHERE unit_id=%s", (unit_id,)).fetchone()
        if not u:
            raise DreamError(f"no such unit: {unit_id!r}")
        rows = con.execute(
            f"SELECT m.msg_id, m.sender, {_TEXT} AS t FROM messages m "
            f"WHERE m.unit_id=%s AND {_TEXT} IS NOT NULL AND {_TEXT} != '' "
            "ORDER BY m.idx", (unit_id,)).fetchall()
        segs, used = [], 0
        for r in rows:
            t = (r["t"] or "").strip()
            if not t:
                continue
            n += 1
            ref = f"e{n}"
            seg = f"[{ref} {r['sender']}]\n{t[:1500]}"
            if used + len(seg) > CONDENSE_CHARS:
                n -= 1
                break
            refmap[ref] = (unit_id, r["msg_id"])
            segs.append(seg)
            used += len(seg)
        parts.append(f"{_unit_header(u)}\n\n" + "\n\n".join(segs))
    if not refmap:
        raise DreamError(f"nothing quotable in units {unit_ids} — refusing to distill")
    return "\n\n---\n\n".join(parts), refmap


# --------------------------------------------------------------------------- #
# Queue + usage audit
# --------------------------------------------------------------------------- #
def _enqueue(con, kind: str, payload: dict, topic_id: str | None = None) -> str:
    item_id = uuid.uuid4().hex
    con.execute(
        "INSERT INTO dream_queue "
        "(item_id,kind,topic_id,payload,state,attempts,created_at,updated_at) "
        "VALUES (%s,%s,%s,%s,'pending',0,%s,%s)",
        (item_id, kind, topic_id, _json(payload), _now(), _now()))
    con.commit()
    return item_id


def _finish(con, item_id: str, state: str, *, usage: dict | None = None,
            error: str | None = None) -> None:
    con.execute(
        "UPDATE dream_queue SET state=%s, usage=%s, error=%s, "
        "attempts=attempts+1, updated_at=%s WHERE item_id=%s",
        (state, _json(usage) if usage else None, error, _now(), item_id))
    con.commit()


def _call(con, kind: str, payload: dict, topic_id: str | None,
          prompt: str, schema: dict, *, system: str,
          effort: str | None = None) -> dict:
    """Run one worker call as a tracked queue item, persisting its `usage` block so
    real consumption is measurable rather than estimated. A failure marks the item
    `failed` with the error retained, then re-raises — the run stops, it never
    silently continues on partial state."""
    item_id = _enqueue(con, kind, payload, topic_id)
    try:
        out, usage = _run_worker(prompt, schema, system=system, effort=effort)
    except Exception as e:
        _finish(con, item_id, "failed", error=str(e)[:4000])
        raise
    _finish(con, item_id, "done", usage=usage)
    return out


# --------------------------------------------------------------------------- #
# Triage — ONE batched call routes every new/changed unit to topics
# --------------------------------------------------------------------------- #
_TRIAGE_SYSTEM = (
    "You route conversation transcripts to knowledge topics. You are given topic "
    "charters and a batch of transcript skeletons. For each unit, list the topics it "
    "carries DURABLE, TRANSFERABLE knowledge about — a stated principle, a considered "
    "position, a reusable technique, a decision and its reasoning.\n\n"
    "Be strict. Most working sessions are mechanical (fix a test, rerun CI, chase a "
    "typo) and carry no durable knowledge: assign them NO topics. Merely mentioning a "
    "topic's subject is not enough — the unit must contain something worth remembering "
    "after the task is finished. An empty list is the correct answer for most units."
)

_TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string"},
                    "topic_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["unit_id", "topic_ids", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assignments"],
    "additionalProperties": False,
}


def triage(con, unit_ids: list[str]) -> dict[str, list[str]]:
    """Route a batch of units to active topics in ONE call. Returns
    {unit_id: [topic_id, ...]}, omitting units assigned nothing.

    Batched deliberately: 10-20 units of skeletons is ~25k tokens, which is one
    call. One call per unit would be the single biggest waste in the nightly path.

    Hallucinated ids fail loud — a returned unit_id or topic_id that wasn't offered
    means the routing is not trustworthy, and silently dropping it would hide that."""
    if not unit_ids:
        return {}
    ts = topics(con)
    if not ts:
        raise DreamError("no active topics — run `clync dream topics --seed` first")
    charters = "\n\n".join(f"[{t['topic_id']}] {t['name']}\n{t['charter']}" for t in ts)
    skeletons = "\n\n---\n\n".join(skeleton(con, u) for u in unit_ids)
    prompt = (f"TOPICS\n{charters}\n\n=====\n\nUNITS ({len(unit_ids)})\n\n{skeletons}\n\n"
              f"=====\nReturn one assignment per unit, using the exact unit_id and "
              f"topic_id strings given above. Use an empty topic_ids list for units "
              f"with no durable knowledge.")
    out = _call(con, "triage", {"unit_ids": unit_ids}, None, prompt, _TRIAGE_SCHEMA,
                system=_TRIAGE_SYSTEM, effort=TRIAGE_EFFORT)

    valid_units, valid_topics = set(unit_ids), {t["topic_id"] for t in ts}
    result: dict[str, list[str]] = {}
    for a in out["assignments"]:
        if a["unit_id"] not in valid_units:
            raise DreamError(f"triage returned an unknown unit_id: {a['unit_id']!r}")
        bad = [t for t in a["topic_ids"] if t not in valid_topics]
        if bad:
            raise DreamError(f"triage invented topic_id(s) {bad} for {a['unit_id']!r}")
        if a["topic_ids"]:
            result[a["unit_id"]] = a["topic_ids"]
    # Every OFFERED unit needs a verdict, exactly as falsify/reconcile demand one
    # per candidate: an omitted unit is indistinguishable from "assigned nothing",
    # and the watermark then moves past it permanently.
    missing = valid_units - {a["unit_id"] for a in out["assignments"]}
    if missing:
        raise DreamError(f"triage returned no verdict for offered unit(s) "
                         f"{sorted(missing)} — an untriaged unit must never be "
                         f"treated as an unassigned one")
    return result


# --------------------------------------------------------------------------- #
# Distill — raw transcript -> candidate insights (cited)
# --------------------------------------------------------------------------- #
_DISTILL_SYSTEM = (
    "You distill durable knowledge from a user's own conversation transcripts into "
    "atomic, attributed, cited claims.\n\n"
    "The transcripts are a DIALOGUE between the user and an AI assistant. Attribution "
    "is the whole point — never blur who held a position:\n"
    "  user_asserted   the user stated it themselves\n"
    "  user_endorsed   the assistant proposed it and the user explicitly accepted it\n"
    "  user_rejected   the assistant proposed it and the user pushed back (KEEP these — "
    "what the user rejected, and why, is knowledge)\n"
    "  co_derived      reached jointly, no clean attribution\n"
    "  claude_proposed present in the transcript but never adjudicated by the user\n\n"
    "RULES\n"
    "1. One claim per insight. A statement that needs 'and' is usually two insights.\n"
    "2. Every insight MUST cite evidence: the short `ref` label of the message it came "
    "from — the token inside that message's [e<N> sender] marker, e.g. \"e7\" — plus a "
    "quote copied VERBATIM from that same message, character for character, no "
    "paraphrasing, no ellipsis, no cleanup. Use only labels that appear in the "
    "transcripts below. If you cannot cite a real label, DROP the insight — never "
    "invent or guess one.\n"
    "3. TRANSFERABLE, not situational. The test: would this still be true and useful "
    "on a different project next year? 'Broken infra blocked us so we fixed it first' "
    "is an event, not knowledge. 'Write good code' is too generic to be knowledge. "
    "'Fix bugs at the layer that should have prevented them, not where they surfaced' "
    "is knowledge. Respect the charter's OUT clause strictly.\n"
    "4. Do not invent. If the transcripts carry nothing durable for this topic, return "
    "an empty list. An empty list is a valid, useful answer."
)

_DISTILL_SCHEMA = {
    "type": "object",
    "properties": {
        "insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "elaboration": {"type": "string"},
                    "stance": {"type": "string", "enum": list(STANCES)},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ref": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["ref", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["statement", "elaboration", "stance", "evidence"],
                "additionalProperties": False,
            },
        },
        "probe_queries_to_add": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["insights", "probe_queries_to_add"],
    "additionalProperties": False,
}


def distill(con, topic: dict, unit_ids: list[str]
            ) -> tuple[list[dict], dict[str, tuple[str, str]]]:
    """One call per topic-batch -> (candidate insights, refmap). Candidates are
    ungrounded and unverified; the refmap resolves their `ref` labels to real ids.

    Also folds back any probe queries the distiller proposes, so vocabulary the
    seed queries missed sharpens the NEXT dig's retrieval."""
    if not unit_ids:
        return [], {}
    bodies, refmap = condense(con, unit_ids)
    prompt = (f"TOPIC: {topic['name']}\nCHARTER: {topic['charter']}\n\n=====\n\n"
              f"TRANSCRIPTS ({len(unit_ids)})\n\n{bodies}\n\n=====\n"
              f"Distill this topic's durable knowledge from the transcripts above. "
              f"Cite each insight with the `ref` label of the message it came from (the "
              f"token in its [e<N> sender] marker) plus a verbatim quote from that same "
              f"message. Also suggest any search phrases that would have found "
              f"this material but are not obvious from the topic name.")
    out = _call(con, "distill", {"topic_id": topic["topic_id"], "unit_ids": unit_ids},
                topic["topic_id"], prompt, _DISTILL_SCHEMA, system=_DISTILL_SYSTEM)
    if out.get("probe_queries_to_add"):
        add_probe_queries(con, topic["topic_id"], out["probe_queries_to_add"])
    return out["insights"], refmap


# --------------------------------------------------------------------------- #
# Gate 1: grounding — a DB check, not an LLM judgement
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return " ".join((s or "").split()).casefold()


# Stances that CLAIM something about what the user said or did. Each one must be
# backed by at least one user-authored turn — mechanically, below.
USER_STANCES = ("user_asserted", "user_endorsed", "user_rejected")


def _require_user_evidence(stance: str, evidence: list[dict], statement: str) -> None:
    """Mechanical attribution gate: a user_* stance must cite at least one turn the
    user actually wrote (`sender` rides on each verified evidence row from
    `ground`). Without this check the only things between an assistant turn and a
    'user_asserted' insight are two LLM opinions — and `passes_substance` waives
    the two-source bar on exactly that stance."""
    if stance in USER_STANCES and not any(
            e.get("sender") in USER_SENDERS for e in evidence):
        senders = sorted({str(e.get("sender")) for e in evidence})
        raise DreamError(
            f"stance {stance!r} cites no user-authored turn "
            f"(cited senders: {senders}) for {statement[:60]!r}")


def ground(con, candidate: dict, refmap: dict[str, tuple[str, str]]) -> list[dict]:
    """Resolve a candidate's citations against `messages`. Returns the verified
    evidence rows, or raises `DreamError` naming exactly what failed.

    Every check is mechanical and cannot be argued with by a model:
      * the `ref` label must be one the distiller was actually shown (`refmap`);
      * the message it resolves to must EXIST and belong to a RAW source — citing a
        dream unit would make the layer circular;
      * the quote must appear VERBATIM (whitespace-normalised) in that message.
    Every citation must resolve. A candidate with one bad reference is not partly
    grounded, it is untrustworthy."""
    ev = candidate.get("evidence") or []
    if not ev:
        raise DreamError(f"candidate cites no evidence: {candidate['statement'][:80]!r}")
    # A message either supports the claim or it doesn't, so the citation key is
    # (src_unit_id, src_msg_id) — that is also `dream_evidence`'s PK. The distiller
    # may legitimately pull two quotes from one message; dedupe HERE, where evidence
    # is normalised, after verifying every quote. Deduping at INSERT instead would
    # discard already-verified data one layer too late and hide the duplication.
    verified, seen = [], set()
    for raw in ev:
        ref = (raw.get("ref") or "").strip()
        if ref not in refmap:
            raise DreamError(
                f"fabricated citation: ref {ref!r} was never shown to the distiller "
                f"for {candidate['statement'][:60]!r}")
        src_unit_id, src_msg_id = refmap[ref]
        row = con.execute(
            f"SELECT {_TEXT} AS t, m.sender, u.source "
            "FROM messages m JOIN units u USING(unit_id) "
            "WHERE m.unit_id=%s AND m.msg_id=%s",
            (src_unit_id, src_msg_id)).fetchone()
        if not row:
            raise DreamError(
                f"citation ref {ref!r} resolves to a missing message "
                f"({src_unit_id}, {src_msg_id}) for {candidate['statement'][:60]!r}")
        if row["source"] not in RAW_SOURCES:
            raise DreamError(
                f"citation points at a non-raw unit (source={row['source']}) — the "
                f"dig must never read derived text")
        e = {"src_unit_id": src_unit_id, "src_msg_id": src_msg_id,
             "quote": raw["quote"], "sender": row["sender"]}
        needle = _norm(e["quote"])[:QUOTE_MATCH_CHARS]
        if not needle or needle not in _norm(row["t"]):
            raise DreamError(
                f"quote not found verbatim in ({e['src_unit_id']}, {e['src_msg_id']}): "
                f"{e['quote'][:80]!r}")
        key = (e["src_unit_id"], e["src_msg_id"])
        if key not in seen:
            seen.add(key)
            verified.append(e)
    # The stance-vs-authorship gate runs on the DISTILLER's stance here, and again
    # in `dig` after falsify (which may rewrite the stance) — both directions of
    # the LLM opinion are checked against the same mechanical fact.
    _require_user_evidence(candidate["stance"], verified, candidate["statement"])
    return verified


# --------------------------------------------------------------------------- #
# Gate 2: falsification — an independent call whose job is to REFUTE
# --------------------------------------------------------------------------- #
_FALSIFY_SYSTEM = (
    "You are a skeptical auditor of candidate knowledge claims. For each candidate you "
    "are given the claim, its attributed stance, and the VERBATIM source quotes it "
    "rests on. Your job is to REFUTE, not to agree.\n\n"
    "Reject a candidate when any of these hold:\n"
    "  * it is SITUATIONAL rather than transferable — a narration of what happened on "
    "one task, or something the topic charter's OUT clause excludes. Apply this test: "
    "would the claim still be true and useful on an unrelated project a year from now? "
    "An event ('the serving stack went down so we diagnosed it first'), a "
    "project-specific implementation detail, or a restatement of one session's plan is "
    "NOT knowledge, however confidently it is phrased. This is the most common failure "
    "— be harsh here;\n"
    "  * the quotes do not actually support the claim;\n"
    "  * the stance is wrong (e.g. claimed as the user's own position when the quote "
    "shows the assistant proposing it, or the user rejecting it);\n"
    "  * it is too generic to be knowledge ('write tests', 'be careful');\n"
    "  * it merely restates something in the EXISTING insights you are shown.\n\n"
    "Mark a claim `contested` (rather than rejecting) when the evidence genuinely "
    "supports competing readings. Default to rejection when uncertain — a rejected "
    "candidate costs nothing, a wrong one poisons the knowledge base."
)

_FALSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {"type": "string",
                                "enum": ["upheld", "rejected", "contested"]},
                    "corrected_stance": {"type": "string", "enum": list(STANCES)},
                    "reason": {"type": "string"},
                },
                "required": ["index", "verdict", "corrected_stance", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def falsify(con, topic: dict, candidates: list[dict], rejections: list | None = None
            ) -> list[dict]:
    """Batched adversarial pass. Returns the surviving candidates, each carrying
    `verdict` ('upheld'|'contested') and a possibly-corrected `stance`.

    Rejection reasons are appended to `rejections` when given. The falsifier's own
    reasoning is the most useful signal there is for tuning a charter or a prompt,
    and a bare reject COUNT throws it away."""
    if not candidates:
        return []
    existing = [r["statement"] for r in con.execute(
        "SELECT statement FROM dream_insights WHERE topic_id=%s AND status=%s",
        (topic["topic_id"], STATUS_ACTIVE)).fetchall()]
    listing = "\n\n".join(
        f"[{i}] CLAIM: {c['statement']}\n    stance: {c['stance']}\n    quotes:\n"
        + "\n".join(f"      - {e['quote'][:300]}" for e in c["evidence"])
        for i, c in enumerate(candidates))
    prompt = (f"TOPIC: {topic['name']}\nCHARTER: {topic['charter']}\n\n"
              f"EXISTING INSIGHTS (a candidate that merely restates one of these must be "
              f"rejected):\n" + ("\n".join(f"  - {s}" for s in existing) or "  (none)")
              + f"\n\n=====\n\nCANDIDATES ({len(candidates)})\n\n{listing}\n\n=====\n"
              f"Return exactly one verdict per candidate index.")
    out = _call(con, "falsify", {"topic_id": topic["topic_id"], "n": len(candidates)},
                topic["topic_id"], prompt, _FALSIFY_SCHEMA, system=_FALSIFY_SYSTEM)

    by_index = {v["index"]: v for v in out["verdicts"]}
    missing = set(range(len(candidates))) - set(by_index)
    if missing:
        raise DreamError(f"falsify skipped candidate index(es) {sorted(missing)} — "
                         f"an unjudged candidate must never be promoted by default")
    survivors = []
    for i, c in enumerate(candidates):
        v = by_index[i]
        if v["verdict"] == "rejected":
            if rejections is not None:
                rejections.append(f"falsify: {c['statement'][:70]} <- {v['reason']}")
            continue
        survivors.append({**c, "stance": v["corrected_stance"],
                          "verdict": v["verdict"], "verdict_reason": v["reason"]})
    return survivors


# --------------------------------------------------------------------------- #
# Gate 3: substance — mechanical
# --------------------------------------------------------------------------- #
def passes_substance(candidate: dict, evidence: list[dict]) -> bool:
    """Blocks single-mention noise without discarding a clearly-stated one-off
    principle: >=2 distinct source units, OR the user asserted it outright."""
    return (len({e["src_unit_id"] for e in evidence}) >= 2
            or candidate["stance"] == "user_asserted")


# --------------------------------------------------------------------------- #
# Reconcile — how a survivor meets what we already believe
# --------------------------------------------------------------------------- #
_RECONCILE_SYSTEM = (
    "You decide how each new knowledge claim relates to a set of claims already held.\n\n"
    "  new         nothing already held covers it\n"
    "  reinforce   an existing claim says the same thing; this is additional evidence "
    "for it, not a change\n"
    "  refine      an existing claim says the same thing but this wording is sharper or "
    "more complete; it should REPLACE that one\n"
    "  contradict  the user's position CHANGED — this claim is incompatible with an "
    "existing one, and this one is the current view\n\n"
    "`contradict` is for a genuine change of mind, not for two claims that merely differ "
    "in emphasis or scope — those are usually both true and independent (`new`). "
    "For reinforce / refine / contradict you MUST name the existing insight's id."
)

_RECONCILE_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "action": {"type": "string",
                               "enum": ["new", "reinforce", "refine", "contradict"]},
                    "target_id": {"type": "string"},   # "" for new
                    "reason": {"type": "string"},
                },
                "required": ["index", "action", "target_id", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


def reconcile(con, topic: dict, survivors: list[dict]) -> list[dict]:
    """Decide new / reinforce / refine / contradict per survivor. Returns the
    survivors annotated with `action` + `target_id`."""
    if not survivors:
        return []
    existing = con.execute(
        "SELECT unit_id, statement, stance, valid_from FROM dream_insights "
        "WHERE topic_id=%s AND status=%s ORDER BY valid_from",
        (topic["topic_id"], STATUS_ACTIVE)).fetchall()
    if not existing:                     # nothing to reconcile against — all new
        return [{**c, "action": "new", "target_id": ""} for c in survivors]

    # The held-since date is shown so the model is not judging recency blind —
    # `contradict` asserts "this one is the current view", which is a temporal
    # claim (persist still gates chronology mechanically; this improves the input).
    held = "\n".join(
        f"  [{r['unit_id']}] ({r['stance']}, held since "
        f"{r['valid_from'].date().isoformat()}) {r['statement']}"
        for r in existing)
    listing = "\n".join(f"  [{i}] ({c['stance']}) {c['statement']}"
                        for i, c in enumerate(survivors))
    prompt = (f"TOPIC: {topic['name']}\n\nALREADY HELD\n{held}\n\n=====\n\n"
              f"NEW CLAIMS\n{listing}\n\n=====\n"
              f"Return exactly one decision per new-claim index. Use the bracketed ids "
              f"verbatim as target_id; use an empty string for `new`.")
    out = _call(con, "reconcile", {"topic_id": topic["topic_id"], "n": len(survivors)},
                topic["topic_id"], prompt, _RECONCILE_SCHEMA, system=_RECONCILE_SYSTEM)

    valid = {r["unit_id"] for r in existing}
    by_index = {d["index"]: d for d in out["decisions"]}
    missing = set(range(len(survivors))) - set(by_index)
    if missing:
        raise DreamError(f"reconcile skipped index(es) {sorted(missing)}")
    annotated = []
    for i, c in enumerate(survivors):
        d = by_index[i]
        if d["action"] != "new" and d["target_id"] not in valid:
            raise DreamError(
                f"reconcile named an unknown target_id {d['target_id']!r} for "
                f"action={d['action']} on {c['statement'][:60]!r}")
        annotated.append({**c, "action": d["action"], "target_id": d["target_id"],
                          "reconcile_reason": d["reason"]})
    return annotated


# --------------------------------------------------------------------------- #
# Persistence — insights ARE units (so search.py indexes them for free)
# --------------------------------------------------------------------------- #
def _evidence_sig(evidence: list[dict]) -> str:
    h = hashlib.sha256()
    for e in sorted(evidence, key=lambda e: (e["src_unit_id"], e["src_msg_id"])):
        h.update(f"{e['src_unit_id']}\x00{e['src_msg_id']}\x00{e['quote']}\x00"
                 .encode("utf-8"))
    return h.hexdigest()


def _evidence_span(con, evidence: list[dict]) -> tuple[datetime, datetime]:
    """When the cited material was AUTHORED — the insight's real temporal extent
    (distinct from when we happened to derive it)."""
    rows = con.execute(
        "SELECT min(m.created_at) AS lo, max(m.created_at) AS hi FROM messages m "
        "JOIN unnest(%s::text[], %s::text[]) AS p(u, g) "
        "  ON m.unit_id = p.u AND m.msg_id = p.g",
        ([e["src_unit_id"] for e in evidence],
         [e["src_msg_id"] for e in evidence])).fetchone()
    now = datetime.now(timezone.utc)
    return (rows["lo"] or now), (rows["hi"] or now)


def _write_unit(con, unit_id: str, kind: str, title: str, body: str,
                created: datetime, updated: datetime) -> None:
    """Write the `units` + `messages` pair that makes a dream row retrievable.
    `search._load_chunks` joins messages, so a unit with no message row would be
    stored but never indexed — silently unfindable."""
    con.execute(
        "INSERT INTO units (unit_id,kind,source,title,summary,lang,created_at,"
        "updated_at,msg_count,synced_at) VALUES (%s,%s,%s,%s,%s,'en',%s,%s,1,%s) "
        "ON CONFLICT (unit_id) DO UPDATE SET title=EXCLUDED.title, "
        "summary=EXCLUDED.summary, updated_at=EXCLUDED.updated_at, "
        "synced_at=EXCLUDED.synced_at",
        (unit_id, kind, DREAM_SOURCE, title, body, created, updated, _now()))
    con.execute(
        "INSERT INTO messages (unit_id,msg_id,idx,sender,text,created_at) "
        "VALUES (%s,%s,0,%s,%s,%s) ON CONFLICT (unit_id,msg_id) DO UPDATE SET "
        "text=EXCLUDED.text",
        (unit_id, f"{unit_id}:0", DREAM_SOURCE, f"{title}\n\n{body}", updated))


def persist(con, topic: dict, decided: list[dict]) -> dict:
    """Apply reconcile decisions. Bi-temporal and append-only: a superseded insight
    keeps its row, gets its validity window CLOSED, and points at its replacement —
    so "when did I change my mind about X" stays answerable.

    Reads each candidate's verified evidence from `_evidence` (attached by the
    grounding gate) — never from a caller-supplied parallel structure."""
    stats = {"new": 0, "reinforce": 0, "refine": 0, "contradict": 0,
             "out_of_order": 0}
    now = datetime.now(timezone.utc)
    for c in decided:
        ev = c["_evidence"]
        lo, hi = _evidence_span(con, ev)
        action = c["action"]
        if action == "reinforce":
            for e in ev:
                con.execute(
                    "INSERT INTO dream_evidence (unit_id,src_unit_id,src_msg_id,quote) "
                    "VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (c["target_id"], e["src_unit_id"], e["src_msg_id"], e["quote"]))
            # support_count is DERIVED from the evidence, never incremented: it is
            # surfaced as "N source(s)", and re-mining the same unit (a re-touched
            # conversation, a re-run after stopped_early) must not inflate it —
            # counting distinct cited units makes the label true by construction.
            con.execute(
                "UPDATE dream_insights SET support_count="
                "(SELECT count(DISTINCT src_unit_id) FROM dream_evidence "
                " WHERE unit_id=%s), "
                "last_seen_at=greatest(last_seen_at,%s), evidence_sig=%s "
                "WHERE unit_id=%s",
                (c["target_id"], hi, _evidence_sig(ev), c["target_id"]))
            stats["reinforce"] += 1
            continue

        if action in ("refine", "contradict"):
            # Chronology gate: superseding demands the NEW evidence not predate the
            # target's window start. Reconcile decides the kind semantically —
            # "this one is the current view" is a temporal claim it cannot verify —
            # so mining an OLD conversation after a newer one would otherwise
            # retire the current position and leave the target an INVERTED window
            # (valid_until < valid_from), invisible at every as_of date. Strictly
            # older contradicting evidence is still information: it lands as a
            # contested NEW row (the digest's Open section), never as a timeline
            # inversion. Equal timestamps (re-mining the same material into a
            # sharper wording) supersede normally — the window stays well-formed.
            target = con.execute(
                "SELECT valid_from FROM dream_insights WHERE unit_id=%s",
                (c["target_id"],)).fetchone()
            if not target:
                raise DreamError(f"persist: {action} names a missing target "
                                 f"{c['target_id']!r}")
            if hi < target["valid_from"]:
                action = "new"
                c = {**c, "verdict": "contested"}
                stats["out_of_order"] += 1

        new_id = f"dream-{uuid.uuid4().hex[:16]}"
        # Stance rides IN the persisted text (title and body), because every
        # surface that reaches an insight through units/messages — search_history
        # snippets, `clync search/list`, get_conversation — would otherwise show
        # the bare claim with nothing saying who held it: a user_rejected approach
        # would read as the user's own. Stance is immutable after persist, so the
        # text cannot go stale (support_count, which changes, stays out of it).
        _write_unit(con, new_id, KIND_INSIGHT,
                    f"[{topic['topic_id']}] ({c['stance']}) {c['statement']}",
                    f"({c['stance']}) {c['elaboration']}", lo, hi)
        con.execute(
            "INSERT INTO dream_insights (unit_id,topic_id,statement,stance,status,"
            "support_count,contested,valid_from,valid_until,first_seen_at,last_seen_at,"
            "distilled_at,model,evidence_sig) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s)",
            (new_id, topic["topic_id"], c["statement"], c["stance"], STATUS_ACTIVE,
             len({e["src_unit_id"] for e in ev}),
             c.get("verdict") == "contested", hi, lo, hi, now, WORKER_MODEL,
             _evidence_sig(ev)))
        for e in ev:
            con.execute(
                "INSERT INTO dream_evidence (unit_id,src_unit_id,src_msg_id,quote) "
                "VALUES (%s,%s,%s,%s)",
                (new_id, e["src_unit_id"], e["src_msg_id"], e["quote"]))
        if action in ("refine", "contradict"):
            # Supersession ALWAYS closes the window and records WHY in
            # `superseded_kind`. A NULL valid_until would satisfy every as_of
            # window forever (the retired wording returned beside its replacement),
            # and consumers must read the recorded kind — never infer it from a
            # NULL timestamp. `refine` is better wording of the SAME position, so
            # only `contradict` is a dated change of mind (the digest's `changed`
            # bucket keys on the kind, not on supersession itself).
            con.execute(
                "UPDATE dream_insights SET status=%s, superseded_by=%s, "
                "superseded_kind=%s, valid_until=%s WHERE unit_id=%s",
                (STATUS_SUPERSEDED, new_id, action, hi, c["target_id"]))
        stats[action] += 1
    con.commit()
    return stats


# --------------------------------------------------------------------------- #
# The dig — retrieval-driven, over the WHOLE corpus (no lookback window)
# --------------------------------------------------------------------------- #
def cited_units(con, topic_id: str) -> set[str]:
    """Source units this topic has already drawn evidence from."""
    return {r["src_unit_id"] for r in con.execute(
        "SELECT DISTINCT e.src_unit_id FROM dream_evidence e "
        "JOIN dream_insights i USING(unit_id) WHERE i.topic_id=%s",
        (topic_id,)).fetchall()}


def candidates(con, topic: dict, *, since: str | None = None,
               exclude: set[str] | None = None, limit: int = DIG_BATCH_UNITS,
               per_query: int = 10) -> list[str]:
    """Candidate source units for a topic, via the EXISTING hybrid index — the dig
    reuses `search.hybrid_search` rather than growing a second retrieval path.

    Unioned across the topic's probe queries and ranked by how many probes found a
    unit (a unit several probes agree on is more on-topic than one a single query
    surfaced). `since` gives the fresh slate; `exclude` skips already-mined units."""
    import search
    exclude = exclude or set()
    hits: dict[str, float] = {}
    for q in topic["probe_queries"]:
        for r in search.hybrid_search(q, topk=per_query, source="all", since=since):
            if r["unit_id"] not in exclude:
                hits[r["unit_id"]] = hits.get(r["unit_id"], 0.0) + r["score"]
    return [u for u, _ in sorted(hits.items(), key=lambda kv: -kv[1])][:limit]


def dig(con, topic: dict, unit_ids: list[str]) -> dict:
    """The full pipeline for one topic-batch: distill -> ground -> falsify ->
    substance -> reconcile -> persist. Returns per-stage counts.

    Rejections are COUNTED and reported, never silently absorbed: a high reject
    rate is information about prompt or gate quality, not noise to hide."""
    report = {"units": len(unit_ids), "candidates": 0, "rejected_grounding": 0,
              "rejected_falsify": 0, "rejected_substance": 0, "written": {},
              "rejections": []}
    raw, refmap = distill(con, topic, unit_ids)
    report["candidates"] = len(raw)
    if not raw:
        return report

    # Verified evidence rides ON the candidate from here on. Parallel lists keyed by
    # position or statement text WILL desynchronise (falsify drops and rewrites
    # entries), and the failure mode is citations silently attached to the wrong
    # claim — which defeats the entire grounding guarantee.
    grounded = []
    for c in raw:
        try:
            grounded.append({**c, "_evidence": ground(con, c, refmap)})
        except DreamError as e:
            # One ungrounded candidate is a rejection, not a run failure — the gate
            # did its job. The reason is recorded so the rate stays visible.
            report["rejected_grounding"] += 1
            report["rejections"].append(f"grounding: {e}")

    survivors = falsify(con, topic, grounded, report["rejections"])
    report["rejected_falsify"] = len(grounded) - len(survivors)

    final = []
    for s in survivors:
        try:
            # falsify may have REWRITTEN the stance (corrected_stance) — re-run the
            # mechanical authorship gate on the corrected value, or an upgrade to
            # user_asserted would slip past the check `ground` ran on the original.
            _require_user_evidence(s["stance"], s["_evidence"], s["statement"])
        except DreamError as e:
            report["rejected_grounding"] += 1
            report["rejections"].append(f"grounding: {e}")
            continue
        if passes_substance(s, s["_evidence"]):
            final.append(s)
        else:
            report["rejected_substance"] += 1
            report["rejections"].append(f"substance: {s['statement'][:70]}")

    decided = reconcile(con, topic, final)
    report["written"] = persist(con, topic, decided)
    con.execute("UPDATE dream_topics SET last_dig_at=%s WHERE topic_id=%s",
                (_now(), topic["topic_id"]))
    con.commit()
    return report


# --------------------------------------------------------------------------- #
# Consolidate — the topic digest (fixed sections; bullets never invented)
# --------------------------------------------------------------------------- #
# The four sections are FIXED and the bullets are assembled from `dream_insights`
# rows in code. The model writes only the short synthesis prose that ties each
# section together, so a digest cannot invent structure or claims — the two things
# a free-prose digest gets wrong. "Changed positions" exists only because
# supersession closes windows instead of deleting (D6); "Rejected approaches" only
# because stance is tracked (D5).
DIGEST_SECTIONS = ("settled", "rejected", "changed", "open")

_CONSOLIDATE_SYSTEM = (
    "You write the connective prose for a knowledge digest. You are given a topic and "
    "its verified claims, already grouped into fixed sections.\n\n"
    "For each section write 1-3 sentences that tie its claims together: the through-line, "
    "any tension between them, what the grouping amounts to. Do NOT restate the claims "
    "one by one (they are printed verbatim beneath your prose), do not add claims that "
    "are not listed, and do not editorialise about the person. If a section has no "
    "claims, return an empty string for it."
)

_CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {s: {"type": "string"} for s in DIGEST_SECTIONS},
    "required": list(DIGEST_SECTIONS),
    "additionalProperties": False,
}


def _digest_buckets(con, topic_id: str) -> dict[str, list[dict]]:
    """Group a topic's insights into the four fixed sections. Purely mechanical."""
    # "Changed positions" is keyed on the RECORDED supersession kind: only a
    # `contradict` supersession is a change of mind. A `refine` also points here
    # via superseded_by, but it is by contract the SAME position in better words —
    # bucketing it as changed would assert a dated change that never happened.
    rows = con.execute(
        "SELECT i.*, s.statement AS superseded_statement FROM dream_insights i "
        "LEFT JOIN dream_insights s "
        "  ON s.superseded_by = i.unit_id AND s.superseded_kind = 'contradict' "
        "WHERE i.topic_id=%s AND i.status=%s ORDER BY i.support_count DESC, i.valid_from",
        (topic_id, STATUS_ACTIVE)).fetchall()
    b: dict[str, list[dict]] = {s: [] for s in DIGEST_SECTIONS}
    for r in rows:
        if r["contested"]:
            b["open"].append(r)
        elif r["stance"] == "user_rejected":
            b["rejected"].append(r)
        elif r["stance"] in TRUSTED_STANCES:
            b["settled"].append(r)
        else:
            b["open"].append(r)          # unadjudicated: not settled knowledge
        if r["superseded_statement"]:
            b["changed"].append(r)
    return b


_SECTION_TITLES = {"settled": "Settled", "rejected": "Rejected approaches",
                   "changed": "Changed positions", "open": "Open / unadjudicated"}


def _stance_label(stance: str, support_count: int) -> str:
    """The one rendering of an insight's attribution+support suffix (digest
    bullets AND recall lines) — two copies would drift the moment either gained
    a field, and this label is what keeps `support_count` honest to the reader."""
    return f"[{stance}, {support_count} source(s)]"


def consolidate(con, topic: dict) -> str | None:
    """Regenerate a topic's digest from its ACTIVE insight rows. Returns the digest
    text, or None when the topic has no insights yet.

    Always re-derived from insight rows — never from a previous digest. That is what
    keeps the abstraction ladder two levels deep and stops derived text feeding
    another derivation."""
    b = _digest_buckets(con, topic["topic_id"])
    if not any(b.values()):
        return None

    def _lines(rows, changed=False):
        out = []
        for r in rows:
            if changed:
                out.append(f"- was: {r['superseded_statement']}\n  now: {r['statement']} "
                           f"(changed {r['valid_from'].date().isoformat()})")
            else:
                out.append(f"- {r['statement']} "
                           f"{_stance_label(r['stance'], r['support_count'])} "
                           f"(last seen {r['last_seen_at'].date().isoformat()})")
        return "\n".join(out)

    listing = "\n\n".join(
        f"{_SECTION_TITLES[s]}:\n{_lines(b[s], changed=(s == 'changed')) or '  (none)'}"
        for s in DIGEST_SECTIONS)
    prose = _call(con, "consolidate", {"topic_id": topic["topic_id"]},
                  topic["topic_id"],
                  f"TOPIC: {topic['name']}\nCHARTER: {topic['charter']}\n\n{listing}",
                  _CONSOLIDATE_SCHEMA, system=_CONSOLIDATE_SYSTEM)

    parts = [f"# {topic['name']}"]
    for s in DIGEST_SECTIONS:
        if not b[s]:
            continue
        parts.append(f"\n## {_SECTION_TITLES[s]}\n")
        if prose.get(s, "").strip():
            parts.append(prose[s].strip() + "\n")
        parts.append(_lines(b[s], changed=(s == "changed")))
    text = "\n".join(parts)

    digest_id = f"dream-digest-{topic['topic_id']}"
    now = datetime.now(timezone.utc)
    oldest = min((r["first_seen_at"] for rows in b.values() for r in rows), default=now)
    _write_unit(con, digest_id, KIND_DIGEST,
                f"[{topic['topic_id']}] digest: {topic['name']}", text, oldest, now)
    con.commit()
    return text


def digest_of(con, topic_id: str) -> dict | None:
    return con.execute("SELECT title, summary, updated_at FROM units WHERE unit_id=%s",
                       (f"dream-digest-{topic_id}",)).fetchone()


# --------------------------------------------------------------------------- #
# Recall — the dream-first surface (explicit tiers, loud coverage gaps)
# --------------------------------------------------------------------------- #
def recall(con, query: str = "", *, topic_id: str | None = None, limit: int = 8,
           as_of: str | None = None, include_evidence: bool = False) -> dict:
    """Dream-first recall. Returns explicit TIERS, never one blended ranking:

        digest    -> the topic's synthesized current position (if any)
        insights  -> matching atomic claims, with stance + support + citations
        raw       -> raw transcript hits, as clearly-marked drill-down
        coverage  -> what is NOT distilled yet, stated LOUDLY

    Tiering rather than a score boost is deliberate: a `dream_boost` multiplier
    inside the ranking would silently interleave a distilled claim with a stray tool
    log and present the result as relevance. The caller must be able to see which
    tier an answer came from.

    `as_of` reads the bi-temporal history — the positions held at that date, rather
    than the current ones."""
    import search
    topic = get_topic(con, topic_id) if topic_id else None

    # --- insight tier: semantic retrieval restricted to dream units, then ranked
    # by trust (stance + support), because an unadjudicated proposal and a
    # five-times-asserted position must not rank as if they were the same object.
    ids: list[str] = []
    no_match = False
    if query.strip():
        # RRF scores are rank-only (they cannot say "nothing here is relevant"),
        # so the floor is on the dense cosine similarity hybrid_search now
        # reports. Below the floor a hit is retrieval filler, and returning it
        # as the user's position on the query is attribution pollution — the
        # exact failure the stance machinery exists to prevent.
        ids = [r["unit_id"] for r in
               search.hybrid_search(query, topk=limit * 4, source=DREAM_SOURCE)
               if (r.get("dense_sim") or 0.0) >= RECALL_MIN_SIM]
        no_match = not ids
    where = ["i.status = %(status)s"] if not as_of else [
        # Window-only PLUS not-retracted: supersession always closes the window
        # (persist's contract), and a retracted claim was withdrawn, not held.
        "i.status <> %(retracted)s",
        "i.valid_from <= %(as_of)s",
        "(i.valid_until IS NULL OR i.valid_until > %(as_of)s)"]
    params: dict = {"status": STATUS_ACTIVE, "retracted": STATUS_RETRACTED,
                    "as_of": as_of}
    if topic:
        where.append("i.topic_id = %(topic)s")
        params["topic"] = topic["topic_id"]
    if query.strip():
        # The id filter applies WHENEVER a query was given — an empty retrieval
        # must yield an EMPTY insight tier, never fall through to the
        # trust-ranked top of the whole table dressed up as an answer.
        where.append("i.unit_id = ANY(%(ids)s)")
        params["ids"] = ids
    insights = con.execute(
        "SELECT i.*, u.summary AS elaboration FROM dream_insights i "
        "JOIN units u USING(unit_id) WHERE " + " AND ".join(where) +
        " ORDER BY (i.stance = ANY(%(trusted)s)) DESC, i.support_count DESC, "
        "i.last_seen_at DESC LIMIT %(limit)s",
        {**params, "trusted": list(TRUSTED_STANCES), "limit": limit}).fetchall()

    # The topic follows from WHAT CAME BACK, not from matching the query's words
    # against charters: the insight tier was already retrieved by BGE-M3, so the
    # topic its hits belong to is a strictly better answer than any keyword overlap
    # (a query like "where should timestamps go in a prompt" shares no vocabulary
    # with the development-principles charter, yet its insight is exactly there).
    if topic is None and insights:
        counts: dict[str, int] = {}
        for i in insights:
            counts[i["topic_id"]] = counts.get(i["topic_id"], 0) + 1
        topic = get_topic(con, max(counts, key=lambda k: counts[k]))

    if include_evidence:
        for i in insights:
            i["evidence"] = con.execute(
                "SELECT src_unit_id, src_msg_id, quote FROM dream_evidence "
                "WHERE unit_id=%s", (i["unit_id"],)).fetchall()

    # --- raw tier: drill-down only, never blended into the tiers above
    raw = search.hybrid_search(query, topk=limit, source="all") if query.strip() else []

    # --- coverage: the gap is information the caller MUST have. A thin digest
    # presented as authoritative is worse than an honest "not distilled yet".
    cov: list[str] = []
    digest = digest_of(con, topic["topic_id"]) if topic else None
    if topic and not digest:
        cov.append(f"topic '{topic['topic_id']}' has no digest yet — nothing distilled")
    if topic and topic["last_dig_at"]:
        # Same eligibility predicate as the nightly selection (_changed_units):
        # counting by raw `updated_at` would hide exactly the units that
        # expression exists to catch (NULL updated_at, late-ingested material).
        newer = con.execute(
            "SELECT count(*) AS n FROM units WHERE source = ANY(%(sources)s) "
            f"AND {_LEARNED_AT_SQL} > %(since)s AND {_RENDERABLE_SQL}",
            {"sources": list(RAW_SOURCES), "senders": list(USER_SENDERS),
             "since": topic["last_dig_at"]}).fetchone()["n"]
        if newer:
            cov.append(f"{newer} raw unit(s) changed since this topic was last dug "
                       f"({topic['last_dig_at'].date().isoformat()}) — not yet distilled")
    elif topic:
        cov.append(f"topic '{topic['topic_id']}' has never been dug")
    if no_match:
        cov.append("no distilled insight matches this query — the knowledge base "
                   "does not cover it (raw transcript hits below are NOT distilled)")
    if not topic:
        cov.append("no topic matched this query — showing raw transcript results only")
    return {"topic": topic, "digest": digest, "insights": insights, "raw": raw,
            "coverage": cov, "as_of": as_of}


def format_recall(result: dict, *, requested_topic: str | None = None,
                  include_evidence: bool = False) -> list[str]:
    """Render `recall`'s tiers as lines. SSOT for both surfaces (the `dream recall`
    CLI and the `recall_knowledge` MCP tool) — they present identical content, and
    two copies of this drift the moment either gains a field."""
    t = result["topic"]
    # Name the topic explicitly: when it was DERIVED rather than requested, an
    # unnamed digest reads as authoritative on whatever was asked, so a wrong
    # derivation would be invisible. Same for as_of — a historical position must
    # never be mistaken for the current one.
    head = f"TOPIC: {t['topic_id']} ({t['name']})" if t else "TOPIC: (none matched)"
    if t and requested_topic is None:
        head += "  [derived from the retrieved insights, not requested]"
    out = [head]
    if result["as_of"]:
        out.append(f"AS OF: {result['as_of']} — positions held THEN, not now")

    out.append("\nCOVERAGE:")
    out += [f"  ! {c}" for c in result["coverage"]] or ["  (no gaps)"]

    out.append("\nDIGEST:")
    out.append(result["digest"]["summary"] if result["digest"] else "  (none)")

    out.append("\nINSIGHTS:")
    if not result["insights"]:
        out.append("  (none)")
    for i in result["insights"]:
        out.append(f"  {i['statement']}  "
                   f"{_stance_label(i['stance'], i['support_count'])}")
        if include_evidence:
            for e in i.get("evidence", []):
                out.append(f"      {e['src_unit_id']}:{e['src_msg_id']}  "
                           f"{e['quote'][:120]!r}")

    out.append("\nRAW TRANSCRIPTS (not distilled):")
    if not result["raw"]:
        out.append("  (none)")
    for r in result["raw"]:
        snippet = r["text"].replace("\n", " ")[:200]
        out.append(f"\n- [{r['source']}] {r['unit_name'] or '(untitled)'}  "
                   f"[{r['unit_id']}]  score={r['score']:.4f}\n  {snippet}")
    return out


# --------------------------------------------------------------------------- #
# Run modes — incremental (nightly) and bulk (explicit). SEPARATE code paths.
# --------------------------------------------------------------------------- #
# The two have cost profiles two orders of magnitude apart (ADR 0004 D9). They do
# not share a loop with a ratio knob, because that is exactly how backfill work
# leaks into a "background" run and makes it permanently expensive.
TRIAGE_BATCH = 15                # units per triage call


class DreamThrottled(DreamError):
    """The subscription's rate limit was hit. The run stops and reports; it does
    NOT retry in a loop or return partial results as if complete."""


_THROTTLE_MARKERS = ("rate limit", "usage limit", "rate_limit", "429",
                     "too many requests", "exceeded your")


def _reraise_throttle(e: Exception) -> None:
    if any(m in str(e).lower() for m in _THROTTLE_MARKERS):
        raise DreamThrottled(str(e)) from e


class _Budget:
    """Caps calls per run. Checked BETWEEN stages so a run always stops on a
    coherent boundary — never half-way through a topic's gate chain.

    `take(n)` RESERVES n calls, because a dig must not begin unless its whole gate
    chain fits. A reservation is an upper bound, not a spend: a dig whose topic has
    no existing insights skips reconcile and uses 2 of its 3. So `reserved` governs
    the cap while the reported call count comes from `dream_queue` — the audit log
    of calls actually made. Reporting reservations as spend would overstate cost."""

    def __init__(self, max_calls: int | None):
        self.max_calls, self.reserved = max_calls, 0

    def take(self, n: int = 1) -> bool:
        if self.max_calls is not None and self.reserved + n > self.max_calls:
            return False
        self.reserved += n
        return True


def _calls_made(con) -> int:
    """Rows in the call audit log — the ground truth for how many calls happened."""
    return con.execute("SELECT count(*) AS n FROM dream_queue").fetchone()["n"]


# A unit is a triage CANDIDATE only if skeleton() can render it: at least one
# non-empty user turn (any turn for a project_doc, which has no user turns).
# Deciding emptiness at SELECTION is what keeps the nightly pass un-wedgeable:
# a genuinely-empty unit (an abandoned claude.ai chat) is skipped VISIBLY here,
# while skeleton()'s fail-loud raise stays reserved for the systemic case — a
# unit that passed this predicate yet renders empty means the render itself broke.
_RENDERABLE_SQL = (
    "EXISTS (SELECT 1 FROM messages m WHERE m.unit_id = units.unit_id "
    "AND (units.kind = 'project_doc' OR m.sender = ANY(%(senders)s)) "
    "AND btrim(coalesce(m.embed_text, m.text, '')) <> '')")

# The watermark compares against when clync LEARNED of a unit, never only when
# the SOURCE last touched it: `updated_at` is source-authored (NULL for project
# docs, months old for late-ingested material), so `updated_at > watermark`
# silently excludes anything that arrives behind the mark. `synced_at` is
# written as _now() on every upsert (both sources, every kind), so this
# expression is total and monotone with ingestion.
_LEARNED_AT_SQL = "greatest(coalesce(updated_at, 'epoch'::timestamptz), synced_at)"


def _changed_units(con, since: str) -> tuple[list[str], list[str]]:
    """Units eligible for triage since the watermark -> (eligible, skipped_empty).

    `skipped_empty` is the changed units excluded because nothing in them is
    renderable — returned so the run REPORT shows the skip instead of hiding it."""
    params = {"sources": list(RAW_SOURCES), "senders": list(USER_SENDERS),
              "since": since}
    base = ("SELECT unit_id FROM units WHERE source = ANY(%(sources)s) "
            "AND (msg_count IS NULL OR msg_count > 0) "
            f"AND {_LEARNED_AT_SQL} > %(since)s")
    eligible = [r["unit_id"] for r in con.execute(
        base + f" AND {_RENDERABLE_SQL} ORDER BY {_LEARNED_AT_SQL}",
        params).fetchall()]
    skipped = [r["unit_id"] for r in con.execute(
        base + f" AND NOT {_RENDERABLE_SQL} ORDER BY {_LEARNED_AT_SQL}",
        params).fetchall()]
    return eligible, skipped


def _queue_pending(con, assignments: dict[str, list[str]]) -> int:
    """Record triage's (topic, unit) assignments so a dig can happen later."""
    n = 0
    for tid, unit_ids in assignments.items():
        for unit_id in unit_ids:
            n += con.execute(
                "INSERT INTO dream_pending (topic_id, unit_id, queued_at) "
                "VALUES (%s,%s,%s) ON CONFLICT DO NOTHING", (tid, unit_id, _now())
            ).rowcount
    con.commit()
    return n


def _ripe_topics(con) -> dict[str, list[str]]:
    """Topics whose pending work is worth a dig now: enough units accumulated, or
    waited long enough. Returns {topic_id: unit_ids} oldest-first.

    A topic that is neither is left alone — that is the whole cost saving, and it
    costs nothing in completeness because the rows stay pending."""
    rows = con.execute(
        "SELECT p.topic_id, count(*) AS n, min(p.queued_at) AS oldest "
        "FROM dream_pending p JOIN dream_topics t USING(topic_id) "
        "WHERE t.status='active' GROUP BY p.topic_id").fetchall()
    cutoff = datetime.now(timezone.utc) - timedelta(days=DIG_MAX_DEFER_DAYS)
    ripe = [r["topic_id"] for r in rows
            if r["n"] >= DIG_MIN_UNITS or r["oldest"] <= cutoff]
    return {tid: [r["unit_id"] for r in con.execute(
        "SELECT unit_id FROM dream_pending WHERE topic_id=%s ORDER BY queued_at",
        (tid,)).fetchall()] for tid in ripe}


def _clear_pending(con, topic_id: str, unit_ids: list[str]) -> None:
    """Drop pending rows for units that were ACTUALLY passed to a dig — never for
    units a batch cap left behind, or they would be silently forgotten."""
    con.execute("DELETE FROM dream_pending WHERE topic_id=%s AND unit_id = ANY(%s)",
                (topic_id, unit_ids))
    con.commit()


def pending_counts(con) -> list[dict]:
    """Per-topic pending backlog, for `dream status` — deferred work must be VISIBLE,
    otherwise 'nothing happened tonight' is indistinguishable from 'nothing to do'."""
    return con.execute(
        "SELECT topic_id, count(*) AS n, min(queued_at) AS oldest "
        "FROM dream_pending GROUP BY topic_id ORDER BY topic_id").fetchall()


def _index_written(con, report: dict) -> None:
    """Embed whatever this run wrote, at the END of the run.

    Without this the layer is silently a day stale: an insight is only reachable
    through `recall`'s insight tier once it is in the vector index, and the
    scheduled job's `build_index` runs BEFORE the dream pass. Tonight's insights
    would therefore be invisible until tomorrow night — and invisible *quietly*,
    since the digest tier would still answer. Indexing once per run (not per
    insight) keeps the embedding model load to a single pass; `build_index` is a
    no-op when no signature changed."""
    wrote = any(any(d["written"].values())
                for digs in report.get("digs", {}).values()
                for d in (digs if isinstance(digs, list) else [digs]))
    if not (wrote or report.get("consolidated")):
        return
    con.commit()                      # the new units must be visible to build_index
    import search
    report["indexed"] = search.build_index()


def run_incremental(con, max_calls: int | None = None) -> dict:
    """The nightly pass. Triage-gated and batched: every stage is gated by the one
    before it, so a quiet day costs ONE call and a day with no new units costs zero.
    Never re-sweeps history — that is `run_backfill`'s job.

    On the very first run there is no watermark, and triaging the whole corpus would
    be bulk work wearing the nightly label. So the watermark is initialised and the
    run reports that `dream backfill` is what mines history."""
    budget = _Budget(max_calls)
    calls_before = _calls_made(con)
    # The new watermark is snapshotted BEFORE selection, and it is what every
    # success path writes. Writing end-of-run _now() instead would open a window
    # the length of the pass (minutes of Opus calls): a unit synced in between
    # was not selected, yet the watermark would move past it — skipped silently,
    # forever. With t0 the window is closed by construction.
    t0 = _now()
    report: dict = {"mode": "incremental", "changed": 0, "skipped_empty": [],
                    "topics_touched": [], "queued": 0, "digs": {},
                    "consolidated": [], "calls": 0, "deferred": [],
                    "stopped_early": None}
    watermark = get_meta(con, WATERMARK_KEY)
    if not watermark:
        set_meta(con, WATERMARK_KEY, t0)
        con.commit()
        report["note"] = ("watermark initialised — nothing distilled. History is "
                          "mined by `clync dream backfill`, not by the nightly run.")
        return report

    units, report["skipped_empty"] = _changed_units(con, watermark)
    # Units ELIGIBLE for triage. Not "triaged": a budget-capped run leaves
    # some of them unread, and reporting them as processed would be a lie.
    report["changed"] = len(units)
    if not units:
        set_meta(con, WATERMARK_KEY, t0)
        con.commit()
        report["note"] = "no new or changed units since last run — nothing to do"
        return report

    touched: dict[str, list[str]] = {}
    try:
        for i in range(0, len(units), TRIAGE_BATCH):
            if not budget.take():
                report["stopped_early"] = "max-calls reached during triage"
                break
            for unit_id, tids in triage(con, units[i:i + TRIAGE_BATCH]).items():
                for tid in tids:
                    touched.setdefault(tid, []).append(unit_id)

        report["topics_touched"] = sorted(touched)
        report["queued"] = _queue_pending(con, touched)

        # Dig only the topics whose backlog is ripe. Cost scales with topics dug, so
        # this — not the unit count — is the nightly budget's real lever.
        for tid, unit_ids in _ripe_topics(con).items():
            # Each dig is distill + falsify + reconcile => up to 3 calls.
            if not budget.take(3):
                report["stopped_early"] = "max-calls reached before digging " + tid
                break
            batch = unit_ids[:DIG_BATCH_UNITS]
            t = get_topic(con, tid)
            report["digs"][tid] = dig(con, t, batch)
            # Only the units actually mined leave the queue. The rest stay pending and
            # are picked up next run, instead of being dropped by the batch cap after
            # the watermark has already moved past them.
            _clear_pending(con, tid, batch)

        for tid, rep in report["digs"].items():
            if not any(rep["written"].values()):
                continue            # insight set unchanged => digest cannot differ
            if not budget.take():
                report["stopped_early"] = "max-calls reached before consolidating " + tid
                break
            consolidate(con, get_topic(con, tid))
            report["consolidated"].append(tid)
    except Exception as e:
        _reraise_throttle(e)
        raise
    finally:
        report["calls"] = _calls_made(con) - calls_before

    # Deferred work must be VISIBLE in the report, or a cheap night is
    # indistinguishable from a broken one.
    report["deferred"] = [{"topic_id": r["topic_id"], "pending": r["n"],
                           "oldest": r["oldest"].date().isoformat()}
                          for r in pending_counts(con)]

    # Only advance the watermark on a run that wasn't cut short, or the units the
    # budget skipped would never be triaged again. It advances to t0 (the
    # pre-selection snapshot), never to the current clock — see t0 above.
    if not report["stopped_early"]:
        set_meta(con, WATERMARK_KEY, t0)
        con.commit()
    _index_written(con, report)
    return report


def run_backfill(con, topic_id: str | None = None, max_calls: int = 30) -> dict:
    """The bulk pass — EXPLICIT only, never scheduled. Walks the deep slate:
    unrestricted relevance retrieval per probe query, minus units this topic has
    already drawn evidence from, in batches until the budget runs out.

    Resumable by construction: progress lives in `dream_evidence` (which units are
    already mined), so the next invocation continues where this one stopped."""
    budget = _Budget(max_calls)
    calls_before = _calls_made(con)
    report: dict = {"mode": "backfill", "digs": {}, "consolidated": [], "calls": 0,
                    "stopped_early": None}
    targets = [get_topic(con, topic_id)] if topic_id else topics(con)
    try:
        for t in targets:
            rounds = []
            while True:
                if not budget.take(3):
                    report["stopped_early"] = "max-calls reached"
                    break
                mined = cited_units(con, t["topic_id"])
                batch = candidates(con, t, exclude=mined, limit=DIG_BATCH_UNITS)
                if not batch:
                    break               # this topic's candidate pool is exhausted
                rounds.append(dig(con, t, batch))
            if rounds:
                report["digs"][t["topic_id"]] = rounds
                if budget.take():
                    consolidate(con, get_topic(con, t["topic_id"]))
                    report["consolidated"].append(t["topic_id"])
            if report["stopped_early"]:
                break
    except Exception as e:
        _reraise_throttle(e)
        raise
    finally:
        report["calls"] = _calls_made(con) - calls_before
    _index_written(con, report)
    return report


def status(con) -> dict:
    """Health + coverage for `clync doctor` / `clync dream status`."""
    rows = con.execute(
        "SELECT t.topic_id, t.status, t.last_dig_at, "
        "count(i.unit_id) FILTER (WHERE i.status='active')     AS active, "
        "count(i.unit_id) FILTER (WHERE i.status='superseded') AS superseded, "
        "count(i.unit_id) FILTER (WHERE i.contested)           AS contested "
        "FROM dream_topics t LEFT JOIN dream_insights i USING(topic_id) "
        "GROUP BY t.topic_id, t.status, t.last_dig_at ORDER BY t.topic_id").fetchall()
    digests = {r["unit_id"].replace("dream-digest-", "") for r in con.execute(
        "SELECT unit_id FROM units WHERE kind=%s", (KIND_DIGEST,)).fetchall()}
    for r in rows:
        r["digest"] = r["topic_id"] in digests
    return {"topics": rows, "watermark": get_meta(con, WATERMARK_KEY),
            "pending": pending_counts(con), "usage": usage_totals(con)}


def usage_totals(con) -> dict:
    """Aggregate what the layer has actually spent, from the audit log."""
    rows = con.execute(
        "SELECT kind, count(*) AS calls, "
        "coalesce(sum((usage->>'input_tokens')::bigint),0) AS input_tokens, "
        "coalesce(sum((usage->>'output_tokens')::bigint),0) AS output_tokens, "
        "coalesce(sum((usage->>'cache_read_input_tokens')::bigint),0) AS cache_read, "
        "coalesce(sum((usage->>'cache_creation_input_tokens')::bigint),0) AS cache_create "
        "FROM dream_queue WHERE state='done' GROUP BY kind ORDER BY kind").fetchall()
    return {"by_kind": rows,
            "failed": con.execute("SELECT count(*) AS n FROM dream_queue "
                                  "WHERE state='failed'").fetchone()["n"]}
