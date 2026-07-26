# ADR 0004 — The Dream layer: topic-driven knowledge distillation

- **Status:** Accepted — implemented in `dream.py` (2026-07-26). The cost model in D9
  and the topic-status vocabulary in D3 were **corrected by measurement during the
  build**; both now describe what the code does, not what was estimated.
- **Date:** 2026-07-26
- **Scope:** A third content layer in clync's store: *derived knowledge* distilled
  from the raw transcripts of both existing sources, plus the retrieval surface
  that prefers it. Builds on [ADR 0003](0003-postgres-only-and-cc-ingest.md)'s
  unified `units`/`messages`/`chunks` model.

## Context

### The gap

clync today answers *"where did I discuss X?"*. It cannot answer *"what do I
actually think about X?"* — because the only thing in the store is raw
transcript. The knowledge is in there, but it is spread across hundreds of
conversations, restated with variations, contradicted and then corrected, and
buried under tool output.

The corpus, measured 2026-07-26:

| | count |
|---|---|
| units | 603 (354 claude.ai chats, 237 cc sessions, 12 project docs) |
| messages | 241,668 |
| message text | 99,519,885 chars ≈ **25M tokens** |
| indexed chunks | 171,205 |

25M tokens is far too much to hand an LLM. Any design that "reads the corpus"
is dead on arrival; the layer must be **retrieval-driven**.

### The constraint that shapes everything: no API quota

There is no Anthropic API billing and there will not be. The only available
inference is the **local `claude` CLI under the existing subscription OAuth**.
This was verified, not assumed (2026-07-26, `claude` 2.1.220):

```sh
claude -p "<prompt>" --model opus --output-format json \
  --json-schema '<schema>' --system-prompt '<role>' \
  --tools "" --strict-mcp-config --setting-sources "" \
  --disable-slash-commands --no-session-persistence
```

| Verified property | Evidence |
|---|---|
| Subscription auth, no API billing | `ANTHROPIC_API_KEY` unset; result JSON reports `"provider":"firstParty"`. `total_cost_usd` is **notional accounting only** — nothing is charged. |
| Validated structured output | `--json-schema` populates a parsed `structured_output` object in the result JSON. No prose-parsing, no regex. |
| Per-call prompt overhead is controllable | Total prefill (`input + cache_creation + cache_read`), one measurement set: default `claude -p` = **76,068** tok · `+ --tools "" --strict-mcp-config --setting-sources "" --disable-slash-commands` = **9,209** · `+ --system-prompt` (replacing CC's) = **515**. The floor is not the operating cost: with the layer's *real* system prompt and `--json-schema`, a triage call prefills **1,576** tok and a distill call **2,140**. So budget **~1.6-2.1k overhead per call** (~40x below default), not the 515 floor. |
| No self-pollution | `--no-session-persistence` writes **no** transcript under `~/.claude/projects` — verified by grepping for the probe's session id. The Dream worker therefore cannot be ingested by `sync-cc`. |
| Prompt caching works across separate invocations | Two identical triage calls in separate processes: call 1 `cache_creation=1,295`, call 2 `cache_read=1,295`. The per-call system+schema prefix is therefore paid once per window, so grouping same-stage calls together is cheaper than interleaving stages. |
| Explicit failure signal | result JSON carries `is_error`, `subtype`, `api_error_status`, `permission_denials`, `terminal_reason` — enough to fail loud without inference. |

`--bare` is **not** usable: its own help text states auth is "strictly
`ANTHROPIC_API_KEY` or `apiKeyHelper`… OAuth and keychain are never read".

The **Claude Agent SDK** (`claude-agent-sdk`) is the same harness in-process and
the same subscription auth; it is a viable later swap for streaming/in-process
pooling. The CLI is chosen now because the numbers above are measured, it adds
no Python dependency, and a subprocess-per-item is trivially resumable and
poolable. *(Unverified: SDK per-call overhead — not benchmarked.)*

The real budget is **subscription rate limits** (rolling 5h + weekly windows),
not dollars. So the layer must be paced, capped, resumable, and must fail loud
on throttling rather than retry forever.

### Prior art baseline

What the field currently does, and where this design deliberately diverges:

| Practice | Prior art | This design |
|---|---|---|
| Background consolidation while idle, strong model for the consolidation pass | Letta **sleep-time compute**; Supermemory **Dynamic Dreaming** | Adopt. Opus for every knowledge judgement; nightly launchd pass after sync+index. |
| Reflections written back into the *same* retrievable stream as observations | Stanford **Generative Agents** | Adopt. Insights become `units` → indexed by `search.py` for free. |
| Retrieval = relevance + recency + importance | **Generative Agents** (linear, α=1) | Adopt recency bias in the *dig*, but reject an opaque score blend at query time (see D8). |
| Bi-temporal validity; contradictions **close a validity window** instead of deleting | **Zep / Graphiti** | Adopt, at insight level. the user's positions evolve — "you used to think X; since May you think Y" is the valuable answer, not an overwrite. |
| Every inference must trace to source or it is rejected | **Supermemory** | Adopt as a hard gate, not a guideline. |
| Multi-signal promotion gates + phases, only the last phase writes durable memory, audit log | **OpenClaw dreaming** (`minScore .8`, `minRecallCount 3`, `minUniqueQueries 3`) | Adopt the gate concept; replace search-hit-count proxies with **real evidence signals** (distinct source units, time spread, stance). |
| Recursive abstraction ladder | **RAPTOR** — but its own paper reports ~4% of summaries carry minor hallucinations, and hierarchical merging can amplify them | **Reject recursion.** Exactly two levels, digests always re-derived from *raw* evidence (D2). |
| Un-consolidated content still queryable; derived state catches up in background | **Supermemory** | Adopt, and make the coverage gap *loud* rather than silent (D8). |
| Lookback window over recent files | OpenClaw (7 days), Supermemory (recent context) | **Reject the window.** We have a BGE-M3 hybrid index over the whole 25M-token corpus, so a topic can be dug across all of history, not just last week (D4). |

Two things nobody in the baseline does, and which matter most here:

1. **Dialogue attribution.** Every system above treats memory as a flat stream
   of facts. This corpus is a *dialogue*: some statements are the user's assertions,
   some are Claude proposals he endorsed, and some are Claude proposals he
   **rejected**. Flattening those produces a knowledge base that confidently
   attributes to the user things he argued against. See D5.
2. **Falsification before promotion.** No baseline tries to *refute* a candidate
   insight before durably storing it. Given the repo's own rules (evidence gates,
   adversarial review, "be the skeptic"), a refute pass is the natural bar. See D6.

## Decisions

### D1. Execution substrate: headless `claude -p`, one subprocess per work item

Opus for every call that makes a knowledge judgement. Triage (routing only) runs
Opus at `--effort low`; distillation, falsification and consolidation run at
default effort. The invocation is the measured minimal form from Context, always
with `--json-schema`.

Every call is a pure function of its prompt: no tools (`--tools ""`), no MCP, no
settings, no session. The Dream worker never reads the filesystem and never
writes anywhere except through `dream.py`'s own DB writes. This is what makes
the layer auditable and re-runnable.

**Fail loud:** a call whose result JSON has `is_error`, a non-`success` `subtype`,
or a missing `structured_output` raises. There is no "skip this item and carry
on" path — the item is marked `failed` with the raw error retained, the run
aborts if failures exceed a threshold, and `clync doctor` surfaces it.

### D2. Two levels, never recursive

- **`dream_insight`** — one atomic claim. "Prefer fixing at the layer that
  should have prevented the bug over the layer where it surfaced." Carries
  statement, elaboration, stance, evidence, validity window.
- **`dream_digest`** — one per topic: the synthesized current state of the user's
  thinking on that topic, assembled from its *active* insights, in **fixed
  structured sections** (schema-enforced, not free prose):

  | section | contents |
  |---|---|
  | **Settled** | active insights, stance `user_asserted`/`user_endorsed`, with support count |
  | **Rejected approaches** | active insights with stance `user_rejected` — what he argued against, and why |
  | **Changed positions** | supersession chains: `old → new`, dated from the validity windows |
  | **Open** | insights the falsification gate marked contested, or questions raised and never resolved |

  Fixed slots make the digest diffable across runs, mechanically assemblable from
  the insight rows (so the consolidate call *writes prose per bullet*, it does not
  invent structure), and directly actionable by an agent. "Changed positions" is
  the section that only exists because of D6's supersede-never-overwrite rule.

A digest is **always regenerated from the insight set**, and insights are
**always derived from raw transcript evidence**. Derived text is never input to
another derivation. This caps hallucination amplification at one hop — the
specific failure RAPTOR measures and a deeper ladder would compound.

There is no level 3. If cross-topic synthesis is wanted later, it is a *query-time*
composition over digests, not a stored third tier.

### D3. `dream_topics`: curated seeds, plus proposals

Topics are rows, not code. Each has a name, a charter (what counts as in-scope,
written for the distiller), and **probe queries** (the natural-language strings
the dig feeds to `hybrid_search`). Seeds from the stated interests:

development principles · writing manners · architecture thinking · AI frontier
research · local AI research · Claude Code token economy & output quality

`dream_topics` is **hand-edited configuration**, not derived data: it is the one
dream table the stale-shape guard never drops, and the only thing that decides
what gets distilled. `status ∈ {active, proposed, retired}` gates digging — only
`active` topics are ever dug, so a topic can be parked (`proposed`) or turned off
(`retired`) without losing the insights already attached to it. Nothing personal
is hardcoded: the seeds ship as `dream topics --seed` operating on the same table.

Automatic topic *discovery* (clustering insights that fit no active topic and
writing `proposed` rows for review) is deliberately **not** built. It would be a
further inference stage in service of a problem that does not exist yet — with six
charters covering the stated interests, the failure mode to watch for is a charter
that is too broad, which auto-proposal would not fix. Adding a topic is one INSERT.

### D4. The dig: two distinct modes — cheap nightly increment, one-time bulk

These are **separate modes**, not one loop with a ratio knob. Conflating them is
what makes a "background" layer quietly expensive forever.

#### Incremental (the nightly default) — triage-gated, batched, fresh only

1. **Triage, batched.** *One* Opus call (`--effort low`) takes the condensed
   skeletons of **all** units new or changed since the last run and returns, per
   unit, which active topics it touches (or none). 10–20 sessions/day of
   skeletons is ~20–30k tokens — one call, occasionally two. Never one call
   per unit.
2. **Gate.** Topics no unit touched are **skipped entirely — zero calls.** Most
   nights that means 1–3 of 6 topics do any work at all.
3. **Fresh slate only.** For each touched topic, retrieve via
   `search.hybrid_search` with `since = last_dig_at`, restricted to raw sources,
   unioned across the topic's probe queries. History is *not* re-swept.
4. **Condense** — candidates reduced to the turns that carry intent (user
   prompts, assistant text, `Agent` results) by reusing the existing `embed_text`
   tier; tool stdout is already excluded there.
5. **Distill** — one call per *topic* (not per unit), schema-constrained,
   emitting candidate insights with mandatory evidence refs.
6. **Reconcile + falsify + consolidate** — batched per topic (D6), and a digest
   is regenerated **only if** its active insight set actually changed.

Every stage is gated by the one before it. A quiet day costs one triage call and
nothing else.

#### Bulk (explicit, one-time-ish) — `dream backfill`

The *deep slate*: unrestricted `sort=relevance` retrieval per probe query, minus
units already in the topic's evidence set, walked in batches until the topic's
candidate pool is exhausted. Never runs on the nightly schedule. Invoked
deliberately, capped by `--max-calls`, resumable from `dream_queue`.

`dream backfill` is also the right tool after adding a new topic or materially
rewriting a charter — those are the only recurring reasons to re-sweep history.

The topic's probe queries are revisable in both modes: a distill call may propose
query additions when it finds vocabulary the probes missed. That is the "repeated
dig" — each run's understanding sharpens the next run's retrieval, without
re-reading history.

**Non-circularity is enforced at the query, not by convention:** the dig filters
to `source IN ('claude_ai','claude_code')`. Combined with `--no-session-persistence`
(verified), there are no paths by which the layer can read its own output.

### D5. Stance: the differentiator

Every insight carries `stance`, and the distiller is required to justify it from
the transcript:

| stance | meaning |
|---|---|
| `user_asserted` | the user stated it himself |
| `user_endorsed` | Claude proposed, the user explicitly accepted / adopted / merged it |
| `user_rejected` | Claude proposed, the user pushed back — **kept**, because knowing what he rejected and why is knowledge |
| `co_derived` | reached jointly, no clean attribution |
| `claude_proposed` | in the transcript but never adjudicated — lowest trust |

Without this the layer cannot honestly answer "what do *I* think", which is the
actual ask. It also gives retrieval a real trust axis: a `user_asserted` insight
with five independent sources is not the same object as an unadjudicated
suggestion, and they must not rank as if they were.

### D6. Lifecycle: falsify, then promote; supersede, never overwrite

A candidate insight passes three gates before it becomes durable:

1. **Grounding gate (mechanical).** Every evidence ref must resolve to a real
   `(unit_id, msg_id)` in the store, and the cited text must actually exist there
   verbatim. Unresolvable ref, or a quote that isn't in that message → the candidate
   is rejected. This is a DB check, not an LLM judgement, so it cannot be talked
   around. A candidate with one bad reference is rejected whole: partly grounded is
   not grounded.

   **The model never handles ids.** Each quotable message is labelled `[e7 sender]`
   in the prompt, and code owns the `e7 → (unit_id, msg_id)` map. This is a measured
   correction, not a style choice: with the ids inline, a real dig was shown the
   correct 36-char UUID and cited `…ad4d227fceff` for `…ad4d277fceff` — one
   transposed digit — and the gate correctly discarded **three genuinely-grounded
   insights, 20% of that dig**, over a copy error. Fuzzy-matching the id was
   rejected as the fix: the nearest id is not necessarily the intended one, and a
   citation silently resolved to the WRONG message is far worse than a rejected one.
   Removing the transcription burden eliminates the whole failure class instead, and
   an unknown label still fails loud.
2. **Falsification gate (Opus).** A separate call is given the candidate plus its
   cited evidence and asked to *refute* it: is it actually supported, is the
   stance right, is it a restatement of an existing insight, is it too generic to
   be knowledge? Default-to-refuted on uncertainty.
3. **Substance gate (mechanical).** Distinct source units ≥ 2 **or** stance =
   `user_asserted` with an explicit statement. Blocks single-mention noise
   without discarding a clearly-stated one-off principle.

On reconcile against existing insights, the outcome is one of:

- **reinforce** — same claim, new evidence → append refs, bump `support_count`,
  extend `last_seen_at`.
- **refine** — same claim, better wording → new version, old row
  `status='superseded'`, `superseded_by` set.
- **contradict** — the user's position changed → old row gets
  `valid_until = new.valid_from` and `status='superseded'`; the new row is
  active. **Nothing is deleted.** The history stays queryable, so the layer can
  answer "when did I change my mind about X".
- **new** — no match.

Retraction (`status='retracted'`) is reserved for insights whose evidence
disappeared (source unit purged) — mirroring the existing deletion-reconciliation
behaviour of `cc_sync_state`.

### D7. Schema

Insights are **`units` rows**, extended 1:1 — not a parallel text store. This is
the SSOT-preserving choice: `search.py` already iterates all units, so insights
get BGE-M3 dense+sparse indexing, facets, and `get_conversation` with no second
retrieval path.

```sql
-- units gains: kind ∈ {…, 'dream_insight', 'dream_digest'}
--              source ∈ {claude_ai, claude_code, 'dream'}

CREATE TABLE dream_topics (
    topic_id      text PRIMARY KEY,      -- slug
    name          text NOT NULL,
    charter       text NOT NULL,         -- in/out of scope, written for the distiller
    probe_queries jsonb NOT NULL,        -- text[] driving the dig
    status        text NOT NULL,         -- active | proposed | retired
    last_dig_at   timestamptz,           -- recency-bias watermark
    created_at    timestamptz NOT NULL
);

CREATE TABLE dream_insights (
    unit_id       text PRIMARY KEY REFERENCES units(unit_id) ON DELETE CASCADE,
    topic_id      text NOT NULL REFERENCES dream_topics(topic_id),
    statement     text NOT NULL,         -- the claim, one sentence
    stance        text NOT NULL,         -- D5
    status        text NOT NULL,         -- active | superseded | retracted
    superseded_by text REFERENCES units(unit_id),
    support_count int NOT NULL,
    valid_from    timestamptz NOT NULL,  -- bi-temporal: when the position held
    valid_until   timestamptz,           -- NULL = still current
    first_seen_at timestamptz NOT NULL,  -- when evidence was authored
    last_seen_at  timestamptz NOT NULL,
    distilled_at  timestamptz NOT NULL,  -- when *we* derived it
    model         text NOT NULL,
    evidence_sig  text NOT NULL          -- sha256 over evidence -> re-derive trigger
);

CREATE TABLE dream_evidence (
    unit_id     text NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE, -- the insight
    src_unit_id text NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE, -- the transcript
    src_msg_id  text,
    quote       text,                    -- must verify against messages.text
    PRIMARY KEY (unit_id, src_unit_id, src_msg_id)
);

CREATE TABLE dream_queue (               -- resumable, budget-governed work
    item_id    text PRIMARY KEY,
    kind       text NOT NULL,            -- triage | distill | falsify | reconcile | consolidate
    topic_id   text,
    payload    jsonb NOT NULL,
    state      text NOT NULL,            -- pending | running | done | failed
    attempts   int NOT NULL,
    error      text,
    usage      jsonb,                    -- the result JSON's usage block, per call
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL
);

CREATE TABLE dream_pending (             -- triaged, awaiting a dig (see D9)
    topic_id  text NOT NULL REFERENCES dream_topics(topic_id) ON DELETE CASCADE,
    unit_id   text NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
    queued_at timestamptz NOT NULL,
    PRIMARY KEY (topic_id, unit_id)
);
```

`dream_pending` is what makes the nightly pass both cheap and lossless: an
assignment lives in a table rather than in a local variable, so a run may defer a
dig (or hit its per-dig unit cap) without dropping units the watermark has already
moved past.

`evidence_sig` mirrors `indexed_units.signature`: if a cited source unit's text
changes (a cc session resumed and re-parsed), the signature flips and the insight
is re-derived rather than silently resting on stale evidence.

### D8. Retrieval: an explicit dream-first surface, not a score fudge

A new MCP tool alongside the existing two:

```
recall_knowledge(query|topic, limit, include_evidence, as_of)
```

It returns **tiered, labelled** output:

1. the topic **digest** (if the query maps to a topic),
2. matching **active insights**, each with stance, support count, and citations,
3. **raw transcript** hits — as drill-down, clearly marked as un-distilled.

Two deliberate choices:

- **Tiering, not boosting.** A `dream_boost` multiplier inside the existing
  scored CTE would silently interleave a distilled claim with a stray tool log
  and call the ordering relevance. Tiering keeps the epistemic difference visible
  to the caller. Ranking *within* the dream tier uses the existing RRF plus
  stance and support (an insight the user asserted, supported five times, outranks an
  unadjudicated proposal).
- **Coverage gaps are loud.** If the query's topic has no digest, or its last dig
  predates the newest matching raw unit, the response says so explicitly and
  falls back to raw. Supermemory's un-dreamt fallback, but never dressing thin
  coverage up as an authoritative answer.

`as_of` exposes the bi-temporal history ("what did I think in March?").

Existing behaviour is preserved: `VALID_SOURCES` gains `'dream'`, but
`source='all'` continues to mean *raw only*. Dreams are reachable via `source='dream'`
or `recall_knowledge`. So the non-circularity filter is also the default, and no
existing query changes meaning.

### D9. Pacing: budget-governed, resumable, loud on throttle

- `dream run [--topic T] [--max-calls N]` — the **incremental** mode (D4).
  Enqueues and drains the triage-gated nightly work, then stops. It **cannot**
  enqueue deep-slate items; that separation is enforced in code, not by a default
  flag value, so backfill can never leak into the scheduled run.
- `dream backfill [--topic T] --max-calls N` — the **bulk** mode. Explicit only;
  never scheduled. State lives in `dream_queue`, so a capped run resumes exactly
  where the last one stopped.
- The nightly pass runs **inside the existing `clync scheduled` launchd job**,
  immediately after sync+index — not as a second job at a later hour. The dig is
  only as good as the index it queries, and sequencing it in one job makes that
  ordering true by construction rather than dependent on the sync having finished
  by a guessed-at second wake time. It also avoids a second plist, log, and
  teardown path. Throttling there is a `warn` (expected, resumable); any other
  failure notifies `fail` and re-raises.
- Rate-limit / throttle response: **stop the run, record state, notify** via the
  existing `notify()` path. No retry storms, no silent partial results.
- Every call's `usage` block is persisted to `dream_queue.usage`, so real
  consumption is measurable rather than estimated after week one.

Cost model, **corrected by measurement**. The first real nightly run on this corpus
cost **14 calls for one ordinary day** (6 changed units) — not the 3–8 originally
estimated here. The reason is structural and worth stating plainly: **nightly cost
scales with TOPICS DUG, not units changed.** Six units triaged into four topics, and
each topic then costs a distill + falsify (+ reconcile) + consolidate. At the stated
daily volume of 10–20 sessions essentially every topic gets touched every day, which
would have made the nightly pass ~20+ calls forever — a "background" layer that is
permanently expensive, i.e. exactly the failure the two-mode split exists to prevent.

So digs are **batched across nights**. Triage assignments are persisted in
`dream_pending`, and a topic is only dug once it has `DIG_MIN_UNITS` (3) pending
units, or its oldest pending unit has waited `DIG_MAX_DEFER_DAYS` (7). Nothing is
lost by waiting — the assignment is in the table, not in a variable — and a
knowledge layer does not care whether a position is distilled tonight or Thursday.
Deferred backlogs are reported by `dream run` and `dream status`, because a night
that digs nothing must be distinguishable from a night that is broken.

This queue also fixes a **silent data-loss bug** the E2E exposed: the dig takes at
most `DIG_BATCH_UNITS` units, and the watermark advanced regardless — so on a busy
day the surplus units were dropped and never distilled. Now only the units actually
mined leave the queue.

Per-call figures **measured** from `dream_queue.usage` (in = `input +
cache_creation + cache_read`; small samples, n in the table):

| stage | measured per call (in / out) | n |
|---|---|---|
| triage — all new/changed units, batched (15/call) | 5.5k / 0.8k | 2 |
| distill — one call per topic dug | 13.0k / 8.8k | 5 |
| falsify — batched per topic dug | 3.0k / 4.9k | 5 |
| reconcile — only if the topic already holds insights | ~2k / ~3k | 1 |
| consolidate — only digests whose insight set changed | 1.1k / 0.4k | 5 |

| nightly pass | calls |
|---|---|
| no changed units | **0** (returns before triage) |
| changed units, no topic ripe | **1** (the triage that decided so) |
| one ripe topic | **4–5** |
| worst case, all 6 topics ripe the same night | ~20, and only every ~3rd night |

Nothing in the nightly path re-reads history.

**Bulk is where the cost lives, and it is a separate, explicit mode.** A full
first `dream backfill` across 6 topics and 603 units is on the order of 100–200
calls and several million tokens — rationed by `--max-calls` and resumable, so it
can be spread over nights or driven hard in one attended session. It then does not
recur except when a topic is added or a charter is rewritten.

This split is the reason budget governance is a D-level decision: the two modes
have cost profiles two orders of magnitude apart and must not share a code path
that lets backfill leak into the nightly run.

### D10. Module boundaries

`dream.py` — topics, dig loop, worker invocation, insight lifecycle. It *calls*
`search.hybrid_search` and `clync.connect_pg`; it does not reimplement retrieval
or own the store. `search.py` gains only the dream-tier ranking and the `'dream'`
source. `clync.py` gains the schema and the CLI wiring. No store or query logic
is duplicated — the ADR 0003 boundary holds.

## Consequences

- **clync gains a second question it can answer.** "What do I think about X",
  with citations, stance, and a change history — not just "where did I say X".
- **The layer is auditable end to end.** Every insight resolves to real quotes in
  real units; `dream_queue.usage` records what each derivation cost; supersession
  chains make position changes inspectable rather than destructive.
- **Derived data stays derivable.** Insights are rebuildable from raw transcripts
  the same way `chunks` are. A lost Dream layer is a re-run, not data loss.
- **New failure mode: a confidently wrong insight.** Mitigated by the grounding
  gate (mechanical), falsification (adversarial), stance (no false attribution),
  and no recursion (no amplification) — but not eliminated. Retraction is a
  first-class state for this reason.
- **The store's schema grows** by four tables and two enum values. Accepted:
  the alternative is a parallel text store and a second retrieval path.
- **Ongoing subscription consumption is near-negligible.** ~3–8 calls a night in
  steady state; one call on a quiet day. The one-time bulk dream is the only
  material spend, and it is explicitly invoked and rationed.

## Open questions (need a decision before implementation)

1. **Insight granularity.** Target ~50 or ~500 insights per topic? Drives whether
   digests stay readable and whether reconcile stays cheap.
2. **cc sessions are noisy.** A large share are mechanical (fix CI, rerun tests)
   and carry no durable knowledge. Is batched skeleton triage enough to drop them,
   or should there be a cheap pre-filter before triage even sees them?
3. **Bulk timing.** Ration the first `dream backfill` across nights, or drive it
   in one attended session? (Recommend: one topic end-to-end first, to validate
   prompt and gate quality on real output before committing all six.)

**Resolved:** digest shape — fixed structured sections (D2). Nightly cost — the
incremental/bulk mode split (D4, D9); nightly must never re-sweep history.
