# ADR 0004 — The Dream layer: topic-driven knowledge distillation

- **Status:** Accepted — implemented in `dream.py`. **Revised 2026-07-27** after
  measuring what the built layer actually read and actually cost. Three decisions
  changed rather than being patched around, each marked "Revised 2026-07-27" in
  place with the measurement that forced it:
  - **D4** — the dig read each unit's OPENING, not the passage retrieval had
    matched. 42 of 45 matches (93%) fell outside the rendered window. It now
    renders the hit plus surrounding turns, and the dreamer can request more.
  - **D2** — the stored, model-written digest is deleted. It was the only
    ungrounded text in the store, a precomputed answer to an unasked question, and
    it had already gone stale in a way that made recall report "nothing distilled"
    above a list of distilled insights. Rendered from the insight rows at read time.
  - **D1/D4** — `reconcile` is folded into `falsify`, which was already shown the
    same held-insight list. A ripe topic costs 3 calls, down from 5.
  The cost model in D9 and the topic-status vocabulary in D3 were corrected by
  measurement during the original build; they describe what the code does.
- **Date:** 2026-07-26 (revised 2026-07-27)
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
Opus at `--effort low`; distillation and falsification run at default effort. The
invocation is the measured minimal form from Context, always with `--json-schema`.

Every call is a pure function of its prompt: no tools (`--tools ""`), no MCP, no
settings, no session. The Dream worker never reads the filesystem and never
writes anywhere except through `dream.py`'s own DB writes. This is what makes the
layer auditable and re-runnable.

**Revised 2026-07-27 — why navigation did NOT become a tool.** The dreamer needs
to read further into a conversation than its seed window (D4), and the obvious
shape is a scoped read-only MCP tool. Measured on this CLI (2.1.220, one probe
set, prefill = input + cache_creation + cache_read):

| worker configuration | prefill | nav tool reachable |
|---|---|---|
| `--tools ""` (this layer's flags) | 515 floor, ~2.1k real | no |
| `--tools ""` + `--mcp-config` | 659 | **no** — `""` strips MCP tools too |
| `--mcp-config`, no `--tools` | 85,709 | yes |
| `--tools "Read"` + `--mcp-config` | 5,421 | yes |

`--tools` governs the BUILT-IN set only, and MCP tools ride along only when a
non-empty built-in selection is present. So a nav tool costs a 10x prefill floor
AND forces a filesystem-wide `Read` schema into every call, scoped only by a
permission denial rather than by absence. Rejected.

Instead the dreamer returns `read_requests` in its structured output and **code**
fulfils them, re-rendering and re-invoking (`MAX_NAV_ROUNDS`, currently two —
i.e. two asks can be served. One bound on every real dig measured, 3/3: the first
ask was served but the informed follow-up, made after seeing the wider window, was
always refused. Two serves that. It does not end the tail — a large transcript
always invites another ask — so this is a deliberate budget, and unserved requests
are reported to keep it auditable). Same capability, fewer parts: no
MCP process, no second tool surface, no
new trust boundary, prefill unchanged, and reads are scoped to raw sources *by
construction* and logged because code performs them. The purity property above
therefore survives navigation — which a tool-using session would have ended.

**Fail loud:** a call whose result JSON has `is_error`, a non-`success` `subtype`,
or a missing `structured_output` raises. There is no "skip this item and carry
on" path — the item is marked `failed` with the raw error retained, the run
aborts if failures exceed a threshold, and `clync doctor` surfaces it.

### D2. One stored level, and a digest rendered at read time

- **`dream_insight`** — one atomic claim, and the ONLY thing this layer stores.
  "Prefer fixing at the layer that should have prevented the bug over the layer
  where it surfaced." Carries statement, elaboration, stance, evidence, validity
  window.
- **The digest** — one per topic: the current state of the user's thinking on that
  topic, in **fixed structured sections**, assembled from its *active* insight rows
  by `dream.render_digest` **when it is asked for**. It is not stored and there is
  no model call in it.

  | section | contents |
  |---|---|
  | **Settled** | active insights, stance `user_asserted`/`user_endorsed`, with support count |
  | **Rejected approaches** | active insights with stance `user_rejected` — what was argued against, and why |
  | **Changed positions** | supersessions recorded as `contradict` (never `refine`): `old → new`, dated from the validity windows |
  | **Open** | insights the falsification gate marked contested |

**Revised 2026-07-27.** This ADR originally specified a stored `dream_digest`
unit, written by a `consolidate` model call that composed connective prose around
bullets code had already assembled. That call is deleted. Three reasons, in order
of weight:

1. **It was the only ungrounded text in the store.** Every other artifact traces
   to a verbatim quote through a mechanical gate. The digest's prose traced to
   nothing, and grounding it would have meant building a whole second
   verification path for text nobody had asked for.
2. **It was a precomputed answer to an unasked question.** A reader arrives with
   an actual question; a nightly synthesis cannot know it. Rendering the sections
   on read lets the caller (or the model reading them) synthesize with the
   question in hand, which is strictly better than a stored guess at it.
3. **A stored summary of rows that keep changing is a staleness bug waiting.** It
   had already produced one: a budget-capped backfill spent its whole allowance
   digging a topic and then never wrote the digest, so recall reported "nothing
   distilled" above a list of twenty distilled insights, permanently.

Deleting it also removed a nightly model call per changed topic, a `units` row per
topic, and the `dream-digest-` id prefix that was duplicated across three call
sites. Rows of the retired kind are cleaned up on provisioning, since stale
synthesis prose left behind would go on answering `search_history`.

The non-recursion guarantee is unchanged and now trivially true: insights derive
**only** from raw transcript evidence (`source='all'` means raw only, and the
grounding gate rejects any citation to a non-raw unit), and the digest is a pure
function of insight rows computed at read time. Derived text is never input to
another derivation, so hallucination amplification is capped at one hop — the
specific failure RAPTOR measures, which a deeper ladder compounds.

There is no level 3. Cross-topic synthesis, if ever wanted, is a query-time
composition, not a stored tier.

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
3. **Queue, then wait for ripeness.** A triaged unit is queued against its topic
   in `dream_pending`; the topic is dug once `DIG_MIN_UNITS` have accumulated or
   its oldest pending unit has waited `DIG_MAX_DEFER_DAYS`. Cost scales with
   *topics dug*, not units changed — measured, digging every flagged topic
   immediately cost 14 calls for one ordinary day. Nothing is lost by waiting:
   the queue is a table, and only units actually mined leave it.
4. **Seed from the retrieval HIT, and render the window around it.** For each
   unit, the position of the matching chunk (`msg_idx`, carried out of
   `hybrid_search`) plus `WINDOW_BEFORE`/`WINDOW_AFTER` neighbouring turns, with
   the matched turn marked and the unit's true size stated.

   **Revised 2026-07-27.** This step originally said "condense": order the unit's
   messages by index and take turns until a char budget runs out. That reads the
   conversation's OPENING. Retrieval had located the unit by a chunk match and the
   match was then discarded. Measured against the committed pre-fix code over 45
   (query, unit) pairs on units with >= 40 messages: **42/45 = 93% of matches fell
   outside the rendered window**, median match at message index 1284 against a
   median 23 messages rendered, every window starting at index 0; worst case a
   23,653-message unit matched at index 20,927 and rendered from 0..19. Every
   insight the layer held had been distilled from a preamble.

   A window is a guess whatever size it is, so the dreamer can ask for more:
   `read_requests` fulfilled by code, not a tool (D1). The size budget within a
   range is spent OUTWARD FROM THE MATCH — spending it from the range start let a
   wide requested expansion evict the matched message, reintroducing this very bug
   through the feature meant to fix it.
5. **Distill** — one call per *topic* (not per unit), schema-constrained,
   emitting candidate insights with mandatory evidence refs. Refs are
   `<slot>#<index>` and code owns the mapping, so the model never handles a uuid:
   shown a correct 36-char id, a real dig transposed one digit and the grounding
   gate correctly discarded 3 of 15 insights. Making the ref a RULE rather than a
   table is also what lets navigation cite messages no earlier rendering showed.
6. **Falsify** — batched per topic (D6). The new/reinforce/refine/contradict
   decision rides on the same call: falsify already had to be shown the held
   insights (a candidate that merely restates one must be rejected), which is
   exactly what that decision needs, so the separate `reconcile` call was
   redundant and is gone. Its targets are `[hN]` labels for the same
   no-uuids-in-model-output reason.

Every stage is gated by the one before it. A quiet day costs one triage call and
nothing else; a ripe topic costs **3 calls** (triage + distill + falsify), with a
fourth only when the dreamer asks to read further. It was 5.

`distill` and `falsify` are deliberately NOT merged, unlike `reconcile`:
generating and refuting in one completion is self-review, and the independent pass
measurably earns its keep, having rejected 10 of 14 and 7 of 27 candidates with
substantive reasoning.

#### Bulk (explicit, one-time-ish) — `dream backfill`

The *deep slate*: unrestricted `sort=relevance` retrieval per probe query, minus
the transcript RANGES already in the topic's ATTEMPT record (`dream_attempted`),
walked in batches until the topic's candidate pool is exhausted. Never runs on
the nightly schedule. Invoked deliberately, capped by `--max-calls`, resumable
from `dream_attempted`.

Progress deliberately derives from the **attempt**, never from what survived the
gates: the gates reject whole batches by design (10 of 14, 7 of 27 measured), so
inferring "already mined" from `dream_evidence` re-selected every fully-rejected
batch at the same rank and re-dug it until the budget expired. The same record
sets `dream_topics.last_dig_at`, so recall's coverage tier reports "never been
dug" only when that is literally true.

The attempt record is exactly the work performed, no more. It is written only
AFTER a distill call returns — a throttled or crashed call read nothing, and a
pre-written row would exclude its batch from every later backfill while coverage
reported it dug. And it is scoped to the (unit, from_idx, to_idx) ranges the
call was actually shown, never the whole unit: a dig reads a ~15-message window
around one retrieval hit, and marking the unit mined on that basis would retire
a 20k-message conversation off 0.1% of it — the exact "dug across all of
history" property this mode exists for. Exclusion is therefore per HIT (a hit
inside a recorded range is skipped; one outside re-enters the pool even in a
unit already dug elsewhere), a unit retires only when its ranges cover its hits,
and recall's coverage tier states the unrendered remainder from the same record.

`dream backfill` is also the right tool after adding a new topic or materially
rewriting a charter — those are the only recurring reasons to re-sweep history.

The topic's probe queries are revisable in both modes: a distill call may propose
query additions when it finds vocabulary the probes missed. That is the "repeated
dig" — each run's understanding sharpens the next run's retrieval, without
re-reading history. **Bounded** (`PROBE_QUERIES_MAX`): every dig could propose more
and nothing removed any, so one live topic reached 49 queries from 5 seeds, and
`candidates` runs one hybrid search per query. Past the cap the newest learned
queries displace the oldest; the hand-written seeds are never displaced, since they
are the charter expressed as retrieval.

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

Reconciliation against existing insights rides on the falsify verdict (D4), and
its outcome is one of:

- **reinforce** — same claim, new evidence → append refs, extend
  `last_seen_at`; `support_count` is *derived* (distinct cited source units),
  never incremented, so re-mining the same unit cannot inflate it.
- **refine** — same claim, better wording → new version, old row
  `status='superseded'`, `superseded_by` set, `superseded_kind='refine'`,
  `valid_until = new.valid_from` (every supersession CLOSES the window; the
  *kind* — not a NULL timestamp — is what says no change of mind happened).
- **contradict** — the user's position changed → old row gets
  `valid_until = new.valid_from`, `status='superseded'`,
  `superseded_kind='contradict'`; the new row is active. **Nothing is
  deleted.** The history stays queryable, so the layer can answer "when did I
  change my mind about X". A supersession whose new evidence *predates* the
  target's `valid_from` is refused (it would invert the timeline) and lands as
  a contested new row instead.
- **new** — no match.

Retraction (`status='retracted'`) is reserved for insights whose evidence
disappeared (source unit purged) — mirroring the existing deletion-reconciliation
behaviour of `cc_sync_state`.

### D7. Schema

Insights are **`units` rows**, extended 1:1 — not a parallel text store. This is
the SSOT-preserving choice: `search.py` already iterates all units, so insights
get BGE-M3 dense+sparse indexing, facets, and `get_conversation` with no second
retrieval path.

**Revised 2026-07-27.** `dream_digest` is gone with the `consolidate` call (D2):
the digest is rendered from insight rows at query time, and `ensure_dream_schema`
deletes any `kind='dream_digest'` unit left behind. `dream_evidence.src_msg_id`
and `quote` are `NOT NULL` — an evidence row that cannot name its message or
reproduce its quote cannot be re-checked by the gate that admitted it, so it is
not a row. `dream_insights.contested` is listed below. **`DREAM_SCHEMA` in
`dream.py` is the SSOT**; this block is illustrative and a divergence between the
two is a bug in this document.

```sql
-- units gains: kind ∈ {…, 'dream_insight'}
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
    superseded_kind text,                -- 'refine' | 'contradict' (why superseded)
    support_count int NOT NULL,          -- derived: distinct cited source units
    contested     boolean NOT NULL,      -- a contradicting insight also holds
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
    src_msg_id  text NOT NULL,
    quote       text NOT NULL,           -- must verify against messages.text
    PRIMARY KEY (unit_id, src_unit_id, src_msg_id)
);

CREATE TABLE dream_queue (               -- resumable, budget-governed work
    item_id    text PRIMARY KEY,
    kind       text NOT NULL,            -- triage | distill | falsify
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

CREATE TABLE dream_attempted (           -- the ATTEMPT record (see D4, bulk mode):
    topic_id  text NOT NULL REFERENCES dream_topics(topic_id) ON DELETE CASCADE,
    unit_id   text NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
    from_idx  int NOT NULL,              -- the ranges actually rendered to the
    to_idx    int NOT NULL,              -- distiller, written after the call returns
    dug_at    timestamptz NOT NULL,
    PRIMARY KEY (topic_id, unit_id, from_idx)
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

- `dream run [--max-calls N]` — the **incremental** mode (D4). No `--topic`: the
  nightly pass digs whatever triage flagged, and restricting that to one topic
  would leave the other topics' pending units queued behind an advanced
  watermark. Use `dream backfill --topic T` to work one topic deliberately.
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
each topic then cost a distill + falsify + reconcile + consolidate. At the stated
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

`reconcile` (~2k/~3k) and `consolidate` (1.1k/0.4k) were measured and then DELETED
— see D2 and D4. Distill's input rises somewhat with hit-centered windows (it
renders more of each unit than the old 6k-char opening) and rises again on a
navigation round; the call COUNT is what the nightly budget is governed by.

| nightly pass | calls |
|---|---|
| no changed units | **0** (returns before triage) |
| changed units, no topic ripe | **1** (the triage that decided so) |
| one ripe topic | **3** (4 if the dreamer asks to read further) |
| worst case, all 6 topics ripe the same night | ~13-19, and only every ~3rd night |

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
- **The store's schema grows** by five tables and two enum values. Accepted:
  the alternative is a parallel text store and a second retrieval path.
- **Ongoing subscription consumption is near-negligible.** ~3–8 calls a night in
  steady state; one call on a quiet day. The one-time bulk dream is the only
  material spend, and it is explicitly invoked and rationed.

## Open questions (need a decision before implementation)

1. **cc sessions are noisy.** A large share are mechanical (fix CI, rerun tests)
   and carry no durable knowledge. Is batched skeleton triage enough to drop them,
   or should there be a cheap pre-filter before triage even sees them?
2. **Bulk timing.** Ration the first `dream backfill` across nights, or drive it
   in one attended session? (Recommend: one topic end-to-end first, to validate
   prompt and gate quality on real output before committing all six.)

**Resolved:** digest shape — fixed structured sections (D2). Nightly cost — the
incremental/bulk mode split (D4, D9); nightly must never re-sweep history.
Insight granularity / falsify's held-insight listing — the listing is bounded to
`FALSIFY_HELD_MAX` held insights nearest the candidates by dense similarity (the
vectors the index already holds), so per-dig cost no longer grows with the
topic's total insight count.
