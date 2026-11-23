"""Index-free lexical session search; source text is evidence, never instructions."""
from __future__ import annotations

import json
import hashlib
import re
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import live_files
from clync import DEFAULT_TOPK
from live_types import Record, Session

SOURCES = ("claude_code", "codex_cli", "opencode", "grok", "pi")
NATIVE_SOURCES = ("claude_code", "codex_cli", "pi")
PASSAGE_CHARS = 1200
READ_CHARS = 16000
SOURCE_READERS = 8
COMPLETE = "complete"
PARTIAL = "partial"
UNAVAILABLE = "unavailable"
_WIDTH_FOLD = str.maketrans({code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)})
_WORDS = re.compile(r"[^\W_]+")


def _normalize(text: str) -> str:
    return text.translate(_WIDTH_FOLD).lower()


def _terms(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text.translate(_WIDTH_FOLD))
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", text)
    return list(dict.fromkeys(_WORDS.findall(text.lower())))


def _one_edit(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    i = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
    if i == min(len(left), len(right)):
        return True
    if len(left) == len(right):
        return (left[i + 1:] == right[i + 1:] or
                (i + 1 < len(left) and left[i] == right[i + 1] and left[i + 1] == right[i]
                 and left[i + 2:] == right[i + 2:]))
    if len(left) < len(right):
        left, right = right, left
    return left[i + 1:] == right[i:]


def _needles(terms: list[str], fuzzy: bool) -> list[str]:
    word = max(terms, key=len)
    needles = [word]
    if fuzzy and len(word) >= 5 and word.isascii():
        middle = len(word) // 2
        # A transposition across the partition can break both otherwise-conservative halves.
        needles.extend((
            word[:middle], word[middle:],
            word[:middle - 1] + word[middle] + word[middle - 1] + word[middle + 1:]))
    characters = set(word + word.upper())
    characters.update(chr(ord(c) + 0xFEE0) for c in list(characters) if 0x21 <= ord(c) <= 0x7E)
    escapes = {f"{int.from_bytes(c.encode('utf-16-be')[i:i + 2], 'big'):04x}"
               for c in characters for i in range(0, len(c.encode('utf-16-be')), 2)}
    specials = [c for c in ("K", "İ") if c.lower()[0] in word]
    return [*needles, *(r"\u" + escape for escape in sorted(escapes)),
            *(c for c in sorted(characters) if 0xFF01 <= ord(c) <= 0xFF5E), *specials]


def _match(text: str, terms: list[str], fuzzy: bool) -> tuple[float, str, int] | None:
    normalized = _normalize(text)
    gates = {}
    for term in terms:
        if fuzzy and len(term) >= 5 and term.isascii():
            a, b = len(term) // 3, 2 * len(term) // 3
            # A single edit preserves two thirds, except for swaps across either partition.
            gates[term] = (term[:a], term[a:b], term[b:],
                           term[:a - 1] + term[a] + term[a - 1] + term[a + 1:],
                           term[:b - 1] + term[b] + term[b - 1] + term[b + 1:])
    for start in range(0, max(1, len(normalized)), PASSAGE_CHARS // 2):
        passage = normalized[start:start + PASSAGE_CHARS]
        words = None
        exact = 0
        for term in terms:
            if term in passage:
                exact += 1
                continue
            anchors = gates.get(term)
            if anchors is None or (sum(part in passage for part in anchors[:3]) < 2
                                   and anchors[3] not in passage and anchors[4] not in passage):
                break
            if words is None:
                words = set(_WORDS.findall(passage))
            length = len(term)
            if not any(_one_edit(term, word) for word in words
                       if length - 1 <= len(word) <= length + 1
                       and (word[0] == term[0] or word[-1] == term[-1])):
                break
        else:
            score = (exact + 0.7 * (len(terms) - exact)) / len(terms)
            position = start
            if len(normalized) != len(text):
                low, high = 0, len(text)
                while low < high:
                    middle = (low + high) // 2
                    if len(_normalize(text[:middle])) < start:
                        low = middle + 1
                    else:
                        high = middle
                position = low
            return score, text[position:position + PASSAGE_CHARS], position
    return None


def _discover(source: str, include_subagents: bool) -> list[Session]:
    if source == "pi":
        import pi_source
        return pi_source.sessions(include_subagents)
    if source in SOURCES[:2]:
        return live_files.sessions(source, include_subagents)
    import live_external
    return getattr(live_external, f"{source}_sessions")(include_subagents=include_subagents)


def _external_records(session: Session, include_tools: bool) -> list[Record]:
    import live_external
    return getattr(live_external, f"{session.source}_records")(session, include_tools=include_tools)


def _date(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _eligible(session: Session, repo: str | None, since: datetime | None) -> bool:
    if repo and session.cwd and repo != session.metadata.get("repo") and repo not in Path(session.cwd).parts:
        return False
    return since is None or _date(session.updated_at) >= since


def _public(session: Session) -> dict:
    return {"source": session.source, "session_id": session.session_id,
            "reference": f"{session.source}:{session.session_id}", "title": session.title[:120],
            "cwd": session.cwd, "updated_at": session.updated_at, "parent_id": session.parent_id}


def _content_anchor(record: Record, text_offset: int = 0) -> str:
    return record.anchor + "@" + hashlib.sha256(record.text.encode()).hexdigest()[:16] + f"#{text_offset}"


def search_sessions(query: str, *, source: str = "all", repo: str | None = None,
                    sender: str | None = None, since: str | None = None, limit: int = DEFAULT_TOPK,
                    include_tools: bool = False, include_subagents: bool = False,
                    exact: bool = False, exhaustive: bool = False) -> dict:
    terms = _terms(query)
    selected = list(SOURCES) if source == "all" else list(dict.fromkeys(source.split(",")))
    if not terms or any(s not in SOURCES for s in selected):
        raise ValueError("A text query and known live sources are required")
    if sender not in (None, "user", "assistant", "tool") or not 1 <= limit <= 100:
        raise ValueError("Invalid sender or limit (1-100)")
    since_date = _date(since)
    started = time.perf_counter()
    coverage, native, external, hits = {}, [], [], {}
    candidate_mode = "exhaustive" if exhaustive else "anchored"

    def accept(session: Session, record: Record):
        role = "user" if record.sender == "user_answer" else record.sender
        if sender and role != sender:
            return
        found = _match(record.text, terms, not exact)
        if found is None:
            return
        score, passage, position = found
        if record.kind == "title":
            score -= 0.05
        reference = f"{session.source}:{session.session_id}"
        at = record.anchor
        if record.kind != "title":
            at = _content_anchor(record, position)
        hit = {**_public(session), "anchor": at, "sender": record.sender,
               "kind": record.kind, "text": passage, "score": score}
        if reference not in hits or hits[reference]["score"] < score:
            hits[reference] = hit

    def scan(needles, decode_all):
        try:
            for session, offset, raw in live_files.candidates(
                    native, needles, decode_all, all(term.isascii() for term in terms)):
                cov = coverage[session.source]
                cov["candidates"] += 1
                if offset >= session.metadata["size"] or not raw.endswith(b"\n"):
                    cov["status"] = PARTIAL
                    continue
                if not live_files.searchable_record(session.source, raw, include_tools):
                    continue
                try:
                    cov["decoded_records"] += 1
                    event = json.loads(raw)
                    if not session.cwd and event.get("cwd"):
                        session.cwd = event["cwd"]
                    if not _eligible(session, repo, since_date):
                        continue
                    if repo and not session.cwd and not session.metadata.get("repo"):
                        raise ValueError("Repository metadata unavailable")
                    records = live_files.records(session, event, "", include_tools)
                    at = live_files.anchor(offset, raw) if records else ""
                    for ordinal, record in enumerate(records):
                        record.anchor = f"{at}:{ordinal}"
                        accept(session, record)
                except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as error:
                    cov["status"] = PARTIAL
                    message = f"Invalid candidate in {session.session_id}: {type(error).__name__}"
                    if message not in cov["errors"]:
                        cov["errors"].append(message)
        except (OSError, RuntimeError) as error:
            for session in native:
                coverage[session.source]["status"] = PARTIAL
                if str(error) not in coverage[session.source]["errors"]:
                    coverage[session.source]["errors"].append(str(error))

    with ThreadPoolExecutor(max_workers=SOURCE_READERS) as pool:
        pending = {pool.submit(_discover, s, include_subagents): s for s in selected if s != "opencode"}
        if "opencode" in selected:
            from live_opencode import Snapshot

            cov = coverage["opencode"] = {"status": COMPLETE, "sessions": 0, "candidates": 0,
                                           "decoded_records": 0, "errors": []}
            discovered = False
            try:
                with Snapshot() as snapshot:
                    sessions = [s for s in snapshot.sessions(include_subagents) if _eligible(s, repo, since_date)]
                    discovered = True
                    cov.update(sessions=len(sessions), storage="native SQLite snapshot", version=snapshot.info["version"])
                    for session in sessions:
                        if not sender and session.title:
                            accept(session, Record("metadata", "metadata", session.title, "title"))
                    for session, row in snapshot.candidates(sessions, _needles(terms, not exact), include_tools):
                        cov["candidates"] += 1
                        try:
                            cov["decoded_records"] += 1
                            for record in snapshot.records(row, include_tools):
                                accept(session, record)
                        except (ValueError, TypeError, KeyError, sqlite3.Error) as error:
                            cov["status"] = PARTIAL
                            cov["errors"].append(f"{session.session_id}/{row['id']}: {error}")
                if snapshot.changed:
                    cov["status"] = PARTIAL
                    cov["errors"].append("OpenCode source changed during search; results use the earlier SQLite snapshot")
            except (OSError, ValueError, TypeError, KeyError, RuntimeError, sqlite3.Error,
                    subprocess.TimeoutExpired) as error:
                cov["status"] = PARTIAL if discovered else UNAVAILABLE
                cov["errors"].append(str(error))
        for future in as_completed(pending):
            name = pending[future]
            coverage[name] = {"status": COMPLETE, "sessions": 0, "candidates": 0,
                              "decoded_records": 0, "errors": []}
            try:
                sessions = [s for s in future.result() if _eligible(s, repo, since_date)]
            except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
                coverage[name].update(status=UNAVAILABLE, errors=[str(error)])
                continue
            coverage[name]["sessions"] = len(sessions)
            if name == "pi" and sender == "user":
                coverage[name]["status"] = PARTIAL
                coverage[name]["errors"].append(
                    "Pi user-role messages have no verified human provenance; omit sender to search pi_user text.")
            for session in sessions:
                if session.metadata.get("error"):
                    coverage[name]["status"] = PARTIAL
                    coverage[name]["errors"].append(session.metadata["error"])
                    continue
                if not sender and session.title:
                    accept(session, Record("metadata", "metadata", session.title, "title"))
                if session.path is not None and name in NATIVE_SOURCES:
                    native.append(session)
                    if session.metadata.get("pending_tail"):
                        coverage[name]["status"] = PARTIAL
                        coverage[name]["errors"].append(f"Incomplete trailing record: {session.session_id}")
                else:
                    external.append(session)

        external_pending = {pool.submit(_external_records, s, include_tools): s for s in external}
        if native:
            anchors = [term for term in terms if len(term) >= 3] or terms
            needles = list(dict.fromkeys(n for term in anchors for n in _needles([term], False)))
            scan(needles, exhaustive)
        for future in as_completed(external_pending):
            session = external_pending[future]
            cov = coverage[session.source]
            try:
                for record in future.result():
                    cov["candidates"] += 1
                    accept(session, record)
                warnings = session.metadata.get("warnings", [])
                if warnings:
                    cov["status"] = PARTIAL
                    cov["errors"].extend(warnings)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
                cov["status"] = PARTIAL
                cov["errors"].append(f"{session.session_id}: {error}")
        if native and not hits and not exact and not exhaustive:
            candidate_mode = "broadened"
            scan(_needles(terms, True), False)
        for session in native:
            try:
                stat = session.path.stat()
                if (stat.st_mtime_ns, stat.st_size) == (session.metadata["mtime_ns"], session.metadata["size"]):
                    continue
                message = f"Source changed during search: {session.session_id}"
            except OSError as error:
                message = f"Source unavailable after search: {session.session_id}: {error}"
            coverage[session.source]["status"] = PARTIAL
            coverage[session.source]["errors"].append(message)
    ranked = sorted(hits.values(), key=lambda h: (-h["score"], -_date(h["updated_at"]).timestamp(), h["reference"]))
    return {"query": query, "matching": "exact" if exact else "one-edit (ASCII terms of length >=5)",
            "candidate_mode": candidate_mode,
            "matching_coverage": ("all passages" if exhaustive or exact or candidate_mode == "broadened"
                                  or not any(s in NATIVE_SOURCES for s in selected)
                                  else "anchored passages and titles; use exhaustive for fuzzy-only matches"),
            "observed_at": datetime.now(UTC).isoformat(), "coverage": coverage,
            "elapsed_ms": round((time.perf_counter() - started) * 1000), "hits": ranked[:limit]}


def read_session(reference: str, anchor: str, *, context: int = 3,
                 include_tools: bool = False) -> dict:
    source, separator, sid = reference.partition(":")
    valid_id = (re.fullmatch(r"[A-Za-z0-9_.-]+", sid) and sid not in (".", "..")
                if source == "pi" else re.fullmatch(r"[A-Za-z0-9_-]+", sid))
    if not separator or source not in SOURCES or not valid_id:
        raise ValueError("Invalid live session reference")
    if not 0 <= context <= 10:
        raise ValueError("Context must be between 0 and 10")
    session = next((s for s in _discover(source, False) if s.session_id == sid), None)
    if session is None:
        session = next((s for s in _discover(source, True) if s.session_id == sid), None)
    if session is None:
        raise ValueError("Live session no longer exists")
    text_offset = 0
    if anchor != "metadata":
        anchor, separator, position = anchor.rpartition("#")
        if not separator or not position.isdecimal():
            raise ValueError("Invalid passage offset")
        text_offset = int(position)
    at, separator, digest = anchor.rpartition("@")
    if anchor != "metadata" and (not separator or not re.fullmatch(r"[0-9a-f]{16}", digest)):
        raise ValueError("Invalid live content anchor")
    if source == "pi":
        import live_pi
        if session.metadata.get("error"):
            raise ValueError(session.metadata["error"])
        window = live_pi.read(session, "metadata" if anchor == "metadata" else at, context,
                              include_tools, expected_hash=digest if anchor != "metadata" else None)
    elif session.path is not None and source in SOURCES[:2]:
        if anchor == "metadata":
            with session.path.open("rb") as stream:
                at = live_files.anchor(0, stream.readline())
        window = live_files.read(session, at, context, include_tools,
                                 expected_hash=digest if anchor != "metadata" else None)
    else:
        records = _external_records(session, include_tools)
        index = 0 if anchor == "metadata" else next((i for i, r in enumerate(records) if r.anchor == at), -1)
        if index < 0:
            raise ValueError("Stale source anchor; search again")
        if anchor != "metadata" and hashlib.sha256(records[index].text.encode()).hexdigest()[:16] != digest:
            raise ValueError("Stale source anchor; search again")
        window = {"records": records[max(0, index - context):index + context + 1], "window_bounded": True}
    output, remaining = [], READ_CHARS
    for record in window["records"]:
        item = asdict(record)
        matched = record.anchor == at and anchor != "metadata"
        item["anchor"] = _content_anchor(record, text_offset if matched else 0)
        allowance = min(remaining, READ_CHARS // (2 * context + 1))
        if matched:
            allowance = min(remaining, max(allowance, PASSAGE_CHARS * 2))
        item["truncated"] = len(record.text) > allowance
        if matched and text_offset > len(record.text):
            raise ValueError("Passage offset outside source record")
        start = max(0, text_offset - allowance // 4) if matched else 0
        item["text_offset"] = start
        item["text"] = record.text[start:start + allowance]
        output.append(item)
        remaining -= len(item["text"])
        if remaining == 0:
            break
    return {**_public(session), "records": output, "window_bounded": True,
            "warnings": [*session.metadata.get("warnings", []), *window.get("warnings", [])],
            "truncated": any(r["truncated"] for r in output) or remaining == 0}


def format_search(result: dict) -> str:
    lines = [f"Live search: {result['query']!r} ({result['elapsed_ms']} ms; {result['matching']})"]
    lines.append(f"Matching coverage: {result['candidate_mode']}; {result['matching_coverage']}")
    for source, cov in sorted(result["coverage"].items()):
        lines.append(f"{source}: {cov['status']}; {cov['sessions']} session(s)")
        lines.extend(f"  {error}" for error in cov["errors"])
    for hit in result["hits"]:
        lines.append(f"\n[{hit['source']}] {hit['title'] or hit['session_id']}\n"
                     f"  reference={hit['reference']} anchor={hit['anchor']} sender={hit['sender']}\n"
                     f"  {hit['text'][:500]}")
    if not result["hits"]:
        lines.append("No verified matches in the reported coverage.")
    return "\n".join(lines)
