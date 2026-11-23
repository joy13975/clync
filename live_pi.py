"""Pi live readback follows native tree edges, never append-order siblings."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import live_files
import pi_source
from live_types import Session


def read(session: Session, at: str, context: int, include_tools: bool,
         *, expected_hash: str | None) -> dict:
    match = re.fullmatch(r"(\d+:[0-9a-f]{16}):(\d+)", at) if at != "metadata" else None
    if at != "metadata" and match is None:
        raise ValueError("Invalid Pi file anchor")
    meta, entries = pi_source.load_entries(session.path)
    if meta["id"] != session.session_id:
        raise ValueError("Stale Pi session identity; search again")
    children, parents, by_id, ordered = defaultdict(list), {}, {}, []
    target = None
    for offset, raw, event in entries:
        eid, parent = event["id"], event["parentId"]
        children[parent].append(eid)
        parents[eid] = parent
        items = pi_source.records(event, "", include_tools)
        for ordinal, record in enumerate(items):
            record.anchor = f"{live_files.anchor(offset, raw)}:{ordinal}"
            ordered.append((eid, record))
        by_id[eid] = items
        if match is not None and offset == int(match[1].split(":", 1)[0]):
            if live_files.anchor(offset, raw) != match[1]:
                raise ValueError("Stale Pi source anchor; search again")
            ordinal = int(match[2])
            if ordinal >= len(items):
                raise ValueError("Pi anchor content is not available with these content filters")
            target = eid, ordinal
            if hashlib.sha256(items[ordinal].text.encode()).hexdigest()[:16] != expected_hash:
                raise ValueError("Pi anchor content changed or has different content filters; search again")
    if at == "metadata" and ordered:
        target = ordered[0][0], 0
    if target is None:
        raise ValueError("Stale or empty Pi source anchor; search again")
    eid, ordinal = target
    ancestors = []
    parent = parents[eid]
    while parent is not None:
        ancestors.append(parent)
        parent = parents[parent]
    before = [record for ancestor in reversed(ancestors) for record in by_id[ancestor]]
    before.extend(by_id[eid][:ordinal])
    matched = by_id[eid][ordinal]
    after = list(by_id[eid][ordinal + 1:])
    warnings = [pi_source.HISTORY, "Pi user-role text is pi_user, not verified human authorship."]
    while len(after) < context and children[eid]:
        choices = children[eid]
        if len(choices) > 1:
            warnings.append(f"Branch ambiguity after entry {eid}: {len(choices)} children; showing only the first append-order branch.")
        eid = choices[0]
        after.extend(by_id[eid])
    return {"records": [*(before[-context:] if context else []), matched, *after[:context]],
            "window_bounded": True, "warnings": warnings}
