"""Source records shared by the live finder and its native adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

TOOL_SENDER = "tool"
REASONING_KIND = "reasoning"


@dataclass
class Session:
    source: str
    session_id: str
    title: str
    cwd: str
    updated_at: str
    path: Path | None = None
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Record:
    anchor: str
    sender: str
    text: str
    kind: str = "text"
    created_at: str | None = None
