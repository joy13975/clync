"""Read-only lexical projection of OpenCode's native message store."""
from __future__ import annotations

import json
import re
import socket
import sqlite3
import subprocess
from pathlib import Path

from live_external import _string, opencode_message_records
from live_types import Session
from opencode_source import Client, _MACHINE_TYPES, _iso, _validate_info

# Projection contract: anomalyco/opencode@dc2ffcf8, session/history.ts and session/blob.ts.
_VERSION = "2.2.6"
_COLUMNS = {
    "session_v2": {"id": "TEXT", "project_id": "TEXT", "parent_id": "TEXT",
                   "directory": "TEXT", "title": "TEXT", "time_created": "INTEGER",
                   "time_updated": "INTEGER"},
    "project": {"id": "TEXT", "worktree": "TEXT"},
    "session_message": {"id": "TEXT", "session_id": "TEXT", "type": "TEXT",
                        "seq": "INTEGER", "data": "TEXT"},
    "session_part": {"message_id": "TEXT", "session_id": "TEXT", "ordinal": "INTEGER",
                     "type": "TEXT", "data": "TEXT"},
    "blob": {"hash": "TEXT"},
}


def _native_path(client, info):
    if info["version"] != _VERSION:
        raise ValueError(f"OpenCode native search supports {_VERSION}, got {info['version']}")
    if info["hostname"] != socket.gethostname() or type(info["pid"]) is not int or info["pid"] <= 0:
        raise ValueError("OpenCode native search requires a verified local service hostname and PID")
    result = subprocess.run([client.command, "debug", "paths", "db"], capture_output=True,
                            text=True, timeout=10, check=False)
    if result.returncode or len(result.stdout.splitlines()) != 1:
        raise ValueError(f"OpenCode native database path unavailable: {result.stderr.strip()}")
    path = Path(result.stdout.strip())
    if not path.is_absolute():
        raise ValueError("OpenCode native database path must be absolute")
    path = path.resolve(strict=True)
    opened = subprocess.run(["lsof", "-a", "-p", str(info["pid"]), "-Fn"],
                            capture_output=True, text=True, timeout=10, check=False)
    lines = opened.stdout.splitlines()
    if (opened.returncode or f"p{info['pid']}" not in lines or
            not any(line.startswith("n/") and Path(line[1:]).resolve() == path for line in lines)):
        raise ValueError("OpenCode service PID does not hold the configured native database open")
    return path


class Snapshot:
    def __enter__(self):
        self.client = Client()
        self.info = self.client.get("/api/info")
        self.path = _native_path(self.client, self.info)
        self.con = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.con.row_factory = sqlite3.Row
        try:
            self.con.execute("PRAGMA query_only=ON")
            self.version = self.con.execute("PRAGMA data_version").fetchone()[0]
            self.con.execute("BEGIN")
            for table, required in _COLUMNS.items():
                columns = {row["name"]: row["type"] for row in self.con.execute(f'PRAGMA table_info("{table}")')}
                if any(columns.get(key) != value for key, value in required.items()):
                    raise ValueError(f"Unsupported OpenCode native schema: {table}")
            return self
        except (sqlite3.Error, ValueError):
            self.con.close()
            raise

    def __exit__(self, *error):
        try:
            self.con.rollback()
            self.changed = self.version != self.con.execute("PRAGMA data_version").fetchone()[0]
        finally:
            self.con.close()
        after = self.client.get("/api/info")
        self.changed |= any(after[key] != self.info[key] for key in ("pid", "hostname", "version"))

    def sessions(self, include_subagents=False):
        projects = {row["id"]: Path(_string(row["worktree"], "OpenCode project root")).name
                    for row in self.con.execute("SELECT id, worktree FROM project")}
        sessions = []
        for row in self.con.execute("SELECT id, project_id, parent_id, directory, title, "
                                    "time_created, time_updated FROM session_v2 ORDER BY id"):
            info = {"id": row["id"], "projectID": row["project_id"], "parentID": row["parent_id"],
                    "location": {"directory": row["directory"]},
                    "time": {"created": row["time_created"], "updated": row["time_updated"]}}
            _validate_info(info)
            if include_subagents or not row["parent_id"]:
                sessions.append(Session("opencode", row["id"], _string(row["title"] or "", "OpenCode title"),
                                        row["directory"], _iso(row["time_updated"]),
                                        parent_id=row["parent_id"],
                                        metadata={"repo": projects.get(row["project_id"]),
                                                  "created_at": _iso(row["time_created"])}))
        return sessions

    def candidates(self, sessions, needles, include_tools=False):
        by_id = {session.session_id: session for session in sessions}
        if not by_id:
            return
        # SQLite's case fold is ASCII-only; decode Unicode/escaped rows rather than risk omissions.
        literals = [n for n in needles if n.isascii() and not n.startswith(r"\u")]
        predicates = []
        for column in ("m.data", "p.data"):
            predicates.append(" OR ".join([f"{column} LIKE ?"] * len(literals) + [
                f"{column} GLOB '*[^ -~]*'", f"instr({column}, '\\u') > 0", f"NOT json_valid({column})"] +
                ([f"instr({column}, 'media:') > 0"] if include_tools else [])))
        params = ["%" + needle + "%" for needle in literals]
        batch_size = self.con.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER) - 2 * len(params)
        if batch_size < 1:
            raise ValueError("SQLite parameter limit is too small for OpenCode candidate search")
        ids = list(by_id)
        for offset in range(0, len(ids), batch_size):
            batch = ids[offset:offset + batch_size]
            placeholders = ",".join("?" for _ in batch)
            for table, known, query in (
                ("session_message", {"user", "assistant", *_MACHINE_TYPES},
                 f"SELECT DISTINCT type FROM session_message WHERE session_id IN ({placeholders})"),
                ("session_part", {"text", "reasoning", "tool"},
                 "SELECT DISTINCT p.type FROM session_message m JOIN session_part p ON p.message_id=m.id "
                 f"WHERE m.session_id IN ({placeholders}) AND m.type='assistant'"),
            ):
                unknown = {row[0] for row in self.con.execute(query, batch)} - known
                if unknown:
                    raise ValueError(f"Unknown OpenCode {table} types: {sorted(unknown)}")
            selected = ",".join("(?)" for _ in batch)
            query = f"""
                WITH selected(id) AS (VALUES {selected}), candidates(id) AS (
                    SELECT m.id FROM selected s JOIN session_message m ON m.session_id=s.id
                    WHERE m.type IN ('user', 'assistant') AND ({predicates[0]})
                    UNION
                    SELECT p.message_id FROM selected s JOIN session_message m ON m.session_id=s.id
                    JOIN session_part p ON p.message_id=m.id
                    WHERE m.type='assistant' AND ({predicates[1]})
                )
                SELECT m.* FROM session_message m JOIN candidates c ON m.id=c.id
                ORDER BY m.session_id, m.seq
            """
            for row in self.con.execute(query, [*batch, *params, *params]):
                yield by_id[row["session_id"]], row

    def records(self, row, include_tools=False):
        data = json.loads(row["data"])
        if row["type"] == "assistant":
            parts = [json.loads(part[0]) for part in self.con.execute(
                "SELECT data FROM session_part WHERE message_id=? ORDER BY ordinal", (row["id"],))]
            if parts:
                inline = data.get("content")
                data["content"] = parts + (inline if isinstance(inline, list) else [])
            if include_tools:
                self._hydrate_text(data)
        return opencode_message_records({**data, "id": row["id"], "type": row["type"]},
                                        include_tools=include_tools)

    def _hydrate_text(self, data):
        for part in data["content"]:
            if part["type"] != "tool" or part["state"]["status"] not in ("completed", "error"):
                continue
            content = part["state"].get("content", [])
            for index, item in enumerate(content):
                if item["type"] != "file" or not re.fullmatch(r"media:[0-9a-f]{64}", item["uri"]):
                    continue
                if self.con.execute("SELECT 1 FROM blob WHERE hash=?", (item["uri"][6:],)).fetchone():
                    continue
                mime = _string(item["mime"], "OpenCode media MIME type")
                name = _string(item["name"], "OpenCode media name") if "name" in item else None
                label = mime if name is None else f"{name} ({mime})"
                content[index] = {"type": "text", "text": f"[media unavailable: {label}]"}
