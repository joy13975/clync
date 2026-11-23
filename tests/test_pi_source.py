from __future__ import annotations

import json
from pathlib import Path

import pytest

import pi_source


def header(sid="primary", version=3, **extra):
    return {"type": "session", "version": version, "id": sid,
            "cwd": "/code/actual-repo", "timestamp": "2026-01-01T00:00:00Z", **extra}


def message(eid, parent, role, text, **extra):
    content = text if role == "user" else [{"type": "text", "text": text}]
    return {"type": "message", "id": eid, "parentId": parent,
            "timestamp": "2026-01-01T00:00:01Z",
            "message": {"role": role, "content": content, **extra}}


def transcript(root, events, name="project/session.jsonl"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))
    return path


def test_archive_raw_tree_history_and_attribution(tmp_path):
    path = transcript(tmp_path, [header(), message("u", None, "user", "First request"),
        message("a", "u", "assistant", "First answer"),
        {"type": "compaction", "id": "c", "parentId": "a", "summary": "NOT original",
         "firstKeptEntryId": "c"},
        message("left", "c", "assistant", "Left branch"),
        message("right", "a", "assistant", "Right branch"),
        message("t", "right", "toolResult", "SECRET tool body"),
        {"type": "custom_message", "id": "custom", "parentId": "t", "content": "Injected"}])
    session = pi_source.parse_session(path)
    assert session.session_id == "pi:primary"
    assert session.repo == "actual-repo" and session.cwd == "/code/actual-repo"
    assert [(m.sender, m.text) for m in session.messages] == [
        ("pi_user", "First request"), ("assistant", "First answer"),
        ("assistant", "Left branch"), ("assistant", "Right branch")]
    left, right = session.messages[2:]
    assert "entry=left" in left.role_detail and "ancestor=a" in left.role_detail
    assert "entry=right" in right.role_detail and "ancestor=a" in right.role_detail
    assert all(m.embed_text != "" for m in session.messages)
    assert "append-order" in session.summary


@pytest.mark.parametrize("version", [1, 2, 3])
def test_versions_are_read_only_and_deterministic(tmp_path, version):
    entries = [message("u", None, "user", "Legacy prompt"),
               message("a", "u", "assistant", "Legacy answer")]
    if version == 1:
        for entry in entries:
            entry.pop("id")
            entry.pop("parentId")
    path = transcript(tmp_path, [header(version=version), *entries])
    before = path.read_bytes()
    first, second = pi_source.parse_session(path), pi_source.parse_session(path)
    assert [m.text for m in first.messages] == ["Legacy prompt", "Legacy answer"]
    assert [m.msg_id for m in first.messages] == [m.msg_id for m in second.messages]
    assert path.read_bytes() == before


def test_primary_fork_child_and_exact_suffix(tmp_path, monkeypatch):
    primary = transcript(tmp_path, [header(), message("u", None, "user", "Primary")])
    fork = transcript(tmp_path, [header("fork.with.dot", parentSession=str(primary)),
        message("u", None, "user", "Fork")], "flat.jsonl")
    child = transcript(tmp_path, [header("child"),
        {"type": "custom", "id": "meta", "parentId": None,
         "customType": "agentstrap.child", "data": {"parentSessionID": "primary"}},
        message("u", "meta", "user", "Machine task")], "agentstrap-children/x/child.jsonl")
    transcript(tmp_path, [header("backup")], "project/backup.jsonl.bak")
    monkeypatch.setattr(pi_source, "PI_ROOT", tmp_path)
    assert set(pi_source.iter_session_files()) == {primary, fork}
    assert pi_source.parse_session(child) is None


def test_duplicate_headers_fail_before_archive_writes(tmp_path):
    transcript(tmp_path, [header("same")], "a.jsonl")
    transcript(tmp_path, [header("same")], "project/export.jsonl")
    with pytest.raises(ValueError, match="[Dd]uplicate.*same"):
        pi_source.iter_session_files(tmp_path)


def test_invalid_and_pending_archive_never_silently_import(tmp_path):
    path = transcript(tmp_path, [header(), message("a", None, "assistant", "good")])
    with path.open("ab") as stream:
        stream.write(b'{"type":"message",')
    with pytest.raises(ValueError, match="[Ii]ncomplete|[Pp]ending"):
        pi_source.parse_session(path)
    path.write_text(json.dumps(header()) + '\n{"broken":\n')
    with pytest.raises(ValueError, match="[Mm]alformed"):
        pi_source.parse_session(path)


def test_missing_configured_root_and_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(pi_source, "PI_ROOT", tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="Pi.*root"):
        pi_source.iter_session_files()
    root = tmp_path / "root"
    root.mkdir()
    outside = transcript(tmp_path, [header()], "outside.jsonl")
    (root / "escape.jsonl").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        pi_source.iter_session_files(root)


def test_root_environment_contract(tmp_path):
    import os
    import subprocess
    import sys
    code = "import pi_source; print(pi_source.PI_ROOT)"
    env = {k: v for k, v in os.environ.items() if k not in ("CLYNC_PI_ROOT", "PI_CODING_AGENT_DIR")}
    env["PI_CODING_AGENT_DIR"] = str(tmp_path / "agent")
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                            capture_output=True, check=True)
    assert Path(result.stdout.strip()) == tmp_path / "agent/sessions"
    env["CLYNC_PI_ROOT"] = str(tmp_path / "flat")
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                            capture_output=True, check=True)
    assert Path(result.stdout.strip()) == tmp_path / "flat"
