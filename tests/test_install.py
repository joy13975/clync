"""Install-surface behaviour: a Claude-Code-only install (no claude.ai Chrome
profile), the bootstrap-skill supersede, and PG17 bin discovery. These are the
parts a NEW user hits first, and each one used to be hardcoded or hard-required."""
from __future__ import annotations

import types
from pathlib import Path

import pytest

import clync


# ---------------------------------------------------------------- bootstrap skill
def _skill_dir(root: Path, name: str, extra: dict[str, str] | None = None) -> Path:
    d = root
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: x\n---\n\nbody\n")
    for fn, body in (extra or {}).items():
        (d / fn).write_text(body)
    return d


def test_bootstrap_copy_recognised_only_when_it_really_is_one(tmp_path):
    target = _skill_dir(tmp_path / "repo" / "skill", "clync")
    assert clync._is_bootstrap_copy(_skill_dir(tmp_path / "same", "clync"), target)
    # a different skill's dir is somebody else's file, not our bootstrap
    assert not clync._is_bootstrap_copy(_skill_dir(tmp_path / "other", "not-clync"), target)
    # extra files mean it is not a bare bootstrap drop
    assert not clync._is_bootstrap_copy(
        _skill_dir(tmp_path / "rich", "clync", {"notes.md": "mine"}), target)
    # no frontmatter name at all
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "SKILL.md").write_text("no frontmatter here\n")
    assert not clync._is_bootstrap_copy(plain, target)


def test_relink_supersedes_a_bootstrap_copy_but_refuses_real_files(tmp_path):
    target = _skill_dir(tmp_path / "repo" / "skill", "clync")

    link = _skill_dir(tmp_path / "skills" / "clync", "clync")
    clync._relink(link, target)
    assert link.is_symlink() and link.resolve() == target.resolve()

    # a real directory that is NOT a bootstrap copy must never be clobbered
    real = _skill_dir(tmp_path / "skills" / "mine", "clync", {"keep.md": "precious"})
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        clync._relink(real, target)
    assert (real / "keep.md").read_text() == "precious"


# ------------------------------------------------------------- PG17 bin discovery
def test_default_pg_bin_picks_an_existing_prefix(tmp_path, monkeypatch):
    missing, present = tmp_path / "a" / "bin", tmp_path / "b" / "bin"
    present.mkdir(parents=True)
    (present / "initdb").write_text("")
    monkeypatch.setattr(clync, "PG17_BIN_CANDIDATES", (str(missing), str(present)))
    assert clync._default_pg_bin() == str(present)


def test_default_pg_bin_returns_first_candidate_so_pg_fails_loud(tmp_path, monkeypatch):
    """On a total miss it must NOT silently pick something wrong — it returns the
    first candidate precisely so `_pg()` raises its own actionable error."""
    a, b = tmp_path / "a" / "bin", tmp_path / "b" / "bin"
    monkeypatch.setattr(clync, "PG17_BIN_CANDIDATES", (str(a), str(b)))
    assert clync._default_pg_bin() == str(a)


# --------------------------------------------------------- local-only scheduling
def _stub_launchctl(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(clync.subprocess, "run", fake_run)
    monkeypatch.setattr(clync, "PLIST_PATH", tmp_path / "agent.plist")
    monkeypatch.setattr(clync, "SCHED_LOG", tmp_path / "log" / "sched.log")
    return calls


def test_install_without_profile_marks_the_job_local_only(tmp_path, monkeypatch):
    _stub_launchctl(monkeypatch, tmp_path)
    clync.cmd_install(types.SimpleNamespace(profile=None, at="09:00", org=None))
    plist = (tmp_path / "agent.plist").read_text()
    assert clync.LOCAL_ONLY_ENV in plist          # deliberate local-only install
    assert clync.PROFILE_ENV not in plist         # and no phantom profile


def test_install_with_profile_carries_the_profile(tmp_path, monkeypatch):
    _stub_launchctl(monkeypatch, tmp_path)
    clync.cmd_install(types.SimpleNamespace(profile="Person 1", at="09:00", org=None))
    plist = (tmp_path / "agent.plist").read_text()
    assert f"<key>{clync.PROFILE_ENV}</key><string>Person 1</string>" in plist
    assert clync.LOCAL_ONLY_ENV not in plist


def test_scheduled_without_profile_or_marker_still_fails_loud(monkeypatch):
    """A LOST profile env var is a broken install, not a local-only config: the
    scheduled job must notify and raise, exactly as before this mode existed."""
    monkeypatch.delenv(clync.PROFILE_ENV, raising=False)
    monkeypatch.delenv(clync.LOCAL_ONLY_ENV, raising=False)
    notes = []
    monkeypatch.setattr(clync, "notify", lambda *a: notes.append(a))
    with pytest.raises(RuntimeError, match=clync.LOCAL_ONLY_ENV):
        clync.cmd_scheduled(types.SimpleNamespace(profile=None, org=None))
    assert notes and notes[0][0] == "fail"       # loud, not silent


# ------------------------------------------- a local source that is not installed
def test_missing_local_root_is_skipped_when_nothing_is_stored(pg_test_db, monkeypatch):
    """First run on a machine with no Codex CLI: `~/.codex` does not exist, nothing
    is stored, so there is nothing deletion reconciliation could purge — skip the
    leg instead of failing the whole sync (which is what new users used to hit)."""
    import codex
    monkeypatch.setattr(codex, "CODEX_ROOT", Path("/nonexistent/codex/root"))
    stats = clync.ingest_codex(full=True)
    assert stats["sessions_ingested"] == 0 and stats["sessions_in_db"] == 0


def test_missing_local_root_still_raises_once_a_corpus_is_stored(pg_test_db, tmp_path,
                                                                monkeypatch):
    """The guard's real case: a root that vanishes AFTER sessions were stored would
    purge them through deletion reconciliation. That must stay loud."""
    import cc
    from tests.test_cc import _cli_events, _write_session
    _write_session(tmp_path / "proj" / "sess-1.jsonl", _cli_events())
    monkeypatch.setattr(cc, "CC_ROOT", tmp_path)
    assert clync.ingest_cc(full=True)["sessions_ingested"] == 1   # corpus now stored

    monkeypatch.setattr(cc, "CC_ROOT", tmp_path / "gone")         # root disappears
    with pytest.raises(FileNotFoundError):
        clync.ingest_cc(full=True)

    con = clync.connect()                                          # and nothing purged
    try:
        assert con.execute("SELECT COUNT(*) FROM units WHERE source='claude_code'"
                           ).fetchone()["count"] == 1
    finally:
        con.close()
