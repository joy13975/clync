import os
import subprocess
import sys

import clync


def test_cli_reads_archive_window_and_rejects_missing_unit(pg_test_db):
    unit = "cli-archive-window"
    texts = ["excluded first turn", "retained decision", "assistant interpretation",
             "excluded last turn"]
    with clync.connect() as con:
        con.execute(
            "INSERT INTO units(unit_id,kind,source,msg_count,synced_at) VALUES(%s,%s,%s,%s,%s)",
            (unit, "cc_session", "claude_code", len(texts), clync._now()))
        for idx, text in enumerate(texts):
            con.execute(
                "INSERT INTO messages(unit_id,msg_id,idx,sender,text) VALUES(%s,%s,%s,%s,%s)",
                (unit, str(idx), idx, "assistant" if idx == 2 else "user", text))
    command = [sys.executable, str(clync.REPO_DIR / "clync.py"), "read"]
    env = {**os.environ, "CLYNC_PG_DB": clync.PG_DB}
    read = subprocess.run(
        [*command, unit, "--from-idx", "1", "--to-idx", "2", "--sender", "user"],
        env=env, capture_output=True, text=True, check=False)
    assert read.returncode == 0, read.stderr
    assert texts[1] in read.stdout
    assert all(text not in read.stdout for text in (texts[0], texts[2], texts[3]))
    missing = subprocess.run([*command, unit + "-missing"], env=env,
                             capture_output=True, text=True, check=False)
    assert missing.returncode != 0
    assert unit + "-missing" in missing.stderr
