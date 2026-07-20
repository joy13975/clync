"""Pure-function units — no Postgres, no model, no network."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

import clync

search = pytest.importorskip("search")


def test_message_text_content_blocks():
    msg = {"content": [{"type": "text", "text": "hello"},
                       {"type": "tool_use"},                     # non-text ignored
                       {"type": "text", "text": "world"}]}
    assert clync._message_text(msg) == "hello\nworld"


def test_message_text_fallback_and_attachments():
    # no content blocks -> top-level text; attachment content appended
    msg = {"text": "top", "attachments": [{"file_name": "a.txt",
                                            "extracted_content": "doc body"}]}
    out = clync._message_text(msg)
    assert "top" in out and "[attachment: a.txt]" in out and "doc body" in out


def test_message_text_hollow_is_empty():
    # the "hollow conversation" class: no usable text -> "" (not a crash, not junk)
    assert clync._message_text({"content": [{"type": "tool_use"}]}) == ""
    assert clync._message_text({}) == ""


def test_sparse_literal_is_1_based():
    # BGE token ids are 0-based; pgvector sparsevec is 1-based -> +1, sorted
    lit = search._sparse_literal({5: 0.2, 0: 0.5})
    assert lit == f"{{1:0.5,6:0.2}}/{search.SPARSE_DIM}"


def test_sparse_literal_empty_raises():
    with pytest.raises(ValueError):
        search._sparse_literal({})              # no positive weights
    with pytest.raises(ValueError):
        search._sparse_literal({3: 0.0})        # non-positive dropped -> empty


def test_lang_detection():
    assert search._lang("hello world") == "en"
    assert search._lang("こんにちは") == "ja"
    assert search._lang("你好世界") == "zh"


def test_empty_query_returns_empty_without_touching_cluster():
    # guard is the first line, before ensure_cluster/_embed -> safe with no cluster
    assert search.hybrid_search("") == []
    assert search.hybrid_search("   \n\t ") == []


def test_pg_db_name_rejects_injection_at_import():
    # $CLYNC_PG_DB is interpolated into SQL identifiers -> must be validated at load
    env = {**os.environ, "CLYNC_PG_DB": "clync; DROP DATABASE postgres"}
    r = subprocess.run([sys.executable, "-c", "import search"],
                       cwd=os.path.dirname(os.path.dirname(__file__)),
                       env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert "invalid $CLYNC_PG_DB" in (r.stderr + r.stdout)
