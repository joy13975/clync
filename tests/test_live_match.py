from types import SimpleNamespace

import pytest

import live_search


def test_disjoint_passages_do_not_tokenize_or_compare_every_word(monkeypatch):
    def unnecessary_work(*args):
        pytest.fail("An unrelated passage needs neither tokenization nor edit comparisons")

    monkeypatch.setattr(live_search, "_terms", unnecessary_work)
    monkeypatch.setattr(live_search, "_WORDS", SimpleNamespace(findall=unnecessary_work))
    monkeypatch.setattr(live_search, "_one_edit", unnecessary_work)
    assert live_search._match("unrelated lexical noise " * 1000, ["postgres"], True) is None


def test_impossible_word_lengths_and_endpoints_do_not_need_edit_comparisons(monkeypatch):
    def unnecessary_comparison(*args):
        pytest.fail("These words cannot match by one edit")

    monkeypatch.setattr(live_search, "_one_edit", unnecessary_comparison)
    assert live_search._match("po stg completelywrongword zzpoqqqq", ["postgres"], True) is None


@pytest.mark.parametrize("boundary", [2, 5])
@pytest.mark.parametrize("edit", ["substitute", "insert", "delete", "transpose"])
@pytest.mark.parametrize("context", ["start", "edge", "tail", "second_tail"])
def test_edit_gate_preserves_score_passage_and_offset_at_both_splits(boundary, edit, context):
    term = "postgres"
    value = {
        "substitute": term[:boundary] + "x" + term[boundary + 1:],
        "insert": term[:boundary] + "x" + term[boundary:],
        "delete": term[:boundary] + term[boundary + 1:],
        "transpose": term[:boundary - 1] + term[boundary] + term[boundary - 1] + term[boundary + 1:],
    }[edit]
    size = live_search.PASSAGE_CHARS
    prefix, offset = {
        "start": ("", 0),
        "edge": ("x" * (size - 1) + " ", size // 2),
        "tail": ("x" * (size // 2), size // 2),
        "second_tail": ("x" * size, size),
    }[context]
    text = prefix + value
    assert live_search._match(text, [term], True) == (0.7, text[offset:offset + size], offset)
    assert live_search._match(text, [term], False) is None


def test_final_partial_window_may_match_a_suffix_inside_a_larger_token():
    offset = live_search.PASSAGE_CHARS // 2
    assert live_search._match("x" * offset + "postgras", ["postgres"], True) == (0.7, "postgras", offset)


def test_normalized_tokenization_preserves_unicode_width_score_and_original_offset():
    size = live_search.PASSAGE_CHARS
    text = "İ" + "x" * (size - 2) + " ＭＡＲＩＧＯＬＤ ＰＯＳＴＧＲＡＳ ２０２７ 認証"
    offset = size // 2 - 1
    expected = ((3 + 0.7) / 4, text[offset:offset + size], offset)
    assert live_search._match(text, ["marigold", "postgres", "2027", "認証"], True) == expected
