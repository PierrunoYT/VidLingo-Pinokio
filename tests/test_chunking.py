"""Every chunk handed to TranslateGemma must respect the size cap.

Chunks were split only at `.`/`!`/`?` followed by a space, so unpunctuated
transcripts and Chinese/Japanese text were one chunk of any length, which
`max_new_tokens` then cut off part-way without any error.
"""

from __future__ import annotations

from translate import _WORD_RE, _split_into_chunks


def _size(chunk: str) -> int:
    return len(_WORD_RE.findall(chunk))


def test_unpunctuated_text_is_split():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = _split_into_chunks(text, max_words=300)
    assert len(chunks) == 4
    assert all(_size(c) <= 300 for c in chunks)
    assert " ".join(chunks) == text, "splitting lost or reordered words"


def test_cjk_sentences_are_split_at_full_width_punctuation():
    text = "今日は晴れです。明日は雨です！本当ですか？"
    assert _split_into_chunks(text, max_words=8) == [
        "今日は晴れです。",
        "明日は雨です！",
        "本当ですか？",
    ]


def test_long_unpunctuated_cjk_is_split():
    text = "字" * 1000
    chunks = _split_into_chunks(text, max_words=300)
    assert all(_size(c) <= 300 for c in chunks)
    assert "".join(chunks) == text


def test_ordinary_sentences_are_still_grouped():
    text = "One two. Three four. Five six."
    assert _split_into_chunks(text, max_words=4) == ["One two. Three four.", "Five six."]
