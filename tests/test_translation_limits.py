"""Output that hits the token limit must fail, not pass as a full translation."""

from __future__ import annotations

import pytest


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class _Pipe:
    tokenizer = _Tokenizer()

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def __call__(self, **kwargs):
        return [{"generated_text": [{"content": self.reply}]}]


def test_truncated_chunk_raises_with_partial(loaded_translator, monkeypatch):
    # 64 tokens out with a 64-token limit: generation stopped on the limit.
    monkeypatch.setattr(loaded_translator, "pipe", _Pipe(" ".join(["w"] * 64)))
    with pytest.raises(loaded_translator.TranslationError) as info:
        loaded_translator.translate_text_block("Hello.", "English", "Spanish", 64)
    assert "token limit" in info.value.message


def test_short_chunk_passes(loaded_translator, monkeypatch):
    monkeypatch.setattr(loaded_translator, "pipe", _Pipe("Hola."))
    assert (
        loaded_translator.translate_text_block("Hello.", "English", "Spanish", 64)
        == "Hola."
    )


def test_chinese_and_japanese_parts_join_without_spaces(loaded_translator, monkeypatch):
    monkeypatch.setattr(loaded_translator, "_split_into_chunks", lambda t, max_words=300: ["a", "b"])
    replies = iter(["你好。", "再见。"])
    monkeypatch.setattr(
        loaded_translator, "_translate_single_chunk", lambda *a: next(replies)
    )
    out = loaded_translator.translate_text_block("x", "English", "Chinese (Simplified)", 64)
    assert out == "你好。再见。"


def test_full_width_sentences_rejoin_without_spaces(loaded_translator):
    assert loaded_translator._split_into_chunks("你好。再见。", max_words=300) == ["你好。再见。"]


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("Auto", None),
        ("Spanish (Mexico)", "es"),
        ("Chinese (Simplified)", "zh"),
        ("Arabic", "arb"),
        ("Hindi", "hi"),
    ],
)
def test_tts_language_maps_to_omnivoice_code(name, code):
    import tts

    assert tts.omnivoice_language(name) == code
