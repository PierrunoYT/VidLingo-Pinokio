"""Translation failures must not travel through the content channel.

`translate_text_block()` used to return strings like "Translation error: ..."
as ordinary output, which the TTS pipelines then synthesized as speech and
logged as success.
"""

from __future__ import annotations

import pytest

from translate import TranslationError, translate_text_block


def test_empty_input_raises():
    with pytest.raises(TranslationError, match="No text to translate"):
        translate_text_block("   ", "English", "French", 512)


def test_unloaded_model_raises(monkeypatch):
    import translate

    monkeypatch.setattr(translate, "pipe", None)
    monkeypatch.setattr(translate, "model", None)
    with pytest.raises(TranslationError, match="Load TranslateGemma first"):
        translate_text_block("Hello.", "English", "French", 512)


def test_chunk_failure_raises_with_type_and_message(loaded_translator, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(loaded_translator, "_translate_single_chunk", boom)
    with pytest.raises(TranslationError) as excinfo:
        translate_text_block("Hello.", "English", "French", 512)
    assert "RuntimeError: decoder failed" in excinfo.value.message


def test_partial_output_is_preserved_but_kept_out_of_the_result(
    loaded_translator, monkeypatch
):
    """Chunks that succeeded are carried on the exception, for display only."""
    calls: list[str] = []

    def flaky(text, *args, **kwargs):
        calls.append(text)
        if len(calls) == 1:
            return "Bonjour."
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(loaded_translator, "_translate_single_chunk", flaky)
    # Two sentences over the chunk limit would be one chunk; force two chunks.
    monkeypatch.setattr(
        loaded_translator, "_split_into_chunks", lambda text, max_words=300: ["a", "b"]
    )
    with pytest.raises(TranslationError) as excinfo:
        translate_text_block("a b", "English", "French", 512)
    assert excinfo.value.partial == "Bonjour."


def test_success_returns_only_the_translation(loaded_translator, monkeypatch):
    monkeypatch.setattr(
        loaded_translator,
        "_translate_single_chunk",
        lambda text, src, tgt, mt: "Bonjour le monde.",
    )
    assert translate_text_block("Hello world.", "English", "French", 512) == (
        "Bonjour le monde."
    )
