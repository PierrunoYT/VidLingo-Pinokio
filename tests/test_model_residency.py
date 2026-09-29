"""Only the model the next stage needs may stay resident.

Handlers released only the model their own pipeline had just used, so
OmniVoice stayed on the GPU through every later transcription and
translation, and TranslateGemma stayed loaded through the Transcribe tab.
"""

from __future__ import annotations

import pytest

import pipeline
from tests.test_pipeline_guards import TTS_ARGS


@pytest.fixture
def unloads(monkeypatch):
    calls: list[str] = []
    for name in ("unload_asr_model", "unload_translate_model", "unload_omnivoice_model"):
        monkeypatch.setattr(pipeline, name, lambda name=name: calls.append(name))
    return calls


def test_transcription_releases_translation_and_tts(unloads, monkeypatch):
    monkeypatch.setattr(pipeline, "transcribe_short", lambda *a, **k: ("hi", "ok"))
    pipeline.transcribe_upload("a.mp3", "English", True, False, "", 256)
    assert set(unloads) == {"unload_translate_model", "unload_omnivoice_model"}


def test_translation_releases_asr_and_tts(unloads, monkeypatch):
    monkeypatch.setattr(pipeline, "load_translate_model", lambda *a, **k: "loaded")
    monkeypatch.setattr(pipeline, "translate_model_is_loaded", lambda: False)
    pipeline.translate_only("Hello.", "", "English", "French", "4B", 512)
    assert set(unloads) == {"unload_asr_model", "unload_omnivoice_model"}


def test_synthesis_releases_asr_and_translation(unloads, monkeypatch):
    monkeypatch.setattr(
        pipeline, "generate_omnivoice_tts", lambda **k: (None, "TTS done.")
    )
    pipeline.omnivoice_synthesize_only("Bonjour.", *TTS_ARGS)
    assert set(unloads) == {"unload_asr_model", "unload_translate_model"}
