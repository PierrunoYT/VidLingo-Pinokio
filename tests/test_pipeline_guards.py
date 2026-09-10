"""A failed stage must stop the pipeline, not feed the next one.

Both TTS pipelines previously passed a translation error message straight into
OmniVoice, appended "Done." and returned success-shaped results.
"""

from __future__ import annotations

import pytest

TTS_ARGS = (
    "Auto",  # tts_language
    "design",  # tts_mode
    None,  # ref_audio
    "",  # ref_text
    "",  # tts_instruct
    32,  # tts_steps
    2.0,  # tts_guidance
    True,  # tts_denoise
    1.0,  # tts_speed
    0,  # tts_duration
    True,  # tts_preprocess
    True,  # tts_postprocess
    "auto",  # tts_device
)


@pytest.fixture
def failing_translation(loaded_translator, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(loaded_translator, "_translate_single_chunk", boom)


@pytest.fixture
def working_translation(loaded_translator, monkeypatch):
    monkeypatch.setattr(
        loaded_translator,
        "_translate_single_chunk",
        lambda text, src, tgt, mt: "Bonjour.",
    )


def _translate_and_synthesize():
    import pipeline

    return pipeline.translate_and_synthesize(
        "Hello.", "", "English", "French", "4B", 512, *TTS_ARGS
    )


def test_translation_failure_is_never_spoken(failing_translation, spy_tts):
    text, status, audio, tts_status, _, _, _ = _translate_and_synthesize()
    assert spy_tts == [], "a failure message reached OmniVoice"
    assert audio is None
    assert "decoder failed" in status
    assert "Skipped TTS" in tts_status
    assert "decoder failed" not in text


def test_successful_translation_is_spoken(working_translation, spy_tts):
    text, _, audio, tts_status, _, _, _ = _translate_and_synthesize()
    assert spy_tts == ["Bonjour."]
    assert audio is not None
    assert text == "Bonjour."
    assert tts_status == "TTS done."


@pytest.fixture
def stub_download(monkeypatch, tmp_path):
    """Full pipeline with download and transcription already succeeded."""
    import pipeline

    mp3 = tmp_path / "audio.mp3"
    mp3.write_bytes(b"")
    monkeypatch.setattr(
        pipeline, "download_youtube_mp3", lambda *a, **k: (str(mp3), "Downloaded")
    )
    monkeypatch.setattr(
        pipeline, "transcribe_long", lambda *a, **k: ("hello", "stats")
    )
    monkeypatch.setattr(
        pipeline, "transcribe_short", lambda *a, **k: ("hello", "stats")
    )


def _run_full_pipeline_tts():
    import pipeline

    return pipeline.run_full_pipeline_tts(
        "https://www.youtube.com/watch?v=x",
        "",
        "English",
        True,
        True,
        256,
        "English",
        "French",
        "4B",
        512,
        *TTS_ARGS,
    )


def test_full_pipeline_stops_before_tts_on_translation_failure(
    stub_download, failing_translation, spy_tts
):
    result = _run_full_pipeline_tts()
    assert spy_tts == []
    assert result[5] is None, "audio was returned for a failed run"
    assert "decoder failed" in result[6]
    assert "Stopped before TTS." in result[7]
    assert "Done." not in result[7], "a failed run reported completion"


def test_full_pipeline_failure_matches_success_arity(
    stub_download, working_translation, spy_tts, monkeypatch
):
    """Gradio binds a fixed number of outputs; both paths must return them all."""
    ok = _run_full_pipeline_tts()

    import translate

    monkeypatch.setattr(
        translate,
        "_translate_single_chunk",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("decoder failed")),
    )
    failed = _run_full_pipeline_tts()
    assert len(ok) == len(failed) == 8
