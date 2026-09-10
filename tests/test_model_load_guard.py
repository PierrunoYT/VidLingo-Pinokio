"""A translation model that failed to load must stop the run.

Callers used to decide whether loading worked by looking for "Error" or
"Authentication" in the status message, so any failure phrased differently —
an unknown model size, for one — read as success and the next stage ran with
no model resident.
"""

from __future__ import annotations

import pytest

from tests.test_pipeline_guards import TTS_ARGS

UNRECOGNISED_FAILURE = "Unknown TranslateGemma size '9B'. Pin a revision in constants.py first."


@pytest.fixture
def failed_load(monkeypatch, spy_tts):
    """Loading reports a failure the old string check would have missed."""
    import pipeline
    import translate

    monkeypatch.setattr(translate, "pipe", None)
    monkeypatch.setattr(translate, "model", None)
    monkeypatch.setattr(
        pipeline, "load_translate_model", lambda *a, **k: UNRECOGNISED_FAILURE
    )
    return spy_tts


def test_unknown_size_is_reported_as_an_error():
    import translate

    msg = translate.load_translate_model("9B")
    assert msg.startswith("Error"), msg


def test_translate_only_stops_when_nothing_loaded(failed_load):
    import pipeline

    out, status = pipeline.translate_only("Hello.", "", "English", "French", "9B", 512)
    assert out == ""
    assert status == UNRECOGNISED_FAILURE


def test_translate_and_synthesize_stops_before_tts(failed_load):
    import pipeline

    result = pipeline.translate_and_synthesize(
        "Hello.", "", "English", "French", "9B", 512, *TTS_ARGS
    )
    assert failed_load == [], "TTS ran without a translation"
    assert result[0] == ""
    assert result[1] == UNRECOGNISED_FAILURE
    assert result[2] is None


def test_full_pipeline_stops_when_nothing_loaded(failed_load, monkeypatch, tmp_path):
    import pipeline

    mp3 = tmp_path / "audio.mp3"
    mp3.write_bytes(b"")
    monkeypatch.setattr(
        pipeline, "download_youtube_mp3", lambda *a, **k: (str(mp3), "Downloaded")
    )
    monkeypatch.setattr(pipeline, "transcribe_long", lambda *a, **k: ("hello", "stats"))

    result = pipeline.run_full_pipeline_tts(
        "https://www.youtube.com/watch?v=x",
        "",
        "English",
        True,
        True,
        256,
        "English",
        "French",
        "9B",
        512,
        *TTS_ARGS,
    )
    assert failed_load == [], "TTS ran without a translation"
    assert len(result) == 8
    assert result[5] is None
    assert UNRECOGNISED_FAILURE in result[7]
