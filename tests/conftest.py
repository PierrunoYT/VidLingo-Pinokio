"""Test harness for the VidLingo app modules.

The app modules import Gradio, yt-dlp, Transformers and huggingface_hub at
module scope, and the real packages are neither needed nor wanted here: these
tests cover our own control flow — what happens when a stage fails — so every
external dependency is replaced with a stub before the modules are imported.
Only NumPy and Torch are real, because the waveform and dtype logic under test
is written against them.

Consequence: the suite needs no models, no network and no GPU.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"


def _stub_gradio() -> types.ModuleType:
    gradio = types.ModuleType("gradio")

    class Progress:
        """gr.Progress() is called for its side effect only."""

        def __call__(self, *args, **kwargs) -> None:
            return None

    gradio.Progress = Progress
    gradio.update = lambda **kwargs: {"__update__": kwargs}
    return gradio


def _stub_transformers() -> list[tuple[str, types.ModuleType]]:
    transformers = types.ModuleType("transformers")

    class _Loadable:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("tests must not load real models")

    transformers.AutoProcessor = _Loadable
    transformers.CohereAsrForConditionalGeneration = _Loadable
    transformers.AutoModelForImageTextToText = _Loadable
    transformers.pipeline = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("tests must not build real pipelines")
    )

    audio_utils = types.ModuleType("transformers.audio_utils")
    audio_utils.load_audio = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("tests must not load real audio")
    )
    return [("transformers", transformers), ("transformers.audio_utils", audio_utils)]


def _install_stubs() -> None:
    stubs: list[tuple[str, types.ModuleType]] = [
        ("imageio_ffmpeg", types.SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg")),
        ("gradio", _stub_gradio()),
        ("yt_dlp", types.ModuleType("yt_dlp")),
        ("huggingface_hub", types.SimpleNamespace(login=lambda **kwargs: None)),
    ]
    stubs += _stub_transformers()
    for name, module in stubs:
        sys.modules[name] = module


if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
_install_stubs()


@pytest.fixture
def downloads_dir(tmp_path, monkeypatch):
    """Point the download module at a scratch OUTPUT_DIR."""
    import youtube

    target = tmp_path / "downloads"
    target.mkdir()
    monkeypatch.setattr(youtube, "OUTPUT_DIR", str(target))
    return target


@pytest.fixture
def loaded_translator(monkeypatch):
    """Pretend TranslateGemma is loaded, with a stubbed per-chunk translator."""
    import translate

    monkeypatch.setattr(translate, "pipe", object())
    monkeypatch.setattr(translate, "model", None)
    monkeypatch.setattr(translate, "processor", None)
    return translate


@pytest.fixture
def spy_tts(monkeypatch):
    """Record every text handed to OmniVoice; return canned success."""
    import pipeline

    spoken: list[str] = []

    def _fake_tts(**kwargs):
        spoken.append(kwargs["text"])
        return (24000, "audio"), "TTS done."

    monkeypatch.setattr(pipeline, "generate_omnivoice_tts", _fake_tts)
    monkeypatch.setattr(pipeline, "load_translate_model", lambda *a, **k: "loaded")
    monkeypatch.setattr(pipeline, "unload_translate_model", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "unload_asr_model", lambda *a, **k: None)
    return spoken
