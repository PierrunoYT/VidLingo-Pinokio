"""ASR must always return the advertised `(text, status)` pair.

Only audio loading used to be guarded, so an expected gated-model auth error
aborted the Gradio event instead of returning a result. The model cache was
also keyed on the raw Hugging Face token.
"""

from __future__ import annotations

import torch

import asr


def test_model_load_failure_returns_a_result(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("401 Client Error: Unauthorized for url: ...")

    monkeypatch.setattr(asr, "get_asr_model", boom)
    text, status = asr.transcribe_short("audio.mp3", "English", True, None)

    assert text.startswith("Error"), "pipeline detects failures by this prefix"
    assert status == ""
    assert "huggingface.co" in text, "auth failures should say how to fix them"


def test_generation_failure_returns_a_result(monkeypatch):
    class _Processor:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(asr, "get_asr_model", lambda **kw: (_Processor(), object()))
    monkeypatch.setattr(asr, "load_audio", lambda *a, **k: [0.0] * 16000)

    text, status = asr.transcribe_long("audio.mp3", "English", True, None)

    assert text.startswith("Error")
    assert status == ""
    assert "out of memory" in text.lower()


def test_audio_load_failure_still_returns_a_result(monkeypatch):
    monkeypatch.setattr(asr, "get_asr_model", lambda **kw: (object(), object()))

    def boom(*args, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(asr, "load_audio", boom)
    text, status = asr.transcribe_short("missing.mp3", "English", True, None)

    assert text.startswith("Error")
    assert status == ""


def test_cache_is_not_keyed_on_the_token(monkeypatch):
    """A token is a credential, not cache identity."""
    loaded: list[str] = []

    class _Loaded:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            loaded.append(kwargs.get("revision", ""))
            return object()

    monkeypatch.setattr(asr, "AutoProcessor", _Loaded)
    monkeypatch.setattr(asr, "CohereAsrForConditionalGeneration", _Loaded)
    monkeypatch.setattr(asr, "_model_cache_asr", {})

    asr.get_asr_model(hf_token="hf_first_token")
    calls_after_first = len(loaded)
    asr.get_asr_model(hf_token="hf_a_different_token")

    assert len(loaded) == calls_after_first, "changing the token reloaded the model"
    assert all(token not in str(asr._model_cache_asr.keys()) for token in ("hf_first_token",))


def test_cpu_only_machine_does_not_request_float16(monkeypatch):
    """`device="auto"` resolves to CPU without CUDA; float16 there is unsupported."""
    captured: dict = {}

    class _Proc:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return object()

    class _Model:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(asr, "AutoProcessor", _Proc)
    monkeypatch.setattr(asr, "CohereAsrForConditionalGeneration", _Model)
    monkeypatch.setattr(asr, "_model_cache_asr", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    asr.get_asr_model(device="auto")

    assert captured["torch_dtype"] is torch.float32
