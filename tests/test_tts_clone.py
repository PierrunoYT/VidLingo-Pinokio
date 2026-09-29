"""A bad clone reference must be reported, not raised.

`create_voice_clone_prompt()` ran outside the error guard, so an unreadable
reference file aborted the Gradio event and every pipeline output was lost.
"""

from __future__ import annotations

import tts


class _Config:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Model:
    def create_voice_clone_prompt(self, **kwargs):
        raise RuntimeError("could not decode reference audio")

    def generate(self, **kwargs):
        raise AssertionError("generate must not run without a clone prompt")


def test_unreadable_reference_returns_a_status(monkeypatch):
    monkeypatch.setattr(tts, "get_omnivoice_model", lambda device="auto": (_Model(), "ok"))
    monkeypatch.setattr(tts, "OmniVoiceGenerationConfig", _Config)

    audio, status = tts.generate_omnivoice_tts(
        text="Bonjour.",
        tts_language="Auto",
        tts_mode="clone",
        ref_audio="broken.wav",
        ref_text="",
        tts_instruct="",
        num_step=32,
        guidance_scale=2.0,
        denoise=True,
        speed=1.0,
        duration=0,
        preprocess_prompt=True,
        postprocess_output=True,
    )

    assert audio is None
    assert "could not decode reference audio" in status
