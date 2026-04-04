"""OmniVoice text-to-speech."""

from __future__ import annotations

import gc
import os
from typing import Optional, Tuple

import numpy as np
import torch

from constants import OMNIVOICE_CHECKPOINT, OMNIVOICE_LOAD_ASR_DEFAULT

try:
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig
except Exception:
    OmniVoice = None
    OmniVoiceGenerationConfig = None

ov_model = None
ov_sampling_rate = 24000
ov_device = None


def _resolve_ov_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_ov_dtype(device: str):
    return torch.float16 if device == "cuda" else torch.float32


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def unload_omnivoice_model() -> None:
    global ov_model, ov_sampling_rate, ov_device
    if ov_model is not None:
        del ov_model
        ov_model = None
    ov_sampling_rate = 24000
    ov_device = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_omnivoice_model(device: str = "auto") -> Tuple[Optional[object], str]:
    global ov_model, ov_sampling_rate, ov_device
    if OmniVoice is None:
        return None, (
            "OmniVoice is not installed. Re-run Install (includes `uv pip install omnivoice --no-deps`)."
        )
    target_device = _resolve_ov_device(None if device == "auto" else device)
    if ov_model is not None and ov_device == target_device:
        return ov_model, "OmniVoice already loaded."

    unload_omnivoice_model()
    try:
        load_asr = _env_flag("OMNIVOICE_LOAD_ASR", OMNIVOICE_LOAD_ASR_DEFAULT)
        ov_model = OmniVoice.from_pretrained(
            OMNIVOICE_CHECKPOINT,
            device_map=target_device,
            dtype=_resolve_ov_dtype(target_device),
            load_asr=load_asr,
        )
        ov_sampling_rate = getattr(ov_model, "sampling_rate", 24000)
        ov_device = target_device
        return ov_model, f"OmniVoice loaded ({target_device}, load_asr={load_asr})."
    except Exception as exc:
        unload_omnivoice_model()
        return None, f"Error loading OmniVoice: {exc}"


def generate_omnivoice_tts(
    text: str,
    tts_language: str,
    tts_mode: str,
    ref_audio,
    ref_text: str,
    tts_instruct: str,
    num_step: int,
    guidance_scale: float,
    denoise: bool,
    speed: float,
    duration: float,
    preprocess_prompt: bool,
    postprocess_output: bool,
    tts_device: str = "auto",
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    if not text or not text.strip():
        return None, "No text to synthesize."

    model_ov, load_msg = get_omnivoice_model(device=tts_device)
    if model_ov is None:
        return None, load_msg

    if OmniVoiceGenerationConfig is None:
        return None, "OmniVoice generation config is unavailable."

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )
    lang = None if not tts_language or tts_language == "Auto" else tts_language
    kwargs = {
        "text": text.strip(),
        "language": lang,
        "generation_config": gen_config,
    }

    if speed is not None and float(speed) != 1.0:
        kwargs["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kwargs["duration"] = float(duration)
    if tts_mode == "clone":
        if not ref_audio:
            return None, "Clone mode needs a reference audio."
        kwargs["voice_clone_prompt"] = model_ov.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=(ref_text or "").strip() or None,
        )
    elif tts_mode == "design" and (tts_instruct or "").strip():
        kwargs["instruct"] = tts_instruct.strip()

    try:
        audio = model_ov.generate(**kwargs)
        tensor = audio[0].squeeze(0)
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu()
        waveform = (tensor.numpy() * 32767).astype(np.int16)
        return (ov_sampling_rate, waveform), "TTS done."
    except Exception as exc:
        return None, f"TTS error: {type(exc).__name__}: {exc}"
