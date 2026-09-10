"""OmniVoice text-to-speech."""

from __future__ import annotations

import gc
import inspect
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch

from constants import (
    OMNIVOICE_CHECKPOINT,
    OMNIVOICE_LOAD_ASR_DEFAULT,
    OMNIVOICE_REVISION,
)

_log = logging.getLogger(__name__)

try:
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

    OMNIVOICE_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:  # keep the cause; a broken install is not "not installed"
    OmniVoice = None
    OmniVoiceGenerationConfig = None
    OMNIVOICE_IMPORT_ERROR = exc

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


def _revision_kwargs() -> dict:
    """`revision=` for OmniVoice, when it can actually be honoured.

    OmniVoice ships its own `from_pretrained`, and the checkpoint can be
    overridden to a local directory via `OMNIVOICE_MODEL`. Pin where that is
    meaningful; never turn a pin attempt into a TypeError at load time.
    """
    if not OMNIVOICE_REVISION or os.path.isdir(OMNIVOICE_CHECKPOINT):
        return {}
    try:
        params = inspect.signature(OmniVoice.from_pretrained).parameters.values()
    except (TypeError, ValueError):
        return {}
    accepted = any(
        p.name == "revision" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params
    )
    if not accepted:
        _log.warning(
            "OmniVoice.from_pretrained takes no `revision`; loading %s unpinned.",
            OMNIVOICE_CHECKPOINT,
        )
        return {}
    return {"revision": OMNIVOICE_REVISION}


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
        detail = ""
        if OMNIVOICE_IMPORT_ERROR is not None:
            detail = (
                f" Import failed with "
                f"{type(OMNIVOICE_IMPORT_ERROR).__name__}: {OMNIVOICE_IMPORT_ERROR}"
            )
        return None, (
            "OmniVoice is not importable. Re-run Install "
            f"(includes `uv pip install omnivoice --no-deps`).{detail}"
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
            **_revision_kwargs(),
        )
        ov_sampling_rate = getattr(ov_model, "sampling_rate", 24000)
        ov_device = target_device
        return ov_model, f"OmniVoice loaded ({target_device}, load_asr={load_asr})."
    except Exception as exc:
        unload_omnivoice_model()
        return None, f"Error loading OmniVoice: {exc}"


def _to_pcm16(audio) -> Optional[np.ndarray]:
    """Convert an OmniVoice `generate()` result into a mono int16 waveform.

    `OmniVoice.generate()` returns `list[np.ndarray]`, each a one-dimensional
    float waveform, but torch tensors and extra leading axes are tolerated.
    """
    raw = audio
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        raw = raw[0]
    if hasattr(raw, "detach"):  # torch tensor
        raw = raw.detach().to("cpu", torch.float32).numpy()
    # Not in place: `asarray`/`reshape` hand back a view of the model's own
    # output buffer when it is already contiguous float32, and clipping through
    # that view would quietly edit OmniVoice's result.
    waveform = np.clip(np.asarray(raw, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (waveform * 32767.0).astype(np.int16)


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
        waveform = _to_pcm16(audio)
        if waveform is None or waveform.size == 0:
            return None, "TTS produced no audio."
        return (ov_sampling_rate, waveform), "TTS done."
    except Exception as exc:
        return None, f"TTS error: {type(exc).__name__}: {exc}"
