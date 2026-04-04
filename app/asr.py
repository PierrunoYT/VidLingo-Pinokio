"""Cohere ASR (transcription)."""

from __future__ import annotations

import gc
import logging
import time
from typing import Optional, Tuple

import gradio as gr
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio

from constants import MODEL_ID_ASR, SUPPORTED_LANGUAGES

_log = logging.getLogger(__name__)

_model_cache_asr: dict = {}


def unload_asr_model() -> None:
    global _model_cache_asr
    for k in list(_model_cache_asr.keys()):
        entry = _model_cache_asr.pop(k, None)
        if entry:
            m = entry.get("model")
            p = entry.get("processor")
            del m, p
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_asr_model(device: str = "auto", hf_token: Optional[str] = None):
    cache_key = hf_token or "no_token"
    if cache_key not in _model_cache_asr:
        auth_kwargs = {"token": hf_token} if hf_token else {}
        proc = AutoProcessor.from_pretrained(MODEL_ID_ASR, **auth_kwargs)
        mdl = CohereAsrForConditionalGeneration.from_pretrained(
            MODEL_ID_ASR,
            device_map=device,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            **auth_kwargs,
        )
        _model_cache_asr[cache_key] = {"processor": proc, "model": mdl}
    return _model_cache_asr[cache_key]["processor"], _model_cache_asr[cache_key]["model"]


def download_asr_model(hf_token: str, progress=gr.Progress()) -> str:
    progress(0, desc="Caching ASR model...")
    token = (hf_token or "").strip() or None
    try:
        get_asr_model(hf_token=token)
        progress(1.0, desc="Done")
        return "Cohere Transcribe model is ready."
    except Exception as e:
        return f"ASR model download failed: {e}"


def transcribe_short(
    audio_path: str,
    language: str,
    punctuation: bool,
    hf_token: Optional[str],
    asr_max_tokens: int = 256,
    progress=gr.Progress(),
) -> Tuple[str, str]:
    token = (hf_token or "").strip() or None
    mt = max(32, int(asr_max_tokens))
    progress(0, desc="Loading ASR...")
    processor, model_asr = get_asr_model(hf_token=token)
    progress(0.3, desc="Loading audio...")
    try:
        audio = load_audio(audio_path, sampling_rate=16000)
    except Exception as e:
        return f"Error loading audio: {e}", ""
    lang_code = SUPPORTED_LANGUAGES.get(language, "en")
    progress(0.5, desc="Transcribing...")
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        language=lang_code,
        punctuation=punctuation,
    )
    inputs.to(model_asr.device, dtype=model_asr.dtype)
    progress(0.7, desc="Generating...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model_asr.generate(**inputs, max_new_tokens=mt)
    elapsed = time.time() - start_time
    text = processor.decode(outputs, skip_special_tokens=True)
    return text, f"Transcribed in {elapsed:.2f}s"


def transcribe_long(
    audio_path: str,
    language: str,
    punctuation: bool,
    hf_token: Optional[str],
    asr_max_tokens: int = 256,
    progress=gr.Progress(),
) -> Tuple[str, str]:
    token = (hf_token or "").strip() or None
    mt = max(32, int(asr_max_tokens))
    progress(0, desc="Loading ASR...")
    processor, model_asr = get_asr_model(hf_token=token)
    progress(0.2, desc="Loading audio...")
    try:
        audio = load_audio(audio_path, sampling_rate=16000)
    except Exception as e:
        return f"Error loading audio: {e}", ""
    lang_code = SUPPORTED_LANGUAGES.get(language, "en")
    duration_s = len(audio) / 16000
    progress(0.4, desc="Processing...")
    inputs = processor(
        audio=audio,
        sampling_rate=16000,
        return_tensors="pt",
        language=lang_code,
        punctuation=punctuation,
    )
    audio_chunk_index = inputs.get("audio_chunk_index")
    inputs.to(model_asr.device, dtype=model_asr.dtype)
    progress(0.6, desc="Generating...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model_asr.generate(**inputs, max_new_tokens=mt)
    elapsed = time.time() - start_time
    text = processor.decode(
        outputs,
        skip_special_tokens=True,
        audio_chunk_index=audio_chunk_index,
        language=lang_code,
    )[0]
    rtfx = duration_s / elapsed if elapsed > 0 else 0
    stats = (
        f"Duration: {duration_s / 60:.1f} min | {elapsed:.1f}s | RTFx: {rtfx:.1f}x"
    )
    return text, stats
