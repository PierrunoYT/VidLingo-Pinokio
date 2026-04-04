"""End-to-end pipelines and tab helpers."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np

from asr import (
    transcribe_long,
    transcribe_short,
    unload_asr_model,
)
from constants import COHERE_TO_TRANSLATE_SOURCE
from translate import (
    load_translate_model,
    set_hf_token,
    translate_text_block,
    unload_translate_model,
)
from tts import generate_omnivoice_tts
from youtube import download_youtube_mp3


def _audio_preview_path(path: Optional[str]) -> Optional[str]:
    if path and path.lower().endswith(".mp3") and os.path.isfile(path):
        return path
    return None


def _resolve_downloaded_mp3(
    only_audio: object, only_file: object
) -> Tuple[Optional[str], str]:
    if only_audio is not None:
        if isinstance(only_audio, str) and os.path.isfile(only_audio):
            if only_audio.lower().endswith(".mp3"):
                return only_audio, ""
        if isinstance(only_audio, dict) and only_audio.get("path"):
            p = only_audio["path"]
            if isinstance(p, str) and os.path.isfile(p) and p.lower().endswith(".mp3"):
                return p, ""
    if only_file is not None:
        p = only_file
        if not isinstance(p, str):
            p = getattr(only_file, "name", None) or str(only_file)
        if isinstance(p, str) and os.path.isfile(p) and p.lower().endswith(".mp3"):
            return p, ""
    return (
        None,
        "No MP3 path found. Download a video that yields a single MP3 (ZIP bundles are not supported here).",
    )


def send_mp3_to_transcribe(
    only_audio: object, only_file: object
) -> Tuple[object, object, str]:
    path, err = _resolve_downloaded_mp3(only_audio, only_file)
    if not path:
        return gr.update(), gr.update(), err
    upd = gr.update(value=path)
    return upd, upd, f"Loaded into Transcribe: {os.path.basename(path)}"


def send_transcription_to_translate(out_short: str, out_long: str) -> Tuple[object, str]:
    long_t = (out_long or "").strip()
    short_t = (out_short or "").strip()
    text = long_t or short_t
    if not text:
        return gr.update(), "No transcription yet — run Transcribe on the Short-form or Long-form tab first."
    return gr.update(value=text), f"Sent {len(text)} characters to Translate tab."


def download_only_ui(link: str, progress=gr.Progress()) -> Tuple[Optional[str], Optional[str], str]:
    path, msg = download_youtube_mp3(link, progress=progress)
    if path is None:
        return None, None, msg
    return _audio_preview_path(path), path, msg


def transcribe_upload(
    audio_file,
    language: str,
    punctuation: bool,
    use_long_form: bool,
    hf_token: str,
    asr_max_tokens: int,
    progress=gr.Progress(),
) -> Tuple[str, str]:
    if audio_file is None:
        return "Please upload or record audio.", ""
    token = (hf_token or "").strip() or None
    if use_long_form:
        return transcribe_long(
            audio_file,
            language,
            punctuation,
            token,
            asr_max_tokens,
            progress=progress,
        )
    return transcribe_short(
        audio_file,
        language,
        punctuation,
        token,
        asr_max_tokens,
        progress=progress,
    )


def run_full_pipeline(
    youtube_url: str,
    hf_token: str,
    transcribe_language: str,
    punctuation: bool,
    use_long_form: bool,
    asr_max_tokens: int,
    translate_source: str,
    translate_target: str,
    tg_model_size: str,
    max_tokens: int,
    progress=gr.Progress(),
) -> Tuple[Optional[str], Optional[str], str, str, str, str]:
    log_lines: List[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)

    token = (hf_token or "").strip() or None

    progress(0.0, desc="Step 1/4: Downloading audio...")
    mp3_path, dl_msg = download_youtube_mp3(youtube_url, progress=progress)
    if mp3_path is None:
        return None, None, dl_msg, "", "", "\n".join(log_lines + [dl_msg])
    log(dl_msg)
    preview = _audio_preview_path(mp3_path)
    if not mp3_path.lower().endswith(".mp3"):
        return (
            None,
            mp3_path,
            dl_msg,
            "",
            "",
            "\n".join(
                log_lines
                + [
                    "Multiple MP3s were zipped. Download the ZIP below; preview works for a single MP3 only."
                ]
            ),
        )

    progress(0.25, desc="Step 2/4: Preparing ASR...")
    unload_translate_model()
    progress(0.3, desc="Step 3/4: Transcribing...")
    if use_long_form:
        transcript, tr_stats = transcribe_long(
            mp3_path,
            transcribe_language,
            punctuation,
            token,
            int(asr_max_tokens),
            progress=progress,
        )
    else:
        transcript, tr_stats = transcribe_short(
            mp3_path,
            transcribe_language,
            punctuation,
            token,
            int(asr_max_tokens),
            progress=progress,
        )
    log(tr_stats)
    if transcript.startswith("Error") or transcript.startswith("Please"):
        unload_asr_model()
        return (
            preview,
            mp3_path,
            dl_msg,
            transcript,
            "",
            "\n".join(log_lines),
        )

    progress(0.55, desc="Releasing ASR model...")
    unload_asr_model()

    progress(0.6, desc="Step 4/4: Loading translation model...")
    if token:
        set_hf_token(hf_token)
    load_msg = load_translate_model(tg_model_size, use_pipeline=True)
    log(load_msg)
    if "Error" in load_msg or "Authentication" in load_msg:
        return (
            preview,
            mp3_path,
            dl_msg,
            transcript,
            "",
            "\n".join(log_lines + [load_msg]),
        )

    src = translate_source or COHERE_TO_TRANSLATE_SOURCE.get(
        transcribe_language, "English"
    )
    progress(0.85, desc="Translating...")
    translated = translate_text_block(
        transcript, src, translate_target, int(max_tokens)
    )
    log("Done.")
    return preview, mp3_path, dl_msg, transcript, translated, "\n".join(log_lines)


def run_full_pipeline_tts(
    youtube_url: str,
    hf_token: str,
    transcribe_language: str,
    punctuation: bool,
    use_long_form: bool,
    asr_max_tokens: int,
    translate_source: str,
    translate_target: str,
    tg_model_size: str,
    max_tokens: int,
    tts_language: str,
    tts_mode: str,
    ref_audio,
    ref_text: str,
    tts_instruct: str,
    tts_steps: int,
    tts_guidance: float,
    tts_denoise: bool,
    tts_speed: float,
    tts_duration: float,
    tts_preprocess: bool,
    tts_postprocess: bool,
    tts_device: str,
    progress=gr.Progress(),
) -> Tuple[Optional[str], Optional[str], str, str, str, Optional[Tuple[int, np.ndarray]], str, str]:
    log_lines: List[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)

    token = (hf_token or "").strip() or None

    progress(0.0, desc="Step 1/5: Downloading audio...")
    mp3_path, dl_msg = download_youtube_mp3(youtube_url, progress=progress)
    if mp3_path is None:
        return None, None, dl_msg, "", "", None, "", "\n".join(log_lines + [dl_msg])
    log(dl_msg)
    preview = _audio_preview_path(mp3_path)
    if not mp3_path.lower().endswith(".mp3"):
        msg = "Multiple MP3s were zipped. Download the ZIP and retry with a single track for TTS."
        return None, mp3_path, dl_msg, "", "", None, msg, "\n".join(log_lines + [msg])

    progress(0.2, desc="Step 2/5: Preparing ASR...")
    unload_translate_model()
    progress(0.3, desc="Step 3/5: Transcribing...")
    if use_long_form:
        transcript, tr_stats = transcribe_long(
            mp3_path,
            transcribe_language,
            punctuation,
            token,
            int(asr_max_tokens),
            progress=progress,
        )
    else:
        transcript, tr_stats = transcribe_short(
            mp3_path,
            transcribe_language,
            punctuation,
            token,
            int(asr_max_tokens),
            progress=progress,
        )
    log(tr_stats)
    if transcript.startswith("Error") or transcript.startswith("Please"):
        unload_asr_model()
        return preview, mp3_path, dl_msg, transcript, "", None, "", "\n".join(log_lines)

    progress(0.55, desc="Releasing ASR model...")
    unload_asr_model()
    progress(0.62, desc="Step 4/5: Loading translation model...")
    if token:
        set_hf_token(hf_token)
    load_msg = load_translate_model(tg_model_size, use_pipeline=True)
    log(load_msg)
    if "Error" in load_msg or "Authentication" in load_msg:
        return preview, mp3_path, dl_msg, transcript, "", None, "", "\n".join(log_lines + [load_msg])
    src = translate_source or COHERE_TO_TRANSLATE_SOURCE.get(transcribe_language, "English")
    progress(0.78, desc="Translating...")
    translated = translate_text_block(transcript, src, translate_target, int(max_tokens))

    progress(0.86, desc="Step 5/5: Preparing OmniVoice...")
    unload_translate_model()
    tts_audio, tts_status = generate_omnivoice_tts(
        text=translated,
        tts_language=tts_language,
        tts_mode=tts_mode,
        ref_audio=ref_audio,
        ref_text=ref_text,
        tts_instruct=tts_instruct,
        num_step=int(tts_steps),
        guidance_scale=float(tts_guidance),
        denoise=bool(tts_denoise),
        speed=float(tts_speed),
        duration=float(tts_duration),
        preprocess_prompt=bool(tts_preprocess),
        postprocess_output=bool(tts_postprocess),
        tts_device=tts_device,
    )
    log(tts_status)
    log("Done.")
    return (
        preview,
        mp3_path,
        dl_msg,
        transcript,
        translated,
        tts_audio,
        tts_status,
        "\n".join(log_lines),
    )


def translate_only(
    transcript: str,
    hf_token: str,
    translate_source: str,
    translate_target: str,
    tg_model_size: str,
    max_tokens: int,
    progress=gr.Progress(),
) -> Tuple[str, str]:
    token = (hf_token or "").strip() or None
    progress(0.2, desc="Loading translation model...")
    unload_asr_model()
    if token:
        set_hf_token(hf_token)
    msg = load_translate_model(tg_model_size, use_pipeline=True)
    if "Error" in msg or "Authentication" in msg:
        return "", msg
    progress(0.7, desc="Translating...")
    out = translate_text_block(
        transcript, translate_source, translate_target, int(max_tokens)
    )
    return out, msg


def send_translation_to_omnivoice(translation: str) -> Tuple[object, str]:
    t = (translation or "").strip()
    if not t:
        return gr.update(), "No translation to send — translate first."
    return gr.update(value=t), f"Loaded {len(t)} characters into the OmniVoice tab."


def translate_and_synthesize(
    transcript: str,
    hf_token: str,
    translate_source: str,
    translate_target: str,
    tg_model_size: str,
    max_tokens: int,
    tts_language: str,
    tts_mode: str,
    ref_audio,
    ref_text: str,
    tts_instruct: str,
    tts_steps: int,
    tts_guidance: float,
    tts_denoise: bool,
    tts_speed: float,
    tts_duration: float,
    tts_preprocess: bool,
    tts_postprocess: bool,
    tts_device: str,
    progress=gr.Progress(),
) -> Tuple[str, str, Optional[Tuple[int, np.ndarray]], str, Optional[Tuple[int, np.ndarray]], str, object]:
    """
    Translate with TranslateGemma, then speak **only the translated text** with OmniVoice
    (same order as the full pipeline TTS step).
    """
    if not transcript or not transcript.strip():
        return "", "No text to translate.", None, "", None, "", gr.update()

    token = (hf_token or "").strip() or None
    progress(0.1, desc="Loading TranslateGemma…")
    unload_asr_model()
    if token:
        set_hf_token(hf_token)
    msg = load_translate_model(tg_model_size, use_pipeline=True)
    if "Error" in msg or "Authentication" in msg:
        return "", msg, None, "", None, "", gr.update()

    progress(0.35, desc="Translating…")
    translated = translate_text_block(
        transcript, translate_source, translate_target, int(max_tokens)
    )

    progress(0.65, desc="Unloading TranslateGemma, loading OmniVoice…")
    unload_translate_model()
    tts_audio, tts_status = generate_omnivoice_tts(
        text=translated,
        tts_language=tts_language,
        tts_mode=tts_mode,
        ref_audio=ref_audio,
        ref_text=ref_text,
        tts_instruct=tts_instruct,
        num_step=int(tts_steps),
        guidance_scale=float(tts_guidance),
        denoise=bool(tts_denoise),
        speed=float(tts_speed),
        duration=float(tts_duration),
        preprocess_prompt=bool(tts_preprocess),
        postprocess_output=bool(tts_postprocess),
        tts_device=tts_device,
    )
    progress(1.0, desc="Done")
    ov_text_update = gr.update(value=translated)
    return translated, msg, tts_audio, tts_status, tts_audio, tts_status, ov_text_update


def omnivoice_synthesize_only(
    text: str,
    tts_language: str,
    tts_mode: str,
    ref_audio,
    ref_text: str,
    tts_instruct: str,
    tts_steps: int,
    tts_guidance: float,
    tts_denoise: bool,
    tts_speed: float,
    tts_duration: float,
    tts_preprocess: bool,
    tts_postprocess: bool,
    tts_device: str,
    progress=gr.Progress(),
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    progress(0.05, desc="Releasing translation model (if loaded)…")
    unload_translate_model()
    progress(0.15, desc="Synthesizing…")
    audio, status = generate_omnivoice_tts(
        text=text,
        tts_language=tts_language,
        tts_mode=tts_mode,
        ref_audio=ref_audio,
        ref_text=ref_text,
        tts_instruct=tts_instruct,
        num_step=int(tts_steps),
        guidance_scale=float(tts_guidance),
        denoise=bool(tts_denoise),
        speed=float(tts_speed),
        duration=float(tts_duration),
        preprocess_prompt=bool(tts_preprocess),
        postprocess_output=bool(tts_postprocess),
        tts_device=tts_device,
    )
    progress(1.0, desc="Done")
    return audio, status
