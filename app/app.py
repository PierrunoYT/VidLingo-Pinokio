"""
VidLingo: YouTube → MP3 (yt-dlp) → Cohere Transcribe → TranslateGemma.
Combines workflows from Youtube2DL-Pinokio, cohere-transcribe-pinokio, TranslateGemma-Pinokio.
"""
from __future__ import annotations

import gc
import logging
import os
import shutil
import time
import zipfile
from typing import List, Optional, Tuple

import gradio as gr
import imageio_ffmpeg
import torch
import yt_dlp
from huggingface_hub import login
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    CohereAsrForConditionalGeneration,
    pipeline,
)
from transformers.audio_utils import load_audio

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "downloads")
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# --- Cohere ASR ---
MODEL_ID_ASR = "CohereLabs/cohere-transcribe-03-2026"
SUPPORTED_LANGUAGES = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Spanish": "es",
    "Portuguese": "pt",
    "Greek": "el",
    "Dutch": "nl",
    "Polish": "pl",
    "Arabic": "ar",
    "Vietnamese": "vi",
    "Chinese (Mandarin)": "zh",
    "Japanese": "ja",
    "Korean": "ko",
}
_model_cache_asr: dict = {}

# Map Cohere UI labels to TranslateGemma source language names
COHERE_TO_TRANSLATE_SOURCE = {
    "English": "English",
    "French": "French",
    "German": "German",
    "Italian": "Italian",
    "Spanish": "Spanish",
    "Portuguese": "Portuguese",
    "Greek": "Greek",
    "Dutch": "Dutch",
    "Polish": "Polish",
    "Arabic": "Arabic",
    "Vietnamese": "Vietnamese",
    "Chinese (Mandarin)": "Chinese (Simplified)",
    "Japanese": "Japanese",
    "Korean": "Korean",
}

# --- TranslateGemma (subset of full LANGUAGES for UI; full dict for lookup) ---
LANGUAGES = {
    "Arabic": "ar",
    "Bengali": "bn",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese (Simplified)": "zh",
    "Chinese (Traditional)": "zh-TW",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "English (US)": "en-US",
    "English (UK)": "en-GB",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "French (Canada)": "fr-CA",
    "German": "de",
    "German (Austria)": "de-AT",
    "German (Switzerland)": "de-CH",
    "Greek": "el",
    "Gujarati": "gu",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Kannada": "kn",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Norwegian": "no",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Portuguese (Brazil)": "pt-BR",
    "Portuguese (Portugal)": "pt-PT",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian": "sr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Spanish (Mexico)": "es-MX",
    "Spanish (Spain)": "es-ES",
    "Swedish": "sv",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
}

# TranslateGemma globals
model = None
processor = None
pipe = None
current_model_size: Optional[str] = None
hf_token_set = False

YOUTUBE_HOSTS = ("youtube.com", "youtu.be")

_log = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _collect_output_files(output_dir: str) -> List[str]:
    files = []
    for name in os.listdir(output_dir):
        if name.lower().endswith(".mp3"):
            files.append(os.path.join(output_dir, name))
    return sorted(files)


def _zip_if_needed(output_dir: str, downloaded_files: List[str]) -> Tuple[str, str]:
    if len(downloaded_files) == 1:
        return downloaded_files[0], "Downloaded 1 file."
    zip_path = os.path.join(output_dir, "downloads.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path in downloaded_files:
            zipf.write(file_path, arcname=os.path.basename(file_path))
    return zip_path, f"Downloaded {len(downloaded_files)} files (zipped)."


def _yt_dlp_download(
    targets: List[str], output_dir: str, progress: gr.Progress
) -> List[str]:
    _ensure_dir(output_dir)
    status = {"current": "", "percent": 0}

    def _hook(d):
        st = d.get("status")
        if st == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            if total:
                status["percent"] = int(downloaded * 100 / total)
            status["current"] = d.get("filename") or d.get("tmpfilename") or ""
            pct = status["percent"]
            name = os.path.basename(status["current"]) if status["current"] else "?"
            line = f"[yt-dlp] downloading {pct}% — {name}"
            _log.info(line)
            print(line, flush=True)
            progress(
                min(pct / 100, 0.95),
                desc=f"Downloading {name}",
            )
        elif st == "finished":
            fn = d.get("filename", "")
            line = f"[yt-dlp] finished: {fn}"
            _log.info(line)
            print(line, flush=True)
        elif st == "postprocessing":
            info = d.get("postprocessor") or "ffmpeg"
            line = f"[yt-dlp] post-processing ({info}) — converting to MP3…"
            _log.info(line)
            print(line, flush=True)
            progress(0.92, desc="Converting to MP3 (ffmpeg)…")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        # Keep yt-dlp quiet here; we log via progress_hooks so Pinokio / terminal still shows activity.
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": FFMPEG_EXE,
        "progress_hooks": [_hook],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for idx, target in enumerate(targets, start=1):
            line = f"[yt-dlp] starting {idx}/{len(targets)}: {target}"
            _log.info(line)
            print(line, flush=True)
            progress(0.05, desc=f"Preparing {idx}/{len(targets)}")
            ydl.download([target])

    progress(0.98, desc="Finalizing")
    _log.info("[yt-dlp] download pass complete, collecting files…")
    print("[yt-dlp] download pass complete, collecting files…", flush=True)
    return _collect_output_files(output_dir)


def download_youtube_mp3(link: str, progress=gr.Progress()) -> Tuple[Optional[str], str]:
    if not link or not link.strip():
        return None, "Please provide a YouTube link."
    link = link.strip()
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    progress(0.01, desc="Validating link")
    _log.info("[yt-dlp] fetching audio for: %s", link)
    print(f"[yt-dlp] fetching audio for: {link}", flush=True)
    try:
        if any(host in link for host in YOUTUBE_HOSTS):
            files = _yt_dlp_download([link], OUTPUT_DIR, progress)
            if not files:
                return None, "No files were downloaded. Check the link or ffmpeg."
            out_path, msg = _zip_if_needed(OUTPUT_DIR, files)
            return out_path, msg
        return None, "Unsupported link. Please use a YouTube URL."
    except Exception as exc:
        return None, f"Error: {exc}"


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


def set_hf_token(token: str) -> str:
    global hf_token_set
    if not token or not token.strip():
        return "Enter a Hugging Face token (required for gated models)."
    try:
        login(token=token.strip(), add_to_git_credential=False)
        hf_token_set = True
        return "Hugging Face token accepted."
    except Exception as e:
        return f"Token error: {e}"


def unload_translate_model() -> None:
    global model, processor, pipe, current_model_size
    if pipe is not None:
        del pipe
        pipe = None
    if model is not None:
        del model
        model = None
    if processor is not None:
        del processor
        processor = None
    current_model_size = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_translate_model(model_size: str = "12B", use_pipeline: bool = True) -> str:
    global model, processor, pipe, current_model_size
    if current_model_size == model_size and (pipe is not None or model is not None):
        return f"TranslateGemma {model_size} already loaded."

    unload_translate_model()

    model_id = f"google/translategemma-{model_size.lower()}-it"
    try:
        if use_pipeline:
            pipe = pipeline(
                "image-text-to-text",
                model=model_id,
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            current_model_size = model_size
            return f"TranslateGemma {model_size} loaded (CUDA: {torch.cuda.is_available()})."
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        current_model_size = model_size
        return f"TranslateGemma {model_size} loaded (CUDA: {torch.cuda.is_available()})."
    except Exception as e:
        err = str(e)
        if "401" in err or "authentication" in err.lower():
            return (
                f"Authentication error. Set HF token and accept the license: "
                f"https://huggingface.co/{model_id}"
            )
        return f"Error loading TranslateGemma: {err}"


def _split_into_chunks(text: str, max_words: int = 300) -> list[str]:
    """Split text into sentence-aware chunks of at most *max_words* words."""
    import re

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        words = len(sentence.split())
        if current_words + words > max_words and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_words = words
        else:
            current.append(sentence)
            current_words += words

    if current:
        chunks.append(" ".join(current))

    return chunks


def _translate_single_chunk(
    text: str,
    source_code: str,
    target_code: str,
    max_tokens: int,
) -> str:
    global pipe, model, processor

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_code,
                    "target_lang_code": target_code,
                    "text": text,
                }
            ],
        }
    ]
    if pipe is not None:
        # Pass generation kwargs directly — pipeline does not accept generation_config
        output = pipe(text=messages, max_new_tokens=max_tokens, do_sample=False)
        return output[0]["generated_text"][-1]["content"]
    inputs = (
        processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        .to(model.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    )
    input_len = len(inputs["input_ids"][0])
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
        generation = generation[0][input_len:]
    return processor.decode(generation, skip_special_tokens=True)


def translate_text_block(
    text: str,
    source_lang: str,
    target_lang: str,
    max_tokens: int,
) -> str:
    global pipe, model, processor

    if not text or not text.strip():
        return "No text to translate."
    if pipe is None and model is None:
        return "Load TranslateGemma first (pipeline will load it automatically)."

    source_code = LANGUAGES.get(source_lang, "en")
    target_code = LANGUAGES.get(target_lang, "es")
    mt = max(64, int(max_tokens))

    chunks = _split_into_chunks(text, max_words=300)
    translated_parts: list[str] = []
    try:
        for chunk in chunks:
            part = _translate_single_chunk(chunk, source_code, target_code, mt)
            translated_parts.append(part)
        return " ".join(translated_parts)
    except Exception as e:
        partial = " ".join(translated_parts)
        return f"{partial}\n\nTranslation error: {e}" if partial else f"Translation error: {e}"


def _audio_preview_path(path: Optional[str]) -> Optional[str]:
    """Only single-file MP3 can be played in the browser; ZIP / other formats return None."""
    if path and path.lower().endswith(".mp3") and os.path.isfile(path):
        return path
    return None


def _resolve_downloaded_mp3(
    only_audio: object, only_file: object
) -> Tuple[Optional[str], str]:
    """Get a filesystem path to a single MP3 from YouTube-only tab outputs."""
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
    """Fill both short- and long-form Audio inputs with the downloaded MP3 path."""
    path, err = _resolve_downloaded_mp3(only_audio, only_file)
    if not path:
        return gr.update(), gr.update(), err
    upd = gr.update(value=path)
    return upd, upd, f"Loaded into Transcribe: {os.path.basename(path)}"


def send_transcription_to_translate(out_short: str, out_long: str) -> Tuple[object, str]:
    """Copy the latest transcription into the Translate text tab."""
    long_t = (out_long or "").strip()
    short_t = (out_short or "").strip()
    text = long_t or short_t
    if not text:
        return gr.update(), "No transcription yet — run Transcribe on the Short-form or Long-form tab first."
    return gr.update(value=text), f"Sent {len(text)} characters to Translate tab."


def download_only_ui(link: str, progress=gr.Progress()) -> Tuple[Optional[str], Optional[str], str]:
    """YouTube → file + optional MP3 playback path."""
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
    """Transcribe an uploaded / recorded audio file (no YouTube step)."""
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
    """
    Returns: audio_preview_path, file_path, download_status, transcription, translation, log
    """
    log_lines: List[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)

    token = (hf_token or "").strip() or None

    # 1) YouTube → MP3
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

    # 2) Transcribe (unload translation model to free VRAM)
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

    # 3) Unload ASR before TranslateGemma
    progress(0.55, desc="Releasing ASR model...")
    unload_asr_model()

    # 4) HF login + TranslateGemma
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


_THEME = gr.themes.Soft(primary_hue="cyan", secondary_hue="slate", neutral_hue="slate")


def build_ui() -> gr.Blocks:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    with gr.Blocks(title="VidLingo") as demo:
        gr.Markdown(
            """
            # VidLingo
            **YouTube → MP3 → transcribe (Cohere) → translate (TranslateGemma).**
            """
        )

        hf_token = gr.Textbox(
            label="Hugging Face token",
            type="password",
            placeholder="hf_... (needed for gated models: Cohere ASR, TranslateGemma)",
        )
        with gr.Row():
            token_btn = gr.Button("Log in to Hugging Face", variant="secondary")
            asr_cache_btn = gr.Button("Pre-download ASR model", variant="secondary")
        token_status = gr.Textbox(label="Token / cache status", interactive=False, lines=2)

        token_btn.click(set_hf_token, [hf_token], [token_status])
        asr_cache_btn.click(download_asr_model, [hf_token], [token_status])

        def _sync_src(cohere_label: str):
            return COHERE_TO_TRANSLATE_SOURCE.get(cohere_label, "English")

        with gr.Tabs():
            # --- Full pipeline ---
            with gr.Tab("Full pipeline"):
                gr.Markdown(
                    "Download audio from YouTube, transcribe, then translate. "
                    "**Results appear below in order: audio → file → transcription → translation.**"
                )
                yt = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
                with gr.Row():
                    lang_asr = gr.Dropdown(
                        choices=list(SUPPORTED_LANGUAGES.keys()),
                        value="English",
                        label="Spoken language (transcription)",
                    )
                    punct = gr.Checkbox(label="Punctuation", value=True)
                    long_form = gr.Checkbox(
                        label="Long-form transcription (recommended for videos)",
                        value=True,
                    )
                asr_max_tok = gr.Slider(
                    64,
                    2048,
                    value=256,
                    step=32,
                    label="Max transcription tokens (Cohere)",
                )
                gr.Markdown(
                    "*Caps Cohere ASR output length — increase if transcription is cut off.*"
                )
                with gr.Row():
                    src_tr = gr.Dropdown(
                        choices=list(LANGUAGES.keys()),
                        value="English",
                        label="Translation: source language",
                    )
                    tgt_tr = gr.Dropdown(
                        choices=list(LANGUAGES.keys()),
                        value="Spanish",
                        label="Translation: target language",
                    )
                with gr.Row():
                    tg_size = gr.Radio(
                        choices=["4B", "12B", "27B"],
                        value="4B",
                        label="TranslateGemma size",
                    )
                    max_tok = gr.Slider(
                        50, 2048, value=512, step=10, label="Max translation tokens (per chunk)"
                    )

                run_btn = gr.Button(
                    "Run: Download → Transcribe → Translate", variant="primary", size="lg"
                )

                gr.Markdown("#### Listen — converted MP3")
                pipeline_audio = gr.Audio(
                    label="Play audio",
                    type="filepath",
                    interactive=False,
                )
                gr.Markdown("#### Download — file on disk (MP3 or ZIP)")
                mp3_out = gr.File(
                    label="Download file",
                    file_count="single",
                )
                dl_status = gr.Textbox(label="Download status", interactive=False, lines=2)

                gr.Markdown("#### 1 — Transcription (Cohere)")
                transcript_out = gr.Textbox(
                    label="Transcription",
                    lines=12,
                )
                gr.Markdown("#### 2 — Translation (TranslateGemma)")
                translation_out = gr.Textbox(
                    label="Translation",
                    lines=12,
                )
                pipeline_log = gr.Textbox(label="Pipeline log", lines=5, interactive=False)

                lang_asr.change(_sync_src, [lang_asr], [src_tr])

                run_btn.click(
                    run_full_pipeline,
                    [
                        yt,
                        hf_token,
                        lang_asr,
                        punct,
                        long_form,
                        asr_max_tok,
                        src_tr,
                        tgt_tr,
                        tg_size,
                        max_tok,
                    ],
                    [
                        pipeline_audio,
                        mp3_out,
                        dl_status,
                        transcript_out,
                        translation_out,
                        pipeline_log,
                    ],
                )

            # --- YouTube MP3 only ---
            with gr.Tab("YouTube → MP3 only"):
                gr.Markdown("Download audio as MP3 (or ZIP if multiple tracks). Playback works for a **single MP3**.")
                yt_only = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
                dl_only_btn = gr.Button("Download", variant="primary")
                only_audio = gr.Audio(
                    label="Play MP3",
                    type="filepath",
                    interactive=False,
                )
                only_file = gr.File(label="Download file", file_count="single")
                only_status = gr.Textbox(label="Status", interactive=False, lines=2)
                dl_only_btn.click(
                    download_only_ui,
                    [yt_only],
                    [only_audio, only_file, only_status],
                )
                send_to_asr_btn = gr.Button(
                    "Send MP3 to Transcribe tab",
                    variant="secondary",
                )
                send_mp3_status = gr.Textbox(
                    label="Send to transcribe",
                    interactive=False,
                    lines=1,
                )
                gr.Markdown(
                    "*Opens the same file in **Short-form** and **Long-form** audio inputs — switch to the Transcribe tab to run.*"
                )

            # --- Transcribe upload ---
            with gr.Tab("Transcribe audio"):
                gr.Markdown(
                    "Upload or record audio and transcribe with **Cohere** (same models as the pipeline). "
                    "Use **Short** for clips ~30s; **Long** for full tracks with chunking."
                )
                asr_upload_tokens = gr.Slider(
                    64,
                    2048,
                    value=256,
                    step=32,
                    label="Max transcription tokens (Cohere)",
                )
                with gr.Tabs():
                    with gr.Tab("Short-form"):
                        au_short = gr.Audio(
                            label="Audio",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        lang_s = gr.Dropdown(
                            choices=list(SUPPORTED_LANGUAGES.keys()),
                            value="English",
                            label="Language",
                        )
                        punct_s = gr.Checkbox(label="Punctuation", value=True)
                        btn_s = gr.Button("Transcribe", variant="primary")
                        out_s = gr.Textbox(label="Transcription", lines=10)
                        stats_s = gr.Textbox(label="Statistics", interactive=False, lines=2)

                        def _ts_short(audio, lang, punc, tok, asr_mt):
                            return transcribe_upload(
                                audio, lang, punc, False, tok, asr_mt
                            )

                        btn_s.click(
                            _ts_short,
                            [au_short, lang_s, punct_s, hf_token, asr_upload_tokens],
                            [out_s, stats_s],
                        )

                    with gr.Tab("Long-form"):
                        au_long = gr.Audio(
                            label="Audio",
                            type="filepath",
                            sources=["upload"],
                        )
                        lang_l = gr.Dropdown(
                            choices=list(SUPPORTED_LANGUAGES.keys()),
                            value="English",
                            label="Language",
                        )
                        punct_l = gr.Checkbox(label="Punctuation", value=True)
                        btn_l = gr.Button("Transcribe long audio", variant="primary")
                        out_l = gr.Textbox(
                            label="Transcription", lines=12
                        )
                        stats_l = gr.Textbox(label="Statistics", interactive=False, lines=2)

                        def _ts_long(audio, lang, punc, tok, asr_mt):
                            return transcribe_upload(
                                audio, lang, punc, True, tok, asr_mt
                            )

                        btn_l.click(
                            _ts_long,
                            [au_long, lang_l, punct_l, hf_token, asr_upload_tokens],
                            [out_l, stats_l],
                        )

                send_to_tr_btn = gr.Button(
                    "Send transcription to Translate tab",
                    variant="secondary",
                )
                send_tr_status = gr.Textbox(
                    label="Send to translate",
                    interactive=False,
                    lines=1,
                )
                gr.Markdown(
                    "*Prefers **Long-form** transcription if present, otherwise Short-form. Switch to **Translate text** to run.*"
                )

            # --- Translate text ---
            with gr.Tab("Translate text"):
                gr.Markdown("Translate text with **TranslateGemma** (loads the model on first use).")
                manual_text = gr.Textbox(
                    label="Text to translate",
                    lines=8,
                )
                with gr.Row():
                    manual_src = gr.Dropdown(
                        choices=list(LANGUAGES.keys()),
                        value="English",
                        label="From",
                    )
                    manual_tgt = gr.Dropdown(
                        choices=list(LANGUAGES.keys()),
                        value="French",
                        label="To",
                    )
                with gr.Row():
                    manual_size = gr.Radio(
                        choices=["4B", "12B", "27B"],
                        value="4B",
                        label="Model size",
                    )
                    manual_max = gr.Slider(
                        50, 2048, value=512, step=10, label="Max tokens (per chunk)"
                    )
                tr_only_btn = gr.Button("Translate", variant="primary")
                manual_out = gr.Textbox(
                    label="Translation",
                    lines=10,
                )
                manual_status = gr.Textbox(label="Model / status", interactive=False, lines=2)

                gr.Examples(
                    examples=[
                        ["Hello, how are you today?", "English", "Spanish"],
                        ["Bonjour, comment allez-vous?", "French", "English"],
                        ["こんにちは、元気ですか？", "Japanese", "English"],
                    ],
                    inputs=[manual_text, manual_src, manual_tgt],
                    label="Examples",
                )

                tr_only_btn.click(
                    translate_only,
                    [
                        manual_text,
                        hf_token,
                        manual_src,
                        manual_tgt,
                        manual_size,
                        manual_max,
                    ],
                    [manual_out, manual_status],
                )

        # Cross-tab: YouTube MP3 → Transcribe; Transcribe → Translate (after all components exist)
        send_to_asr_btn.click(
            send_mp3_to_transcribe,
            [only_audio, only_file],
            [au_short, au_long, send_mp3_status],
        )
        send_to_tr_btn.click(
            send_transcription_to_translate,
            [out_s, out_l],
            [manual_text, send_tr_status],
        )

        gr.Markdown(
            """
            ### Notes
            - Accept model licenses on Hugging Face for
              [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) and
              [TranslateGemma](https://huggingface.co/google/translategemma-12b-it).
            - VRAM: the full pipeline unloads the ASR model before loading TranslateGemma to reduce peak memory.
            """
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=_THEME,
    )
