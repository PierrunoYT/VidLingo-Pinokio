"""Gradio UI for VidLingo."""

from __future__ import annotations

import logging
import os

import gradio as gr

from asr import download_asr_model
from constants import COHERE_TO_TRANSLATE_SOURCE, LANGUAGES, SUPPORTED_LANGUAGES
from pipeline import (
    download_only_ui,
    omnivoice_synthesize_only,
    run_full_pipeline_tts,
    send_mp3_to_transcribe,
    send_transcription_to_translate,
    send_translation_to_omnivoice,
    transcribe_upload,
    translate_only,
)
from translate import set_hf_token


def build_ui() -> gr.Blocks:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    with gr.Blocks(title="VidLingo") as demo:
        gr.Markdown(
            """
            # VidLingo
            **YouTube → MP3 → transcribe (Cohere) → translate (TranslateGemma) → TTS (OmniVoice).**
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
            with gr.Tab("Full pipeline"):
                gr.Markdown(
                    "Download audio from YouTube, transcribe, translate, then synthesize speech with OmniVoice. "
                    "**Results appear below in order: audio → file → transcription → translation → TTS.**"
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
                gr.Markdown("### OmniVoice TTS settings")
                with gr.Row():
                    tts_lang = gr.Dropdown(
                        choices=["Auto"] + list(SUPPORTED_LANGUAGES.keys()),
                        value="Auto",
                        label="TTS language",
                    )
                    tts_mode = gr.Radio(
                        choices=["design", "clone"],
                        value="design",
                        label="TTS mode",
                    )
                    tts_device = gr.Dropdown(
                        choices=["auto", "cuda", "cpu", "mps"],
                        value="auto",
                        label="OmniVoice device",
                    )
                with gr.Row():
                    ref_audio = gr.Audio(
                        label="Reference audio (clone mode)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    ref_text = gr.Textbox(
                        label="Reference text (optional)",
                        lines=2,
                        placeholder="Transcript of the reference audio (improves cloning).",
                    )
                tts_instruct = gr.Textbox(
                    label="Voice design instruction (design mode)",
                    lines=2,
                    placeholder="Example: Warm, calm narration voice with clear pronunciation.",
                )
                with gr.Row():
                    tts_steps = gr.Slider(8, 64, value=32, step=1, label="TTS steps")
                    tts_guidance = gr.Slider(0.5, 6.0, value=2.0, step=0.1, label="TTS guidance")
                    tts_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="TTS speed")
                with gr.Row():
                    tts_duration = gr.Slider(0, 120, value=0, step=1, label="Target duration (0=auto, sec)")
                    tts_denoise = gr.Checkbox(label="Denoise", value=True)
                    tts_preprocess = gr.Checkbox(label="Preprocess prompt", value=True)
                    tts_postprocess = gr.Checkbox(label="Postprocess output", value=True)

                run_btn = gr.Button(
                    "Run: Download → Transcribe → Translate → TTS", variant="primary", size="lg"
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
                gr.Markdown("#### 3 — TTS (OmniVoice)")
                tts_audio_out = gr.Audio(
                    label="Synthesized speech",
                    type="numpy",
                    interactive=False,
                )
                tts_status = gr.Textbox(label="TTS status", interactive=False, lines=2)
                pipeline_log = gr.Textbox(label="Pipeline log", lines=5, interactive=False)

                lang_asr.change(_sync_src, [lang_asr], [src_tr])

                run_btn.click(
                    run_full_pipeline_tts,
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
                        tts_lang,
                        tts_mode,
                        ref_audio,
                        ref_text,
                        tts_instruct,
                        tts_steps,
                        tts_guidance,
                        tts_denoise,
                        tts_speed,
                        tts_duration,
                        tts_preprocess,
                        tts_postprocess,
                        tts_device,
                    ],
                    [
                        pipeline_audio,
                        mp3_out,
                        dl_status,
                        transcript_out,
                        translation_out,
                        tts_audio_out,
                        tts_status,
                        pipeline_log,
                    ],
                )

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

            with gr.Tab("OmniVoice TTS"):
                gr.Markdown(
                    "Synthesize speech from any text with **OmniVoice** (same options as the full pipeline). "
                    "Unload **TranslateGemma** automatically before TTS when VRAM is tight."
                )
                ov_text = gr.Textbox(
                    label="Text to speak",
                    lines=8,
                    placeholder="Paste translated text, or any script for TTS.",
                )
                with gr.Row():
                    ov_tts_lang = gr.Dropdown(
                        choices=["Auto"] + list(SUPPORTED_LANGUAGES.keys()),
                        value="Auto",
                        label="TTS language",
                    )
                    ov_tts_mode = gr.Radio(
                        choices=["design", "clone"],
                        value="design",
                        label="TTS mode",
                    )
                    ov_tts_device = gr.Dropdown(
                        choices=["auto", "cuda", "cpu", "mps"],
                        value="auto",
                        label="Device",
                    )
                with gr.Row():
                    ov_ref_audio = gr.Audio(
                        label="Reference audio (clone mode)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    ov_ref_text = gr.Textbox(
                        label="Reference text (optional)",
                        lines=2,
                        placeholder="Transcript of the reference audio (improves cloning).",
                    )
                ov_tts_instruct = gr.Textbox(
                    label="Voice design instruction (design mode)",
                    lines=2,
                    placeholder="Example: Warm, calm narration voice with clear pronunciation.",
                )
                with gr.Row():
                    ov_tts_steps = gr.Slider(8, 64, value=32, step=1, label="TTS steps")
                    ov_tts_guidance = gr.Slider(0.5, 6.0, value=2.0, step=0.1, label="TTS guidance")
                    ov_tts_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="TTS speed")
                with gr.Row():
                    ov_tts_duration = gr.Slider(0, 120, value=0, step=1, label="Target duration (0=auto, sec)")
                    ov_tts_denoise = gr.Checkbox(label="Denoise", value=True)
                    ov_tts_preprocess = gr.Checkbox(label="Preprocess prompt", value=True)
                    ov_tts_postprocess = gr.Checkbox(label="Postprocess output", value=True)

                ov_synth_btn = gr.Button("Synthesize speech", variant="primary", size="lg")
                ov_tts_audio = gr.Audio(
                    label="Synthesized speech",
                    type="numpy",
                    interactive=False,
                )
                ov_tts_status = gr.Textbox(label="TTS status", interactive=False, lines=2)

                ov_synth_btn.click(
                    omnivoice_synthesize_only,
                    [
                        ov_text,
                        ov_tts_lang,
                        ov_tts_mode,
                        ov_ref_audio,
                        ov_ref_text,
                        ov_tts_instruct,
                        ov_tts_steps,
                        ov_tts_guidance,
                        ov_tts_denoise,
                        ov_tts_speed,
                        ov_tts_duration,
                        ov_tts_preprocess,
                        ov_tts_postprocess,
                        ov_tts_device,
                    ],
                    [ov_tts_audio, ov_tts_status],
                )

            with gr.Tab("Translate text"):
                gr.Markdown(
                    "Translate text with **TranslateGemma** (loads the model on first use). "
                    "Use **Send translation to OmniVoice** to copy the result into the **OmniVoice TTS** tab."
                )
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
                send_to_ov_btn = gr.Button(
                    "Send translation to OmniVoice tab",
                    variant="secondary",
                )
                send_to_ov_status = gr.Textbox(
                    label="Send to OmniVoice",
                    interactive=False,
                    lines=1,
                )

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
                send_to_ov_btn.click(
                    send_translation_to_omnivoice,
                    [manual_out],
                    [ov_text, send_to_ov_status],
                )

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
