"""
VidLingo: YouTube → MP3 (yt-dlp) → Cohere Transcribe → TranslateGemma → OmniVoice TTS.

Entry point — implementation lives in sibling modules (`ui`, `pipeline`, `asr`, …).
"""

from __future__ import annotations

import os

import gradio as gr

from ui import build_ui

_THEME = gr.themes.Soft(primary_hue="cyan", secondary_hue="slate", neutral_hue="slate")

if __name__ == "__main__":
    build_ui().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=_THEME,
    )
