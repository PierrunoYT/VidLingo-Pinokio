"""
VidLingo: YouTube → MP3 (yt-dlp) → Cohere Transcribe → TranslateGemma → OmniVoice TTS.

Entry point — implementation lives in sibling modules (`ui`, `pipeline`, `asr`, …).
"""

from __future__ import annotations

import os

from ui import THEME, build_ui

if __name__ == "__main__":
    # Gradio 6 moved app-level settings such as `theme` from `gr.Blocks()` to
    # `launch()`; passing it to Blocks is deprecated and warns on every start.
    build_ui().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=THEME,
    )
