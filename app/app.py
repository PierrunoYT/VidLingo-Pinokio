"""
VidLingo: YouTube → MP3 (yt-dlp) → Cohere Transcribe → TranslateGemma → OmniVoice TTS.

Entry point — implementation lives in sibling modules (`ui`, `pipeline`, `asr`, …).
"""

from __future__ import annotations

import os

from ui import build_ui

if __name__ == "__main__":
    # The theme belongs to `gr.Blocks(...)`, not `launch()`: `launch(theme=...)`
    # only exists from Gradio 6 onward and raises TypeError on the 5.x the
    # requirement spec still allows, which would fail the app at startup.
    build_ui().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
