# VidLingo

Repository: [https://github.com/PierrunoYT/VidLingo-Pinokio](https://github.com/PierrunoYT/VidLingo-Pinokio)

**VidLingo** is a Pinokio Gradio app that chains:

1. **YouTube → MP3** — `yt-dlp` + FFmpeg.
2. **Transcription** — [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).
3. **Translation** — [TranslateGemma](https://huggingface.co/google/translategemma-12b-it).
4. **TTS** — [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) for voice design / cloning from translated text.

Between ASR and translation, the ASR model is unloaded from GPU/RAM so TranslateGemma can load; accept both model licenses on Hugging Face and use a read token where required.

## Supported platforms

Windows, Linux, and **Apple Silicon** macOS.

**Intel macOS is not supported.** OmniVoice requires PyTorch 2.4 or newer, and PyTorch publishes no Intel-Mac (x86-64 macOS) builds past 2.2.2. Install, Update, and Start stop with an explanation on that platform rather than building an environment that would fail at the TTS step.

## How to use (Pinokio)

1. Install the app from this folder, then **Start**.
2. Open the Web UI, paste a **Hugging Face token** if you have not logged in on the machine.
3. Optionally click **Pre-download ASR model** to cache Cohere weights.
4. Paste a **YouTube URL**, set **spoken language**, **translation source/target**, **TranslateGemma size**, and **OmniVoice TTS settings**, then **Run: Download → Transcribe → Translate → TTS**.

This repo’s `app/` implements the combined workflow. The standalone launchers it draws on (`Youtube2DL-Pinokio`, `cohere-transcribe-pinokio`, `TranslateGemma-Pinokio`, `OmniVoice-Pinokio`) are separate projects — they are not part of this repository, and `.gitignore` excludes them so they can be cloned alongside it for reference.

## Programmatic API (Gradio)

After the server is running, discover endpoints with the Gradio client:

### Python (`gradio_client`)

```python
from gradio_client import Client

base = "http://127.0.0.1:7860"  # use the URL Pinokio shows
client = Client(base)
# List callable APIs (names depend on Gradio version):
print(client.view_api())
# Example (adjust fn_index / api_name to match view_api output):
# result = client.predict(
#     "https://www.youtube.com/watch?v=...",  # youtube_url
#     "hf_xxx",  # hf_token
#     "English",  # transcribe_language
#     True,  # punctuation
#     True,  # use_long_form
#     256,  # asr_max_tokens
#     "English",  # translate_source
#     "Spanish",  # translate_target
#     "4B",  # tg_model_size
#     512,  # max_tokens
#     # ...followed by the OmniVoice TTS arguments (language, mode, reference
#     # audio/text, instruction, steps, guidance, denoise, speed, duration,
#     # preprocess, postprocess, device) — see view_api() for the exact order.
#     api_name="/run_full_pipeline_tts",
# )
```

### JavaScript

Use the same base URL and call the Gradio HTTP API (see `/info` or `/openapi.json` on the Gradio server) or use `@gradio/client` in the browser with the same `api_name` values as in `view_api()`.

### curl

Gradio exposes REST routes under the app root; exact paths vary by version. Prefer `GET {base}/openapi.json` or `client.view_api()` to obtain the current `api_name` and payload order.

## Development

Dependencies are locked. `app/requirements.txt` is the input spec; the launcher
installs the resolved `app/requirements.lock.txt`. After editing the spec:

```bash
python tools/relock.py
```

Remote models are pinned to commit SHAs in `app/constants.py` — bump them
deliberately after testing, or override per-machine with the `VIDLINGO_*_REVISION`
environment variables.

Tests cover the failure paths that previously broke silently, and stub every
heavy dependency, so they need no models and no network:

```bash
python -m pytest
```
