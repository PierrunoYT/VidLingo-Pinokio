# Transcribe Studio (Pinokio)

Single Gradio app that chains:

1. **YouTube → MP3** — `yt-dlp` + FFmpeg (same idea as [Youtube2DL-Pinokio](./Youtube2DL-Pinokio)).
2. **Transcription** — [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) (same stack as [cohere-transcribe-pinokio](./cohere-transcribe-pinokio)).
3. **Translation** — [TranslateGemma](https://huggingface.co/google/translategemma-12b-it) (same idea as [TranslateGemma-Pinokio](./TranslateGemma-Pinokio)).

Between ASR and translation, the ASR model is unloaded from GPU/RAM so TranslateGemma can load; accept both model licenses on Hugging Face and use a read token where required.

## How to use (Pinokio)

1. Install the app from this folder, then **Start**.
2. Open the Web UI, paste a **Hugging Face token** if you have not logged in on the machine.
3. Optionally click **Pre-download ASR model** to cache Cohere weights.
4. Paste a **YouTube URL**, set **spoken language**, **translation source/target**, and **TranslateGemma size**, then **Run: Download → Transcribe → Translate**.

The sibling folders `Youtube2DL-Pinokio`, `cohere-transcribe-pinokio`, and `TranslateGemma-Pinokio` remain standalone references; this repo’s `app/` implements the combined workflow.

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
#     "English",  # translate_source
#     "Spanish",  # translate_target
#     "12B",  # tg_model_size
#     400,  # max_tokens
#     api_name="/run_full_pipeline",
# )
```

### JavaScript

Use the same base URL and call the Gradio HTTP API (see `/info` or `/openapi.json` on the Gradio server) or use `@gradio/client` in the browser with the same `api_name` values as in `view_api()`.

### curl

Gradio exposes REST routes under the app root; exact paths vary by version. Prefer `GET {base}/openapi.json` or `client.view_api()` to obtain the current `api_name` and payload order.

---

Subprojects for reference: [Youtube2DL-Pinokio](./Youtube2DL-Pinokio), [cohere-transcribe-pinokio](./cohere-transcribe-pinokio), [TranslateGemma-Pinokio](./TranslateGemma-Pinokio).
