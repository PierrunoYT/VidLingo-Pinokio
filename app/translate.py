"""TranslateGemma loading and text translation."""

from __future__ import annotations

import gc
import re
from typing import Optional

import torch
from huggingface_hub import login
from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline

from constants import LANGUAGES

model = None
processor = None
pipe = None
current_model_size: Optional[str] = None
hf_token_set = False


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
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
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
