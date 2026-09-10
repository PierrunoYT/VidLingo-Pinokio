"""Shared paths and language / model constants for VidLingo."""

from __future__ import annotations

import os

import imageio_ffmpeg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "downloads")
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

MODEL_ID_ASR = "CohereLabs/cohere-transcribe-03-2026"

# Every remote model is pinned to a commit SHA. Tracking a branch means the
# weights can change under a working install; bump these deliberately after
# testing rather than letting an install pick up whatever is on `main`.
# Resolved from the Hugging Face API on 2026-09-10.
ASR_REVISION = os.environ.get(
    "VIDLINGO_ASR_REVISION", "b1eacc2686a3d08ceaae5f24a88b1d519620bc09"
)
TRANSLATEGEMMA_REVISIONS = {
    "4B": os.environ.get(
        "VIDLINGO_TG_4B_REVISION", "10042cb0e6e7fdce748996a71dc3dc432a4e0c89"
    ),
    "12B": os.environ.get(
        "VIDLINGO_TG_12B_REVISION", "d1b225e1caa17f1ddc7e62065d8637d0923f34e2"
    ),
    "27B": os.environ.get(
        "VIDLINGO_TG_27B_REVISION", "7d10f0b72f89a2d0f268cea30727d8b77c0d25c2"
    ),
}
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

YOUTUBE_HOSTS = ("youtube.com", "youtu.be")

# How many past download job directories to keep. Each request downloads into
# its own directory, so jobs cannot delete each other's files; older ones are
# pruned on the way in.
DOWNLOAD_RETENTION = max(1, int(os.environ.get("VIDLINGO_DOWNLOAD_RETENTION", "5")))

OMNIVOICE_CHECKPOINT = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
OMNIVOICE_REVISION = os.environ.get(
    "VIDLINGO_OMNIVOICE_REVISION", "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
)
OMNIVOICE_LOAD_ASR_DEFAULT = False
