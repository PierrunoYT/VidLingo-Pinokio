"""Shared paths and language / model constants for VidLingo."""

from __future__ import annotations

import os

import imageio_ffmpeg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "downloads")
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

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

OMNIVOICE_CHECKPOINT = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
OMNIVOICE_LOAD_ASR_DEFAULT = False
