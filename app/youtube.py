"""YouTube → MP3 via yt-dlp."""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
import uuid
import zipfile
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import gradio as gr
import yt_dlp

from constants import DOWNLOAD_RETENTION, FFMPEG_EXE, OUTPUT_DIR, YOUTUBE_HOSTS

_log = logging.getLogger(__name__)

# Only directories this module created are ever pruned.
_JOB_DIR_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_youtube_url(link: str) -> bool:
    """Whether `link` is an http(s) URL on YouTube itself.

    A substring test accepts `https://youtube.com.evil.test/x`,
    `https://example.com/?youtube.com`, and `http://169.254.169.254/?youtube.com`;
    the host has to be parsed and compared to be meaningful.
    """
    try:
        parsed = urlparse(link)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return any(host == h or host.endswith(f".{h}") for h in YOUTUBE_HOSTS)


def _prune_old_jobs(keep: int = DOWNLOAD_RETENTION) -> None:
    """Best-effort retention: drop all but the `keep` newest job directories."""
    try:
        jobs = sorted(
            name
            for name in os.listdir(OUTPUT_DIR)
            if _JOB_DIR_RE.match(name)
            and os.path.isdir(os.path.join(OUTPUT_DIR, name))
        )
    except OSError:
        return
    for name in jobs[: max(0, len(jobs) - keep)]:
        shutil.rmtree(os.path.join(OUTPUT_DIR, name), ignore_errors=True)


def _new_job_dir() -> str:
    """A private directory for one download, so concurrent jobs cannot collide."""
    _ensure_dir(OUTPUT_DIR)
    _prune_old_jobs(keep=max(0, DOWNLOAD_RETENTION - 1))
    job_dir = os.path.join(
        OUTPUT_DIR, f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


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
    # Validate before touching disk: a rejected link must not destroy the
    # output of a previous download.
    progress(0.01, desc="Validating link")
    if not _is_youtube_url(link):
        return None, "Unsupported link. Please use a YouTube URL."
    _log.info("[yt-dlp] fetching audio for: %s", link)
    print(f"[yt-dlp] fetching audio for: {link}", flush=True)
    try:
        job_dir = _new_job_dir()
        files = _yt_dlp_download([link], job_dir, progress)
        if not files:
            return None, "No files were downloaded. Check the link or ffmpeg."
        return _zip_if_needed(job_dir, files)
    except Exception as exc:
        return None, f"Error: {exc}"
