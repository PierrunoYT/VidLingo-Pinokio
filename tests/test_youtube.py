"""Link validation and download storage.

The validator was a substring test, and every request deleted the shared
output directory before validating — so an unsupported link destroyed a
previous download.
"""

from __future__ import annotations

import os

import pytest

from youtube import _is_youtube_url, _new_job_dir, _prune_old_jobs, download_youtube_mp3


@pytest.mark.parametrize(
    "link",
    [
        "https://www.youtube.com/watch?v=abc",
        "https://youtube.com/watch?v=abc",
        "http://m.youtube.com/watch?v=abc",
        "https://music.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
    ],
)
def test_accepts_youtube(link):
    assert _is_youtube_url(link)


@pytest.mark.parametrize(
    "link",
    [
        pytest.param("https://example.com/?youtube.com", id="host-in-query"),
        pytest.param("https://youtube.com.evil.test/video", id="suffix-lookalike"),
        pytest.param(
            "http://169.254.169.254/latest/meta-data/?youtube.com", id="link-local-ssrf"
        ),
        pytest.param("file:///etc/passwd?youtube.com", id="non-http-scheme"),
        pytest.param("ftp://youtube.com/x", id="ftp"),
        pytest.param("not a url", id="not-a-url"),
        pytest.param("", id="empty"),
    ],
)
def test_rejects_everything_else(link):
    assert not _is_youtube_url(link)


def test_invalid_link_does_not_destroy_previous_downloads(downloads_dir):
    previous = downloads_dir / "20200101-000000-aaaaaaaa"
    previous.mkdir()
    kept = previous / "already-downloaded.mp3"
    kept.write_bytes(b"audio")

    path, message = download_youtube_mp3("https://example.com/?youtube.com")

    assert path is None
    assert "Unsupported link" in message
    assert kept.is_file(), "a rejected link deleted an earlier download"


def test_each_job_gets_its_own_directory(downloads_dir):
    first = _new_job_dir()
    second = _new_job_dir()
    assert first != second
    assert os.path.isdir(first) and os.path.isdir(second)


def test_retention_prunes_oldest_jobs_only(downloads_dir):
    for name in (
        "20200101-000000-aaaaaaaa",
        "20200102-000000-bbbbbbbb",
        "20200103-000000-cccccccc",
    ):
        (downloads_dir / name).mkdir()

    _prune_old_jobs(keep=2)

    remaining = sorted(p.name for p in downloads_dir.iterdir())
    assert remaining == ["20200102-000000-bbbbbbbb", "20200103-000000-cccccccc"]


def test_retention_never_touches_unrecognised_entries(downloads_dir):
    (downloads_dir / "20200101-000000-aaaaaaaa").mkdir()
    users_own = downloads_dir / "my-saved-audio"
    users_own.mkdir()
    stray = downloads_dir / "loose-file.mp3"
    stray.write_bytes(b"audio")

    _prune_old_jobs(keep=0)

    assert users_own.is_dir(), "pruned a directory it did not create"
    assert stray.is_file()
    assert not (downloads_dir / "20200101-000000-aaaaaaaa").exists()
