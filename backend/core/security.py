"""
Security utilities — file validation, rate limiting, input sanitization.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request, UploadFile

from backend.core.config import settings


# ─── File-type magic bytes ────────────────────────────────────────────────────

_MAGIC = {
    # Images
    b"\xff\xd8\xff":            "image/jpeg",
    b"\x89PNG\r\n\x1a\n":      "image/png",
    b"RIFF":                    "image/webp",   # RIFF....WEBP
    b"BM":                      "image/bmp",
    b"GIF87a":                  "image/gif",
    b"GIF89a":                  "image/gif",
    # Video
    b"\x00\x00\x00\x1cftyp":    "video/mp4",
    b"\x00\x00\x00\x18ftyp":    "video/mp4",
    b"\x00\x00\x00\x20ftyp":    "video/mp4",
    b"\x1aE\xdf\xa3":          "video/webm",
}


def _detect_mime(header: bytes) -> Optional[str]:
    """Sniff MIME type from leading bytes."""
    for magic, mime in _MAGIC.items():
        if header[: len(magic)] == magic:
            return mime
    # Relaxed MP4 check (ftyp anywhere in first 12 bytes)
    if b"ftyp" in header[:12]:
        return "video/mp4"
    return None


async def validate_upload(file: UploadFile, allowed: str = "both") -> bytes:
    """
    Read and validate an uploaded file.

    Args:
        file:    FastAPI UploadFile
        allowed: 'image', 'video', or 'both'

    Returns:
        Raw bytes of the file.

    Raises:
        HTTPException on invalid / oversized / malicious uploads.
    """
    data = await file.read()

    if len(data) == 0:
        raise HTTPException(400, "Empty file uploaded.")

    if len(data) > settings.max_file_bytes:
        raise HTTPException(
            413,
            f"File too large ({len(data) / 1024 / 1024:.1f} MB). "
            f"Max allowed: {settings.MAX_FILE_SIZE_MB} MB.",
        )

    mime = _detect_mime(data[:16])
    if mime is None:
        raise HTTPException(400, "Unrecognised file format. Upload an image or video.")

    if allowed == "image" and not mime.startswith("image"):
        raise HTTPException(400, f"Expected an image file, got {mime}.")
    if allowed == "video" and not mime.startswith("video"):
        raise HTTPException(400, f"Expected a video file, got {mime}.")
    if allowed == "both" and not (mime.startswith("image") or mime.startswith("video")):
        raise HTTPException(400, f"Unsupported file type: {mime}.")

    return data


# ─── Simple in-memory rate limiter ────────────────────────────────────────────

class RateLimiter:
    """Token-bucket rate limiter keyed by client IP."""

    def __init__(self, rpm: int = 60):
        self.rpm = rpm
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def check(self, request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._buckets[ip]
        # Purge expired entries
        self._buckets[ip] = [t for t in window if now - t < 60]
        if len(self._buckets[ip]) >= self.rpm:
            raise HTTPException(429, "Rate limit exceeded. Try again in a minute.")
        self._buckets[ip].append(now)


rate_limiter = RateLimiter(settings.RATE_LIMIT_RPM)


def file_hash(data: bytes) -> str:
    """SHA-256 hex digest of file contents."""
    return hashlib.sha256(data).hexdigest()[:16]
