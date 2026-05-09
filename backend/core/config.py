"""
Application configuration — loads from environment variables with sane defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes")


@dataclass
class Settings:
    """Central application settings."""

    # ── Paths ─────────────────────────────────────────────────────────
    BASE_DIR: Path          = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    MODEL_DIR: Path         = field(default_factory=lambda: Path(_env("MODEL_DIR", "models")))
    UPLOAD_DIR: Path        = field(default_factory=lambda: Path(_env("UPLOAD_DIR", "uploads")))
    LOCAL_MODEL_PATH: Path  = field(default_factory=lambda: Path(_env("LOCAL_MODEL_PATH", "models/best_model.pt")))

    # ── Model ─────────────────────────────────────────────────────────
    HF_MODEL_NAME: str      = field(default_factory=lambda: _env(
        "HF_MODEL_NAME",
        "capcheck/ai-human-generated-image-detection,"
        "prithivMLmods/deepfake-detector-model-v1,"
        "king1oo1/deepfake-model,"
        "Smogy/SMOGY-Ai-images-detector",
    ))
    BACKBONE: str           = field(default_factory=lambda: _env("BACKBONE", "efficientnet_b4"))
    DEVICE: str             = field(default_factory=lambda: _env("DEVICE", "auto"))
    IMAGE_SIZE: int         = field(default_factory=lambda: _env_int("IMAGE_SIZE", 224))
    THRESHOLD: float        = field(default_factory=lambda: float(_env("THRESHOLD", "0.40")))
    VIDEO_FRAME_THRESHOLD: float = field(default_factory=lambda: float(_env("VIDEO_FRAME_THRESHOLD", "0.40")))
    VIDEO_OVERALL_THRESHOLD: float = field(default_factory=lambda: float(_env("VIDEO_OVERALL_THRESHOLD", "0.40")))
    VIDEO_FAKE_FRAME_RATIO_THRESHOLD: float = field(default_factory=lambda: float(_env("VIDEO_FAKE_FRAME_RATIO_THRESHOLD", "0.20")))

    # ── Server ────────────────────────────────────────────────────────
    HOST: str               = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    PORT: int               = field(default_factory=lambda: _env_int("PORT", 8000))
    DEBUG: bool             = field(default_factory=lambda: _env_bool("DEBUG", False))
    CORS_ORIGINS: str       = field(default_factory=lambda: _env("CORS_ORIGINS", "*"))

    # ── Limits ────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int   = field(default_factory=lambda: _env_int("MAX_FILE_SIZE_MB", 200))
    MAX_BATCH_SIZE: int     = field(default_factory=lambda: _env_int("MAX_BATCH_SIZE", 20))
    MAX_VIDEO_FRAMES: int   = field(default_factory=lambda: _env_int("MAX_VIDEO_FRAMES", 32))
    RATE_LIMIT_RPM: int     = field(default_factory=lambda: _env_int("RATE_LIMIT_RPM", 60))

    # ── Video ─────────────────────────────────────────────────────────
    VIDEO_EXTENSIONS: tuple = (".mp4", ".avi", ".mov", ".webm", ".mkv")
    IMAGE_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")

    def __post_init__(self):
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def max_file_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


settings = Settings()
