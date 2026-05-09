"""
API route handlers for the deepfake detection service.
"""

from __future__ import annotations

import io
import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from PIL import Image

from backend.core.config import settings
from backend.core.security import rate_limiter, validate_upload
from backend.models.schemas import (
    BatchPredictionResult,
    HealthResponse,
    ModelInfoResponse,
    PredictionResult,
    VideoPredictionResult,
)
from backend.services.detector import DeepfakeDetectorService

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-initialised detector singleton
_detector: DeepfakeDetectorService | None = None


def get_detector() -> DeepfakeDetectorService:
    global _detector
    if _detector is None:
        logger.info("Initialising DeepfakeDetectorService …")
        _detector = DeepfakeDetectorService()
        logger.info("Detector ready.")
    return _detector


def _bytes_to_pil(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Invalid image data: {exc}")


# ─── Health & Info ────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Server health and readiness check."""
    detector = get_detector()
    return HealthResponse(
        status="ok",
        device=str(detector.device),
        model_loaded=detector.is_loaded,
        model_type=detector.model_type,
    )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    """Model metadata and configuration."""
    info = get_detector().get_info()
    return ModelInfoResponse(**info)


# ─── Image Prediction ────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResult, tags=["Detection"])
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyse"),
    heatmap: bool = Query(False, description="Generate Grad-CAM heatmap"),
    threshold: float = Query(settings.THRESHOLD, ge=0.0, le=1.0),
    face_detection: bool = Query(True, description="Use face detection"),
):
    """
    Analyse a single image for deepfake / AI-generated content.

    Supports JPG, PNG, WebP, BMP formats up to 200 MB.
    """
    rate_limiter.check(request)

    data = await validate_upload(file, allowed="image")
    image = _bytes_to_pil(data)
    detector = get_detector()

    result = detector.predict_image(
        image,
        threshold=threshold,
        use_face_detection=face_detection,
        generate_heatmap=heatmap,
    )

    return PredictionResult(**result)


# ─── Batch Prediction ────────────────────────────────────────────────────────

@router.post("/predict/batch", response_model=BatchPredictionResult, tags=["Detection"])
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="Image files (max 20)"),
    threshold: float = Query(settings.THRESHOLD, ge=0.0, le=1.0),
):
    """Analyse multiple images in one request (max 20)."""
    rate_limiter.check(request)

    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(400, f"Max {settings.MAX_BATCH_SIZE} files per batch.")

    images = []
    for f in files:
        data = await validate_upload(f, allowed="image")
        images.append(_bytes_to_pil(data))

    detector = get_detector()
    results, total_latency = detector.predict_batch(images, threshold=threshold)

    fake_count = sum(1 for r in results if r["label"] == "fake")
    summary = {
        "total": len(results),
        "fake_count": fake_count,
        "real_count": len(results) - fake_count,
        "fake_ratio": round(fake_count / max(len(results), 1), 4),
    }

    return BatchPredictionResult(
        results=[PredictionResult(**r) for r in results],
        total_latency_ms=total_latency,
        summary=summary,
    )


# ─── Video Prediction ────────────────────────────────────────────────────────

@router.post("/predict/video", response_model=VideoPredictionResult, tags=["Detection"])
async def predict_video(
    request: Request,
    file: UploadFile = File(..., description="Video file to analyse"),
    threshold: float = Query(settings.VIDEO_FRAME_THRESHOLD, ge=0.0, le=1.0),
    max_frames: int = Query(settings.MAX_VIDEO_FRAMES, ge=1, le=64),
):
    """
    Analyse a video for deepfake content.

    Extracts frames uniformly, detects faces in each frame,
    and aggregates predictions into an overall verdict.
    """
    rate_limiter.check(request)

    data = await validate_upload(file, allowed="video")
    detector = get_detector()

    result = detector.predict_video(
        data, threshold=threshold, max_frames=max_frames
    )

    return VideoPredictionResult(**result)
