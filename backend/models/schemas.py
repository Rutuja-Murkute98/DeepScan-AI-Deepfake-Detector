"""
Pydantic response / request schemas for the API.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ─── Health & Info ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    model_type: str
    version: str = "4.0.0"


class ModelInfoResponse(BaseModel):
    model_type: str
    model_name: str
    device: str
    num_params: Optional[int] = None
    image_size: int
    threshold: float


# ─── Prediction results ──────────────────────────────────────────────────────

class FaceRegion(BaseModel):
    """Bounding box for a detected face."""
    x: int
    y: int
    width: int
    height: int
    confidence: float


class PredictionResult(BaseModel):
    """Single image prediction result."""
    label: str = Field(..., description="'real' or 'fake'")
    confidence: float = Field(..., ge=0, le=1, description="Confidence 0–1")
    prob_real: float = Field(..., ge=0, le=1)
    prob_fake: float = Field(..., ge=0, le=1)
    latency_ms: float
    faces_detected: int = 0
    face_regions: List[FaceRegion] = []
    heatmap_b64: Optional[str] = None
    analysis_details: Optional[dict] = None


class BatchPredictionResult(BaseModel):
    results: List[PredictionResult]
    total_latency_ms: float
    summary: dict


# ─── Video ────────────────────────────────────────────────────────────────────

class FrameResult(BaseModel):
    frame_index: int
    timestamp_sec: float
    label: str
    prob_fake: float
    faces_detected: int = 0


class VideoPredictionResult(BaseModel):
    overall_label: str
    overall_confidence: float
    overall_prob_fake: float
    frames_analyzed: int
    fake_frame_count: int
    real_frame_count: int
    frame_results: List[FrameResult]
    latency_ms: float
    duration_sec: Optional[float] = None
    fps: Optional[float] = None
    analysis_details: Optional[dict] = None
