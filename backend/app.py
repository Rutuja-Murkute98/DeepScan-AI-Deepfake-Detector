"""
FastAPI application entry point.

Run:
    uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.core.config import settings

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deepscan")

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DeepScan — AI Deepfake Detection API",
    description=(
        "Production-grade deepfake and AI-generated content detection. "
        "Supports images (JPG, PNG, WebP) and videos (MP4, AVI, MOV). "
        "Powered by EfficientNet-B4 / ViT with face detection pipeline."
    ),
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API Routes ───────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api")
# Also mount routes at root for backward compatibility
app.include_router(router)

# ─── Frontend (static) ───────────────────────────────────────────────────────

frontend_dir = Path(__file__).parent.parent / "frontend"

if frontend_dir.exists():
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def serve_frontend():
        index = frontend_dir / "index.html"
        if index.exists():
            return index.read_text(encoding="utf-8")
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


# ─── Startup event ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("  DeepScan — AI Deepfake Detection System")
    logger.info("=" * 60)
    logger.info(f"  Device : {settings.DEVICE}")
    logger.info(f"  Debug  : {settings.DEBUG}")
    logger.info(f"  Docs   : http://{settings.HOST}:{settings.PORT}/docs")
    logger.info("=" * 60)

    # Pre-load the model
    from backend.api.routes import get_detector
    try:
        get_detector()
        logger.info("Model loaded successfully at startup.")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.warning("Server running without model — /health will show model_loaded=false")
