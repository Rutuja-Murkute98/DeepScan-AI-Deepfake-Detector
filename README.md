
# 🛡️ DeepScan — AI Deepfake Detection System

Production-grade deepfake and AI-generated content detection system powered by deep learning.

## Features

- **Image Analysis** — Detect deepfakes and AI-generated images instantly
- **Video Analysis** — Frame-by-frame deepfake detection with aggregated verdict
- **Batch Processing** — Analyse up to 20 images in one request
- **Face Detection** — Automatic face extraction for focused analysis
- **Explainability** — Grad-CAM heatmaps showing suspicious regions
- **Pre-trained Model** — Works out-of-the-box with HuggingFace ViT model
- **Custom Training** — Full pipeline for training on FaceForensics++/DFDC
- **Professional UI** — Dark/light mode, drag-and-drop, real-time results
- **REST API** — FastAPI with auto-generated docs at `/docs`

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open the UI

Navigate to **http://localhost:8000** in your browser.

The pre-trained model downloads automatically on first run (~350 MB, cached).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check |
| GET | `/model/info` | Model metadata |
| POST | `/predict` | Single image analysis |
| POST | `/predict/batch` | Batch image analysis (≤20) |
| POST | `/predict/video` | Video analysis |
| GET | `/docs` | Interactive API docs |

## Project Structure

```
├── backend/
│   ├── app.py                 # FastAPI entry point
│   ├── api/routes.py          # API route handlers
│   ├── core/config.py         # Configuration
│   ├── core/security.py       # Validation & rate limiting
│   ├── models/schemas.py      # Pydantic schemas
│   └── services/
│       ├── detector.py        # Main detection orchestrator
│       ├── face_detector.py   # Face detection (MTCNN/Haar)
│       ├── video_processor.py # Frame extraction
│       └── explainability.py  # Grad-CAM heatmaps
├── frontend/
│   └── index.html             # Web UI
├── training/
│   ├── train.py               # Training pipeline
│   ├── dataset.py             # Dataset loader
│   └── evaluate.py            # Evaluation & metrics
├── tests/
│   └── test_api.py            # API tests
├── models/                    # Model weights
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Detection Pipeline

1. **Image Upload** → File validation & security checks
2. **Face Detection** → Extract face regions (MTCNN or Haar cascade)
3. **Classification** → Pre-trained ViT or custom EfficientNet-B4
4. **Aggregation** — Area-weighted face predictions (images) / trimmed mean (videos)
5. **Explainability** → Grad-CAM attention heatmaps (optional)

## Custom Training

To train on your own dataset (e.g., FaceForensics++):

```bash
# Prepare data in: data/train/real/, data/train/fake/, data/val/real/, data/val/fake/
python -m training.train --data_dir data --epochs 60 --backbone efficientnet_b4

# Evaluate
python -m training.evaluate --data_root data --checkpoint models/best_model.pt
```

## Docker

```bash
docker-compose up --build
```

## Configuration

Copy `.env.example` to `.env` and adjust settings. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `THRESHOLD` | `0.5` | Classification threshold |
| `MAX_VIDEO_FRAMES` | `32` | Max frames sampled from video |
| `RATE_LIMIT_RPM` | `60` | API rate limit per minute |

## Tech Stack

- **Backend**: FastAPI, PyTorch, timm, HuggingFace Transformers
- **Frontend**: Vanilla HTML/CSS/JS with Inter font
- **Models**: EfficientNet-B4 / ViT (pre-trained)
- **Face Detection**: MTCNN / OpenCV Haar cascade
- **Deployment**: Docker, uvicorn

## License

MIT
