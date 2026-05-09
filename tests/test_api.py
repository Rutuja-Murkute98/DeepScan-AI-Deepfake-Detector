"""
API tests for the deepfake detection service.

Usage:
    pytest tests/test_api.py -v
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app

client = TestClient(app)


def _make_test_image(size=(224, 224), color=(128, 128, 128)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


class TestHealth:
    def test_health_endpoint(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "device" in data

    def test_model_info(self):
        r = client.get("/model/info")
        assert r.status_code == 200
        data = r.json()
        assert "model_type" in data


class TestImagePrediction:
    def test_predict_image(self):
        buf = _make_test_image()
        r = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert r.status_code == 200
        data = r.json()
        assert data["label"] in ("real", "fake")
        assert 0 <= data["confidence"] <= 1
        assert 0 <= data["prob_real"] <= 1
        assert 0 <= data["prob_fake"] <= 1

    def test_predict_empty_file(self):
        r = client.post("/predict", files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")})
        assert r.status_code == 400


class TestVideoPrediction:
    def test_predict_invalid_video(self):
        r = client.post("/predict/video", files={"file": ("test.txt", io.BytesIO(b"not a video"), "video/mp4")})
        assert r.status_code == 400


class TestBatchPrediction:
    def test_predict_batch(self):
        files = [("files", (f"img{i}.jpg", _make_test_image(), "image/jpeg")) for i in range(3)]
        r = client.post("/predict/batch", files=files)
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) == 3
        assert "summary" in data
