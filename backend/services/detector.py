"""
Main deepfake detection orchestrator.

Detection pipeline:
    1. Try to load a locally trained EfficientNet-B4 model (models/best_model.pt)
    2. If unavailable, download and use a pre-trained HuggingFace model
       (dima806/deepfake_vs_real_image_detection — ViT fine-tuned on 140k images)
    3. For each uploaded image:
       a. Optionally detect and extract faces
       b. Classify each face crop (or full image if no faces)
       c. Generate Grad-CAM heatmap if requested
    4. For videos: extract frames → detect faces → classify → aggregate
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from backend.core.config import settings
from backend.services.face_detector import FaceDetector
from backend.services.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


# ─── EfficientNet model architecture (for locally trained models) ─────────────

class EfficientNetDetector(nn.Module):
    """EfficientNet-B4 binary classifier: Real (0) vs Fake (1)."""

    def __init__(self, backbone: str = "efficientnet_b4", dropout: float = 0.4):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            backbone, pretrained=False, num_classes=0, global_pool="avg"
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_last_conv_layer(self) -> nn.Module:
        last_conv = None
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("No Conv2d found")
        return last_conv


# ─── Main Detector Service ────────────────────────────────────────────────────

class DeepfakeDetectorService:
    """
    High-level detection service.

    Supports two backends:
        • "huggingface" — Pre-trained ViT from HuggingFace (default, zero-config)
        • "local"       — Custom-trained EfficientNet-B4 (from models/best_model.pt)
    """

    def __init__(self):
        self.device = self._resolve_device()
        self.model = None
        self.processor = None
        self.hf_detectors: list[tuple[str, Any, Any]] = []
        self.transform = None
        self.model_type: str = "none"
        self.model_name: str = "none"

        self.face_detector = FaceDetector()
        self.video_processor = VideoProcessor(max_frames=settings.MAX_VIDEO_FRAMES)

        self._load_model()

    def _resolve_device(self) -> torch.device:
        if settings.DEVICE == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(settings.DEVICE)

    def _load_model(self) -> None:
        """Load the best available model."""
        local_path = Path(settings.LOCAL_MODEL_PATH)

        # Priority 1: Local trained model
        if local_path.exists():
            try:
                self._load_local_model(local_path)
                return
            except Exception as e:
                logger.warning(f"Failed to load local model: {e}")

        # Priority 2: HuggingFace pre-trained model
        try:
            self._load_hf_model()
            return
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise RuntimeError(
                "No detection model available. Either:\n"
                "  1. Place a trained model at models/best_model.pt\n"
                "  2. Install transformers: pip install transformers\n"
                "  3. Ensure internet access for HuggingFace download"
            )

    def _load_hf_model(self) -> None:
        """Load pre-trained model from HuggingFace."""
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        model_names = [
            name.strip()
            for name in settings.HF_MODEL_NAME.split(",")
            if name.strip()
        ]
        errors = []

        for model_name in model_names:
            try:
                logger.info(f"Loading HuggingFace model: {model_name}")
                processor = AutoImageProcessor.from_pretrained(model_name)
                model = AutoModelForImageClassification.from_pretrained(model_name)
                model.to(self.device).eval()
                self.hf_detectors.append((model_name, processor, model))
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
                logger.warning(f"Skipping HuggingFace model {model_name}: {exc}")

        if not self.hf_detectors:
            raise RuntimeError("; ".join(errors) or "No HuggingFace models configured")

        first_model_name, self.processor, self.model = self.hf_detectors[0]
        self.model_type = "huggingface"
        self.model_name = (
            first_model_name
            if len(self.hf_detectors) == 1
            else "ensemble: " + ", ".join(name for name, _, _ in self.hf_detectors)
        )

        logger.info(
            f"Loaded {len(self.hf_detectors)} HuggingFace model(s) on {self.device}"
        )

    def _load_local_model(self, path: Path) -> None:
        """Load a locally trained EfficientNet model."""
        logger.info(f"Loading local model from {path}")

        self.model = EfficientNetDetector(
            backbone=settings.BACKBONE, dropout=0.4
        )

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))

        # Remap legacy keys
        remapped = {}
        for key, val in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            remapped[new_key] = val

        self.model.load_state_dict(remapped, strict=False)
        self.model.to(self.device).eval()
        self.model_type = "local"
        self.model_name = f"EfficientNet-B4 ({path.name})"

        # Standard ImageNet transform for local model
        self.transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        logger.info(f"Local model loaded on {self.device}")

    # ─── Image Prediction ─────────────────────────────────────────────────

    def predict_image(
        self,
        image: Image.Image,
        threshold: float = None,
        use_face_detection: bool = True,
        generate_heatmap: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict whether an image is real or AI-generated/deepfake.

        Pipeline:
            1. Detect faces (optional)
            2. Classify face crops OR full image
            3. Generate heatmap (optional)
            4. Return comprehensive result

        Returns dict matching PredictionResult schema.
        """
        threshold = threshold or settings.THRESHOLD
        t0 = time.perf_counter()
        full_real, full_fake = self._classify(image)

        # Detect faces
        face_results = []
        face_regions = []

        if use_face_detection:
            faces = self.face_detector.detect(image)
            if faces:
                for face_crop, bbox in faces:
                    prob_real, prob_fake = self._classify(face_crop)
                    face_results.append((prob_real, prob_fake))
                    face_regions.append(bbox)

        # If faces found, aggregate face predictions
        if face_results:
            # Weighted average by face area (larger faces = higher weight)
            areas = [r["width"] * r["height"] for r in face_regions]
            total_area = sum(areas) or 1
            weights = [a / total_area for a in areas]

            face_real = sum(w * r[0] for w, r in zip(weights, face_results))
            face_fake = sum(w * r[1] for w, r in zip(weights, face_results))
            avg_real = (0.6 * full_real) + (0.4 * face_real)
            avg_fake = (0.6 * full_fake) + (0.4 * face_fake)
        else:
            # No faces detected — classify the full image
            avg_real, avg_fake = full_real, full_fake

        raw_real, raw_fake = avg_real, avg_fake
        label = "fake" if raw_fake >= threshold else "real"
        avg_real, avg_fake = self._calibrate_display_probs(raw_fake, threshold)
        confidence = round(avg_fake if label == "fake" else avg_real, 4)

        # Heatmap
        heatmap_b64 = None
        if generate_heatmap and self.model_type == "local":
            try:
                from backend.services.explainability import generate_heatmap_b64
                heatmap_b64 = generate_heatmap_b64(
                    self.model,
                    self.model.get_last_conv_layer(),
                    image,
                    self.transform(image),
                    self.device,
                    class_idx=1,
                )
            except Exception as e:
                logger.warning(f"Heatmap generation failed: {e}")

        latency = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "label": label,
            "confidence": confidence,
            "prob_real": round(avg_real, 4),
            "prob_fake": round(avg_fake, 4),
            "latency_ms": latency,
            "faces_detected": len(face_regions),
            "face_regions": face_regions,
            "heatmap_b64": heatmap_b64,
            "analysis_details": {
                "model_type": self.model_type,
                "model_name": self.model_name,
                "threshold": threshold,
                "raw_prob_real": round(raw_real, 4),
                "raw_prob_fake": round(raw_fake, 4),
                "face_detection_used": use_face_detection,
                "face_backend": self.face_detector.backend_name,
            },
        }

    @staticmethod
    def _calibrate_display_probs(prob_fake: float, threshold: float) -> Tuple[float, float]:
        """
        Convert a threshold-based fake score into UI-friendly verdict scores.

        The classifier decision is threshold based, so a raw fake score of 0.46
        with a 0.40 threshold is a fake verdict. Displaying raw probabilities
        would show real > fake, which is confusing. This maps scores around the
        threshold to 50/50 and makes the winning verdict visually higher.
        """
        threshold = min(max(threshold, 0.01), 0.99)
        if prob_fake >= threshold:
            fake_display = 0.5 + 0.5 * ((prob_fake - threshold) / (1.0 - threshold))
        else:
            fake_display = 0.5 * (prob_fake / threshold)

        fake_display = min(max(fake_display, 0.0), 1.0)
        return 1.0 - fake_display, fake_display

    def _classify(self, image: Image.Image) -> Tuple[float, float]:
        """
        Classify a single image. Returns (prob_real, prob_fake).
        """
        if self.model_type == "huggingface":
            return self._classify_hf(image)
        else:
            return self._classify_local(image)

    def _classify_hf(self, image: Image.Image) -> Tuple[float, float]:
        """Classify using the HuggingFace ensemble."""
        predictions = [
            self._classify_hf_single(image, processor, model)
            for _, processor, model in self.hf_detectors
        ]
        fake_scores = [prob_fake for _, prob_fake in predictions]

        # Balance false positives and false negatives:
        # - one over-sensitive model alone is not enough to call a real photo fake
        # - two models with moderate/strong fake evidence should flag AI content
        # - one extreme fake score can flag only when the ensemble mean supports it
        median_fake = float(np.median(fake_scores))
        mean_fake = float(np.mean(fake_scores))
        max_fake = max(fake_scores)
        medium_fake_votes = sum(score >= 0.45 for score in fake_scores)
        high_fake_votes = sum(score >= 0.60 for score in fake_scores)

        if high_fake_votes >= 2:
            fake_prob = max(0.65, 0.55 * median_fake + 0.45 * max_fake)
        elif medium_fake_votes >= 2:
            fake_prob = max(0.56, 0.65 * mean_fake + 0.35 * median_fake)
        elif max_fake >= 0.90 and mean_fake >= 0.35:
            fake_prob = max(0.56, 0.65 * mean_fake + 0.35 * max_fake)
        else:
            fake_prob = min(0.49, 0.7 * mean_fake + 0.3 * median_fake)

        real_prob = 1.0 - fake_prob
        return real_prob, fake_prob

    def _classify_hf_single(
        self,
        image: Image.Image,
        processor: Any,
        model: Any,
    ) -> Tuple[float, float]:
        """Classify using one HuggingFace model."""
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        # Map common detector labels robustly. Different HuggingFace models use
        # labels such as Real/Fake, human/AI-generated, or synthetic/authentic.
        id2label = model.config.id2label
        prob_dict = {}
        for idx, prob in enumerate(probs):
            label_name = id2label.get(idx, f"class_{idx}").lower()
            prob_dict[label_name] = float(prob)

        real_terms = ("real", "human", "authentic", "natural", "photograph")
        fake_terms = (
            "fake",
            "ai",
            "generated",
            "synthetic",
            "deepfake",
            "manipulated",
            "artificial",
        )

        prob_real = sum(
            score
            for label, score in prob_dict.items()
            if any(term in label for term in real_terms)
        )
        prob_fake = sum(
            score
            for label, score in prob_dict.items()
            if any(term in label for term in fake_terms)
        )

        # If labels don't match expected, use positional
        if prob_real == 0.0 and prob_fake == 0.0:
            prob_real = float(probs[0])
            prob_fake = float(probs[1])

        return prob_real, prob_fake

    @torch.no_grad()
    def _classify_local(self, image: Image.Image) -> Tuple[float, float]:
        """Classify using local EfficientNet model."""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        return float(probs[0]), float(probs[1])

    # ─── Video Prediction ─────────────────────────────────────────────────

    def predict_video(
        self,
        video_bytes: bytes,
        threshold: float = None,
        max_frames: int = None,
    ) -> Dict[str, Any]:
        """
        Predict whether a video contains deepfake content.

        Pipeline:
            1. Extract frames uniformly across the video
            2. For each frame: detect faces → classify
            3. Aggregate frame-level predictions into overall verdict
        """
        frame_threshold = threshold or settings.VIDEO_FRAME_THRESHOLD
        max_frames = min(max_frames or settings.MAX_VIDEO_FRAMES, settings.MAX_VIDEO_FRAMES)

        t0 = time.perf_counter()

        frames, metadata = self.video_processor.extract_frames(video_bytes, max_frames)
        timestamps = metadata.get("timestamps", [0.0] * len(frames))

        frame_results = []
        all_fake_probs = []

        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            prob_real, prob_fake = self._classify(frame)

            # Also try face-based detection for face frames
            faces = self.face_detector.detect(frame)
            n_faces = len(faces)

            if faces:
                face_probs = []
                for face_crop, _ in faces:
                    _, fp = self._classify(face_crop)
                    face_probs.append(fp)
                # Use max face-level fake probability
                face_fake = max(face_probs)
                # Ensemble: average of full-frame and face-level prediction
                prob_fake = 0.4 * prob_fake + 0.6 * face_fake
                prob_real = 1.0 - prob_fake

            all_fake_probs.append(prob_fake)
            frame_results.append({
                "frame_index": i,
                "timestamp_sec": ts,
                "label": "fake" if prob_fake >= frame_threshold else "real",
                "prob_fake": round(prob_fake, 4),
                "faces_detected": n_faces,
            })

        # Aggregate: use trimmed mean (robust to outlier frames)
        sorted_probs = sorted(all_fake_probs)
        if len(sorted_probs) > 4:
            # Trim top/bottom 10%
            trim = max(1, len(sorted_probs) // 10)
            trimmed = sorted_probs[trim:-trim]
            overall_fake = float(np.mean(trimmed))
        else:
            overall_fake = float(np.mean(sorted_probs))

        fake_count = sum(1 for r in frame_results if r["label"] == "fake")
        fake_frame_ratio = fake_count / max(len(frame_results), 1)
        video_fake_score = max(overall_fake, fake_frame_ratio)
        overall_label = (
            "fake"
            if overall_fake >= settings.VIDEO_OVERALL_THRESHOLD
            or fake_frame_ratio >= settings.VIDEO_FAKE_FRAME_RATIO_THRESHOLD
            else "real"
        )

        latency = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "overall_label": overall_label,
            "overall_confidence": round(
                video_fake_score if overall_label == "fake" else 1 - video_fake_score, 4
            ),
            "overall_prob_fake": round(video_fake_score, 4),
            "frames_analyzed": len(frame_results),
            "fake_frame_count": fake_count,
            "real_frame_count": len(frame_results) - fake_count,
            "frame_results": frame_results,
            "latency_ms": latency,
            "duration_sec": metadata.get("duration_sec"),
            "fps": metadata.get("fps"),
            "analysis_details": {
                "model_type": self.model_type,
                "model_name": self.model_name,
                "frame_threshold": frame_threshold,
                "overall_threshold": settings.VIDEO_OVERALL_THRESHOLD,
                "fake_frame_ratio_threshold": settings.VIDEO_FAKE_FRAME_RATIO_THRESHOLD,
                "mean_fake_probability": round(overall_fake, 4),
                "fake_frame_ratio": round(fake_frame_ratio, 4),
                "video_resolution": metadata.get("resolution"),
                "total_video_frames": metadata.get("total_frames"),
            },
        }

    # ─── Batch Prediction ─────────────────────────────────────────────────

    def predict_batch(
        self,
        images: List[Image.Image],
        threshold: float = None,
    ) -> Tuple[List[Dict], float]:
        """Predict a batch of images. Returns (results_list, total_latency_ms)."""
        threshold = threshold or settings.THRESHOLD
        t0 = time.perf_counter()

        results = []
        for img in images:
            result = self.predict_image(
                img, threshold=threshold, use_face_detection=True, generate_heatmap=False
            )
            results.append(result)

        total_latency = round((time.perf_counter() - t0) * 1000, 2)
        return results, total_latency

    # ─── Model Info ───────────────────────────────────────────────────────

    def get_info(self) -> Dict[str, Any]:
        num_params = None
        if self.model is not None:
            try:
                num_params = sum(p.numel() for p in self.model.parameters())
            except Exception:
                pass

        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "ensemble_models": [name for name, _, _ in self.hf_detectors],
            "device": str(self.device),
            "num_params": num_params,
            "image_size": settings.IMAGE_SIZE,
            "threshold": settings.THRESHOLD,
        }

    @property
    def is_loaded(self) -> bool:
        return self.model is not None
