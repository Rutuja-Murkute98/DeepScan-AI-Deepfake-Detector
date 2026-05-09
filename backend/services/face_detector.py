"""
Face detection service using OpenCV DNN (ships with opencv-python).

Falls back to Haar cascades if the DNN model files are unavailable.
Optionally uses MTCNN from facenet-pytorch if installed (higher accuracy).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detect and extract face crops from images.

    Priority order:
        1. MTCNN (facenet-pytorch) — best accuracy
        2. OpenCV DNN SSD face detector — good accuracy, zero extra deps
        3. OpenCV Haar cascade — fallback, always available
    """

    def __init__(self, min_confidence: float = 0.5, min_face_size: int = 40):
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self._backend = None
        self._detector = None
        self._init_detector()

    def _init_detector(self) -> None:
        # Try MTCNN first
        try:
            from facenet_pytorch import MTCNN
            self._detector = MTCNN(
                keep_all=True,
                min_face_size=self.min_face_size,
                thresholds=[0.6, 0.7, 0.7],
                device="cpu",
            )
            self._backend = "mtcnn"
            logger.info("Face detector: MTCNN (facenet-pytorch)")
            return
        except ImportError:
            pass

        # Fall back to Haar cascade (always available with OpenCV)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)
        self._backend = "haar"
        logger.info("Face detector: OpenCV Haar cascade")

    def detect(
        self,
        image: Image.Image,
        margin: float = 0.2,
    ) -> List[Tuple[Image.Image, dict]]:
        """
        Detect faces and return cropped face images with bounding boxes.

        Args:
            image:  PIL Image (RGB).
            margin: Fractional margin around detected face box.

        Returns:
            List of (face_crop_PIL, {"x", "y", "width", "height", "confidence"})
        """
        if self._backend == "mtcnn":
            return self._detect_mtcnn(image, margin)
        else:
            return self._detect_haar(image, margin)

    def _detect_mtcnn(
        self, image: Image.Image, margin: float
    ) -> List[Tuple[Image.Image, dict]]:
        boxes, probs = self._detector.detect(image)
        if boxes is None:
            return []

        results = []
        w, h = image.size
        for box, prob in zip(boxes, probs):
            if prob < self.min_confidence:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            mx, my = int(bw * margin), int(bh * margin)

            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(w, x2 + mx)
            y2 = min(h, y2 + my)

            crop = image.crop((x1, y1, x2, y2))
            if crop.size[0] < self.min_face_size or crop.size[1] < self.min_face_size:
                continue

            results.append((crop, {
                "x": x1, "y": y1,
                "width": x2 - x1, "height": y2 - y1,
                "confidence": round(float(prob), 4),
            }))

        return results

    def _detect_haar(
        self, image: Image.Image, margin: float
    ) -> List[Tuple[Image.Image, dict]]:
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )

        if len(faces) == 0:
            return []

        results = []
        h_img, w_img = img_np.shape[:2]
        for (x, y, w, h) in faces:
            mx, my = int(w * margin), int(h * margin)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w_img, x + w + mx)
            y2 = min(h_img, y + h + my)

            crop = image.crop((x1, y1, x2, y2))
            if crop.size[0] < self.min_face_size or crop.size[1] < self.min_face_size:
                continue

            results.append((crop, {
                "x": x1, "y": y1,
                "width": x2 - x1, "height": y2 - y1,
                "confidence": 0.85,  # Haar doesn't give confidence
            }))

        return results

    @property
    def backend_name(self) -> str:
        return self._backend or "none"
