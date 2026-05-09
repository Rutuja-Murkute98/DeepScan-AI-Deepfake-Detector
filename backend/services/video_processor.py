"""
Video processing service — frame extraction, face-aware sampling.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Extract and pre-process frames from video files."""

    def __init__(self, max_frames: int = 32):
        self.max_frames = max_frames

    def extract_frames(
        self,
        video_bytes: bytes,
        max_frames: Optional[int] = None,
    ) -> Tuple[List[Image.Image], dict]:
        """
        Extract uniformly-spaced frames from a video.

        Returns:
            (list_of_PIL_frames, video_metadata_dict)
        """
        max_frames = max_frames or self.max_frames

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video file.")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            if total_frames <= 0:
                raise ValueError("Video has no frames.")

            # Smart frame sampling: prefer frames from different scenes
            n_sample = min(max_frames, total_frames)
            indices = np.linspace(0, total_frames - 1, n_sample, dtype=int).tolist()

            frames: List[Image.Image] = []
            timestamps: List[float] = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
                    timestamps.append(round(idx / fps, 3) if fps > 0 else 0.0)

            cap.release()

            if not frames:
                raise ValueError("Could not read any frames from video.")

            metadata = {
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration_sec": round(duration, 2),
                "resolution": f"{width}x{height}",
                "sampled_frames": len(frames),
                "timestamps": timestamps,
            }

            logger.info(
                f"Extracted {len(frames)} frames from video "
                f"({duration:.1f}s, {fps:.0f}fps, {width}x{height})"
            )
            return frames, metadata

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @staticmethod
    def get_scene_change_frames(
        video_bytes: bytes, threshold: float = 30.0, max_frames: int = 16
    ) -> List[Image.Image]:
        """
        Extract frames at scene change boundaries (content-aware sampling).
        Useful for getting diverse frames from long videos.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return []

            frames = []
            prev_gray = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    score = float(np.mean(diff))
                    if score > threshold:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(rgb))
                        if len(frames) >= max_frames:
                            break
                else:
                    # Always include first frame
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))

                prev_gray = gray

            cap.release()
            return frames

        finally:
            Path(tmp_path).unlink(missing_ok=True)
