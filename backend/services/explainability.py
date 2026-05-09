"""
Explainability service — Grad-CAM heatmap generation for model predictions.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN-based models.

    Highlights which regions of the input image influenced the model's decision,
    making it possible to see WHERE the model detected fake artifacts.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_inp, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        x: torch.Tensor,
        class_idx: int = 1,
        size: tuple = (224, 224),
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            x:         Input tensor (1×C×H×W).
            class_idx: Class to explain (1 = fake).
            size:      Output heatmap size (H, W).

        Returns:
            Normalised heatmap as numpy array (0–1).
        """
        self.model.eval()
        self.model.zero_grad()

        # Enable gradients for this forward pass
        x = x.requires_grad_(True)
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("Grad-CAM hooks did not fire.")

        weights = self._gradients[0].mean(dim=(1, 2))
        cam_map = (weights[:, None, None] * self._activations[0]).sum(0)
        cam_map = torch.relu(cam_map).cpu().numpy()

        # Normalise
        cam_map -= cam_map.min()
        if cam_map.max() > 0:
            cam_map /= cam_map.max()

        return cv2.resize(cam_map, (size[1], size[0]))

    @staticmethod
    def overlay(
        image_bgr: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay heatmap on BGR image."""
        heatmap_u8 = np.uint8(255 * heatmap)
        colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        # Resize colored to match image
        if colored.shape[:2] != image_bgr.shape[:2]:
            colored = cv2.resize(colored, (image_bgr.shape[1], image_bgr.shape[0]))
        return cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)


def generate_heatmap_b64(
    model: nn.Module,
    target_layer: nn.Module,
    image: Image.Image,
    input_tensor: torch.Tensor,
    device: torch.device,
    class_idx: int = 1,
) -> Optional[str]:
    """
    Generate a Grad-CAM heatmap overlay and return as base64 PNG.

    Returns None on failure (non-fatal).
    """
    try:
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(
            input_tensor.unsqueeze(0).to(device),
            class_idx=class_idx,
            size=(224, 224),
        )

        img_rgb = np.array(image.resize((224, 224)))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        overlay = cam.overlay(img_bgr, heatmap, alpha=0.4)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        buf = io.BytesIO()
        Image.fromarray(overlay_rgb).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.warning(f"Heatmap generation failed: {e}")
        return None
