"""
Dataset loader for deepfake detection training.

Expected layout:
    <root>/train/real/*.jpg   <root>/train/fake/*.jpg
    <root>/val/real/*.jpg     <root>/val/fake/*.jpg
    <root>/test/real/*.jpg    <root>/test/fake/*.jpg
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

VALID_EXT = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
CLASS_MAP = {"real": 0, "fake": 1}


class DeepfakeDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", transform: Optional[Callable] = None, max_samples: Optional[int] = None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        split_dir = Path(root) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        for cls, label in CLASS_MAP.items():
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            paths = sorted(p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT)
            if max_samples:
                paths = paths[:max_samples]
            self.samples.extend((p, label) for p in paths)
        if not self.samples:
            raise FileNotFoundError(f"No images in {split_dir}")
        real_n = sum(1 for _, l in self.samples if l == 0)
        fake_n = len(self.samples) - real_n
        print(f"[{split.upper():5s}] {len(self.samples)} samples | real={real_n} fake={fake_n}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        labels = [l for _, l in self.samples]
        counts = [labels.count(0), labels.count(1)]
        weights = [1.0 / max(c, 1) for c in counts]
        sample_weights = [weights[l] for l in labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
