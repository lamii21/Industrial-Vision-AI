"""E5 (complementary): deterministic perturbations of the validation images
(docs/phase3_protocol.md §9.6). Descriptive only, never used for selection.

Levels are outside the training augmentation range (brightness/contrast ±20 %).
Perturbations act on intensities in [0, 1] **before** normalisation; JPEG acts on the
uint8 image. The noise of image ``i`` uses the seed ``NOISE_SEED + i``.
"""

from __future__ import annotations

import io
import math

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as F

NOISE_SEED = 0
PERTURBATIONS: tuple[tuple[str, float], ...] = (
    ("brightness", 0.6), ("brightness", 1.4),
    ("contrast", 0.6), ("contrast", 1.4),
    ("gaussian_noise", 0.02), ("gaussian_noise", 0.05),
    ("gaussian_blur", 1.0), ("gaussian_blur", 2.0),
    ("jpeg", 50), ("jpeg", 20),
)


def perturb(image_uint8: torch.Tensor, kind: str, level: float, index: int) -> torch.Tensor:
    """uint8 (1,H,W) -> float (1,H,W) in [0, 1], perturbed."""
    if image_uint8.dtype != torch.uint8:
        raise TypeError("expects the raw uint8 image")
    if kind == "jpeg":  # Pillow (torchvision's JPEG encoder is deprecated since 0.29)
        buf = io.BytesIO()
        Image.fromarray(image_uint8[0].numpy(), mode="L").save(buf, format="JPEG", quality=int(level))
        decoded = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("L"))
        return torch.from_numpy(decoded).float().unsqueeze(0) / 255.0
    x = image_uint8.float() / 255.0
    if kind == "brightness":
        return F.adjust_brightness(x, level).clamp(0, 1)
    if kind == "contrast":
        return F.adjust_contrast(x, level).clamp(0, 1)
    if kind == "gaussian_noise":
        g = torch.Generator().manual_seed(NOISE_SEED + int(index))
        return (x + level * torch.randn(x.shape, generator=g)).clamp(0, 1)
    if kind == "gaussian_blur":
        k = 2 * math.ceil(3 * level) + 1
        return F.gaussian_blur(x, kernel_size=[k, k], sigma=[level, level])
    raise ValueError(f"Unknown perturbation {kind!r}")


class PerturbedDataset(torch.utils.data.Dataset):
    """Wraps a dataset built with ``transform=None`` (uint8 images) and applies one
    perturbation followed by the evaluation transform of the model."""

    def __init__(self, raw_dataset, kind: str, level: float, eval_transform):
        self.raw, self.kind, self.level, self.tf = raw_dataset, kind, level, eval_transform

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, i: int):
        x, y = self.raw[i]
        if self.kind == "clean":  # exactly the standard evaluation pipeline
            return self.tf(x), y
        return self.tf(perturb(x, self.kind, self.level, i)), y
