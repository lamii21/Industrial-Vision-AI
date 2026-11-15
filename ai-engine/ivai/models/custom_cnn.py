"""Custom CNN baseline for NEU-CLS (6 classes, 1 x 200 x 200 grayscale input).

Architecture (default channels 32-64-128-256)::

    input                      1 x 200 x 200
    block1  Conv3x3 -> BN -> ReLU -> MaxPool2     32 x 100 x 100
    block2  Conv3x3 -> BN -> ReLU -> MaxPool2     64 x  50 x  50
    block3  Conv3x3 -> BN -> ReLU -> MaxPool2    128 x  25 x  25
    block4  Conv3x3 -> BN -> ReLU -> MaxPool2    256 x  12 x  12   (floor(25/2))
    global average pooling                       256
    dropout(0.5)                                 256
    linear                                       6   (logits, no softmax)

Design notes
* ``bias=False`` in the convolutions: BatchNorm subtracts the mean and adds its own
  learnable shift (beta), so a conv bias would be redundant.
* Global average pooling instead of flattening: 256 parameters per class in the
  classifier instead of 256*12*12 = 36 864, which strongly limits overfitting on a
  1 260-image training set, and makes the model independent of the input size.
* GAP is written as ``x.mean(dim=(2, 3))``: mathematically identical to
  ``AdaptiveAvgPool2d(1)`` but its backward pass is deterministic on CUDA, which
  keeps ``torch.use_deterministic_algorithms(True)`` usable.
* No softmax in ``forward``: ``CrossEntropyLoss`` applies log-softmax internally
  (numerically stable). Probabilities are obtained at inference with ``softmax``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class CustomCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        channels: Sequence[int] = (32, 64, 128, 256),
        dropout: float = 0.5,
    ):
        super().__init__()
        blocks, prev = [], in_channels
        for ch in channels:
            blocks.append(ConvBlock(prev, ch))
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(prev, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        # He (Kaiming) initialisation, suited to ReLU activations.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Last convolutional feature map (N, 256, 12, 12): used later by Grad-CAM."""
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = x.mean(dim=(2, 3))  # global average pooling
        x = self.dropout(x)
        return self.classifier(x)  # logits
