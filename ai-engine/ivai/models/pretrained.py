"""torchvision ResNet50 / MobileNetV2 for NEU-CLS (Phase 3, docs/phase3_protocol.md §5-6).

Design rules (frozen protocol):
* The torchvision architecture is kept unchanged; only the **last linear layer** is
  replaced by a 6-output layer (``fc`` for ResNet50, ``classifier[1]`` for MobileNetV2).
* ``AdaptiveAvgPool2d`` is kept: with a 1x1 output PyTorch computes a plain ``mean``,
  which is deterministic on CUDA (verified in Phase 3 step 1, protocol §13.2).
* Pretrained weights = ImageNet ``IMAGENET1K_V1`` (chosen a priori).
* **No implicit download.** Weights are read from the local torch hub cache only;
  if the file is missing, ``WeightsNotAvailableError`` is raised. Downloading is an
  explicit, separate action (``allow_download=True``, used only by
  ``scripts/download_weights.py``).
* Same random-number consumption with or without pretraining: the model is always
  built with ``weights=None`` (random init), then the ImageNet weights are loaded,
  then the new head is created. The pretrained and from-scratch variants therefore
  differ only by the loaded weights.
* Linear probe (``freeze_backbone=True``): every parameter outside the head has
  ``requires_grad=False`` and the backbone is kept in eval mode even when
  ``model.train()`` is called, so BatchNorm running statistics stay frozen.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, ResNet50_Weights, mobilenet_v2, resnet50
from torchvision.models._api import WeightsEnum

WEIGHTS: dict[str, WeightsEnum] = {
    "resnet50": ResNet50_Weights.IMAGENET1K_V1,
    "mobilenetv2": MobileNet_V2_Weights.IMAGENET1K_V1,
}
# Name of the head module (the only part trained in a linear probe, and the
# parameters that get the "head" learning rate in fine-tuning).
HEAD_MODULE = {"resnet50": "fc", "mobilenetv2": "classifier"}


class WeightsNotAvailableError(FileNotFoundError):
    """The ImageNet weights are not in the local cache and downloading was not allowed."""


def weights_cache_path(weights: WeightsEnum) -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name


def load_imagenet_state_dict(weights: WeightsEnum, allow_download: bool = False) -> dict:
    path = weights_cache_path(weights)
    if not path.exists() and not allow_download:
        raise WeightsNotAvailableError(
            f"{path.name} is not in the local cache ({path.parent}). Downloading is disabled: "
            "run scripts/download_weights.py explicitly once it has been authorised."
        )
    # check_hash=True: torchvision verifies the SHA-256 prefix embedded in the file name.
    return weights.get_state_dict(progress=False, check_hash=True)


_PROBE_CLASSES: dict[type, type] = {}


def _probe_train(self, mode: bool = True):
    nn.Module.train(self, mode)
    for name, child in self.named_children():
        if name != self.probe_head:  # backbone BatchNorm statistics and dropout stay frozen
            child.train(False)
    return self


def _freeze_backbone(model: nn.Module, head_name: str) -> None:
    head_params = {id(p) for p in getattr(model, head_name).parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_params
    # Subclass (not a monkey-patched bound method) so that copy.deepcopy keeps the behaviour.
    base = type(model)
    if base not in _PROBE_CLASSES:
        _PROBE_CLASSES[base] = type(f"LinearProbe{base.__name__}", (base,), {"train": _probe_train})
    model.__class__ = _PROBE_CLASSES[base]
    model.probe_head = head_name
    model.train(model.training)


def build_torchvision_model(name: str, num_classes: int = 6, pretrained: bool = True,
                            freeze_backbone: bool = False, allow_download: bool = False) -> nn.Module:
    if name not in WEIGHTS:
        raise ValueError(f"Unknown torchvision model {name!r}; expected one of {sorted(WEIGHTS)}")
    if freeze_backbone and not pretrained:
        raise ValueError("A linear probe (freeze_backbone=True) only makes sense on pretrained weights.")
    model = resnet50(weights=None) if name == "resnet50" else mobilenet_v2(weights=None)
    if pretrained:
        model.load_state_dict(load_imagenet_state_dict(WEIGHTS[name], allow_download))
    if name == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    if freeze_backbone:
        _freeze_backbone(model, HEAD_MODULE[name])
    return model


def head_parameter_names(model: nn.Module, name: str) -> set[str]:
    prefix = HEAD_MODULE[name] + "."
    return {n for n, _ in model.named_parameters() if n.startswith(prefix)}
