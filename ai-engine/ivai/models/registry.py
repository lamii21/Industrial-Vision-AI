"""Single entry point to build any model of the project from its YAML ``model:`` section,
and to describe its input preprocessing and optimiser parameter groups.

Model section keys:
    name: custom_cnn | resnet50 | mobilenetv2
    custom_cnn  -> in_channels, num_classes, channels, dropout (Phase 2, unchanged)
    torchvision -> num_classes, pretrained (bool), freeze_backbone (bool)
"""

from __future__ import annotations

from torch import nn

from ivai.models.custom_cnn import CustomCNN
from ivai.models.pretrained import HEAD_MODULE, WEIGHTS, build_torchvision_model, head_parameter_names

TORCHVISION = set(WEIGHTS)
MODEL_NAMES = ("custom_cnn", *sorted(TORCHVISION))


def build_model(mcfg: dict, allow_download: bool = False) -> nn.Module:
    name = mcfg["name"]
    kwargs = {k: v for k, v in mcfg.items() if k != "name"}
    if name == "custom_cnn":
        return CustomCNN(**kwargs)
    if name in TORCHVISION:
        return build_torchvision_model(name, allow_download=allow_download, **kwargs)
    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")


def default_input(mcfg: dict) -> dict:
    """Input convention (protocol §4): the custom CNN takes 1 channel normalised with the
    train statistics; torchvision models take the grey channel replicated 3x and ImageNet
    normalisation (pretrained or not, so that only the pretraining differs)."""
    if mcfg["name"] == "custom_cnn":
        return {"channels": 1, "normalization": "dataset"}
    return {"channels": 3, "normalization": "imagenet"}


def parameter_groups(model: nn.Module, mcfg: dict, lr: float, head_lr: float | None) -> list | None:
    """Optimiser parameter groups.

    * ``None`` (historical Phase 2 path): ``fit`` builds ``Adam(model.parameters())``.
    * frozen backbone: one group with the trainable (head) parameters only.
    * ``head_lr`` set (fine-tuning): two groups, backbone at ``lr`` and head at ``head_lr``.
    """
    name = mcfg["name"]
    if name == "custom_cnn":
        if head_lr is not None:
            raise ValueError("head_lr is not defined for custom_cnn")
        return None
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if mcfg.get("freeze_backbone", False):
        if head_lr is not None:
            raise ValueError("Linear probe: use 'lr' only (the head is the only trained part)")
        return [{"params": [p for _, p in trainable], "lr": lr, "name": "head"}]
    if head_lr is None:
        return [{"params": [p for _, p in trainable], "lr": lr, "name": "all"}]
    head = head_parameter_names(model, name)
    return [
        {"params": [p for n, p in trainable if n not in head], "lr": lr, "name": "backbone"},
        {"params": [p for n, p in trainable if n in head], "lr": head_lr, "name": "head"},
    ]


__all__ = ["HEAD_MODULE", "MODEL_NAMES", "build_model", "default_input", "parameter_groups"]
