"""Model registry and helpers for transfer learning."""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import torch.nn as nn
import torchvision.models as tv_models

MODEL_REGISTRY: Dict[str, Tuple] = {
    "efficientnet_b0": (tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights.IMAGENET1K_V2),
    "convnext_tiny": (tv_models.convnext_tiny, tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
}


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Instantiate a torchvision classification backbone with a new classification head."""

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{model_name}'. Available: {', '.join(MODEL_REGISTRY)}")

    constructor, weight_enum = MODEL_REGISTRY[model_name]
    weights = weight_enum if pretrained else None
    try:
        model = constructor(weights=weights)
    except Exception as exc:  # noqa: BLE001 (surface network errors gracefully)
        if pretrained:
            warnings.warn(f"Falling back to random initialization because pretrained weights failed: {exc}")
            model = constructor(weights=None)
        else:
            raise

    head_module = _replace_classifier(model, num_classes, dropout)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in head_module.parameters():
            param.requires_grad = True

    return model


def _replace_classifier(model: nn.Module, num_classes: int, dropout: float) -> nn.Module:
    """Swap the final classification layer across common torchvision backbones."""

    if hasattr(model, "classifier"):
        classifier = getattr(model, "classifier")
        if isinstance(classifier, nn.Sequential):
            in_features = classifier[-1].in_features
            new_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        else:
            in_features = classifier.in_features
            new_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        model.classifier = new_head
        return model.classifier

    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model.fc

    if hasattr(model, "head"):
        in_features = model.head.in_features
        model.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model.head

    raise AttributeError("Could not locate a classifier/fc/head attribute to replace.")
