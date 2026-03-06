"""ResNet50 baseline model for Real vs Synthetic MRI classification."""

import torch.nn as nn
from torchvision import models


def build_resnet50_baseline(pretrained: bool = True):
    """Build ResNet50 with a custom binary classification head.

    Architecture
    ------------
    ResNet50 (ImageNet pretrained) → Global Avg Pool →
        Linear(2048 → 512) → ReLU → Dropout(0.3) → Linear(512 → 1)

    Returns raw logits (use BCEWithLogitsLoss for training,
    apply sigmoid at inference for probability).
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    backbone = models.resnet50(weights=weights)

    # Replace the final fully-connected layer
    backbone.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
    )
    return backbone
