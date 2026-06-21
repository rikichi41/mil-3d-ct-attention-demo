from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class Small3DEncoder(nn.Module):
    """Small 3D CNN encoder used for the synthetic demo."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class AttentionMILClassifier(nn.Module):
    """Attention or gated-attention MIL classifier for one patient bag."""

    def __init__(self, embedding_dim: int = 64, attention_dim: int = 64, gated: bool = False):
        super().__init__()
        self.encoder = Small3DEncoder(embedding_dim=embedding_dim)
        self.gated = gated
        self.attention_v = nn.Linear(embedding_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        if gated:
            self.attention_u = nn.Linear(embedding_dim, attention_dim)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(bag)
        attention_hidden = torch.tanh(self.attention_v(features))
        if self.gated:
            attention_hidden = attention_hidden * torch.sigmoid(self.attention_u(features))
        attention_logits = self.attention_w(attention_hidden).squeeze(-1)
        weights = torch.softmax(attention_logits, dim=0)
        bag_feature = torch.sum(weights.unsqueeze(-1) * features, dim=0)
        logit = self.classifier(bag_feature).squeeze(-1)
        return logit, weights


class MaxMILClassifier(nn.Module):
    """Max-pooling MIL baseline for one patient bag."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.encoder = Small3DEncoder(embedding_dim=embedding_dim)
        self.instance_classifier = nn.Linear(embedding_dim, 1)

    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(bag)
        instance_logits = self.instance_classifier(features).squeeze(-1)
        max_index = torch.argmax(instance_logits)
        logit = instance_logits[max_index]
        weights = torch.zeros_like(instance_logits)
        weights[max_index] = 1.0
        return logit, weights
