from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .backbones import build_backbone


class NormalInvGammaHead(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 4)

    def forward(self, x: torch.Tensor):
        out = self.fc(x)
        mu, logv, loga, logb = torch.chunk(out, 4, dim=-1)
        v = F.softplus(logv) + 1e-6
        alpha = F.softplus(loga) + 1.0 + 1e-6
        beta = F.softplus(logb) + 1e-6
        return mu, v, alpha, beta


class ResNetDER(nn.Module):
    """Evidential regression model with a configurable image backbone."""

    def __init__(
        self,
        architecture: str = "resnet18",
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.encoder, in_dim = build_backbone(architecture=architecture, pretrained=pretrained)

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            NormalInvGammaHead(hidden_dim, 1),
        )

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.pixel_mean) / self.pixel_std

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)
        return feat

    def forward(self, x: torch.Tensor):
        feat = self.extract_features(x)
        return self.head(feat)


def evidential_regression_loss(pred, y: torch.Tensor, lamb: float = 1e-3) -> torch.Tensor:
    mu, v, alpha, beta = pred
    two_b_lambda = 2.0 * beta * (1.0 + v)
    nll = (
        0.5 * torch.log(torch.tensor(math.pi, device=y.device, dtype=y.dtype) / v)
        - alpha * torch.log(two_b_lambda)
        + (alpha + 0.5) * torch.log(v * (y - mu).pow(2) + two_b_lambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    ).mean()
    error = torch.abs(y - mu)
    evidence = 2.0 * v + alpha
    reg = (error * evidence).mean()
    return nll + lamb * reg


def evidential_regression_elbo_loss(pred, y: torch.Tensor, lamb: float = 1e-3) -> torch.Tensor:
    mu, v, alpha, beta = pred
    two_b_lambda = 2.0 * beta * (1.0 + v)
    nll = (
        0.5 * torch.log(torch.tensor(math.pi, device=y.device, dtype=y.dtype) / v)
        - alpha * torch.log(two_b_lambda)
        + (alpha + 0.5) * torch.log(v * (y - mu).pow(2) + two_b_lambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    ).mean()
    kl = (
        -0.5 * torch.log(v)
        + alpha * torch.log(beta)
        - torch.lgamma(alpha)
        + torch.lgamma(alpha + 0.5)
        - (alpha + 0.5)
        + (alpha + 0.5) * (v * (y - mu).pow(2) + 2.0 * beta * (1.0 + v)) / (2.0 * beta * (1.0 + v))
    ).mean()
    return nll + lamb * kl
