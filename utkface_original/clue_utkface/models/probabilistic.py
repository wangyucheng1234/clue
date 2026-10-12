from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .backbones import build_backbone


class MonteCarloDropout(nn.Dropout):
    """Dropout layer that can stay active during evaluation for MC sampling."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=p, inplace=inplace)
        self.mc_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, self.p, training=self.training or self.mc_enabled, inplace=self.inplace)


class GaussianRegressionHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        dropout_layer: nn.Module = nn.Identity()
        if dropout > 0:
            dropout_layer = MonteCarloDropout(dropout)

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            dropout_layer,
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean, raw_var = torch.chunk(out, 2, dim=-1)
        var = F.softplus(raw_var) + 1e-6
        return mean, var

    def set_mc_dropout(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, MonteCarloDropout):
                module.mc_enabled = enabled


class BackboneGaussianRegressor(nn.Module):
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
        self.head = GaussianRegressionHead(in_features=in_dim, hidden_dim=hidden_dim, dropout=dropout)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.pixel_mean) / self.pixel_std

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        feat = self.encoder(x)
        return torch.flatten(feat, 1)

    def set_mc_dropout(self, enabled: bool) -> None:
        self.head.set_mc_dropout(enabled)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        return self.head(feat)


class BayesLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        init_rho: float = -5.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if prior_std <= 0:
            raise ValueError("prior_std must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.init_rho = float(init_rho)

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_rho = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        self.register_buffer("prior_std", torch.tensor(float(prior_std), dtype=torch.float32), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, self.init_rho)
        if self.bias_mu is not None:
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, self.init_rho)

    @staticmethod
    def _sigma(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-6

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            weight_sigma = self._sigma(self.weight_rho)
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            bias = None
            if self.bias_mu is not None:
                bias_sigma = self._sigma(self.bias_rho)
                bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        prior_var = self.prior_std.pow(2)
        weight_sigma = self._sigma(self.weight_rho)
        weight_kl = torch.log(self.prior_std / weight_sigma)
        weight_kl = weight_kl + 0.5 * (weight_sigma.pow(2) + self.weight_mu.pow(2)) / prior_var - 0.5
        total_kl = weight_kl.sum()

        if self.bias_mu is not None:
            bias_sigma = self._sigma(self.bias_rho)
            bias_kl = torch.log(self.prior_std / bias_sigma)
            bias_kl = bias_kl + 0.5 * (bias_sigma.pow(2) + self.bias_mu.pow(2)) / prior_var - 0.5
            total_kl = total_kl + bias_kl.sum()

        return total_kl


class BayesianGaussianRegressionHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        prior_std: float = 1.0,
        init_rho: float = -5.0,
    ) -> None:
        super().__init__()
        self.hidden = BayesLinear(
            in_features=in_features,
            out_features=hidden_dim,
            prior_std=prior_std,
            init_rho=init_rho,
        )
        self.out = BayesLinear(
            in_features=hidden_dim,
            out_features=2,
            prior_std=prior_std,
            init_rho=init_rho,
        )

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.hidden(x, sample=sample), inplace=False)
        out = self.out(hidden, sample=sample)
        mean, raw_var = torch.chunk(out, 2, dim=-1)
        var = F.softplus(raw_var) + 1e-6
        return mean, var

    def kl_divergence(self) -> torch.Tensor:
        return self.hidden.kl_divergence() + self.out.kl_divergence()


class BackboneBBBRegressor(nn.Module):
    def __init__(
        self,
        architecture: str = "resnet18",
        pretrained: bool = True,
        hidden_dim: int = 256,
        prior_std: float = 1.0,
        init_rho: float = -5.0,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.encoder, in_dim = build_backbone(architecture=architecture, pretrained=pretrained)
        self.head = BayesianGaussianRegressionHead(
            in_features=in_dim,
            hidden_dim=hidden_dim,
            prior_std=prior_std,
            init_rho=init_rho,
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
        return torch.flatten(feat, 1)

    def kl_divergence(self) -> torch.Tensor:
        return self.head.kl_divergence()

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        return self.head(feat, sample=sample)


def gaussian_nll_loss(mean: torch.Tensor, y: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    var = var.clamp_min(1e-6)
    return (0.5 * torch.log(2.0 * math.pi * var) + 0.5 * (y - mean).pow(2) / var).mean()


__all__ = [
    "MonteCarloDropout",
    "GaussianRegressionHead",
    "BackboneGaussianRegressor",
    "BayesLinear",
    "BayesianGaussianRegressionHead",
    "BackboneBBBRegressor",
    "gaussian_nll_loss",
]
