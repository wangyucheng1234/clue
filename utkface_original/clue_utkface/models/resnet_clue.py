from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone


class ResNetCLUE(nn.Module):
    """
    CLUE image regressor with a configurable image backbone.

    Variants
    --------
    fixed_noise:
        forward(x) -> mu, var_f
        forward(x, return_features=True) -> mu, var_f, feat

    variable_noise:
        forward(x) -> mu_f, var_f, mu_logsigma, var_logsigma
        forward(x, return_features=True) -> mu_f, var_f, mu_logsigma, var_logsigma, feat
    """

    def __init__(
        self,
        variant: str = "fixed_noise",
        architecture: str = "resnet18",
        pretrained: bool = True,
        hidden_dim: int = 256,
        proj_dim: int = 256,
        dropout: float = 0.0,
        l2_normalize_features: bool = True,
    ) -> None:
        super().__init__()
        if variant not in {"fixed_noise", "variable_noise"}:
            raise ValueError(f"Unsupported variant: {variant}")

        self.variant = variant
        self.architecture = architecture
        self.l2_normalize_features = l2_normalize_features

        self.encoder, in_dim = build_backbone(architecture=architecture, pretrained=pretrained)
        dropout_layer = nn.Identity() if dropout <= 0 else nn.Dropout(dropout)

        self.projector = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            dropout_layer,
        )
        self.shared_head = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Identity() if dropout <= 0 else nn.Dropout(dropout),
        )

        out_dim = 2 if variant == "fixed_noise" else 4
        self.out = nn.Linear(hidden_dim, out_dim)
        self.softplus = nn.Softplus()

        # Keep normalization inside the model so anchor sampling can stay in [0, 1].
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
        feat = self.projector(feat)
        if self.l2_normalize_features:
            feat = F.normalize(feat, p=2, dim=-1)
        return feat

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.extract_features(x)
        h = self.shared_head(feat)
        out = self.out(h)

        if self.variant == "fixed_noise":
            mu, raw_var = torch.chunk(out, 2, dim=-1)
            var_f = self.softplus(raw_var) + 1e-6
            if return_features:
                return mu, var_f, feat
            return mu, var_f

        mu_f, raw_var_f, mu_logsigma, raw_var_logsigma = torch.chunk(out, 4, dim=-1)
        var_f = self.softplus(raw_var_f) + 1e-6
        var_logsigma = self.softplus(raw_var_logsigma) + 1e-6
        if return_features:
            return mu_f, var_f, mu_logsigma, var_logsigma, feat
        return mu_f, var_f, mu_logsigma, var_logsigma


@torch.no_grad()
def sample_anchor_images(x: torch.Tensor, severity: float = 0.10) -> torch.Tensor:
    noise = torch.randn_like(x) * severity
    return torch.clamp(x + noise, 0.0, 1.0)


def rbf_rho_features(feat_q: torch.Tensor, feat_x: torch.Tensor, ell: float) -> torch.Tensor:
    if ell <= 0:
        raise ValueError("ell must be > 0")
    d2 = torch.cdist(feat_q, feat_x, p=2.0).pow(2)
    return torch.exp(-0.5 * d2 / (ell ** 2))
