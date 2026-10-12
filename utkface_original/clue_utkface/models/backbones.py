from __future__ import annotations

import warnings
from typing import Tuple

from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    VGG11_Weights,
    VGG13_Weights,
    VGG16_Weights,
    resnet18,
    vgg11,
    vgg13,
    vgg16,
)

AVAILABLE_ARCHITECTURES = (
    "resnet18",
    "cnn3",
    "cnn4",
    "vgg11",
    "vgg13",
    "vgg16",
)

_ARCHITECTURE_ALIASES = {
    "res18": "resnet18",
    "resnet": "resnet18",
    "simple_cnn3": "cnn3",
    "simple_cnn4": "cnn4",
    "small_cnn3": "cnn3",
    "small_cnn4": "cnn4",
    "vgg": "vgg11",
}


class SmallCNNEncoder(nn.Module):
    def __init__(self, channels: Tuple[int, ...]) -> None:
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        return self.pool(x)


def normalize_architecture(architecture: str) -> str:
    key = architecture.strip().lower()
    return _ARCHITECTURE_ALIASES.get(key, key)


def build_backbone(architecture: str = "resnet18", pretrained: bool = True):
    architecture = normalize_architecture(architecture)

    if architecture == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        encoder = nn.Sequential(*list(backbone.children())[:-1])
        return encoder, backbone.fc.in_features

    if architecture == "cnn3":
        if pretrained:
            warnings.warn(
                "cnn3 does not have torchvision pretrained weights; using random initialization.",
                stacklevel=2,
            )
        return SmallCNNEncoder((64, 128, 256)), 256

    if architecture == "cnn4":
        if pretrained:
            warnings.warn(
                "cnn4 does not have torchvision pretrained weights; using random initialization.",
                stacklevel=2,
            )
        return SmallCNNEncoder((64, 128, 256, 512)), 512

    vgg_builders = {
        "vgg11": (vgg11, VGG11_Weights),
        "vgg13": (vgg13, VGG13_Weights),
        "vgg16": (vgg16, VGG16_Weights),
    }
    if architecture in vgg_builders:
        builder, weight_enum = vgg_builders[architecture]
        weights = weight_enum.IMAGENET1K_V1 if pretrained else None
        backbone = builder(weights=weights)
        encoder = nn.Sequential(backbone.features, backbone.avgpool)
        return encoder, backbone.classifier[0].in_features

    raise ValueError(
        f"Unsupported architecture: {architecture}. "
        f"Available options: {', '.join(AVAILABLE_ARCHITECTURES)}"
    )


__all__ = ["AVAILABLE_ARCHITECTURES", "build_backbone", "normalize_architecture"]
