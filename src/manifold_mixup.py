"""
manifold_mixup.py
Manifold / Latent Layer Mixup and Table 5 Ablation Study Modules.
Implements feature map interpolation at residual block layers (Layers 1-5).
Paper: Section 3.8 & Table 5
Author: Student (Roll No: 24CS3055)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ManifoldPreActResNet18(nn.Module):
    """
    PreAct ResNet-18 with Manifold Mixup capability.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        from .models import PreActResidualBlock
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.bn_last = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        from .models import PreActResidualBlock
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(PreActResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward_manifold(self, x: torch.Tensor, target: torch.Tensor = None, alpha: float = 1.0, mix_layer: int = 0):
        if self.training and alpha > 0.0:
            lam = np.random.beta(alpha, alpha)
            perm = torch.randperm(x.size(0), device=x.device)
            y_a, y_b = target, target[perm]
        else:
            lam = 1.0
            perm = None
            y_a, y_b = target, target

        if mix_layer == 0 and self.training and alpha > 0.0:
            x = lam * x + (1.0 - lam) * x[perm]

        out = self.conv1(x)
        out = self.layer1(out)
        if mix_layer == 1 and self.training and alpha > 0.0:
            out = lam * out + (1.0 - lam) * out[perm]

        out = self.layer2(out)
        if mix_layer == 2 and self.training and alpha > 0.0:
            out = lam * out + (1.0 - lam) * out[perm]

        out = self.layer3(out)
        if mix_layer == 3 and self.training and alpha > 0.0:
            out = lam * out + (1.0 - lam) * out[perm]

        out = self.layer4(out)
        if mix_layer == 4 and self.training and alpha > 0.0:
            out = lam * out + (1.0 - lam) * out[perm]

        out = F.relu(self.bn_last(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        if mix_layer == 5 and self.training and alpha > 0.0:
            out = lam * out + (1.0 - lam) * out[perm]

        logits = self.linear(out)
        return logits, y_a, y_b, lam
