"""
ablations_table5.py
Implements alternative augmentation baselines matching Table 5 of Section 3.8:
- Label Smoothing (epsilon = 0.05, 0.1, 0.2)
- Input Gaussian Noise (sigma = 0.05, 0.1, 0.2)
- Mixup + Label Smoothing
- Weight decay variations (1e-4 vs 5e-4)
Author: Student (Roll No: 24CS3055)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing loss as described in Szegedy et al. (2016) & tested in Table 5.
    """
    def __init__(self, epsilon: float = 0.1, num_classes: int = 10):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = (1.0 - self.epsilon) * nll_loss + self.epsilon * smooth_loss
        return loss.mean()


def add_gaussian_noise(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """
    Adds Gaussian noise N(0, sigma^2) to input image tensors.
    """
    if sigma > 0.0:
        noise = torch.randn_like(x) * sigma
        return x + noise
    return x
