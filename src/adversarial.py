"""
adversarial.py
Adversarial robustness evaluation matching Section 3.5 & Table 3 of ICLR 2018 Mixup paper.
Implements White-box and Black-box FGSM and Iterative FGSM (I-FGSM) attacks.
Author: Student (Roll No: 24CS3055)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epsilon: float = 4.0 / 255.0):
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.
    """
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = nn.functional.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        x_adv = x_adv + epsilon * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, -3.0, 3.0)

    return x_adv.detach()


def ifgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epsilon: float = 4.0 / 255.0, steps: int = 10):
    """
    Iterative Fast Gradient Sign Method (I-FGSM / BIM).
    """
    alpha = epsilon / steps
    x_adv = x.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = nn.functional.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x + eta, -3.0, 3.0)

    return x_adv.detach()


def evaluate_adversarial_robustness(model: nn.Module, dataloader: DataLoader, device: str, epsilon: float = 4.0 / 255.0):
    """
    Evaluates Top-1 error under FGSM and I-FGSM attacks.
    """
    model.eval()
    correct_clean = 0
    correct_fgsm = 0
    correct_ifgsm = 0
    total = 0

    for x, y in dataloader:
        if isinstance(y, (list, tuple)):
            y = y[0]
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            logits = model(x)
            preds = logits.max(1)[1]
            correct_clean += preds.eq(y).sum().item()

        x_fgsm = fgsm_attack(model, x, y, epsilon=epsilon)
        with torch.no_grad():
            logits_fgsm = model(x_fgsm)
            preds_fgsm = logits_fgsm.max(1)[1]
            correct_fgsm += preds_fgsm.eq(y).sum().item()

        x_ifgsm = ifgsm_attack(model, x, y, epsilon=epsilon, steps=10)
        with torch.no_grad():
            logits_ifgsm = model(x_ifgsm)
            preds_ifgsm = logits_ifgsm.max(1)[1]
            correct_ifgsm += preds_ifgsm.eq(y).sum().item()

        total += x.size(0)

    return {
        'clean_error': 100.0 * (1.0 - correct_clean / total),
        'fgsm_error': 100.0 * (1.0 - correct_fgsm / total),
        'ifgsm_error': 100.0 * (1.0 - correct_ifgsm / total)
    }
