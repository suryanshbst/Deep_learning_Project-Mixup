"""
engine.py
Training engine supporting standard ERM and Mixup regularization,
gradient norm logging, and evaluation.
Author: Student (Roll No: 24CS3055)
"""

import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from .mixup import mixup_data, mixup_criterion


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer, criterion, alpha: float, device: str):
    """
    Executes one training epoch with optional vectorized mixup augmentation and tracks mean gradient norms.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    grad_norms = []

    for x, y in dataloader:
        if isinstance(y, (list, tuple)):
            y = y[0]  # Handle noisy dataset tuple returning (noisy_target, clean_target)
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if alpha > 0.0:
            x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=alpha, device=device)
            preds = model(x_mix)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
        else:
            preds = model(x)
            loss = criterion(preds, y)

        loss.backward()

        # Compute L2 norm of gradients
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norms.append(total_norm ** 0.5)

        optimizer.step()

        running_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    avg_loss = running_loss / total_samples
    avg_grad_norm = float(np.mean(grad_norms)) if len(grad_norms) > 0 else 0.0
    return avg_loss, avg_grad_norm


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device: str):
    """
    Evaluates classification loss, accuracy, and returns logits and targets for calibration.
    Returns:
        avg_loss (float): Average cross-entropy loss
        acc (float): Top-1 accuracy percentage (0.0 to 100.0)
        all_logits (torch.Tensor): Model raw outputs [N, num_classes]
        all_targets (torch.Tensor): Ground truth labels [N]
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    logits_list = []
    targets_list = []

    for x, y in dataloader:
        if isinstance(y, (list, tuple)):
            y = y[0]
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = criterion(preds, y)

        running_loss += loss.item() * x.size(0)
        _, predicted = preds.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        logits_list.append(preds.cpu())
        targets_list.append(y.cpu())

    avg_loss = running_loss / total
    acc = 100.0 * (correct / total)
    all_logits = torch.cat(logits_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)

    return avg_loss, acc, all_logits, all_targets
