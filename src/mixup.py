"""
mixup.py
Vectorized implementation of Mixup (Vicinal Risk Minimization) for PyTorch.
Author: Student (Roll No: 24CS3055)
"""

import torch
import torch.nn as nn
import numpy as np


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Computes mixed inputs, target pairs, and Beta mixing coefficient lambda.
    """
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    if use_cuda and torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes linear cross-entropy loss over target pairs.
    """
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
