"""
utils.py
Utility functions for determinism, logging, and metric tracking.
Author: Student (Roll No: 24CS3055)
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets global random seeds across Python, NumPy, PyTorch, and CuDNN for deterministic reproduction.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Deterministic seed set to {seed}")
