"""
noise.py
Synthetic label noise injection module matching Section 3.4 of the ICLR 2018 paper.
Author: Student (Roll No: 24CS3055)
"""

import copy
import numpy as np
import torch
from torch.utils.data import Dataset


def add_label_noise(train_dataset, noise_rate=0.2, num_classes=10, seed=42):
    """
    Injects symmetric uniform label noise into a PyTorch Dataset and returns a noisy dataset copy.
    """
    noisy_dataset = copy.deepcopy(train_dataset)
    if hasattr(noisy_dataset, 'targets'):
        clean_targets = np.array(noisy_dataset.targets)
    elif hasattr(noisy_dataset, 'labels'):
        clean_targets = np.array(noisy_dataset.labels)
    else:
        clean_targets = np.array([y for _, y in noisy_dataset])

    noisy_targets = np.copy(clean_targets)

    if noise_rate > 0.0:
        np.random.seed(seed)
        n_samples = len(clean_targets)
        n_noisy = int(noise_rate * n_samples)
        noisy_idx = np.random.choice(n_samples, size=n_noisy, replace=False)

        for idx in noisy_idx:
            orig_class = clean_targets[idx]
            other_classes = [c for c in range(num_classes) if c != orig_class]
            noisy_targets[idx] = np.random.choice(other_classes)

    if hasattr(noisy_dataset, 'targets'):
        noisy_dataset.targets = noisy_targets.tolist()

    return SymmetricNoisyCIFAR10(noisy_dataset, clean_targets, noisy_targets)


class SymmetricNoisyCIFAR10(Dataset):
    """
    Dataset wrapper holding image data, clean targets, and corrupted noisy targets.
    """
    def __init__(self, dataset, clean_targets, noisy_targets):
        self.dataset = dataset
        self.clean_targets = clean_targets
        self.noisy_targets = noisy_targets

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        return img, self.noisy_targets[index], self.clean_targets[index]

    def __len__(self):
        return len(self.dataset)
