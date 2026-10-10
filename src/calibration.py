"""
calibration.py
Calculates Expected Calibration Error (ECE) and plots reliability diagrams.
Author: Student (Roll No: 24CS3055)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    """
    Computes Expected Calibration Error (ECE) across confidence bins.
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.tensor(0.0)
    bin_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += torch.abs(torch.tensor(avg_confidence_in_bin - accuracy_in_bin)) * prop_in_bin
            bin_data.append((avg_confidence_in_bin, accuracy_in_bin, prop_in_bin))
        else:
            bin_data.append((0.0, 0.0, 0.0))

    return ece.item(), bin_data


def plot_reliability_diagram(bin_data, title: str, save_path: str = None):
    """
    Plots a reliability diagram comparing confidence vs empirical accuracy.
    """
    n_bins = len(bin_data)
    bin_centers = np.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins)
    accs = [b[1] for b in bin_data]

    plt.figure(figsize=(6, 5))
    plt.bar(bin_centers, accs, width=1.0 / n_bins, alpha=0.7, color='#1f77b4', edgecolor='black', label='Model Outputs')
    plt.plot([0, 1], [0, 1], '--', color='red', linewidth=2, label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
