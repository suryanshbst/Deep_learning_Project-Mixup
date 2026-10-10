"""
train_noisy.py
Corrupted label noise training script evaluating ERM vs Mixup under 20% and 50% symmetric noise.
Author: Student (Roll No: 24CS3055)
"""

import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision
import torchvision.transforms as transforms
from src import PreActResNet18, add_label_noise, set_seed, train_one_epoch, evaluate


def run_noisy_label_experiment(alpha=None, noise=None, epochs=40, batch_size=128):
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    clean_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    noise_rates = [noise] if noise is not None else [0.2, 0.5]
    alphas = [alpha] if alpha is not None else [0.0, 8.0]

    for rate in noise_rates:
        print(f"\n==================================================")
        print(f"Evaluating Label Corruption Rate: {int(rate*100)}%")
        print(f"==================================================")
        
        noisy_ds = add_label_noise(clean_trainset, noise_rate=rate, seed=42)
        noisy_loader = torch.utils.data.DataLoader(noisy_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        crit = torch.nn.CrossEntropyLoss()

        for a in alphas:
            method_name = 'ERM' if a == 0.0 else f'Mixup (alpha={a})'
            print(f"Training {method_name} with {int(rate*100)}% symmetric noise for {epochs} epochs...")
            model = PreActResNet18().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            for ep in range(1, epochs + 1):
                train_loss, _ = train_one_epoch(model, noisy_loader, optimizer, crit, alpha=a, device=device)
                scheduler.step()
                if ep % 10 == 0 or ep == epochs:
                    print(f"Epoch [{ep:3d}/{epochs}] - Train Loss: {train_loss:.4f}")

            val_loss, acc, _, _ = evaluate(model, testloader, crit, device)
            print(f"[{int(rate*100)}% Noise - {method_name}] Top-1 Error: {100.0 - acc:.2f}% | Test Loss: {val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CIFAR-10 with Corrupted Labels')
    parser.add_argument('--alpha', type=float, default=None, help='Mixup alpha parameter (e.g. 0.0 for ERM, 8.0 for paper Table 2)')
    parser.add_argument('--noise', type=float, default=None, help='Symmetric label noise rate (e.g. 0.2 or 0.5)')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs (default: 40)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    args = parser.parse_args()

    run_noisy_label_experiment(alpha=args.alpha, noise=args.noise, epochs=args.epochs, batch_size=args.batch_size)
