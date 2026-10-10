"""
train_clean.py
Clean CIFAR-10 training script comparing ERM (alpha=0.0) vs Mixup (alpha=1.0).
Author: Student (Roll No: 24CS3055)
"""

import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision
import torchvision.transforms as transforms
from src import PreActResNet18, mixup_data, mixup_criterion, compute_ece, set_seed, train_one_epoch, evaluate


def run_clean_training(alpha=None, epochs=50, batch_size=128):
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    if alpha is not None:
        methods = [('Mixup' if alpha > 0 else 'ERM', alpha)]
    else:
        methods = [('ERM', 0.0), ('Mixup (alpha=1.0)', 1.0)]

    for method, a in methods:
        print(f"\n==================================================")
        print(f"Training {method} (alpha={a}) on Clean CIFAR-10 for {epochs} Epochs")
        print(f"==================================================")
        
        model = PreActResNet18().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = torch.nn.CrossEntropyLoss()

        for ep in range(1, epochs + 1):
            train_loss, grad_norm = train_one_epoch(model, trainloader, optimizer, criterion, a, device)
            scheduler.step()
            if ep % 10 == 0 or ep == epochs:
                print(f"Epoch [{ep:3d}/{epochs}] - Train Loss: {train_loss:.4f} | Grad Norm: {grad_norm:.4f}")

        val_loss, val_acc, logits, targets = evaluate(model, testloader, criterion, device)
        ece_score, _ = compute_ece(logits, targets)

        print(f"[{method}] Top-1 Error: {100.0 - val_acc:.2f}% | Test Loss: {val_loss:.4f} | ECE: {ece_score:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CIFAR-10 with ERM or Mixup')
    parser.add_argument('--alpha', type=float, default=None, help='Mixup alpha parameter (default: runs both 0.0 and 1.0)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
    args = parser.parse_args()

    run_clean_training(alpha=args.alpha, epochs=args.epochs, batch_size=args.batch_size)
