"""
run_ablation.py
Automated hyperparameter ablation sweep script over alpha in {0.0, 0.2, 0.5, 1.0, 2.0}.
Author: Student (Roll No: 24CS3055)
"""

import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision
import torchvision.transforms as transforms
from src import PreActResNet18, compute_ece, set_seed, train_one_epoch, evaluate


def run_alpha_ablation_sweep(alphas=None, epochs=40, batch_size=128):
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

    alpha_list = alphas if alphas is not None else [0.0, 0.2, 0.5, 1.0, 2.0]
    results = {}

    for alpha in alpha_list:
        print(f"\n==================================================")
        print(f"Running Mixup Hyperparameter Sweep | Alpha = {alpha}")
        print(f"==================================================")
        
        model = PreActResNet18().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = torch.nn.CrossEntropyLoss()

        for ep in range(1, epochs + 1):
            train_loss, grad_norm = train_one_epoch(model, trainloader, optimizer, criterion, alpha, device)
            scheduler.step()
            if ep % 10 == 0 or ep == epochs:
                print(f"Epoch [{ep:3d}/{epochs}] - Train Loss: {train_loss:.4f} | Grad Norm: {grad_norm:.4f}")

        test_loss, top1_acc, logits, targets = evaluate(model, testloader, criterion, device)
        ece_score, _ = compute_ece(logits, targets)

        results[alpha] = {
            'top1_err': 100.0 - top1_acc,
            'test_loss': test_loss,
            'ece': ece_score
        }
        print(f"[Results alpha={alpha:0.1f}] Top-1 Error: {100.0 - top1_acc:.2f}% | Test Loss: {test_loss:.4f} | ECE: {ece_score:.4f}")

    print("\n==================================================")
    print("SUMMARY OF ALPHA ABLATION SWEEP")
    print("==================================================")
    for a, res in results.items():
        print(f"Alpha: {a:0.1f} -> Top-1 Err: {res['top1_err']:.2f}%, Test Loss: {res['test_loss']:.4f}, ECE: {res['ece']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Alpha Ablation Sweep on CIFAR-10')
    parser.add_argument('--alphas', nargs='+', type=float, default=None, help='List of alpha values (e.g. --alphas 0.0 0.2 0.5 1.0 2.0)')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs per alpha (default: 40)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    args = parser.parse_args()

    run_alpha_ablation_sweep(alphas=args.alphas, epochs=args.epochs, batch_size=args.batch_size)
