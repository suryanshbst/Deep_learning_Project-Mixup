"""
epoch_scaling.py
Evaluates model behavior when increasing training epochs (50 vs 120 epochs).
Paper Core Insight:
- ERM plateaus after 50 epochs (starts memorizing data).
- Mixup continues to improve between 50 and 120 epochs because vicinal linear interpolation acts as an ongoing regularizer.
Author: Student (Roll No: 24CS3055)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision
import torchvision.transforms as transforms
from src import PreActResNet18, set_seed, train_one_epoch, evaluate



def run_epoch_scaling_experiment(epoch_milestones=[50, 120], total_epochs=120, batch_size=128):
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    results = {}

    for method, alpha in [('ERM', 0.0), ('Mixup (alpha=1.0)', 1.0)]:
        print(f"\n==================================================")
        print(f"Training {method} up to {total_epochs} Epochs for Epoch Scaling Analysis")
        print(f"==================================================")
        
        model = PreActResNet18().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        criterion = torch.nn.CrossEntropyLoss()

        milestone_accs = {}
        for ep in range(1, total_epochs + 1):
            train_loss, grad_norm = train_one_epoch(model, trainloader, optimizer, criterion, alpha, device)
            scheduler.step()

            if ep in epoch_milestones:
                val_loss, val_acc, _, _ = evaluate(model, testloader, criterion, device)
                milestone_accs[ep] = {
                    'test_loss': val_loss,
                    'top1_acc': val_acc,
                    'top1_error': 100.0 - val_acc
                }
                print(f"[{method}] Epoch {ep:3d}/{total_epochs} -> Test Loss: {val_loss:.4f} | Top-1 Error: {100.0 - val_acc:.2f}%")

        results[method] = milestone_accs

    print("\n==================================================")
    print("EPOCH SCALING ANALYSIS SUMMARY (50 vs 120 Epochs)")
    print("==================================================")
    for method, res in results.items():
        err_50 = res[50]['top1_error']
        err_120 = res[120]['top1_error']
        diff = err_120 - err_50
        print(f"{method:18s} | Epoch 50 Error: {err_50:.2f}% | Epoch 120 Error: {err_120:.2f}% | Delta: {diff:+.2f}%")

    return results


if __name__ == '__main__':
    run_epoch_scaling_experiment()
