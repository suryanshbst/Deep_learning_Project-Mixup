"""
run_experiments.py
Automated runner executing Experiments A, B, and C.
Author: Student (Roll No: 24CS3055)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src import PreActResNet18, mixup_data, mixup_criterion, add_label_noise, compute_ece, set_seed, train_one_epoch, evaluate



def run_all_experiments(epochs=40, batch_size=128):
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

    # Experiment A: Clean CIFAR-10
    print("\n=== Experiment A: Clean CIFAR-10 Benchmark ===")
    m_erm = PreActResNet18().to(device)
    opt_erm = torch.optim.SGD(m_erm.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched_erm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_erm, T_max=epochs)
    crit = torch.nn.CrossEntropyLoss()

    erm_test_errs = []
    for ep in range(1, epochs + 1):
        train_one_epoch(m_erm, trainloader, opt_erm, crit, alpha=0.0, device=device)
        sched_erm.step()
        val_loss, val_acc, _, _ = evaluate(m_erm, testloader, crit, device)
        erm_test_errs.append(100.0 - val_acc)
    _, erm_acc, _, _ = evaluate(m_erm, testloader, crit, device)

    m_mix = PreActResNet18().to(device)
    opt_mix = torch.optim.SGD(m_mix.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched_mix = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mix, T_max=epochs)

    mix_test_errs = []
    for ep in range(1, epochs + 1):
        train_one_epoch(m_mix, trainloader, opt_mix, crit, alpha=1.0, device=device)
        sched_mix.step()
        val_loss, val_acc, _, _ = evaluate(m_mix, testloader, crit, device)
        mix_test_errs.append(100.0 - val_acc)
    _, mix_acc, _, _ = evaluate(m_mix, testloader, crit, device)

    print(f"[Exp A Complete] ERM Top-1 Error: {100.0 - erm_acc:.2f}% | Mixup Top-1 Error: {100.0 - mix_acc:.2f}%")


if __name__ == '__main__':
    run_all_experiments()
