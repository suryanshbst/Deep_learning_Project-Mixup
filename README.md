# mixup: Beyond Empirical Risk Minimization — PyTorch Reproduction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

> **Student**: Suryansh (Roll No: `24CS3055`)  
> **Department**: Department of Computer Science & Engineering, Rajiv Gandhi Institute of Petroleum Technology  
> **Course**: Deep Learning 2026 Paper Implementation Project (Paper ID: `DL-008`)  
> **Original Paper**: *"mixup: Beyond Empirical Risk Minimization"* by Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz (ICLR 2018)

---

## 📌 Executive Summary

This repository presents an empirical reproduction and analysis of the ICLR 2018 paper **"mixup: Beyond Empirical Risk Minimization"** by Zhang et al. Deep neural networks trained under standard Empirical Risk Minimization (ERM) tend to memorize corrupted labels, exhibit vulnerability to adversarial perturbations, and display predictive overconfidence.

**mixup** introduces a generic, data-agnostic Vicinal Risk Minimization (VRM) principle. Instead of training models exclusively on discrete training points, mixup trains neural networks on convex linear combinations of pairs of feature vectors and target labels:
$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j$$
where $(x_i, y_i)$ and $(x_j, y_j)$ are pairs sampled at random from the training set, and mixing factor $\lambda \in [0, 1]$ is drawn from a Beta distribution $\text{Beta}(\alpha, \alpha)$.

Our experiments on CIFAR-10 confirm that mixup regularizes deep networks by enforcing smooth linear transitions across decision boundaries, suppressing corrupted label memorization under 20% and 50% symmetric noise, and reducing Expected Calibration Error (ECE).

---

## 📂 Repository Directory Tree

```text
DL_PROJECT/
├── notebooks/
│   └── mixup_cifar10_reproduction.ipynb   # Interactive Google Colab Notebook
├── src/
│   ├── __init__.py                         # Package initialization
│   ├── mixup.py                            # Vectorized mixup_data & mixup_criterion
│   ├── models.py                           # PreAct ResNet-18 for 32x32 CIFAR-10
│   ├── noise.py                            # Symmetric label noise injector (add_label_noise)
│   ├── calibration.py                      # ECE calculation & reliability diagrams
│   ├── adversarial.py                      # Adversarial robustness evaluation
│   ├── manifold_mixup.py                   # Manifold mixup extension
│   ├── ablations_table5.py                 # Table 5 ablation sweep
│   └── utils.py                            # Deterministic seed & helper functions
├── scripts/
│   ├── train_clean.py                      # Clean CIFAR-10 training (ERM vs Mixup)
│   ├── train_noisy.py                      # Corrupted label noise experiments (20%, 50%)
│   ├── run_ablation.py                     # Alpha hyperparameter ablation sweep
│   ├── epoch_scaling.py                    # 50 vs 120 epoch scaling dynamics
│   └── run_experiments.py                  # Full experiment suite runner
├── results/
│   ├── figures/                            # Exported comparison plots
│   └── tables/                             # Numerical performance logs
├── report/
│   ├── main.tex                            # 5-Page IEEE format technical report
│   └── DL_008_Mixup_Report.pdf             # Compiled report PDF
├── requirements.txt                        # Python package dependencies
├── viva_defense_guide.md                   # Presentation and viva defense guide
└── README.md                               # Repository documentation
```

---

## 🛠️ Setup & Installation

### 1. Requirements
* **Python**: `3.10+`
* **PyTorch**: `2.0+` (CUDA 11.8 / 12.1 GPU support recommended)
* **GPU**: Google Colab Nvidia T4 GPU (16GB VRAM) or local Nvidia GPU

### 2. Environment Setup
```bash
git clone https://github.com/your-username/DL_PROJECT.git
cd DL_PROJECT
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Clean CIFAR-10 Training (ERM vs Mixup)
```bash
# Standard ERM (alpha = 0.0)
python scripts/train_clean.py --alpha 0.0 --epochs 50 --batch_size 128

# mixup Regularization (alpha = 1.0)
python scripts/train_clean.py --alpha 1.0 --epochs 50 --batch_size 128
```

### 2. Corrupted Label Noise Experiments (Table 2 Reproduction)
```bash
# ERM with 20% symmetric label noise
python scripts/train_noisy.py --alpha 0.0 --noise 0.2 --epochs 50

# Mixup with 20% symmetric label noise
python scripts/train_noisy.py --alpha 1.0 --noise 0.2 --epochs 50

# Mixup with 50% symmetric label noise
python scripts/train_noisy.py --alpha 1.0 --noise 0.5 --epochs 50
```

### 3. Alpha Hyperparameter Sweep (Figure 3b Reproduction)
```bash
python scripts/run_ablation.py --alphas 0.0 0.2 0.5 1.0 2.0 --epochs 50
```

---

## 🎯 Seeds & Reproducibility Notice

All experiments set exact deterministic seeds (`seed=42`) across Python, NumPy, and PyTorch backends:
```python
from src.utils import set_seed
set_seed(42)
```

---

## 📊 Summary of Results (CIFAR-10, PreAct ResNet-18)

| Metric / Setting | ERM ($\alpha=0.0$) | mixup ($\alpha=1.0$) | Original Paper Reference |
|---|:---:|:---:|:---:|
| Clean Test Error (50 ep) | 7.42% | **5.98%** | 5.5% (200 ep) |
| Clean Test Error (120 ep) | 6.84% | **5.45%** | 5.5% (200 ep) |
| 20% Label Noise Test Error | 14.15% | **8.22%** | 8.8% |
| 50% Label Noise Test Error | 31.80% | **14.65%** | 16.5% |
| Expected Calibration Error (ECE) | 0.0840 | **0.0310** | Significantly lower ECE |

---

## 📚 Citations & References

```bibtex
@inproceedings{zhang2018mixup,
  title={mixup: Beyond Empirical Risk Minimization},
  author={Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N. and Lopez-Paz, David},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}
```
