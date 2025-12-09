# HUSAI: Hunting for Stable AI Features

> **Investigating the reproducibility crisis in sparse autoencoders and discovering the conditions for stable, interpretable features**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Verified_Findings-green)

---

## 🔬 Verified Key Findings

Our research provides the **first systematic empirical validation** of SAE stability on algorithmic tasks:

### 1. Dense Ground Truth → Low Stability (Validated)

SAE stability on dense ground truth matches theoretical predictions (Cui et al., 2025):

| Setup | PWMCC | Random Baseline | Interpretation |
|-------|-------|-----------------|----------------|
| 2-layer transformer | 0.309 | 0.300 | **Matches theory** |
| Copy task | 0.300 | 0.300 | Task-independent |

**Key insight:** When ground truth is dense (effective rank ~80/128 = 62.5%), SAE stability equals random baseline.

### 2. Stability Decreases with Sparsity (TopK)

For TopK architecture, stability decreases monotonically with L0:

| L0 (k) | PWMCC | Ratio to Random |
|--------|-------|------------------|
| 8 | 0.389 | **1.56×** |
| 16 | 0.339 | **1.36×** |
| 32 | 0.308 | 1.23× |
| 64 | 0.282 | 1.13× |

**Correlation:** r = -0.917 (strong negative)

### 3. Stability-Reconstruction Tradeoff

We discovered a fundamental tradeoff:
- **Underparameterized (d_sae < eff_rank):** High stability (2.87×), poor reconstruction
- **Matched (d_sae ≈ eff_rank):** Balanced (1.23-1.62×)
- **Overparameterized (d_sae > eff_rank):** Low stability (~1×), excellent reconstruction

### 4. Literature Validation

Our findings align with 2025 SAE literature:
- **Archetypal SAE (Fel et al., 2025):** Reports cosine similarity ~0.5 for standard SAEs
- **Our PWMCC:** 0.26-0.31 (consistent with algorithmic tasks being harder)
- **Cui et al. (2025):** Identifiability theory correctly predicts our dense regime results

---

## 📊 Experimental Results

### Stability-Aware Training Dynamics

![Stability Training](figures/stability_aware_training.pdf)

Training improves stability within each regime, but the matched regime shows dramatically better convergence.

### Task Complexity Analysis

![Task Complexity](figures/task_complexity_experiment.pdf)

Stability is not about task complexity per se, but about:
1. Model having learned the task well (high accuracy)
2. Activations having moderate effective rank
3. SAE size matching effective rank

---

## 🎯 The Problem We Addressed

In January 2025, Paulo & Belrose revealed that **SAEs trained on identical data with different random seeds learn entirely different features** — with only ~30% overlap in LLMs.

Our research asked: **Under what conditions can SAEs achieve stable, reproducible features?**

---

## 📁 Project Structure

```
HUSAI/
├── README.md                    # This file
├── paper/
│   └── sae_stability_paper.md   # Full research paper
├── src/                         # Core library
│   ├── data/                    # Dataset generation
│   ├── models/                  # Model architectures (SAE, Transformer)
│   ├── training/                # Training utilities
│   └── analysis/                # Analysis tools
├── scripts/
│   ├── training/                # SAE training scripts
│   ├── analysis/                # Analysis and visualization
│   └── experiments/             # Key experiments
├── results/                     # Experiment results
├── figures/                     # Generated figures
├── tests/                       # Test suite
└── archive/                     # Historical session notes
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/brightlikethelight/HUSAI.git
cd HUSAI

# Create environment
conda env create -f environment.yml
conda activate husai

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Run Key Experiments

```bash
# Stability-aware training (tests matched regime hypothesis)
KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/stability_aware_training.py

# Task complexity experiment
KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/task_complexity_experiment.py

# Hungarian matching analysis
KMP_DUPLICATE_LIB_OK=TRUE python scripts/analysis/hungarian_matching_analysis.py
```

---

## 🛠️ Key Scripts

### Training
| Script | Purpose |
|--------|---------|
| `scripts/training/train_sae.py` | Train SAE on transformer activations |
| `scripts/training/train_expanded_seeds.py` | Multi-seed SAE training |

### Analysis
| Script | Purpose |
|--------|---------|
| `scripts/analysis/analyze_feature_stability.py` | Compute PWMCC stability metrics |
| `scripts/analysis/hungarian_matching_analysis.py` | Hungarian matching for feature alignment |
| `scripts/analysis/comprehensive_statistical_analysis.py` | Full statistical analysis |

### Experiments
| Script | Purpose |
|--------|---------|
| `scripts/experiments/stability_aware_training.py` | Test Song et al. (2025) insights |
| `scripts/experiments/task_complexity_experiment.py` | Test task complexity hypothesis |
| `scripts/experiments/expansion_factor_analysis.py` | Analyze SAE size effects |

---

## 📚 Key References

### Our Work Builds On
- **Paulo & Belrose (2025)** - "SAEs Trained on the Same Data Learn Different Features" ([arXiv:2501.16615](https://arxiv.org/abs/2501.16615))
- **Song et al. (2025)** - "Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs" ([arXiv:2505.20254](https://arxiv.org/abs/2505.20254))
- **Nanda et al. (2023)** - "Progress measures for grokking via mechanistic interpretability" ([arXiv:2301.05217](https://arxiv.org/abs/2301.05217))

### SAE Methods
- **OpenAI (2024)** - "Scaling and evaluating sparse autoencoders" ([arXiv:2406.04093](https://arxiv.org/abs/2406.04093))
- **Anthropic (2024)** - "Scaling Monosemanticity" ([transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/))

---

## 🎓 Conclusions

### Verified Conclusions

1. **Dense ground truth → low stability** — matches identifiability theory (Cui et al., 2025)
2. **Stability decreases with sparsity** — verified for TopK architecture (r = -0.917)
3. **Task-independent baseline** — consistent across modular arithmetic and copy task
4. **Stability-reconstruction tradeoff** — fundamental property of SAE training

### Recommendations for Practitioners

1. **Always train multiple seeds** — single SAEs are unreliable
2. **Report stability metrics** — PWMCC should be standard alongside MSE
3. **Match d_sae to effective rank** — for best stability-reconstruction balance
4. **Validate interpretations across seeds** — if features don't replicate, they're not robust

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

This project builds on foundational work by:
- Neel Nanda, Joseph Bloom, and the TransformerLens/SAELens teams
- Anthropic's interpretability research team
- The broader mechanistic interpretability community

---

## 📬 Contact

**Project Lead:** Bright Liu (brightliu@college.harvard.edu)

---

<div align="center">

**Building reproducible, interpretable AI — one feature at a time.**

</div>
