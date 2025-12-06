# HUSAI: Hunting for Stable AI Features

> **Investigating the reproducibility crisis in sparse autoencoders and discovering the conditions for stable, interpretable features**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Research_Complete-green)

---

## 🔬 Key Findings

Our research confirms and extends the findings of Paulo & Belrose (2025) on SAE feature instability, while identifying **critical conditions for achieving stability**:

### 1. The Matched Regime Insight (Song et al. 2025)

SAE stability depends critically on matching dictionary size to the effective rank of activations:

| Configuration | d_sae | Effective Rank | PWMCC | Ratio to Random |
|---------------|-------|----------------|-------|-----------------|
| **Matched (small)** | 64 | ~80 | 0.422 | **1.83×** |
| **Matched (medium)** | 128 | ~80 | 0.341 | **1.38×** |
| Overparameterized | 1024 | ~80 | 0.307 | 1.03× |

**Key insight:** When `d_sae ≈ effective_rank`, stability improves by 38-83% over random baseline!

### 2. Model Quality Matters

Stability correlates with how well the underlying model has learned the task:

| Task | Model Accuracy | PWMCC | Stability Ratio |
|------|---------------|-------|-----------------|
| Multiplication | 99.3% | 0.392 | **1.57×** |
| Addition | 66% | 0.312 | 1.25× |
| Combined | 87% | 0.295 | 1.18× |

### 3. Modular Arithmetic Shows Extreme Instability

For modular arithmetic tasks with overparameterized SAEs:
- **PWMCC ≈ 0.30** (indistinguishable from random baseline)
- **0% shared features** above 0.5 similarity threshold
- This contrasts with Paulo & Belrose's ~65% shared features in LLM SAEs

**Root cause:** Modular arithmetic activations lack the interpretable structure found in LLM activations.

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

### What We Learned

1. **SAE stability is achievable** — but requires careful configuration
2. **The matched regime is key** — d_sae should approximate effective rank
3. **Model quality matters** — well-trained models produce more stable SAE features
4. **Task structure affects stability** — tasks with interpretable structure show better stability

### Recommendations for Practitioners

1. **Compute effective rank** of your activations before choosing SAE size
2. **Use TopK SAEs** — they naturally enforce sparsity constraints
3. **Train models well** — SAE stability depends on underlying model quality
4. **Normalize decoder weights** — critical for training stability

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
