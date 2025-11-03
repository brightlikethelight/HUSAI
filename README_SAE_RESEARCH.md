# SAE Feature Stability Research

**Investigating reproducibility in Sparse Autoencoder feature learning**

[![Status](https://img.shields.io/badge/status-phase%202%20complete-success)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red)]()

---

## 🎯 Research Question

**Do Sparse Autoencoders learn stable, reproducible features across different random seeds?**

**Answer:** ❌ **NO** - SAEs show systematic feature instability (PWMCC ~0.30), independent of architecture.

---

## 📊 Key Findings

### Major Result: Low Feature Stability

Both TopK and ReLU SAEs achieve excellent reconstruction metrics but show **low feature stability** across random seeds:

| Architecture | PWMCC | Explained Variance | Dead Neurons | Fourier Overlap |
|--------------|-------|-------------------|--------------|-----------------|
| **TopK** | 0.302 ± 0.001 | 0.923 ± 0.002 | 0.4% ± 0.1% | 0.258 ± 0.002 |
| **ReLU** | 0.300 ± 0.001 | 0.980 ± 0.002 | 15.6% ± 2.9% | 0.258 ± 0.002 |

**Interpretation:**
- ✅ Excellent reconstruction (EV ~0.92-0.98)
- ✅ Correct sparsity levels
- ✅ Low dead neurons
- ❌ **Features DO NOT converge** (PWMCC ~0.30 vs target >0.7)
- ❌ **Ground truth NOT recovered** (Fourier overlap ~0.26 vs expected 0.6-0.8)

### Critical Insight

**Good reconstruction metrics ≠ stable, interpretable features!**

SAEs can perfectly reconstruct activations while learning completely different, unstable features across training runs.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/brightlikethelight/HUSAI.git
cd HUSAI

# Install dependencies
pip install torch transformers-lens numpy matplotlib tqdm
```

### Run Complete Experiment

```bash
# Set OpenMP workaround (macOS)
export KMP_DUPLICATE_LIB_OK=TRUE

# Train 5 TopK SAEs
for seed in 42 123 456 789 1011; do
    python scripts/train_simple_sae.py \
        --transformer results/transformer_5000ep/transformer_best.pt \
        --architecture topk \
        --seed $seed \
        --save-dir results/saes/topk_seed${seed}
done

# Analyze stability
python scripts/analyze_feature_stability.py \
    --sae-dir results/saes \
    --pattern "topk_seed*/sae_final.pt" \
    --architecture topk \
    --save-plots results/analysis
```

### Expected Runtime

- **Per SAE:** ~5 minutes (20 epochs, CPU)
- **Complete experiment:** ~25 minutes (5 SAEs)
- **Analysis:** <1 minute

---

## 📁 Repository Structure

```
HUSAI/
├── src/                           # Source code
│   ├── models/
│   │   ├── transformer.py        # Modular arithmetic transformer
│   │   └── simple_sae.py        # Custom TopK & ReLU SAE
│   ├── analysis/
│   │   ├── fourier_validation.py # Ground truth validation
│   │   └── feature_matching.py   # PWMCC implementation
│   └── data/
│       └── modular_arithmetic.py # Dataset generation
├── scripts/                       # Executable scripts
│   ├── train_simple_sae.py       # Main SAE training
│   └── analyze_feature_stability.py # PWMCC analysis
├── results/                       # Experimental results
│   ├── transformer_5000ep/       # Trained transformer
│   ├── saes/                     # 10 trained SAEs
│   └── analysis/                 # Stability analysis
├── docs/                          # Documentation
│   └── results/
│       ├── phase1_topk_stability.md
│       └── phase2_architecture_comparison.md
├── RESEARCH_SUMMARY.md            # Complete findings
└── README_SAE_RESEARCH.md         # This file
```

---

## 🔬 Methodology

### Experimental Setup

**Transformer (Ground Truth):**
- Task: Modular arithmetic (mod 113)
- Architecture: 2-layer, d_model=128
- Training: 5000 epochs with grokking
- Performance: 100% accuracy

**SAE Configuration:**
- **TopK:** k=32, 8× expansion (128 → 1024)
- **ReLU:** L1=1e-3, 8× expansion
- **Training:** 20 epochs, LR=3e-4
- **Seeds:** 5 per architecture (42, 123, 456, 789, 1011)

### Metrics

**PWMCC (Pairwise Maximum Cosine Correlation):**
- Measures feature alignment across SAEs
- Range: [0, 1], higher = more stable
- Threshold: >0.7 for "high stability"

**Fourier Overlap:**
- Ground truth validation for modular arithmetic
- Range: [0, 1], higher = better circuit recovery
- Expected: 0.6-0.8

---

## 📈 Results

### Phase 1: TopK Stability

**5 TopK SAEs trained with different seeds:**

| Seed | L0 | EV | Dead % | Fourier |
|------|----|----|--------|---------|
| 42 | 32.00 | 0.9215 | 0.3% | 0.259 |
| 123 | 32.00 | 0.9226 | 0.4% | 0.254 |
| 456 | 32.00 | 0.9257 | 0.2% | 0.260 |
| 789 | 32.00 | 0.9224 | 0.5% | 0.257 |
| 1011 | 32.00 | 0.9222 | 0.5% | 0.259 |

**PWMCC:** 0.302 ± 0.001 (❌ LOW - features unstable)

### Phase 2: Architecture Comparison

**TopK vs ReLU stability:**

| Metric | TopK | ReLU | Difference |
|--------|------|------|------------|
| PWMCC | 0.302 | 0.300 | 0.002 (insignificant) |
| EV | 0.923 | 0.980 | ReLU better reconstruction |
| L0 | 32.0 | 427.0 | Different sparsity |
| Dead % | 0.4% | 15.6% | TopK fewer dead |
| Fourier | 0.258 | 0.258 | Identical (both low) |

**Conclusion:** Instability is **architecture-independent**!

---

## 🎓 Implications

### For SAE Research

1. **Reconstruction ≠ Interpretability**
   - Standard metrics (EV, L0, dead %) insufficient
   - Need stability and ground truth validation

2. **Architecture Won't Solve It**
   - TopK and ReLU equally unstable
   - Problem lies in training dynamics, not architecture

3. **Reproducibility Crisis Confirmed**
   - Matches Paulo & Belrose baseline (~0.30)
   - Fundamental challenge in SAE training

### For Mechanistic Interpretability

1. **Be Cautious with SAE Features**
   - Features may not be reproducible
   - Verify stability before making claims

2. **Ground Truth When Available**
   - Use known structures for validation
   - Don't rely on reconstruction alone

3. **New Training Methods Needed**
   - Stability-promoting objectives
   - Better optimization procedures

---

## 🔧 Technical Details

### Custom SAE Implementation

We use a **custom PyTorch implementation** (no SAELens dependency) for:
- ✅ Full control over training dynamics
- ✅ Transparent implementation
- ✅ Easy modification for experiments
- ✅ Research reproducibility

**Key features:**
- Decoder normalization after every step (critical!)
- Auxiliary loss for dead neuron revival
- TopK and ReLU architectures
- Clean, documented code (~400 lines)

### System Requirements

- **Hardware:** MacBook Pro (CPU only, no GPU needed)
- **Memory:** <2GB per SAE
- **Time:** ~5 min per SAE training
- **OS:** macOS (with OpenMP workaround)

### Dependencies

```txt
torch>=2.0.0
transformer-lens>=1.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## 📚 Documentation

- **[RESEARCH_SUMMARY.md](RESEARCH_SUMMARY.md)** - Complete findings and methodology
- **[Phase 1 Results](docs/results/phase1_topk_stability.md)** - TopK stability analysis
- **[Phase 2 Results](docs/results/phase2_architecture_comparison.md)** - Architecture comparison
- **[SAE Guide](docs/02-Product/SAE_COMPREHENSIVE_GUIDE.md)** - Technical reference

---

## 🚧 Future Work

### Immediate Investigations

1. **Low Fourier Overlap**
   - Verify transformer learned Fourier circuits
   - Try different layers/positions
   - Longer training (40-50 epochs)

2. **Hyperparameter Sensitivity**
   - Learning rates
   - Expansion factors
   - Auxiliary loss coefficients

3. **Feature Analysis**
   - What features ARE being learned?
   - Semantic consistency across seeds?
   - Which features overlap/don't overlap?

### Longer-term Research

4. **Stability-Promoting Training**
   - Consistency losses
   - Feature alignment objectives
   - Two-stage training

5. **Alternative Approaches**
   - Different initialization
   - Learning rate schedules
   - Regularization techniques

6. **Scale to Real Models**
   - Apply to language models
   - Validate on complex tasks
   - Compare with literature

---

## 📄 Citation

If you use this code or findings in your research:

```bibtex
@misc{husai2024sae,
  title={SAE Feature Stability: Architecture-Independent Reproducibility Crisis},
  author={HUSAI Research Team},
  year={2024},
  url={https://github.com/brightlikethelight/HUSAI}
}
```

---

## 📞 Contact

- **Repository:** https://github.com/brightlikethelight/HUSAI
- **Issues:** https://github.com/brightlikethelight/HUSAI/issues

---

## ⚖️ License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Paulo & Belrose** - Baseline PWMCC methodology
- **Anthropic** - SAE architecture insights
- **TransformerLens** - Modular arithmetic implementation
- **Research Community** - Open science practices

---

**Last Updated:** November 3, 2025  
**Status:** Phase 1 & 2 Complete ✅  
**Next:** Investigating low Fourier overlap and stability-promoting training methods
