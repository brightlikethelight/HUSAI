# SAE Stability Research: Comprehensive Summary

**Date:** December 5, 2025  
**Status:** Phase 4.3 Complete, Phase 3 In Progress

---

## Executive Summary

This research investigates the stability of Sparse Autoencoder (SAE) features across random seeds. Our key findings fundamentally reframe the SAE stability problem.

### Core Findings

| Finding | Value | Significance |
|---------|-------|--------------|
| **Trained PWMCC (20 epochs)** | 0.302 ± 0.001 | Matches random baseline |
| **Random PWMCC** | 0.300 ± 0.001 | Baseline (chance) |
| **Trained PWMCC (50 epochs)** | 0.358 ± 0.006 | 20% above random |
| **Reconstruction** | 4-8× better than random | SAEs work functionally |
| **Layer independence** | Confirmed | All layers show same pattern |

### The Core Paradox

SAEs are **functionally successful** but **representationally unstable**:
- Reconstruction: Trained >> Random (4-8× better MSE)
- Feature matching: Trained ≈ Random (at 20 epochs)
- With longer training: Trained > Random by ~20%

---

## Key Discoveries

### 1. Training Duration Matters

| Epochs | PWMCC | vs Random |
|--------|-------|-----------|
| 0 | 0.300 | Baseline |
| 20 | 0.302 | +0.7% |
| 50 | 0.358 | +19.3% |

**Implication:** Standard training (20 epochs) produces near-random stability. Longer training (50 epochs) improves stability by ~20%.

### 2. Layer 0 "Anomaly" Was a Bug

- **Original finding:** Layer 0 PWMCC = 0.047 (6× below random)
- **Root cause:** Activation-based PWMCC fails for sparse TopK SAEs
- **Corrected finding:** Layer 0 PWMCC = 0.309 (same as random)
- **Lesson:** Use decoder-based PWMCC for sparse SAEs

### 3. Architecture Doesn't Matter

| Architecture | PWMCC | Difference |
|--------------|-------|------------|
| TopK | 0.302 | - |
| ReLU | 0.300 | -0.002 |

Cohen's d = 1.8 (large) but absolute difference is 0.002 (0.7%) - statistically significant but practically meaningless.

### 4. Alternative Metrics Show Learning

| Metric | Random | Trained | Improvement |
|--------|--------|---------|-------------|
| MSE Loss | 7.44 | 1.85 | **4.0× better** |
| MSE Loss (ReLU) | 17.95 | 2.15 | **8.4× better** |

SAEs DO learn - they just don't learn consistent features.

---

## Methodology

### Experimental Setup

- **Model:** 2-layer modular arithmetic transformer (d_model=128)
- **SAE:** 8× expansion (d_sae=1024), TopK (k=32) or ReLU (L1=1e-3)
- **Seeds:** 5 original (42, 123, 456, 789, 1011) + 2 new (2022, 2023)
- **Training:** 20-50 epochs, Adam optimizer, lr=3e-4

### Metrics

1. **PWMCC (Decoder-based):** Pairwise Maximum Cosine Correlation between decoder columns
2. **Explained Variance:** 1 - (residual_var / total_var)
3. **MSE Loss:** Mean squared reconstruction error
4. **L0 Sparsity:** Average number of active features

---

## Files and Scripts

### Key Documents

| File | Purpose |
|------|---------|
| `TRAINING_DYNAMICS_FINDING.md` | Training dynamics analysis |
| `LAYER0_DIAGNOSIS.md` | Layer 0 bug resolution |
| `FINDINGS_SYNTHESIS.md` | Overall findings synthesis |
| `paper/sae_stability_paper.md` | Draft paper |

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `scripts/analyze_training_dynamics.py` | Track PWMCC over training |
| `scripts/train_expanded_seeds.py` | Train additional SAEs |
| `scripts/comprehensive_statistical_analysis.py` | Full statistical analysis |
| `scripts/alternative_stability_metrics.py` | Alternative metrics |
| `scripts/diagnose_layer0.py` | Layer 0 bug diagnosis |

### Results

| File | Contents |
|------|----------|
| `results/training_dynamics/` | Training dynamics data and figures |
| `results/analysis/` | Statistical analysis results |
| `figures/` | Publication figures |

---

## Revised Narrative

### Old Narrative
> "SAE features have low stability (PWMCC ≈ 0.30), far below the 0.70 threshold."

### New Narrative
> "SAE features start at random baseline (0.30) and converge during training. After 20 epochs (standard training), PWMCC remains near random. After 50 epochs, PWMCC reaches 0.36 (~20% above random). Training DOES improve stability, but the improvement is modest and far below the 0.70 threshold. The fundamental issue is underconstrained reconstruction - many equally-good solutions exist."

---

## Remaining Work

### Phase 3: Expanded Seed Training
- **Status:** 2/10 new seeds trained
- **Remaining:** 8 TopK + 10 ReLU SAEs
- **Time:** ~25 hours

### Phase 4: Validation Experiments
- **4.1 Feature Sensitivity:** Not started
- **4.2 Task Generalization:** Not started
- **4.4 Intervention Validation:** Not started

### To Run

```bash
# Complete expanded seed training
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_expanded_seeds.py

# Run statistical analysis on expanded data
KMP_DUPLICATE_LIB_OK=TRUE python scripts/comprehensive_statistical_analysis.py
```

---

## Conclusions

1. **SAE training works** - reconstruction is 4-8× better than random
2. **Feature stability is low** - PWMCC ≈ 0.30-0.36 vs 0.70 target
3. **Longer training helps** - 50 epochs gives ~20% improvement over random
4. **Architecture doesn't matter** - TopK ≈ ReLU for stability
5. **Layer doesn't matter** - All layers show same pattern
6. **The problem is underconstrained reconstruction** - many solutions exist

---

*Generated: December 5, 2025*
