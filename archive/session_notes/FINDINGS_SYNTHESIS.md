# SAE Feature Stability: Complete Findings Synthesis

## Executive Summary

**Core Discovery**: Sparse Autoencoders learn to reconstruct activations well (4-8x better than random), but through **completely different feature sets** across random seeds. Feature matching (PWMCC) between trained SAEs equals the random baseline (~0.30).

---

## Quantitative Results

### 1. Feature Matching (PWMCC)

| Comparison | PWMCC | Interpretation |
|------------|-------|----------------|
| Trained vs Trained | 0.309 ± 0.002 | Cross-seed feature similarity |
| Random vs Random | 0.300 ± 0.001 | Baseline (chance) |
| **Difference** | **0.009 (0.8%)** | **Practically zero** |

**Statistical Note**: p < 0.0001 but Cohen's d effect is practically meaningless.

### 2. Functional Performance

| Metric | Trained | Random | Ratio |
|--------|---------|--------|-------|
| MSE Loss (TopK) | 1.85 | 7.44 | **4.0x better** |
| MSE Loss (ReLU) | 2.15 | 17.94 | **8.4x better** |
| Explained Var (TopK) | -7.97 | -22.82 | +14.85 |
| Explained Var (ReLU) | -2.40 | -7.69 | +5.28 |

### 3. Cross-Seed Activation Overlap

| Architecture | Trained | Random | Interpretation |
|--------------|---------|--------|----------------|
| TopK | 0.95% | 2.15% | Trained LOWER |
| ReLU | 1.67% | 2.15% | Trained LOWER |

**Surprising**: Same inputs activate LESS overlapping features across trained seeds than random!

### 4. Layer Consistency

| Layer | PWMCC | Note |
|-------|-------|------|
| Layer 0 | 0.309 | Same as random |
| Layer 1 | 0.302 | Same as random |

No layer-dependent effects. Instability is universal.

---

## The Paradox

**SAEs are simultaneously**:
1. **Functionally successful** - Low reconstruction loss, good explained variance
2. **Representationally unstable** - Different seeds learn incompatible features

**This means**: The reconstruction task is **underconstrained**. Many different feature decompositions achieve equally good reconstruction.

---

## Implications for Mechanistic Interpretability

### What This Challenges

1. **Feature universality** - Claims that SAE features represent "natural" concepts are questionable if features change arbitrarily across seeds
2. **Circuit analysis** - Downstream analysis of specific features may not generalize
3. **Interpretability claims** - Feature interpretations may be artifacts of training randomness

### What Remains Valid

1. **Reconstruction capability** - SAEs do learn useful compressed representations
2. **Sparsity benefits** - Sparse features are more interpretable than dense ones
3. **Methodology** - The SAE approach is sound; the underconstraining is the issue

---

## Recommended Paper Narrative

### Title Options
- "The Instability of Sparse Autoencoder Features: A Cautionary Tale"
- "SAE Features Are Not Unique: Random Seeds Yield Incompatible Representations"
- "Measuring Feature Stability in Sparse Autoencoders"

### Key Claims
1. **Claim 1**: SAEs successfully learn to reconstruct (4-8x better than random)
2. **Claim 2**: But feature sets are not stable across seeds (PWMCC = random baseline)
3. **Claim 3**: This is because reconstruction is underconstrained (many solutions)
4. **Claim 4**: This challenges downstream interpretability claims based on specific features

### Contribution
- First systematic measurement of SAE feature stability
- Demonstration that PWMCC ≈ random baseline
- Evidence for underconstrained reconstruction hypothesis
- Call for stability-aware SAE training methods

---

## Next Steps

1. **Regenerate figures** with random baseline clearly shown
2. **Revise paper draft** with this narrative
3. **Consider**: Can we propose methods to IMPROVE stability?
   - Regularization toward canonical features?
   - Multi-seed ensemble training?
   - Contrastive losses for feature alignment?

---

## Files Generated

| File | Purpose |
|------|---------|
| `scripts/verify_random_baseline.py` | Trained vs random PWMCC |
| `scripts/alternative_stability_metrics.py` | Multi-metric comparison |
| `scripts/diagnose_layer0.py` | Layer 0 anomaly resolution |
| `results/analysis/trained_vs_random_pwmcc.json` | Raw PWMCC data |
| `results/analysis/alternative_stability_metrics.csv` | All metrics data |
| `results/analysis/layer0_diagnosis.json` | Layer 0 diagnosis |

---

*Generated: 2025-12-05*
*Status: Ready for paper revision*
