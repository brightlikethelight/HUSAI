# Sparse Ground Truth Validation: Unexpected Findings

**Date:** December 8, 2025
**Status:** CRITICAL - Theory contradiction discovered
**Experiment:** Extension 1 - Sparse Ground Truth Validation

---

## Executive Summary 🚨

We tested Cui et al. (2025)'s identifiability theory prediction that **sparse ground truth** should lead to **high SAE stability** (PWMCC > 0.70).

**RESULT: HYPOTHESIS REJECTED**
- Synthetic sparse ground truth (10/128 = 7.8% sparsity)
- **Observed PWMCC = 0.255** (LOWER than dense setup's 0.309!)
- **Theory predicted: PWMCC > 0.90**

This is the **opposite** of theoretical prediction and represents a **fundamental contradiction** that requires investigation.

---

## Experimental Results

### Experiment 1: Fourier Transformer (1-layer)
**Goal:** Train transformer known to learn sparse Fourier circuits

**Result:**
- ✅ Transformer grokked (100% accuracy after epoch 100)
- ❌ Did NOT learn Fourier circuits (R² = -252, should be > 0.90)
- Learned different algorithm (not sparse Fourier)

**Conclusion:** Cannot use for sparse validation (wrong algorithm)

---

### Experiment 2: Synthetic Sparse Data (Option B)
**Goal:** Perfect control over sparse ground truth

**Setup:**
```
Ground truth: 10 orthonormal features in d=128 space
Sparsity: 10/128 = 7.8% (EXTREMELY SPARSE)
Per-sample sparsity: L0 = 3 (30% of true features)
Effective rank: 9-10 (matches ground truth exactly)
Noise: 0.01 (minimal)

SAE configuration:
- d_sae = 64 (enough for 10 true features)
- k = 5 (matched to L0 = 3 sparsity)
- 5 seeds: 42, 123, 456, 789, 1011
- Epochs: 100
```

**Results:**
| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| **PWMCC** | **0.255 ± 0.008** | **> 0.90** | ❌ **FAILED** |
| Ground truth recovery | 0.6/10 features | ~10/10 | ❌ FAILED |
| Mean max similarity | 0.597 | > 0.90 | ❌ FAILED |
| Reconstruction loss | 0.035 | < 0.01 | ⚠️  Partial |

**Pairwise PWMCC:**
```
Seed 42 vs 123: 0.246
Seed 42 vs 456: 0.261
Seed 42 vs 789: 0.255
Seed 42 vs 1011: 0.252
Seed 123 vs 456: 0.263
Seed 123 vs 789: 0.246
Seed 123 vs 1011: 0.244
Seed 456 vs 789: 0.265
Seed 456 vs 1011: 0.264
Seed 789 vs 1011: 0.254

Mean: 0.255 ± 0.008
```

---

## Comparison to Dense Setup

| Setup | Ground Truth | Sparsity | Theory Predicts | Empirical PWMCC | Δ from Theory |
|-------|--------------|----------|-----------------|-----------------|---------------|
| **2-layer (dense)** | eff_rank=80/128 | 62.5% | PWMCC ≈ 0.30 | 0.309 ± 0.023 | +0.009 ✅ |
| **Synthetic (sparse)** | 10/128 features | 7.8% | PWMCC > 0.90 | 0.255 ± 0.008 | **-0.645** ❌ |
| **Random baseline** | N/A | N/A | PWMCC ≈ 0.30 | 0.300 ± 0.000 | Reference |

**Key finding:** Sparse setup LOWER than dense (0.255 vs 0.309) - opposite of prediction!

---

## Possible Explanations

### 1. SAE Hyperparameter Mismatch ⭐⭐⭐⭐
**Hypothesis:** d_sae=64 and k=5 don't match the sparse structure

**Evidence:**
- Poor ground truth recovery (0.6/10 features)
- Low mean similarity (0.597, should be > 0.90)
- SAEs learning 64 features when only 10 exist

**Test:**
- Retry with d_sae = 10 (exact match to true features)
- Retry with k = 3 (exact match to L0 sparsity)
- Expected: If this fixes it, PWMCC should jump to > 0.80

**Likelihood:** **HIGH** - This is the most plausible explanation

---

### 2. SAE Training Not Converged ⭐⭐⭐
**Hypothesis:** 100 epochs insufficient, loss = 0.035 too high

**Evidence:**
- Reconstruction loss: 0.035 (not near zero)
- Ground truth recovery poor
- May need 500-1000 epochs

**Test:**
- Train for 1000 epochs
- Monitor ground truth recovery over training
- Expected: Recovery should improve, PWMCC should increase

**Likelihood:** **MEDIUM** - Could contribute but unlikely to explain full gap

---

### 3. TopK SAE Fundamental Limitation ⭐⭐⭐⭐⭐
**Hypothesis:** TopK SAEs inherently unstable even with sparse ground truth

**Evidence:**
- Even with PERFECT sparse setup, still get PWMCC ≈ 0.25
- Theory applies to general sparse coding, not specifically TopK
- TopK creates hard selection (discrete optimization)
- Multiple optimal solutions exist (different k features can reconstruct equally well)

**Test:**
- Try ReLU SAE instead of TopK
- Try Gated SAE
- Expected: If TopK-specific, other architectures should work better

**Likelihood:** **VERY HIGH** - This could be a fundamental discovery

---

### 4. Identifiability Theory Assumptions Violated ⭐⭐
**Hypothesis:** Theory assumes continuous optimization, we have discrete (TopK)

**Evidence:**
- Cui et al. theory derived for ℓ0-penalized continuous optimization
- TopK is hard selection (non-differentiable)
- May break uniqueness guarantees

**Test:**
- Review Cui et al. paper carefully - does it apply to TopK?
- Try "soft TopK" (differentiable approximation)
- Expected: If theory doesn't apply to TopK, this explains everything

**Likelihood:** **HIGH** - Theory may not generalize to our setting

---

### 5. Optimization Landscape Issue ⭐⭐⭐
**Hypothesis:** Multiple local minima, each seed finds different one

**Evidence:**
- SAEs recover different subsets of true features
- All achieve similar reconstruction loss (0.035)
- Features form different bases that span similar subspace

**Test:**
- Visualize learned features vs ground truth
- Check if different SAEs span the same subspace (even if features differ)
- Measure subspace overlap, not just feature overlap

**Likelihood:** **MEDIUM-HIGH** - Consistent with observations

---

## Critical Questions to Investigate

### Question 1: Is this a TopK-specific problem?
**Experiment:** Train ReLU SAE on same synthetic data
- If ReLU gets PWMCC > 0.70: TopK is the problem
- If ReLU also gets PWMCC ≈ 0.25: More fundamental issue

### Question 2: Does d_sae = 10 (exact match) fix it?
**Experiment:** Retry with d_sae = 10, k = 3
- If PWMCC jumps to > 0.80: Hyperparameter mismatch
- If PWMCC stays ≈ 0.25: Deeper problem

### Question 3: Do SAEs learn the correct SUBSPACE?
**Experiment:** Measure subspace overlap, not just feature overlap
- Compute: rank(span(D1) ∩ span(D2))
- Check if D1, D2 span same 10D subspace even if features differ
- This would indicate "basis ambiguity" not "wrong features"

### Question 4: Does Cui et al. theory apply to TopK?
**Literature review:** Re-read Cui et al. (2025) carefully
- Check assumptions: continuous optimization? ℓ0 penalty vs hard TopK?
- May need to email authors for clarification

---

## Implications for Paper

### Scenario A: Hyperparameter Fix Works (PWMCC > 0.80 with d_sae=10, k=3)
**Narrative:**
- Identifiability theory VALIDATED, but with caveats
- Requires **precise hyperparameter matching** to ground truth
- Add section showing both failed (d_sae=64) and successful (d_sae=10) attempts
- Design principle: d_sae must match ground truth sparsity exactly

**Impact:** Still validates theory, adds important practical guidance

---

### Scenario B: TopK Fundamental Problem (ReLU works, TopK doesn't)
**Narrative:**
- Identifiability theory applies to continuous sparse coding
- TopK SAEs have inherent instability due to discrete selection
- First demonstration of architecture-specific stability properties
- Recommendation: Use ReLU or Gated SAEs for stable features

**Impact:** Major finding about SAE architecture choice

---

### Scenario C: Theory Doesn't Apply (Nothing works)
**Narrative:**
- Cui et al. theory may not generalize to neural network SAEs
- Theory assumes idealized sparse coding, real SAEs have:
  - Overparameterization (d_sae > true features)
  - Optimization difficulties (local minima)
  - Basis ambiguity (many equivalent solutions)
- SAE instability may be fundamental, not fixable by sparsity alone

**Impact:** **Paradigm shift** - challenges identifiability theory applicability

---

### Scenario D: Subspace Stability (Features differ but subspaces match)
**Narrative:**
- SAEs learn the correct SUBSPACE but choose different bases within it
- This is "basis ambiguity" not "feature instability"
- For interpretability, need basis-invariant methods
- PWMCC measures feature-level overlap, not subspace overlap

**Impact:** Reframes problem - stability at subspace level, instability at feature level

---

## Immediate Next Steps (Prioritized)

### PRIORITY 1: Test Exact Hyperparameter Match ⭐⭐⭐⭐⭐
**Why:** Fastest test, highest likelihood of explaining result

```bash
python scripts/synthetic_sparse_validation.py \
  --d-sae 10 \  # Exact match to 10 true features
  --k 3 \       # Exact match to L0 = 3
  --n-samples 50000 \
  --output-dir results/synthetic_sparse_exact
```

**Expected time:** 10 minutes
**Expected outcome:** If this fixes it, PWMCC should jump to > 0.80

---

### PRIORITY 2: Test ReLU SAE ⭐⭐⭐⭐
**Why:** Tests if TopK is the problem

```python
# Modify synthetic_sparse_validation.py to use ReLUSAE
from src.models.simple_sae import ReLUSAE

# Train ReLU SAE with L1 penalty
sae = ReLUSAE(d_model=128, d_sae=64, l1_coef=0.01)
# ... train and measure PWMCC
```

**Expected time:** 15 minutes
**Expected outcome:** If TopK is the problem, ReLU should get PWMCC > 0.70

---

### PRIORITY 3: Measure Subspace Overlap ⭐⭐⭐
**Why:** Tests "basis ambiguity" hypothesis

```python
def compute_subspace_overlap(D1, D2, k=10):
    """Measure if D1 and D2 span the same k-dimensional subspace."""
    # SVD to get top-k subspaces
    U1, S1, _ = torch.svd(D1)
    U2, S2, _ = torch.svd(D2)

    # Top-k principal components
    U1_k = U1[:, :k]  # [d_model, k]
    U2_k = U2[:, :k]  # [d_model, k]

    # Compute overlap: ||U1_k^T U2_k||_F^2 / k
    overlap = (U1_k.T @ U2_k).pow(2).sum() / k
    return overlap.item()

# overlap ≈ 1: Same subspace
# overlap ≈ 0: Different subspaces
```

**Expected time:** 5 minutes
**Expected outcome:** If > 0.90, SAEs learn same subspace with different bases

---

### PRIORITY 4: Literature Review ⭐⭐
**Why:** Understand if theory applies to our setting

**Tasks:**
- Re-read Cui et al. (2025) assumptions carefully
- Check if theory assumes continuous optimization
- Look for TopK SAE-specific identifiability results
- Consider emailing authors

**Expected time:** 2 hours
**Expected outcome:** Clarify if theory should apply

---

## Potential Research Outcomes

### Outcome 1: Validation with Caveats ✅
- Theory works but requires exact hyperparameter matching
- Add practical guidelines to paper
- Still validates Cui et al. in principle

### Outcome 2: Architecture-Specific Stability ✅✅
- TopK unstable, ReLU stable
- First characterization of architecture stability properties
- Major contribution beyond original plan

### Outcome 3: Basis Ambiguity Discovery ✅✅✅
- SAEs learn correct subspace, choose different bases
- Reframes interpretability problem
- Suggests new research direction: basis-invariant interpretability

### Outcome 4: Theory Limitation ✅✅✅✅
- Identifiability theory doesn't apply to neural SAEs
- **Paradigm shift** in how we think about SAE stability
- Extremely high-impact finding

---

## Files and Data

**Generated:**
- `results/sparse_ground_truth/` - Fourier transformer experiment (failed Fourier validation)
- `results/synthetic_sparse/` - Synthetic sparse experiment (PWMCC = 0.255)
  - `sae_seed_*.pt` - 5 trained SAEs
  - `results.json` - Complete results
- `SPARSE_GROUND_TRUTH_EXPERIMENT.md` - Experimental design doc
- `SPARSE_VALIDATION_FINDINGS.md` - This analysis

**To generate:**
- `results/synthetic_sparse_exact/` - Exact hyperparameter match test
- `results/synthetic_sparse_relu/` - ReLU SAE test
- Updated paper section with findings

---

## Timeline

**Completed (2-3 hours):**
- ✅ Fourier transformer experiment
- ✅ Synthetic sparse experiment
- ✅ Analysis and documentation

**Next (1-2 hours):**
- 🔄 Priority 1: Exact hyperparameter test
- 🔄 Priority 2: ReLU SAE test
- 🔄 Priority 3: Subspace overlap analysis

**Then (2-4 hours):**
- Update paper with findings
- Create visualizations
- Write discussion section

---

## Conclusion

This is a **critical and unexpected finding** that challenges our understanding of SAE stability.

**Three possible outcomes:**
1. **Technical fix** (hyperparameters) → Validates theory with caveats
2. **Architecture discovery** (TopK vs ReLU) → Novel contribution
3. **Theory limitation** (doesn't apply to SAEs) → Paradigm shift

All three are publishable and scientifically valuable. The investigation continues.

---

*Status: Active investigation*
*Next update: After Priority 1-3 tests complete*
*Expected resolution: Within 4-6 hours*
