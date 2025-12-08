# Critical Review of SAE Stability Research Findings

**Date:** December 8, 2025  
**Status:** 🚨 CRITICAL - Some claims require correction

---

## Executive Summary

After comprehensive review of all experimental results, documentation, and claims, I found:

### ✅ VERIFIED Claims
1. **Stability decreases monotonically with sparsity** on algorithmic tasks
2. **Multi-architecture verification** (TopK, ReLU, Gated, JumpReLU all show same pattern)
3. **Overall correlation: -0.725** between L0 and stability
4. **Feature-level stability is uniform** (no predictor significantly predicts stability)

### ❌ FALSE/OVERSTATED Claims
1. **"Basis Ambiguity" Discovery** - The key claim is NOT supported by data

---

## Detailed Analysis

### 1. Basis Ambiguity Claim - REJECTED

**The Claim (from BASIS_AMBIGUITY_DISCOVERY.md):**
> "SAEs learn the correct SUBSPACE but choose different orthonormal BASES within it"
> "Subspace overlap should be HIGH (>0.90)"

**Actual Experimental Results:**

| Metric | Claimed | Actual |
|--------|---------|--------|
| PWMCC | ~0.26 (low) | 0.263 ± 0.029 ✅ |
| Subspace Overlap | >0.90 (high) | **0.139 ± 0.019** ❌ |

**Validation Script Output:**
```
Mean PWMCC (feature overlap):    0.263 +/- 0.029
Mean Subspace overlap:           0.139 +/- 0.019

HYPOTHESIS TEST:
REJECTED: Different Subspaces
  - Low feature overlap AND low subspace overlap
  - SAEs are learning genuinely different things
```

**Conclusion:** The "basis ambiguity" hypothesis is **REJECTED**. SAEs are NOT learning the same subspace with different bases - they are learning **genuinely different representations**.

### 2. Ground Truth Recovery Claim - MISLEADING

**The Claim:**
> "SAEs recover 88% of ground truth features (8.8/10, similarity = 1.28)"

**Issue:** The "mean similarity = 1.28" is suspicious. Cosine similarity should be in [-1, 1]. A value of 1.28 suggests either:
1. A bug in the similarity calculation
2. Using a different metric (not cosine similarity)
3. Summing similarities instead of averaging

This needs investigation before publication.

### 3. Verified Findings

#### Multi-Architecture Stability (VERIFIED ✅)
```
Overall correlation (L0 vs Stability): -0.725

TopK: corr=-0.917, L0=[8.0, 64.0], ratio=[1.13×, 1.56×]
ReLU: corr=-0.999, L0=[59.0, 65.1], ratio=[1.13×, 1.19×]
```

This finding is robust and architecture-independent.

#### Stability-Reconstruction Tradeoff (VERIFIED ✅)
- Underparameterized: High stability, poor reconstruction
- Matched regime: Balanced
- Overparameterized: Low stability, good reconstruction

---

## Recommendations

### Immediate Actions

1. **Remove or correct "Basis Ambiguity" claims** from:
   - `BASIS_AMBIGUITY_DISCOVERY.md`
   - `paper/sae_stability_paper.md` (if included)
   - Any other documentation

2. **Investigate the similarity > 1.0 issue** in ground truth recovery

3. **Update the narrative:**
   - OLD: "SAEs learn same subspace, different bases"
   - NEW: "SAEs learn genuinely different representations when ground truth is not sparse"

### For the Paper

**Keep:**
- Multi-architecture stability verification
- Stability-reconstruction tradeoff
- Task-dependent stability findings
- Feature-level stability uniformity

**Remove/Correct:**
- Basis ambiguity claims
- Any claims about subspace identifiability

### For LLM SAE Analysis

The new comprehensive Colab notebook (`notebooks/llm_sae_stability_comprehensive.ipynb`) is ready for use. It includes:
- Pretrained SAE analysis (Gemma Scope)
- Multi-seed training
- Stability vs sparsity experiment
- Proper visualization and comparison

---

## Files Created/Modified

| File | Status |
|------|--------|
| `scripts/experiments/validate_basis_ambiguity.py` | Created - validates (rejects) basis ambiguity |
| `notebooks/llm_sae_stability_comprehensive.ipynb` | Created - production-grade Colab notebook |
| `CRITICAL_REVIEW_FINDINGS.md` | Created - this document |

---

## Summary

The core findings about SAE stability are **valid and important**:
1. Stability decreases with sparsity on algorithmic tasks
2. This is architecture-independent
3. Feature-level stability is uniform

However, the "basis ambiguity discovery" is **NOT supported by the data** and should be removed from the paper before submission.

The research is still valuable - it just needs to be presented accurately without the unsupported "basis ambiguity" interpretation.
