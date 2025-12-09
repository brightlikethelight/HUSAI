# Corrected Numbers - Quick Reference Card

**Last Updated:** December 8, 2025 (after fixing normalization bug)

---

## Sparse Ground Truth Validation (CORRECTED)

### Ground Truth Recovery

| Metric | ❌ Buggy (Reported) | ✅ Corrected | Notes |
|--------|---------------------|--------------|-------|
| **Features recovered** | 8.8/10 (88%) | **0/10 (0%)** | Threshold >0.9 similarity |
| **Mean max similarity** | 1.284 | **0.390 ± 0.02** | Buggy value was >1.0 (impossible!) |
| **Reconstruction loss** | ~0.027 | ~0.027 | Unchanged (not affected by bug) |

**Per-seed (corrected):**
- Seed 42: 0/10, similarity = 0.366
- Seed 123: 0/10, similarity = 0.364
- Seed 456: 0/10, similarity = 0.415
- Seed 789: 0/10, similarity = 0.398
- Seed 1011: 0/10, similarity = 0.406

---

### Feature Stability (PWMCC)

| Setup | Ground Truth Sparsity | PWMCC (Mean ± Std) | Status |
|-------|----------------------|---------------------|--------|
| **Dense (2-layer)** | 62.5% (80/128) | 0.309 ± 0.002 | ✅ Matches theory |
| **Sparse (synthetic)** | 7.8% (10/128) | 0.270 ± 0.054 | ❌ Theory predicted >0.90 |
| **Random baseline** | N/A | ~0.30 | Reference |

**Conclusion:** Sparse ground truth does NOT improve stability (PWMCC unchanged)

---

### Subspace Overlap

| Comparison | Overlap | Interpretation |
|-----------|---------|----------------|
| **Top-9 dimensions** | 0.124 ± 0.016 | Nearly orthogonal |
| **Top-10 dimensions** | 0.139 ± 0.019 | Nearly orthogonal |
| **Predicted (basis ambiguity)** | >0.90 | ❌ REJECTED |

**Conclusion:** SAEs learn nearly orthogonal subspaces, NOT different bases of same subspace

---

### Singular Value Analysis

**All SAEs show:**
- Effective rank: **9D** (not 10D)
- σ₉ / σ₁₀ ratio: **2.5-5.9×** (mean: 4.0×)
- 90-95% variance explained by first 9 dimensions

| Seed | σ₉ | σ₁₀ | Drop |
|------|-----|-----|------|
| 42 | 0.723 | 0.173 | 4.2× |
| 123 | 0.728 | 0.208 | 3.5× |
| 456 | 0.764 | 0.130 | 5.9× |
| 789 | 0.713 | 0.289 | 2.5× |
| 1011 | 0.756 | 0.181 | 4.2× |

**Conclusion:** Weak 10th dimension is consistent, suggests 9D effective subspace

---

## Multi-Architecture Comparison (TopK Only Verified)

| Architecture | L0 Range | PWMCC Range | Correlation | Verdict |
|--------------|----------|-------------|-------------|---------|
| **TopK** | 8 - 64 (8×) | 0.28 - 0.39 | r = -0.917 | ✅ **Clear decreasing trend** |
| **ReLU** | 59 - 65 (1.1×) | 0.28 - 0.30 | r = -0.999 | ⚠️ Weak trend (narrow range) |
| **Gated** | 67.1 - 67.8 (1.01×) | 0.302 - 0.304 | r = +0.9998 | ❌ **No trend (constant L0)** |
| **JumpReLU** | 26.9 (constant) | 0.307 (constant) | N/A | ❌ No variation |

**Key correction:** "ALL architectures" → "TopK architecture" (only TopK has sufficient L0 variation)

---

## Dense Regime Validation (VERIFIED)

| Task | PWMCC | Effective Rank | Status |
|------|-------|----------------|--------|
| **Modular arithmetic** | 0.309 ± 0.002 | 80/128 (62.5%) | ✅ Matches theory |
| **Copy task** | 0.300 ± 0.004 | Not measured | ✅ Task-independent |
| **Theory prediction** | ~0.25-0.35 | Dense ground truth | ✅ Confirmed |

**Conclusion:** Cui et al. identifiability theory correctly predicts low stability for dense ground truth

---

## Training Dynamics (VERIFIED)

| Metric | Initial | Epoch 50 | Epoch 100 | Trend |
|--------|---------|----------|-----------|-------|
| **PWMCC** | ~0.30 | ~0.33 | ~0.36 | ✅ Converging (not diverging) |
| **Reconstruction loss** | ~0.60 | ~0.04 | ~0.03 | Decreasing |

**Conclusion:** Features converge toward random-like overlap, not diverge

---

## Critical Statistical Corrections

### Fixed Errors

| Original Claim | Corrected | Issue |
|----------------|-----------|-------|
| "0.309 ± 0.023" | **0.309 ± 0.002** | Std value not in data |
| "1.284 similarity" | **0.390 similarity** | >1.0 cosine impossible |
| "8.8/10 recovery" | **0/10 recovery** | Normalization bug |

### Standard Deviation Notes

- All experiments use **population std** (ddof=0), not sample std (ddof=1)
- Should use ddof=1 for n=5 samples (minor correction)
- Impact: ~10-20% increase in reported std values

---

## What Changed After Bug Fix

**The Bug (line 143):**
```python
# WRONG:
decoder = F.normalize(decoder, dim=1)  # Normalizes rows

# CORRECT:
decoder = F.normalize(decoder, dim=0)  # Normalizes columns
```

**Impact:**

| Metric | Change |
|--------|--------|
| Ground truth recovery | 88% → **0%** |
| Mean similarity | 1.28 → **0.39** (3.3× deflation) |
| Subspace overlap | Unchanged (0.14) |
| PWMCC | Unchanged (0.27) |

**Only ground truth recovery metric was affected by bug** - all other measurements were correct.

---

## Quick Lookup: Valid vs Invalid Claims

### ❌ REMOVE from Paper

- "88% ground truth recovery"
- "Similarity = 1.28"
- "Basis ambiguity phenomenon"
- "ALL architectures show decreasing trend"
- "Sparse ground truth improves stability"

### ✅ KEEP in Paper

- "PWMCC ≈ 0.30 (random baseline)"
- "Dense ground truth → low stability"
- "TopK stability decreases with L0"
- "Task-independent baseline"
- "Features converge during training"
- "Unstable features are causal"

### ➕ ADD to Paper

- "0/10 ground truth recovery"
- "Similarity = 0.39 (random-level)"
- "Sparse ground truth does NOT improve stability"
- "SAEs learn 9D subspaces (not 10D)"
- "TopK breaks identifiability theory"
- "Reconstruction ≠ ground truth recovery"

---

## Where to Find Full Details

| Topic | Document |
|-------|----------|
| **Complete paradox resolution** | `OPTION_B_RESOLUTION_COMPLETE.md` |
| **Claim-by-claim verification** | `FINAL_VERIFICATION_REPORT.md` |
| **Paper correction timeline** | `PUBLICATION_CHECKLIST.md` |
| **Session summary** | `SESSION_SUMMARY_DEC8.md` |
| **Diagnostic output** | `results/synthetic_sparse_exact/paradox_diagnosis_output.txt` |

---

**Last verified:** December 8, 2025, 10:30 PM
**Bug fix commit:** Line 143, `scripts/synthetic_sparse_validation.py`
**Rerun completed:** `results/synthetic_sparse_exact_corrected/`
