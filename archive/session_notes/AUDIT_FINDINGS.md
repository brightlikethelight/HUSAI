# SAE Stability Research: Critical Audit Findings

**Date:** December 6, 2025  
**Status:** CRITICAL ISSUES IDENTIFIED

---

## Executive Summary

A comprehensive audit of the codebase, methodology, and findings revealed **two critical issues** that require immediate correction:

1. **Intervention Validation is NOT Causal** - The "unstable features are causal" claim is unsupported
2. **Feature Sensitivity Metric is Misleading** - It measures sparsity, not semantic sensitivity

---

## Issue 1: Intervention Validation is NOT Causal (CRITICAL)

### The Problem

The `intervention_validation.py` script claims to show that "unstable features have significant causal effects." However, the actual implementation uses a **proxy metric**, not causal ablation.

### What the Code Does

```python
# From intervention_validation.py, line 330-335
def perform_simplified_ablation(...):
    # Importance = mean magnitude * standard deviation
    importance_score = mean_activation * std_activation
    return importance_score
```

This computes `activation_magnitude × variance`, which is:
- ❌ NOT causal ablation
- ❌ NOT measuring downstream task effects
- ✅ Just measuring activation statistics

### Verification

I implemented **proper causal ablation** using TransformerLens hooks:

```python
# Proper causal ablation results:
Stable features:   Mean effect = 0.0000
Unstable features: Mean effect = 0.0000
```

**Both stable and unstable features show ZERO causal effect** when properly ablated.

### Why Zero Effect?

1. **Model has 100% accuracy** - No room for accuracy to drop
2. **Single feature ablation is negligible** - Ablating 1/1024 features changes activations by only 0.0016
3. **Features are sparse** - Most features active in <1% of samples

### Corrected Conclusion

**Original claim:** "Unstable features ARE causally important (p=0.008)"

**Corrected claim:** "Single-feature ablation shows no measurable causal effect for either stable or unstable features. The model is robust to individual feature ablation."

---

## Issue 2: Feature Sensitivity Metric is Misleading (CRITICAL)

### The Problem

The feature sensitivity analysis reported:
- TopK sensitivity: 0.059 (very low)
- ReLU sensitivity: 0.465 (moderate)

This was interpreted as "TopK features are input-specific" while "ReLU features generalize better."

### What the Metric Actually Measures

The sensitivity metric tests whether features that activate on input (a, b) also activate on "similar" inputs (a, b±δ).

**But in modular arithmetic:**
- (10 + 20) mod 113 = 30
- (10 + 21) mod 113 = 31

These are **semantically DIFFERENT**, not similar! Different answers should activate different features.

### Verification

```python
# TopK: (10,20) vs (10,21) feature overlap: 2/32 = 6.25%
# ReLU: (10,20) has 428 active, (10,21) has 402 active, overlap: 217
```

TopK has 6% overlap because it's **correctly** using different features for different answers.

ReLU has higher "sensitivity" because it activates **400+ features per input**, so there's ~50% overlap by chance.

### Corrected Conclusion

**Original claim:** "TopK has very low sensitivity (0.06), features are input-specific"

**Corrected claim:** "The sensitivity metric is not meaningful for modular arithmetic because 'similar' inputs have different answers. The apparent difference between TopK (0.06) and ReLU (0.47) reflects sparsity differences, not semantic sensitivity."

---

## Issue 3: Inconsistent PWMCC Implementations (MODERATE)

### The Problem

Multiple PWMCC implementations exist in the codebase:

| Script | Method | Symmetric? | Status |
|--------|--------|------------|--------|
| `src/analysis/feature_matching.py` | Decoder-based | Yes | ✅ Correct |
| `verify_random_baseline.py` | Decoder-based | Yes | ✅ Correct |
| `training_dynamics_analysis.py` | Decoder-based | Yes | ✅ Correct |
| `cross_layer_validation.py` | **Activation-based** | No | ❌ WRONG |
| `pwmcc_sensitivity_analysis.py` | Decoder-based | **No** | ⚠️ Inconsistent |

### Impact

- The Layer 0 anomaly (PWMCC = 0.047) was caused by `cross_layer_validation.py` using activation-based PWMCC
- This was correctly identified and fixed in the diagnosis
- Main results use the correct decoder-based method

### Recommendation

Fix `cross_layer_validation.py` to use decoder-based PWMCC for consistency.

---

## Findings That Remain Valid

Despite the issues above, the following findings are **still valid**:

### 1. PWMCC ≈ Random Baseline ✅

| Architecture | PWMCC | Random |
|--------------|-------|--------|
| TopK (20 epochs) | 0.302 | 0.300 |
| ReLU (20 epochs) | 0.300 | 0.300 |

This is measured using decoder-based PWMCC and is robust.

### 2. Training Dynamics: Features Converge ✅

| Epochs | PWMCC |
|--------|-------|
| 0 | 0.300 |
| 20 | 0.302 |
| 50 | 0.357 |

Longer training improves stability by ~19%.

### 3. Layer Independence ✅

| Layer | PWMCC |
|-------|-------|
| Layer 0 | 0.309 |
| Layer 1 | 0.302 |

Both layers show same PWMCC (after fixing the activation-based bug).

### 4. Reconstruction Quality ✅

| Architecture | MSE (Trained) | MSE (Random) | Improvement |
|--------------|---------------|--------------|-------------|
| TopK | 1.85 | 7.44 | 4.0× |
| ReLU | 2.15 | 17.95 | 8.4× |

SAEs achieve excellent reconstruction.

---

## Required Corrections

### 1. Paper Updates

Remove or heavily caveat:
- Section 4.8 "Intervention Validation: Instability ≠ Failure"
- Any claims about "unstable features being causal"
- Feature sensitivity comparison between TopK and ReLU

### 2. Documentation Updates

Update:
- `VALIDATION_COMPLETE_REPORT.md`
- `COMPREHENSIVE_SUMMARY.md`
- `TRAINING_DYNAMICS_FINDING.md`

### 3. Code Fixes

- Fix `cross_layer_validation.py` to use decoder-based PWMCC
- Add warnings to `intervention_validation.py` about proxy metric
- Add warnings to `measure_feature_sensitivity.py` about metric limitations

---

## Lessons Learned

1. **Proxy metrics can be misleading** - Always verify with proper causal methods
2. **Task structure matters** - Metrics designed for one task may not apply to another
3. **Code review is essential** - Multiple implementations can diverge
4. **Verify assumptions** - "Similar inputs" may not be similar in all tasks

---

*Generated: December 6, 2025*
