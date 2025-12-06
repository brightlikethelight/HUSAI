# SAE Stability Research: Second Audit Findings

**Date:** December 6, 2025  
**Status:** Comprehensive verification complete

---

## Executive Summary

A second comprehensive audit verified our core findings and added important context from recent literature. Key results:

1. **All findings verified** - PWMCC, training dynamics, layer independence all confirmed
2. **Hungarian matching confirms extreme instability** - 0% of features are "shared" (>0.5 similarity)
3. **Our results are MORE extreme than Paulo & Belrose (2025)** - They found ~65% shared, we found 0%
4. **Cross-regime analysis reveals nuance** - 50-epoch SAEs are similar to each other but not to 20-epoch SAEs

---

## Verification Results

### 1. Random Baseline Verification ✅

Theoretical analysis confirms PWMCC ≈ 0.30 is expected for random unit vectors:

| Metric | Value |
|--------|-------|
| Random PWMCC | 0.2986 ± 0.0009 |
| Trained PWMCC | 0.3017 ± 0.0010 |
| Difference | +0.0031 (1%) |

**Conclusion:** Trained SAEs are essentially at random baseline.

### 2. Per-Feature Stability Analysis ✅

| Metric | Value |
|--------|-------|
| Mean stability | 0.3017 |
| Max stability | 0.3660 |
| Features > 0.5 | 0 (0%) |
| Features > random | 51.6% (expected by chance) |

**Conclusion:** No features are highly stable. All features are at random baseline.

### 3. Hungarian Matching Analysis ✅

Using the methodology from Paulo & Belrose (2025):

| Metric | Our Results | Paulo & Belrose |
|--------|-------------|-----------------|
| Mean matched similarity | 0.29 | ~0.5-0.7 |
| % shared (>0.5) | **0%** | ~65% |
| % shared (>0.7) | **0%** | ~35% |

**Conclusion:** Our SAEs show MORE extreme instability than LLM SAEs.

### 4. Training Dynamics Verification ✅

| Epoch | PWMCC | Change |
|-------|-------|--------|
| 0 | 0.300 | Baseline |
| 20 | 0.320 | +7% |
| 50 | 0.358 | +19% |

**Conclusion:** Training improves within-regime stability, but cross-regime remains at random.

### 5. Cross-Regime Analysis (NEW)

| Comparison | PWMCC |
|------------|-------|
| Old vs Old (20 epochs) | 0.3017 |
| New vs New (50 epochs) | 0.3567 |
| Old vs New (cross) | 0.3103 |

**Key insight:** 50-epoch SAEs converge to a DIFFERENT solution space than 20-epoch SAEs. The "improvement" is regime-specific.

---

## Comparison with Paulo & Belrose (2025)

### Their Findings (on LLMs)
- ~65% of features shared across seeds (>0.5 similarity)
- ~35% "orphan" features found in only one SAE
- TopK more seed-dependent than ReLU
- Longer training increases overlap

### Our Findings (on modular arithmetic)
- **0%** of features shared (>0.5 similarity)
- **100%** "orphan" features
- TopK ≈ ReLU (both at random baseline)
- Longer training increases within-regime overlap

### Why the Difference?

1. **Task complexity:** Modular arithmetic is simpler, may have more equivalent solutions
2. **Model size:** Our transformer is much smaller (2 layers, 128 dims)
3. **SAE size:** 1024 features may be too many for the task
4. **Training data:** We use all possible inputs (12,769 samples)

---

## Implications

### 1. For Interpretability Research

Our results suggest that on simple, well-defined tasks:
- SAE features are **completely arbitrary**
- There is no "ground truth" decomposition to discover
- Different seeds find **entirely different** features

### 2. For SAE Methodology

- **PWMCC alone is insufficient** - Need Hungarian matching for precise analysis
- **Training duration matters** - But only within the same regime
- **Task structure affects stability** - Simple tasks may have more equivalent solutions

### 3. For Future Work

- **Investigate why our results are more extreme** than LLM results
- **Test on intermediate complexity tasks** to understand the transition
- **Develop stability-aware training objectives** (like Song et al., 2025)

---

## Valid Conclusions

| Finding | Status | Evidence |
|---------|--------|----------|
| PWMCC ≈ random baseline | ✅ Valid | Multiple verification methods |
| Training improves stability | ✅ Valid | Training dynamics analysis |
| Layer independence | ✅ Valid | Layer 0 = Layer 1 after bug fix |
| 0% shared features | ✅ Valid | Hungarian matching |
| Architecture doesn't matter | ✅ Valid | TopK ≈ ReLU |

## Corrected/Removed Claims

| Original Claim | Status | Reason |
|----------------|--------|--------|
| "Unstable features are causal" | ❌ Removed | Proxy metric, not causal |
| "TopK has low sensitivity" | ❌ Removed | Measures sparsity, not semantics |

---

## Best Next Steps

### High Priority

1. **Update paper** with Hungarian matching results and Paulo & Belrose comparison
2. **Investigate why our results are more extreme** than LLM results
3. **Test stability-aware training** (Song et al., 2025 approach)

### Medium Priority

4. **Analyze feature interpretability** - Are any features meaningful despite instability?
5. **Test on intermediate tasks** - Bridge between modular arithmetic and LLMs
6. **Implement multi-feature ablation** - Test causal importance of feature groups

### Low Priority

7. **SAGE framework** - Ground truth validation (complex, may not apply)
8. **Task generalization** - Test on sentiment analysis (requires new model)

---

*Generated: December 6, 2025*
