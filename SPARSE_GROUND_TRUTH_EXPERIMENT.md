# Sparse Ground Truth Validation Experiment

**Date:** December 7, 2025
**Status:** Running
**Extension:** 1 of 5 Novel Research Extensions

---

## Executive Summary

This experiment tests the **critical prediction** of Cui et al. (2025)'s identifiability theory:

> **Hypothesis:** When ground truth features are SPARSE, SAEs should achieve high stability (PWMCC > 0.70)

This would **definitively validate** the identifiability theory, completing our story:
- Dense ground truth → PWMCC ≈ 0.30 ✅ (already validated)
- Sparse ground truth → PWMCC > 0.70 ❓ (testing now)

---

## Background: Why This Experiment Matters

### Our Main Finding
- **Trained SAE PWMCC = 0.309** (essentially random baseline = 0.300)
- This was initially surprising - why do SAEs match random?

### Theoretical Explanation (Cui et al., 2025)
Cui et al. proved that SAEs can only learn **unique, stable features** when THREE conditions hold:

1. **Extreme sparsity of ground truth features** ← CRITICAL
2. Sparse SAE activation (k small)
3. Sufficient hidden dimensions (d_sae ≥ true features)

### Our Setup Analysis
| Condition | Requirement | Our 2-Layer Setup | Status |
|-----------|-------------|-------------------|--------|
| 1. Ground truth sparsity | Extremely sparse (<10%) | Dense (eff_rank=80/128=62.5%) | ❌ **VIOLATED** |
| 2. SAE sparsity | Small k | k=32 | ✅ Satisfied |
| 3. Dimensions | d_sae ≥ true features | d_sae=1024 >> 80 | ✅ Satisfied |

**Conclusion:** Condition 1 violation → Theory predicts PWMCC ≈ 0.30 → Observed 0.309 ✅

### The Missing Piece
We've shown that **dense** ground truth leads to low stability (0.30).
But we haven't shown that **sparse** ground truth leads to high stability (>0.70).

**This experiment completes the validation.**

---

## Experimental Design

### Approach: 1-Layer Fourier Transformer

**Why 1-layer?**
- Nanda et al. (2023) showed that 1-layer transformers on modular arithmetic learn **sparse Fourier circuits**
- Ground truth: ~10-20 key frequencies (highly sparse!)
- Expected R² > 0.90 (variance explained by Fourier basis)

**Why not 2-layer?**
- 2-layer transformers learn more complex, **dense** representations
- Our existing 2-layer transformer: R² < 0.30 (not Fourier, dense)

### Three Phases

#### Phase 1: Train 1-Layer Transformer
```python
Architecture:
  - 1 attention layer (not 2)
  - d_model = 128
  - n_heads = 4
  - MLP with d_ff = 512

Training:
  - Task: (a + b) mod 113
  - Epochs: 1000-5000 (until grokking)
  - Expected: 100% accuracy after grokking
```

#### Phase 2: Validate Fourier Structure
```python
Method: Nanda et al. (2023) - DFT on embeddings

Validation criteria:
  - R² > 0.90: ✅ Strong sparse Fourier structure
  - R² > 0.60: ⚠️  Partial structure (proceed with caution)
  - R² < 0.60: ❌ Failed (not suitable)

Expected:
  - Top frequencies: k=1, k=5 (or similar low freqs)
  - R² ≈ 0.93-0.98 (matching Nanda et al.)
```

#### Phase 3: Train SAEs and Measure PWMCC
```python
Configuration:
  - d_sae = 256 (matched to ~20-30 Fourier features)
  - k = 16 (matched to sparse structure)
  - Seeds: [42, 123, 456, 789, 1011]

Measurement:
  - Compute pairwise PWMCC across all 5 seeds
  - Compare to dense setup (0.309)

Expected outcomes:
  1. PWMCC > 0.70: ✅ Hypothesis confirmed
  2. PWMCC > 0.50: ⚠️  Partial validation
  3. PWMCC ≈ 0.30: ❌ Hypothesis rejected
```

---

## Expected Results Table

| Setup | Ground Truth Type | Effective Sparsity | Theory Predicts | Empirical Result |
|-------|-------------------|--------------------|-----------------|------------------|
| **2-layer (ours)** | Dense features | eff_rank=80/128 (62.5%) | PWMCC ≈ 0.30 | 0.309 ± 0.023 ✅ |
| **1-layer (test)** | Sparse Fourier | ~15 frequencies (<15%) | PWMCC > 0.70 | **Running...** |
| **Random baseline** | N/A (random) | N/A | PWMCC ≈ 0.30 | 0.300 ± 0.000 ✅ |

---

## Interpretation Guidelines

### Scenario 1: PWMCC > 0.70 ✅
**Result:** HYPOTHESIS CONFIRMED!

**Interpretation:**
- Sparse ground truth → High stability
- Dense ground truth → Low stability (random)
- **Definitive validation of identifiability theory**

**Impact:**
- Transforms paper from "empirical observation" to "theoretical validation"
- Provides clear design principle: **SAEs only work with sparse ground truth**
- Explains why LLMs get 65% shared features (may have sparser structure than our toy task)

**Next steps:**
- Add Section 4.11 or 5.X to paper with sparse validation results
- Update abstract: "We validate the theory in BOTH regimes (dense and sparse)"
- Strengthen conclusion with design guidelines

### Scenario 2: 0.50 < PWMCC < 0.70 ⚠️
**Result:** PARTIAL VALIDATION

**Possible reasons:**
1. Fourier structure not perfectly sparse (R² < 0.95)
2. SAE hyperparameters suboptimal (k too large, d_sae too small)
3. Effective rank still higher than theoretical Fourier features
4. Theory's threshold (>0.70) may be optimistic

**Interpretation:**
- Evidence that sparsity improves stability
- But not to the degree theory predicts
- May indicate need to refine theory or experiment

**Next steps:**
- Debug: Check R², effective rank, feature overlap
- Retry with stricter sparsity (smaller k, larger d_sae)
- Consider synthetic sparse data (Extension 1, Option B)

### Scenario 3: PWMCC ≈ 0.30 (no improvement) ❌
**Result:** HYPOTHESIS NOT CONFIRMED

**Possible reasons:**
1. Transformer failed to learn Fourier circuits (check R²)
2. SAE training failed (check reconstruction loss)
3. Identifiability theory may not apply to this setup
4. Our interpretation of "sparse ground truth" may be wrong

**Interpretation:**
- Unexpected result that challenges theory
- Need deep investigation to understand why
- May indicate fundamental limitation of approach

**Next steps:**
- Investigate transformer: Did it grok? What algorithm did it learn?
- Investigate SAEs: Are they learning anything meaningful?
- Consider alternative: Use **synthetic sparse data** with known ground truth
- Re-examine identifiability theory assumptions

---

## Connection to Paper

### Current Paper Status
**Section 4.10:** "Theoretical Grounding: Why PWMCC = 0.30"
- Explains Cui et al.'s three conditions
- Shows our setup violates Condition 1 (dense ground truth)
- Theory predicts PWMCC ≈ 0.30 → Observed 0.309 ✅

### Addition with This Experiment

**New Section 4.11:** "Validation in Sparse Regime"

```markdown
To definitively test identifiability theory, we trained a 1-layer transformer
known to learn sparse Fourier circuits (Nanda et al., 2023). This setup
SATISFIES Condition 1: ground truth has ~15 key frequencies (<15% sparsity).

[Table showing Fourier validation: R² = 0.XX]

We then trained SAEs on these sparse activations and measured PWMCC:

| Setup | Ground Truth | PWMCC | Interpretation |
|-------|--------------|-------|----------------|
| 2-layer (dense) | 80/128 features (62.5%) | 0.309 | Matches theory |
| 1-layer (sparse) | ~15 Fourier (13%) | 0.XXX | **Testing theory** |

[Interpretation based on result - see scenarios above]

This completes our empirical validation of Cui et al.'s identifiability theory,
demonstrating that SAE stability is fundamentally determined by ground truth
sparsity structure.
```

---

## Alternative if Fourier Fails: Synthetic Sparse Data

If the 1-layer transformer doesn't learn sufficiently sparse Fourier circuits,
we can use **Option B: Synthetic sparse data with known ground truth**.

### Synthetic Data Design
```python
# Generate data with KNOWN sparse ground truth
K = 10  # Number of true features
d = 128  # Ambient dimension
L0 = 3  # Sparsity per sample (only 3 features active)

# True features: K orthonormal directions
true_features = torch.randn(d, K)
true_features = F.normalize(true_features, dim=0)

# Generate samples: each uses 3 random features
for _ in range(n_samples):
    # Pick 3 random features
    active_features = random.sample(range(K), L0)
    coefficients = torch.randn(L0)

    # Linear combination + small noise
    activation = true_features[:, active_features] @ coefficients
    activation += 0.01 * torch.randn(d)  # Small noise
```

### Expected Result
- True sparsity: 10 features, L0=3 (30% per sample, but only 10 total)
- Theory predicts: PWMCC > 0.90 (extremely sparse)
- SAEs should recover the 10 true directions exactly

**Advantage:** Perfect control over ground truth sparsity
**Disadvantage:** Not a "real" task (less convincing for reviewers)

---

## Timeline

**Started:** December 7, 2025
**Expected duration:** 2-6 hours (depends on CPU speed and grokking time)

**Checkpoints:**
1. Transformer training (1-3 hours): Check if grokking occurs
2. Fourier validation (immediate): Check R²
3. SAE training (30-60 min per seed × 5): Train all SAEs
4. PWMCC computation (immediate): Get final result
5. Analysis and paper integration (1-2 hours)

---

## Success Criteria

**Minimum success:**
- Transformer groks (100% accuracy) ✅
- Fourier validation passes (R² > 0.60) ✅
- PWMCC improves over dense setup (>0.35) ✅

**Full success:**
- Transformer groks with strong Fourier (R² > 0.90) ✅
- PWMCC exceeds theoretical threshold (>0.70) ✅
- Clear separation from dense setup (>2× improvement) ✅

**Impact:**
- Minimum: Evidence that sparsity matters (good)
- Full: Definitive validation of theory (excellent, publishable)

---

## Files Generated

**During experiment:**
- `results/sparse_ground_truth/transformer/` - Transformer checkpoints
- `results/sparse_ground_truth/sae_seed_*.pt` - Trained SAEs
- `results/sparse_ground_truth/fourier_validation.json` - Fourier analysis
- `results/sparse_ground_truth/results.json` - Complete results
- `results/sparse_ground_truth/experiment_log.txt` - Full output log

**After analysis:**
- Updated `paper/sae_stability_paper.md` with Section 4.11
- Updated `COMPREHENSIVE_RESEARCH_SUMMARY.md`
- Updated `QUICK_ACTION_PLAN.md` with results

---

## References

**Identifiability Theory:**
- Cui, Y., et al. (2025). On the Theoretical Understanding of Identifiable Sparse Autoencoders and Beyond. *arXiv:2506.15963*.

**Fourier Circuits:**
- Nanda, N., et al. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR 2023*.

**Related Work:**
- Paulo, F., & Belrose, N. (2025). SAEs trained on same data learn different features. *arXiv:2501.16615*.
- Song, E., et al. (2025). Position: MI should prioritize feature consistency in SAEs. *arXiv:2505.20254*.

---

*Document created: December 7, 2025*
*Status: Experiment running*
*Expected completion: Within 6 hours*
