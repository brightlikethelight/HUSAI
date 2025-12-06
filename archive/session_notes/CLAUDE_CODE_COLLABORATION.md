# 🤝 Claude Code Collaboration: Critical Findings Synthesis

**Date:** December 5, 2025  
**Purpose:** Share critical findings from Cascade's ultrathink analysis with Claude Code's execution plan

---

## 🚨 CRITICAL FINDING CLAUDE CODE MUST KNOW

### Random Baseline = Trained SAEs!

I ran a random baseline experiment that Claude Code's plan doesn't account for:

```
Random SAE PWMCC:   0.3000 ± 0.0007 (45 pairwise comparisons)
Trained SAE PWMCC:  0.3000 ± 0.001  (10 pairwise comparisons)
Difference:         ~0.00002 (essentially ZERO)
```

**This fundamentally changes the interpretation of ALL findings!**

### Implications for Claude Code's Plan

1. **Layer 0 PWMCC = 0.047 is BELOW random (0.30)!**
   - This is actually the INTERESTING finding
   - Layer 0 shows LESS consistency than random chance
   - This suggests something is actively DESTROYING feature alignment at layer 0

2. **Layer 1 PWMCC = 0.302 is AT random baseline**
   - This means trained SAEs at layer 1 show NO improvement over random
   - The "instability" isn't low stability - it's ZERO stability above chance

3. **The "consistency gap" (0.30 vs 0.80) needs reframing**
   - Current practice: 0.30 = random baseline
   - Song et al. achievable: 0.80 = actually learned structure
   - The gap is between "random" and "learned", not "low" and "high"

---

## 📊 Alternative Metrics Show SOME Learning

I also tested alternative metrics that DO distinguish trained from random:

| Metric | Random | Trained | Diff | Significant? |
|--------|--------|---------|------|--------------|
| PWMCC | 0.299 | 0.302 | +0.003 | ❌ No |
| Mutual NN (>0.3) | 0.312 | 0.354 | **+0.042** | ✅ Yes |
| Subspace Overlap (k=50) | 0.386 | 0.439 | **+0.053** | ✅ Yes |

**Key insight:** PWMCC is a flawed metric for this comparison. Subspace overlap and mutual nearest neighbors show trained SAEs DO learn something!

---

## 🔧 Confirmed Bugs (Agree with Claude Code)

### Bug 1: Explained Variance Formula ✅ CONFIRMED

Location: `scripts/train_simple_sae.py:167-169`

```python
# Current (WRONG):
data_var = activations.var()  # Scalar across ALL dimensions
error_var = (activations - all_recon).var()
explained_var = 1 - (error_var / data_var)

# Correct:
# Per-dimension variance, then average
per_dim_var = activations.var(dim=0)  # [d_model]
per_dim_mse = ((activations - all_recon) ** 2).mean(dim=0)  # [d_model]
explained_var = 1 - (per_dim_mse / per_dim_var).mean()
```

### Bug 2: Paper Inaccuracies ✅ CONFIRMED

| Issue | Paper Says | Reality |
|-------|-----------|---------|
| Expansion factor | 32× | **8×** |
| d_sae | 4096 | **1024** |
| Cohen's d interpretation | "negligible" | **1.92 is LARGE** |
| Sparsity matching | "matched" | **TopK L0=32, ReLU L0=427** |

---

## 🎯 Revised Priority for Claude Code's Plan

Given the random baseline finding, I recommend reprioritizing:

### HIGHEST PRIORITY: Investigate Layer 0 Anomaly

**Why:** Layer 0 PWMCC (0.047) is BELOW random (0.30). This is the real finding!

Questions to answer:
- Why does Layer 0 show LESS consistency than random?
- Is there dimensional collapse at Layer 0?
- Are SAEs failing to train properly on Layer 0?

### HIGH PRIORITY: Fix EV Bug + Paper Inaccuracies

Claude Code is already working on this. Good.

### MEDIUM PRIORITY: Test Alternative Metrics

Since PWMCC = random baseline, we need metrics that actually distinguish trained from random:
- Subspace overlap (shows +5% improvement)
- Mutual nearest neighbors (shows +4% improvement)
- Activation-based stability (which inputs activate which features)

### LOWER PRIORITY: Increase Sample Size

If PWMCC = random baseline, adding more seeds won't change that. Focus on understanding WHY first.

---

## 📁 Files I Created

1. **`scripts/compute_random_baseline.py`** - Random baseline experiment
2. **`scripts/alternative_stability_metrics.py`** - Alternative metrics
3. **`CRITICAL_ANALYSIS.md`** - Full critical analysis
4. **`results/analysis/random_baseline.json`** - Random baseline results
5. **`results/analysis/alternative_metrics.json`** - Alternative metrics results

---

## 🔬 Suggested New Experiments

### Experiment 1: Layer 0 Deep Dive (HIGHEST PRIORITY)

```python
# Why is Layer 0 PWMCC (0.047) BELOW random (0.30)?

# Hypotheses:
# 1. Dimensional collapse - fewer effective dimensions
# 2. SAE training failure - poor reconstruction
# 3. Activation scale issues - very small/large values
# 4. True phenomenon - embeddings are fundamentally different

# Tests:
# - Compute effective rank of Layer 0 vs Layer 1 activations
# - Compare reconstruction quality (EV) at each layer
# - Visualize activation distributions
# - Check if random SAEs at Layer 0 also show 0.047 PWMCC
```

### Experiment 2: Random Baseline at Each Layer

```python
# Critical: Does random PWMCC vary by layer?

# If Layer 0 random PWMCC = 0.047, then trained = random (no learning)
# If Layer 0 random PWMCC = 0.30, then trained < random (WORSE than random!)

# This determines interpretation of Layer 0 finding
```

### Experiment 3: Activation-Based Stability

```python
# Instead of comparing decoder weights, compare:
# - Which inputs activate which features
# - Do matched features activate on same inputs?

# This might show learning even if PWMCC doesn't
```

---

## 📝 Recommended Paper Narrative

### Old Narrative (INCORRECT)
> "SAEs show low feature stability (PWMCC ≈ 0.30), far below the 0.70 threshold."

### New Narrative (CORRECT)
> "SAE feature consistency (PWMCC ≈ 0.30) is indistinguishable from randomly initialized SAEs, suggesting standard training produces no reproducible decoder structure. However, alternative metrics (subspace overlap, mutual nearest neighbors) show trained SAEs DO learn some structure. Interestingly, Layer 0 shows BELOW-random consistency (0.047), suggesting active interference with feature alignment at early layers."

---

## ✅ Action Items for Claude Code

1. **IMMEDIATE:** Share this document with Claude Code
2. **HIGH:** Run random baseline at Layer 0 to interpret the 0.047 finding
3. **HIGH:** Fix EV bug (already in progress)
4. **MEDIUM:** Add alternative metrics to analysis pipeline
5. **MEDIUM:** Fix paper inaccuracies (expansion factor, Cohen's d)
6. **LOWER:** Increase sample size (only after understanding random baseline)

---

## 💡 Key Insight

**The story isn't "SAEs are unstable" - it's "PWMCC doesn't capture what SAEs learn."**

Subspace overlap shows trained > random, meaning SAEs DO learn something. But PWMCC (which looks at individual feature matching) doesn't capture it.

This suggests SAEs learn the RIGHT SUBSPACE but not the RIGHT BASIS within that subspace. Different seeds find different bases for the same subspace.

**This is actually a more nuanced and interesting finding than "SAEs are unstable"!**

---

---

## 🔬 UPDATE: Layer 0 Anomaly RESOLVED

**The Layer 0 PWMCC = 0.047 was a COMPUTATION BUG, not a real phenomenon!**

### Root Cause
The `cross_layer_validation.py` used **activation-based PWMCC**, which fails for TopK SAEs:
- TopK SAEs activate only k=32 out of 1024 features per sample
- Each feature is zero for 96.9% of samples
- Normalizing mostly-zero vectors produces artificially low cosine similarities

### Corrected Results
Using **decoder-based PWMCC** (the correct method):

| Layer | Trained PWMCC | Random PWMCC | Interpretation |
|-------|---------------|--------------|----------------|
| Layer 0 | **0.309** | 0.30 | **AT random** |
| Layer 1 | 0.302 | 0.30 | **AT random** |

### Key Insight
**ALL layers show random-baseline stability.** There are NO layer-dependent effects.
SAE instability is universal across the transformer.

### Methodological Lesson
For TopK SAEs (or any sparse SAEs):
- ❌ **Don't use** activation-based PWMCC
- ✅ **Do use** decoder-based PWMCC (compares decoder weight columns directly)

See `LAYER0_DIAGNOSIS.md` for full investigation details.

---

**Last updated:** December 5, 2025  
**Status:** Ready for Claude Code integration
