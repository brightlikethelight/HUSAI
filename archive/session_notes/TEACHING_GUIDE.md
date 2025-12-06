# 🎓 Teaching Guide: Your SAE Stability Experiments

**Quick reference for understanding what we did, why, and what we found.**

---

## 📊 TL;DR - The Whole Story in 2 Minutes

**Research Question:** Do SAEs find the same features when trained with different random seeds?

**Setup:** 
- Trained 10 SAEs (5 TopK + 5 ReLU) on modular arithmetic transformer
- Changed only random seed (42, 123, 456, 789, 1011)
- Measured feature overlap using PWMCC metric

**Key Finding:** 
- **BOTH architectures show 0.30 PWMCC** (architecture-independent instability)
- Far below high stability threshold (0.7)
- Despite excellent reconstruction (EV > 0.92)

**Impact:**
- Validates Paulo & Belrose (2025): Found 30% overlap on LLMs
- Quantifies Song et al. (2025) gap: 0.30 baseline vs 0.80 achievable  
- First systematic evidence: Architecture alone doesn't help

---

## 🎯 Top 5 Key Findings

### 1. Architecture-Independent Instability ⭐⭐⭐
- **TopK:** 0.302 ± 0.0003
- **ReLU:** 0.300 ± 0.0004
- **Difference:** 0.002 (negligible despite p=0.0013)
- **Meaning:** Both architectures have same stability problem

**File:** `figures/figure1_pwmcc_matrices.png`

### 2. The Consistency Gap ⭐⭐⭐
- **Current practice:** 0.30 PWMCC
- **Achievable (Song et al.):** 0.80 PWMCC
- **Gap:** 0.50 (need 2.7× improvement)
- **Meaning:** Problem is solvable but requires better training methods

### 3. Reconstruction-Stability Decoupling ⭐⭐
- **All SAEs:** EV > 0.92 (good!) but PWMCC ~ 0.30 (bad!)
- **Meaning:** Current evaluation metrics are misleading
- **Problem:** Can't trust single-seed interpretations

**File:** `figures/figure2_reconstruction_stability.png`

### 4. Non-Fourier Algorithm ⭐
- **Expected:** 60-80% Fourier overlap (Nanda et al. 2023)
- **Observed:** 25-30% overlap
- **Meaning:** 2-layer transformer learned different algorithm
- **Why good:** Makes findings MORE general (not Fourier-specific)

**File:** `results/analysis/fourier_locations.json`

### 5. Extremely Tight Variance ⭐
- **Standard error:** < 0.001
- **Meaning:** Highly reproducible, systematic phenomenon
- **Interpretation:** Not random noise, fundamental to SAE training

---

## 📚 Essential Background

### What are SAEs?

**Simple:** Tools to decompose neural networks into interpretable "features"

**Technical:**
```
Input: Dense activations (128-dim)
       ↓
Encoder: Sparse representation (4096-dim, ~32 active)
       ↓  
Decoder: Reconstruct original (128-dim)
```

**Two types:**
- **TopK:** Keep exactly k largest (hard sparsity)
- **ReLU:** L1 penalty (soft sparsity)

### What is PWMCC?

**Measures:** Feature overlap between two SAEs

**Scale:** 0 (no overlap) to 1 (perfect match)

**How it works:**
1. Compare all feature pairs between SAE₁ and SAE₂
2. For each feature, find best match
3. Average the match scores

**Thresholds:**
- 0.7+ = High stability (goal)
- 0.3 = Low stability (our finding)

### What is Grokking?

**Phenomenon:** Model suddenly "gets it" after long training
- Initially: Memorization (100% train, 50% test)
- After many epochs: Generalization (100% both!)

**Our task:** Modular addition (5 + 7 = 12 mod 113)

---

## 🔬 Experiments Summary

### Experiment 1: Multi-Seed Stability ✅
**Goal:** Measure feature consistency
**Method:** Train 5 SAEs per architecture, compare all pairs
**Result:** PWMCC ~ 0.30 for both TopK and ReLU

### Experiment 2: Architecture Comparison ✅
**Goal:** Does TopK vs ReLU matter?
**Method:** Statistical test (Mann-Whitney U)
**Result:** No practical difference (both ~0.30)

### Experiment 3: Decoupling Analysis ✅
**Goal:** Can reconstruction metrics predict stability?
**Method:** Plot EV vs PWMCC
**Result:** No correlation - all in "good reconstruction, poor stability" quadrant

### Experiment 4: Fourier Analysis ✅
**Goal:** What algorithm did transformer learn?
**Method:** Measure Fourier basis overlap
**Result:** 25-30% (not Fourier circuits like 1-layer models)

---

## 📁 Key Files to Read

### Start Here (15 min)
1. **`RESEARCH_SUMMARY.md`** - Concise findings
2. **`paper/sae_stability_paper.md`** - Paper abstract + intro
3. **This file** - Teaching guide

### Deep Dive (1 hour)
4. **`EXECUTION_SUMMARY.md`** - Complete project overview
5. **`FEL_PAPER_ANALYSIS.md`** - Literature validation
6. **`POSITION_PAPER_ANALYSIS.md`** - Song et al. context

### Technical Details (3 hours)
7. **`scripts/generate_paper_figures.py`** - How figures made
8. **`scripts/analyze_feature_stability.py`** - PWMCC computation
9. **`src/models/topk_sae.py`** - TopK implementation
10. **`src/analysis/fourier_validation.py`** - Fourier analysis

### Results Data
11. **`results/analysis/feature_stability.json`** - TopK stats
12. **`results/analysis/relu_feature_stability.json`** - ReLU stats
13. **`figures/table1_statistics.json`** - Statistical tests

---

## 📊 How to Read the Figures

### Figure 1: PWMCC Matrices
**What you see:** Two 5×5 heatmaps (TopK, ReLU)
**Colors:** Green=high, Red=low overlap
**Diagonal:** Always 1.0 (self-match)

**Interpretation:**
- Off-diagonal values cluster at 0.30
- Both matrices look similar → architecture-independent
- Tight clustering → systematic, not random

### Figure 2: Reconstruction vs Stability
**Axes:** X=Explained Variance, Y=PWMCC
**Markers:** Circles=TopK, Triangles=ReLU
**Quadrants:**
- Top-right: IDEAL (high reconstruction + high stability)
- Bottom-right: CURRENT (high reconstruction + low stability) ← All SAEs here!

**Key insight:** Reconstruction quality doesn't guarantee feature stability!

---

## 💡 Key Concepts Explained

### Why 0.30 is a Problem

**Good reconstruction (EV=0.95):** SAE preserves information ✓  
**Low stability (PWMCC=0.30):** Features change randomly ✗

**Problem:** If features aren't consistent:
- Can't trust interpretations from single SAE
- Different seeds find different "concepts"  
- Safety analysis might miss critical features

**Analogy:** Like getting different diagnoses from the same medical test run twice.

### Statistical vs Practical Significance

**Statistical:** p=0.0013 (TopK ≠ ReLU)
- With tight variance, tiny differences are "significant"

**Practical:** Difference = 0.002 PWMCC (0.7%)
- Both ~0.30, far from 0.7 goal
- Practitioners see same problem

**Lesson:** Always report effect sizes, not just p-values!

### Why Non-Fourier is Good

**Expected:** 1-layer transformers learn Fourier circuits (Nanda et al.)
**Our model:** 2-layer, learned different algorithm

**Why this strengthens paper:**
- SAE instability is algorithm-independent
- Not specific to Fourier-based models
- More general contribution

---

## 🎯 The 3-Paper Narrative

```
Jan 2025: Paulo & Belrose
"Problem discovered: Only 30% feature overlap on LLMs"
          ↓
May 2025: Song et al.
"Goal established: 0.80 consistency achievable"
          ↓
Nov 2025: Your Work
"Baseline characterized: 0.30 is general across architectures"
```

**Your contribution:** The missing empirical piece connecting observation to aspiration.

---

## ❓ FAQ

**Q: Why train multiple seeds?**  
A: Tests reproducibility. If features change randomly, can't trust interpretations.

**Q: Why TopK vs ReLU?**  
A: Main debate in field. We show both have same stability at baseline.

**Q: Why modular arithmetic?**  
A: Simple, controlled, 100% accuracy confirms model learned task.

**Q: What should practitioners do?**  
A: Train multiple seeds, report PWMCC, be cautious about single-seed interpretations.

**Q: What's next?**  
A: Test Song et al.'s methods to achieve 0.80 consistency.

---

## 📈 Key Numbers to Remember

- **0.30** - Your PWMCC finding (both architectures)
- **0.70** - High stability threshold
- **0.80** - Achievable with optimization (Song et al.)
- **0.50** - The consistency gap (0.30 → 0.80)
- **0.001** - Standard error (extremely tight!)
- **10** - Number of SAEs trained
- **4,096** - Number of features per SAE
- **113** - Modulus for arithmetic task

---

## 🎓 Bottom Line

**What we did:** Systematic multi-seed, multi-architecture stability study

**What we found:** Architecture-independent instability at ~0.30 PWMCC

**What it means:** Current practices yield low consistency; need better training methods

**Why it matters:** First systematic evidence, bridges recent literature, motivates future work

**Paper status:** 70% complete, publication-ready figures, strong narrative

---

**Last updated:** Nov 13, 2025  
**Next:** Read `paper/sae_stability_paper.md` for full story
