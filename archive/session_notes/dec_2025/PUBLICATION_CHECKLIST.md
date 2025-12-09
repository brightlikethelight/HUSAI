# Publication Checklist: SAE Stability Research
## Quick Reference for Paper Submission

**Date:** December 8, 2025
**Source:** FINAL_VERIFICATION_REPORT.md (932 lines)
**Status:** 🚨 BLOCKERS IDENTIFIED - 4 hours to publication-ready

---

## TL;DR

### What's Wrong
- ❌ "Basis ambiguity" claims contradict data (must remove)
- ❌ Recovery metric shows values >1.0 (mathematically impossible - bug)
- ⚠️ Overclaiming identifiability theory validation

### What's Right
- ✅ 8 robust empirical findings (random baseline, paradox, multi-architecture, etc.)
- ✅ Strong core contribution
- ✅ Novel and important for interpretability community

### Time to Fix
**4 hours** of targeted revisions → ready for workshop submission

---

## CRITICAL BLOCKERS (Must Fix Before Submission)

### Blocker 1: Basis Ambiguity Claims ❌
**Time:** 2 hours

**Problem:**
- Claimed: "SAEs learn same subspace, different bases" (overlap >0.90 expected)
- Reality: Subspace overlap = 0.14 (nearly orthogonal, not rotated)
- **HYPOTHESIS REJECTED by data**

**Files to Fix:**
- [ ] Paper Section 4.11 (if exists) - DELETE entire section
- [ ] Paper Discussion - Remove all "basis ambiguity" mentions
- [ ] Paper Abstract - Remove if mentioned
- [ ] BASIS_AMBIGUITY_DISCOVERY.md - Add disclaimer at top

**Search for:**
```bash
grep -r "basis ambiguity" /Users/brightliu/School_Work/HUSAI/paper/
grep -r "same subspace" /Users/brightliu/School_Work/HUSAI/paper/
grep -r "rotated bases" /Users/brightliu/School_Work/HUSAI/paper/
```

---

### Blocker 2: Recovery Metric Bug ❌
**Time:** 1 hour

**Problem:**
- Mean similarity = 1.28 (impossible for cosine similarity ∈ [-1, 1])
- Creates unresolved mathematical paradox
- Cannot publish with contradictory metrics

**Action:**
```python
# Re-run ground truth recovery with verified calculation
# File: scripts/synthetic_sparse_validation.py or similar

# Verify:
1. Using cosine similarity (not dot product)
2. Values are in [-1, 1]
3. Not summing when should average
4. Not counting multiple matches per feature

# Expected fix: Either
- Bug in calculation → fix → recalculate
- Or different metric → clarify in paper what it is
```

**Files to Check:**
- [ ] scripts/synthetic_sparse_validation.py (lines with similarity calculation)
- [ ] Paper Section 4.11 or sparse results section
- [ ] SPARSE_VALIDATION_FINDINGS.md

---

### Blocker 3: Identifiability Overclaims ⚠️
**Time:** 1 hour

**Problem:**
- Paper claims: "First validation of Cui et al. identifiability theory"
- Reality: Dense case validates ✓, Sparse case contradicts ✗
- Only **partial validation**

**Action: Rewrite Section 4.10**

**OLD Title:** "Theoretical Grounding: Identifiability Theory Validation"

**NEW Title:** "Partial Agreement with Identifiability Theory"

**OLD Content:**
```
We validate Cui et al.'s identifiability theory...
Theory predicts PWMCC ≈ 0.30 for dense → Observed 0.309 ✓
Theory predicts PWMCC > 0.90 for sparse → [omitted or overstated]
```

**NEW Content:**
```
We test Cui et al.'s identifiability theory predictions:

DENSE ground truth (eff_rank=80/128):
- Theory predicts: PWMCC ≈ 0.25-0.35
- Observed: PWMCC = 0.309
- ✓ Excellent agreement

SPARSE ground truth (10/128 features):
- Theory predicts: PWMCC > 0.90
- Observed: PWMCC = 0.263
- ✗ Contradicts prediction

This partial validation suggests:
- Theory applies to dense regime
- Sparse regime may require different conditions (ReLU vs TopK,
  higher sparsity, longer training)
- Open question for future work
```

**Files to Fix:**
- [ ] Paper Section 4.10
- [ ] Abstract (if claims "validation")
- [ ] Introduction contributions list
- [ ] QUICK_ACTION_PLAN.md (update status)

---

## VERIFIED CLAIMS (Safe to Keep) ✅

### Core Findings (All Well-Supported)

1. ✅ **Random baseline phenomenon**
   - PWMCC = 0.309 (trained) vs 0.300 (random)
   - +0.009 improvement (3%, negligible)
   - **Keep this - it's the main contribution**

2. ✅ **Task independence**
   - Modular arithmetic: 0.309
   - Sequence copying: 0.300
   - **Keep - shows generality**

3. ✅ **Functional-representational paradox**
   - MSE 4-8× better than random
   - But features match random baseline
   - **Keep - striking finding**

4. ✅ **Architecture independence**
   - TopK: 0.302
   - ReLU: 0.300
   - Gated, JumpReLU: similar
   - **Keep - rare multi-architecture validation**

5. ✅ **Stability-reconstruction tradeoff**
   - Three regimes verified
   - Overall correlation: -0.725
   - **Keep - practical implications**

6. ✅ **Feature-level uniformity**
   - No predictor explains stability
   - **Keep - negative result but important**

7. ✅ **Training dynamics**
   - Features converge 0.30→0.36, don't diverge
   - **Keep - contradicts intuition**

8. ✅ **Task complexity correlation**
   - Stability correlates with transformer accuracy
   - **Keep - novel finding**

9. ✅ **Dense ground truth theory match**
   - Cui et al. predicts 0.25-0.35 → Observed 0.309
   - **Keep - partial validation**

---

## RECOMMENDED QUICK WINS (Optional, 5 min - 2 hours)

### Quick Win 1: 9D Subspace Test
**Time:** 5 minutes

**Action:**
```bash
python scripts/validate_subspace_overlap.py --k 9
```

**If overlap >0.70:**
- Include finding: "SAEs share 9D core, differ in 10th dimension"
- Explains weak 10th singular value
- Interesting nuance

**If overlap still low:**
- Confirms genuinely different representations
- Strengthens current narrative

---

### Quick Win 2: ReLU Sparse Test (Optional)
**Time:** 2 hours

**Action:**
```bash
# Train ReLU SAE on synthetic sparse data
python scripts/synthetic_sparse_validation.py \
  --architecture relu \
  --d-sae 10 \
  --k 3 \
  --output-dir results/synthetic_sparse_relu
```

**If ReLU PWMCC >0.70:**
- Major finding: "TopK-specific instability"
- Clear recommendation: Use ReLU for sparse tasks

**If ReLU also ≈0.30:**
- Confirms fundamental property, not architectural

---

## PAPER STRUCTURE (After Fixes)

### Keep These Sections ✅

- Abstract (minus basis ambiguity, fix identifiability claim)
- 1. Introduction
- 2. Related Work
- 3. Methods
- 4.1 Random Baseline Phenomenon ✅
- 4.2 Functional-Representational Paradox ✅
- 4.3 Architecture Independence ✅
- 4.4 Stability-Reconstruction Tradeoff ✅
- 4.5 Cross-Task Generalization ✅
- 4.6 Training Dynamics ✅
- 4.7 Feature-Level Analysis ✅
- 4.8 Effective Rank Study ✅
- 4.9 Task Complexity Correlation ✅
- 4.10 **Partial Agreement** with Identifiability Theory (REWRITE)
- 5. Discussion
- 6. Conclusion
- References
- Appendix (experimental details)

### Remove These Sections ❌

- 4.11 Basis Ambiguity Discovery (if exists) - DELETE
- Any "Subspace Identifiability" subsections - DELETE
- Sparse ground truth "validation" claims - REVISE to "preliminary/contradictory"

---

## SEARCH & REPLACE CHECKLIST

Run these searches in paper to find problematic claims:

```bash
cd /Users/brightliu/School_Work/HUSAI/paper

# Find basis ambiguity mentions
grep -n "basis ambiguity" sae_stability_paper.md
grep -n "same subspace" sae_stability_paper.md
grep -n "rotated" sae_stability_paper.md

# Find overclaimed validation
grep -n "first validation" sae_stability_paper.md
grep -n "definitive" sae_stability_paper.md
grep -n "validates.*identifiability" sae_stability_paper.md

# Find recovery metric mentions
grep -n "88%" sae_stability_paper.md
grep -n "8.8/10" sae_stability_paper.md
grep -n "1.28" sae_stability_paper.md
grep -n "similarity.*1\." sae_stability_paper.md
```

**Action for each match:**
- Basis ambiguity → DELETE
- "First validation" → Change to "partial agreement" or "tests"
- Recovery >1.0 → VERIFY or DELETE

---

## 4-HOUR TIMELINE

### Hour 1: Remove Basis Ambiguity
- [ ] Search paper for all mentions
- [ ] Delete Section 4.11 (if exists)
- [ ] Remove from abstract/intro/discussion
- [ ] Update narrative to "genuinely different representations"

### Hour 2: Fix Recovery Metric
- [ ] Re-run calculation with verified cosine similarity
- [ ] Confirm values ∈ [-1, 1]
- [ ] Update paper with corrected values
- [ ] Or remove if can't be fixed

### Hour 3: Rewrite Section 4.10
- [ ] Change title to "Partial Agreement"
- [ ] Clearly separate dense (✓) and sparse (✗) results
- [ ] Add discussion of possible explanations
- [ ] Frame sparse as "open question"

### Hour 4: Final Cleanup
- [ ] Update abstract to reflect changes
- [ ] Fix introduction contributions list
- [ ] Proofread revised sections
- [ ] Run all search queries to verify removals

**Result:** Paper ready for workshop submission

---

## PUBLICATION READINESS

### After 4-Hour Fixes

**Strength:** STRONG
- Random baseline phenomenon (novel, striking)
- Multi-architecture validation (rigorous)
- 8 verified empirical findings
- Honest about limitations

**Venues:**
- ✅ ICLR Workshop: 90% acceptance probability
- ✅ NeurIPS Workshop: 85% acceptance probability
- ⚠️ Main Conference: 40-50% (solid empirics, limited scope)
- ✅ arXiv: Always viable

**Recommended:** Submit to workshop first, get feedback, potentially extend for main conference

---

## CRITICAL WARNINGS

### DO NOT Submit Until:
- ❌ Basis ambiguity removed from paper
- ❌ Recovery metric verified or removed
- ❌ Identifiability claims toned down

### DO Submit When:
- ✅ All 3 blockers fixed (4 hours)
- ✅ Paper accurately represents data
- ✅ No mathematical contradictions
- ✅ Clear about what's verified vs uncertain

---

## CONTACT

**Questions about specific claims?** See FINAL_VERIFICATION_REPORT.md:
- Part 1: VERIFIED claims (lines 50-270)
- Part 2: REJECTED claims (lines 272-430)
- Part 3: UNCERTAIN claims (lines 432-540)
- Part 4: Paradoxes (lines 542-600)

**Need detailed evidence?** Each claim has:
- Source files
- Experimental results
- Verdict with reasoning

---

**Last Updated:** December 8, 2025
**Status:** Pre-publication verification complete, blockers identified
**Timeline:** 4 hours to publication-ready
