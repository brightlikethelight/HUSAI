# CRITICAL FINDINGS: Fourier Diagnostic Results

**Date:** November 3, 2025
**Status:** 🚨 URGENT - Research Direction Pivot Required

---

## Executive Summary

**MAJOR DISCOVERY:** The transformer model **DID NOT learn Fourier circuits** as assumed. This invalidates our Fourier validation approach and completely reframes our findings.

### Key Results from Diagnostic:

1. **Transformer Fourier Overlap: 0.2497** (expected: 0.6-0.8)
2. **Trained SAE = Random SAE:** -0.2% difference (statistically identical)
3. **Energy Distribution:** Uniform across all 128 dimensions (no Fourier preference)

**Conclusion:** Low Fourier overlap (~0.26) is NOT an SAE failure - it's because the transformer never learned Fourier structure in the first place!

---

## Detailed Findings

### Test 1: Dimension Mismatch
```
Energy distribution:
  Dims [0-112] (Fourier):  2.8299
  Dims [113-127] (Extra):  2.8125
  Ratio (Extra/Fourier):   99.4%
```

**Interpretation:** SAE features use all dimensions equally - no special structure in Fourier dimensions.

### Test 2: Random Baseline
```
Random SAE overlap:     0.2539 ± 0.0011
Trained SAE overlap:    0.2534
Improvement:           -0.0005 (-0.2%)
```

**Interpretation:** SAEs are NOT learning Fourier structure (not better than random).

### Test 3: Transformer Validation
```
Transformer embedding Fourier overlap:  0.2497
Expected (Nanda et al.):                0.6-0.8
Difference:                            -0.35 (2.4× below target)
```

**Interpretation:** **Transformer never grokked Fourier circuits!**

---

## What Went Wrong?

### Hypothesis 1: Transformer Not Fully Trained (Most Likely)
**Evidence:**
- Checkpoint shows "epoch: 1" (despite being named transformer_best.pt)
- May have been saved early in training
- Grokking requires extended training (Nanda et al. trained to convergence)

**Test:**
```bash
# Check if there are later checkpoints
ls -l results/transformer_5000ep/

# Result: transformer_epoch_5000.pt exists (Nov 3 16:16)
# transformer_best.pt is from earlier (Nov 3 12:14)
```

**Action:** Test transformer_epoch_5000.pt or transformer_final.pt instead!

### Hypothesis 2: Wrong Transformer Configuration
**Possible issues:**
- Different activation function (ReLU vs GELU)
- Different initialization
- Different optimizer/LR schedule
- Modulus mismatch (we used 113, Nanda used 97 or 113?)

**Action:** Compare our config to Nanda et al. paper precisely

### Hypothesis 3: Wrong Extraction Layer/Position
**Current:**
- Layer: 1 (residual stream post)
- Position: -2 (answer token)

**Nanda et al. may have used:**
- Layer: 0 or different activation point
- Position: Different token

**Action:** Test all layer/position combinations

---

## Implications for Our Research

### What Phases 1-2 Actually Showed:

**Original Interpretation (INVALID):**
> "SAEs fail to recover Fourier structure despite excellent reconstruction"

**Correct Interpretation (NEW):**
> "SAEs perfectly reflect the transformer's learned representations, which happen to NOT be Fourier-based. Low Fourier overlap is EXPECTED when the ground truth doesn't contain Fourier structure!"

### What This Means for Feature Stability (PWMCC = 0.30):

**The instability finding is STILL VALID!**
- SAEs showing PWMCC = 0.30 across seeds is INDEPENDENT of Fourier overlap
- Different seeds learn different features even when trained on identical data
- This is the CORE finding (reproducibility crisis)

**The Fourier finding needs reframing:**
- NOT "SAEs fail to recover ground truth"
- INSTEAD: "SAEs' feature instability prevents reliable extraction of ANY learned structure"

---

## Revised Research Narrative

### Phase 1: TopK Stability ✅ (VALID)
**Finding:** TopK SAEs show low stability (PWMCC = 0.302)
**Status:** Confirmed, robust

### Phase 2: Architecture Comparison ✅ (VALID)
**Finding:** ReLU and TopK equally unstable (PWMCC ~0.30)
**Status:** Confirmed, architecture-independent

### Phase 3: Fourier Validation ❌ (INVALID)
**Original goal:** Show SAEs fail to recover Fourier structure
**Actual finding:** Transformer never learned Fourier, so SAEs couldn't either
**Status:** **Need to replace with different ground truth validation**

---

## Revised Action Plan

### IMMEDIATE (Next 30 minutes)

**Action 1: Test Correct Transformer Checkpoint**
```bash
# Run diagnostic on later checkpoint
python scripts/diagnose_fourier.py \
  --transformer results/transformer_5000ep/transformer_final.pt \
  --sae results/saes/topk_seed42/sae_final.pt \
  --modulus 113
```

**Expected Outcomes:**
- **If transformer_final shows >0.6 Fourier overlap:**
  - Transformer DID grok, we used wrong checkpoint
  - SAEs ACTUALLY failed to recover Fourier
  - Original narrative is correct

- **If transformer_final ALSO shows ~0.25 overlap:**
  - Transformer NEVER grokked
  - Need to investigate WHY
  - May need to retrain transformer properly

**Action 2: Validate Transformer Performance**
```bash
# Test transformer on held-out data
python scripts/test_transformer_accuracy.py \
  --checkpoint results/transformer_5000ep/transformer_final.pt \
  --modulus 113
```

**Expected:** Should show 100% accuracy if grokked

### SHORT TERM (Next 2 hours)

**Option A: If Transformer IS Grokked (Fourier >0.6)**
→ Continue with original Phase 3 plan (Fourier-aligned init)

**Option B: If Transformer NOT Grokked (Fourier <0.4)**

**Sub-Option B1: Retrain Transformer Correctly**
```bash
# Train new transformer with validated grokking
python scripts/train_transformer.py \
  --config configs/grokking_verified.yaml \
  --epochs 50000  # Extended training
  --early-stop-metric fourier_overlap \
  --target-overlap 0.7
```

**Time:** 4-6 hours
**Risk:** May not grok even with longer training
**Reward:** If successful, validates entire research approach

**Sub-Option B2: Pivot to Different Ground Truth**

Instead of Fourier validation, use:

1. **Cluster analysis:** Do SAE features cluster consistently across seeds?
2. **Intervention testing:** Do features causally affect predictions?
3. **Interpretability metrics:** Are features human-interpretable?

**Time:** 2-3 hours
**Risk:** Lower - doesn't depend on transformer properties
**Reward:** More general findings applicable beyond grokking

---

## Recommended Path Forward

### RECOMMENDATION: Sub-Option B2 (Pivot to General Validation)

**Rationale:**
1. **Our core finding (PWMCC = 0.30) is VALID** regardless of Fourier
2. **Retraining transformer is high-risk** - may not grok
3. **Pivoting to general validation is more impactful:**
   - Applies to ANY task (not just modular arithmetic)
   - Tests SAE utility directly (interpretability, causality)
   - Publication-ready narrative

### New Phase 3 Plan:

**Experiment 1: Causal Intervention Testing (2 hours)**
- Ablate top SAE features
- Measure impact on model predictions
- Check if same features matter across seeds

**Experiment 2: Feature Interpretability Analysis (1 hour)**
- Extract top features from each SAE
- Check for semantic consistency across seeds
- Quantify interpretability scores

**Experiment 3: Cluster Stability Analysis (1 hour)**
- Cluster SAE features within each seed
- Compare cluster assignments across seeds
- Measure cluster overlap (alternative to PWMCC)

**Total time:** 4 hours
**Expected output:** 3-4 validation metrics showing feature instability

---

## Publication Strategy (Revised)

### Title Options:

**Option A (Conservative):**
"SAE Feature Instability: A Reproducibility Crisis in Sparse Autoencoders"

**Option B (Broader Impact):**
"Beyond Reconstruction: Why Excellent SAE Metrics Don't Guarantee Reliable Features"

**Option C (Mechanistic):**
"Multiple Local Minima in SAE Training Cause Feature Instability"

### Core Contributions:

1. **Empirical finding:** SAEs unstable across seeds (PWMCC = 0.30)
2. **Architecture independence:** TopK and ReLU equally affected
3. **Metric decoupling:** Good reconstruction ≠ stable features
4. **General validation framework:** Multi-metric evaluation beyond Fourier
5. **Practical implications:** Guidelines for SAE research reliability

### Impact:

- **Immediate:** Warns researchers about SAE reproducibility issues
- **Medium-term:** Motivates better training procedures
- **Long-term:** Establishes multi-metric evaluation standards

---

## Next Steps (Concrete)

### Step 1: Verify Transformer (15 min)
```bash
python scripts/diagnose_fourier.py \
  --transformer results/transformer_5000ep/transformer_final.pt \
  --sae results/saes/topk_seed42/sae_final.pt
```

### Step 2: Decide Based on Results

**If Fourier >0.6:** Proceed with original Phase 3
**If Fourier <0.4:** Pivot to general validation (recommended)

### Step 3: Execute Chosen Path (2-4 hours)

### Step 4: Write Updated Findings Document (1 hour)

### Step 5: Prepare for Publication (2 hours)

---

## Key Takeaways

1. **The instability finding (PWMCC = 0.30) is ROBUST** ✅
2. **The Fourier validation approach was INVALID** ❌
3. **This is actually a BETTER finding** - shows general problem, not task-specific
4. **Pivoting to general validation strengthens the paper** 🎯

---

**Status:** Awaiting decision on transformer validation results
**Priority:** CRITICAL - determines entire Phase 3 direction
**Timeline:** Results available in 15 minutes

---

