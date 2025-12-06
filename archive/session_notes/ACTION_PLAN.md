# Action Plan - Next Steps

**Date:** November 3, 2025, 5:45 PM  
**Current Status:** Phase 1 & 2 Complete ✅  
**Next Phase:** Phase 3 - Deep Investigation

---

## ✅ Completed Today

### Phase 1: TopK Stability Analysis
- [x] Trained 5 TopK SAEs with different seeds
- [x] Computed PWMCC stability matrix
- [x] Result: 0.302 ± 0.001 (LOW stability)
- [x] Documented in `docs/results/phase1_topk_stability.md`

### Phase 2: Architecture Comparison
- [x] Trained 5 ReLU SAEs with same seeds
- [x] Compared TopK vs ReLU stability
- [x] Result: Both show ~0.30 PWMCC (architecture-independent!)
- [x] Documented in `docs/results/phase2_architecture_comparison.md`

### Infrastructure & Documentation
- [x] Custom SAE implementation (no SAELens)
- [x] Complete training pipeline
- [x] PWMCC analysis tools
- [x] Fourier validation integrated
- [x] Comprehensive documentation
- [x] Repository organized and clean

---

## 🚨 Critical Issues to Investigate

### Issue 1: Low Fourier Overlap (~0.26 vs expected 0.6-0.8)

**Possible causes:**
1. Transformer didn't actually learn Fourier circuits
2. Wrong layer/position for extraction
3. SAE hyperparameters need tuning
4. Fourier basis computation error

**Immediate actions:**
```bash
# Check transformer Fourier structure directly
python scripts/analyze_transformer_fourier.py \
    --checkpoint results/transformer_5000ep/transformer_best.pt

# Try layer 0 instead of layer 1
python scripts/train_simple_sae.py \
    --transformer results/transformer_5000ep/transformer_best.pt \
    --layer 0 \
    --seed 42

# Try different position (all tokens, not just answer)
python scripts/train_simple_sae.py \
    --transformer results/transformer_5000ep/transformer_best.pt \
    --position all \
    --seed 42
```

### Issue 2: Low PWMCC Stability (~0.30)

**This is the main research finding, but we should:**
1. Verify it's not due to implementation bugs
2. Try stability-promoting training
3. Investigate what features ARE being learned

**Actions:**
```bash
# Longer training (maybe 20 epochs not enough)
python scripts/train_simple_sae.py \
    --transformer results/transformer_5000ep/transformer_best.pt \
    --epochs 40 \
    --seed 42

# Different learning rate
python scripts/train_simple_sae.py \
    --transformer results/transformer_5000ep/transformer_best.pt \
    --lr 1e-4 \
    --seed 42

# Larger expansion factor
python scripts/train_simple_sae.py \
    --transformer results/transformer_5000ep/transformer_best.pt \
    --expansion 16 \
    --seed 42
```

---

## 📋 Phase 3: Deep Investigation (Days 4-5)

### Priority 1: Validate Transformer Fourier Structure (HIGH IMPACT)

**Goal:** Confirm transformer actually learned Fourier circuits

**Tasks:**
1. Create `scripts/analyze_transformer_fourier.py`
2. Extract and analyze transformer embeddings
3. Check attention pattern structure
4. Verify against known Fourier basis
5. Document findings

**Expected outcome:** Either:
- ✅ Transformer HAS Fourier structure → SAE problem
- ❌ Transformer LACKS Fourier structure → Training problem

**Time estimate:** 2-3 hours

### Priority 2: Hyperparameter Sensitivity Analysis (MEDIUM IMPACT)

**Goal:** Determine if Fourier overlap can be improved

**Experiments:**
- Longer training: 30, 40, 50 epochs
- Learning rates: 1e-4, 5e-4, 1e-3
- Expansion factors: 4×, 16×, 32×
- Different layers: 0, 1, both
- Different positions: answer, all tokens

**Expected outcome:** 
- Best hyperparameters for Fourier recovery
- Understanding of what affects overlap

**Time estimate:** 4-6 hours (10-15 training runs)

### Priority 3: Feature Analysis (MEDIUM IMPACT)

**Goal:** Understand what features SAEs ARE learning

**Tasks:**
1. Create `scripts/analyze_features.py`
2. Visualize top features from each SAE
3. Check semantic consistency
4. Identify which features overlap across seeds
5. Analyze non-overlapping features

**Questions to answer:**
- Are features semantically meaningful?
- Do some features consistently appear?
- What causes variation?

**Time estimate:** 3-4 hours

### Priority 4: Stability-Promoting Training (HIGH IMPACT, LONGER-TERM)

**Goal:** Develop training procedures that improve stability

**Approaches:**
1. **Consistency loss:** Add loss term penalizing feature variation
2. **Two-stage training:** Train for reconstruction, then stabilize
3. **Feature alignment:** Explicitly match features during training
4. **Better initialization:** Use Fourier basis as initialization

**Implementation:**
```python
# Example: Add consistency loss
def train_with_consistency(sae, reference_sae, activations):
    reconstruction_loss = mse_loss(sae(act), act)
    consistency_loss = feature_alignment(sae.decoder, reference_sae.decoder)
    total_loss = reconstruction_loss + alpha * consistency_loss
```

**Time estimate:** 1-2 days (requires new training code)

---

## 📊 Experiments to Run

### Experiment Set 1: Fourier Investigation (URGENT)

**Hypothesis:** Transformer learned Fourier, but SAE hyperparameters wrong

```bash
# 1. Verify transformer Fourier structure
python scripts/analyze_transformer_fourier.py

# 2. Try layer 0
python scripts/train_simple_sae.py --layer 0 --seed 42

# 3. Try 40 epochs
python scripts/train_simple_sae.py --epochs 40 --seed 42

# 4. Try lower LR
python scripts/train_simple_sae.py --lr 1e-4 --seed 42

# 5. Try 16x expansion
python scripts/train_simple_sae.py --expansion 16 --seed 42
```

**Time:** ~2 hours total
**Expected result:** One or more show improved Fourier overlap

### Experiment Set 2: Multi-Seed Validation (IMPORTANT)

**Hypothesis:** Instability persists across hyperparameters

```bash
# For best hyperparameters from Set 1, train 5 seeds
for seed in 42 123 456 789 1011; do
    python scripts/train_simple_sae.py \
        --<best_params> \
        --seed $seed
done

# Analyze stability
python scripts/analyze_feature_stability.py \
    --sae-dir results/saes_optimized \
    --pattern "*/sae_final.pt"
```

**Time:** ~1 hour
**Expected result:** PWMCC still ~0.30 OR improved stability

### Experiment Set 3: Feature Analysis (EXPLORATORY)

**Hypothesis:** Some features are stable, others aren't

```bash
# Analyze which features overlap
python scripts/analyze_features.py \
    --sae1 results/saes/topk_seed42/sae_final.pt \
    --sae2 results/saes/topk_seed123/sae_final.pt \
    --visualize-top 20

# Check semantic consistency
python scripts/cluster_features.py \
    --sae-dir results/saes \
    --pattern "topk_seed*/sae_final.pt"
```

**Time:** ~2 hours (after script creation)
**Expected result:** Understanding of feature structure

---

## 🎯 Success Criteria

### For Phase 3

**Minimum (Must achieve):**
- [ ] Verified whether transformer has Fourier structure
- [ ] Tested 5+ hyperparameter configurations
- [ ] Documented findings in Phase 3 report

**Target (Should achieve):**
- [ ] Found hyperparameters that improve Fourier overlap >0.4
- [ ] Understood which features are stable/unstable
- [ ] Created feature analysis tools

**Stretch (Nice to have):**
- [ ] Implemented stability-promoting training
- [ ] Achieved PWMCC >0.5 with new methods
- [ ] Ready for publication draft

---

## 📅 Timeline

### Day 4 (Tomorrow)

**Morning (3 hours):**
- Create transformer Fourier analysis script
- Verify transformer structure
- Test layer 0 extraction

**Afternoon (3 hours):**
- Run hyperparameter sweep (5 experiments)
- Analyze results
- Document findings

**Evening:**
- Start feature analysis script
- Plan Day 5 experiments

### Day 5

**Morning (3 hours):**
- Complete feature analysis
- Visualize results
- Identify patterns

**Afternoon (3 hours):**
- Write Phase 3 report
- Update documentation
- Commit all work

**Evening:**
- Plan future research direction
- Prepare presentation/paper outline

---

## 🛠️ Scripts to Create

### 1. `scripts/analyze_transformer_fourier.py` (URGENT)

**Purpose:** Verify transformer Fourier structure

**Features:**
- Load trained transformer
- Extract embeddings and attention patterns
- Compute Fourier basis overlap
- Visualize results

**Time:** 1-2 hours

### 2. `scripts/analyze_features.py` (IMPORTANT)

**Purpose:** Understand SAE features

**Features:**
- Load multiple SAEs
- Compute feature similarities
- Identify stable/unstable features
- Visualize top features

**Time:** 2-3 hours

### 3. `scripts/train_sae_with_consistency.py` (FUTURE)

**Purpose:** Stability-promoting training

**Features:**
- Reference SAE for consistency loss
- Feature alignment loss
- Two-stage training option

**Time:** 4-6 hours

---

## 💭 Research Questions to Answer

### Critical Questions (Phase 3)

1. **Does the transformer actually have Fourier structure?**
   - If NO: Need to fix transformer training
   - If YES: Continue SAE investigation

2. **Can hyperparameters improve Fourier overlap?**
   - Try: longer training, different LR, bigger SAE
   - Target: Get overlap >0.4

3. **What features DO converge across seeds?**
   - Are ANY features stable?
   - What characterizes stable features?

4. **Why is sparsity level irrelevant?**
   - TopK (L0=32) and ReLU (L0=427) both unstable
   - What does this tell us?

### Longer-term Questions

5. **Can we design stability-promoting training?**
   - Consistency losses?
   - Better initialization?
   - Different optimization?

6. **Does this generalize to larger models?**
   - Test on real LLMs
   - Same instability?

7. **How does this compare to literature?**
   - Reproduce Paulo & Belrose exactly
   - Compare with other papers

---

## 📊 Expected Outcomes

### Best Case (30% probability)

- Find hyperparameters that improve Fourier overlap >0.5
- Identify architectural features that promote stability
- PWMCC improves to >0.5 with optimized training
- **Paper story:** "How to train stable SAEs"

### Realistic Case (50% probability)

- Confirm transformer has Fourier structure
- Fourier overlap improves modestly (0.3-0.4)
- PWMCC remains low (~0.30-0.35)
- **Paper story:** "SAE instability is fundamental, here's evidence"

### Worst Case (20% probability)

- Transformer doesn't have Fourier structure (training issue)
- No hyperparameters improve stability
- **Paper story:** "We identified a training issue and how to fix it"

**All outcomes are publishable!** Negative results are valuable.

---

## ✅ Quality Checks

Before moving to publication:

### Code Quality
- [ ] All scripts documented
- [ ] Tests for key functions
- [ ] Clean, readable code
- [ ] No hardcoded paths

### Reproducibility
- [ ] All experiments documented
- [ ] Seeds specified
- [ ] Hyperparameters recorded
- [ ] Results version controlled

### Documentation
- [ ] README complete
- [ ] Each phase documented
- [ ] Code comments clear
- [ ] Research summary updated

### Analysis
- [ ] Statistics reported correctly
- [ ] Visualizations clear
- [ ] Interpretations justified
- [ ] Alternative explanations considered

---

## 🚀 Long-term Vision

### Week 2-3: Core Research
- Complete Phase 3 investigation
- Implement stability improvements
- Write preliminary paper draft

### Week 4: Polish & Extend
- Create publication-quality figures
- Write full methodology section
- Prepare code release
- Documentation complete

### Month 2: Extension & Submission
- Test on larger models
- Compare with literature thoroughly
- Respond to reviewer feedback
- Submit to conference/journal

### Month 3+: Follow-up
- Implement community feedback
- Extend to new domains
- Collaborate with other researchers
- Open-source release

---

**Status:** Ready for Phase 3! 🚀  
**Next action:** Create `scripts/analyze_transformer_fourier.py`  
**Timeline:** Phase 3 completion by Day 5 (November 5)
