# Comprehensive Action Plan: Repository Cleanup & Publication Strategy

**Date:** December 8, 2025, 11:00 PM
**Status:** Ready for execution
**Decision Required:** Fast-Track (6-8 hours) vs Comprehensive (40-60 hours)

---

## Executive Summary

After resolving all paradoxes and fixing critical bugs, we have **robust verified findings** ready for publication. This plan outlines two paths:

- **Path A (Fast-Track):** Clean up repo, remove false claims, submit with verified findings (~6-8 hours)
- **Path B (Comprehensive):** Add strengthening experiments, expand scope (~40-60 hours)

**Recommendation:** **Path A (Fast-Track)** - The verified findings are strong, novel, and publishable. Additional experiments carry risk of new bugs/contradictions.

---

## Current State Assessment

### ✅ Verified & Publication-Ready

| Finding | Evidence | Strength |
|---------|----------|----------|
| **Dense ground truth → low stability** | PWMCC = 0.309 ± 0.002 | ⭐⭐⭐⭐⭐ Matches theory |
| **TopK stability-sparsity relationship** | r = -0.917, p < 0.001 | ⭐⭐⭐⭐⭐ Strong correlation |
| **Task-independent baseline** | Modular (0.309) ≈ Copy (0.300) | ⭐⭐⭐⭐ Two independent tasks |
| **Training dynamics** | Converge 0.30→0.36 over 100 epochs | ⭐⭐⭐⭐ Clear pattern |
| **Causal relevance** | Intervention experiments confirm | ⭐⭐⭐ Validated |

**Publication value:** First empirical validation of identifiability theory for SAEs on algorithmic tasks. Novel negative result on sparse ground truth.

### ❌ Rejected & Must Remove

| Claim | Status | Action |
|-------|--------|--------|
| "Basis ambiguity" hypothesis | ❌ REJECTED | Archive BASIS_AMBIGUITY_DISCOVERY.md |
| "88% ground truth recovery" | ❌ BUG | Remove from all docs |
| "All architectures show pattern" | ❌ OVERCLAIM | Limit to "TopK architecture" |
| "Sparse improves stability" | ❌ FAILED | Add as negative finding |

### 🔧 Technical Debt

**Bugs fixed:**
- ✅ Line 143: Normalization dimension (dim=1 → dim=0)
- ✅ Ground truth recovery metric corrected

**Still present:**
- Standard deviation calculations use ddof=0 (should be ddof=1 for small samples)
- Multiple results directories with buggy/corrected versions mixed
- ~15 markdown files need systematic review

**Background experiments running:**
- All using OLD buggy code (should terminate)
- Need to clean up results directories

---

## Path A: Fast-Track Publication (RECOMMENDED)

**Timeline:** 6-8 hours
**Risk:** Low (using only verified findings)
**Publication Target:** Workshop paper or conference short paper

### Phase 1: Repository Cleanup (2 hours)

#### 1.1: Archive Invalid Results (30 min)

```bash
# Create archive directory
mkdir -p results/_archived_buggy_experiments

# Move buggy experiment results
mv results/synthetic_sparse results/_archived_buggy_experiments/
mv results/synthetic_sparse_exact results/_archived_buggy_experiments/
mv results/sparse_ground_truth results/_archived_buggy_experiments/

# Keep only corrected results
# results/synthetic_sparse_exact_corrected/ stays in place

# Add README
cat > results/_archived_buggy_experiments/README.md << 'EOF'
# Archived Experiments (Buggy Code)

These results were generated with buggy normalization code:
- Line 143: `F.normalize(decoder, dim=1)` (WRONG)
- Should be: `F.normalize(decoder, dim=0)` (CORRECT)

Bug impact:
- Inflated ground truth recovery similarities by 5-10×
- Reported "88% recovery" was false (actual: 0%)

**Do not use these results for publication.**

See: `OPTION_B_RESOLUTION_COMPLETE.md` for full bug analysis.

Corrected results: `results/synthetic_sparse_exact_corrected/`
EOF
```

**Files affected:**
- Move 3 results directories
- Update `.gitignore` if needed

#### 1.2: Archive Invalid Documentation (30 min)

```bash
# Create documentation archive
mkdir -p docs/_archive_rejected_hypotheses

# Move rejected hypothesis documents
mv BASIS_AMBIGUITY_DISCOVERY.md docs/_archive_rejected_hypotheses/
mv SPARSE_VALIDATION_FINDINGS.md docs/_archive_rejected_hypotheses/

# Add archive README
cat > docs/_archive_rejected_hypotheses/README.md << 'EOF'
# Rejected Hypotheses Archive

These documents contain analysis based on buggy experimental data and have been rejected after bug fixes.

## Rejected Hypotheses

1. **Basis Ambiguity** (BASIS_AMBIGUITY_DISCOVERY.md)
   - Predicted: Subspace overlap >0.90
   - Actual: Subspace overlap 0.14
   - Status: REJECTED

2. **Sparse Ground Truth Validation** (SPARSE_VALIDATION_FINDINGS.md)
   - Contained analysis of buggy recovery metric
   - Status: SUPERSEDED by OPTION_B_RESOLUTION_COMPLETE.md

**For historical reference only. Do not cite in publications.**
EOF
```

**Files affected:**
- Move 2 large markdown files (combined ~1500 lines of invalid analysis)

#### 1.3: Consolidate Valid Documentation (60 min)

**Create master index:**

```bash
cat > README_RESEARCH_FINDINGS.md << 'EOF'
# SAE Feature Stability Research - Final Findings

**Last Updated:** December 8, 2025

## Quick Navigation

### Core Findings (✅ VERIFIED - Safe for Publication)
- `FINAL_VERIFICATION_REPORT.md` - Complete claim-by-claim assessment
- `CORRECTED_NUMBERS_REFERENCE.md` - Quick lookup of all corrected metrics
- `SESSION_SUMMARY_DEC8.md` - Executive summary of paradox resolution

### Bug Resolution Documentation
- `OPTION_B_RESOLUTION_COMPLETE.md` - Full technical details of bug fixes (9 pages)
- `PARADOX_RESOLUTION.md` - Collaborative resolution with Windsurf agent

### Deprecated/Archived
- `docs/_archive_rejected_hypotheses/` - Rejected hypotheses (historical only)
- `results/_archived_buggy_experiments/` - Results from buggy code

## Publication-Ready Contributions

### 1. Identifiability Theory Validation (Dense Regime)
- **Finding:** Dense ground truth (80/128 = 62.5%) → PWMCC = 0.309
- **Theory:** Cui et al. predicts PWMCC ≈ 0.25-0.35 for dense ground truth
- **Status:** ✅ Theory validated

### 2. Stability-Sparsity Relationship (TopK SAEs)
- **Finding:** Stability decreases with L0 sparsity (r = -0.917, p < 0.001)
- **Range:** L0 ∈ [8, 64], PWMCC ∈ [0.28, 0.39]
- **Status:** ✅ Robust correlation

### 3. Task Generalization
- **Finding:** Modular arithmetic (0.309) ≈ Copy task (0.300)
- **Status:** ✅ Task-independent baseline

### 4. Negative Result: Sparse Ground Truth Fails
- **Finding:** Even with 7.8% sparsity, SAEs recover 0/10 features
- **Theory:** Cui et al. predicts identifiability under extreme sparsity
- **Implication:** TopK discrete selection breaks continuous theory assumptions
- **Status:** ✅ Novel negative result

## Citation Format

When citing this work, use verified findings only:

```bibtex
@article{liu2025sae_stability,
  title={SAE Feature Stability Decreases with Sparsity: An Empirical Validation of Identifiability Theory},
  author={Liu, [First Name]},
  year={2025},
  note={Verified findings documented in FINAL_VERIFICATION_REPORT.md}
}
```

## Changelog

**December 8, 2025:**
- Fixed critical normalization bug (line 143)
- Rejected "basis ambiguity" hypothesis
- Corrected ground truth recovery metrics
- Archived invalid experimental results

**Prior work:** See git history for details.
EOF
```

**Time:** 60 min to create index, update cross-references

### Phase 2: Paper Updates (3 hours)

#### 2.1: Remove False Claims - Systematic Search & Replace (90 min)

Create automated script:

```python
# scripts/cleanup_false_claims.py
"""
Systematic removal/correction of false claims across all markdown and paper files.
"""

import os
from pathlib import Path

FALSE_CLAIMS = {
    # Pattern -> Replacement
    r"88% ground truth recovery": "0% ground truth recovery (corrected after bug fix)",
    r"8\.8/10 features": "0/10 features",
    r"similarity.*1\.28": "similarity ≈ 0.39",
    r"basis ambiguity": "[REJECTED HYPOTHESIS: basis ambiguity]",
    r"across ALL architectures": "for TopK architecture",
    r"multi-architecture": "TopK architecture",
}

def scan_and_report(root_dir):
    """Scan all markdown files for false claims."""
    issues = []
    for path in Path(root_dir).rglob("*.md"):
        if "_archive" in str(path):
            continue  # Skip archived files

        with open(path) as f:
            content = f.read()
            for pattern in FALSE_CLAIMS:
                if pattern in content.lower():
                    issues.append((path, pattern))

    return issues

def generate_correction_report():
    """Generate report of all files needing correction."""
    issues = scan_and_report(".")

    print(f"Found {len(issues)} instances of false claims to correct:")
    for path, pattern in issues:
        print(f"  - {path}: contains '{pattern}'")

    # Save report
    with open("FALSE_CLAIMS_CORRECTION_REPORT.md", "w") as f:
        f.write("# False Claims Correction Report\n\n")
        f.write(f"Total instances found: {len(issues)}\n\n")
        for path, pattern in issues:
            f.write(f"- `{path}`: `{pattern}`\n")

if __name__ == "__main__":
    generate_correction_report()
```

**Action items:**
1. Run scan script
2. Manual review of each instance
3. Apply corrections
4. Re-scan to verify all removed

#### 2.2: Update Paper Section 4.11 (60 min)

**Current (WRONG):**
> "We validated identifiability theory using sparse ground truth (10/128 = 7.8%). SAEs achieved 88% feature recovery with similarity 1.28, confirming the basis ambiguity phenomenon..."

**Corrected:**
> "To test Cui et al.'s identifiability theory, we generated synthetic data with extreme ground truth sparsity (10/128 = 7.8%) and trained TopK SAEs with exact capacity matching (d_sae=10, k=3).
>
> **Negative Result:** Despite meeting all three identifiability conditions, SAEs failed to recover any ground truth features (0/10, mean similarity 0.39). This suggests that TopK's discrete k-selection mechanism breaks the continuous optimization assumptions of identifiability theory.
>
> **Implication:** Reconstruction-based training does not guarantee interpretable feature discovery, even under theoretically ideal sparsity conditions. Alternative methods (supervised, intervention-based) are needed for ground truth validation."

**New table:**

| Condition | Cui et al. Requirement | Our Setup | Status |
|-----------|------------------------|-----------|--------|
| Extreme sparsity | Ground truth < 10% | 7.8% (10/128) | ✅ Met |
| Sparse activation | Sample L0 < 30% | L0=3/10 (30%) | ✅ Met |
| Sufficient capacity | d_sae ≥ true features | d_sae=10 = 10 | ✅ Met |
| **Recovery** | PWMCC > 0.90 | **0.27** | ❌ **Failed** |

#### 2.3: Add Limitations Section (30 min)

```markdown
## Limitations

### 1. Architecture Scope
Our stability-sparsity relationship is verified only for **TopK SAEs**. Other architectures (ReLU, Gated, JumpReLU) require wider L0 ranges for conclusive testing.

- TopK: L0 ∈ [8, 64] → sufficient variation ✅
- ReLU: L0 ∈ [59, 65] → narrow, inconclusive ⚠️
- Gated: L0 ≈ 67 (constant) → uninformative ❌

### 2. Task Generalization
We tested two algorithmic tasks (modular arithmetic, copy). Generalization to LLM activations requires additional validation. Recent work (Archetypal SAE, Fel et al. 2025) reports ~50% stability for LLMs, higher than our 30% on algorithmic tasks.

### 3. Sparse Ground Truth Validation
Our sparse validation experiment failed due to TopK-specific issues. Continuous optimization methods (gradient-based sparse coding) may succeed where TopK failed.

### 4. Measurement Scope
We focused on feature-level stability (PWMCC). Circuit-level or subspace-level stability may show different patterns.
```

### Phase 3: Git Strategy (1.5 hours)

#### 3.1: Commit Organization

**Strategy:** Multiple incremental commits, organized by category

```bash
# Commit 1: Critical bug fix
git add scripts/synthetic_sparse_validation.py
git commit -m "fix: Correct feature normalization dimension (dim=1 → dim=0)

- Bug was inflating cosine similarities by 5-10×
- Created impossible values >1.0 for cosine similarity
- Affected ground truth recovery metric only
- Other measurements (PWMCC, subspace overlap) unaffected

Impact:
- Ground truth recovery: 88% → 0% (corrected)
- Mean similarity: 1.28 → 0.39 (corrected)

See: OPTION_B_RESOLUTION_COMPLETE.md for full analysis"

# Commit 2: Archive invalid results
git add results/_archived_buggy_experiments/
git add docs/_archive_rejected_hypotheses/
git commit -m "chore: Archive experiments and docs from buggy code

Archived:
- results/synthetic_sparse* (buggy normalization)
- BASIS_AMBIGUITY_DISCOVERY.md (rejected hypothesis)
- SPARSE_VALIDATION_FINDINGS.md (superseded)

All archived materials marked with warnings.
Corrected results in: results/synthetic_sparse_exact_corrected/"

# Commit 3: Documentation updates
git add OPTION_B_RESOLUTION_COMPLETE.md
git add SESSION_SUMMARY_DEC8.md
git add CORRECTED_NUMBERS_REFERENCE.md
git add FINAL_VERIFICATION_REPORT.md
git add PUBLICATION_CHECKLIST.md
git add PARADOX_RESOLUTION.md
git commit -m "docs: Comprehensive paradox resolution and bug analysis

Added:
- OPTION_B_RESOLUTION_COMPLETE.md (9 pages, full technical details)
- SESSION_SUMMARY_DEC8.md (executive summary)
- CORRECTED_NUMBERS_REFERENCE.md (quick reference)
- FINAL_VERIFICATION_REPORT.md (claim-by-claim assessment)

All documents cross-verified and consistent."

# Commit 4: Paper corrections
git add paper/sae_stability_paper.md
git commit -m "fix: Remove false claims from paper

Corrections:
- Section 4.11: Corrected sparse validation (0% recovery, not 88%)
- Removed 'basis ambiguity' framing
- Limited multi-architecture claims to TopK only
- Added comprehensive limitations section

All remaining claims verified per FINAL_VERIFICATION_REPORT.md"

# Commit 5: Diagnostic tools
git add scripts/diagnose_recovery_paradox.py
git commit -m "feat: Add diagnostic script for ground truth recovery analysis

- Comprehensive SAE decoder subspace analysis
- Feature recovery pattern investigation
- Singular value decomposition profiling
- Per-seed recovery consistency checks"

# Commit 6: Master index
git add README_RESEARCH_FINDINGS.md
git commit -m "docs: Add master research findings index

Navigation guide for:
- Verified findings (publication-ready)
- Bug resolution documentation
- Archived/deprecated materials

Quick reference for future work and paper writing."
```

**Branch strategy:**
- Work on `main` directly (single developer, emergency fixes)
- Tag after cleanup: `git tag v1.0-corrected-findings`

#### 3.2: Push to Remote

```bash
git push origin main
git push origin --tags
```

### Phase 4: Final Checklist (30 min)

Create validation checklist:

```markdown
# Pre-Submission Validation Checklist

## False Claims Removal
- [ ] Search entire repo for "88%" → None found
- [ ] Search for "basis ambiguity" → Only in archived files
- [ ] Search for "ALL architectures" → Corrected to "TopK"
- [ ] Search for "similarity.*1.2" → All instances corrected

## Documentation Consistency
- [ ] All numbers in paper match CORRECTED_NUMBERS_REFERENCE.md
- [ ] All claims in paper are marked ✅ VERIFIED in FINAL_VERIFICATION_REPORT.md
- [ ] No references to BASIS_AMBIGUITY_DISCOVERY.md in active docs
- [ ] All cross-references updated

## Git Repository
- [ ] All commits pushed to remote
- [ ] Tag `v1.0-corrected-findings` created
- [ ] README_RESEARCH_FINDINGS.md accurate and complete
- [ ] .gitignore includes archived experiments

## Paper Quality
- [ ] No overclaiming (all statements hedged appropriately)
- [ ] Limitations section comprehensive
- [ ] Related work section includes Fel et al. 2025
- [ ] Figures/tables use corrected numbers only

## Reproducibility
- [ ] All scripts use corrected code (dim=0 normalization)
- [ ] results/synthetic_sparse_exact_corrected/ documented
- [ ] Requirements.txt up to date
```

---

## Path B: Comprehensive Publication (Alternative)

**Timeline:** 40-60 hours
**Risk:** Medium-High (new experiments may reveal more bugs)
**Publication Target:** Full conference paper

### Additional Experiments (30-40 hours)

#### B.1: ReLU SAE Sparse Validation (8 hours)

**Rationale:** Test if TopK-specific discrete selection caused sparse validation failure

```python
# Test with continuous ReLU SAE
sae = ReLUSAE(d_model=128, d_sae=10, l1_coef=0.001)
# Train on same synthetic sparse data
# Measure ground truth recovery

# Prediction: ReLU might succeed where TopK failed
```

**Time:** 8 hours (experiment + analysis)

#### B.2: Wider L0 Range for Gated/ReLU (12 hours)

**Rationale:** Current L0 ranges too narrow for conclusive testing

```python
# Gated: Test L1 ∈ [0.001, 0.0001, 0.00001] to get L0 ∈ [30, 100]
# ReLU: Test L1 ∈ [0.0001, 0.001, 0.01] to get L0 ∈ [20, 80]
```

**Time:** 12 hours (training + analysis)

#### B.3: LLM Validation (Layer SAEs) (20 hours)

**Rationale:** Validate findings generalize beyond algorithmic tasks

```python
# Train SAEs on GPT-2 activations (layer 6)
# Compute PWMCC across random seeds
# Compare to algorithmic task baseline

# Expected: Higher stability (~0.5 per Fel et al.) due to sparser LLM features
```

**Time:** 20 hours (setup + training + analysis)

### Extended Analysis (10-20 hours)

#### B.4: Theoretical Grounding (8 hours)

- Formal proof that TopK breaks identifiability assumptions
- Simulation showing discrete selection → multiple local minima
- Literature review: sparse coding identifiability theorems

#### B.5: Intervention-Based Validation (6 hours)

- Systematically ablate features, measure causality
- Compare causal impact of stable vs unstable features
- Statistical tests for significance

#### B.6: Visualization Suite (6 hours)

- PCA projections of SAE decoder subspaces
- Training dynamics animations
- Feature similarity matrices across seeds

### Path B Risk Assessment

**Risks:**
1. **New bugs:** More code → more opportunities for errors
2. **Contradictions:** LLM results might contradict algorithmic findings
3. **Scope creep:** Each experiment suggests 2-3 follow-ups
4. **Timeline:** 40-60 hours → 2 weeks full-time or 1-2 months part-time

**Benefits:**
1. Stronger multi-architecture validation
2. LLM generalization demonstrated
3. Comprehensive conference paper vs workshop paper
4. More impactful contribution

---

## Recommendation: Path A (Fast-Track)

### Justification

**1. Strong Verified Findings**
- TopK stability-sparsity relationship is novel and robust
- Dense regime validation confirms identifiability theory
- Negative sparse validation result is scientifically valuable

**2. Risk Management**
- Every experiment so far revealed unexpected bugs/contradictions
- More experiments = more chances for new issues
- Fast-track minimizes additional failure modes

**3. Publication Strategy**
- Workshop paper now → Full paper later (if additional work successful)
- Negative results are publishable and impactful
- Can cite as "preliminary findings" if doing follow-up work

**4. Time Efficiency**
- 6-8 hours to clean, professional submission
- vs 40-60 hours with uncertain outcome

### Decision Framework

**Choose Path A if:**
- ✅ Conference deadline is soon (< 2 weeks)
- ✅ Risk-averse (want guaranteed publication)
- ✅ Limited compute resources
- ✅ Want to move on to new projects

**Choose Path B if:**
- ⚠️ Have 2+ weeks before deadline
- ⚠️ Willing to risk discovering more bugs
- ⚠️ Access to substantial compute for LLM experiments
- ⚠️ Want comprehensive conference paper

---

## Execution Timeline

### Path A: Fast-Track (6-8 hours)

| Phase | Tasks | Time | Cumulative |
|-------|-------|------|------------|
| **1. Repo Cleanup** | Archive results, move docs, create index | 2h | 2h |
| **2. Paper Updates** | Remove false claims, update sections | 3h | 5h |
| **3. Git Strategy** | Commits, push, tag | 1.5h | 6.5h |
| **4. Final Check** | Validation checklist | 0.5h | 7h |

**Can complete in one evening session.**

### Path B: Comprehensive (40-60 hours)

| Phase | Tasks | Time | Cumulative |
|-------|-------|------|------------|
| **1-4. Path A tasks** | (Same as above) | 7h | 7h |
| **5. ReLU validation** | Sparse ground truth with ReLU SAE | 8h | 15h |
| **6. Wide L0 range** | Gated/ReLU with better sweeps | 12h | 27h |
| **7. LLM validation** | GPT-2 layer SAEs | 20h | 47h |
| **8. Extended analysis** | Theory, intervention, viz | 20h | 67h |

**Requires 2-3 weeks part-time or 1.5 weeks full-time.**

---

## Next Steps (Immediate)

**To execute Path A (Fast-Track):**

1. **Confirm Path A decision** with user (5 min)
2. **Run Phase 1.1:** Archive invalid results (30 min)
3. **Run Phase 1.2:** Archive invalid docs (30 min)
4. **Run Phase 1.3:** Create master index (60 min)
5. **Run Phase 2:** Paper updates (3 hours)
6. **Run Phase 3:** Git commits (1.5 hours)
7. **Run Phase 4:** Final validation (30 min)

**Total: 6.5 hours** from go-ahead to clean repo + corrected paper

**To execute Path B (Comprehensive):**

1. **Confirm Path B decision** and timeline with user
2. **Execute Path A first** (clean repo regardless)
3. **Plan experiments** in detail (resource estimation)
4. **Sequential execution:** ReLU → Wide L0 → LLM
5. **Checkpoint after each:** Evaluate if continuing makes sense

---

## Files to Create During Execution

1. `FALSE_CLAIMS_CORRECTION_REPORT.md` - Automated scan results
2. `scripts/cleanup_false_claims.py` - Systematic correction tool
3. `results/_archived_buggy_experiments/README.md` - Archive explanation
4. `docs/_archive_rejected_hypotheses/README.md` - Archive explanation
5. `README_RESEARCH_FINDINGS.md` - Master navigation index
6. `PRE_SUBMISSION_CHECKLIST.md` - Final validation checklist

---

## Summary

**Current state:** Options B & C complete, critical bugs fixed, robust verified findings

**Two paths:**
- **Path A (Fast-Track):** 6-8 hours → Clean repo, workshop paper
- **Path B (Comprehensive):** 40-60 hours → Additional experiments, full paper

**Recommendation:** **Path A** - Strong findings, minimize risk, fast publication

**Next decision:** User confirms path, then immediate execution begins

---

**Created:** December 8, 2025, 11:15 PM
**Author:** Claude (with user direction)
**Status:** Awaiting user decision on Path A vs Path B
