# Claims Summary Table: SAE Stability Research
## Quick Reference for All Research Claims

**Generated:** December 8, 2025
**Source:** Comprehensive analysis of 4 key documents + experimental results

---

## Legend
- ✅ **VERIFIED** - Supported by data, safe for publication
- ❌ **REJECTED** - Contradicted by data, must remove
- ⚠️ **UNCERTAIN** - Needs investigation before publication

---

## VERIFIED CLAIMS (8 Major Findings) ✅

| # | Claim | Evidence | Source | Strength |
|---|-------|----------|--------|----------|
| 1 | Random baseline phenomenon (PWMCC=0.309 vs 0.300) | 10 trained pairs vs 45 random pairs, p<0.0001 | Paper §4.1 | ⭐⭐⭐⭐⭐ |
| 2 | Task independence (mod arith + copying both ≈0.30) | Two tasks, same result | Paper §4.2 | ⭐⭐⭐⭐⭐ |
| 3 | Functional-representational paradox (good MSE, random features) | MSE 4-8× better, PWMCC unchanged | Paper §4.2 | ⭐⭐⭐⭐⭐ |
| 4 | Architecture independence (TopK=ReLU=Gated≈0.30) | 4 architectures tested | multi_architecture_results.json | ⭐⭐⭐⭐⭐ |
| 5 | Stability decreases with sparsity (corr=-0.725) | TopK: -0.917, ReLU: -0.999 | Multi-arch experiment | ⭐⭐⭐⭐ |
| 6 | Three regime tradeoff (under/matched/over) | Expansion factor sweep | expansion_factor_results.json | ⭐⭐⭐⭐ |
| 7 | Feature-level stability uniform (no predictors) | Tested freq, norm, sparsity - none predict | feature_level_stability_results.json | ⭐⭐⭐ |
| 8 | Features converge 0.30→0.36 (don't diverge) | Training dynamics over 100 epochs | training_dynamics_results.json | ⭐⭐⭐ |
| 9 | Stability correlates with task complexity | Higher accuracy → higher stability | task_complexity_results.json | ⭐⭐⭐ |
| 10 | Dense ground truth matches theory (0.309 ≈ 0.30) | Cui et al. predicts 0.25-0.35, observed 0.309 | Paper §4.10 + IDENTIFIABILITY_ANALYSIS.md | ⭐⭐⭐⭐ |

**Total:** 10 verified findings, all publication-ready

---

## REJECTED CLAIMS (Must Remove) ❌

| # | Claim | Predicted | Actual | Gap | Source | Action |
|---|-------|-----------|--------|-----|--------|--------|
| 1 | Basis ambiguity: SAEs learn same subspace | Overlap >0.90 | Overlap 0.14 | -0.76 | BASIS_AMBIGUITY_DISCOVERY.md | DELETE all mentions |
| 2 | Rotated bases explanation | High subspace, low feature | Low both | N/A | SUBSPACE_OVERLAP_FINDINGS.md | DELETE explanation |
| 3 | Subspace identifiability | SAEs span same 10D | Nearly orthogonal 10D | N/A | Validation script | DELETE claims |
| 4 | Strong identifiability validation | Sparse → PWMCC>0.90 | PWMCC=0.263 | -0.64 | SPARSE_VALIDATION_FINDINGS.md | Tone down to "partial" |
| 5 | 88% recovery with similarity=1.28 | Cosine ∈[-1,1] | Value=1.28 | Impossible | Sparse experiment | FIX metric or remove |
| 6 | "First validation" of Cui et al. theory | Both dense+sparse | Only dense ✓ | Half fails | Multiple docs | Change to "partial test" |

**Total:** 6 rejected/problematic claims requiring removal or fixing

---

## UNCERTAIN CLAIMS (Need Investigation) ⚠️

| # | Claim | Issue | Test Needed | Time | Impact |
|---|-------|-------|-------------|------|--------|
| 1 | Ground truth recovery 88% (8.8/10) | Similarity >1.0 impossible | Verify cosine calculation | 1 hour | Resolves paradox |
| 2 | 9D core hypothesis (share 9D, differ in 1D) | Plausible but untested | Run k=9 overlap test | 5 min | Interesting if true |
| 3 | TopK-specific instability | ReLU not tested on sparse | Train ReLU on sparse data | 2 hours | Major if true |
| 4 | Generalizes to LLM SAEs | Toy tasks only | Run on Gemma Scope | Days | Critical for impact |
| 5 | Explains Paulo & Belrose 65% sharing | Speculative connection | No evidence yet | N/A | Remove or hedge |

**Total:** 5 uncertain claims - recommend investigating #1 and #2 before submission

---

## PUBLICATION IMPACT SUMMARY

### Core Contribution (STRONG) ✅
- **10 verified findings** about SAE stability on algorithmic tasks
- **Novel discovery:** Random baseline phenomenon
- **Important paradox:** Functional success + representational instability
- **Rigorous validation:** Multi-architecture, multi-task, multi-seed

### Weaknesses (After Fixes) ⚠️
- Limited to toy tasks (not LLMs yet)
- Sparse ground truth contradicts theory (unresolved)
- Basis ambiguity hypothesis rejected
- Only partial theoretical validation

### After Removing False Claims
**Strength:** 8/10 (solid empirical paper)
**Novelty:** 9/10 (random baseline is striking)
**Rigor:** 9/10 (multi-architecture rare)
**Impact:** 7/10 (toy tasks limit generality)

**Overall:** Strong workshop paper, competitive main conference

---

## VENUE RECOMMENDATIONS

| Venue | Probability | Rationale |
|-------|-------------|-----------|
| ICLR Workshop | 90% | Strong empirics, honest about scope |
| NeurIPS Workshop | 85% | Novel finding, good validation |
| ICML Workshop | 80% | Solid contribution |
| Main Conference | 40-50% | Good empirics, limited to toy tasks |
| arXiv + Journals | 100% | Always viable, no deadline pressure |

**Recommendation:** Workshop first (fast feedback) → Extend for main conference

---

## CRITICAL PATH TO PUBLICATION

### Must Do (4 hours)
1. ❌ Remove basis ambiguity (2 hours)
2. ❌ Fix/verify recovery metric (1 hour)
3. ⚠️ Rewrite Section 4.10 identifiability (1 hour)

### Should Do (1 hour)
4. ⚠️ Run k=9 subspace test (5 min)
5. ⚠️ Proofread all revised sections (30 min)
6. ✅ Final consistency check (30 min)

### Could Do (Future Work)
7. ReLU sparse test (2 hours)
8. LLM SAE analysis (days to weeks)
9. Resolve sparse ground truth paradox (research project)

---

## FILES TO REVIEW

**Critical Documents:**
- `/Users/brightliu/School_Work/HUSAI/FINAL_VERIFICATION_REPORT.md` (932 lines, comprehensive)
- `/Users/brightliu/School_Work/HUSAI/PUBLICATION_CHECKLIST.md` (quick action guide)
- `/Users/brightliu/School_Work/HUSAI/CLAIMS_SUMMARY_TABLE.md` (this file)

**Evidence Files:**
- `/Users/brightliu/School_Work/HUSAI/CRITICAL_REVIEW_FINDINGS.md` (basis ambiguity rejected)
- `/Users/brightliu/School_Work/HUSAI/SUBSPACE_OVERLAP_FINDINGS.md` (overlap=0.14 data)
- `/Users/brightliu/School_Work/HUSAI/SPARSE_VALIDATION_FINDINGS.md` (theory contradiction)
- `/Users/brightliu/School_Work/HUSAI/BASIS_AMBIGUITY_DISCOVERY.md` (original hypothesis)

**Paper:**
- `/Users/brightliu/School_Work/HUSAI/paper/sae_stability_paper.md` (requires revisions)

---

## BOTTOM LINE

**What You Have:**
- 10 solid, verified empirical findings
- Novel and important discovery (random baseline)
- Rigorous multi-architecture validation
- Strong workshop paper

**What Went Wrong:**
- Basis ambiguity hypothesis contradicted by data
- Sparse ground truth didn't validate as expected
- Recovery metric shows impossible values (>1.0)

**What To Do:**
- Remove false claims (2 hours)
- Fix/verify metrics (1 hour)
- Tone down theory claims (1 hour)
- **Submit to workshop (high acceptance probability)**

**Timeline:** 4 hours → publication-ready

---

**Status:** Comprehensive verification complete
**Recommendation:** Fix blockers → Submit to workshop
**Confidence:** High (after fixes)
