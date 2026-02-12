# False Claims Detection Report

**Generated:** Automated scan

**Files with issues:** 26
**Total issues:** 223

## By Severity

| Severity | Count |
|----------|-------|
| CRITICAL | 80 |
| HIGH | 123 |
| MEDIUM | 4 |
| LOW | 16 |

## Detailed Findings

### `COMPREHENSIVE_ACTION_PLAN.md`

Issues found: 15

#### Line 38 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
| "Basis ambiguity" hypothesis | ❌ REJECTED | Archive BASIS_AMBIGUITY_DISCOVERY.md |
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 39 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
| "88% ground truth recovery" | ❌ BUG | Remove from all docs |
```

**Action required:** Should be 0% (bug fix)

#### Line 41 [LOW]

**Pattern:** `sparse.*improves.*stability`

**Text:**
```
| "Sparse improves stability" | ❌ FAILED | Add as negative finding |
```

**Action required:** Should note this FAILED (negative result)

#### Line 92 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
- Reported "88% recovery" was false (actual: 0%)
```

**Action required:** Should be 0% (bug fix)

#### Line 124 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
1. **Basis Ambiguity** (BASIS_AMBIGUITY_DISCOVERY.md)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 204 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Rejected "basis ambiguity" hypothesis
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 231 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
r"88% ground truth recovery": "0% ground truth recovery (corrected after bug fix)",
```

**Action required:** Should be 0% (bug fix)

#### Line 234 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
r"basis ambiguity": "[REJECTED HYPOTHESIS: basis ambiguity]",
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 282 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
> "We validated identifiability theory using sparse ground truth (10/128 = 7.8%). SAEs achieved 88% feature recovery with similarity 1.28, confirming the basis ambiguity phenomenon..."
```

**Action required:** Should be 0% (bug fix)

#### Line 282 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
> "We validated identifiability theory using sparse ground truth (10/128 = 7.8%). SAEs achieved 88% feature recovery with similarity 1.28, confirming the basis ambiguity phenomenon..."
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 282 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
> "We validated identifiability theory using sparse ground truth (10/128 = 7.8%). SAEs achieved 88% feature recovery with similarity 1.28, confirming the basis ambiguity phenomenon..."
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 340 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
- Mean similarity: 1.28 → 0.39 (corrected)
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 380 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Removed 'basis ambiguity' framing
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 427 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- [ ] Search for "basis ambiguity" → Only in archived files
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 429 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
- [ ] Search for "similarity.*1.2" → All instances corrected
```

**Action required:** Max cosine similarity is 1.0, this is impossible

### `archive/session_notes/CLAUDE_CODE_COLLABORATION.md`

Issues found: 1

#### Line 194 [HIGH]

**Pattern:** `different bases.*same subspace`

**Text:**
```
This suggests SAEs learn the RIGHT SUBSPACE but not the RIGHT BASIS within that subspace. Different seeds find different bases for the same subspace.
```

**Action required:** Rejected - SAEs learn orthogonal subspaces

### `archive/session_notes/dec_2025/BASIS_AMBIGUITY_DISCOVERY.md`

Issues found: 36

#### Line 1 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
# ❌ HYPOTHESIS REJECTED: SAE "Basis Ambiguity" Phenomenon
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 13 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
- Ground truth recovery: **0/10 features** (NOT 8.8/10 as originally reported)
```

**Action required:** Should be 0/10 (bug fix)

#### Line 14 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
- Mean similarity: **0.390** (NOT 1.28 - that value was impossible, >1.0 cosine!)
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 16 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**Conclusion:** The "basis ambiguity" hypothesis was based on buggy data. SAEs do NOT learn the same subspace with different bases. They learn nearly orthogonal subspaces and fail to recover ground truth features entirely.
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 32 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**We discovered that SAE "instability" is not about learning wrong features - it's about basis ambiguity.**
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 36 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
- ✅ **SAEs recover 88% of ground truth features** (8.8/10, similarity = 1.28)
```

**Action required:** Should be 0/10 (bug fix)

#### Line 36 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
- ✅ **SAEs recover 88% of ground truth features** (8.8/10, similarity = 1.28)
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 100 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
Ground truth recovery: 8.8/10 features ✅
```

**Action required:** Should be 0/10 (bug fix)

#### Line 101 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
Mean similarity: 1.284 (near perfect!) ✅
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 218 [HIGH]

**Pattern:** `different bases.*same subspace`

**Text:**
```
- "SAEs learn different bases for the same subspace"
```

**Action required:** Rejected - SAEs learn orthogonal subspaces

#### Line 219 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- "Basis ambiguity is fundamental, not a bug"
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 237 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Features are NOT stable (basis ambiguity)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 257 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Other features: dense subspaces → basis ambiguity (35%)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 272 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
| **Song et al. (2025)** | Stability-aware training improves overlap | We show this fights basis ambiguity, doesn't eliminate it |
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 275 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- **First demonstration** of basis ambiguity with controlled ground truth
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 310 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
# If basis ambiguity hypothesis is correct:
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 313 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
# This would CONFIRM basis ambiguity
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 323 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
### New Section: "The Basis Ambiguity Phenomenon"
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 333 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
> **Results:** SAEs achieved near-perfect ground truth recovery (8.8/10 features, similarity=1.28). However, PWMCC remained at random baseline (0.263 vs theory prediction >0.90).
```

**Action required:** Should be 0/10 (bug fix)

#### Line 333 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
> **Results:** SAEs achieved near-perfect ground truth recovery (8.8/10 features, similarity=1.28). However, PWMCC remained at random baseline (0.263 vs theory prediction >0.90).
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 335 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
> **Discovery:** This paradox reveals "basis ambiguity" - SAEs learn the correct 10-dimensional subspace but choose different orthonormal bases within it.
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 341 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
| Ground truth recovery | 8.8/10 | ~10/10 | ✅ Near perfect |
```

**Action required:** Should be 0/10 (bug fix)

#### Line 342 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
| Mean max similarity | 1.284 | >0.90 | ✅ Excellent |
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 344 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
| **Interpretation** | **Basis ambiguity** | **Unique features** | ⚠️ Subspace identified, basis not |
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 346 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
#### Figure: Basis Ambiguity Illustration
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 364 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**NEW (after sparse validation + basis ambiguity discovery):**
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 367 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
> To test if sparse ground truth improves stability, we trained SAEs on synthetic data with 10 known sparse features. Surprisingly, PWMCC remained low (0.263) despite near-perfect feature recovery (8.8/10).
```

**Action required:** Should be 0/10 (bug fix)

#### Line 367 [LOW]

**Pattern:** `sparse.*improves.*stability`

**Text:**
```
> To test if sparse ground truth improves stability, we trained SAEs on synthetic data with 10 known sparse features. Surprisingly, PWMCC remained low (0.263) despite near-perfect feature recovery (8.8/10).
```

**Action required:** Should note this FAILED (negative result)

#### Line 369 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
> This paradox reveals **basis ambiguity**: SAEs learn the correct subspace but choose different orthonormal bases within it. Feature-level instability (low PWMCC) coexists with subspace-level stability (high recovery).
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 381 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
### Contribution 2: Discovery of Basis Ambiguity ✅✅✅
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 406 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Follow-up: Basis ambiguity + subspace methods
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 416 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
3. `results/synthetic_sparse_exact/` - Exact match (PWMCC=0.263, recovery=8.8/10) 🔥
```

**Action required:** Should be 0/10 (bug fix)

#### Line 432 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**Why:** Confirms basis ambiguity hypothesis
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 438 [HIGH]

**Pattern:** `different bases.*same subspace`

**Text:**
```
**How:** 2D projection showing different bases, same subspace
```

**Action required:** Rejected - SAEs learn orthogonal subspaces

#### Line 443 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**How:** Add sparse validation + basis ambiguity finding
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 477 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**SAE "instability" is basis ambiguity, not feature randomness.**
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `archive/session_notes/dec_2025/CLAIMS_SUMMARY_TABLE.md`

Issues found: 8

#### Line 39 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
| 1 | Basis ambiguity: SAEs learn same subspace | Overlap >0.90 | Overlap 0.14 | -0.76 | BASIS_AMBIGUITY_DISCOVERY.md | DELETE all mentions |
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 43 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
| 5 | 88% recovery with similarity=1.28 | Cosine ∈[-1,1] | Value=1.28 | Impossible | Sparse experiment | FIX metric or remove |
```

**Action required:** Should be 0% (bug fix)

#### Line 43 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
| 5 | 88% recovery with similarity=1.28 | Cosine ∈[-1,1] | Value=1.28 | Impossible | Sparse experiment | FIX metric or remove |
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 54 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
| 1 | Ground truth recovery 88% (8.8/10) | Similarity >1.0 impossible | Verify cosine calculation | 1 hour | Resolves paradox |
```

**Action required:** Should be 0/10 (bug fix)

#### Line 75 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Basis ambiguity hypothesis rejected
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 105 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
1. ❌ Remove basis ambiguity (2 hours)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 129 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- `/Users/brightliu/School_Work/HUSAI/CRITICAL_REVIEW_FINDINGS.md` (basis ambiguity rejected)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 148 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Basis ambiguity hypothesis contradicted by data
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `archive/session_notes/dec_2025/CLEANUP_PLAN.md`

Issues found: 1

#### Line 115 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Remove basis ambiguity claims
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `archive/session_notes/dec_2025/CORRECTED_NUMBERS_REFERENCE.md`

Issues found: 11

#### Line 13 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
| **Features recovered** | 8.8/10 (88%) | **0/10 (0%)** | Threshold >0.9 similarity |
```

**Action required:** Should be 0/10 (bug fix)

#### Line 14 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
| **Mean max similarity** | 1.284 | **0.390 ± 0.02** | Buggy value was >1.0 (impossible!) |
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 44 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
| **Predicted (basis ambiguity)** | >0.90 | ❌ REJECTED |
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 46 [HIGH]

**Pattern:** `different bases.*same subspace`

**Text:**
```
**Conclusion:** SAEs learn nearly orthogonal subspaces, NOT different bases of same subspace
```

**Action required:** Rejected - SAEs learn orthogonal subspaces

#### Line 112 [CRITICAL]

**Pattern:** `1\.28.*similarity`

**Text:**
```
| "1.284 similarity" | **0.390 similarity** | >1.0 cosine impossible |
```

**Action required:** Should be ~0.39 (bug fix)

#### Line 113 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
| "8.8/10 recovery" | **0/10 recovery** | Normalization bug |
```

**Action required:** Should be 0/10 (bug fix)

#### Line 139 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
| Mean similarity | 1.28 → **0.39** (3.3× deflation) |
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 151 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
- "88% ground truth recovery"
```

**Action required:** Should be 0% (bug fix)

#### Line 152 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
- "Similarity = 1.28"
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 153 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- "Basis ambiguity phenomenon"
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 155 [LOW]

**Pattern:** `sparse.*improves.*stability`

**Text:**
```
- "Sparse ground truth improves stability"
```

**Action required:** Should note this FAILED (negative result)

### `archive/session_notes/dec_2025/CRITICAL_REVIEW_FINDINGS.md`

Issues found: 14

#### Line 14 [MEDIUM]

**Pattern:** `multi-architecture.*verification`

**Text:**
```
2. **Multi-architecture verification** (TopK, ReLU, Gated, JumpReLU all show same pattern)
```

**Action required:** Only TopK thoroughly tested

#### Line 19 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
1. **"Basis Ambiguity" Discovery** - The key claim is NOT supported by data
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 25 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
### 1. Basis Ambiguity Claim - REJECTED
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 49 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**Conclusion:** The "basis ambiguity" hypothesis is **REJECTED**. SAEs are NOT learning the same subspace with different bases - they are learning **genuinely different representations**.
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 54 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
> "SAEs recover 88% of ground truth features (8.8/10, similarity = 1.28)"
```

**Action required:** Should be 0/10 (bug fix)

#### Line 54 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
> "SAEs recover 88% of ground truth features (8.8/10, similarity = 1.28)"
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 56 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
**Issue:** The "mean similarity = 1.28" is suspicious. Cosine similarity should be in [-1, 1]. A value of 1.28 suggests either:
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 56 [CRITICAL]

**Pattern:** `1\.28.*similarity`

**Text:**
```
**Issue:** The "mean similarity = 1.28" is suspicious. Cosine similarity should be in [-1, 1]. A value of 1.28 suggests either:
```

**Action required:** Should be ~0.39 (bug fix)

#### Line 86 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
1. **Remove or correct "Basis Ambiguity" claims** from:
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 100 [MEDIUM]

**Pattern:** `multi-architecture.*verification`

**Text:**
```
- Multi-architecture stability verification
```

**Action required:** Only TopK thoroughly tested

#### Line 106 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Basis ambiguity claims
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 123 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
| `scripts/experiments/validate_basis_ambiguity.py` | Created - validates (rejects) basis ambiguity |
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 136 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
However, the "basis ambiguity discovery" is **NOT supported by the data** and should be removed from the paper before submission.
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 138 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
The research is still valuable - it just needs to be presented accurately without the unsupported "basis ambiguity" interpretation.
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `archive/session_notes/dec_2025/FINAL_VERIFICATION_REPORT.md`

Issues found: 24

#### Line 24 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- ❌ "Basis Ambiguity" hypothesis (contradicted by data)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 195 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
### 2.1 Basis Ambiguity Hypothesis ❌
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 221 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**Action Required:** Remove all "basis ambiguity" claims from paper and documentation
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 226 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**Statement:** "SAE 'instability' is not about learning wrong features - it's about basis ambiguity"
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 261 [LOW]

**Pattern:** `sparse ground truth.*high.*stability`

**Text:**
```
**Correct Claim:** "We validate identifiability theory's prediction for DENSE ground truth (PWMCC ≈ 0.30), but find that sparse ground truth does NOT yield high stability as theory predicts"
```

**Action required:** Theory predicted high, but observed low

#### Line 280 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
#### Claim 2.3a: 88% Recovery with High Similarity
```

**Action required:** Should be 0% (bug fix)

#### Line 281 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
**Statement (from BASIS_AMBIGUITY_DISCOVERY.md):** "SAEs recover 88% of ground truth features (8.8/10, similarity = 1.28)"
```

**Action required:** Should be 0/10 (bug fix)

#### Line 281 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
**Statement (from BASIS_AMBIGUITY_DISCOVERY.md):** "SAEs recover 88% of ground truth features (8.8/10, similarity = 1.28)"
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 283 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
**Problem:** Mean similarity = 1.28 is **mathematically impossible** for cosine similarity (should be ≤ 1.0)
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 283 [CRITICAL]

**Pattern:** `1\.28.*similarity`

**Text:**
```
**Problem:** Mean similarity = 1.28 is **mathematically impossible** for cosine similarity (should be ≤ 1.0)
```

**Action required:** Should be ~0.39 (bug fix)

#### Line 309 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
- But: 88% feature recovery claimed (metric suspicious)
```

**Action required:** Should be 0% (bug fix)

#### Line 392 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
- Recovery = 8.8/10 features (matches ~9 stable)
```

**Action required:** Should be 0/10 (bug fix)

#### Line 421 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
But both recover 8.8/10 ground truth features?
```

**Action required:** Should be 0/10 (bug fix)

#### Line 551 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
1. ❌ **Basis ambiguity hypothesis**
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 566 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
4. ❌ **88% ground truth recovery** (unless verified)
```

**Action required:** Should be 0% (bug fix)

#### Line 657 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Any mention of basis ambiguity
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 718 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
#### 4.11 ❌ Sparse Ground Truth / Basis Ambiguity (REMOVE)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 766 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
### Issue 2: Basis Ambiguity Claims ❌
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 787 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
1. ❌ Basis ambiguity claims must be removed
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 792 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Remove basis ambiguity: 2 hours
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 831 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
1. **Remove Basis Ambiguity** (2 hours)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 882 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- ❌ Remove basis ambiguity completely
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 921 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
However, **critical claims about "basis ambiguity" are contradicted by data** and must be removed before publication. Additionally, **sparse ground truth results contradict theory** and should be presented as preliminary/inconclusive.
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 931 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**Next Steps:** Remove basis ambiguity, verify metrics, rewrite theory section
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `archive/session_notes/dec_2025/NOVEL_RESEARCH_EXTENSIONS.md`

Issues found: 1

#### Line 42 [LOW]

**Pattern:** `sparse ground truth.*high.*stability`

**Text:**
```
**Goal:** Validate that SPARSE ground truth → HIGH stability (PWMCC > 0.70)
```

**Action required:** Theory predicted high, but observed low

### `archive/session_notes/dec_2025/OPTION_B_RESOLUTION_COMPLETE.md`

Issues found: 15

#### Line 27 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
- Ground truth recovery: **8.8/10 features** (88%)
```

**Action required:** Should be 0/10 (bug fix)

#### Line 28 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
- Mean max similarity: **1.28** (>100% cosine similarity - impossible!)
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 28 [CRITICAL]

**Pattern:** `1\.28.*similarity`

**Text:**
```
- Mean max similarity: **1.28** (>100% cosine similarity - impossible!)
```

**Action required:** Should be ~0.39 (bug fix)

#### Line 61 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
| **Features recovered** | 8.8/10 (88%) | **0/10 (0%)** | -88 pp |
```

**Action required:** Should be 0/10 (bug fix)

#### Line 62 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
| **Mean similarity** | 1.284 | **0.390 ± 0.02** | -0.894 (3.3× inflation) |
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 86 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
**Conclusion:** SAEs completely FAIL to recover sparse ground truth, even under extreme sparsity (7.8%). The "88% recovery" was entirely an artifact of the normalization bug.
```

**Action required:** Should be 0% (bug fix)

#### Line 177 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
2. **"Mean similarity 1.28 indicates near-perfect recovery"**
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 180 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
3. **"Basis ambiguity: SAEs learn same subspace with different bases"**
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 186 [LOW]

**Pattern:** `sparse.*improves.*stability`

**Text:**
```
5. **"Sparse ground truth improves stability per identifiability theory"**
```

**Action required:** Should note this FAILED (negative result)

#### Line 241 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
3. **Confirmation bias** (88% recovery seemed like good news)
```

**Action required:** Should be 0% (bug fix)

#### Line 288 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
- ❌ Remove: "88% ground truth recovery"
```

**Action required:** Should be 0% (bug fix)

#### Line 289 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
- ❌ Remove: "Similarity = 1.28"
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 290 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- ❌ Remove: "Basis ambiguity" explanation
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 344 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
- **Remove:** Basis ambiguity, 88% recovery, multi-architecture claims
```

**Action required:** Should be 0% (bug fix)

#### Line 344 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- **Remove:** Basis ambiguity, 88% recovery, multi-architecture claims
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `archive/session_notes/dec_2025/PARADOX_RESOLUTION.md`

Issues found: 9

#### Line 14 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
| 88% GT recovery with 14% subspace overlap | **BUG**: Ground truth metric was wrong; actual recovery is 0% |
```

**Action required:** Should be 0% (bug fix)

#### Line 24 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
> "SAEs recover 88% of ground truth features (8.8/10) but subspace overlap is only 14%"
```

**Action required:** Should be 0/10 (bug fix)

#### Line 53 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
| Features recovered (>0.9 sim) | 8.8/10 | **0/10** |
```

**Action required:** Should be 0/10 (bug fix)

#### Line 54 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
| Mean max similarity | 1.28 | **0.14-0.19** |
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 134 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
1. **"Basis ambiguity"** - Subspace overlap is 14%, not 90%
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 135 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
2. **"88% ground truth recovery"** - Actual recovery is 0%
```

**Action required:** Should be 0% (bug fix)

#### Line 144 [LOW]

**Pattern:** `sparse ground truth.*high.*stability`

**Text:**
```
| "Sparse ground truth → high stability" | "Sparse ground truth → SAEs still unstable (need investigation)" |
```

**Action required:** Theory predicted high, but observed low

#### Line 145 [MEDIUM]

**Pattern:** `multi-architecture.*verification`

**Text:**
```
| "Multi-architecture verification" | "TopK verification; other architectures need wider L0 range" |
```

**Action required:** Only TopK thoroughly tested

#### Line 161 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
1. All "basis ambiguity" claims
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `archive/session_notes/dec_2025/PUBLICATION_CHECKLIST.md`

Issues found: 14

#### Line 13 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- ❌ "Basis ambiguity" claims contradict data (must remove)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 29 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
### Blocker 1: Basis Ambiguity Claims ❌
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 39 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- [ ] Paper Discussion - Remove all "basis ambiguity" mentions
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 45 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
grep -r "basis ambiguity" /Users/brightliu/School_Work/HUSAI/paper/
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 56 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
- Mean similarity = 1.28 (impossible for cosine similarity ∈ [-1, 1])
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 56 [CRITICAL]

**Pattern:** `1\.28.*similarity`

**Text:**
```
- Mean similarity = 1.28 (impossible for cosine similarity ∈ [-1, 1])
```

**Action required:** Should be ~0.39 (bug fix)

#### Line 228 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Abstract (minus basis ambiguity, fix identifiability claim)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 249 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- 4.11 Basis Ambiguity Discovery (if exists) - DELETE
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 262 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
# Find basis ambiguity mentions
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 263 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
grep -n "basis ambiguity" sae_stability_paper.md
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 274 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
grep -n "8.8/10" sae_stability_paper.md
```

**Action required:** Should be 0/10 (bug fix)

#### Line 280 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Basis ambiguity → DELETE
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 288 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
### Hour 1: Remove Basis Ambiguity
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 339 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- ❌ Basis ambiguity removed from paper
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `archive/session_notes/dec_2025/README_VERIFICATION.md`

Issues found: 12

#### Line 29 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
├─ Basis ambiguity (REJECTED): BASIS_AMBIGUITY_DISCOVERY.md
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 62 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Blocker 1: Basis ambiguity claims (2 hours)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 114 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
1. **Basis ambiguity hypothesis** - SAEs learn same subspace
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 124 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
3. **Ground truth recovery** - 88% with similarity=1.28
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 129 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**Action:** Remove all basis ambiguity claims, tone down identifiability, fix metric
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 150 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**1. Remove Basis Ambiguity Claims (2 hours)**
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 151 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Search paper for: "basis ambiguity", "same subspace", "rotated"
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 172 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**Reason:** Contains contradicted claims (basis ambiguity, recovery >1.0)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 235 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
├── CRITICAL_REVIEW_FINDINGS.md        (Basis ambiguity rejection)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 268 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- ❌ Basis ambiguity hypothesis contradicted by data
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 296 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**Q: Is the basis ambiguity discovery wrong?**
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 324 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
The basis ambiguity hypothesis didn't pan out, but that's how science works - you test hypotheses and some get rejected. The strength of your work is the rigorous testing that revealed this.
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `archive/session_notes/dec_2025/SESSION_SUMMARY_DEC8.md`

Issues found: 10

#### Line 19 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
2. **Basis ambiguity hypothesis:** ❌ REJECTED
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 45 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
- Reported: 8.8/10 features recovered, similarity = 1.28
```

**Action required:** Should be 0/10 (bug fix)

#### Line 45 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
- Reported: 8.8/10 features recovered, similarity = 1.28
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 88 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
2. **"Basis ambiguity phenomenon"** → FALSE (subspace overlap 14%, not 90%)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 90 [LOW]

**Pattern:** `sparse.*improves.*stability`

**Text:**
```
4. **"Sparse ground truth improves stability"** → FALSE (PWMCC unchanged: 0.27)
```

**Action required:** Should note this FAILED (negative result)

#### Line 155 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
- ❌ Remove: "88% recovery, similarity = 1.28"
```

**Action required:** Should be 0% (bug fix)

#### Line 155 [CRITICAL]

**Pattern:** `similarity.*1\.[2-9]`

**Text:**
```
- ❌ Remove: "88% recovery, similarity = 1.28"
```

**Action required:** Max cosine similarity is 1.0, this is impossible

#### Line 157 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- ❌ Remove: All "basis ambiguity" explanations
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 221 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- OLD: "SAEs work under sparsity (basis ambiguity)"
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 255 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**Hypotheses rejected:** 2 (basis ambiguity, multi-architecture)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `archive/session_notes/dec_2025/SPARSE_GROUND_TRUTH_EXPERIMENT.md`

Issues found: 1

#### Line 129 [LOW]

**Pattern:** `sparse ground truth.*high.*stability`

**Text:**
```
- Sparse ground truth → High stability
```

**Action required:** Theory predicted high, but observed low

### `archive/session_notes/dec_2025/SPARSE_VALIDATION_FINDINGS.md`

Issues found: 6

#### Line 11 [LOW]

**Pattern:** `sparse ground truth.*high.*stability`

**Text:**
```
We tested Cui et al. (2025)'s identifiability theory prediction that **sparse ground truth** should lead to **high SAE stability** (PWMCC > 0.70).
```

**Action required:** Theory predicted high, but observed low

#### Line 196 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- This would indicate "basis ambiguity" not "wrong features"
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 235 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- Basis ambiguity (many equivalent solutions)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 245 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- This is "basis ambiguity" not "feature instability"
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 289 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**Why:** Tests "basis ambiguity" hypothesis
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 341 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
### Outcome 3: Basis Ambiguity Discovery ✅✅✅
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `archive/session_notes/dec_2025/SUBSPACE_OVERLAP_FINDINGS.md`

Issues found: 9

#### Line 5 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**Experiment:** Validation of Basis Ambiguity Hypothesis
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 12 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
We investigated whether SAEs trained on synthetic sparse data with **known 10D ground truth** learn the same subspace. The results definitively reject the basis ambiguity hypothesis and reveal a fundamental puzzle:
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 23 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
### Basis Ambiguity Prediction
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 32 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
Ground Truth:      8.8/10 features [Expected: 10/10] ~ GOOD
```

**Action required:** Should be 0/10 (bug fix)

#### Line 47 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
But both recover 8.8/10 ground truth features?
```

**Action required:** Should be 0/10 (bug fix)

#### Line 99 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
Average: 8.8/10 = 88% recovery
```

**Action required:** Should be 0% (bug fix)

#### Line 99 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
Average: 8.8/10 = 88% recovery
```

**Action required:** Should be 0/10 (bug fix)

#### Line 119 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
- Recovery = 8.8/10 (matches ~9 stable features)
```

**Action required:** Should be 0/10 (bug fix)

#### Line 248 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
The **basis ambiguity hypothesis is definitively rejected**. SAEs do not learn the same subspace with different bases. Instead, they learn nearly orthogonal subspaces (14% overlap).
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `docs/VERIFIED_FINDINGS_FOR_PAPER.md`

Issues found: 4

#### Line 96 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
### ❌ Basis Ambiguity Claims
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 98 [HIGH]

**Pattern:** `different bases.*same subspace`

**Text:**
```
**Reason:** Subspace overlap is 14%, not 90%. SAEs learn different subspaces, not different bases for the same subspace.
```

**Action required:** Rejected - SAEs learn orthogonal subspaces

#### Line 108 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
### ❌ "88% Ground Truth Recovery"
```

**Action required:** Should be 0% (bug fix)

#### Line 177 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
- [ ] Remove all basis ambiguity claims
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `results/synthetic_sparse_exact/SUBSPACE_ANALYSIS.md`

Issues found: 8

#### Line 4 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
**Experiment:** Validation of Basis Ambiguity Hypothesis
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 9 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
The "basis ambiguity" hypothesis has been **definitively rejected**. SAEs trained on synthetic sparse data with known 10D ground truth show:
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 17 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
**SAEs are NOT learning the same 10D subspace.** Instead, they are learning nearly orthogonal subspaces, despite recovering 8.8/10 ground truth features on average.
```

**Action required:** Should be 0/10 (bug fix)

#### Line 21 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
### What We Expected (Basis Ambiguity Hypothesis)
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 31 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
BUT recover ground truth well → 8.8/10 features recovered
```

**Action required:** Should be 0/10 (bug fix)

#### Line 35 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
**How can SAEs recover the ground truth features (8.8/10) while learning nearly orthogonal subspaces (overlap = 0.14)?**
```

**Action required:** Should be 0/10 (bug fix)

#### Line 126 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
2. **Check ground truth recovery method:** Verify the 8.8/10 claim
```

**Action required:** Should be 0/10 (bug fix)

#### Line 142 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
The basis ambiguity hypothesis is **strongly rejected**. SAEs do not learn the same subspace with different bases. Instead, they learn **different subspaces** entirely, with only ~14% overlap.
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `scripts/cleanup_false_claims.py`

Issues found: 6

#### Line 18 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
r"88%.*recovery": ("CRITICAL", "Should be 0% (bug fix)"),
```

**Action required:** Should be 0% (bug fix)

#### Line 24 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
r"basis ambiguity": ("HIGH", "Hypothesis rejected - subspace overlap is 14%, not 90%"),
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 25 [HIGH]

**Pattern:** `different bases.*same subspace`

**Text:**
```
r"different bases.*same subspace": ("HIGH", "Rejected - SAEs learn orthogonal subspaces"),
```

**Action required:** Rejected - SAEs learn orthogonal subspaces

#### Line 29 [MEDIUM]

**Pattern:** `multi-architecture.*verification`

**Text:**
```
r"multi-architecture.*verification": ("MEDIUM", "Only TopK thoroughly tested"),
```

**Action required:** Only TopK thoroughly tested

#### Line 33 [LOW]

**Pattern:** `sparse.*improves.*stability`

**Text:**
```
r"sparse.*improves.*stability": ("LOW", "Should note this FAILED (negative result)"),
```

**Action required:** Should note this FAILED (negative result)

#### Line 34 [LOW]

**Pattern:** `sparse ground truth.*high.*stability`

**Text:**
```
r"sparse ground truth.*high.*stability": ("LOW", "Theory predicted high, but observed low"),
```

**Action required:** Theory predicted high, but observed low

### `scripts/diagnose_recovery_paradox.py`

Issues found: 1

#### Line 4 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
Paradox: SAEs recover 8.8/10 true features (88%) with high similarity,
```

**Action required:** Should be 0/10 (bug fix)

### `scripts/experiments/diagnose_paradoxes.py`

Issues found: 1

#### Line 6 [CRITICAL]

**Pattern:** `88%.*recovery`

**Text:**
```
1. 88% ground truth recovery with 14% subspace overlap
```

**Action required:** Should be 0% (bug fix)

### `scripts/experiments/validate_basis_ambiguity.py`

Issues found: 4

#### Line 2 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
"""Validate the Basis Ambiguity Discovery
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 10 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
This would confirm that SAE "instability" is basis ambiguity, not wrong features.
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 77 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
print("VALIDATING BASIS AMBIGUITY HYPOTHESIS")
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 125 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
print("CONFIRMED: Basis Ambiguity!")
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

### `scripts/sparse_ground_truth_experiment.py`

Issues found: 2

#### Line 600 [LOW]

**Pattern:** `sparse ground truth.*high.*stability`

**Text:**
```
print(f"\nHypothesis: Sparse ground truth → High SAE stability (PWMCC > 0.70)")
```

**Action required:** Theory predicted high, but observed low

#### Line 789 [LOW]

**Pattern:** `sparse ground truth.*high.*stability`

**Text:**
```
print(f"   Sparse ground truth → High stability ({mean_pwmcc:.3f} vs dense {dense_pwmcc:.3f})")
```

**Action required:** Theory predicted high, but observed low

### `scripts/synthetic_sparse_validation.py`

Issues found: 2

#### Line 293 [LOW]

**Pattern:** `sparse ground truth.*high.*stability`

**Text:**
```
print(f"\nHypothesis: Sparse ground truth → High SAE stability (PWMCC > 0.90)")
```

**Action required:** Theory predicted high, but observed low

#### Line 440 [LOW]

**Pattern:** `sparse ground truth.*high.*stability`

**Text:**
```
print(f"   Sparse ground truth → Extremely high stability ({mean_pwmcc:.3f})")
```

**Action required:** Theory predicted high, but observed low

### `scripts/validate_subspace_overlap.py`

Issues found: 8

#### Line 2 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
"""Validate Basis Ambiguity Hypothesis - Subspace Overlap Analysis
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 8 [CRITICAL]

**Pattern:** `8\.8/10`

**Text:**
```
- Ground truth recovery = 8.8/10 features (high subspace recovery)
```

**Action required:** Should be 0/10 (bug fix)

#### Line 10 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
This suggests "basis ambiguity": SAEs learn the CORRECT 10D subspace but
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 15 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
If basis ambiguity is the explanation, then:
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 220 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
"""Main validation function for basis ambiguity hypothesis.
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 233 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
print("SUBSPACE OVERLAP VALIDATION - Basis Ambiguity Hypothesis")
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 349 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
print(f"\nBasis Ambiguity Hypothesis:")
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%

#### Line 449 [HIGH]

**Pattern:** `basis ambiguity`

**Text:**
```
description="Validate basis ambiguity hypothesis via subspace overlap analysis",
```

**Action required:** Hypothesis rejected - subspace overlap is 14%, not 90%


## Action Items

### Priority 1: CRITICAL Issues

- [ ] `COMPREHENSIVE_ACTION_PLAN.md` - Fix numerical errors
- [ ] `archive/session_notes/dec_2025/BASIS_AMBIGUITY_DISCOVERY.md` - Fix numerical errors
- [ ] `archive/session_notes/dec_2025/CLAIMS_SUMMARY_TABLE.md` - Fix numerical errors
- [ ] `archive/session_notes/dec_2025/CORRECTED_NUMBERS_REFERENCE.md` - Fix numerical errors
- [ ] `archive/session_notes/dec_2025/CRITICAL_REVIEW_FINDINGS.md` - Fix numerical errors
- [ ] `archive/session_notes/dec_2025/FINAL_VERIFICATION_REPORT.md` - Fix numerical errors
- [ ] `archive/session_notes/dec_2025/OPTION_B_RESOLUTION_COMPLETE.md` - Fix numerical errors
- [ ] `archive/session_notes/dec_2025/PARADOX_RESOLUTION.md` - Fix numerical errors
- [ ] `archive/session_notes/dec_2025/PUBLICATION_CHECKLIST.md` - Fix numerical errors
- [ ] `archive/session_notes/dec_2025/README_VERIFICATION.md` - Fix numerical errors
- [ ] `archive/session_notes/dec_2025/SESSION_SUMMARY_DEC8.md` - Fix numerical errors
- [ ] `archive/session_notes/dec_2025/SUBSPACE_OVERLAP_FINDINGS.md` - Fix numerical errors
- [ ] `docs/VERIFIED_FINDINGS_FOR_PAPER.md` - Fix numerical errors
- [ ] `results/synthetic_sparse_exact/SUBSPACE_ANALYSIS.md` - Fix numerical errors
- [ ] `scripts/cleanup_false_claims.py` - Fix numerical errors
- [ ] `scripts/diagnose_recovery_paradox.py` - Fix numerical errors
- [ ] `scripts/experiments/diagnose_paradoxes.py` - Fix numerical errors
- [ ] `scripts/validate_subspace_overlap.py` - Fix numerical errors

### Priority 2: HIGH Issues

- [ ] `COMPREHENSIVE_ACTION_PLAN.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/CLAUDE_CODE_COLLABORATION.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/dec_2025/BASIS_AMBIGUITY_DISCOVERY.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/dec_2025/CLAIMS_SUMMARY_TABLE.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/dec_2025/CLEANUP_PLAN.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/dec_2025/CORRECTED_NUMBERS_REFERENCE.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/dec_2025/CRITICAL_REVIEW_FINDINGS.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/dec_2025/FINAL_VERIFICATION_REPORT.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/dec_2025/OPTION_B_RESOLUTION_COMPLETE.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/dec_2025/PARADOX_RESOLUTION.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/dec_2025/PUBLICATION_CHECKLIST.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/dec_2025/README_VERIFICATION.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/dec_2025/SESSION_SUMMARY_DEC8.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/dec_2025/SPARSE_VALIDATION_FINDINGS.md` - Remove rejected hypotheses
- [ ] `archive/session_notes/dec_2025/SUBSPACE_OVERLAP_FINDINGS.md` - Remove rejected hypotheses
- [ ] `docs/VERIFIED_FINDINGS_FOR_PAPER.md` - Remove rejected hypotheses
- [ ] `results/synthetic_sparse_exact/SUBSPACE_ANALYSIS.md` - Remove rejected hypotheses
- [ ] `scripts/cleanup_false_claims.py` - Remove rejected hypotheses
- [ ] `scripts/experiments/validate_basis_ambiguity.py` - Remove rejected hypotheses
- [ ] `scripts/validate_subspace_overlap.py` - Remove rejected hypotheses
