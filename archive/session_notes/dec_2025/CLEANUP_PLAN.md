# Repository Cleanup and Reorganization Plan

**Date:** December 8, 2025  
**Status:** Ready for execution

---

## Current State Analysis

### Root Directory Issues

**26 markdown files in root** - too cluttered. Should have only:
- README.md
- CONTRIBUTING.md  
- LICENSE
- QUICK_START.md (optional)

### Files to Archive (Move to archive/session_notes/)

These are session-specific notes that should not be in root:

1. `BASIS_AMBIGUITY_DISCOVERY.md` - **RETRACTED** claim, archive
2. `CLAIMS_SUMMARY_TABLE.md` - Session work, archive
3. `COMPREHENSIVE_RESEARCH_SUMMARY.md` - Session work, archive
4. `CORRECTED_NUMBERS_REFERENCE.md` - Session work, archive
5. `CRITICAL_REVIEW_FINDINGS.md` - Session work, archive
6. `EXEC_SUMMARY_TASK_GENERALIZATION.md` - Session work, archive
7. `FILES_GENERATED.md` - Session work, archive
8. `FINAL_VERIFICATION_REPORT.md` - Session work, archive
9. `IDENTIFIABILITY_ANALYSIS.md` - Move to docs/
10. `NOVEL_RESEARCH_DIRECTIONS.md` - Session work, archive
11. `NOVEL_RESEARCH_EXTENSIONS.md` - Session work, archive
12. `OPTION_B_RESOLUTION_COMPLETE.md` - Session work, archive
13. `PARADOX_RESOLUTION.md` - Session work, archive
14. `PUBLICATION_CHECKLIST.md` - Session work, archive
15. `QUICK_ACTION_PLAN.md` - Session work, archive
16. `QUICK_SUMMARY.md` - Session work, archive
17. `README_VERIFICATION.md` - Session work, archive
18. `SESSION_SUMMARY_DEC8.md` - Session work, archive
19. `SPARSE_GROUND_TRUTH_EXPERIMENT.md` - Session work, archive
20. `SPARSE_VALIDATION_FINDINGS.md` - Session work, archive
21. `SUBSPACE_OVERLAP_FINDINGS.md` - Session work, archive
22. `TASK_GENERALIZATION_RESULTS.md` - Session work, archive
23. `VERIFIED_FINDINGS_FOR_PAPER.md` - Move to docs/

### Files to Keep in Root

- `README.md` - Main readme (needs update)
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - License file
- `QUICK_START.md` - Quick start guide

### Scripts to Clean Up

Move to archive/experimental_scripts/:
- `stability_monotonicity_analysis.py` (in root, should be in scripts/)

---

## Proposed New Structure

```
HUSAI/
├── README.md                    # Updated with verified findings
├── CONTRIBUTING.md              
├── LICENSE
├── QUICK_START.md
│
├── docs/
│   ├── VERIFIED_FINDINGS.md     # Clean summary of verified findings
│   ├── METHODOLOGY.md           # Experimental methodology
│   ├── IDENTIFIABILITY_ANALYSIS.md
│   └── ...existing docs...
│
├── paper/
│   └── sae_stability_paper.md   # Updated with corrections
│
├── archive/
│   └── session_notes/
│       └── dec_2025/            # All session-specific files
│
├── scripts/
│   ├── experiments/             # Main experiment scripts
│   ├── analysis/                # Analysis scripts
│   └── training/                # Training scripts
│
├── src/                         # Source code
├── tests/                       # Tests
├── notebooks/                   # Jupyter notebooks
├── results/                     # Experiment results
└── figures/                     # Generated figures
```

---

## Execution Steps

### Step 1: Archive Session Notes

```bash
mkdir -p archive/session_notes/dec_2025
mv BASIS_AMBIGUITY_DISCOVERY.md archive/session_notes/dec_2025/
mv CLAIMS_SUMMARY_TABLE.md archive/session_notes/dec_2025/
# ... etc
```

### Step 2: Update README.md

- Remove references to retracted claims
- Add verified findings summary
- Update project status

### Step 3: Update Paper

- Remove basis ambiguity claims
- Correct multi-architecture claims
- Add Archetypal SAE citation

### Step 4: Clean Up Scripts

- Move root-level scripts to scripts/
- Remove duplicate/outdated scripts

### Step 5: Git Commit and Push

```bash
git add -A
git commit -m "Major cleanup: Archive session notes, update paper with verified findings"
git push
```

---

## Files to Create

1. `docs/VERIFIED_FINDINGS.md` - Clean summary for external readers
2. Updated `README.md` - Professional project overview
3. Updated `paper/sae_stability_paper.md` - Corrected paper

---

## Verification Checklist

- [ ] All retracted claims removed from paper
- [ ] README reflects verified findings only
- [ ] Root directory has ≤5 markdown files
- [ ] All session notes archived
- [ ] Git history preserved
- [ ] Tests still pass
