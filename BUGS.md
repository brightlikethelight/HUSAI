# Bugs and Reliability Gaps

Date: 2026-03-05

## Fixed in This Pass

1. TopK auxiliary loss dropped from objective (`src/training/train_sae.py`) - fixed.
2. Small-dataset divide-by-zero risk in SAE trainer - fixed.
3. Hard `wandb` import dependency - fixed.
4. Singleton feature-stat crash in `feature_matching` - fixed.
5. Routed frontier invalid expert config leading to degenerate features - fixed.
6. Assignment-v2 empty list parse/summarize crash path - fixed.
7. CE-Bench model-name key errors and artifact-path restoration on exception - fixed.
8. Official benchmark harness shell command execution risk (`shell=True`) - fixed.

## Open High-Priority Issues

### P1-Open-1: Final evidence package is not fully mirrored locally
- Impact: exact final-cycle candidate/metric claims remain partially remote-dependent.
- Evidence: see `EVIDENCE_STATUS.md`.
- Suggested fix: export remote final package metadata into `docs/evidence/` with checksums.

### P1-Open-2: External strict gate remains failing
- Impact: no release-ready external transfer claim.
- Evidence: both evidence tiers report `pass_all=false`.
- Suggested fix: execute `EXPERIMENT_PLAN.md` phase4b/4c/4d external-focused program.

### P2-Open-1: Documentation drift risk in archived files
- Impact: users may read stale historical docs as current status.
- Evidence: many archived files under `archive/project_docs_2026_02/`.
- Suggested fix: keep canonical index enforced and add periodic stale-doc checks.

## Prioritized Fix Plan (Next PR-Sized Chunks)

1. Evidence reconciliation package mirror and checksum index.
2. Seed-complete grouped-LCB rerun batch for top candidates.
3. Matched-protocol external baseline recalibration.
4. External-aware objective branch with stress-aware validation.
