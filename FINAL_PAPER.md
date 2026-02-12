# Reliability-First Reproduction of SAE Stability Research in HUSAI

## Abstract
We present a reliability-first execution of the HUSAI SAE-stability repository. The original codebase contained important research artifacts but also execution blockers in script pathing, SAE integration, and test/documentation consistency. We performed a structured audit, repaired core training/extraction/SAE pathways, and established a reproducible baseline workflow with passing tests and end-to-end smoke validation. This work does not claim final new state-of-the-art stability numbers yet; instead, it establishes the engineering and evaluation substrate required for credible multi-seed SAE stability science. We provide a literature-grounded experiment program aligned with current benchmark practice.

## 1. Introduction
Sparse autoencoders are widely used in mechanistic interpretability, but cross-seed consistency remains an active concern in recent literature. Reliable empirical conclusions require both valid metrics and reproducible execution. In this repository, core reliability issues were preventing robust experimentation from documented entrypoints.

## 2. Repository Objective
The repository's intended objective is to measure SAE feature stability (primarily with PWMCC-style decoder matching) under controlled algorithmic tasks and relate stability to sparsity, parameterization, and reconstruction quality.

## 3. Methods: Reliability and Audit Procedure
We used a staged process:
1. architecture and dependency mapping
2. correctness/reproducibility audit with concrete runtime checks
3. targeted P0/P1 fixes on execution-critical surfaces
4. revalidation with unit/integration/e2e smoke tests
5. literature-grounded experiment-plan reconstruction

## 4. Implementation Fixes
Key fixes implemented:
- corrected script-root path bootstrapping in core scripts
- corrected SAE CLI extraction import path
- replaced fragile SAE wrapper coupling with stable local SAE wrapper path
- harmonized transformer checkpoint extras/model interface expectations
- fixed config test drift and pipeline script API drift
- modernized Makefile/run command targets

## 5. Validation Results
Post-fix validation:
- `pytest tests -q`: 83 passed
- baseline training smoke command succeeds
- activation extraction smoke command succeeds
- SAE training smoke command succeeds
- pipeline script `tests/test_sae_pipeline.py` now runs end-to-end

These are engineering-validity results, not final scientific leaderboard claims.

## 6. Literature Alignment
We grounded the next-stage program using recent primary sources on:
- consistency-focused SAE analysis
- scaling/evaluation frameworks
- benchmark suites (SAEBench/CE-Bench)
- architecture variants and theory-grounded methods

Details and citations are in `LIT_REVIEW.md`.

## 7. Limitations
- no new full multi-seed benchmark campaign was executed in this pass
- benchmark integration with SAEBench/CE-Bench is planned, not yet implemented
- environment remains sensitive on this machine (`KMP_DUPLICATE_LIB_OK`, `TMPDIR` workaround)

## 8. Conclusion
The repository has been moved from partially non-executable to reproducible baseline operation. The highest leverage next work is scientific execution of the staged experiment plan (`EXPERIMENT_PLAN.md`) now that the reliability substrate is in place.

## Reproducibility Checklist
- runbook: `RUNBOOK.md`
- architecture map: `ARCHITECTURE.md`
- audit: `AUDIT.md`
- bug list: `BUGS.md`
- experiment plan: `EXPERIMENT_PLAN.md`
- command/run log: `EXPERIMENT_LOG.md`
- literature grounding: `LIT_REVIEW.md`
