# Executive Summary (Cycle 4 Reflective Update)

Date: 2026-02-15

## Repo Purpose

HUSAI tests a central mechanistic-interpretability question: whether SAE feature consistency gains are real across seeds, and whether those gains transfer to external benchmark validity (SAEBench, CE-Bench) under strict release gates.

## Current Scientific Bottom Line

- Internal consistency progress: **real and replicated**.
- External competitiveness: **not yet achieved**.
- Reliability and reproducibility hygiene: **strong**.
- Strict release decision: **fail** (`pass_all=False`).

Canonical current-status artifacts:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`
- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.json`
- `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`

## Cycle 4 Key Metrics (Latest)

Run IDs:
- Followups orchestrator: `run_20260215T190004Z`
- Release gate: `run_20260215T191137Z`
- Transcoder sweep: `run_20260215T184609Z`
- OOD stress: `run_20260215T190404Z`

Gate outcomes:
- random_model: `True`
- transcoder: `True`
- ood: `True`
- external_saebench: `False`
- external_cebench: `False`
- pass_all: `False`

Numerical highlights:
- `trained_random_delta_lcb = 6.18e-05`
- `transcoder_delta = +0.0049161`
- `ood_drop = 0.0209946`
- `saebench_delta_ci95_low = -0.0447896`
- `cebench_interp_delta_vs_baseline_ci95_low = -40.4670`

## Top 10 Issues (Severity-Ranked)

1. `P0` External gate fails with large negative CE-Bench delta.
2. `P0` No candidate currently satisfies both external gates jointly.
3. `P0` Matryoshka frontier run collapsed (`l0=0`) and external eval crashed in cycle4 artifacts.
4. `P1` Assignment-v3 external stage skipped (`d_model` mismatch), so external claims are unresolved there.
5. `P1` Known-circuit closure artifacts are not trustworthy yet (pre-fix basis mismatch path).
6. `P1` Canonical docs were cycle3-pinned and drifted from latest evidence.
7. `P1` W&B coverage is inconsistent across experiment scripts (some runs are file-artifact only).
8. `P2` Determinism warnings around CuBLAS were present in queue logs.
9. `P2` Historical docs can still be mistaken as canonical if entrypoint is ignored.
10. `P2` External benchmark protocol variants can drift unless baseline-map strictness is enforced.

## What Changed in This Update

Code fixes:
- `scripts/experiments/husai_custom_sae_adapter.py`
  - Added dead-decoder-row repair + encoder masking before SAEBench/CE-Bench adapter checks.
- `scripts/experiments/run_known_circuit_recovery_closure.py`
  - Fixed SAE Fourier overlap geometry to use **model-space projected Fourier basis**.
  - Added skip-reason accounting for checkpoint decode/dimension failures.
- `scripts/experiments/run_matryoshka_frontier_external.py`
  - Switched Matryoshka training to HUSAI `TopKSAE` (with auxiliary dead-feature revival) to prevent all-dead collapse.

Tests added:
- `tests/unit/test_husai_custom_sae_adapter.py`
- `tests/unit/test_known_circuit_recovery_closure.py`

Documentation sync:
- `START_HERE.md`, `README.md`, `PROJECT_STUDY_GUIDE.md`, `REPO_NAVIGATION.md`, `HIGH_IMPACT_FOLLOWUPS_REPORT.md`, `PROPOSAL_COMPLETENESS_REVIEW.md`, `FINAL_READINESS_REVIEW.md`, `ADVISOR_BRIEF.md`
- New canonical reflective synthesis: `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`

## Highest-Leverage Next 5 (Ranked)

1. Re-run Matryoshka frontier with fixed training+adapter and compare against topk grouped-LCB candidate.
2. Re-run known-circuit closure with fixed model-space basis and report trained-vs-random CIs.
3. Run assignment-v3 on external-compatible activations (`d_model` matched) so external gates are actually evaluated.
4. Add RouteSAE family under matched budget protocol and grouped-LCB selection.
5. Make grouped-LCB + strict joint external gate the only release-eligible path in all queue scripts.

## Read Next

1. `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
2. `START_HERE.md`
3. `PROJECT_STUDY_GUIDE.md`
4. `RUNBOOK.md`
5. `EXPERIMENT_LOG.md`
