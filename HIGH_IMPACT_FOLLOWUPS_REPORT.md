# High-Impact Follow-Ups Report (Cycle 7/8/9 Live)

Date: 2026-02-16

## Current Queue Status

- `cycle7` run: `results/experiments/cycle7_pareto_push/run_20260216T062213Z`
  - routed stage `p1..p5`: complete
  - assignment `a1`: complete
  - assignment `a2`: complete
  - assignment `a3`: in progress (`run_20260216T201509Z`, checkpoints `46/56` at latest check)
- `cycle8` run: `results/experiments/cycle8_robust_pareto_push/run_20260216T163502Z`
  - waiting for cycle7 completion
  - will `git pull` before execution
- `cycle9` run: `results/experiments/cycle9_novelty_push/run_20260216T184628Z`
  - waiting behind cycle8/cycle7
  - will `git pull` before execution

## What Is Already Complete (artifact-backed)

1. Direct HUSAI-checkpoint CE-Bench adapter with matched baseline.
- Evidence: `docs/evidence/high_impact_adapter_check/run_20260214T202232Z_husai_custom_cebench_summary.json`

2. Matched-budget architecture frontier on external benchmarks.
- Evidence: `docs/evidence/cycle5_external_push_run_20260215T232351Z/routed/`
- Additional cycle7 routed sweep now complete in:
  - `results/experiments/phase4b_routed_frontier_external_sweep_cycle7_pareto/`

3. External-metric scaling study (`token budget`, `hook layer`, `d_sae`).
- Evidence: `docs/evidence/cycle3_queue_final/scaling_multiseed_results_run_20260214T212435Z.json`

4. Assignment-aware objective with external-aware checkpoint policy.
- Evidence: `scripts/experiments/run_assignment_consistency_v3.py`
- Cycle7 a1 output: `results/experiments/phase4d_assignment_consistency_v3_cycle7_pareto/run_20260216T142558Z/results.json`

5. Stress-gated release policy.
- Evidence: `scripts/experiments/run_stress_gated_release_policy.py`
- Prior strict-gate outputs: `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.json`

## New Engineering Hardening Added Today

Commit: `14b6c59`
- selector/ranking correctness hardening:
  - preserve valid `0.0` metrics in ranking/gating
  - ignore non-finite values in normalization paths

Commit: `95f567c`
- assignment training throughput hardening:
  - added interval-cached Hungarian update path in `run_assignment_consistency_v2.py`
  - threaded `--assignment-update-interval` into `run_assignment_consistency_v3.py`
  - wired cycle queues (`cycle4`..`cycle9`) with `ASSIGN_UPDATE_INTERVAL` env (default `4`)
  - added targeted unit tests: `tests/unit/test_assignment_consistency_v2.py`

Validation:
- `pytest -q tests/unit` -> `104 passed`
- `python -m py_compile scripts/experiments/run_assignment_consistency_v2.py scripts/experiments/run_assignment_consistency_v3.py`
- `bash -n scripts/experiments/run_cycle4_followups_after_queue.sh scripts/experiments/run_cycle5_external_push.sh scripts/experiments/run_cycle6_saeaware_push.sh scripts/experiments/run_cycle7_pareto_push.sh scripts/experiments/run_cycle8_robust_pareto_push.sh scripts/experiments/run_cycle9_novelty_push.sh`

## Current Scientific Bottom Line

- Internal consistency: strong and reproducible.
- External competitiveness: still below strict release gates.
- Main open problem: improve SAEBench and CE-Bench deltas jointly under grouped-LCB selection.

## Updated Highest-Leverage Next 5 (post cycle7/cycle8)

1. Assignment-v4 with supervised external proxy objective.
2. Robust-routed expansion around best cycle8 robust condition.
3. PolySAE-inspired matched-budget family run.
4. CI-LCB-first release selection as strict default everywhere.
5. Known-circuit closure as mandatory pre-release gate.

Detailed plan: `CYCLE9_NOVELTY_PLAN.md`
