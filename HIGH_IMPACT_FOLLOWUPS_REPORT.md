# High-Impact Follow-Ups Report (Cycle 7/8/9 Live)

Date: 2026-02-17

## Current Queue Status

- `cycle7` run (`results/experiments/cycle7_pareto_push/run_20260216T062213Z`): complete.
- `cycle8` run (`results/experiments/cycle8_robust_pareto_push/run_20260216T163502Z`): active.
  - routed stage complete (`b0`, `r1`, `r2`, `r3`, `r4`)
  - assignment `a1` complete (`run_20260217T061919Z`)
  - assignment `a2` complete (`run_20260217T084709Z`)
  - assignment `a3` active (`run_20260217T111709Z`, checkpoints=33 at 2026-02-17T13:36:19Z)
- `cycle9` run (`results/experiments/cycle9_novelty_push/run_20260217T052929Z`): active-waiting behind cycle8.
  - supervised-proxy assignment config:
    - `SUPERVISED_PROXY_MODE=file_id`
    - `SUPERVISED_PROXY_WEIGHT=0.10`
    - `SUPERVISED_PROXY_NUM_CLASSES=0`

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

Commit: `eca2c32`
- assignment supervised-proxy extension:
  - added optional file-ID supervised proxy loss/metrics in `run_assignment_consistency_v2.py`
  - added external-cache file-label loading mode in `run_assignment_consistency_v3.py`
  - wired `cycle9` assignment stage to pass supervised-proxy flags
  - added unit coverage in `tests/unit/test_assignment_consistency_v2.py` and `tests/unit/test_assignment_consistency_v3.py`

Commit: `d1ac12d`
- queue reliability hardening:
  - replaced broad queue wait `pgrep` patterns with anchored process checks in `run_cycle8_robust_pareto_push.sh` and `run_cycle9_novelty_push.sh`
  - avoids stale wrapper false positives that can stall queue progression

Validation:
- `pytest -q tests/unit` -> `106 passed`
- `python -m py_compile scripts/experiments/run_assignment_consistency_v2.py scripts/experiments/run_assignment_consistency_v3.py`
- `bash -n scripts/experiments/run_cycle4_followups_after_queue.sh scripts/experiments/run_cycle5_external_push.sh scripts/experiments/run_cycle6_saeaware_push.sh scripts/experiments/run_cycle7_pareto_push.sh scripts/experiments/run_cycle8_robust_pareto_push.sh scripts/experiments/run_cycle9_novelty_push.sh`

## Current Scientific Bottom Line

- Internal consistency: strong and reproducible.
- External competitiveness: still below strict release gates.
- Main open problem: improve SAEBench and CE-Bench deltas jointly under grouped-LCB selection.

## Cycle8 Interim Evidence (important)

Routed stage summaries (`b0`/`r1`/`r2`/`r3`/`r4`) show:
- Best SAEBench delta so far: `-0.06319` (`r4`, `run_20260217T060602Z`)
- Best CE-Bench delta so far: `-36.18278` (`r4`, `run_20260217T060602Z`)
- Neither meets strict external-positive release criteria yet.

Assignment stage summaries show:
- `a1` best lambda `0.10`: `saebench=-0.04060`, `cebench=-34.86151`, `ev_drop=0.27625`
- `a2` best lambda `0.05`: `saebench=-0.03976`, `cebench=-35.48915`, `ev_drop=0.24146`
- Both fail `gate_saebench` and `gate_ev_drop`; both pass `gate_cebench`.
- `a3` is now the key near-term decision point.

Primary snapshot: `docs/evidence/cycle8_cycle9_live_snapshot_20260217T1334Z/monitoring_summary.md`

## Updated Highest-Leverage Next 5 (post cycle7/cycle8)

1. Assignment-v4 with supervised external proxy objective.
2. Robust-routed expansion around best cycle8 robust condition.
3. PolySAE-inspired matched-budget family run.
4. CI-LCB-first release selection as strict default everywhere.
5. Known-circuit closure as mandatory pre-release gate.

Detailed plan: `CYCLE9_NOVELTY_PLAN.md`
