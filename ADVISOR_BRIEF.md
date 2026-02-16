# Advisor Brief: HUSAI Live State

Date: 2026-02-16

## 1) Core Question

Can we train SAEs that are both internally consistent across seeds and externally competitive under modern interpretability benchmarks (SAEBench + CE-Bench), with strict uncertainty-aware release gates?

## 2) Where We Are Now

- Queue status:
  - `cycle7`: running (`a2` assignment condition in progress)
  - `cycle8`: queued and waiting; will pull latest `main` automatically
- Engineering/reliability status: strong.
- Scientific status:
  - internal consistency improvements are real
  - external metrics still negative under strict gates

## 3) Strongest Current Evidence

- Cycle7 routed complete:
  - `results/experiments/phase4b_routed_frontier_external_sweep_cycle7_pareto/`
- Cycle7 assignment a1 complete:
  - `results/experiments/phase4d_assignment_consistency_v3_cycle7_pareto/run_20260216T142558Z/results.json`
  - best lambda `0.15`, internal LCB `0.83984`, SAEBench delta `-0.04355`, CE-Bench delta `-34.46848`
- Live monitoring snapshot:
  - `docs/evidence/cycle7_live_snapshot_20260216T165714Z/monitoring_summary.md`

## 4) Critical Fixes Applied Today

Commit `14b6c59`:
- fixed selection/gating handling of valid `0.0` values in assignment and selector logic
- filtered non-finite values from assignment normalization path

Commit `95f567c`:
- added interval-cached Hungarian updates for assignment training (`--assignment-update-interval`)
- threaded the flag through assignment-v3 and cycle queue scripts (`cycle4`..`cycle9`)
- added unit coverage for interval behavior
- full unit suite green (`104 passed`)

## 5) What Blocks "Finished"

1. External gates remain red under strict criteria.
2. Need one candidate with credible SAEBench + CE-Bench joint improvement under grouped-LCB selection.
3. Need closure on known-circuit confidence-bound criteria before final claim set.

## 6) Immediate Next Program (after cycle7/cycle8)

1. Assignment-v4 supervised external proxy objective.
2. Robust-routed expansion around cycle8 best condition.
3. PolySAE-inspired matched-budget run.
4. CI-LCB-first candidate selection and strict release gate rerun.
5. Known-circuit closure rerun with explicit trained-vs-random lower-bound thresholds.

Full plan: `CYCLE9_NOVELTY_PLAN.md`
