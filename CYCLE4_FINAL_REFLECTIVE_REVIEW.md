# Cycle 4 Final Reflective Review

Date: 2026-02-15

## 1) Original Goal vs What We Actually Learned

Original research goal:
- Find a reproducible "Goldilocks zone" where SAEs recover stable, meaningful features and this stability translates into externally validated interpretability gains.

What is now evidence-backed:
- We can improve internal consistency metrics reproducibly.
- Those internal gains do not automatically transfer to SAEBench/CE-Bench gains.
- The tradeoff is structured, not random; strict stress gates expose it reliably.

## 2) Latest Cycle 4 Evidence (Canonical)

Cycle-4 followups manifest:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/followups/manifest.json`

Strict release gate:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.json`
- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`

Current gate state:
- `pass_all=False`
- `random_model=True`
- `transcoder=True`
- `ood=True`
- `external_saebench=False`
- `external_cebench=False`

Key gate metrics:
- `trained_random_delta_lcb = 6.183e-05`
- `transcoder_delta = +0.004916`
- `ood_drop = 0.020995`
- `saebench_delta_ci95_low = -0.044790`
- `cebench_interp_delta_vs_baseline_ci95_low = -40.467037`

## 3) What Is Finished vs Not Finished

Finished (engineering/reliability):
- Reproducible queue orchestration, manifests, logs, evidence syncing, strict release gating.
- Grouped uncertainty-aware (LCB) candidate selection and policy wiring.
- External benchmark adapters and matched-baseline comparison paths.

Not finished (scientific):
- No release-eligible external-positive candidate yet.
- Known-circuit closure not complete in current published artifacts.
- New architecture family track (Matryoshka) did not produce valid external eval in cycle4 evidence.

## 4) Critical Issues Found in Cycle 4 Artifacts

1. Matryoshka external eval failed for all seeds.
- Evidence: `docs/evidence/cycle4_followups_run_20260215T190004Z/matryoshka/logs/*.log`
- Failure: `ValueError: Failed to normalize decoder rows for custom SAE`
- Train signal in that run: `l0=0.0` (collapsed features).

2. Assignment-v3 external evaluation was skipped.
- Evidence: `docs/evidence/cycle4_followups_run_20260215T190004Z/assignment_v3/results.json`
- Reason: `d_model_mismatch checkpoint=128 external_cache=512`
- Implication: external-aware acceptance in that run is not informative.

3. Known-circuit closure did not evaluate SAE checkpoints meaningfully.
- Evidence: `docs/evidence/cycle4_followups_run_20260215T190004Z/known_circuit/closure_summary.json`
- Reported `checkpoints_evaluated=0` and failed gates.

## 5) High-Impact Fixes Applied in This Pass

1. Adapter robustness fix.
- File: `scripts/experiments/husai_custom_sae_adapter.py`
- Added deterministic dead-decoder-row repair + encoder masking before norm checks.

2. Known-circuit geometry fix.
- File: `scripts/experiments/run_known_circuit_recovery_closure.py`
- SAE overlap now uses model-space Fourier basis from token-space projection via embedding matrix.
- Added explicit skipped-checkpoint reason accounting.

3. Matryoshka training stabilization path.
- File: `scripts/experiments/run_matryoshka_frontier_external.py`
- Uses HUSAI `TopKSAE` with auxiliary dead-feature recovery objective instead of bare custom TopK path.

4. Unit-test coverage for both bug classes.
- Files:
  - `tests/unit/test_husai_custom_sae_adapter.py`
  - `tests/unit/test_known_circuit_recovery_closure.py`

## 6) What To Read (If You Want Full Understanding Fast)

1. `START_HERE.md`
2. `PROJECT_STUDY_GUIDE.md`
3. `EXECUTIVE_SUMMARY.md`
4. `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`
5. `docs/evidence/cycle4_followups_run_20260215T190004Z/transcoder_sweep/summary.md`
6. `docs/evidence/cycle4_followups_run_20260215T190004Z/ood/ood_stress_summary.md`
7. `docs/evidence/cycle4_followups_run_20260215T190004Z/matryoshka/summary.md`
8. `docs/evidence/cycle4_followups_run_20260215T190004Z/assignment_v3/summary.md`
9. `docs/evidence/cycle4_followups_run_20260215T190004Z/known_circuit/closure_summary.md`
10. `RUNBOOK.md`

## 7) Are We Satisfied / "Project Finished"?

Not yet scientifically.

We are close to "finished" on engineering hygiene and reproducibility. We are not yet finished on the core scientific promise (joint internal+external improvement with strict gate pass).

## 8) Recommended Immediate Experiment Queue (B200)

1. Matryoshka rerun (post-fix) under matched budget.
- Goal: verify non-collapsed `l0`, successful SAEBench/CE-Bench adapter execution.

2. Known-circuit closure rerun (post-fix).
- Goal: obtain non-empty SAE overlap statistics with CIs.

3. Assignment-v3 external-compatible run.
- Goal: run v3 where checkpoint `d_model` matches external cache/model.

4. New architecture family run (RouteSAE) under same protocol.
- Goal: test whether external frontier can improve without violating internal/OOD/transcoder gates.

5. Strict gate rerun from grouped-LCB selected candidate.
- Goal: update final `pass_all` truth after fixes.

## 9) Claim Policy (Unchanged and Strict)

No strong external claim unless strict release gate passes:
- `pass_all=True`
- with LCB-based external criteria.
