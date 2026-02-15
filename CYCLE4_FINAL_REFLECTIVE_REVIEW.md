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

## 3) Post-Fix Reruns in This Pass

### 3.1 Known-circuit closure rerun (fixed metric geometry)

Artifact:
- `docs/evidence/cycle4_postfix_reruns/known_circuit_run_20260215T203809Z_summary.json`

Result:
- Previously: `checkpoints_evaluated=0` (not scientifically useful).
- Now: `checkpoints_discovered=20`, `checkpoints_evaluated=20`, `skipped_dimension_mismatch=0`.
- Gates still fail (`pass_all=False`), but evidence is now valid and interpretable.

### 3.2 Matryoshka rerun (fixed training + adapter path)

Artifacts:
- `docs/evidence/cycle4_postfix_reruns/matryoshka/run_20260215T203710Z_summary.md`
- `docs/evidence/cycle4_postfix_reruns/matryoshka/run_20260215T203710Z_results.json`

Result:
- Previous cycle4 run: collapsed (`l0=0`) and adapter crash.
- New run: external eval succeeds for all 3 seeds.
- Aggregate metrics:
  - `train_ev_mean = 0.6166`
  - `train_l0_mean = 32.0`
  - `saebench_best_minus_llm_auc_mean = -0.03444`
  - `cebench_interpretability_mean = 7.7842`
  - `cebench_delta_vs_baseline_mean = -40.1674`

Interpretation:
- Crash is fixed and training is no longer degenerate.
- External deltas remain negative, so release gate remains blocked.

## 4) What Is Finished vs Not Finished

Finished (engineering/reliability):
- Reproducible queue orchestration, manifests, logs, evidence syncing, strict release gating.
- Grouped uncertainty-aware (LCB) candidate selection and policy wiring.
- External benchmark adapters and matched-baseline comparison paths.
- Fixed critical evaluation bugs (adapter + known-circuit geometry + Matryoshka training path).

Not finished (scientific):
- No release-eligible external-positive candidate yet.
- Assignment-v3 external stage still needs a `d_model`-compatible rerun.

## 5) Remaining Critical Issues

1. External gate still fails with large negative CE-Bench delta.
2. Assignment-v3 external-aware acceptance is unresolved due dimensional mismatch in current artifact run.
3. Strict gate remains failing; no candidate meets all acceptance criteria simultaneously.

## 6) What To Read (Fast Path)

1. `START_HERE.md`
2. `PROJECT_STUDY_GUIDE.md`
3. `EXECUTIVE_SUMMARY.md`
4. `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`
5. `docs/evidence/cycle4_postfix_reruns/known_circuit_run_20260215T203809Z_summary.md`
6. `docs/evidence/cycle4_postfix_reruns/matryoshka/run_20260215T203710Z_summary.md`
7. `RUNBOOK.md`

## 7) Are We Scientifically Finished?

Not yet.

We are close to finished on engineering/reproducibility and clear on the scientific bottleneck. We are not yet finished on the original goal of joint internal + external improvement under strict gates.

## 8) Recommended Immediate B200 Queue

1. Assignment-v3 rerun with external-compatible `d_model` setup.
2. RouteSAE matched-budget run (same SAEBench/CE-Bench protocol).
3. Grouped-LCB selection across frontier + scaling + new family.
4. OOD + transcoder stress on selected candidate.
5. Strict release gate rerun and canonical status refresh.

## 9) Claim Policy

No strong external claim unless strict release gate passes:
- `pass_all=True`
- with LCB-based external criteria.
