# Cycle 4 Final Reflective Review

Date: 2026-02-15

## 1) Original Goal vs What We Learned

Original goal:
- Find a reproducible SAE "Goldilocks zone" where feature stability is high and external interpretability metrics improve.

Evidence-backed outcome:
- Internal stability gains are real and reproducible.
- External benchmark deltas remain negative at strict LCB thresholds.
- Reliability/gating infrastructure is strong and prevents overclaiming.

## 2) Canonical Evidence (Latest)

- Followups manifest: `docs/evidence/cycle4_followups_run_20260215T220728Z/followups/manifest.json`
- Selection summary: `docs/evidence/cycle4_followups_run_20260215T220728Z/selector/selection_summary.json`
- Release gate: `docs/evidence/cycle4_followups_run_20260215T220728Z/release/release_policy.json`
- Assignment-v3 external: `docs/evidence/cycle4_followups_run_20260215T220728Z/assignment_external/results.json`
- Routed frontier: `docs/evidence/cycle4_followups_run_20260215T220728Z/routed/results.json`
- Matryoshka frontier: `docs/evidence/cycle4_followups_run_20260215T220728Z/matryoshka/results.json`
- Known-circuit closure: `docs/evidence/cycle4_followups_run_20260215T220728Z/known_circuit/closure_summary.json`

## 3) Latest Gate State

From `docs/evidence/cycle4_followups_run_20260215T220728Z/release/release_policy.json`:

- `pass_all=False`
- `random_model=True`
- `transcoder=True`
- `ood=True`
- `external_saebench=False`
- `external_cebench=False`

Key metrics:
- `trained_random_delta_lcb = 0.00006183199584486321`
- `transcoder_delta = +0.004916101694107056`
- `ood_drop = 0.015173514260201082`
- `saebench_delta_ci95_low = -0.04478959689939781`
- `cebench_interp_delta_vs_baseline_ci95_low = -40.467037470119465`

## 4) What Changed in This Latest Pass

1. Assignment-v3 external path completed.
- The prior dimensional-compatibility blocker is resolved.
- Best lambda selected (`0.3`) but does not satisfy external acceptance thresholds.

2. New family run completed: routed frontier.
- Added matched-budget routed track.
- Result: external deltas still negative; also low effective activation (`train_l0`), suggesting further tuning is needed.

3. Grouped LCB selector rerun on updated pool.
- Selected candidate remains `topk_seed123`.

## 5) Critical Findings

1. Internal gains do not guarantee external gains.
2. External gate is the dominant bottleneck.
3. Reliability controls are functioning as intended.
4. Current best candidate is robust on stress gates but weak on external metrics.

## 6) Remaining Gaps

1. No release-eligible candidate under strict external LCB criteria.
2. Known-circuit closure gates remain below threshold.
3. New-family routing track likely under-tuned and currently under-utilizes features.

## 7) What To Read Next

1. `EXECUTIVE_SUMMARY.md`
2. `PROJECT_STUDY_GUIDE.md`
3. `RUNBOOK.md`
4. `EXPERIMENT_LOG.md`

## 8) Claim Policy

No strong external claim unless strict release gate passes:
- `pass_all=True`
- with LCB-based external criteria.
