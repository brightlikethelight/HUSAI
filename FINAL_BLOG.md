# What We Learned After Cycle 5: Reliable SAE Research Without Overclaiming

Date: 2026-02-16

The core question stayed the same:

Can we make SAE features more consistent and also improve external interpretability benchmarks?

After cycle-5 external push on B200, the answer is sharper.

## What We Executed

1. Routed-family hyper-sweep with a new `expert_topk` mode.
2. Assignment-v3 external-aware sweep at higher capacity (`d_sae=2048`).
3. Grouped LCB reselection with assignment integration.
4. OOD + strict release gate rerun on the selected checkpoint.

Evidence root:
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/`

## The Short Scientific Answer

- Internal consistency improvements: yes.
- CE-Bench improved in routed/assignment sweeps: yes.
- External benchmark superiority under strict gate: no.
- Strict release decision: fail (`pass_all=false`).

Latest gate evidence:
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.md`

## What Passed and What Failed

Passed:
- random-model gate
- transcoder gate
- OOD gate

Failed:
- SAEBench external gate
- CE-Bench external gate

Key values:
- `trained_random_delta_lcb = 0.00006183199584486321`
- `transcoder_delta = +0.004916101694107056`
- `ood_drop = 0.020994556554025268`
- `saebench_delta_ci95_low = -0.04478959689939781`
- `cebench_interp_delta_vs_baseline_ci95_low = -40.467037470119465`

## What Cycle 5 Added

1. Routed sparsity pathology fix.
- `expert_topk` restored effective sparsity (`l0=32/48`) vs legacy `global_mask` collapse (`l0≈4.3`).
- Best routed CE-Bench delta improved to `-37.260996`.

2. Assignment external track improved CE-Bench materially.
- Best assignment run: `d_sae=2048`, `best_lambda=0.05`.
- `cebench_delta = -34.345572`, better than prior baseline candidate.
- SAEBench remained negative (`-0.049864`).

3. Selection-policy sensitivity is now explicit.
- Default grouped selector (`min_seeds_per_group=3`) selected baseline `topk`.
- Relaxed check (`min_seeds_per_group=2`) selected assignment group.
- Neither candidate passes strict external-positive thresholds.

## Why This Is Still a Strong Result

- We did not stop at internal metrics.
- We enforced external and stress gates before claims.
- We exposed policy sensitivity (group threshold effects) instead of hiding it.
- We fixed orchestration bugs and reran decision-critical stages.

This means future wins will be meaningful, not accidental.

## What Still Needs to Be Done

1. Improve SAEBench without giving back CE-Bench/internal consistency.
2. Expand assignment/routed seeds to stabilize grouped-LCB selection.
3. Add SAEBench-aware objective terms (not only CE-focused improvements).
4. Close known-circuit track with trained-vs-random confidence bounds.

## Final Takeaway

HUSAI is strong where many research repos are weak: reliability, reproducibility, and claim discipline.

The remaining challenge is a real research challenge: finding a method that wins internally and externally at the same time under strict gates.
