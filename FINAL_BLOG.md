# What We Learned After Cycle 4: Reliable SAE Research Without Overclaiming

Date: 2026-02-15

The core question stayed the same:

Can we make SAE features more consistent and also improve external interpretability benchmarks?

After the full B200 queue and cycle-4 followups, the answer is clearer.

## What We Executed

1. Matched-baseline CE-Bench adapter path for direct HUSAI checkpoints.
2. Multiseed external architecture frontier (`topk`, `relu`, `batchtopk`, `jumprelu`).
3. Multiseed external scaling study (`token budget`, `hook layer`, `d_sae`).
4. Assignment-aware objective tracks (v2 + v3), including external-aware v3 selection.
5. New architecture family: routed frontier.
6. Transcoder and OOD stress checks.
7. Strict release gate with grouped uncertainty-aware (LCB) candidate selection.

Evidence root:
- `docs/evidence/cycle4_followups_run_20260215T220728Z/`

## The Short Scientific Answer

- Internal consistency improvements: yes.
- External benchmark superiority: no.
- Strict release decision: fail (`pass_all=false`).

Latest gate evidence:
- `docs/evidence/cycle4_followups_run_20260215T220728Z/release/release_policy.md`

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
- `ood_drop = 0.015173514260201082`
- `saebench_delta_ci95_low = -0.04478959689939781`
- `cebench_delta_ci95_low = -40.467037470119465`

## What This Pass Added

1. Assignment-v3 external track is now complete.
- Best lambda: `0.3`.
- Internal margin is strong, but external deltas remain below thresholds.

2. Routed family is now benchmarked under matched budget.
- Run completed and integrated into selection.
- External deltas remain negative.

3. Candidate selection rerun stayed conservative.
- Grouped LCB selector still chooses `topk_seed123`.

## Why This Is Still a Strong Result

- We did not stop at internal metrics.
- We enforced external and stress gates before claims.
- We fixed real evaluation bugs and reran affected tracks.

This means future wins will be meaningful, not accidental.

## What Still Needs to Be Done

1. Tune routed-family regime (current `train_l0` suggests under-utilization).
2. Expand assignment-v3 external-aware sweep and Pareto checkpointing.
3. Re-run grouped-LCB selection and strict release gate on expanded pool.
4. Improve known-circuit closure above trained-vs-random thresholds.

## Final Takeaway

HUSAI is strong where many research repos are weak: reliability, reproducibility, and claim discipline.

The remaining challenge is a real research challenge: finding a method that wins internally and externally at the same time.
