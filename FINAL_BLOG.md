# What We Learned After Cycle 4: Reliable SAE Research Without Overclaiming

Date: 2026-02-15

The core question remained the same:

Can we make SAE features more consistent and also improve external interpretability benchmarks?

After running the B200 program, cycle4 followups, and post-fix reruns, the answer is clearer.

## What We Executed

1. Matched-baseline CE-Bench adapter path for direct HUSAI checkpoints.
2. Multiseed external architecture frontier (`topk`, `relu`, `batchtopk`, `jumprelu`).
3. Multiseed external scaling study (token budget, hook layer, `d_sae`).
4. Assignment-aware objective tracks (v2 and v3).
5. Transcoder and OOD stress checks.
6. Strict release gate with grouped uncertainty-aware (LCB) candidate selection.
7. Post-fix reruns for known-circuit closure and Matryoshka frontier.

Evidence roots:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/`
- `docs/evidence/cycle4_postfix_reruns/`

## The Short Scientific Answer

- Internal consistency improvements: yes.
- External benchmark superiority: no.
- Strict release decision: fail (`pass_all=false`).

Latest gate evidence:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`

## What Passed and What Failed

Passed:
- random-model gate
- transcoder gate (after hyper-sweep)
- OOD gate

Failed:
- SAEBench external gate
- CE-Bench external gate

Key values:
- `transcoder_delta = +0.004916`
- `ood_drop = 0.020995`
- `saebench_delta_ci95_low = -0.044790`
- `cebench_delta_ci95_low = -40.467037`

## What the Post-Fix Reruns Changed

1. Known-circuit closure is now measurable (not empty).
- We now evaluate 20 SAE checkpoints (previous artifact had 0).
- Gate still fails, but the metric is now valid.

2. Matryoshka no longer collapses/crashes.
- Previous cycle4 artifact: `l0=0`, adapter normalization crash.
- Post-fix rerun: `l0=32`, full SAEBench/CE-Bench outputs for all 3 seeds.
- External deltas are still negative.

## Why This Is Still a Strong Outcome

This is valuable because it is hard to fake:
- We did not stop at internal metrics.
- We forced claims through external and stress gates.
- We found and fixed concrete methodological bugs.

That makes future improvements scientifically meaningful.

## What Still Needs to Be Done

1. Assignment-v3 rerun with external-compatible dimensional setup.
2. Add RouteSAE family under matched-budget protocol.
3. Re-run grouped-LCB candidate selection with new family included.
4. Re-run strict gate and update canonical status.

## Final Takeaway

HUSAI is now strong where many research repos are weak: reliability, reproducibility, and claim discipline.

The remaining challenge is a real research challenge: finding a method that wins internally and externally at the same time.
