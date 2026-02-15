# What We Learned After Cycle 4: Reliable SAE Research Without Overclaiming

Date: 2026-02-15

The core question remained the same:

Can we make SAE features more consistent and also improve external interpretability benchmarks?

After running the B200 program and cycle4 followups, the answer is now clearer.

## What We Executed

1. Matched-baseline CE-Bench adapter path for direct HUSAI checkpoints.
2. Multiseed external architecture frontier (`topk`, `relu`, `batchtopk`, `jumprelu`).
3. Multiseed external scaling study (token budget, hook layer, `d_sae`).
4. Assignment-aware objective tracks (v2 and v3).
5. Transcoder and OOD stress checks.
6. Strict release gate with grouped uncertainty-aware (LCB) candidate selection.

Cycle4 canonical artifact root:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/`

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

## Why This Is Still a Strong Outcome

This is a high-value result because it is hard to fake:
- We did not stop at internal metrics.
- We forced claims through external and stress gates.
- We identified concrete failure modes and fixed real code issues.

That makes the next iterations scientifically meaningful, not cosmetic.

## Critical Fixes We Added

1. Robust custom-SAE adapter handling for dead decoder rows.
- File: `scripts/experiments/husai_custom_sae_adapter.py`

2. Corrected known-circuit overlap geometry.
- File: `scripts/experiments/run_known_circuit_recovery_closure.py`
- SAE overlap now uses model-space projected Fourier basis.

3. Matryoshka training path stabilization.
- File: `scripts/experiments/run_matryoshka_frontier_external.py`
- Uses HUSAI TopK with dead-feature recovery auxiliary objective.

4. Unit tests for both bug classes.
- `tests/unit/test_husai_custom_sae_adapter.py`
- `tests/unit/test_known_circuit_recovery_closure.py`

## What Still Needs to Be Done

1. Re-run Matryoshka frontier post-fix.
2. Re-run known-circuit closure post-fix.
3. Re-run assignment-v3 with external-compatible dimensions.
4. Add RouteSAE family under matched-budget protocol.
5. Re-run strict gate and update canonical status from fresh artifacts.

## Final Takeaway

HUSAI is now strong where many research repos are weak: reliability, reproducibility, and claim discipline.

The remaining challenge is a real research challenge: finding a method that wins internally and externally at the same time.
