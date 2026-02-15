# Reliability-First SAE Evaluation in HUSAI (Cycle 4 Update)

Date: 2026-02-15

## Abstract
We present a reliability-first SAE research program that explicitly tests whether internal consistency improvements transfer to external benchmark gains. HUSAI integrates multiseed internal ablations, external evaluations (SAEBench, CE-Bench), stress tests (random-model, transcoder, OOD), and strict release gating. Across cycle3 and cycle4 runs, internal consistency gains were reproducible, but external deltas remained negative for release-candidate settings under uncertainty-aware (LCB) criteria. The strict gate remained failing (`pass_all=false`). We also identify and fix two critical evaluation risks: custom-SAE dead-decoder normalization failures and a known-circuit SAE overlap geometry mismatch. The resulting repository is methodologically strong and reproducible, while scientific closure on external competitiveness remains open.

## 1. Introduction
SAE interpretability research often emphasizes internal metrics without robust external or stress controls. HUSAI addresses this by enforcing a strict gate over:
- random-model baseline,
- transcoder stress,
- OOD robustness,
- external benchmark deltas.

Primary question:
- Can internal consistency gains be achieved without external regressions?

## 2. Method

### 2.1 Reliability Protocol
- Reproducible run manifests + config hashing.
- Matched-baseline external comparisons.
- Grouped uncertainty-aware (LCB) candidate selection.
- Strict release policy and fail-fast semantics.

### 2.2 Experiment Tracks
1. Trained-vs-random and core ablations.
2. Assignment-aware consistency objectives (v2/v3).
3. External architecture frontier and scaling studies.
4. Transcoder/OOD stress evaluations.
5. Known-circuit closure track.

## 3. Experimental Setup

Compute:
- B200 queue execution for high-impact programs.

External frontier:
- Architectures: `topk`, `relu`, `batchtopk`, `jumprelu`
- Multiseed matched-budget evaluation.

Scaling:
- token budgets: `10k`, `30k`
- hook layers: `0`, `1`
- widths: `d_sae=1024,2048`

Cycle4 followup root:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/`

## 4. Results

### 4.1 Internal vs External
Internal consistency gains are reproducible across tracks, but external improvements do not follow automatically.

### 4.2 Strict Release Gate (Latest)
From `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.json`:
- `random_model=True`
- `transcoder=True`
- `ood=True`
- `external=False`
- `pass_all=False`

Key metrics:
- `transcoder_delta = +0.004916101694107056`
- `ood_drop = 0.020994556554025268`
- `saebench_delta_ci95_low = -0.04478959689939781`
- `cebench_interp_delta_vs_baseline_ci95_low = -40.467037470119465`

### 4.3 New-Family and Closure Status
- Matryoshka run in cycle4 artifacts failed due dead-feature collapse and adapter normalization failure.
- Assignment-v3 external stage skipped in latest artifact due `d_model` mismatch.
- Known-circuit closure artifact run was pre-fix and not final.

## 5. Engineering Corrections Applied

1. `scripts/experiments/husai_custom_sae_adapter.py`
- Added deterministic dead-decoder-row repair and encoder masking before norm checks.

2. `scripts/experiments/run_known_circuit_recovery_closure.py`
- Corrected SAE overlap metric to model-space Fourier projection.
- Added skipped-checkpoint reason accounting.

3. `scripts/experiments/run_matryoshka_frontier_external.py`
- Switched to HUSAI TopK training path with dead-feature recovery auxiliary objective.

4. Added unit tests:
- `tests/unit/test_husai_custom_sae_adapter.py`
- `tests/unit/test_known_circuit_recovery_closure.py`

## 6. Discussion

### 6.1 Supported Claims
- Internal consistency gains are real.
- Reliability-first infrastructure and strict claim gating are effective.

### 6.2 Unsupported Claims
- External superiority for current candidates.
- Full closure on known-circuit recovery from current published artifacts.

### 6.3 Main Scientific Insight
Internal consistency and external interpretability objectives are not automatically aligned; explicit multi-objective optimization remains necessary.

## 7. Limitations
- External-positive candidate not yet found under strict LCB gates.
- New-family frontier still requires post-fix rerun.
- Known-circuit closure requires post-fix rerun for final interpretation.

## 8. Reproducibility Checklist
- Entrypoint: `START_HERE.md`
- Runbook: `RUNBOOK.md`
- Experiment provenance: `EXPERIMENT_LOG.md`
- Latest reflective synthesis: `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
- Latest gate evidence: `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`

## 9. Conclusion
HUSAI is now a robust reliability-first SAE research platform. It delivers strong evidence for internal consistency progress while transparently showing that external transfer remains unresolved under strict controls. This narrows the true frontier and sets up a clear next experiment program.
