# Reliability-First SAE Evaluation in HUSAI (Cycle 4 + Post-Fix Reruns)

Date: 2026-02-15

## Abstract
We present a reliability-first SAE research program that explicitly tests whether internal consistency improvements transfer to external benchmark gains. HUSAI integrates multiseed internal ablations, external evaluations (SAEBench, CE-Bench), stress tests (random-model, transcoder, OOD), and strict release gating. Across cycle3/cycle4 runs, internal consistency gains were reproducible, but external deltas remained negative for release-candidate settings under uncertainty-aware (LCB) criteria. The strict gate remained failing (`pass_all=false`). We additionally fixed two critical methodological bugs (custom-SAE dead-decoder normalization failures and known-circuit SAE overlap geometry mismatch) and validated both via post-fix reruns.

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
6. Post-fix reruns after bug corrections.

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

Evidence roots:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/`
- `docs/evidence/cycle4_postfix_reruns/`

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

### 4.3 Post-Fix Rerun Validation
Known-circuit closure rerun:
- `docs/evidence/cycle4_postfix_reruns/known_circuit_run_20260215T203809Z_summary.json`
- now evaluates 20 checkpoints (vs 0 in earlier artifact run).

Matryoshka rerun:
- `docs/evidence/cycle4_postfix_reruns/matryoshka/run_20260215T203710Z_results.json`
- no crash, no collapse (`l0=32`), full external outputs for all seeds.
- external deltas remain negative.

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
- Post-fix reruns resolve previously invalid failure modes.

### 6.2 Unsupported Claims
- External superiority for current candidates.
- Full proposal closure on joint internal+external improvement.

### 6.3 Main Scientific Insight
Internal consistency and external interpretability objectives are not automatically aligned; explicit multi-objective optimization remains necessary.

## 7. Limitations
- External-positive candidate not yet found under strict LCB gates.
- Assignment-v3 external stage needs dimension-compatible rerun.
- New-family exploration should be extended (e.g., RouteSAE) under matched budgets.

## 8. Reproducibility Checklist
- Entrypoint: `START_HERE.md`
- Runbook: `RUNBOOK.md`
- Experiment provenance: `EXPERIMENT_LOG.md`
- Reflective synthesis: `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
- Latest gate evidence: `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`

## 9. Conclusion
HUSAI is now a robust reliability-first SAE research platform. It delivers strong evidence for internal consistency progress while transparently showing that external transfer remains unresolved under strict controls. Post-fix reruns improved methodological validity and narrowed the remaining frontier.
