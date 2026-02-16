# Reliability-First SAE Evaluation in HUSAI (Cycle 5)

Date: 2026-02-16

## Abstract
We present a reliability-first SAE research program that tests whether internal consistency improvements transfer to external benchmark gains. HUSAI integrates multiseed internal ablations, external evaluations (SAEBench, CE-Bench), stress tests (random-model, transcoder, OOD), and strict release gating. Across cycle-5 B200 runs, internal consistency gains remained reproducible, CE-Bench improved for routed and assignment sweeps, but external deltas remained negative at LCB thresholds for selected candidates. The strict gate remained failing (`pass_all=false`).

## 1. Introduction
SAE interpretability work often emphasizes internal metrics without robust external/stress controls. HUSAI enforces a strict gate over:
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
4. New families under matched budget (matryoshka, routed).
5. Transcoder/OOD stress evaluations.
6. Known-circuit closure track.

## 3. Experimental Setup

Compute:
- RunPod B200 queue execution for high-impact programs.

Cycle-5 external push stages:
1. Routed hyper-sweep (`expert_topk` + `global_mask` control).
2. Assignment-v3 external-aware sweep (`d_sae=1024,2048`).
3. Grouped-LCB reselection.
4. OOD + strict release gate.

Evidence root:
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/`

## 4. Results

### 4.1 Internal vs External
Internal consistency gains are reproducible; external transfer remains unresolved.

### 4.2 Routed-family correction
- `expert_topk` restored effective sparsity (`train_l0=32/48`) vs `global_mask` collapse (`train_l0≈4.3`).
- Best routed CE-Bench delta improved to `-37.260996`.
- SAEBench deltas remained negative.

### 4.3 Assignment-v3 external sweep
Best condition (`d_sae=2048`, `best_lambda=0.05`):
- `internal_lcb = 0.838793`
- `cebench_delta = -34.345572`
- `saebench_delta = -0.049864`
- `pass_all = false`

### 4.4 Strict release gate (cycle-5 canonical)
From `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.json`:
- `random_model=True`
- `transcoder=True`
- `ood=True`
- `external=False`
- `pass_all=False`

Metrics:
- `trained_random_delta_lcb = 0.00006183199584486321`
- `transcoder_delta = +0.004916101694107056`
- `ood_drop = 0.020994556554025268`
- `saebench_delta_ci95_low = -0.04478959689939781`
- `cebench_interp_delta_vs_baseline_ci95_low = -40.467037470119465`

## 5. Discussion

### 5.1 Supported claims
- Internal consistency gains are real and replicated.
- Reliability-first infrastructure and strict gating are effective.
- Routed implementation correction improves effective sparsity behavior.

### 5.2 Unsupported claims
- External superiority for current candidates.
- Full proposal closure on joint internal+external improvement.

### 5.3 Main scientific insight
Internal consistency and external interpretability objectives are not automatically aligned; explicit multi-objective optimization is required.

## 6. Limitations
- External-positive candidate not yet found under strict LCB gates.
- Selector sensitivity to group-size constraints can change winner identity.
- Known-circuit closure remains below trained-vs-random thresholds.

## 7. Reproducibility Checklist
- Entrypoint: `START_HERE.md`
- Runbook: `RUNBOOK.md`
- Experiment provenance: `EXPERIMENT_LOG.md`
- Reflective synthesis: `CYCLE5_EXTERNAL_PUSH_REFLECTIVE_REVIEW.md`
- Latest gate evidence: `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.md`

## 8. Conclusion
HUSAI is a robust reliability-first SAE research platform. It provides clear evidence of internal progress while transparently showing unresolved external transfer under strict controls.
