# Reliability-First SAE Evaluation in HUSAI (Cycle 4)

Date: 2026-02-15

## Abstract
We present a reliability-first SAE research program that explicitly tests whether internal consistency improvements transfer to external benchmark gains. HUSAI integrates multiseed internal ablations, external evaluations (SAEBench, CE-Bench), stress tests (random-model, transcoder, OOD), and strict release gating. Across cycle-4 runs on B200 compute, internal consistency gains were reproducible, but external deltas remained negative for release-candidate settings under uncertainty-aware (LCB) criteria. The strict gate remained failing (`pass_all=false`).

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
4. New routed family under matched budget.
5. Transcoder/OOD stress evaluations.
6. Known-circuit closure track.

## 3. Experimental Setup

Compute:
- B200 queue execution for high-impact programs.

External frontier:
- Architectures: `topk`, `relu`, `batchtopk`, `jumprelu`, `matryoshka`, `routed_topk`
- Multiseed matched-budget evaluation.

Scaling:
- token budgets: `10k`, `30k`
- hook layers: `0`, `1`
- widths: `d_sae=1024,2048`

Evidence root:
- `docs/evidence/cycle4_followups_run_20260215T220728Z/`

## 4. Results

### 4.1 Internal vs External
Internal consistency gains are reproducible across tracks, but external improvements do not follow automatically.

### 4.2 Strict Release Gate (Latest)
From `docs/evidence/cycle4_followups_run_20260215T220728Z/release/release_policy.json`:
- `random_model=True`
- `transcoder=True`
- `ood=True`
- `external=False`
- `pass_all=False`

Key metrics:
- `trained_random_delta_lcb = 0.00006183199584486321`
- `transcoder_delta = +0.004916101694107056`
- `ood_drop = 0.015173514260201082`
- `saebench_delta_ci95_low = -0.04478959689939781`
- `cebench_interp_delta_vs_baseline_ci95_low = -40.467037470119465`

### 4.3 Assignment-v3 External Completion
From `docs/evidence/cycle4_followups_run_20260215T220728Z/assignment_external/results.json`:
- Best lambda selected: `0.3`
- Internal LCB signal remains strong
- External SAEBench/CE-Bench deltas remain negative under acceptance thresholds

### 4.4 Routed Family Addition
From `docs/evidence/cycle4_followups_run_20260215T220728Z/routed/results.json`:
- Routed family runs complete under matched budget
- External deltas remain negative
- Effective activation (`train_l0`) indicates a likely under-tuned routing regime

## 5. Discussion

### 5.1 Supported Claims
- Internal consistency gains are real.
- Reliability-first infrastructure and strict claim gating are effective.
- External gap remains robust under grouped LCB selection.

### 5.2 Unsupported Claims
- External superiority for current candidates.
- Full proposal closure on joint internal+external improvement.

### 5.3 Main Scientific Insight
Internal consistency and external interpretability objectives are not automatically aligned; explicit multi-objective optimization remains necessary.

## 6. Limitations
- External-positive candidate not yet found under strict LCB gates.
- Known-circuit closure remains below target thresholds.
- Routed-family hyperparameters require further tuning before fair final judgment.

## 7. Reproducibility Checklist
- Entrypoint: `START_HERE.md`
- Runbook: `RUNBOOK.md`
- Experiment provenance: `EXPERIMENT_LOG.md`
- Reflective synthesis: `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
- Latest gate evidence: `docs/evidence/cycle4_followups_run_20260215T220728Z/release/release_policy.md`

## 8. Conclusion
HUSAI is now a robust reliability-first SAE research platform. It delivers strong evidence for internal consistency progress while transparently showing that external transfer remains unresolved under strict controls.
