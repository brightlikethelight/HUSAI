# Reliability-First SAE Evaluation in HUSAI

Date: 2026-02-15

## Abstract
We executed an end-to-end reliability-first experiment program for sparse autoencoders (SAEs) in HUSAI to test whether internal consistency improvements transfer to external benchmark gains. The program covered direct HUSAI-checkpoint CE-Bench integration with matched baseline comparisons, a multiseed architecture frontier on SAEBench/CE-Bench, a multiseed external scaling study (`token budget`, `hook layer`, `d_sae`), assignment-aware consistency objective v2, and strict stress-gated release checks. Internal consistency improvements were reproducible, but external gains remained negative relative to matched baselines and LLM references. Final strict gates failed (`pass_all=false`) due transcoder and external criteria. These results support methodological rigor and reproducibility, but do not support external superiority claims.

## 1. Introduction
SAE interpretability research often overindexes on internal metrics without fully stress-testing external validity. HUSAI targets this gap with a claim-gated workflow: every major claim must tie to artifact-backed metrics and pass strict release criteria.

Primary question:
- Can internal consistency gains be achieved without external regressions?

## 2. Method

### 2.1 Reliability Protocol
- Reproducible run manifests and config hashing.
- Matched-baseline external comparisons.
- Stress gates for random-model, transcoder, OOD, external metrics.
- Consistency audit linking narrative claims to artifacts.

### 2.2 Experiment Tracks
1. Trained-vs-random baseline and core ablations.
2. Assignment-aware consistency objective v2.
3. Direct custom checkpoint external eval (SAEBench + CE-Bench).
4. Architecture frontier (multiseed).
5. External scaling study (multiseed).
6. Stress and strict release policy.

## 3. Experimental Setup

Compute and orchestration:
- B200 queue execution for cycle-3 high-impact runs.

External frontier setup:
- Architectures: `topk,relu,batchtopk,jumprelu`
- Seeds: `42,123,456,789,1011`
- Matched CE-Bench budget with baseline-comparable protocol.

Scaling setup:
- Token budgets: `10000,30000`
- Hook layers: `0,1`
- `d_sae`: `1024,2048`
- Seeds: `42,123,456`

Primary synthesis artifact:
- `docs/evidence/cycle3_queue_final/cycle3_final_synthesis_run_20260214T210734Z.md`

## 4. Results

### 4.1 Architecture Frontier (Multiseed)
Source: `docs/evidence/cycle3_queue_final/frontier_multiseed_results_run_20260214T202538Z.json`

SAEBench best-minus-LLM AUC mean deltas:
- `relu = -0.024691`
- `jumprelu = -0.030577`
- `topk = -0.040593`
- `batchtopk = -0.043356`

CE-Bench interpretability means:
- `topk = 7.726768`
- `batchtopk = 6.537639`
- `jumprelu = 4.379002`
- `relu = 4.257686`

Interpretation:
- Benchmarks favor different architectures.
- No tested architecture closed external gaps.

### 4.2 External Scaling (Multiseed)
Source: `docs/evidence/cycle3_queue_final/scaling_multiseed_results_run_20260214T212435Z.json`

By hook layer:
- Layer 0: SAEBench mean `-0.077427`, CE-Bench mean `6.749414`
- Layer 1: SAEBench mean `-0.093996`, CE-Bench mean `9.139791`

By width:
- `d_sae=1024`: SAEBench mean `-0.082122`, CE-Bench mean `7.167310`
- `d_sae=2048`: SAEBench mean `-0.089301`, CE-Bench mean `8.721896`

Interpretation:
- CE-Bench improves in regions that worsen SAEBench deltas.

### 4.3 Stress-Gated Release
Source: `docs/evidence/cycle3_queue_final/release_policy_run_20260214T225029Z.json`

Gate status:
- `random_model = true`
- `transcoder = false`
- `ood = true`
- `external = false`
- `pass_all = false`

Key metrics:
- `transcoder_delta = -0.002227966984113039`
- `ood_drop = 0.01445406161520213`
- `external_delta = -0.017257680751151527`

Interpretation:
- Strict gating correctly blocks promotion.

## 5. Discussion

### 5.1 Supported Claims
- Internal consistency can be improved.
- External evaluation stack is reproducible and matched-baseline aware.
- Claim-gating infrastructure is operational and useful.

### 5.2 Unsupported Claims
- External superiority over matched/public baselines.
- SOTA-style performance claims for current methods.

### 5.3 Core Scientific Takeaway
Internal consistency optimization and external benchmark performance are misaligned in this regime; solving this likely requires explicit multi-objective optimization and selection.

## 6. Limitations
- Known-ground-truth circuit recovery from the original proposal is incomplete.
- Candidate selection for release gating should be formalized beyond single-summary input wiring.
- Architecture space is still limited relative to newest SAE families.

## 7. Reproducibility Checklist
- Canonical orientation: `START_HERE.md`
- Runbook: `RUNBOOK.md`
- Full provenance: `EXPERIMENT_LOG.md`
- Consistency audit: `results/analysis/experiment_consistency_report.md`
- Final queue evidence mirror: `docs/evidence/cycle3_queue_final/`

## 8. Next Steps
1. Multi-objective training/selection over internal + external metrics.
2. Add one new architecture family under matched protocol.
3. Close known-circuit recovery track.
4. CI guardrail for narrative-to-gate consistency.

## 9. Conclusion
HUSAI is now a robust reliability-first SAE research platform. It provides strong evidence for internal consistency gains and equally strong evidence that those gains do not yet transfer externally in tested settings. This clarifies the real frontier and prevents false confidence.
