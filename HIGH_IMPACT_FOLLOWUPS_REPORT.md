# High-Impact Follow-Ups: Cycle 3 Final Closure Report

Date: 2026-02-15

This report closes the ranked follow-ups and records what is complete, what failed, and what remains.

## Requested Follow-Ups and Final Status

| Follow-up | Status | Primary evidence |
|---|---|---|
| 1) Direct HUSAI-checkpoint CE-Bench adapter/eval with matched baseline | complete | `docs/evidence/high_impact_adapter_check/run_20260214T202232Z_husai_custom_cebench_summary.json` |
| 2) Matched-budget architecture frontier sweep on external benchmarks | complete (multiseed) | `docs/evidence/cycle3_queue_final/frontier_multiseed_results_run_20260214T202538Z.json` |
| 3) External-metric scaling study (`token budget`, `hook layer`, `d_sae`) | complete (multiseed) | `docs/evidence/cycle3_queue_final/scaling_multiseed_results_run_20260214T212435Z.json` |
| 4) Assignment-aware consistency objective v2 with external acceptance criteria | complete | `results/experiments/phase4d_assignment_consistency_v2/run_20260213T203957Z/results.json` |
| 5) Stress-gated release policy (`random-model`, `transcoder`, `OOD`, `external`) | complete and enforced | `docs/evidence/cycle3_queue_final/release_policy_run_20260214T225029Z.json` |

## Final Outcomes

### 1) Direct HUSAI CE-Bench Adapter Path

- Adapter/eval path works end-to-end under matched settings.
- Matched baseline reference is operational: `docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json`.
- Result: custom checkpoints remain far below matched baseline in current tested regimes.

### 2) Architecture Frontier (Multiseed)

Source: `docs/evidence/cycle3_queue_final/frontier_multiseed_results_run_20260214T202538Z.json`

- `4 architectures x 5 seeds`, 20 complete records.
- SAEBench best-minus-LLM mean deltas:
  - `relu = -0.024691`
  - `jumprelu = -0.030577`
  - `topk = -0.040593`
  - `batchtopk = -0.043356`
- CE-Bench interpretability means:
  - `topk = 7.726768`
  - `batchtopk = 6.537639`
  - `jumprelu = 4.379002`
  - `relu = 4.257686`

Conclusion:
- SAEBench and CE-Bench prefer different architectures.
- No tested architecture closes external gaps.

### 3) External Scaling Study (Multiseed)

Source: `docs/evidence/cycle3_queue_final/scaling_multiseed_results_run_20260214T212435Z.json`

- 24 completed conditions.
- By token budget:
  - `10000`: SAEBench mean `-0.086291`, CE-Bench mean `7.862273`
  - `30000`: SAEBench mean `-0.085132`, CE-Bench mean `8.026933`
- By hook layer:
  - `0`: SAEBench mean `-0.077427`, CE-Bench mean `6.749414`
  - `1`: SAEBench mean `-0.093996`, CE-Bench mean `9.139791`
- By `d_sae`:
  - `1024`: SAEBench mean `-0.082122`, CE-Bench mean `7.167310`
  - `2048`: SAEBench mean `-0.089301`, CE-Bench mean `8.721896`

Conclusion:
- Larger width and layer 1 help CE-Bench but worsen SAEBench deltas.

### 4) Assignment-Aware Objective v2

Source: `results/experiments/phase4d_assignment_consistency_v2/run_20260213T203957Z/results.json`

Best tested condition:
- `lambda = 0.2`
- internal delta PWMCC `+0.070804`
- conservative LCB `+0.054419`
- EV drop `0.000878`

Conclusion:
- Internal consistency improved strongly.
- External acceptance remained unsatisfied.

### 5) Stress-Gated Release Policy

Source: `docs/evidence/cycle3_queue_final/release_policy_run_20260214T225029Z.json`

Gate results:
- `random_model = true`
- `transcoder = false`
- `ood = true`
- `external = false`
- `pass_all = false`

Key metric inputs:
- `transcoder_delta = -0.002227966984113039`
- `ood_drop = 0.01445406161520213`
- `external_delta = -0.017257680751151527`

Conclusion:
- Policy is working correctly and blocks unsupported release claims.

## What Was Fixed/Upgraded to Enable This Cycle

- CE-Bench adapter path and matched-baseline comparison flow.
- Multiseed external frontier/scaling orchestration.
- Stress evaluation runners for transcoder and OOD.
- Strict release gate enforcement with fail-fast behavior.
- Artifact-backed consistency audit and evidence syncing.

## Final Interpretation

1. The ranked follow-ups are executed and closed from an engineering standpoint.
2. Scientific outcome is mixed: internal gains are real; external gains are not yet competitive.
3. The repository now supports honest, reproducible claim-gating at release time.

## New Highest-Leverage Next 5 (Post-Closure)

1. Add explicit multi-objective optimization/selection (internal consistency + external metrics).
2. Add one newer SAE family (Matryoshka/RouteSAE/HierarchicalTopK) in the same matched protocol.
3. Close known-ground-truth circuit recovery from the original proposal.
4. Tighten deterministic CUDA settings in production run scripts.
5. Add CI check that blocks summary docs when gate status and narrative diverge.
