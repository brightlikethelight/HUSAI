# High-Impact Follow-Ups: Cycle 2 Completion Report

Date: 2026-02-13

## Scope

This report closes the five highest-leverage follow-ups requested for this cycle:
1. direct HUSAI-checkpoint CE-Bench adapter/eval with matched baseline,
2. matched-budget architecture frontier on external metrics,
3. external-metric scaling study (`token budget`, `hook layer`, `d_sae`),
4. assignment-aware consistency objective v2 with external acceptance,
5. stress-gated release policy (`random-model`, `transcoder`, `OOD`, `external`).

All runs are artifact-backed and reproducible.

## Completion Status

| Follow-up | Status | Primary artifact |
|---|---|---|
| Direct HUSAI CE-Bench + matched baseline | complete | `docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json` |
| Architecture frontier (external) | complete | `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_results.json` |
| External scaling study | complete | `docs/evidence/phase4e_external_scaling_study/run_20260213T203923Z_results.json` |
| Assignment-aware consistency v2 | complete | `docs/evidence/phase4d_assignment_consistency_v2/run_20260213T203957Z_results.json` |
| Stress-gated release policy | complete | `docs/evidence/phase4e_stress_gated_release/run_20260213T204120Z_release_policy.json` |

## Key Results

### 1) Direct HUSAI CE-Bench with matched baseline

Matched baseline (public SAE, `max_rows=200`):
- artifact: `docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json`
- interpretability max: `47.9516`

HUSAI architecture frontier deltas vs matched baseline:
- artifact: `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_cebench_deltas_vs_matched200.md`
- topk delta interpretability: `-40.3662`
- relu delta interpretability: `-43.7235`
- batchtopk delta interpretability: `-41.4848`
- jumprelu delta interpretability: `-43.6006`

Conclusion:
- direct HUSAI CE-Bench path is operational and reproducible,
- all tested custom checkpoints are substantially below matched public baseline.

### 2) Matched-budget architecture frontier (TopK/ReLU/BatchTopK/JumpReLU)

Artifact:
- `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_summary_table.md`

Per-architecture external snapshot:
- TopK: SAEBench delta `-0.1328`, CE-Bench interpretability `7.5854`
- ReLU: SAEBench delta `-0.0346`, CE-Bench interpretability `4.2281`
- BatchTopK: SAEBench delta `-0.0630`, CE-Bench interpretability `6.4668`
- JumpReLU: SAEBench delta `-0.0488`, CE-Bench interpretability `4.3510`

Conclusion:
- benchmark preference is split: ReLU/JumpReLU better on SAEBench delta, TopK/BatchTopK better on CE-Bench,
- no tested architecture closes the external baseline gap.

### 3) External scaling study (`token/layer/d_sae`)

Artifact:
- `docs/evidence/phase4e_external_scaling_study/run_20260213T203923Z_results.json`
- extracted table: `docs/evidence/phase4e_external_scaling_study/run_20260213T203923Z_summary_table.md`

Grid:
- token budgets: `10000, 30000`
- hook layers: `0, 1`
- d_sae: `1024, 2048`
- seeds: `42`
- total conditions: `8`

Axis aggregates:
- by token budget:
  - `10000`: SAEBench delta mean `-0.07854`, CE-Bench interpretability mean `8.0104`
  - `30000`: SAEBench delta mean `-0.08497`, CE-Bench interpretability mean `8.1203`
- by hook layer:
  - layer `0`: SAEBench delta mean `-0.06873`, CE-Bench interpretability mean `6.8769`
  - layer `1`: SAEBench delta mean `-0.09479`, CE-Bench interpretability mean `9.2538`
- by d_sae:
  - `1024`: SAEBench delta mean `-0.07996`, CE-Bench interpretability mean `7.1746`
  - `2048`: SAEBench delta mean `-0.08355`, CE-Bench interpretability mean `8.9561`

Conclusion:
- external metric tradeoff is systematic (layer 1 and larger `d_sae` help CE-Bench but hurt SAEBench delta),
- additional tokens did not recover SAEBench delta.

### 4) Assignment-aware consistency objective v2

Artifact:
- `docs/evidence/phase4d_assignment_consistency_v2/run_20260213T203957Z_results.json`

Best condition:
- lambda: `0.2`
- delta PWMCC: `+0.07080`
- conservative delta LCB: `+0.05442`
- EV drop vs baseline: `0.000878` (well under `0.05` budget)

Acceptance gates:
- internal consistency gates: pass,
- external gate (`min_external_delta >= 0`): fail (`-0.1328`),
- overall: `pass_all = False`.

Conclusion:
- assignment-aware regularization improves internal consistency strongly,
- external gate blocks release claim upgrades.

### 5) Stress-gated release policy

Artifact:
- `docs/evidence/phase4e_stress_gated_release/run_20260213T204120Z_release_policy.json`

Strict gate result:
- random-model gate: pass,
- transcoder gate: fail (missing result),
- OOD gate: fail (missing result),
- external gate: fail (negative external delta),
- overall: `pass_all = False`.

Additional enforcement upgrade:
- `scripts/experiments/run_stress_gated_release_policy.py` now supports `--fail-on-gate-fail` for CI fail-fast.

## Major Engineering Fixes Landed

Code fixes that unblocked valid experiments:
- `run_architecture_frontier_external.py`
  - BatchTopK threshold train/inference handling,
  - custom SAE non-degenerate init,
  - explicit SAEBench dataset passing,
  - path normalization and dataset-limit controls.
- `run_cebench_compat.py`
  - `--max-rows` support for matched-budget CE-Bench runs.
- `run_external_metric_scaling_study.py`
  - fail-fast dataset resolution for SAEBench,
  - explicit dataset forwarding into custom SAEBench eval,
  - CE-Bench `--max-rows` forwarding.
- `run_assignment_consistency_v2.py`
  - new `--fail-on-acceptance-fail` flag.
- `run_stress_gated_release_policy.py`
  - new `--fail-on-gate-fail` flag.

Validation:
- `python -m py_compile ...` passes on modified scripts.
- `pytest -q` remains green (`83 passed`).

## Updated Highest-Leverage Next 5 (Ranked)

1. Add matched transcoder + OOD stress benchmarks and wire them into strict release gates.
- Why: current release policy fails primarily because these gates are missing.

2. Expand architecture frontier to Matryoshka/RouteSAE/HierarchicalTopK with matched external protocol.
- Why: current 4-architecture frontier exposes tradeoffs but no win-region.

3. Run multi-seed external confidence intervals for frontier/scaling best candidates.
- Why: current frontier/scaling is mostly seed-1; uncertainty on external metrics is undercharacterized.

4. Add dual-objective model selection (internal consistency + external score Pareto) for checkpoint promotion.
- Why: objective v2 improves internal signal but fails external gate; we need explicit multi-objective selection.

5. Add layer-aware architecture routing (e.g., layer0 TopK-family, layer1 ReLU-family) under fixed parameter budget.
- Why: scaling suggests layer-specific metric preferences; a single architecture may be suboptimal.

## Primary Sources

- SAEBench (ICML 2025): https://proceedings.mlr.press/v267/karvonen25a.html
- SAEBench repo: https://github.com/adamkarvonen/SAEBench
- CE-Bench (2025): https://arxiv.org/abs/2509.00691
- CE-Bench repo: https://github.com/Yusen-Peng/CE-Bench
- JumpReLU: https://arxiv.org/abs/2407.14435
- BatchTopK: https://arxiv.org/abs/2412.06410
- Matryoshka SAEs: https://arxiv.org/abs/2503.17547
- RouteSAE: https://aclanthology.org/2025.emnlp-main.346/
- HierarchicalTopK SAEs: https://aclanthology.org/2025.emnlp-main.515/
- Transcoders Beat SAEs: https://arxiv.org/abs/2501.18823
- Automated metrics vs random-transformer controls: https://arxiv.org/abs/2501.17727

## 2026-02-14 Update (Cycle 3 In Progress)

### Newly executed/high-impact updates
- Direct adapter parity re-check (matched CE-Bench baseline) completed:
  - `docs/evidence/high_impact_adapter_check/run_20260214T202232Z_husai_custom_cebench_summary.json`
- Matched-budget architecture frontier **multiseed** run launched on B200 and in progress:
  - `results/experiments/phase4b_architecture_frontier_external_multiseed/run_20260214T202538Z/`
  - Current progress snapshot: 13 checkpoints, 13 SAEBench summaries, 12 CE-Bench summaries.

### Integrity correction
- Consistency audit has been upgraded to include assignment-v2 and stress-gate artifacts so report status cannot be falsely green when external gates fail.
  - Script: `scripts/analysis/verify_experiment_consistency.py`
  - Latest report: `results/analysis/experiment_consistency_report.md` (`overall_pass=False`)

### Updated highest-leverage next 5 (ranked)
1. Add direct HUSAI-checkpoint CE-Bench adapter/eval path with matched baselines. (completed)
2. Run matched-budget architecture frontier sweep on external benchmarks. (multiseed run in progress)
3. Run external-metric scaling study (token budget, hook layer, d_sae). (single-seed complete; multiseed extension pending)
4. Implement assignment-aware consistency objective v2 with external acceptance criteria. (completed)
5. Add stress-gated release policy (transcoder, random-model, OOD). (implemented; strict gate currently failing pending fresh transcoder/OOD + external-positive candidate)
