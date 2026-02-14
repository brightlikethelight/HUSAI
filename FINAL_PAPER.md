# Reliability-First SAE Evaluation in HUSAI: Internal Consistency Gains Under External Benchmark Constraints

## Abstract
We executed a reliability-first, end-to-end experiment program for sparse autoencoders (SAEs) in HUSAI. The cycle targeted five high-impact follow-ups: direct HUSAI-checkpoint CE-Bench integration with matched baseline, matched-budget architecture frontier, external scaling (`token budget`, `hook layer`, `d_sae`), assignment-aware consistency objective v2 with acceptance criteria, and stress-gated release policy. Internal consistency improved under assignment-aware training (best delta PWMCC `+0.070804`, conservative lower bound `+0.054419`) with negligible explained-variance drop (`0.000878`). However, external acceptance failed across tested conditions: SAEBench best-minus-LLM AUC deltas remained negative, and CE-Bench custom checkpoints were far below matched public baseline (`interpretability` deltas approximately `-37` to `-44`). Scaling revealed systematic tradeoffs: layer-1 and larger width improved CE-Bench interpretability while worsening SAEBench deltas. Stress-gated release policy correctly blocked claim upgrades (`pass_all=False`) due negative external delta and missing transcoder/OOD controls. These results support stronger methodological rigor and reproducibility, but do not support external SOTA claims.

## 1. Introduction
SAE research is vulnerable to two failure modes: unstable internal features and overclaiming without external validation. This work prioritizes reliability and reproducibility first, then evaluates whether consistency improvements transfer to external benchmark gains.

Our primary research question is:
- Can internal consistency gains be achieved without external metric regressions?

## 2. Related Work
- SAEBench benchmark design and evaluation protocol: Karvonen et al. (ICML 2025), https://proceedings.mlr.press/v267/karvonen25a.html
- CE-Bench contrastive interpretability benchmark: Peng et al. (2025), https://arxiv.org/abs/2509.00691
- Architecture references: JumpReLU (https://arxiv.org/abs/2407.14435), BatchTopK (https://arxiv.org/abs/2412.06410), Matryoshka SAEs (https://arxiv.org/abs/2503.17547), RouteSAE (https://aclanthology.org/2025.emnlp-main.346/), HierarchicalTopK (https://aclanthology.org/2025.emnlp-main.515/)
- Stress-control framing: Transcoders vs SAEs (https://arxiv.org/abs/2501.18823), random-transformer controls (https://arxiv.org/abs/2501.17727)

## 3. Method and Engineering Protocol

### 3.1 Reliability prerequisites
Before new claims, we enforced:
- deterministic benchmark harness artifacts,
- explicit dataset resolution for SAEBench custom evals,
- path normalization for remote portability,
- CI-compatible fail-fast gating flags.

### 3.2 High-impact execution tracks
1. direct HUSAI CE-Bench + matched baseline,
2. architecture frontier on external stack,
3. external scaling sweep,
4. assignment-aware consistency objective v2,
5. stress-gated release policy.

### 3.3 Core metrics
- internal: PWMCC trained-vs-random deltas, conservative CI lower bounds, EV drop,
- external: SAEBench best-minus-LLM AUC delta, CE-Bench interpretability/contrastive/independent maxima,
- release gating: random-model, transcoder, OOD, external pass/fail.

## 4. Experiments

### 4.1 Direct HUSAI CE-Bench with matched baseline
Matched baseline (`max_rows=200`) artifact:
- `docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json`

Baseline metrics:
- contrastive max: `50.5113`
- independent max: `50.9993`
- interpretability max: `47.9516`

### 4.2 Matched-budget architecture frontier
Artifact:
- `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_results.json`

Architectures:
- TopK, ReLU, BatchTopK, JumpReLU

### 4.3 External scaling study
Artifact:
- `docs/evidence/phase4e_external_scaling_study/run_20260213T203923Z_results.json`

Grid:
- token budgets: `10000, 30000`
- hook layers: `0, 1`
- widths: `1024, 2048`
- total conditions: `8`

### 4.4 Assignment-aware consistency objective v2
Artifact:
- `docs/evidence/phase4d_assignment_consistency_v2/run_20260213T203957Z_results.json`

### 4.5 Stress-gated release policy
Artifact:
- `docs/evidence/phase4e_stress_gated_release/run_20260213T204120Z_release_policy.json`

## 5. Results

### 5.1 Architecture frontier external behavior
From `run_20260213T173707Z_summary_table.md`:
- TopK: SAEBench delta `-0.1328`, CE-Bench interpretability `7.5854`
- ReLU: SAEBench delta `-0.0346`, CE-Bench interpretability `4.2281`
- BatchTopK: SAEBench delta `-0.0630`, CE-Bench interpretability `6.4668`
- JumpReLU: SAEBench delta `-0.0488`, CE-Bench interpretability `4.3510`

Observation:
- SAEBench and CE-Bench rankings disagree.

### 5.2 Frontier vs matched CE-Bench baseline
From `run_20260213T173707Z_cebench_deltas_vs_matched200.md`:
- TopK interpretability delta: `-40.3662`
- ReLU interpretability delta: `-43.7235`
- BatchTopK interpretability delta: `-41.4848`
- JumpReLU interpretability delta: `-43.6006`

Observation:
- All tested custom checkpoints remain well below matched baseline.

### 5.3 Scaling study aggregate trends
From `run_20260213T203923Z_results.json`:
- by token budget:
  - `10000`: SAEBench delta mean `-0.07854`, CE-Bench interpretability mean `8.0104`
  - `30000`: SAEBench delta mean `-0.08497`, CE-Bench interpretability mean `8.1203`
- by hook layer:
  - layer `0`: SAEBench delta mean `-0.06873`, CE-Bench interpretability mean `6.8769`
  - layer `1`: SAEBench delta mean `-0.09479`, CE-Bench interpretability mean `9.2538`
- by `d_sae`:
  - `1024`: SAEBench delta mean `-0.07996`, CE-Bench interpretability mean `7.1746`
  - `2048`: SAEBench delta mean `-0.08355`, CE-Bench interpretability mean `8.9561`

Observation:
- Layer/width changes can improve CE-Bench while worsening SAEBench delta.

### 5.4 Assignment-aware v2 outcome
Best setting (`lambda=0.2`):
- delta PWMCC: `+0.070804`
- conservative delta LCB: `+0.054419`
- EV drop: `0.000878`
- external delta input: `-0.132836`
- acceptance `pass_all=False` (external gate failure)

Observation:
- internal consistency can be improved robustly,
- external constraints still block promotion.

### 5.5 Stress-gated policy outcome
Gate results:
- random-model: pass
- transcoder: fail (missing evidence)
- OOD: fail (missing evidence)
- external: fail (negative delta)
- overall: fail

Strict fail-fast is now supported by:
- `run_stress_gated_release_policy.py --fail-on-gate-fail`

## 6. Discussion

### 6.1 What is supported
- The experiment stack is now reproducible and evidence-first.
- Internal consistency gains are achievable with assignment-aware regularization.
- External benchmark tradeoffs are now explicit and quantified.

### 6.2 What is not supported
- External superiority claims.
- SOTA claims in current tested regimes.

### 6.3 Main scientific takeaway
Internal consistency optimization and external interpretability performance are not aligned by default in this setting. Reliable progress requires multi-objective selection and stricter stress controls.

## 7. Limitations
- External frontier/scaling is currently single-seed per condition.
- Transcoder and OOD stress artifacts are missing (gates fail by design).
- Architecture frontier currently covers 4 families; broader variants remain untested.

## 8. Reproducibility Checklist
- CI/tests: `pytest -q` (`83 passed`)
- high-impact report: `HIGH_IMPACT_FOLLOWUPS_REPORT.md`
- experiment ledger: `EXPERIMENT_LOG.md`
- external evidence roots:
  - `docs/evidence/phase4b_architecture_frontier_external/`
  - `docs/evidence/phase4e_cebench_matched200/`
  - `docs/evidence/phase4e_external_scaling_study/`
  - `docs/evidence/phase4d_assignment_consistency_v2/`
  - `docs/evidence/phase4e_stress_gated_release/`

## 9. Next Steps (Ranked)
1. Add matched transcoder and OOD stress tracks; enforce strict gates in CI.
2. Expand architecture frontier to Matryoshka/RouteSAE/HierarchicalTopK.
3. Run multi-seed external CIs for selected frontier/scaling candidates.
4. Implement Pareto checkpoint selection over internal+external objectives.
5. Evaluate layer-aware architecture routing under fixed budget.

## 10. Broader Impact
The main contribution of this cycle is methodological: converting speculative benchmark narratives into strict, reproducible evidence with explicit gate failures. This reduces overclaiming risk and produces a clearer roadmap for genuinely competitive interpretability research.

## 11. 2026-02-14 Postscript: Claim-Gating Outcome

A post-cycle consistency audit that includes assignment-v2 and release-stress gates reports `overall_pass=False` (`results/analysis/experiment_consistency_report.md`). This reflects negative external deltas and missing/failed stress-gate evidence in the latest closure artifacts, and should be treated as the canonical claim status until those gates are satisfied.
