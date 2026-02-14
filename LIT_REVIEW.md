# Literature and Competitive Landscape (Phase 3)

Date: 2026-02-14
Scope: SAE consistency, benchmark practice, architecture/objective frontier, and near-term novelty opportunities for this repository.

## 1) Primary Sources (Current and High-Relevance)

### Consistency, benchmarking, and claim discipline
1. Paulo & Belrose (2025), *SAEs Trained on the Same Data Learn Different Features* - https://arxiv.org/abs/2501.16615
2. Song et al. (2025), *Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs* - https://arxiv.org/abs/2505.20254
3. Karvonen et al. (ICML 2025), *SAEBench* - https://proceedings.mlr.press/v267/karvonen25a.html
4. SAEBench preprint - https://arxiv.org/abs/2503.09532
5. Peng et al. (BlackboxNLP 2025), *CE-Bench* - https://arxiv.org/abs/2509.00691
6. SAEBench repository - https://github.com/adamkarvonen/SAEBench
7. CE-Bench repository - https://github.com/Yusen-Peng/CE-Bench

### Architecture/objective frontier
8. OpenAI (2024), *Scaling and Evaluating Sparse Autoencoders* - https://arxiv.org/abs/2406.04093
9. Rajamanoharan et al. (2024), *JumpReLU SAEs* - https://arxiv.org/abs/2407.14435
10. Bussmann et al. (2024), *BatchTopK SAEs* - https://arxiv.org/abs/2412.06410
11. Gao et al. (2025), *Nested Sparse Autoencoders / Matryoshka SAEs* - https://arxiv.org/abs/2503.17547
12. Li et al. (EMNLP 2025), *Route Sparse Autoencoders* - https://aclanthology.org/2025.emnlp-main.346/
13. Wan et al. (EMNLP 2025), *HierarchicalTopK SAEs* - https://aclanthology.org/2025.emnlp-main.515/
14. Cao et al. (2026), *PolySAE* - https://arxiv.org/abs/2602.01322

### Frontier controls and competing lenses
15. Makelov et al. (2025), *Transcoders Beat SAEs for Interpretability* - https://arxiv.org/abs/2501.18823
16. Arditi et al. (2025), *Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers* - https://arxiv.org/abs/2501.17727
17. Yu et al. (2025), *MIB: A New Benchmark for Mechanistic Interpretability* - https://arxiv.org/abs/2504.13151

## 2) What Benchmark-Credible Practice Requires (Synthesis)

Across SAEBench/CE-Bench and recent control papers, credible claims require:
- Multi-seed reporting with confidence intervals and effect sizes.
- Trained-vs-control comparisons (including random/model controls).
- Full manifests (command, commit, config hash, artifacts).
- External benchmark execution on the exact checkpoint family under study.
- Stress controls (random-model, OOD, transcoder/alternative-method) before narrative upgrades.

Inference for this repo:
- Internal PWMCC gains are necessary but insufficient for external-quality claims.
- Promotion decisions should be multi-objective (internal consistency + external benchmark deltas).

## 3) Gap Analysis vs This Repository (Updated)

### Where we were behind
1. Official benchmark execution was initially preflight-only.
2. Architecture frontier coverage lagged 2025 methods.
3. Objective-level consistency gains were weak/unresolved.
4. Stress testing was not a hard release gate.

### Where we are now
1. Official SAEBench command execution completed via harness (`run_20260212T201204Z`).
2. HUSAI custom-checkpoint SAEBench path completed and reproduced across 3 seeds (`run_20260213T024329Z`, `run_20260213T031247Z`, `run_20260213T032116Z`).
3. Official CE-Bench compatibility execution completed (`run_20260213T103218Z`) with tracked evidence in `docs/evidence/phase4e_cebench_official/`.
4. Direct HUSAI custom-checkpoint CE-Bench path is implemented and exercised (frontier/scaling runs under `docs/evidence/phase4b_architecture_frontier_external/` and `docs/evidence/phase4e_external_scaling_study/`).
5. Stress-gated release policy is implemented with fail-fast mode (`--fail-on-gate-fail`).

### Remaining critical gaps
- External deltas are still negative in tested regimes (SAEBench best-minus-LLM AUC and CE-Bench matched-baseline deltas).
- Frontier/scaling uncertainty is underpowered (mostly single-seed external runs).
- Transcoder and OOD stress runners are now implemented, but fresh gate artifacts are still required before any release upgrade.

## 4) External Evidence Snapshot (Current)

Public SAEBench target run:
- `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`
- mean delta (best SAE over k {1,2,5} minus baseline logreg):
  - `test_f1`: `-0.0952`
  - `test_acc`: `-0.0513`
  - `test_auc`: `-0.0651`

HUSAI custom multi-seed SAEBench summary:
- `docs/evidence/phase4e_husai_custom_multiseed/summary.json`
- best AUC mean ± std: `0.622601 ± 0.000615`
- delta AUC vs baseline mean ± std: `-0.051801 ± 0.000615`

CE-Bench matched-200 baseline:
- `docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json`
- interpretability max: `47.9516`

HUSAI architecture frontier CE-Bench deltas vs matched baseline:
- `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_cebench_deltas_vs_matched200.md`
- interpretability delta range: approximately `-43.7` to `-40.4`

Interpretation:
- infrastructure is benchmark-capable and reproducible,
- external performance remains the primary bottleneck,
- nearest leverage is method improvement with matched external constraints and stronger uncertainty quantification.

## 5) Novel, Feasible Contribution Opportunities (Ranked)

1. Multi-seed external confidence frontier
- run >=3 seeds for best frontier/scaling candidates and report CIs on SAEBench and CE-Bench deltas.

2. Pareto checkpoint selection for release
- promote checkpoints only if they satisfy internal consistency + external gates jointly.

3. Architecture expansion under matched protocol
- add Matryoshka/RouteSAE/HierarchicalTopK with fixed token and compute budgets.

4. Assignment-aware objective + external proxy coupling
- extend assignment-v2 with weak external proxy regularization; track EV guardrails.

5. Stress-gated claim automation
- CI gate that blocks doc claim changes when release gate artifacts fail.

## 6) Practical Replication Protocol to Keep

For each major update:
1. Log command, commit, config hash, and artifact paths in `EXPERIMENT_LOG.md`.
2. Run `python scripts/analysis/verify_experiment_consistency.py`.
3. For benchmark-facing claims, archive `commands.json`, `preflight.json`, `summary.md`, and logs.
4. Report gains and regressions with equal visibility.

## 7) Bottom Line

The repo has crossed from speculative benchmark narrative to evidence-backed benchmark execution. The remaining scientific challenge is not plumbing; it is finding configurations that improve external metrics without losing reproducibility discipline.
