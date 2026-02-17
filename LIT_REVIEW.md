# Literature and Competitive Landscape (Phase 3)

Date: 2026-02-16
Scope: SAE consistency, benchmark practice, architecture/objective frontier, and near-term novelty opportunities for this repository.

## 1) Primary Sources (Current and High-Relevance)

### Consistency, benchmarking, and claim discipline
1. Paulo & Belrose (2025), *SAEs Trained on the Same Data Learn Different Features* - https://arxiv.org/abs/2501.16615
2. Song et al. (2025), *Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs* - https://arxiv.org/abs/2505.20254
3. Karvonen et al. (ICML 2025), *SAEBench* - https://proceedings.mlr.press/v267/karvonen25a.html
4. SAEBench preprint - https://arxiv.org/abs/2503.09532
5. Peng et al. (2025), *CE-Bench* - https://arxiv.org/abs/2509.00691
6. SAEBench repository - https://github.com/adamkarvonen/SAEBench
7. CE-Bench repository - https://github.com/Yusen-Peng/CE-Bench

### Architecture/objective frontier
8. OpenAI (2024), *Scaling and Evaluating Sparse Autoencoders* - https://arxiv.org/abs/2406.04093
9. Rajamanoharan et al. (2024), *JumpReLU SAEs* - https://arxiv.org/abs/2407.14435
10. Bussmann et al. (2024), *BatchTopK SAEs* - https://arxiv.org/abs/2412.06410
11. Gao et al. (2025), *Nested Sparse Autoencoders / Matryoshka SAEs* - https://arxiv.org/abs/2503.17547
12. Li et al. (2025), *Route Sparse Autoencoders* - https://arxiv.org/abs/2503.08200
13. Wan et al. (2025), *HierarchicalTopK SAEs* - https://aclanthology.org/2025.emnlp-main.515/
14. Lin et al. (2026), *PolySAE* - https://arxiv.org/abs/2602.01322

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
1. Official SAEBench/CE-Bench harness paths are implemented and reproducible.
2. Custom HUSAI-checkpoint SAEBench/CE-Bench adapter paths are complete.
3. Multiseed external frontier and scaling runs are archived with manifests.
4. Stress-gated release policy is enforced and used for decision-making.
5. Cycle-5 added routed-family correction (`expert_topk`) and assignment-v3 selector integration.

### Remaining critical gaps
- External deltas are still negative in strict release settings.
- SAEBench remains the principal blocking metric.
- Known-circuit closure remains below trained-vs-random thresholds.

## 4) External Evidence Snapshot (Current)

Cycle-5 canonical evidence root:
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/`

Strict gate status:
- `pass_all = false`
- `random_model = true`
- `transcoder = true`
- `ood = true`
- `external = false`

Selected strict-gate metrics:
- `saebench_delta_ci95_low = -0.04478959689939781`
- `cebench_interp_delta_vs_baseline_ci95_low = -40.467037470119465`

Best cycle-5 routed CE delta:
- `-37.260996` (`run_20260215T234257Z`)

Best cycle-5 assignment CE delta:
- `-34.345572` (`run_20260216T005618Z`)

Interpretation:
- Infrastructure is benchmark-capable and reproducible.
- External performance remains the primary bottleneck.
- Nearest leverage is SAEBench-aware objective design with stronger grouped-LCB selection discipline.

## 5) Novel, Feasible Contribution Opportunities (Ranked)

1. SAEBench-aware assignment objective with grouped-LCB selection.
2. Seed-complete grouped external frontier (avoid dropped-group artifacts).
3. Routed `expert_topk` expansion under matched protocol.
4. Joint Pareto release policy with explicit SAEBench floor.
5. Known-circuit closure with confidence-bound thresholds.

## 6) Practical Replication Protocol to Keep

For each major update:
1. Log command, commit, config hash, and artifact paths in `EXPERIMENT_LOG.md`.
2. Run `python scripts/analysis/verify_experiment_consistency.py`.
3. For benchmark-facing claims, archive `commands.json`, `preflight.json`, `summary.md`, and logs.
4. Report gains and regressions with equal visibility.

## 7) Bottom Line

The repo has moved from speculative benchmark narrative to evidence-backed benchmark execution. The remaining scientific challenge is not tooling; it is finding configurations that improve external metrics without sacrificing reproducibility discipline.

## 8) 2026-02-17 Literature Refresh (Web-Verified)

Primary links checked:
1. SAE seed instability (Paulo & Belrose, 2025): https://arxiv.org/abs/2501.16615
2. SAEBench (Karvonen et al., 2025): https://arxiv.org/abs/2503.09532
3. CE-Bench (BlackboxNLP 2025): https://aclanthology.org/2025.blackboxnlp-1.1/
4. RouteSAE (Li et al., 2025): https://arxiv.org/abs/2503.08200
5. Transcoders vs SAEs (Makelov et al., 2025): https://arxiv.org/abs/2501.18823
6. Matryoshka/Nested SAEs (Gao et al., 2025): https://arxiv.org/abs/2503.17547
7. BatchTopK SAEs (Bussmann et al., 2024): https://arxiv.org/abs/2412.06410
8. JumpReLU SAEs (Rajamanoharan et al., 2024): https://arxiv.org/abs/2407.14435
9. RE-SA / relation-constrained SAE direction (2025): https://arxiv.org/abs/2506.09967

Interpretation update:
- The strongest current literature trend is that consistency claims require controls + benchmark coupling, not internal metrics alone.
- Route-style sparse architectures and relation-constrained objectives provide plausible mechanisms to improve external transfer without abandoning sparsity.
- Stress-gated release policies are now standard for credible external claims.

Important citation hygiene note:
- The commonly-circulated `arXiv:2507.18006` link is not a mechanistic-interpretability SAE benchmark anchor (it resolves to unrelated systems work). Treat it as non-authoritative for this project's core claims.

High-impact novel contributions not yet fully executed here:
1. CE-aware/SAEBench-aware multi-task assignment objective with explicit Pareto checkpointing (beyond current file-ID proxy).
2. RouteSAE vs Matryoshka vs TopK matched-budget frontier with identical seed sets and CI-lower-bound release criterion.
3. Cross-layer transfer test: train at layer-0, evaluate portability to layer-1 via fixed adapter vs retrain.
4. Relation-constrained regularization (RE-SA-inspired) ablation in assignment-v3 path.
5. External-aware early-stopping/checkpoint policy where validation objective is weighted external proxy + internal LCB.

## 9) Additional Source Addendum (2026-02-17)

Primary sources reviewed for next-step design:
1. Zhou et al. (ACL Findings 2025), *A Survey on Sparse Autoencoders for Interpretability* - https://aclanthology.org/2025.findings-acl.854/
2. Wu et al. (2025), *Kronecker Product Is All You Need for SAE Interpretability* - https://arxiv.org/abs/2505.22255

Interpretation for HUSAI:
- Survey evidence reinforces that reproducibility + external-benchmark coupling should be treated as first-class acceptance criteria, not optional reporting.
- Kron-style factorization is a concrete efficiency lever for larger-width sweeps under fixed compute budgets; this is a plausible cycle11 direction if cycle10 remains externally negative.
