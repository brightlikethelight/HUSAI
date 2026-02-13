# Literature and Competitive Landscape (Phase 3)

Date: 2026-02-13
Scope: SAE consistency, benchmark practice, architecture/objective frontier, and near-term novelty opportunities for this repository.

## 1) Primary Sources (Current and High-Relevance)

### Consistency, benchmarking, and claim discipline
1. Paulo & Belrose (2025), *SAEs Trained on the Same Data Learn Different Features* - https://arxiv.org/abs/2501.16615
2. Song et al. (2025), *Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs* - https://arxiv.org/abs/2505.20254
3. Karvonen et al. (ICML 2025), *SAEBench* - https://proceedings.mlr.press/v267/karvonen25a.html
4. SAEBench repository - https://github.com/adamkarvonen/SAEBench
5. Gulko et al. (BlackboxNLP 2025), *CE-Bench* - https://arxiv.org/abs/2509.00691
6. CE-Bench repository - https://github.com/Yusen-Peng/CE-Bench

### Architecture/objective frontier
7. OpenAI (2024), *Scaling and Evaluating Sparse Autoencoders* - https://arxiv.org/abs/2406.04093
8. Rajamanoharan et al. (2024), *JumpReLU SAEs* - https://arxiv.org/abs/2407.14435
9. Bussmann et al. (2024), *BatchTopK SAEs* - https://arxiv.org/abs/2412.06410
10. Gao et al. (2025), *Matryoshka SAEs* - https://arxiv.org/abs/2503.17547
11. Li et al. (EMNLP 2025), *Route Sparse Autoencoders* - https://aclanthology.org/2025.emnlp-main.346/
12. Wan et al. (EMNLP 2025), *HierarchicalTopK SAEs* - https://aclanthology.org/2025.emnlp-main.515/
13. Cui et al. (2026), *Polysemantic Sparse Autoencoders* - https://arxiv.org/abs/2602.01322

### Frontier controls and competing lenses
14. Makelov et al. (2025), *Transcoders Beat SAEs for Interpretability* - https://arxiv.org/abs/2501.18823
15. Arditi et al. (2025/2026), *Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers* - https://arxiv.org/abs/2501.17727
16. Yu et al. (2025), *MIB: A New Benchmark for Mechanistic Interpretability* - https://arxiv.org/abs/2504.13151

## 2) What Benchmark-Credible Practice Requires (Synthesis)

From SAEBench/CE-Bench plus recent controls papers:
- Multi-seed reporting with confidence intervals/effect sizes.
- Trained-vs-control comparisons (including strong simple baselines).
- Explicit manifests: command, commit, config hash, artifact paths.
- External benchmark execution on the actual method under study (not only nearby/public targets).
- Stress controls (random-model, OOD/sensitivity) before narrative claim updates.

Inference for this repo:
- Internal PWMCC gains are necessary but insufficient for external-quality claims.
- External benchmarks must include direct HUSAI-produced checkpoint evaluation on both SAEBench and CE-Bench-style protocols.

## 3) Gap Analysis vs This Repository (Updated)

### Where we were behind
1. Official benchmark execution was initially preflight-only.
2. Architecture frontier coverage lagged 2025 methods.
3. Objective-level consistency gains were weak/unresolved.
4. Stress testing was not a hard release gate.

### Where we are now
1. Official SAEBench command execution completed via harness (`run_20260212T201204Z`).
2. HUSAI custom-checkpoint SAEBench path completed and reproduced across 3 seeds:
   - `run_20260213T024329Z`, `run_20260213T031247Z`, `run_20260213T032116Z`
3. Official CE-Bench compatibility execution completed (`run_20260213T103218Z`) with tracked evidence in `docs/evidence/phase4e_cebench_official/`.
4. CE-Bench compatibility drift issues were resolved in runner code (`sae_lens` aliasing, multiprocessing shim, `stw.Stopwatch` API drift, run-local output capture).

### Remaining critical gaps
- CE-Bench currently runs on public SAE targets; direct HUSAI checkpoint CE-Bench support remains missing.
- HUSAI external SAEBench AUC is consistently below baseline probes in current setup.
- No matched-budget architecture frontier run has yet been completed on the external stack.

## 4) External Evidence Snapshot (Current)

Public-SAEBench target run:
- `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`
- Mean delta (best SAE over k {1,2,5} minus baseline logreg):
  - `test_f1`: `-0.0952`
  - `test_acc`: `-0.0513`
  - `test_auc`: `-0.0651`

HUSAI custom multi-seed SAEBench summary:
- `docs/evidence/phase4e_husai_custom_multiseed/summary.json`
- best AUC mean ± std: `0.622601 ± 0.000615`
- best AUC 95% CI: `[0.621905, 0.623297]`
- delta AUC vs baseline mean ± std: `-0.051801 ± 0.000615`
- delta AUC vs baseline 95% CI: `[-0.052496, -0.051105]`

Official CE-Bench compatibility run (public SAE target):
- `results/experiments/phase4e_external_benchmark_official/run_20260213T103218Z/`
- tracked metrics summary: `docs/evidence/phase4e_cebench_official/run_20260213T103218Z_cebench_metrics_summary.json`
- `total_rows`: `5000`
- `contrastive_score_mean.max`: `49.1142`
- `independent_score_mean.max`: `53.6982`
- `interpretability_score_mean.max`: `47.4812`

Interpretation:
- Infrastructure is now benchmark-capable and reproducible.
- Method quality on external SAEBench probing remains the bottleneck.
- CE-Bench execution is unblocked, but direct HUSAI CE-Bench evaluation is the next required milestone.

## 5) Novel, Feasible Contribution Opportunities (Ranked)

1. Direct HUSAI CE-Bench + SAEBench joint reporting
- Novelty: dual-benchmark evidence for the same checkpoint family under one manifest schema.
- Risk: CE-Bench custom-checkpoint integration complexity.

2. Architecture selection conditioned on activation geometry
- Novelty: predict best SAE family (`TopK/JumpReLU/BatchTopK/Matryoshka/Route/Hierarchical`) from effective-rank/anisotropy descriptors.
- Risk: overfitting to narrow task families.

3. Assignment-aware multi-seed consistency objective v2
- Novelty: optimize cross-seed alignment directly (matching-aware loss) while preserving EV.
- Risk: extra compute and optimization instability.

4. Polysemantic-aware scaling ablation (PolySAE-inspired)
- Novelty: test whether polysemantic structure improves stability-external tradeoff in this domain.
- Risk: implementation complexity and interpretation ambiguity.

5. Stress-gated claim pipeline (random-model + OOD + transcoder)
- Novelty: explicit anti-overclaim gate where narrative updates require passing stress controls.
- Risk: slower iteration cadence.

## 6) Practical Replication Protocol to Keep

For each major update:
1. Log command, commit, config hash, and artifact paths in `EXPERIMENT_LOG.md`.
2. Run `python scripts/analysis/verify_experiment_consistency.py`.
3. For benchmark-facing claims, archive harness `commands.json`, `preflight.json`, `summary.md`, and logs.
4. Report gains and regressions with equal visibility.

## 7) Bottom Line

The repo is now substantially stronger on engineering rigor and benchmark execution quality. The highest-leverage path is method improvement under benchmark pressure, starting with direct HUSAI CE-Bench integration and architecture/objective variants that can move external metrics while preserving reproducibility discipline.
