# Literature and Competitive Landscape (Phase 3)

Date: 2026-02-12
Scope: SAE consistency, external benchmark practice, and high-leverage architecture/objective directions for this repository.

## 1) Primary Sources Most Relevant to This Repo

Consistency and evaluation framing:
1. Paulo & Belrose (2025): https://arxiv.org/abs/2501.16615
2. Song et al. (2025): https://arxiv.org/abs/2505.20254
3. SAEBench (ICML 2025): https://proceedings.mlr.press/v267/karvonen25a.html
4. SAEBench repo: https://github.com/adamkarvonen/SAEBench
5. CE-Bench (2025): https://arxiv.org/abs/2509.00691
6. CE-Bench repo: https://github.com/Yusen-Peng/CE-Bench

Strong SAE baseline methods:
7. OpenAI scaling/eval (2024): https://arxiv.org/abs/2406.04093
8. JumpReLU (2024): https://arxiv.org/abs/2407.14435
9. BatchTopK (2024): https://arxiv.org/abs/2412.06410
10. Matryoshka SAEs (2025): https://arxiv.org/abs/2503.17547

Recent frontier variants worth adding to this repo’s baseline set:
11. Route Sparse Autoencoders (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.346/
12. HierarchicalTopK SAEs (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.515/

## 2) What "Good" Looks Like in 2025-2026

Observed standard from benchmark-facing papers:
- multi-seed claims with uncertainty intervals
- random/naive controls to avoid proxy-metric illusions
- multi-metric reporting (consistency + quality + functional metrics)
- external benchmark evidence (SAEBench/CE-Bench), not internal slices alone
- reproducibility manifests (versions, commands, model IDs, seeds)

Inference:
- Internal PWMCC curves are useful, but insufficient for leaderboard-level or SOTA-level claims.

## 3) Gap Analysis vs This Repository

Where this repo is still behind:
1. Official external benchmark execution not yet completed (critical credibility gap).
2. Architecture frontier incomplete relative to 2025 methods (RouteSAE/HierarchicalTopK absent).
3. Objective-level consistency improvement is not yet statistically resolved.
4. Stress-test coverage (OOD/scaling/sensitivity) remains limited.

Where this repo can win:
1. Strong controlled setting for causal debugging of consistency claims.
2. Already-good artifact hygiene (manifested experiments + explicit controls).
3. Clear signal that adaptive L0 is a strong practical lever.
4. Opportunity for a reproducible bridge paper: controlled tasks -> external benchmark protocols.

## 4) Explicit "Behind" and "Can Win" Lists

Behind:
- No official SAEBench/CE-Bench run artifacts from this workspace yet.
- Missing modern architecture comparisons under matched budgets.
- Missing stress-test gate before updating claims.

Can win:
- Deliver benchmark-backed reproducibility with strict manifest discipline.
- Produce architecture-vs-objective regime maps (when does each method win).
- Set a high bar for negative-result clarity (what did not improve, with CIs).

## 5) Replication Details to Import Immediately

- For every headline metric:
  - include trained-vs-random control
  - include CI and effect size
  - include command + commit + config hash
- For every writeup update:
  - run `scripts/analysis/verify_experiment_consistency.py`
- For benchmark claims:
  - run official SAEBench/CE-Bench commands via `scripts/experiments/run_official_external_benchmarks.py`

## 6) Practical Conclusion

The repo is now strong enough to pursue high-impact external credibility, but only if the next cycle is benchmark-first and architecture-frontier aware. The immediate bottleneck is not another internal ablation; it is official external benchmark execution plus modern baseline coverage.
