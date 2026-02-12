# Novel Contributions and High-Leverage Follow-Ups

Updated: 2026-02-12

## Evidence Base (Primary Sources)

1. SAE instability is real across seeds and architectures.
- Paulo & Belrose (arXiv:2501.16615): only ~30% shared features in a large SAE setting; TopK is more seed-sensitive than ReLU.
- Source: https://arxiv.org/abs/2501.16615

2. Proxy metrics alone are insufficient for progress claims.
- SAEBench (ICML 2025 / PMLR 267): benchmark shows proxy gains do not reliably map to practical utility; compares many architectures and tasks.
- Source: https://proceedings.mlr.press/v267/karvonen25a.html
- SAEBench codebase/evals: https://github.com/adamkarvonen/SAEBench

3. Consistency itself is a first-class objective.
- Song et al. (arXiv:2505.20254): argues MI should prioritize feature consistency; reports high consistency is achievable under the right setup.
- Source: https://arxiv.org/abs/2505.20254

4. "Canonical units" are not guaranteed.
- Leask et al. (arXiv:2502.04878): SAE stitching/meta-SAEs show incompleteness/non-atomicity of SAE features.
- Source: https://arxiv.org/abs/2502.04878

5. Modern SAE architectures improve quality/controllability tradeoffs.
- OpenAI scaling paper (arXiv:2406.04093): scaling laws and quality metrics for SAEs.
- JumpReLU (arXiv:2407.14435): stronger reconstruction at fixed sparsity.
- BatchTopK (arXiv:2412.06410): adaptive per-sample activation count at fixed average L0.
- Matryoshka SAEs (arXiv:2503.17547): hierarchical features, better disentanglement/task behavior.
- Sources:
  - https://arxiv.org/abs/2406.04093
  - https://arxiv.org/abs/2407.14435
  - https://arxiv.org/abs/2412.06410
  - https://arxiv.org/abs/2503.17547

6. Benchmarking is converging to reproducible, lower-cost evaluation.
- CE-Bench (arXiv:2509.00691): contrastive, no external LLM judge; reports strong alignment with SAEBench.
- Source: https://arxiv.org/abs/2509.00691

7. Tooling maturity supports reproducible evaluation pipelines.
- SAELens v6+ and active releases.
- Sources:
  - https://github.com/decoderesearch/SAELens
  - https://pypi.org/project/sae-bench/

## Gap Analysis vs This Repo

Current repo strength:
- Clean local pipeline for transformer -> activations -> SAE.
- Multi-seed trained-vs-random comparison implemented with manifests.
- Core ablations and a benchmark-aligned slice now automated.

Current repo gap:
- No official SAEBench/CE-Bench full run yet (only aligned slice).
- No explicit consistency-first training objective beyond post-hoc metrics.
- No adaptive L0 calibration loop.
- No cross-scale canonicality/stitching analysis on this task family.
- Limited causal-disentanglement scoring (RAVEL-like) in the mainline.

## Ranked Top 5 Follow-Ups (Highest Leverage)

1. Adaptive L0 Calibration + Retrain Loop (Highest impact)
- Novelty: bring "Sparse but Wrong"-style L0 correctness into this repo's full pipeline.
- Hypothesis: selecting L0 via decoder-projection diagnostics improves consistency and sparse probing at fixed compute.
- Implementation:
  - add decoder-projection metric tracking during/after SAE training,
  - perform narrow L0 search around candidate values,
  - retrain best-L0 models across multi-seed.
- Success criteria: statistically significant trained-vs-random gap increase and better CE/SAEBench-aligned scores.

2. Consistency-First Objective Sweep (PW-MCC-aware training)
- Novelty: convert consistency from reporting metric into optimized training signal.
- Hypothesis: adding a lightweight cross-seed consistency regularizer improves reproducibility with minimal EV loss.
- Implementation:
  - synchronous paired-seed training mode,
  - decoder-alignment regularizer variants,
  - ablations on regularizer strength vs EV/PWMCC.
- Success criteria: +consistency with <=5% reconstruction degradation.

3. Matryoshka + BatchTopK Head-to-Head on Algorithmic Tasks
- Novelty: first rigorous comparison in this repo's modular-arithmetic regime with matched budgets.
- Hypothesis: Matryoshka preserves higher-level abstractions while BatchTopK keeps stronger reconstruction under controlled sparsity.
- Implementation:
  - integrate both architectures with shared runner,
  - evaluate with core metrics + benchmark-aligned slice,
  - produce Pareto frontier tables.
- Success criteria: clear architecture-dependent frontier, not single-metric wins.

4. Canonicality Study: SAE Stitching/Meta-SAE for Modular Arithmetic
- Novelty: test whether non-canonicality findings from LLM contexts transfer to this small controlled setting.
- Hypothesis: even in algorithmic settings, latent sets are partially non-canonical but subspace/function-level invariants remain stable.
- Implementation:
  - implement stitching protocol between dictionary sizes,
  - meta-SAE over decoder matrices,
  - report novel-vs-reconstruction latent fractions.
- Success criteria: publishable evidence on canonicality boundary conditions.

5. Official External Benchmark Execution (SAEBench or CE-Bench Full)
- Novelty: move from "aligned slice" to official benchmark artifacts.
- Hypothesis: repo improvements that look good internally may partially fail external evals; closing this gap yields more credible claims.
- Implementation:
  - adapter from local SAE checkpoints to benchmark input format,
  - full benchmark run with pinned versions,
  - artifact bundle + reproducibility manifest.
- Success criteria: official benchmark report generated from this repo with exact version/seed provenance.

## Immediate Experimental Ideas (New)

- L0 Curriculum Study: warm-start at higher L0 then anneal to calibrated L0 vs fixed-L0 training.
- Cross-Task Transfer Consistency: train SAE on modular-addition, evaluate feature reuse on multiplication/copy tasks.
- Stability-Quality Frontier Under Data Regime Shift: sweep data fractions (10/30/70/100%) and measure consistency collapse points.
- Causal Isolation Metric for Algorithmic Features: adapt RAVEL-style intervention to modular arithmetic labels.

## Execution Recommendation

Priority order for next sprint:
1. Adaptive L0 Calibration
2. Consistency-First Objective Sweep
3. Official CE-Bench or SAEBench full adapter/run
4. Matryoshka vs BatchTopK matched-budget study
5. Canonicality (stitching/meta-SAE) study
