# Literature and Competitive Landscape (Phase 3)

Date: 2026-02-12
Scope: SAE feature stability/consistency, benchmark practices, and current baselines relevant to this repository.

## 1) Most Relevant Papers (Primary Sources)

1. **Paulo et al., 2025** - *Do Sparse Autoencoders Truly Learn Sparse Representations?* ([arXiv:2501.16615](https://arxiv.org/abs/2501.16615))
- Relevance: directly studies cross-seed feature consistency in LLM SAEs.
- Use for this repo: consistency framing and seed-to-seed methodology.

2. **Song et al., 2025** - *Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs* ([arXiv:2505.20254](https://arxiv.org/abs/2505.20254))
- Relevance: positions consistency as a first-class objective and discusses PW-MCC style evaluation.
- Use: target metric framing and consistency-oriented objectives.

3. **Gao et al., 2024** - *Scaling and Evaluating Sparse Autoencoders* ([arXiv:2406.04093](https://arxiv.org/abs/2406.04093))
- Relevance: large-scale SAE training/evaluation template (TopK-heavy modern baseline practice).
- Use: baseline architecture/training defaults and scaling behavior expectations.

4. **Karvonen et al., 2025** - *SAEBench* ([arXiv:2503.09532](https://arxiv.org/abs/2503.09532))
- Relevance: multi-metric benchmark suite and cross-architecture comparisons.
- Use: external evaluation protocol beyond single-metric stability.

5. **Peng et al., 2025** - *CE-Bench* ([arXiv:2509.00691](https://arxiv.org/abs/2509.00691))
- Relevance: lightweight contrastive benchmark and practical alignment with larger benchmark suites.
- Use: low-cost external benchmark for iterative experiments.

6. **Chen et al., 2025** - *GBA: A General Theoretical Framework for Sparse Autoencoders* ([arXiv:2506.14002](https://arxiv.org/abs/2506.14002))
- Relevance: stronger theory-grounded recovery perspective.
- Use: theory-aligned ablation targets and expected failure modes.

7. **Farnik et al., 2025** - *Jointly Sparse Autoencoders* ([arXiv:2502.18147](https://arxiv.org/abs/2502.18147))
- Relevance: alternative objective (sparse computations/Jacobian) instead of only sparse activations.
- Use: candidate SOTA-chasing variant for this repo.

8. **Luber et al., 2025** - *Matryoshka SAE* ([arXiv:2503.17547](https://arxiv.org/abs/2503.17547))
- Relevance: multiresolution/hierarchical SAE formulation.
- Use: architecture variant for compute-quality tradeoff experiments.

9. **Rajamanoharan et al., 2024** - *JumpReLU SAEs* ([arXiv:2407.14435](https://arxiv.org/abs/2407.14435))
- Relevance: modern alternative SAE architecture used by strong recent work.
- Use: required architecture baseline if claiming state-of-art alignment.

10. **Heap et al., 2026 update** - *Automated Interpretability Metrics Can Spuriously Explain Model Behavior* ([arXiv:2501.17727](https://arxiv.org/abs/2501.17727))
- Relevance: warns that proxy metrics can fail even for random models.
- Use: reinforces need for random baselines and causal checks in this repo.

## 2) Strong Baseline Codebases

1. **SAELens** - official ecosystem and tooling for training/analyzing SAEs ([GitHub](https://github.com/decoderesearch/SAELens))
2. **SAEBench** - benchmark code and eval harness ([GitHub](https://github.com/adamkarvonen/SAEBench))
3. **CE-Bench** - contrastive benchmark datasets/eval scripts ([GitHub](https://github.com/Yusen-Peng/CE-Bench))
4. **Feature consistency code** - PW-MCC focused consistency analyses ([GitHub](https://github.com/xiangchensong/sae-feature-consistency))
5. **GBA code** - theory-grounded recovery implementation ([GitHub](https://github.com/FFishy-git/TamingSAE_GBA))
6. **OpenAI Sparse Autoencoder release context** - large-scale baseline framing via paper/references ([arXiv:2406.04093](https://arxiv.org/abs/2406.04093))
7. **Anthropic scaling monosemanticity context** - major practical reference ([Transformer Circuits](https://transformer-circuits.pub/2024/scaling-monosemanticity/))

## 3) Standard Evaluation Protocols in 2025-2026

From SAEBench/CE-Bench and consistency papers:
- Multi-seed, same-data comparisons with decoder/feature matching metrics.
- Multi-metric evaluation (not only reconstruction): consistency, interpretability quality, and downstream behavior metrics.
- Random/naive baseline controls to avoid proxy-metric false positives.
- Cross-architecture comparisons under matched training budgets.
- External benchmark validation (SAEBench/CE-Bench) instead of only internal tasks.

Inference: the field has moved from "single metric + pretty examples" toward benchmark-driven, multi-objective evaluation.

## 4) What "Good" Looks Like Today

- No universal single "best SAE" across all metrics/benchmarks.
- Strong papers report both quality and consistency, and include baseline controls.
- Benchmark-facing claims typically include multi-architecture and multi-seed evidence.
- Consistency targets are context-dependent; high values are demonstrated in some regimes, but not universal.

Inference: for this repo, claiming SOTA requires external benchmark evidence, not only internal modular arithmetic curves.

## 5) Gap Analysis vs This Repository

### Where this repo is currently behind

1. **Benchmark coverage gap**
- No integrated SAEBench/CE-Bench pipeline.

2. **Architecture coverage gap**
- Core path does not yet robustly compare modern variants (JumpReLU/Matryoshka/JSAE/GBA-level methods).

3. **Execution reliability gap**
- Core SAE training path currently broken by path/import/API issues; reproducibility narrative is fragmented.

4. **Artifact provenance gap**
- Claims are spread across many docs/results files without a single command->artifact manifest.

5. **LLM-scale transfer gap**
- Repo is strong on algorithmic tasks but does not yet provide a robust LLM-scale benchmarked replication path.

### Where this repo can win

1. **Controlled testbed rigor**
- Algorithmic tasks with known structure allow cleaner causal diagnosis than many LLM-only studies.

2. **Random-baseline emphasis**
- Existing work already highlights trained-vs-random comparisons; this aligns with current metric skepticism literature.

3. **Stability-reconstruction tradeoff analysis**
- With fixed engineering, this repo can produce high-quality ablation maps that are still relatively scarce in public codebases.

4. **Bridge paper opportunity**
- A strong contribution is a "controlled-task to benchmark-task" bridge: identical metrics/protocols from modular arithmetic to SAEBench/CE-Bench subsets.

## 6) Replication Details to Import Immediately

- Multi-seed protocol as a hard requirement for every claim.
- Standardized naming and artifact schema (seed, architecture, d_sae, k, dataset/task, commit hash).
- Random baseline runs for every stability metric.
- External benchmark hooks (at least one SAEBench-like and one CE-Bench-like pass).
- Confidence intervals/uncertainty reporting for all headline metrics.

## 7) Primary Sources Used

- [arXiv:2501.16615](https://arxiv.org/abs/2501.16615)
- [arXiv:2505.20254](https://arxiv.org/abs/2505.20254)
- [arXiv:2406.04093](https://arxiv.org/abs/2406.04093)
- [arXiv:2503.09532](https://arxiv.org/abs/2503.09532)
- [arXiv:2509.00691](https://arxiv.org/abs/2509.00691)
- [arXiv:2506.14002](https://arxiv.org/abs/2506.14002)
- [arXiv:2502.18147](https://arxiv.org/abs/2502.18147)
- [arXiv:2503.17547](https://arxiv.org/abs/2503.17547)
- [arXiv:2407.14435](https://arxiv.org/abs/2407.14435)
- [arXiv:2501.17727](https://arxiv.org/abs/2501.17727)
- [SAELens GitHub](https://github.com/decoderesearch/SAELens)
- [SAEBench GitHub](https://github.com/adamkarvonen/SAEBench)
- [CE-Bench GitHub](https://github.com/Yusen-Peng/CE-Bench)
- [Feature Consistency GitHub](https://github.com/xiangchensong/sae-feature-consistency)
- [GBA GitHub](https://github.com/FFishy-git/TamingSAE_GBA)
- [Anthropic Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
