# Paper Outline

Working title:
- *SAE Stability Under Controlled Tasks: Reproducibility, Baselines, and Tradeoffs*

## Abstract
- Problem, method, key results, and practical implications.

## 1. Introduction
- Why reproducibility of SAE features is foundational.
- Prior concern from consistency literature.
- This paper's contributions.

## 2. Related Work
- SAE scaling work.
- Consistency-focused work.
- Benchmark-centric work (SAEBench/CE-Bench).
- Theory-grounded recovery/identifiability work.

## 3. Methods
- Datasets/tasks and model setup.
- SAE architectures and training settings.
- Multi-seed protocol.
- Metrics (PWMCC + random baselines + quality metrics).

## 4. Reliability and Reproducibility Protocol
- Environment controls and deterministic settings.
- Command->artifact manifest strategy.
- CI/test gates.

## 5. Experiments
- 5.1 Baseline reproduction
- 5.2 Baseline suite (ReLU/TopK/linear)
- 5.3 Core ablations (`k`, `d_sae`, normalization, training dynamics)
- 5.4 SOTA-chasing variants (JumpReLU/Matryoshka/JSAE-style)
- 5.5 Stress tests (task/data/activation source)

## 6. Results
- Main quantitative tables.
- Stability-quality Pareto curves.
- Trained-vs-random comparisons.
- Cross-task and benchmark-lite validation.

## 7. Analysis
- Regimes where stability improves/degrades.
- Relationship to theory and benchmark observations.
- Metric robustness and failure cases.

## 8. Limitations
- Controlled-task scope.
- Architecture/compute limits.
- Remaining benchmark coverage gaps.

## 9. Broader Impact
- Scientific reliability implications for interpretability claims.
- Cautions for deployment/safety narratives.

## 10. Conclusion
- Key takeaways and next steps.

## Appendix A: Full Hyperparameters
- Per-run config tables and seeds.

## Appendix B: Reproducibility Checklist
- Environment, data versions, commit hashes, scripts, manifests.

## Appendix C: Additional Plots and Failure Cases
- Negative results, sensitivity plots, and run diagnostics.
