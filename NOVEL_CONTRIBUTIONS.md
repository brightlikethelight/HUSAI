# Novel Contributions and Highest-Leverage Follow-Ups

Updated: 2026-02-12

## Current Reality Check

What is now true in this repository:
- CI smoke + incremental quality gates exist (`.github/workflows/ci.yml`).
- Phase 4a/4c/4e aligned runs and high-impact follow-up runs are artifact-tracked.
- Adaptive L0 calibration has strong evidence in this task regime.
- A reproducible official-benchmark harness now exists:
  - `scripts/experiments/run_official_external_benchmarks.py`
  - `results/experiments/phase4e_external_benchmark_official/run_20260212T151416Z/`
- A machine-readable claim-consistency audit now exists:
  - `scripts/analysis/verify_experiment_consistency.py`
  - `results/analysis/experiment_consistency_report.json`

What is still blocked:
- No official SAEBench/CE-Bench benchmark execution has been run from this workspace yet.
- SOTA-facing claims remain unsupported until official external benchmark artifacts are produced.

## Literature Grounding (Primary Sources)

Core stability/benchmark papers:
- Paulo & Belrose, 2025: https://arxiv.org/abs/2501.16615
- Song et al., 2025: https://arxiv.org/abs/2505.20254
- SAEBench (ICML 2025): https://proceedings.mlr.press/v267/karvonen25a.html
- SAEBench codebase: https://github.com/adamkarvonen/SAEBench
- CE-Bench, 2025: https://arxiv.org/abs/2509.00691
- CE-Bench codebase: https://github.com/Yusen-Peng/CE-Bench

Strong architecture baselines:
- OpenAI scaling/evaluation: https://arxiv.org/abs/2406.04093
- JumpReLU: https://arxiv.org/abs/2407.14435
- BatchTopK: https://arxiv.org/abs/2412.06410
- Matryoshka SAEs: https://arxiv.org/abs/2503.17547

2025 frontier variants worth explicit inclusion:
- Route Sparse Autoencoders (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.346/
- HierarchicalTopK SAEs (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.515/

## Ranked Top 5 Highest-Leverage Follow-Ups (Now)

1. Official external benchmark execution (SAEBench + CE-Bench)
- Why this is #1: claim credibility bottleneck; without this, no defensible SOTA statement.
- What to do:
  - Use `run_official_external_benchmarks.py` with explicit official commands.
  - Produce benchmark score tables + versioned manifests.
- Success criteria:
  - official benchmark artifacts committed under `results/experiments/phase4e_external_benchmark_official/`
  - exact command/version/commit provenance for every score.

2. Modern architecture frontier on fixed compute budget
- Why this matters: architecture choice likely dominates objective tweaks.
- Candidate set:
  - TopK (current), JumpReLU, BatchTopK, Matryoshka, HierarchicalTopK, RouteSAE.
- What to do:
  - matched-budget comparison (same data, epochs, seed count, optimizer schedule).
  - report Pareto fronts: consistency (PWMCC delta) vs quality (EV/MSE) vs external benchmark metrics.
- Success criteria:
  - one architecture beats current TopK baseline on at least two primary axes without catastrophic regression.

3. Consistency-objective v2 (assignment-aware/joint multi-seed)
- Why this matters: current regularizer effect is unresolved.
- What to do:
  - replace single-reference cosine penalty with assignment-aware matching or joint training across seed batch.
  - evaluate against same k=4 and k=32 controls.
- Success criteria:
  - positive trained-PWMCC gain with 95% CI excluding zero and EV drop <= 5%.

4. Transcoder-style baseline vs SAE on algorithmic tasks
- Why this matters: literature now questions whether SAE is always best representational bottleneck.
- What to do:
  - add transcoder baseline with matched parameter budget.
  - compare consistency, reconstruction, and causal intervention faithfulness.
- Success criteria:
  - clear regime map: where SAE wins vs where transcoder wins.

5. Stress-test suite as mandatory gate before claims
- Why this matters: current evidence is strong but narrow.
- What to do:
  - OOD task transfer, data-fraction scaling curves, noise robustness, intervention stability.
  - enforce `verify_experiment_consistency.py` + stress-test pass before writeup updates.
- Success criteria:
  - reproducible stress-test table with confidence intervals and failure mode diagnostics.

## New High-Impact Experiment Ideas

- L0 curriculum schedule:
  - warm start at high k, anneal to calibrated low k, compare against fixed-k baselines.
- Joint dictionary overlap objective:
  - directly optimize overlap with permutation-invariant assignment cost.
- Cross-task dictionary transfer:
  - train on modular addition, test reuse on multiplication/copy-style tasks.
- Causal-faithfulness co-primary endpoint:
  - promote intervention metrics from auxiliary to selection criterion.

## Immediate Execution Sequence

1. Run official benchmark commands through the new harness (SAEBench and/or CE-Bench).
2. Add architecture frontier runner with unified config schema.
3. Implement consistency-objective v2 and compare against existing v1.
4. Run stress-test suite and fold into claim-consistency audit.
5. Update paper/blog only from audited artifact JSONs.
