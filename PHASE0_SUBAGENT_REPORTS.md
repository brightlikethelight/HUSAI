# Phase 0 Subagent Reports (Simulated Due Agent Thread Limit)

Date: 2026-02-12
Constraint: new agent threads were unavailable (`agent thread limit reached`), so the six requested subagent tracks were executed manually with isolated scopes and evidence pointers.

## Subagent A - Repo Archaeologist

Deliverables completed:
- Architecture map: `ARCHITECTURE.md`
- Phase-1 goal/claims summary: `PHASE1_REPO_UNDERSTANDING.md`
- Critical path map: `CRITICAL_PATH.md`

Key findings:
- Core path is transformer -> activation extraction -> SAE training -> stability analysis.
- Main execution entrypoints are in `scripts/training/`, `scripts/analysis/`, and `scripts/experiments/`.
- Minimal smoke workflow is stable and documented in `RUNBOOK.md`.

Actionable tasks:
1. Keep `RUNBOOK.md` as source-of-truth command surface.
2. Add a generated dependency graph artifact in CI for drift detection.

## Subagent B - Reliability/Infra Engineer

Deliverables completed:
- CI workflow: `.github/workflows/ci.yml`
- Smoke script: `scripts/ci/smoke_pipeline.sh`
- Runbook updates: `RUNBOOK.md`
- New Make targets: `Makefile` (`benchmark-official`, `audit-results`)

Key findings:
- Incremental lint/typecheck + full pytest are operational.
- Deterministic experiment manifests are present for core follow-up runs.
- Environment lockfile remains unresolved (still split across specs).

Actionable tasks:
1. Add lockfile strategy (`conda-lock` or `pip-tools`) and pin CUDA matrix.
2. Expand typecheck/lint from incremental to full-repo gates.

## Subagent C - Debugger/Quality Engineer

Deliverables completed:
- Bug audit docs: `BUGS.md`, `AUDIT.md`
- Claim-consistency validator: `scripts/analysis/verify_experiment_consistency.py`
- Result audit artifacts: `results/analysis/experiment_consistency_report.json`

Key findings:
- Current headline conclusions are consistent with artifact JSONs.
- Adaptive L0 effect is strong and stable under fair-control comparison.
- Consistency-regularizer v1 remains unresolved (CI includes zero).

Actionable tasks:
1. Make `audit-results` a required pre-merge check for research docs.
2. Add regression tests for key metric calculators (PWMCC, bootstrap CI).

## Subagent D - Literature + Competitive Landscape

Deliverables completed:
- Literature synthesis: `LIT_REVIEW.md`
- Frontier strategy updates: `NOVEL_CONTRIBUTIONS.md`

Primary-source anchors:
- SAEBench: https://proceedings.mlr.press/v267/karvonen25a.html
- SAEBench repo: https://github.com/adamkarvonen/SAEBench
- CE-Bench: https://arxiv.org/abs/2509.00691
- CE-Bench repo: https://github.com/Yusen-Peng/CE-Bench
- OpenAI scaling/eval: https://arxiv.org/abs/2406.04093
- JumpReLU / BatchTopK / Matryoshka: https://arxiv.org/abs/2407.14435, https://arxiv.org/abs/2412.06410, https://arxiv.org/abs/2503.17547
- RouteSAE / HierarchicalTopK: https://aclanthology.org/2025.emnlp-main.346/, https://aclanthology.org/2025.emnlp-main.515/

Actionable tasks:
1. Execute official SAEBench/CE-Bench via harness before any external-performance claim.
2. Add matched-budget architecture frontier study including RouteSAE/HierarchicalTopK.

## Subagent E - Research Scientist (Experiment Designer)

Deliverables completed:
- Follow-up result synthesis: `HIGH_IMPACT_FOLLOWUPS_REPORT.md`
- Updated ranked roadmap: `NOVEL_CONTRIBUTIONS.md`

Key findings:
- Adaptive L0 is the best validated near-term lever.
- Objective-level consistency requires stronger formulations.
- External benchmark execution is now the highest leverage bottleneck.

Actionable tasks:
1. Run official benchmark suite through `run_official_external_benchmarks.py`.
2. Build architecture frontier runner with strict matched-budget configs.
3. Implement assignment-aware consistency objective v2.

## Subagent F - Product/Writeup Editor

Deliverables completed:
- Paper writeup: `FINAL_PAPER.md`
- Blog writeup: `FINAL_BLOG.md`
- Updated operational narrative: `RUNBOOK.md`, `EXPERIMENT_LOG.md`

Key findings:
- Narrative now aligns with conservative evidence boundaries.
- External benchmark claim boundary is explicit.
- Added audit tooling to keep future claims in sync with artifacts.

Actionable tasks:
1. Auto-generate summary tables/figures in docs from result JSONs.
2. Add a single "claims ledger" doc updated by script in CI.

## Consolidated Handoff

Highest-priority execution order:
1. Official SAEBench/CE-Bench command execution with manifests.
2. Architecture frontier baseline suite under matched budgets.
3. Consistency-objective v2 and stress-test gate integration.
