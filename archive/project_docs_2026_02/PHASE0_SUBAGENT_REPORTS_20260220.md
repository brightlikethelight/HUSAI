# Phase 0 Subagent Reports (2026-02-20)

This document captures scoped subagent audits, evidence, and concrete actions taken in this pass.

## Subagent A: Repo Archaeologist (Path Hygiene)

Findings:
- Cycle/queue launchers hard-coded `/workspace/HUSAI`, `/workspace/CE-Bench`, and `/tmp/...` paths.
- This created environment-coupled behavior and weakened reproducibility portability.

Evidence:
- `scripts/experiments/run_cycle4_followups_after_queue.sh`
- `scripts/experiments/run_cycle5_external_push.sh`
- `scripts/experiments/run_cycle6_saeaware_push.sh`
- `scripts/experiments/run_cycle7_pareto_push.sh`
- `scripts/experiments/run_cycle8_robust_pareto_push.sh`
- `scripts/experiments/run_cycle9_novelty_push.sh`
- `scripts/experiments/run_cycle10_external_recovery.sh`
- `scripts/experiments/run_b200_high_impact_queue.sh`

Actions taken:
- Replaced fixed absolute roots with `ROOT_DIR` detection.
- Introduced launcher defaults:
  - `HUSAI_TMP_ROOT=${ROOT_DIR}/tmp`
  - `HUSAI_MPLCONFIGDIR=${HUSAI_TMP_ROOT}/mpl`
  - `CEBENCH_REPO=${ROOT_DIR}/../CE-Bench`
- Replaced absolute cache/artifact paths with `HUSAI_TMP_ROOT`-based paths.

## Subagent B: Reliability/Infra Engineer (CI/Determinism)

Findings:
- CI had smoke + quality jobs, but no enforced pre-commit parity and no fail-fast pytest option.

Evidence:
- `.github/workflows/ci.yml`
- `.pre-commit-config.yaml`

Actions taken:
- Updated CI to run `pre-commit run --all-files`.
- Updated pytest command to fail fast: `pytest tests -q -x --maxfail=1`.
- Local note: full pre-commit hook execution cannot fully complete in this network-restricted environment (hook fetch blocked).

## Subagent C: Documentation Coherence Reviewer

Findings:
- Canonical docs still pointed readers to remote-only package paths without clear local-vs-remote distinction.
- Archived plan docs could be mistaken for current status.

Evidence:
- `CANONICAL_DOCS.md`
- `docs/evidence/README.md`
- `REPO_NAVIGATION.md`
- `archive/project_docs_2026_02/EXPERIMENT_PLAN.md`
- `archive/project_docs_2026_02/CYCLE10_EXTERNAL_RECOVERY_PLAN.md`

Actions taken:
- Clarified canonical local snapshot vs remote final package references.
- Added historical-status notes in archived plan docs.
- Normalized references away from hard-coded `/workspace/HUSAI/...` paths in canonical docs.

## Subagent D: External Benchmark Reliability Engineer

Findings:
- Artifact writing lacked centralized failure surfacing and pre-write free-space checks.
- This increased fragility during long benchmark runs.

Evidence:
- `scripts/experiments/run_official_external_benchmarks.py`
- `scripts/experiments/run_husai_saebench_custom_eval.py`
- `scripts/experiments/run_cebench_compat.py`
- `scripts/experiments/run_husai_cebench_custom_eval.py`

Actions taken:
- Added `scripts/experiments/benchmark_utils.py` with:
  - `ensure_free_space(...)`
  - safe JSON/markdown persistence wrappers.
- Integrated these guards/writers into all external benchmark runners listed above.
- Added `sys.path` safety in benchmark scripts to avoid import fragility when launched from non-root directories.

## Subagent E: Results Sanity Reviewer

Findings:
- Some architecture/external comparisons are under-seeded and require stronger grouped uncertainty reporting.
- CE-Bench delta comparisons are sensitive to row-budget mismatch and should be protocol-matched before interpretation.

Evidence:
- `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_summary.md`
- `docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json`

Actions taken:
- Reflected these constraints and priorities in:
  - `HIGH_IMPACT_FOLLOWUPS_REPORT.md`
  - `NOVEL_CONTRIBUTIONS.md`
- Prioritized next experiments around grouped-LCB external reruns and matched-protocol CE-Bench deltas.

## Literature and Competitive Alignment (Lead Synthesis)

Primary sources refreshed and integrated into next-step design:
- SAEBench: https://arxiv.org/abs/2503.09532
- CE-Bench: https://arxiv.org/abs/2509.00691
- Seed instability in SAEs: https://arxiv.org/abs/2501.16615
- RouteSAE: https://arxiv.org/abs/2503.08200
- Transcoders: https://arxiv.org/abs/2501.18823
- Open problems in mech interp: https://arxiv.org/abs/2501.16496

## Verification Executed

- `bash -n` on all modified launchers: pass.
- `python -m py_compile` on modified benchmark scripts: pass.
- `pytest tests -q`: `113 passed`.
- Harness preflight smoke: `run_official_external_benchmarks.py --skip-saebench --skip-cebench` succeeded and emitted manifest/config/preflight/summary artifacts.
