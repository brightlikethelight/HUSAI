# High-Impact Follow-Ups Report

Date: 2026-03-05

## Previous Top-5 Status

1. CI (lint + typecheck + pytest) and fail-fast smoke workflow.
- Status: Completed.
- Evidence: `.github/workflows/ci.yml`, `scripts/ci/smoke_pipeline.sh`.

2. Remove absolute paths in core experiment/analysis launchers.
- Status: Completed for active launcher paths.
- Evidence: current `scripts/experiments/*` and `scripts/analysis/*` no longer hardcode `/workspace` or user-home roots.

3. Execute phase4a multi-seed reproduction with manifest logging.
- Status: Completed.
- Evidence: `results/experiments/phase4a_trained_vs_random/` and entries in `EXPERIMENT_LOG.md`.

4. Run core ablations (`k`, `d_sae`) with confidence intervals.
- Status: Completed.
- Evidence: `results/experiments/phase4c_core_ablations/run_20260212T091848Z/`.

5. Add external benchmark-aligned slice before SOTA claims.
- Status: Completed as harness + slice path; strict external gains still unresolved.
- Evidence: `scripts/experiments/run_external_benchmark_slice.py`, `scripts/experiments/run_official_external_benchmarks.py`.

## Highest-Leverage Next 5 (Ranked)

1. Mirror remote final package into repo-local evidence and close the claim gap.
- Why: current Tier1/Tier2 mismatch blocks fully auditable final-candidate claims.
- Deliverable: local copy or export of `results/final_packages/cycle10_final_20260218T141310Z` metadata under `docs/evidence/` with checksum map.

2. Seed-complete external reruns (>=5 seeds/condition) under grouped-LCB selection.
- Why: external ranking remains underpowered/unstable across candidate families.
- Deliverable: harmonized multi-seed reruns for relu/topk/routed/assignment with CIs and LCB table.

3. Matched-protocol CE-Bench and SAEBench calibration sweep.
- Why: baseline/protocol mismatch can distort deltas and candidate comparisons.
- Deliverable: protocol manifest that enforces matched row budgets, dataset slices, model hooks, and summary schema.

4. External-aware objective branch (multi-objective SAEBench+CE with stress constraints).
- Why: internal consistency optimization alone has not transferred externally.
- Deliverable: training branch with weighted external proxies + stress regularization and ablation report.

5. One official benchmark slice run end-to-end in a clean environment for claim hardening.
- Why: custom adapters are strong, but a fully official run is still needed for stronger external comparability.
- Deliverable: reproducible `--execute` run artifact bundle from `run_official_external_benchmarks.py` with command logs.

## New Novel Contribution Opportunities

1. Reliability-calibrated selector scoring.
- Incorporate uncertainty and stress-consistency terms directly in candidate scoring, not only in post-hoc gates.

2. Stress-aware curriculum training.
- Use OOD/transcoder proxy penalties during training to reduce late-stage gate failures.

3. Cross-layer transfer diagnostics.
- Train at one hook layer and evaluate transfer to adjacent layers to measure representation portability.

4. Protocol-conditioned architecture frontier.
- Compare routed, nested/matryoshka, and TopK variants under strict matched-compute and matched-benchmark protocols.

5. Reconciliation-aware reporting standard.
- Publish explicit claim tiers (local verified vs remote-reported) as part of reproducibility checklists.

## Literature Anchors (Primary Sources)

- SAEBench paper: https://arxiv.org/abs/2503.09532
- SAEBench repo: https://github.com/adamkarvonen/SAEBench
- CE-Bench paper: https://aclanthology.org/2025.blackboxnlp-1.1/
- RouteSAE: https://arxiv.org/abs/2503.08200
- Transcoders: https://arxiv.org/abs/2501.18823
- Seed instability in SAEs: https://arxiv.org/abs/2501.16615
- JumpReLU SAEs: https://arxiv.org/abs/2407.14435
- BatchTopK SAEs: https://arxiv.org/abs/2412.06410
- Nested/Matryoshka SAEs: https://arxiv.org/abs/2503.17547
