# Executive Summary (Cycle 2)

Date: 2026-02-13

## Repo Purpose

HUSAI studies sparse autoencoder (SAE) consistency and interpretability, with emphasis on whether feature dictionaries are stable across random seeds and whether improvements transfer to external benchmarks.

This cycle shifted from preflight-level claims to evidence-backed execution: direct HUSAI custom-checkpoint external evaluation, architecture and scaling sweeps, objective-level consistency upgrades, and release-gating policy enforcement.

## Top 10 Issues Found

1. `P0` CE-Bench custom-checkpoint gap: no matched HUSAI CE-Bench baseline path existed at cycle start.
2. `P0` External performance deficit: all tested HUSAI checkpoints remained below matched CE-Bench baseline and below SAEBench LLM baselines.
3. `P1` Frontier initialization collapse risk: custom SAEBench model init could degenerate to near-zero activity.
4. `P1` BatchTopK inference bug risk: threshold-dependent path could fail without post-train calibration.
5. `P1` Dataset-resolution fragility: SAEBench custom eval could silently run with empty datasets unless explicitly validated.
6. `P1` Scaling-run validity risk: layer sweep initially impossible without matched layer-1 activation caches.
7. `P1` Path portability bugs: relative/absolute path assumptions caused remote run failures.
8. `P1` CE-Bench compatibility drift: dependency API mismatch (`stw` / `sae_lens.toolkit`) required shims.
9. `P1` Objective-only optimism risk: assignment-v2 improved internal metrics but failed external acceptance.
10. `P2` Release-policy incompleteness: transcoder/OOD stress evidence missing; strict release should fail.

## What We Changed (with commits)

- `1e3d94e`: fixed BatchTopK threshold calibration behavior in frontier.
- `38a00db`: normalized frontier/scaling path handling (safe repo-relative behavior).
- `bed457c`: added non-degenerate custom SAE initialization for frontier training.
- `dda27d8`: explicit SAEBench dataset propagation + fail-fast empty-dataset behavior.
- `b5aca3a`: SAEBench dataset controls (`--saebench-datasets`, `--saebench-dataset-limit`) for frontier runtime control.
- `7d1c934`: added `--max-rows` to CE-Bench compatibility runner for matched-budget baselines.
- `884a039`: hardened scaling-study dataset resolution and SAEBench dataset forwarding.
- `e2f5e8e`: added fail-fast gate flags:
  - `run_assignment_consistency_v2.py --fail-on-acceptance-fail`
  - `run_stress_gated_release_policy.py --fail-on-gate-fail`

## Best Results Achieved

| Experiment | Best Result | Evidence |
|---|---|---|
| Assignment-aware consistency v2 | delta PWMCC `+0.070804`, conservative LCB `+0.054419`, EV drop `0.000878` | `docs/evidence/phase4d_assignment_consistency_v2/run_20260213T203957Z_results.json` |
| Architecture frontier SAEBench (best among tested) | ReLU SAEBench best-minus-LLM AUC delta `-0.034640` | `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_summary_table.md` |
| Architecture frontier CE-Bench (best among tested) | TopK interpretability max `7.585395` | `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_summary_table.md` |
| Scaling study CE-Bench (best condition) | `tok30000_layer1_dsae2048_seed42` interpretability max `10.552940` | `docs/evidence/phase4e_external_scaling_study/run_20260213T203923Z_summary_table.md` |
| Matched CE-Bench baseline | interpretability max `47.951612` (public SAE, 200 rows) | `docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json` |
| Stress-gated release | `pass_all=False` (random pass; transcoder/OOD/external fail) | `docs/evidence/phase4e_stress_gated_release/run_20260213T204120Z_release_policy.json` |

## Next 5 Highest-Leverage Follow-Ups (Ranked)

1. Add matched transcoder + OOD stress tracks and enforce strict release gates in CI.
2. Expand architecture frontier to Matryoshka/RouteSAE/HierarchicalTopK under matched protocol.
3. Run multi-seed external confidence intervals for frontier/scaling candidates.
4. Add Pareto checkpoint selection over internal consistency and external benchmarks.
5. Test layer-aware architecture routing under fixed parameter budget.

## Paths to Key Artifacts

- Runbook: `RUNBOOK.md`
- Literature review: `LIT_REVIEW.md`
- Experiment plan: `EXPERIMENT_PLAN.md`
- Blog outline: `BLOG_OUTLINE.md`
- Paper outline: `PAPER_OUTLINE.md`
- Final blog: `FINAL_BLOG.md`
- Final paper: `FINAL_PAPER.md`
- High-impact report: `HIGH_IMPACT_FOLLOWUPS_REPORT.md`
- Experiment ledger: `EXPERIMENT_LOG.md`

## Key Plots

- Architecture frontier tradeoff: `docs/evidence/plots/frontier_external_tradeoff.png`
- Scaling axis effects: `docs/evidence/plots/scaling_axis_effects.png`
