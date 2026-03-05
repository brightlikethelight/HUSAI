# Executive Summary

Date: 2026-03-05

## Repo Purpose

HUSAI is a reliability-first SAE research codebase. It evaluates whether learned SAE features can satisfy a strict release policy that jointly requires internal consistency, stress robustness, and external benchmark competitiveness.

## Scientific Bottom Line

- Internal consistency: positive in documented runs.
- Stress gates: passing in documented runs.
- External gates: failing at strict thresholds in both local and remote-reported summaries.
- Overall release status: `pass_all=false`.

## Evidence Status (Important)

Use `EVIDENCE_STATUS.md` for claim provenance.

- Local verified snapshot (2026-02-15): selected candidate `topk_seed123`, strict gate `pass_all=false`.
- Remote-reported final package (2026-02-18): selected candidate `relu_seed42`, strict gate `pass_all=false`.

Conclusion-level claim is stable (`pass_all=false`), but exact final-candidate identity/metric values are tier-dependent until remote package contents are fully mirrored locally.

## What Is Finished vs What Is Open

**Finished:**
- Internal track: trained-vs-random controls, k/d_sae ablations, assignment-aware objectives.
- External track: SAEBench and CE-Bench adapter runs, architecture/frontier/scaling sweeps.
- Reliability track: stress controls (random_model, transcoder, OOD), uncertainty-aware candidate selection, strict release gating.
- Engineering: TopK aux-loss fix, small-batch training, optional wandb, input validation hardening, shell=True removal.

**Open:**
- External benchmark deltas remain negative under strict thresholds.
- Remote final package not fully mirrored locally.
- Seed-complete grouped-LCB external reruns needed for stable candidate ranking.
- External-aware training objectives not yet explored.

## Repository Health Snapshot

High-impact reliability fixes completed:
- SAE trainer now includes TopK auxiliary loss in optimization.
- SAE trainer no longer crashes when dataset size is smaller than batch size.
- `wandb` import is now optional at runtime.
- Feature stability stats now handle single-model inputs safely.
- Routed frontier validates `num_experts`/`k` inputs to prevent zero-feature degeneration.
- Assignment consistency v2 now rejects empty seed/lambda lists with clear errors.
- CE-Bench custom eval now validates model-name maps before dict lookups and restores global artifact path on exceptions.
- Official benchmark harness no longer executes commands via `shell=True`.

Targeted verification:
- `pytest -q tests/unit/test_train_sae_edge_cases.py tests/unit/test_feature_matching_edge_cases.py tests/unit/test_routed_frontier_modes.py tests/unit/test_assignment_consistency_v2.py tests/unit/test_official_external_benchmark_harness.py`
- Result: `17 passed`.

## Canonical Paths

- Orientation: `START_HERE.md`
- Evidence ledger: `EVIDENCE_STATUS.md`
- Paper: `paper/sae_stability_paper.md`
- Technical report: `paper/FINAL_PAPER.md`
- Runbook: `RUNBOOK.md`
- Experiment log: `EXPERIMENT_LOG.md`
- Literature review: `LIT_REVIEW.md`
- Experiment roadmap: `docs/04-Execution/EXPERIMENT_PLAN_2026_02_20.md`
