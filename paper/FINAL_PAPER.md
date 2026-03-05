# HUSAI Manuscript Draft (Working)

Date: 2026-03-05

This file is the cycle-10 technical report companion to the canonical paper in `paper/sae_stability_paper.md`.

## Scope and Status

- Canonical paper: `paper/sae_stability_paper.md`
- Canonical evidence policy: `EVIDENCE_STATUS.md`
- Canonical plan for next experiments: `docs/04-Execution/EXPERIMENT_PLAN_2026_02_20.md`

This draft intentionally avoids placeholders and only states claims that are currently evidence-backed.

## Core Claim

HUSAI established a reliability-first SAE workflow with explicit release gates, but has not yet achieved strict external benchmark pass criteria.

## Evidence Summary

Local verified artifacts:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/selector/selected_candidate.json`
- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.json`

Remote-reported final package path:
- `results/final_packages/cycle10_final_20260218T141310Z`

Both evidence tracks agree on strict outcome:
- `pass_all=false`

## Methods Snapshot

- Candidate selection with grouped uncertainty-aware scoring.
- Stress-gated release policy with explicit threshold checks.
- External benchmark adapters for SAEBench and CE-Bench.

## Engineering and Reproducibility Snapshot

Recent reliability fixes now covered by targeted tests include:
- TopK auxiliary loss optimization wiring.
- Small-batch training stability.
- Optional `wandb` import behavior.
- Input validation and safer command execution in benchmark wrappers.

## Active Research Direction

Prioritize objective-level external-transfer improvements while preserving current stress-gate robustness.
