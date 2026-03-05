# Project Study Guide

Updated: 2026-02-18

This guide explains what HUSAI aimed to discover, what was executed, what was learned, and what remains open.

## 1) Problem Statement

We want to know whether SAE features are trustworthy:

- Do we recover consistent features across seeds?
- Do internal improvements transfer to external interpretability benchmarks?

## 2) Final Bottom Line

- Internal consistency gains: supported.
- Stress controls: supported.
- External gains under strict gates: not supported.
- Final strict release decision: `pass_all=false`.

Primary final evidence:

- `/workspace/HUSAI/results/final_packages/cycle10_final_20260218T141310Z/meta/FINAL_INDEX.md`

## 3) Read in This Order

1. `START_HERE.md`
2. `CANONICAL_DOCS.md`
3. `EXECUTIVE_SUMMARY.md`
4. `CURRENT_STATUS_AND_STUDY_GUIDE.md`
5. `PROPOSAL_COMPLETENESS_REVIEW.md`
6. `RUNBOOK.md`
7. `EXPERIMENT_LOG.md`

## 4) Hypotheses and How We Tested Them

1. Trained-vs-random hypothesis.
- `scripts/experiments/run_phase4a_reproduction.py`

2. Architecture and scaling frontier hypothesis.
- `scripts/experiments/run_architecture_frontier_external.py`
- `scripts/experiments/run_external_metric_scaling_study.py`
- `scripts/experiments/run_matryoshka_frontier_external.py`
- `scripts/experiments/run_routed_frontier_external.py`

3. Assignment-aware objective hypothesis.
- `scripts/experiments/run_assignment_consistency_v3.py`

4. Stress-gated release hypothesis.
- `scripts/experiments/run_transcoder_stress_eval.py`
- `scripts/experiments/run_ood_stress_eval.py`
- `scripts/experiments/run_stress_gated_release_policy.py`

## 5) What We Found

1. Internal gains are reproducible.
2. External transfer remains the unresolved bottleneck.
3. Reliability-first gate policy prevented unsupported release claims.

## 6) Highest-Impact Next Steps

1. Dual-target external recovery objective.
2. Larger grouped-LCB seed support.
3. Assignment objective v4/v5 with external-aware Pareto checkpointing.
4. New architecture family under matched budget.
5. Known-circuit closure with stronger confidence bounds.
