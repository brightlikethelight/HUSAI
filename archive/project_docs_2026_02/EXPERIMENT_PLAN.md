# Experiment Plan (High-Impact Program)

Date: 2026-03-05

## Goal

Find at least one candidate satisfying strict release criteria:
- `random_model=true`
- `transcoder=true`
- `ood=true`
- `saebench_delta_lcb >= 0`
- `cebench_interp_delta_vs_baseline_lcb >= 0`

## Assumptions

- Default hardware: single modern GPU (A100/B200 class) + CPU orchestration.
- Budget strategy: fail-fast checkpoints to stop weak branches early.

## Phase 4a: Reproduction Lock

Experiment 4a.1: phase4a rerun with fresh manifests
- Hypothesis: trained-vs-random signal remains positive after code hardening.
- Change list: no method changes; rerun for integrity.
- Metrics: PWMCC delta, CI, Mann-Whitney p-value.
- Runtime estimate: low.
- Failure modes: config drift, activation cache drift.

## Phase 4b: Baseline Suite

Experiment 4b.1: matched-protocol CE-Bench/SAEBench baseline recalibration
- Hypothesis: protocol mismatch contributes to distorted deltas.
- Change list: enforce identical row budgets, dataset slice, and baseline summary format.
- Metrics: interpretability/contrastive/independent deltas and SAEBench delta.
- Runtime estimate: medium.
- Failure modes: upstream benchmark dependency drift.

Experiment 4b.2: seed-complete external reruns
- Hypothesis: grouped-LCB ranking changes at >=5 seeds/condition.
- Change list: rerun relu/topk/routed/assignment best conditions with harmonized seeds.
- Metrics: mean/std/CI/LCB for both external deltas.
- Runtime estimate: medium-high.
- Failure modes: persistent negative deltas despite reduced variance.

## Phase 4c: Core Ablations

Experiment 4c.1: `k` sweep under external-aware objective
- Hypothesis: external-optimal `k` differs from internal-optimal `k`.
- Metrics: external LCB deltas + EV + stress metrics.
- Runtime estimate: medium.
- Failure modes: reconstruction collapse for aggressive sparsity.

Experiment 4c.2: `d_sae` sweep under external-aware objective
- Hypothesis: width changes improve one benchmark while hurting another.
- Metrics: same as 4c.1 plus compute-normalized frontier.
- Runtime estimate: medium.
- Failure modes: no Pareto gains.

## Phase 4d: SOTA-Chasing Variants

Experiment 4d.1: routed + nested hybrid frontier
- Hypothesis: combined inductive bias improves external transfer under fixed compute.
- Runtime estimate: high.
- Failure modes: optimization instability.

Experiment 4d.2: stress-aware training objective
- Hypothesis: integrating stress proxies in training lowers late-stage gate failures.
- Runtime estimate: high.
- Failure modes: over-regularization suppresses external gains.

Experiment 4d.3: reliability-calibrated selector
- Hypothesis: selection quality improves when stress uncertainty enters scoring directly.
- Runtime estimate: low-medium.
- Failure modes: ranking instability.

## Phase 4e: Stress and Robustness

Experiment 4e.1: OOD/random/transcoder stress on external-positive shortlist
- Hypothesis: external-positive candidates may still fail stress gates.
- Runtime estimate: medium.

Experiment 4e.2: release-policy threshold sweep
- Hypothesis: pass/fail boundary is narrow and must be characterized.
- Runtime estimate: low.

## Logging Specification (Mandatory)

For every run, record:
- command
- git commit
- config hash
- seeds
- dataset slice/version
- hardware/device
- artifact root

Path convention:
- `results/experiments/<program>/run_<UTC>/`

Required artifacts:
- `results.json`
- `summary.md`
- `manifest.json`
- benchmark-specific summary JSONs

Also append each run to `EXPERIMENT_LOG.md`.

## Milestones

1. M1: evidence reconciliation + phase4a lock rerun.
2. M2: seed-complete baseline suite and grouped-LCB table.
3. M3: external-aware ablations with stress-aware shortlist.
4. M4: one official benchmark slice with full reproducibility package.

## Expected Outcome

Most likely near-term win is improved confidence and ranking stability; full strict pass is uncertain and should be treated as exploratory until external LCBs are non-negative.
