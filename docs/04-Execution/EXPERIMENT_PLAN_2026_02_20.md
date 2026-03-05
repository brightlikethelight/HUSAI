# Experiment Plan (2026-02-20)

Status: Active plan for post-cycle10 external-recovery research.

## 0) Goal and Success Criteria

Primary goal:
- Find at least one SAE candidate that preserves current robustness gates and achieves non-negative uncertainty-aware external deltas on both SAEBench and CE-Bench.

Success criteria (strict):
1. `random_model=true`, `transcoder=true`, `ood=true`.
2. `saebench_delta_ci95_low >= 0`.
3. `cebench_interp_delta_vs_baseline_ci95_low >= 0`.
4. Full manifest logging for all promoted runs.

Default execution assumptions (inferred):
- Hardware: single modern GPU (A100/B200-class).
- Budget: staged, stop early on fail-fast gates.

## 1) Phase 4a: Reproduction Lock

Experiment 4a.1: Re-run phase4a trained-vs-random baseline with seed audit
- Hypothesis: baseline signal stays positive under current code and path changes.
- Change list: none (repro validation only).
- Metrics: PWMCC delta, ratio, Mann-Whitney p-value, CI.
- Runtime estimate: low.
- Failure modes: hidden data/config drift.

## 2) Phase 4b: External Baseline Integrity

Experiment 4b.1: CE-Bench matched-protocol baseline recalibration
- Hypothesis: part of historical CE-Bench deficit is protocol mismatch (row budget / data handling).
- Change list: enforce identical row budget and matched-baseline summary generation for every comparison.
- Metrics: interpretability/contrastive/independent max metrics and deltas.
- Runtime estimate: medium.
- Failure modes: residual mismatch in preprocessing or split handling.

Experiment 4b.2: Seed-complete external reruns on top existing candidates
- Hypothesis: candidate ordering changes under grouped uncertainty with >=5 seeds/condition.
- Change list: rerun best routed, assignment, and relu conditions with harmonized seeds.
- Metrics: mean, std, CI, LCB for SAEBench and CE-Bench deltas.
- Runtime estimate: medium-high.
- Failure modes: high variance and no stable winner.

## 3) Phase 4c: Core Ablations (External-Aware)

Experiment 4c.1: `k` sweep under external-aware objective
- Hypothesis: historically good internal-k is not external-optimal; best external `k` shifts lower/higher.
- Change list: external-aware training objective with fixed `d_sae`.
- Metrics: internal + external deltas with CI.
- Runtime estimate: medium.
- Failure modes: objective destabilizes reconstruction.

Experiment 4c.2: `d_sae` sweep under external-aware objective
- Hypothesis: expansion improves one external metric but hurts the other unless regularized.
- Change list: fixed `k`, sweep `d_sae`.
- Metrics: same as above + compute cost.
- Runtime estimate: medium.
- Failure modes: no Pareto improvement.

## 4) Phase 4d: SOTA-Chasing Variants

Experiment 4d.1: Route+Nested hybrid frontier (matched compute)
- Hypothesis: combining routing and nested structure improves external transfer frontier relative to either alone.
- Change list: Route-style gating + nested dictionary constraints in one branch.
- Metrics: strict-gate pass rate, external CI-LCB deltas, compute-normalized gain.
- Runtime estimate: high.
- Failure modes: optimization instability, sparse collapse.

Experiment 4d.2: Gate-aware training (transcoder + OOD constraints in objective)
- Hypothesis: moving stress controls into training reduces late-stage gate failures.
- Change list: add penalties/auxiliary terms from transcoder and OOD proxies.
- Metrics: gate pass rate and external deltas.
- Runtime estimate: high.
- Failure modes: over-regularization harming external scores.

Experiment 4d.3: Causal-faithfulness calibrated selection
- Hypothesis: selector quality improves when causal consistency terms are integrated before final policy gate.
- Change list: extend selection score with causal calibration term and uncertainty penalty.
- Metrics: selected-candidate stability across bootstrap resamples.
- Runtime estimate: low-medium.
- Failure modes: ranking instability.

## 5) Phase 4e: Stress and Robustness

Experiment 4e.1: OOD stress under external-positive candidates only
- Hypothesis: candidates that pass external deltas may regress OOD; this must be measured before any claim.
- Change list: run OOD and random-model checks immediately after each external-positive candidate.
- Metrics: `ood_drop`, trained-vs-random delta LCB.
- Runtime estimate: medium.
- Failure modes: pass external, fail OOD.

Experiment 4e.2: Release-policy sweep (threshold sensitivity)
- Hypothesis: a narrow threshold band flips release decisions; policy stability must be shown.
- Change list: sweep CI-LCB thresholds for SAEBench and CE-Bench.
- Metrics: decision robustness region.
- Runtime estimate: low.
- Failure modes: unstable pass/fail boundary.

## 6) Logging and Artifact Spec (Mandatory)

For every run:
- Record command, git commit, config hash, seeds, dataset versions, and hardware.
- Emit `results.json`, `summary.md`, `manifest.json`, and per-benchmark summaries.
- Append run to `EXPERIMENT_LOG.md` with outcome and artifact paths.

Naming convention:
- `results/experiments/<program_name>/run_<UTC>/...`
- Candidate IDs include architecture + key hyperparameters + seed.

## 7) Fail-Fast Rules

Stop any branch early if:
1. Reconstruction EV collapses below minimum acceptable floor.
2. External deltas are strongly negative for two consecutive seed blocks.
3. Stress gates consistently fail despite objective tuning.

## 8) Literature Anchors

- SAEBench: https://arxiv.org/abs/2503.09532
- CE-Bench: https://arxiv.org/abs/2509.00691
- Seed instability in SAEs: https://arxiv.org/abs/2501.16615
- Route Sparse Autoencoders: https://arxiv.org/abs/2503.08200
- Transcoders: https://arxiv.org/abs/2501.18823
- Open Problems in Mechanistic Interpretability: https://arxiv.org/abs/2501.16496
