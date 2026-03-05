# HUSAI: Reliability-First SAE Evaluation Under Strict Release Gates

Date: 2026-03-05

## Abstract

HUSAI evaluates whether sparse autoencoder (SAE) improvements are trustworthy under a release policy that jointly requires internal consistency, stress robustness, and external benchmark competitiveness. The repository contains a full scripted workflow: reproduction controls, ablations, external benchmark adapters (SAEBench and CE-Bench), uncertainty-aware candidate selection, and stress-gated policy checks. The key empirical conclusion is negative but stable: strict release criteria are not met (`pass_all=false`). Internal and stress metrics improve, but external deltas remain below strict thresholds.

## 1. Problem

Many SAE projects report internal improvements without proving external interpretability gains. HUSAI addresses this by requiring all of the following before release claims:
- Internal trained-vs-random improvement
- Stress robustness (`random_model`, `transcoder`, `OOD`)
- External benchmark deltas (SAEBench and CE-Bench)

## 2. Method Overview

Core workflow:
1. Train/evaluate candidate SAEs across architecture/objective sweeps.
2. Run external evaluations with custom adapters.
3. Select release candidate with grouped uncertainty-aware scoring.
4. Apply strict release-gate policy.

Core implementations:
- Selection: `scripts/experiments/select_release_candidate.py`
- Release gate: `scripts/experiments/run_stress_gated_release_policy.py`
- SAEBench adapter: `scripts/experiments/run_husai_saebench_custom_eval.py`
- CE-Bench adapter: `scripts/experiments/run_husai_cebench_custom_eval.py`

## 3. Results and Evidence Status

Evidence is split across local verified artifacts and remote-reported package references. See `EVIDENCE_STATUS.md`.

### 3.1 Locally Verified Snapshot (Tier 1)

Sources:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/selector/selected_candidate.json`
- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.json`

Observed:
- Selected candidate: `topk_seed123`
- `saebench_delta` (LCB): `-0.04478959689939781`
- `cebench_interp_delta_vs_baseline` (LCB): `-40.467037470119465`
- `pass_all=false`

### 3.2 Remote-Reported Final Package (Tier 2)

Documented path:
- `results/final_packages/cycle10_final_20260218T141310Z`

Canonical docs reference:
- Selected candidate: `relu_seed42`
- `saebench_delta = -0.029153650997086358`
- `cebench_interp_delta_vs_baseline = -43.71286609575971`
- `pass_all=false`

### 3.3 Stable Conclusion

Despite candidate/metric mismatch across evidence tiers, both tracks agree on the release decision:
- External gate not satisfied.
- Strict release remains blocked (`pass_all=false`).

## 4. Engineering Reliability Improvements in This Update

This update fixes high-impact code-path defects and adds regression tests:
- TopK auxiliary loss is now optimized in `src/training/train_sae.py`.
- Small-dataset training no longer crashes when `batch_size > num_samples`.
- `wandb` is optional at runtime.
- Single-SAE feature-stat calls no longer crash in `src/analysis/feature_matching.py`.
- Routed frontier now validates `1 <= num_experts <= d_sae` and `k >= 1`.
- Assignment-v2 now rejects empty seeds/lambdas with clear errors.
- CE-Bench custom eval validates model-name mappings and always restores global artifact path.
- Official benchmark harness no longer uses `shell=True`.

Targeted test result:
- `17 passed` for newly added edge-case regression suite.

## 5. Limitations

1. Final remote package is not fully mirrored locally; exact final-candidate claims require local mirroring for full reproducibility.
2. External metrics remain negative under strict thresholds.
3. Official benchmark execution in fully standardized external environments remains environment-dependent.

## 6. Next Step

Highest leverage is external-transfer recovery under strict gate discipline: objective-level external coupling, seed-complete grouped evaluation, and matched-protocol baseline recalibration.
