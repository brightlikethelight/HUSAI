# Advisor Brief: HUSAI Current State

Date: 2026-02-15

## 1) Problem and Objective

HUSAI tests whether SAE consistency gains are reproducible across seeds and whether those gains transfer to external benchmarks under strict release controls.

Current bottom line:
- Internal consistency progress: yes.
- External competitiveness: no (yet).
- Reliability and claim-gating: strong.

## 2) Strongest Artifact-Backed Evidence

- `docs/evidence/cycle4_followups_run_20260215T220728Z/release/release_policy.json`
- `docs/evidence/cycle4_followups_run_20260215T220728Z/selector/selection_summary.json`
- `docs/evidence/cycle4_followups_run_20260215T220728Z/assignment_external/results.json`
- `docs/evidence/cycle4_followups_run_20260215T220728Z/routed/results.json`

Latest gate:
- random pass, transcoder pass, OOD pass, external fail, `pass_all=false`.

## 3) Scientifically Clear Conclusions

1. Internal consistency gains are real but not sufficient for external success.
2. SAEBench and CE-Bench pressures differ and create a nontrivial tradeoff surface.
3. Strict gate enforcement prevents unsupported external claims.
4. Assignment-v3 external completion removed a key confounder; the external gap persists.

## 4) Current High-Risk Gaps

1. External deltas remain negative at LCB level.
2. Known-circuit closure gate still fails trained-vs-random thresholds.
3. Routed-family run appears under-tuned (`train_l0` too low), so architectural potential is not yet fully tested.

## 5) Highest-Impact Next Work

1. Routed-family hyper-sweep (capacity/router regularization/lr) with external LCB acceptance criteria.
2. Assignment-v3 external-aware sweep with larger seed set and hard external constraints in selection.
3. CI-aware grouped selection rerun across expanded candidate pool.
4. Known-circuit closure improvement track with explicit confidence-bound targets.
5. Uniform W&B logging in queue scripts.

## 6) Literature Anchors (Primary Sources)

- SAEBench: https://arxiv.org/abs/2503.09532
- CE-Bench (ACL Anthology): https://aclanthology.org/2025.findings-acl.854/
- CE-Bench (arXiv): https://arxiv.org/abs/2509.00691
- Route Sparse Autoencoders: https://arxiv.org/abs/2503.08200
- Crosscoders and model diffing (Anthropic): https://www.anthropic.com/research/tracing-thoughts-language-model
- Matryoshka SAEs: https://arxiv.org/abs/2505.24473
