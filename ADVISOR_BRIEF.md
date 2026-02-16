# Advisor Brief: HUSAI Current State

Date: 2026-02-16

## 1) Problem and Objective

HUSAI tests whether SAE consistency gains are reproducible across seeds and whether those gains transfer to external benchmarks under strict release controls.

Current bottom line:
- Internal consistency progress: yes.
- External competitiveness under strict gates: no (yet).
- Reliability and claim-gating: strong.

## 2) Strongest Artifact-Backed Evidence

- `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.json`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/selector/selection_summary.json`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/assignment/run_20260216T005618Z_results.json`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/routed/run_20260215T234257Z_results.json`

Latest gate:
- random pass, transcoder pass, OOD pass, external fail, `pass_all=false`.

## 3) Scientifically Clear Conclusions

1. Internal consistency gains are real but not sufficient for external success.
2. CE-Bench improved in assignment/routed sweeps, but SAEBench remained negative.
3. Selector policy (group-size threshold) changes which family is selected.
4. Strict gate enforcement prevents unsupported external claims.

## 4) Current High-Risk Gaps

1. External deltas remain negative at LCB level for selected candidate.
2. Known-circuit closure gate still fails trained-vs-random thresholds.
3. SAEBench improvement remains the principal blocker.

## 5) Highest-Impact Next Work

1. Assignment-v3 objective extension with explicit SAEBench-aware regularization.
2. Assignment/routed seed expansion so grouped-LCB selection is not threshold-constrained.
3. Joint Pareto selection requiring SAEBench floor + CE-Bench maximization.
4. Known-circuit closure upgrade with confidence-bound pass criteria.
5. Promote deterministic env and selector diagnostics as default queue behavior.

## 6) Literature Anchors (Primary Sources)

- SAEs trained on same data learn different features: https://arxiv.org/abs/2501.16615
- Feature consistency priority paper: https://arxiv.org/abs/2505.20254
- SAEBench (ICML 2025): https://proceedings.mlr.press/v267/karvonen25a.html
- SAEBench preprint: https://arxiv.org/abs/2503.09532
- CE-Bench preprint: https://arxiv.org/abs/2509.00691
- CE-Bench repository: https://github.com/Yusen-Peng/CE-Bench
- Route Sparse Autoencoders: https://arxiv.org/abs/2503.08200
- Nested/Matryoshka SAEs: https://arxiv.org/abs/2503.17547
- Transcoders vs SAEs: https://arxiv.org/abs/2501.18823
- Random-control caution for metrics: https://arxiv.org/abs/2501.17727
