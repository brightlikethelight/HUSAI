# Advisor Brief: HUSAI Current State

Date: 2026-02-15

## 1) Problem and Objective

HUSAI tests whether SAE consistency gains are reproducible across seeds and whether those gains transfer to external benchmarks under strict release controls.

Current bottom line:
- Internal consistency progress: yes.
- External competitiveness: no (yet).
- Reliability and claim-gating: strong.

## 2) Strongest Artifact-Backed Evidence

- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.json`
- `docs/evidence/cycle4_followups_run_20260215T190004Z/transcoder_sweep/summary.md`
- `docs/evidence/cycle4_followups_run_20260215T190004Z/ood/ood_stress_summary.md`

Latest gate:
- random pass, transcoder pass, OOD pass, external fail, `pass_all=false`.

## 3) Scientifically Clear Conclusions

1. Internal consistency gains are real but not sufficient for external success.
2. SAEBench and CE-Bench pressures differ and create a nontrivial frontier.
3. Strict gate enforcement prevents unsupported external claims.

## 4) Current High-Risk Gaps

1. External deltas remain negative at LCB level.
2. Matryoshka frontier evidence run failed and needs post-fix rerun.
3. Known-circuit closure requires rerun after corrected basis mapping.
4. Assignment-v3 must be rerun with external-compatible dimensional setup.

## 5) Highest-Impact Next Work

1. Matryoshka rerun under matched budget with fixed training+adapter path.
2. Known-circuit closure rerun with confidence bounds.
3. Assignment-v3 external-compatible rerun.
4. Add RouteSAE family under matched protocol.
5. Re-run strict release gate and update canonical status from new artifacts.

## 6) Literature Anchors (Primary Sources)

- SAEBench: https://arxiv.org/abs/2503.09532
- CE-Bench (paper page): https://aclanthology.org/2025.blackboxnlp-1.1/
- CE-Bench (arXiv): https://arxiv.org/abs/2509.00691
- Route Sparse Autoencoders: https://arxiv.org/abs/2503.08200
- Transcoders vs SAEs: https://arxiv.org/abs/2501.18823
- Can Sparse Autoencoders Reason?: https://arxiv.org/abs/2503.18878
- Matryoshka SAEs: https://arxiv.org/abs/2505.24473
