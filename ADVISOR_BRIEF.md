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
- `docs/evidence/cycle4_postfix_reruns/matryoshka/run_20260215T203710Z_results.json`
- `docs/evidence/cycle4_postfix_reruns/known_circuit_run_20260215T203809Z_summary.json`

Latest gate:
- random pass, transcoder pass, OOD pass, external fail, `pass_all=false`.

## 3) Scientifically Clear Conclusions

1. Internal consistency gains are real but not sufficient for external success.
2. SAEBench and CE-Bench pressures differ and create a nontrivial frontier.
3. Strict gate enforcement prevents unsupported external claims.
4. Post-fix reruns converted two previously invalid tracks into usable evidence.

## 4) Current High-Risk Gaps

1. External deltas remain negative at LCB level.
2. Assignment-v3 external stage needs dimension-compatible rerun.
3. No candidate currently passes strict release policy end-to-end.

## 5) Highest-Impact Next Work

1. Assignment-v3 rerun with external-compatible `d_model`.
2. Add RouteSAE family under matched protocol.
3. Re-run grouped-LCB selection with updated candidate pool.
4. Re-run stress gates and strict release gate.
5. Refresh canonical summaries from new gate artifacts.

## 6) Literature Anchors (Primary Sources)

- SAEBench: https://arxiv.org/abs/2503.09532
- CE-Bench (paper page): https://aclanthology.org/2025.blackboxnlp-1.1/
- CE-Bench (arXiv): https://arxiv.org/abs/2509.00691
- Route Sparse Autoencoders: https://arxiv.org/abs/2503.08200
- Transcoders vs SAEs: https://arxiv.org/abs/2501.18823
- Can Sparse Autoencoders Reason?: https://arxiv.org/abs/2503.18878
- Matryoshka SAEs: https://arxiv.org/abs/2505.24473
