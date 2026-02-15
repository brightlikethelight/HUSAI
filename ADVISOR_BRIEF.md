# Advisor Brief: HUSAI Current State

Date: 2026-02-15

## 1) Problem and Objective

HUSAI asks whether SAE consistency gains are real across seeds and whether those gains transfer to external interpretability benchmarks.

Current bottom line:
- Internal consistency progress: yes.
- External competitiveness: no (yet).
- Engineering reliability and claim gating: strong.

## 2) Strongest Artifact-Backed Evidence

Primary synthesis files:
- `docs/evidence/cycle3_queue_final/cycle3_final_synthesis_run_20260214T210734Z.md`
- `docs/evidence/cycle3_queue_final/release_policy_run_20260214T225029Z.json`

Cycle-3 outcomes:
- Frontier multiseed completed (`4 architectures x 5 seeds`, 20 records).
- Scaling multiseed completed (24 records).
- Strict release gate: `pass_all=false`.
- Stress gate details:
  - random model: pass
  - OOD: pass
  - transcoder: fail
  - external: fail

## 3) Scientifically Clear Conclusions

1. Internal metrics can improve, but that alone does not transfer externally.
2. SAEBench and CE-Bench preferences currently conflict in the explored region.
3. Strict gate framework prevents overclaiming and is now integrated into workflow.

## 4) Current High-Risk Gaps

1. External deltas remain negative under matched baselines.
2. Transcoder gate remains below threshold.
3. Known-circuit closure track is not yet fully green.

## 5) Highest-Impact Next Work

1. Transcoder stress hyper-sweep with CI-based acceptance.
2. Condition-grouped uncertainty-aware selector as default release criterion.
3. New architecture family trial (RouteSAE or Matryoshka-style) under matched budgets.
4. Assignment-aware objective v3 with external-aware Pareto checkpointing.
5. Known-circuit closure completion with trained-vs-random confidence bounds.

## 6) Verified Literature Anchors

- SAEBench: https://arxiv.org/abs/2503.09532
- CE-Bench: https://aclanthology.org/2025.findings-acl.854/
- Route Sparse Autoencoders: https://arxiv.org/abs/2503.08200
- Transcoders Beat Sparse Autoencoders?: https://arxiv.org/abs/2501.18823
- Can Sparse Autoencoders Reason?: https://arxiv.org/abs/2507.18006
