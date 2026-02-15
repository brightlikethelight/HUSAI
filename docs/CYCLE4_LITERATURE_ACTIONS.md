# Cycle-4 Literature-to-Execution Map

Updated: 2026-02-15

This note maps primary-source findings to concrete HUSAI cycle-4 experiment actions.

## Primary Sources (Verified)

1. SAEBench (Karvonen et al., 2025)
- https://arxiv.org/abs/2503.09532
- Why it matters here:
  - reinforces multi-metric, multi-seed benchmark discipline for SAE claims.
- Direct HUSAI action:
  - keep grouped LCB candidate selection default.
  - keep strict external gate tied to artifact-backed deltas.

2. CE-Bench (Peng et al., 2025)
- https://arxiv.org/abs/2509.00691
- Why it matters here:
  - external evaluation should include contrastive explanation quality, not just internal reconstruction/stability.
- Direct HUSAI action:
  - maintain matched-baseline CE-Bench deltas in selector/gate.
  - keep CE-Bench max-row matched protocol for fair deltas.

3. Route Sparse Autoencoders (Liu et al., 2025)
- https://arxiv.org/abs/2503.08200
- Why it matters here:
  - routing/specialization architecture family may shift external Pareto frontier.
- Direct HUSAI action:
  - run at least one non-TopK family under matched budget (matryoshka now, route next).

4. Transcoders Beat Sparse Autoencoders? (Makelov et al., 2025)
- https://arxiv.org/abs/2501.18823
- Why it matters here:
  - SAE quality claims need direct stress comparisons against stronger alternatives.
- Direct HUSAI action:
  - transcoder stress sweep with CI-LCB gate and fail-fast mode.

5. Jacobian Sparse Autoencoders (Awe et al., 2025)
- https://proceedings.mlr.press/v267/awe25a.html
- Why it matters here:
  - Jacobian/geometric constraints can improve feature consistency and may reduce seed instability.
- Direct HUSAI action:
  - add Jacobian-style regularization variant in next architecture/objective branch if cycle-4 fails external gates.

6. Can Sparse Autoencoders Reason? (de Chantal et al., 2025)
- https://arxiv.org/abs/2503.18878
- Why it matters here:
  - reasoning-focused stress tests are necessary before broad interpretability claims.
- Direct HUSAI action:
  - keep known-circuit closure + trained-vs-random CI track as hard proposal-closure criterion.

## Execution Priorities (Now)

1. Complete current queue and run post-queue cycle-4 script.
2. Require transcoder sweep CI-LCB pass (`delta_lcb >= 0`).
3. Use grouped LCB selector outputs as release candidate default.
4. Run matryoshka external matched-budget sweep.
5. Run assignment-v3 with external-aware Pareto selection.
6. Run known-circuit closure and include CI-based trained-vs-random deltas.

## If Cycle-4 Still Fails External Gate

1. Add RouteSAE family under identical matched-budget protocol.
2. Add Jacobian-style regularization branch in assignment-v3 family.
3. Expand seeds from 3 to 5 for final candidate-only runs and gate on LCB.
