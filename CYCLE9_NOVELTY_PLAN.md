# Cycle-9 Novelty and External-Competitiveness Plan

Date: 2026-02-16
Status: queued after cycle7/cycle8 complete

## 1) Evidence-Grounded Problem Statement

Current strongest completed evidence (cycle7 routed + assignment a1):
- Routed sweeps remain external-negative:
  - best SAEBench delta: `-0.063807`
  - best CE-Bench delta: `-36.183461`
- Assignment a1 improved both external metrics vs routed but still failed strict external gate:
  - SAEBench delta: `-0.043546`
  - CE-Bench delta: `-34.468482`
  - internal LCB remains strong (`0.83984`), so external transfer is the main blocker.

Implication:
- We are no longer blocked on internal consistency engineering.
- We are blocked on **representation quality under external protocols**.

## 2) Literature Anchors Driving Cycle-9 Design

- SAEBench establishes broad SAE evaluation and warns against narrow metric optimism:
  - https://arxiv.org/abs/2503.09532
- CE-Bench introduces contrastive causality/correlation stress and reports weak alignment with prior benchmarks:
  - https://arxiv.org/abs/2509.00691
- RouteSAE reports improvement from conditional routing and stronger scalability than monolithic SAEs:
  - https://arxiv.org/abs/2503.08200
- Robust SAEs show improved behavior under adversarial/distribution-shift settings:
  - https://arxiv.org/abs/2505.24473
- Supervised SAEs report superior concept correspondence and OOD robustness vs unsupervised SAEs:
  - https://arxiv.org/abs/2505.16004
- PolySAE reports better reconstruction at fixed budget via polysemantic pre-bases:
  - https://arxiv.org/abs/2602.01322
- Transcoders vs SAEs highlights metric-dependent conclusions and the need for multi-metric checks:
  - https://arxiv.org/abs/2501.18823

## 3) Highest-Impact Cycle-9 Experiments (Ranked)

1. Assignment-v4 with supervised external proxy objective.
- Hypothesis: adding a lightweight supervised auxiliary objective on cached benchmark-labeled activations will improve SAEBench delta without collapsing CE-Bench.
- Minimal implementation path:
  - extend `run_assignment_consistency_v3.py` to optionally optimize a probe-separability proxy over `saebench_datasets` batches.
  - preserve current selection/gate machinery for apples-to-apples comparison.
- Acceptance:
  - grouped-LCB SAEBench delta improves by >= `+0.01` vs cycle7 a1 best.
  - CE-Bench delta does not regress by more than `-1.0` absolute points.

2. Robust-routed frontier with explicit noise-consistency/diversity (cycle8 baseline) + expanded robust grid.
- Hypothesis: route-consistency + decoder-diversity helps external transfer under stress distributions.
- Immediate next extension after cycle8:
  - expand robustness grid around best cycle8 condition with higher `d_sae` (3072/4096) and longer epochs.
- Acceptance:
  - at least one condition improves both SAEBench and CE-Bench deltas vs routed cycle7 best.

3. Polysemantic pre-basis ablation (PolySAE-inspired) under matched budget.
- Hypothesis: polysemantic basis can improve reconstruction-quality/feature utility tradeoff at fixed parameter budget.
- Plan:
  - add a new architecture family runner analogous to routed/matryoshka scripts.
- Acceptance:
  - improvement on either SAEBench or CE-Bench with no internal-LCB regression below configured floor.

4. External-aware Pareto checkpointing with strict CI lower bounds as default release criterion.
- Hypothesis: CI-LCB-first selection avoids unstable one-seed wins and prevents false release positives.
- Required state:
  - already mostly implemented; continue enforcing in cycle8+.
- Acceptance:
  - selected candidates always include uncertainty metrics and pass/fail rationale in artifacts.

5. Known-circuit closure under trained-vs-random confidence bounds.
- Hypothesis: if known-circuit recovery does not pass CI-lower-bound criteria, external gains are likely fragile.
- Plan:
  - run closure as a hard prerequisite for any final claim set.
- Acceptance:
  - trained-vs-random lower-bound superiority on designated closure metrics.

## 4) Cycle-9 Execution Order

1. Wait for cycle7/cycle8 completion and extract final gate outcome.
2. Launch assignment-v4 supervised-proxy sweep (small pilot: 2 `d_sae` x 2 `k` x 3 seeds).
3. Launch robust-routed expansion sweep around best cycle8 setting.
4. Run selector (`group-by-condition`, `uncertainty-mode lcb`, strict external thresholds).
5. Run strict release gate and known-circuit closure.

## 5) Hard Claim Policy (unchanged)

- No external-improvement claim unless SAEBench + CE-Bench gates pass on configured threshold basis (point or LCB, explicitly declared).
- No “finished” claim unless strict release gate and known-circuit closure are both green.
- Every claim must link to exact artifact JSON/MD path.
