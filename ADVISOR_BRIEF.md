# Advisor Brief: HUSAI After Cycle 3 (B200)

Date: 2026-02-15

## 1) Problem and Goal

HUSAI is testing a core mechanistic-interpretability question: can SAE features be made stable and trustworthy across random seeds, and do those gains transfer to external interpretability benchmarks?

Current scientific bottom line:
- Internal consistency progress: real.
- External competitiveness: not yet achieved.
- Reliability/engineering hygiene: strong.
- Open problem: improve internal consistency and external benchmark deltas simultaneously under strict release gates.

## 2) Strongest Evidence (Artifact-Backed)

Primary synthesis:
- `docs/evidence/cycle3_queue_final/cycle3_final_synthesis_run_20260214T210734Z.md`

Key metrics:
- Frontier multiseed (`4 architectures x 5 seeds`, 20 records):
  - best SAEBench delta among tested: `relu = -0.024691`
  - best CE-Bench mean interpretability among tested: `topk = 7.726768`
  - CE-Bench deltas vs matched baseline remain strongly negative (about `-40` to `-44`).
- Scaling multiseed (24 conditions):
  - layer 1 and larger `d_sae` improve CE-Bench but worsen SAEBench deltas.
- Stress gates:
  - `random_model = true`
  - `transcoder = false` (`transcoder_delta = -0.002227966984113039`)
  - `ood = true` (`ood_drop = 0.01445406161520213`)
  - `external = false` (`external_delta = -0.017257680751151527`)
  - `pass_all = false`

Gate artifact:
- `docs/evidence/cycle3_queue_final/release_policy_run_20260214T225029Z.json`

## 3) What Is Scientifically Clear

1. Internal consistency is improvable (assignment-aware v2 and ablations support this).
2. External validity is the bottleneck.
3. SAEBench and CE-Bench optimize conflicting regions in current search space.
4. Claim discipline is now good: strict gates block unsupported claims.

## 4) Highest-Risk Technical Gaps

1. External gate is candidate-sensitive and currently fed from a single external summary path.
- Risk: release decisions depend on one candidate summary rather than an explicit cross-run selection policy.

2. External gate metric is underspecified across benchmark families.
- `run_stress_gated_release_policy.py` accepts one scalar `external_delta`, which can come from SAEBench or CE-Bench summaries depending on file shape.
- Risk: apples-to-oranges pass criteria.

3. Queue candidate selection is SAEBench-first.
- `run_b200_high_impact_queue.sh` selects best frontier condition by SAEBench only.
- Risk: misses CE-Bench and multi-objective best candidates.

4. Proposal closure gap remains on known-circuit recovery.
- The original proposal explicitly included known ground-truth/circuit recovery; still incomplete.

## 5) Why This Is Publishable Already

As a reliability-first negative/nuanced result:
- It establishes robust internal gains with external non-transfer.
- It quantifies a repeatable cross-benchmark tradeoff frontier.
- It demonstrates strong claim-gating and reproducibility discipline.

## 6) Decision for Next Cycle

Do not chase single-metric wins.
Run a multi-objective program that jointly optimizes:
- internal consistency,
- SAEBench external delta,
- CE-Bench external delta,
- stress-gate robustness.

The detailed execution plan is in:
- `CYCLE4_HIGH_IMPACT_EXECUTION_PLAN.md`

## 7) Current Literature Anchors (Primary Sources)

- SAEBench (ICML 2025): https://proceedings.mlr.press/v267/karvonen25a.html
- CE-Bench (arXiv 2025): https://arxiv.org/abs/2509.00691
- Transcoders beat SAEs (arXiv 2025): https://arxiv.org/abs/2501.18823
- RouteSAEs (arXiv 2025): https://arxiv.org/abs/2505.00570
- Hierarchical Top-K SAEs (arXiv 2025): https://arxiv.org/abs/2506.13617
- Matryoshka SAE (arXiv 2025): https://arxiv.org/abs/2503.17637
