# Current Status and Study Guide

Updated: 2026-02-17

## 1) Original Goal

Determine whether SAE features are trustworthy and reproducible across seeds, and whether internal consistency improvements transfer to external interpretability benchmarks.

Core question:
- Can we jointly improve internal consistency and external benchmark performance under strict uncertainty-aware release gates?

## 2) What We Actually Tested

1. Internal validity
- Trained-vs-random consistency checks.
- Core ablations over `k` and `d_sae`.
- Assignment-aware consistency objectives (`v2`, `v3`, and supervised-proxy extension).

2. External validity
- SAEBench-aligned probing (`run_husai_saebench_custom_eval.py`).
- CE-Bench aligned evaluation (`run_husai_cebench_custom_eval.py`).
- Matched-budget architecture frontier sweeps (TopK/ReLU/BatchTopK/JumpReLU/Matryoshka/Routed).
- External scaling sweeps (token budget, hook layer, `d_sae`).

3. Reliability gates
- Random-model, transcoder, and OOD stress checks.
- Strict release policy with LCB gates for external metrics.

## 3) Current Scientific State

1. Strongly supported
- Internal consistency gains are real and reproducible.
- Engineering/reproducibility stack is robust (manifests, checks, queue scripts, tests).

2. Not yet solved
- No candidate has passed strict external joint gate (`SAEBench` + `CE-Bench`) with positive lower-bound deltas.
- Known-circuit closure remains partial.

3. Current live queue
- `cycle8`: completed through assignment `a3` with final `pass_all=false`.
- `cycle9`: active in routed stage1 (`run_20260217T151852Z`) with supervised-proxy assignment settings queued for stage2.
- `cycle10`: prepared as post-cycle9 external-recovery queue (`scripts/experiments/run_cycle10_external_recovery.sh`).

## 4) Most Important Files To Read (Order)

1. `START_HERE.md`
2. `docs/evidence/cycle8_cycle9_live_snapshot_20260217T152123Z/monitoring_summary.md`
3. `HIGH_IMPACT_FOLLOWUPS_REPORT.md`
4. `CYCLE10_EXTERNAL_RECOVERY_PLAN.md`
5. `RUNBOOK.md`
6. `EXPERIMENT_LOG.md`
7. `LIT_REVIEW.md`
8. `FINAL_PAPER.md`

## 5) Hypotheses and Why They Matter

1. H1: Internal consistency improvements should appear over random controls.
- Importance: validates non-trivial training signal.

2. H2: Architecture/objective choices should create a Pareto frontier, not a single optimum.
- Importance: enables targeted tradeoff optimization for external metrics.

3. H3: Internal gains should transfer externally.
- Importance: required for meaningful interpretability claims.

4. H4: A candidate should pass strict gated release criteria.
- Importance: prevents overclaiming and enforces reliability discipline.

## 6) Highest-Impact Open Questions

1. Can assignment objective variants raise SAEBench without collapsing CE-Bench?
2. Which routed robustness settings (noise/consistency/diversity) improve external deltas under matched compute?
3. Does grouped-LCB selection remain stable under stricter seed/group requirements?
4. Can we close known-circuit recovery with trained-vs-random confidence bounds?

## 7) Immediate Next Experiments (Ranked)

1. Finish cycle9 end-to-end, then rerun strict grouped-LCB selector and release gates.
2. Launch cycle10 routed + assignment external-recovery queue and compare against cycle8/cycle9 under identical gate policy.
3. If cycle10 still fails external gates, run assignment-v4 (relation-constrained + supervised proxy) with external-aware Pareto checkpointing.
4. Run matched-budget RouteSAE vs Matryoshka vs TopK head-to-head with identical seeds and CI-lower-bound decision policy.
5. Final known-circuit closure rerun with explicit acceptance thresholds.

## 8) Guardrails for Claims

1. Do not make external superiority claims unless strict gate `pass_all=true`.
2. Prefer CI/lower-bound comparisons over point estimates.
3. Keep trained-vs-random and stress controls in every release packet.
