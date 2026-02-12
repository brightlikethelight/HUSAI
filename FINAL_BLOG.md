# We Ran the Highest-Impact SAE Follow-Ups End-to-End. Here Is What Actually Held Up.

HUSAI studies a central interpretability question:

If we train sparse autoencoders (SAEs) on the same activations with different seeds, do we recover the same features?

This cycle focused on the highest-leverage operational and research steps:
1. reliability hardening (CI + smoke + portability),
2. full reproduction and ablations on remote GPU,
3. official external benchmark execution,
4. evidence-locked reporting.

## What We Changed First (Before New Claims)

Engineering fixes that were required for trustworthy science:
- Added fail-fast CI smoke and quality gates.
- Fixed `.gitignore` bug that accidentally excluded `src/data/` from version control.
- Resolved NumPy/TransformerLens compatibility constraints.
- Removed absolute-path assumptions in core experiment runners.
- Upgraded benchmark harness logging to stream subprocess logs to disk.

Without these fixes, remote reproduction either failed outright or risked non-portable results.

## Remote Reproduction on RunPod B200

### Phase 4a: trained vs random (5 seeds)
- trained mean PWMCC: `0.300059`
- random mean PWMCC: `0.298829`
- delta: `+0.001230`
- one-sided p-value: `8.629e-03`

Interpretation:
- Signal is statistically detectable but small in absolute magnitude for this rerun.

### Phase 4c: core ablations
Best `k` sweep condition:
- `k=8`, `d_sae=128`, delta `+0.009773`

Best `d_sae` sweep condition:
- `d_sae=64`, `k=32`, delta `+0.119986`

Interpretation:
- Geometry choices can dominate consistency gains.
- Hyperparameter regime matters more than single-metric optimism.

## Adaptive L0 Still Looks Like the Strongest Internal Lever

From the follow-up runs:
- `k=4` vs matched `k=32` control improved trained PWMCC by `+0.05701`.
- 95% bootstrap CI stayed strongly positive.

Interpretation:
- If your immediate goal is more reproducible dictionaries in this repo, L0 calibration remains the highest-value knob.

## Official SAEBench Execution: Completed, and Informative

We executed the official harness run:
- `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`
- command status: success (`returncode=0`)

The run produced SAE-probes outputs over 113 matched datasets. Aggregated against baseline logreg probes:
- mean delta `test_f1`: `-0.0952`
- mean delta `test_acc`: `-0.0513`
- mean delta `test_auc`: `-0.0651`
- AUC wins/losses/ties: `21 / 88 / 4`

Interpretation:
- We now have real external benchmark evidence, not just preflight scaffolding.
- In this setup, the SAE target used for official probing underperforms baseline probes.
- This is exactly why benchmark-first discipline matters.

## What We Can Claim Now (and What We Cannot)

Supported:
- The repo is substantially more reproducible and portable.
- Internal consistency effects are real but regime-sensitive.
- Adaptive low-L0 remains a strong internal strategy.
- Official external benchmarking is operational.

Not supported:
- Any SOTA-style claim.
- Any claim that current method dominates external benchmark baselines.

Still pending:
- CE-Bench execution in this environment.
- Full HUSAI-checkpoint-to-benchmark adapter path.

## Next 5 Highest-Leverage Steps (Ranked)

1. Benchmark HUSAI-produced checkpoints directly on SAEBench and CE-Bench.
2. Run a matched-budget architecture frontier sweep (TopK, JumpReLU, BatchTopK, Matryoshka, HierarchicalTopK, RouteSAE).
3. Implement consistency-objective v2 (assignment-aware or joint multi-seed) with CI-based acceptance criteria.
4. Add a transcoder control arm under equal compute.
5. Add random-model and OOD stress-test gates before any claim update.

## Bottom Line

The biggest win this cycle was not a flashy metric jump; it was converting the repo into a system where claims are tied to reproducible artifacts and external checks. That surfaced a hard but useful truth: our current external benchmark story is not yet strong, which gives us a clear, high-impact roadmap for what to improve next.
