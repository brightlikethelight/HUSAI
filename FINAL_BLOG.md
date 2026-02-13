# We Executed the Highest-Impact SAE Follow-Ups End-to-End. What Survived?

HUSAI asks a direct question:

If we train sparse autoencoders (SAEs) on the same activations with different seeds, do we recover stable, useful features?

This cycle focused on the highest-leverage steps:
1. reliability hardening (CI, smoke, portability),
2. remote reproduction and ablations,
3. official external benchmark execution,
4. direct HUSAI-checkpoint benchmark evaluation with multi-seed uncertainty.

## Reliability First, Then Claims

Before new claims, we fixed infrastructure and reproducibility blockers:
- CI fail-fast smoke + quality gates.
- path portability in core experiment runners.
- benchmark harness log streaming for long runs.
- consistency-audit tooling to check narrative claims against artifact JSONs.

Result: we can now run the core stack cleanly on remote GPU and preserve manifest-level provenance.

## Remote Reproduction Summary (RunPod B200)

### Phase 4a (trained vs random)
- trained mean PWMCC: `0.300059`
- random mean PWMCC: `0.298829`
- delta: `+0.001230`
- one-sided p-value: `8.629e-03`

Interpretation: a real but small consistency signal in this rerun.

### Phase 4c (core ablations)
Best `k`-sweep condition:
- `k=8`, `d_sae=128`, delta `+0.009773`

Best `d_sae`-sweep condition:
- `d_sae=64`, `k=32`, delta `+0.119986`

Interpretation: geometry choices can dominate the consistency outcome.

## Official SAEBench: Completed and Reproducible

Public SAE target run:
- `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`

Aggregate over 113 matched datasets (best SAE over `k in {1,2,5}` minus logreg baseline):
- mean `test_f1` delta: `-0.0952`
- mean `test_acc` delta: `-0.0513`
- mean `test_auc` delta: `-0.0651`

Interpretation: external evidence in this setup is negative relative to baseline.

## Direct HUSAI Checkpoint SAEBench Path: Now Complete

We ran direct custom-checkpoint evaluation for three seeds:
- seed 42: `run_20260213T024329Z`
- seed 123: `run_20260213T031247Z`
- seed 456: `run_20260213T032116Z`

Tracked summary:
- `docs/evidence/phase4e_husai_custom_multiseed/summary.json`

Multi-seed readout:
- best AUC mean ± std: `0.622601 ± 0.000615`
- best AUC 95% CI: `[0.621905, 0.623297]`
- delta AUC vs LLM baseline mean ± std: `-0.051801 ± 0.000615`
- delta AUC 95% CI: `[-0.052496, -0.051105]`

Interpretation:
- the custom path is stable and reproducible,
- but consistently below baseline in current form,
- so no SOTA-style claim is justified.

## What We Can Claim Now

Supported:
- reproducibility and engineering discipline improved substantially,
- internal consistency effects are real but regime-sensitive,
- external benchmark infrastructure is operational,
- direct HUSAI external evaluation is now real (not TODO).

Not supported:
- external superiority claims,
- SOTA claims.

## Next 5 Highest-Leverage Steps (Ranked)

1. Execute CE-Bench for HUSAI + baseline targets with full manifests.
2. Run a matched-budget architecture frontier sweep (TopK, JumpReLU, BatchTopK, Matryoshka, RouteSAE, HierarchicalTopK).
3. Run external-metric-focused scaling (`token budget`, `hook layer`, `d_sae`).
4. Implement assignment-aware consistency objective v2 with CI-based acceptance criteria.
5. Add mandatory stress gates (transcoder control, random-model control, OOD checks) before narrative updates.

## Bottom Line

The highest-value outcome this cycle is clarity: we converted benchmark discussion from assumptions into reproducible evidence, and that evidence says current external performance is not yet competitive. That is a stronger research position than optimistic ambiguity, and it points directly to the next experiments that matter.
