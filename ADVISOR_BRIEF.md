# Advisor Brief: HUSAI Final State

Date: 2026-02-18

## Core Question

Can SAEs be made both internally reliable and externally benchmark-competitive under strict uncertainty-aware release gates?

## Final Outcome

- Internal consistency: improved and reproducible.
- Stress controls: pass.
- External benchmarks: fail strict positivity thresholds.
- Final release gate: `pass_all=false`.

## Selected Final Candidate

- condition: `relu`
- seed: `42`
- checkpoint: `results/experiments/phase4b_architecture_frontier_external_multiseed/run_20260214T202538Z/checkpoints/relu_seed42/sae_final.pt`

## Why Not Released

External metrics remained negative for the selected candidate:

- `saebench_delta = -0.029153650997086358`
- `cebench_interp_delta_vs_baseline = -43.71286609575971`

while non-external gates passed:

- `ood=true`, `transcoder=true`, `random_model=true`

## Canonical Evidence

- `results/final_packages/cycle10_final_20260218T141310Z/meta/FINAL_INDEX.md` (remote RunPod storage)
- `results/experiments/release_stress_gates/run_20260218T070856Z/release_policy.json`
- `results/experiments/release_candidate_selection_cycle10_recovery/run_20260218T070102Z/selected_candidate.json`

## Recommended Next Program

1. External-positive objective sweep with explicit dual-target constraints.
2. Larger grouped-LCB seed counts for selector robustness.
3. Assignment objective v4/v5 with external-aware Pareto checkpointing.
4. New architecture family under matched compute (RouteSAE variants first).
5. Known-circuit closure with trained-vs-random confidence bounds.
