# Evidence Status Ledger

Updated: 2026-03-05

This file is the source of truth for what is *locally verifiable* in this git checkout versus what is *reported from remote compute storage*.

## Evidence Tiers

1. `Tier 1 (Local, reproducible in repo)`
- Artifact files exist under this repository and can be inspected directly.
- Claims from this tier are safe to use without remote access.

2. `Tier 2 (Remote-reported, not fully mirrored locally)`
- Paths are documented in canonical docs but not present in this checkout.
- Claims from this tier must be labeled as remote-reported unless independently mirrored.

## Tier 1: Locally Verified Selector/Gate Snapshot

Primary local selector artifact:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/selector/selected_candidate.json`

Primary local strict gate artifact:
- `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.json`

Locally verified selected candidate and metrics:
- condition: `topk`
- representative seed: `123`
- checkpoint: `results/experiments/phase4b_architecture_frontier_external_multiseed/run_20260214T202538Z/checkpoints/topk_seed123/sae_final.pt`
- `saebench_delta` / LCB: `-0.04478959689939781`
- `cebench_interp_delta_vs_baseline` / LCB: `-40.467037470119465`
- strict gate: `pass_all=false`

## Tier 2: Remote-Reported Final Package (Not Fully Mirrored Locally)

Documented remote package path:
- `results/final_packages/cycle10_final_20260218T141310Z`
- `results/final_packages/cycle10_final_20260218T141310Z/meta/FINAL_INDEX.md`

Remote-reported canonical docs currently cite:
- selected condition: `relu`, seed `42`
- `saebench_delta = -0.029153650997086358`
- `cebench_interp_delta_vs_baseline = -43.71286609575971`
- strict gate: `pass_all=false`

## Reconciliation Status

Current state:
- Local artifacts and remote-reported package agree on the final *decision* (`pass_all=false`).
- Candidate identity and metric values differ between the local selector snapshot and remote-reported final package references.

Interpretation:
- The repository supports a robust negative conclusion (external gate not satisfied).
- Exact final-cycle candidate ranking/metrics require remote package mirroring for full local verification.

## Claim Policy Going Forward

When writing summaries:
- Prefer Tier 1 claims by default.
- Use Tier 2 claims only with explicit "remote-reported" labeling.
- Never mix Tier 1 candidate identity with Tier 2 metrics without explicit caveats.
