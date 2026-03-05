# Phase 1 Repo Understanding

Date: 2026-03-05

## Goal and Claims

Repo goal:
- Evaluate SAE trustworthiness using strict release gates over internal consistency, stress robustness, and external benchmark performance.

Explicit claim:
- Current strict outcome is negative (`pass_all=false`).

Implicit claim:
- Internal improvements should not be treated as external success without benchmark gates.

## Minimal Reproducible Demo

```bash
pytest tests -q
python scripts/experiments/run_phase4a_reproduction.py
python scripts/experiments/run_core_ablations.py
```

## How It Works

data -> activation cache -> SAE training variants -> internal/external eval -> selector -> strict gate -> artifact manifests

## Critical Path Files

1. `scripts/experiments/run_phase4a_reproduction.py`
2. `scripts/experiments/run_core_ablations.py`
3. `scripts/experiments/run_assignment_consistency_v3.py`
4. `scripts/experiments/run_architecture_frontier_external.py`
5. `scripts/experiments/run_husai_saebench_custom_eval.py`
6. `scripts/experiments/run_husai_cebench_custom_eval.py`
7. `scripts/experiments/select_release_candidate.py`
8. `scripts/experiments/run_stress_gated_release_policy.py`
9. `src/training/train_sae.py`
10. `src/analysis/feature_matching.py`

## Evidence Warning

Final-cycle candidate identity/metrics differ across local and remote-reported sources; use `EVIDENCE_STATUS.md` for claim-tier handling.
