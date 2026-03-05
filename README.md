# HUSAI: Reliability-First SAE Research

HUSAI studies whether sparse autoencoder (SAE) features are trustworthy under strict release criteria: internal reproducibility, stress robustness, and external benchmark competitiveness.

## Current Bottom Line

- Internal consistency signal: positive.
- Stress controls (`random_model`, `transcoder`, `OOD`): passing in documented runs.
- External transfer (`SAEBench`, `CE-Bench`): still below strict thresholds.
- Strict release outcome: `pass_all=false`.

## Evidence Integrity Note

The repository currently has two evidence tiers:
- Local verified artifacts in `docs/evidence/...`
- Remote-reported final package paths under `results/final_packages/...` (not fully mirrored in this checkout)

Use `EVIDENCE_STATUS.md` before citing exact candidate identity or final metric values.

## Canonical Entry Points

1. `START_HERE.md`
2. `CANONICAL_DOCS.md`
3. `EVIDENCE_STATUS.md`
4. `EXECUTIVE_SUMMARY.md`
5. `RUNBOOK.md`
6. `EXPERIMENT_LOG.md`

## Core Research Question

Can we improve SAE internal consistency and still meet external benchmark gates (SAEBench + CE-Bench) under uncertainty-aware release policy?

## Quick Validation

```bash
pytest tests -q
make smoke
```

## Core Scripts

Internal baselines and ablations:
- `scripts/experiments/run_phase4a_reproduction.py`
- `scripts/experiments/run_core_ablations.py`
- `scripts/experiments/run_assignment_consistency_v2.py`
- `scripts/experiments/run_assignment_consistency_v3.py`

External benchmark program:
- `scripts/experiments/run_husai_saebench_custom_eval.py`
- `scripts/experiments/run_husai_cebench_custom_eval.py`
- `scripts/experiments/run_architecture_frontier_external.py`
- `scripts/experiments/run_matryoshka_frontier_external.py`
- `scripts/experiments/run_routed_frontier_external.py`
- `scripts/experiments/run_external_metric_scaling_study.py`

Strict gating:
- `scripts/experiments/select_release_candidate.py`
- `scripts/experiments/run_stress_gated_release_policy.py`

## Documentation Artifacts

- Final summary: `EXECUTIVE_SUMMARY.md`
- Final paper-style writeup: `FINAL_PAPER.md`
- Final blog-style writeup: `FINAL_BLOG.md`
- Literature and SOTA review: `LIT_REVIEW.md`
- Experiment roadmap: `EXPERIMENT_PLAN.md`
- Presentation package: `docs/05-Presentation/cycle10_readout/`

## License

MIT (`LICENSE`).
