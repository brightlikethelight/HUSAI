# HUSAI: Reliability-First SAE Research

HUSAI studies whether sparse autoencoder (SAE) features are trustworthy under strict release criteria: internal reproducibility, stress robustness, and external benchmark competitiveness.

## Key Finding

Trained SAE features are **indistinguishable from random baseline** (PWMCC = 0.309 vs 0.300 for untrained SAEs). SAEs reconstruct well but learn arbitrary, non-reproducible feature decompositions across seeds. See `paper/sae_stability_paper.md` for the full paper.

## Current Bottom Line

- Internal consistency signal: positive.
- Stress controls (`random_model`, `transcoder`, `OOD`): passing.
- External transfer (`SAEBench`, `CE-Bench`): below strict thresholds.
- Strict release outcome: `pass_all=false`.

Use `EVIDENCE_STATUS.md` before citing exact metrics (local vs remote evidence tiers).

## Start Here

See `START_HERE.md` for the full reading order and orientation.

## Quick Validation

```bash
conda env create -f environment.yml && conda activate husai
pip install -r requirements-dev.txt
pytest tests -q
make smoke
```

## Core Scripts

Internal baselines and ablations:
- `scripts/experiments/run_phase4a_reproduction.py`
- `scripts/experiments/run_core_ablations.py`
- `scripts/experiments/run_assignment_consistency_v3.py`

Follow-up experiments (Section 4.11 of the paper):
- `scripts/experiments/run_all_followup_experiments.sh` -- run all follow-ups
- `scripts/experiments/exp_1layer_ground_truth.py` -- 1-layer vs 2-layer comparison
- `scripts/experiments/exp_subspace_stability.py` -- subspace vs feature stability
- `scripts/experiments/exp_effective_rank_predictor.py` -- universal stability predictor
- `scripts/experiments/exp_contrastive_stability.py` -- contrastive alignment loss
- `scripts/experiments/exp_intervention_stability.py` -- steering consistency across seeds
- `scripts/experiments/exp_dictionary_pinning.py` -- warm-start with frozen decoder columns
- `scripts/experiments/exp_pythia70m_stability.py` -- scale to Pythia-70M (GPU)

External benchmark program:
- `scripts/experiments/run_husai_saebench_custom_eval.py`
- `scripts/experiments/run_husai_cebench_custom_eval.py`
- `scripts/experiments/run_architecture_frontier_external.py`

Strict gating:
- `scripts/experiments/select_release_candidate.py`
- `scripts/experiments/run_stress_gated_release_policy.py`

## License

MIT (`LICENSE`).
