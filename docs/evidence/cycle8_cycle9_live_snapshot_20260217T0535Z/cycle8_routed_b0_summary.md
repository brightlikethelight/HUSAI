# Routed TopK Frontier External

- Run ID: `run_20260217T051230Z`
- Seeds: `[42, 123, 456, 789]`
- d_sae / k: `1536` / `40`
- Experts: `6`
- Route balance / entropy coef: `0.02` / `0.02`
- Route top-k mode: `expert_topk`
- Rows used: `144042`

| metric | mean | std |
|---|---:|---:|
| train_ev | 0.3627955893902479 | 0.019510851396628337 |
| train_l0 | 40.0 | 0.0 |
| routing_entropy | 1.7917593121528625 | 1.5389853104779787e-07 |
| routing_balance_l2 | 1.1197328973366893e-05 | 1.0559396376842724e-05 |
| route_consistency_loss | 0.0 | 0.0 |
| decoder_diversity_loss | 0.0 | 0.0 |
| saebench_best_minus_llm_auc | -0.06616931831020209 | 0.010360862485332073 |
| cebench_interpretability_max | 11.533530024290085 | 1.9680870560441164 |
| cebench_interp_delta_vs_baseline | -36.418081561326986 | 1.9680870560441173 |
