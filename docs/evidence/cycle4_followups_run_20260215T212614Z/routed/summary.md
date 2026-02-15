# Routed TopK Frontier External

- Run ID: `run_20260215T213621Z`
- Seeds: `[42, 123, 456]`
- d_sae / k: `1024` / `32`
- Experts: `8`
- Route balance / entropy coef: `0.2` / `0.01`
- Rows used: `144042`

| metric | mean | std |
|---|---:|---:|
| train_ev | 0.225768804132668 | 0.03702007120677993 |
| train_l0 | 4.193015495936076 | 0.3930144362721389 |
| routing_entropy | 2.079439163208008 | 1.2615925364802316e-06 |
| routing_balance_l2 | 3.261359658305688e-05 | 1.2907503409047941e-05 |
| saebench_best_minus_llm_auc | -0.06930492591377861 | 0.0038189153287062263 |
| cebench_interpretability_max | 8.125359812577566 | 1.5391865375905698 |
| cebench_interp_delta_vs_baseline | -39.8262517730395 | 1.5391865375905684 |
