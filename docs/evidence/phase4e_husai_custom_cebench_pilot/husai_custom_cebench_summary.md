# HUSAI Custom CE-Bench Summary

- Checkpoint: `results/saes/husai_pythia70m_topk_seed42/sae_final.pt`
- Checkpoint SHA256: `7ef3e6c94f5c8dcd36f9a0aeadec37b6bdf0fc36d296e4a0ed786631c1b4eef2`
- SAE release id: `husai_pythia70m_topk_seed42`
- Model: `pythia-70m-deduped`
- Hook: `blocks.0.hook_resid_pre` (layer `0`)
- Device: `cuda`
- Dataset rows used: `500` / `5000`
- LLM dtype / batch: `float32` / `512`

## Custom Metrics

- contrastive_score_mean.max: `10.735687292099`
- independent_score_mean.max: `11.450095337867737`
- interpretability_score_mean.max: `10.733953895568847`

## Delta vs Matched Baseline

- baseline summary: `results/experiments/phase4e_external_benchmark_official/run_20260213T103218Z/cebench/cebench_metrics_summary.json`
- delta contrastive_score_mean.max (custom - baseline): `-38.37853209877014`
- delta independent_score_mean.max (custom - baseline): `-42.24810237560272`
- delta interpretability_score_mean.max (custom - baseline): `-36.74728686256409`

## Artifacts

- CE-Bench metrics JSON: `results/experiments/phase4e_external_benchmark_official/husai_custom_cebench_pilot_20260213/cebench_metrics_summary.json`
- CE-Bench metrics markdown: `results/experiments/phase4e_external_benchmark_official/husai_custom_cebench_pilot_20260213/cebench_metrics_summary.md`
- Adapter summary JSON: `results/experiments/phase4e_external_benchmark_official/husai_custom_cebench_pilot_20260213/husai_custom_cebench_summary.json`
