# Literature and Competitive Landscape (Phase 3)

Date: 2026-02-12
Scope: SAE consistency, benchmark practice, and architecture/objective directions with direct relevance to this repository.

## 1) Primary Sources (Current and High-Relevance)

Core consistency and evaluation:
1. Paulo & Belrose (2025), *SAEs Trained on the Same Data Learn Different Features* - https://arxiv.org/abs/2501.16615
2. Song et al. (2025), *Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs* - https://arxiv.org/abs/2505.20254
3. Karvonen et al. (ICML 2025), *SAEBench* - https://proceedings.mlr.press/v267/karvonen25a.html
4. SAEBench repository - https://github.com/adamkarvonen/SAEBench
5. Peng et al. (2025), *CE-Bench* - https://arxiv.org/abs/2509.00691
6. CE-Bench repository - https://github.com/Yusen-Peng/CE-Bench

Strong architecture/objective baselines:
7. OpenAI (2024), *Scaling and Evaluating Sparse Autoencoders* - https://arxiv.org/abs/2406.04093
8. Rajamanoharan et al. (2024), *JumpReLU SAEs* - https://arxiv.org/abs/2407.14435
9. Bussmann et al. (2024), *BatchTopK SAEs* - https://arxiv.org/abs/2412.06410
10. Gao et al. (2025), *Matryoshka SAEs* - https://arxiv.org/abs/2503.17547
11. Li et al. (EMNLP 2025), *Route Sparse Autoencoders* - https://aclanthology.org/2025.emnlp-main.346/
12. Wan et al. (EMNLP 2025), *HierarchicalTopK SAEs* - https://aclanthology.org/2025.emnlp-main.515/

Additional frontier signals relevant to novel contributions:
13. Arditi et al. (2025, v2 2026), *Randomly Initialized Transformers Can Provide Strong Representations for Interpretable Tasks* - https://arxiv.org/abs/2501.18823
14. Makelov et al. (2025), *Transcoders Beat SAEs for Interpretability* - https://arxiv.org/abs/2501.17727
15. Marks et al. (2024), *Sparse Feature Circuits* - https://arxiv.org/abs/2409.14507
16. Templeton et al. (2024), *Scaling Monosemanticity* - https://transformer-circuits.pub/2024/scaling-monosemanticity/

## 2) What Current Benchmark-Credible Practice Requires

From SAEBench/CE-Bench and adjacent papers, benchmark-credible claims now require:
- Multi-seed reporting with confidence intervals/effect sizes.
- Trained-vs-control comparisons (not proxy metrics alone).
- Functional/causal metrics, not only feature-overlap metrics.
- Explicit command/config provenance and artifact manifests.
- External benchmark execution on the target method, not only nearby baselines.

Inference for this repo:
- Internal PWMCC progress is necessary but insufficient for SOTA-facing claims.
- Official benchmark runs are required, and they must evaluate HUSAI-produced checkpoints or adapters.

## 3) Gap Analysis vs This Repository

### Where we were behind
1. Official benchmark execution had been preflight-only.
2. Architecture frontier coverage lagged 2025 methods.
3. Objective-level consistency gains remained weak/unresolved.
4. Stress testing (OOD/sensitivity) was not a hard release gate.

### Where we are now
1. Official SAEBench command execution is completed via harness (`run_20260212T201204Z`).
2. Artifact-level reproducibility and consistency audit tooling is in place.
3. Adaptive L0 calibration has strong empirical support in this repo.

### Remaining critical gap
- Official execution used a public SAEBench SAE target (`pythia-70m-deduped-res-sm`), not a full HUSAI-checkpoint adapter path; CE-Bench is still not executed in this environment.

## 4) Evidence from the Completed Official SAEBench Run

Run artifact:
- `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`

Status:
- `commands.json`: SAEBench command attempted and succeeded (`returncode=0`).
- `preflight.json`: SAEBench module available `True`; CE-Bench module available `False`.

Aggregate readout from produced SAE-probes outputs (`/tmp/sae_bench_probe_results/`):
- Matched datasets: `113`
- Mean delta (best SAE over ks {1,2,5} minus baseline logreg):
  - `test_f1`: `-0.0952`
  - `test_acc`: `-0.0513`
  - `test_auc`: `-0.0651`
- Win/loss on `test_auc`: `21` wins, `88` losses, `4` ties.

Interpretation:
- This run validates harness functionality and gives a real external performance signal.
- It does not support SOTA claims; if anything, it highlights a performance gap for this benchmark setup.

## 5) Immediate Replication Protocol to Import

For each major result update:
1. Log command, commit, config hash, and artifact paths in `EXPERIMENT_LOG.md`.
2. Run `python scripts/analysis/verify_experiment_consistency.py`.
3. If benchmark-facing, run official harness command(s) and archive `commands.json`, `preflight.json`, `summary.md`, and logs.
4. Report both gains and regressions (no selective omission).

## 6) Practical Conclusion

The repo is now materially stronger in engineering rigor and reproducibility than at the start of this cycle. The scientific bottleneck has shifted from tooling to method quality on external tasks. Highest leverage now is architecture/objective innovation evaluated through official benchmark pathways with strict controls and uncertainty reporting.
