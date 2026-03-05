# Literature and Competitive Landscape

Date: 2026-03-05

## Scope

This review focuses on SAE reliability, external benchmark practice, and architecture/objective directions relevant to HUSAI.

## Most Relevant Papers (Primary Sources)

1. SAEs trained on same data can learn different features (seed instability): https://arxiv.org/abs/2501.16615
2. SAEBench benchmark paper: https://arxiv.org/abs/2503.09532
3. SAEBench repository and eval harness: https://github.com/adamkarvonen/SAEBench
4. CE-Bench paper (ACL BlackboxNLP 2025): https://aclanthology.org/2025.blackboxnlp-1.1/
5. CE-Bench codebase: https://github.com/Yusen-Peng/CE-Bench
6. Route Sparse Autoencoders (RouteSAE): https://arxiv.org/abs/2503.08200
7. Transcoders vs SAEs: https://arxiv.org/abs/2501.18823
8. JumpReLU SAEs: https://arxiv.org/abs/2407.14435
9. BatchTopK SAEs: https://arxiv.org/abs/2412.06410
10. Nested/Matryoshka SAEs: https://arxiv.org/abs/2503.17547
11. OpenAI sparse autoencoder codebase: https://github.com/openai/sparse_autoencoder
12. End-to-end SAE training reference implementation: https://github.com/ApolloResearch/e2e_sae

## Strong Baselines to Track (5-10)

1. ReLU SAE baseline (internal control).
2. TopK SAE baseline.
3. BatchTopK SAE baseline.
4. JumpReLU variant.
5. RouteSAE variant.
6. Nested/Matryoshka variant.
7. Transcoder baseline (as external comparator/control).
8. Random-feature/random-model control.

In HUSAI, these map to existing script families in `scripts/experiments/` plus stress-policy controls.

## Standard Evaluation Suites and Protocol Expectations

1. SAEBench.
- Use benchmark-provided task slices consistently.
- Report seed-aware statistics (mean/std/CI), not single-run values.

2. CE-Bench.
- Match row budgets and baseline protocol exactly before delta claims.
- Record interpretability/contrastive/independent score summaries with provenance.

3. Stress controls.
- Random-model, OOD, and alternative-method checks (e.g., transcoders) should be release gates, not appendix-only diagnostics.

## Gap Analysis vs This Repository

### Where HUSAI Is Behind

1. Final-cycle candidate identity/metric reconciliation is incomplete locally (local vs remote tier mismatch).
2. External benchmark deltas remain negative under strict thresholds.
3. Some frontier conclusions are sensitive to seed count and grouped uncertainty handling.
4. Fully official benchmark runs in clean external environments are still sparse relative to custom-adapter runs.

### Where HUSAI Can Win

1. Strong claim discipline with explicit gates and evidence tiers.
2. Reproducibility-focused orchestration and manifest logging.
3. Practical benchmark integration across SAEBench and CE-Bench in one pipeline.
4. Fast iteration on candidate selection policy and stress-aware gating.

## Replication Details to Keep Fixed

For all benchmark-facing runs:
- identical hook layer/hook name
- identical row budgets
- explicit seed list and grouped aggregation mode
- persisted `config_hash`, `git_commit`, command, and artifact paths

## Recommended 2026 Experiment Focus

1. Seed-complete grouped-LCB external reruns for top candidate families.
2. Matched-protocol CE/SAE benchmark recalibration before any cross-run comparisons.
3. Objective-level external coupling (not just post-hoc selection) with stress constraints.
4. One official benchmark slice in a clean external environment for stronger comparability.
