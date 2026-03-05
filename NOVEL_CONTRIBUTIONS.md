# Novel Contributions and Opportunity Map

Updated: 2026-03-05

## Demonstrated Contributions

1. Reliability-first release discipline.
- Release claims are blocked unless internal, stress, and external gates all pass.

2. Evidence-tier transparency.
- `EVIDENCE_STATUS.md` formalizes local-verified versus remote-reported claim boundaries.

3. Reproducible benchmark adapters.
- SAEBench and CE-Bench custom-eval paths are implemented with manifest outputs.

4. Multi-family frontier tooling.
- ReLU/TopK/BatchTopK/Matryoshka/Routed candidates are evaluated under shared orchestration.

5. Negative-result integrity.
- The codebase keeps `pass_all=false` visible instead of inflating partial progress.

## Current Scientific State

- Internal signal: positive.
- Stress controls: passing in documented runs.
- External strict criteria: not met.
- Overall decision: `pass_all=false`.

## Highest-Impact Novel Ideas (Ranked)

1. External-aware training objective with stress constraints.
2. Grouped-LCB selector that includes stress calibration terms.
3. Cross-layer transfer and portability benchmark for SAE features.
4. Matched-protocol hybrid frontier (routed + nested + TopK) under compute parity.
5. Evidence-tier reporting standard as a reproducibility contribution.

## Primary References

- Seed instability in SAEs: https://arxiv.org/abs/2501.16615
- SAEBench: https://arxiv.org/abs/2503.09532
- CE-Bench: https://aclanthology.org/2025.blackboxnlp-1.1/
- RouteSAE: https://arxiv.org/abs/2503.08200
- Transcoders: https://arxiv.org/abs/2501.18823
- JumpReLU SAEs: https://arxiv.org/abs/2407.14435
- BatchTopK SAEs: https://arxiv.org/abs/2412.06410
- Nested/Matryoshka SAEs: https://arxiv.org/abs/2503.17547
