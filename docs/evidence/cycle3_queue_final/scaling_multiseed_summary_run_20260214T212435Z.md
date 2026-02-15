# External Metric Scaling Study

- Run ID: `run_20260214T212435Z`
- token budgets: `[10000, 30000]`
- hook layers: `[0, 1]`
- d_sae values: `[1024, 2048]`
- seeds: `[42, 123, 456]`

## Aggregates by Token Budget

| token_budget | SAEBench best-LLM AUC mean | CE-Bench interpretability max mean | CE-Bench interp delta vs baseline mean |
|---:|---:|---:|---:|
| 10000 | -0.08629052824435675 | 7.862272895475228 | -40.08933869014184 |
| 30000 | -0.08513235397555448 | 8.026932807564735 | -39.92467877805234 |
