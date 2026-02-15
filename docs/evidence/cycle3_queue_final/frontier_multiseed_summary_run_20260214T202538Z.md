# Architecture Frontier External

- Run ID: `run_20260214T202538Z`
- Architectures: `['topk', 'relu', 'batchtopk', 'jumprelu']`
- Seeds: `[42, 123, 456, 789, 1011]`
- d_sae / k: `1024` / `32`
- Rows used: `144042`

| architecture | train EV mean | SAEBench best-LLM AUC mean | CE-Bench interpretability max mean | CE-Bench interp delta vs baseline mean |
|---|---:|---:|---:|---:|
| topk | 0.7450903693098755 | -0.04059280026569467 | 7.726768206596374 | -40.22484337902069 |
| relu | 0.9955460129388298 | -0.024691224025891967 | 4.257686385631561 | -43.69392519998551 |
| batchtopk | 0.6775429210982081 | -0.043356197451623536 | 6.537639029979705 | -41.413972555637365 |
| jumprelu | 0.9959328156159207 | -0.03057683519457275 | 4.379002495765685 | -43.57260908985138 |
