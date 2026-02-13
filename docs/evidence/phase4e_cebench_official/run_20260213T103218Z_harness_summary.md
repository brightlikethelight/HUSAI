# Official External Benchmark Harness

- Run ID: `run_20260213T103218Z`
- Git commit: `59876a5d729d91ccb18571ca86fe94ef72b4795b`
- Config hash: `70c5d1b4d2ce1e93bab4592d0ec28e09da742e819e0e020090654d6abe5467b5`

## Preflight

- SAEBench module available: `True`
- SAEBench repo path: `None`
- CE-Bench module available: `False`
- CE-Bench repo path: `/workspace/CE-Bench`
- Local SAE checkpoints indexed: `8`
- HUSAI custom checkpoint provided: `False`

## Command Status

| name | attempted | success | returncode | note |
|---|---:|---:|---:|---|
| cebench | True | True | 0 | completed |

## How to Execute

Run with explicit official commands (examples):
```bash
python scripts/experiments/run_official_external_benchmarks.py \
  --saebench-repo /path/to/SAEBench \
  --cebench-repo /path/to/CE-Bench \
  --saebench-command "<official SAEBench command>" \
  --cebench-command "<official CE-Bench command>" \
  --execute
```

HUSAI custom checkpoint eval example:
```bash
python scripts/experiments/run_official_external_benchmarks.py \
  --husai-saebench-checkpoint results/saes/husai_pythia70m_topk_seed42/sae_final.pt \
  --execute
```

CE-Bench compat execution example:
```bash
python scripts/experiments/run_official_external_benchmarks.py --cebench-repo /path/to/CE-Bench --cebench-use-compat-runner --cebench-sae-regex-pattern <pattern> --cebench-sae-block-pattern <block> --execute
```

SAEBench reference command pattern from official docs:
```bash
python -m sae_bench.evals.sparse_probing.main \
  --sae_regex_pattern "<pattern>" \
  --sae_block_pattern "<block>" \
  --model_name <model>
```
