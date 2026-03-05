# Health Audit

Date: 2026-03-05

Scope: end-to-end research workflow, correctness, maintainability, reproducibility, performance, and claim hygiene.

## Summary

The repository is functionally strong on workflow breadth and experiment tooling, but had several high-risk correctness gaps and evidence-integrity inconsistencies. This pass fixed the highest-impact code reliability defects and introduced explicit evidence-tier policy.

## Findings (P0/P1/P2)

| ID | Severity | Category | Finding | Evidence | Recommendation | Effort | Status |
|---|---|---|---|---|---|---|---|
| A1 | P0 | Research hygiene | Canonical docs mixed local and remote evidence without explicit tiering. | `README.md`, `START_HERE.md`, `EXECUTIVE_SUMMARY.md`, local selector artifact in `docs/evidence/cycle4_followups_run_20260215T190004Z/selector/selected_candidate.json` | Enforce evidence ledger and claim-tier policy. | 0.5 day | Fixed (`EVIDENCE_STATUS.md`) |
| A2 | P1 | Correctness | TopK auxiliary loss was not included in training loss. | `src/training/train_sae.py` | Include aux term in total loss and logging. | 0.5 day | Fixed + tested |
| A3 | P1 | Correctness | Training loop could divide by zero when no batches produced. | `src/training/train_sae.py` | Stop using `drop_last=True` and fail clearly on zero-batch case. | 0.25 day | Fixed + tested |
| A4 | P1 | Reproducibility | `wandb` import hard dependency even when disabled. | `src/training/train_sae.py` | Lazy import with graceful fallback. | 0.25 day | Fixed + tested |
| A5 | P1 | Correctness | Feature-stat code crashed for singleton SAE inputs. | `src/analysis/feature_matching.py` | Guard `n < 2` and emit sentinel stats. | 0.25 day | Fixed + tested |
| A6 | P1 | Correctness | Routed frontier allowed invalid expert partitioning (`num_experts > d_sae`). | `scripts/experiments/run_routed_frontier_external.py` | Validate CLI args and expert-slice constraints. | 0.25 day | Fixed + tested |
| A7 | P1 | Correctness | Assignment-v2 accepted empty seeds/lambdas then failed later. | `scripts/experiments/run_assignment_consistency_v2.py` | Reject empty lists at parse time and in summarize. | 0.25 day | Fixed + tested |
| A8 | P1 | Correctness | CE-Bench custom eval used opaque dict-key access for model mappings. | `scripts/experiments/run_husai_cebench_custom_eval.py` | Validate supported model names before lookup. | 0.25 day | Fixed |
| A9 | P1 | Safety | Official benchmark harness executed commands using `shell=True`. | `scripts/experiments/run_official_external_benchmarks.py` | Parse argv, reject shell operators, run with `shell=False`. | 0.5 day | Fixed + tested |
| A10 | P2 | Maintainability | Placeholder GitHub links and stale URLs remained in contributor/docs metadata. | `pyproject.toml` | Replace placeholders with real repository links. | 0.25 day | Fixed |

## Residual Risks

1. Remote final package is not fully mirrored locally; exact final candidate claims remain tier-dependent.
2. External strict-gate success remains unresolved (`pass_all=false`).
3. Some historical docs and archived notes are stale by design and may be misread without canonical index guidance.

## Validation Performed

Targeted regression suite:

```bash
pytest -q tests/unit/test_train_sae_edge_cases.py \
  tests/unit/test_feature_matching_edge_cases.py \
  tests/unit/test_routed_frontier_modes.py \
  tests/unit/test_assignment_consistency_v2.py \
  tests/unit/test_official_external_benchmark_harness.py
```

Result: `17 passed`.
