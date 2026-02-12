# Health Audit (Phase 2)

Audit date: 2026-02-12
Scope: core training, analysis, reproducibility, and documentation pathways for SAE stability research.

## Summary

The repository contains valuable research artifacts and many experiments, but the core execution path is currently fragile. The highest-risk issues are pathing/API drifts that block the main SAE training flow, followed by test/documentation drift that obscures actual reliability.

## Findings by Category

| Category | Severity | Issue | Evidence | Recommendation | Effort |
|---|---|---|---|---|---|
| Correctness | P0 | Documented script invocations fail due `src` import pathing | runtime failures for `python scripts/training/train_baseline.py` and `python scripts/analysis/extract_activations.py`; bootstrapping at `scripts/training/train_baseline.py:33`, `scripts/analysis/extract_activations.py:38` | Fix root path resolution for nested scripts or remove manual path hacks and enforce module entrypoints | 0.5 day |
| Correctness | P0 | SAE CLI imports wrong extraction module path | `scripts/training/train_sae.py:64` imports `scripts.extract_activations` but file is `scripts/analysis/extract_activations.py` | Correct import, ideally move extraction utility into `src/` package | <1 hour |
| Correctness | P0 | SAELens integration is incompatible with installed API | `src/models/sae.py:64-98`; runtime `TypeError` on `LanguageModelSAERunnerConfig(... architecture=...)` | Update to SAELens v6-compatible API and add compatibility test | 0.5-1.5 days |
| Correctness | P0 | End-to-end pipeline test fails on stale model interface assumptions | `tests/test_sae_pipeline.py:112` uses `model.d_model`; runtime failure confirms | Update test to current API (`model.config.d_model`) and return formats | 0.5 day |
| Correctness | P1 | SAE trainer forward-path dispatch is overly generic and unsafe | `src/training/train_sae.py:285-290` | Replace `hasattr(..., 'forward')` branch with explicit adapter/protocol checks | 0.5 day |
| Correctness | P1 | Decoder normalization axis inconsistent with simple SAE implementation | `src/training/train_sae.py:93` (`dim=1`) vs `src/models/simple_sae.py:97-99` (`dim=0`) | Standardize decoder orientation and add unit tests for normalization behavior | 0.5 day |
| Maintainability | P1 | Command surface drift (Makefile, run script, quickstart) | `Makefile:84-106`, `run_training.sh:10`, `QUICK_START.md` | Rewrite command surfaces to one canonical execution path | 0.5 day |
| Maintainability | P2 | Conflicting status docs and stale implementation narratives | `src/README.md` claims major modules "not implemented" while they exist | Archive or update stale docs; add explicit "source of truth" docs | 0.5 day |
| Maintainability | P2 | Hardcoded absolute paths in many scripts | e.g. `scripts/training/train_expanded_seeds.py:48`; multiple files under `scripts/analysis` and `scripts/experiments` | Replace with repo-relative `Path` logic and CLI args | 1-2 days |
| Reproducibility | P1 | Test-suite drift around vocabulary assumptions | code uses `modulus + 4` (`src/utils/config.py:86-101`) while tests assume `+1` in `tests/unit/test_config.py` | Update tests/docs to current tokenization semantics | 0.5 day |
| Reproducibility | P2 | No CI pipeline currently configured | no `.github/workflows` | Add CPU CI: lint + typecheck + unit/integration smoke | 0.5 day |
| Reproducibility | P2 | Environment locking is split and partially unconstrained | `pyproject.toml` ranges plus separate `requirements*.txt` and `environment.yml` | Choose one canonical environment and add lockfile | 1 day |
| Reproducibility | P2 | OpenMP runtime conflict in local environment | runtime `OMP: Error #15` without workaround | Normalize runtime stack and document supported env matrix | 0.5-1 day |
| Performance | P2 | DataLoader reproducibility not fully controlled for shuffle workers | `src/data/modular_arithmetic.py:369-375` uses shuffle but no explicit loader generator/worker seeding | pass seeded generator and worker init fn where needed | 0.5 day |
| Research Hygiene | P1 | Some paper/doc claims not clearly linked to exact run manifests | broad findings across README/docs without unified command->artifact manifest | create experiment manifest and per-figure provenance table | 1 day |
| Research Hygiene | P1 | Evaluation suite is mostly internal; external benchmark alignment missing | no SAEBench/CE-Bench integration in current pipeline | add benchmark adapters and cross-benchmark reporting | 2-4 days |

## What Is Correct and Valuable

- Modular arithmetic dataset implementation and tests are strong:
  - `tests/unit/test_modular_arithmetic.py` currently passes (43/43).
- Transformer training and activation extraction work via module entrypoints.
- Large body of analysis artifacts exists under `results/` and `docs/results/`.
- Core stability metric implementation (decoder-space PWMCC) is explicit and inspectable.

## Prioritized Remediation Plan

1. Restore executable baseline (`scripts` pathing + `train_sae` import fix).
2. Repair SAELens integration and make a minimal SAE training smoke test pass.
3. Repair test/doc drift (`test_config.py`, quickstart, make targets).
4. Add CI and env-lock strategy.
5. Add experiment manifest and benchmark alignment layer (SAEBench/CE-Bench).

## Post-Fix Snapshot (Current Workspace)

Verified after implementation pass:
- `pytest tests -q` -> 83 passed
- `python tests/test_sae_pipeline.py --transformer-checkpoint results/transformer_5000ep/transformer_best.pt` -> passes end-to-end
- `python scripts/training/train_sae.py ... --use-cached-activations ...` -> runs and saves checkpoints
- `./run_training.sh --config ... --epochs 1 ...` -> runs successfully

Interpretation:
- Core execution blockers were resolved.
- Remaining work is now mostly experiment quality, benchmark alignment, and reproducibility hardening at scale.
