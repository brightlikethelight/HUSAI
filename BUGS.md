# BUGS and Reliability Gaps (Prioritized)

This list includes only issues validated by direct code inspection and/or command execution.

## P0 (Blocks Core Research Pipeline)

### P0-1: Main training scripts fail when run as documented
- Issue: `python scripts/training/train_baseline.py` and similar commands fail with `ModuleNotFoundError: src`.
- Evidence:
  - `scripts/training/train_baseline.py:33`
  - `scripts/training/train_sae.py:57`
  - `scripts/analysis/extract_activations.py:38`
  - runtime: direct invocations fail; module invocations (`python -m ...`) work.
- Impact: README/Makefile/run scripts are not executable as written.
- Fix: correct repo-root bootstrap (`Path(__file__).resolve().parents[2]` for nested scripts) or remove manual `sys.path` edits and rely on package/module execution.
- Estimated effort: 0.5 day.

### P0-2: SAE training CLI import path is wrong
- Issue: `scripts/training/train_sae.py` imports `scripts.extract_activations`, but file is in `scripts/analysis/extract_activations.py`.
- Evidence: `scripts/training/train_sae.py:64`.
- Impact: `python -m scripts.training.train_sae --help` fails immediately.
- Fix: import from `scripts.analysis.extract_activations` or move helper into `src/` and import there.
- Estimated effort: <1 hour.

### P0-3: SAELens API mismatch in SAE construction
- Issue: `LanguageModelSAERunnerConfig` usage in `src/models/sae.py` does not match installed `sae-lens` API.
- Evidence:
  - `src/models/sae.py:64-98` passes `architecture=...`.
  - runtime: `TypeError: LanguageModelSAERunnerConfig.__init__() got an unexpected keyword argument 'architecture'`.
- Impact: `SAEWrapper` creation fails; core SAE training path is blocked.
- Fix: migrate `create_sae` to current SAELens config API or pin a known-compatible sae-lens version with lockfile and tests.
- Estimated effort: 0.5-1.5 days (depends on migration depth).

### P0-4: End-to-end SAE pipeline test is broken
- Issue: `tests/test_sae_pipeline.py` fails before training due stale model API assumptions.
- Evidence: runtime failure `ModularArithmeticTransformer object has no attribute d_model`.
- File references:
  - `tests/test_sae_pipeline.py:112`
  - related API: `src/models/transformer.py` exposes `config.d_model` instead.
- Impact: no reliable e2e guardrail; regressions can pass unnoticed.
- Fix: align test with current APIs and return types, then include in CI smoke.
- Estimated effort: 0.5 day.

## P1 (High Risk, Non-Blocking or Secondary)

### P1-1: `src/training/train_sae.py` forward-path logic is unsafe
- Issue: branch uses `hasattr(sae, 'forward')` and calls `sae(batch_acts, return_latents=True)`.
- Evidence: `src/training/train_sae.py:285-287`.
- Why unsafe:
  - all `nn.Module`s have `forward`; this is not a meaningful capability check.
  - incompatible for models whose `forward` signature differs.
- Impact: fragile runtime behavior, hidden type errors across SAE implementations.
- Fix: explicit type/capability checks, with clear adapter interface.
- Estimated effort: 0.5 day.

### P1-2: Decoder normalization axis likely wrong in shared trainer
- Issue: normalization uses `dim=1` in shared trainer.
- Evidence: `src/training/train_sae.py:93`.
- Counter-evidence in same repo: simple SAE implementations normalize decoder by `dim=0` (`src/models/simple_sae.py:97-99`, `src/models/simple_sae.py:321-323`).
- Impact: potential mismatch with intended feature-direction normalization, affecting stability metrics.
- Fix: standardize decoder tensor convention and assert/normalize with tests.
- Estimated effort: 0.5 day.

### P1-3: Config tests and docs are stale vs code
- Issue: multiple tests expect `vocab_size = modulus + 1`, but implementation is `modulus + 4` for sequence format.
- Evidence:
  - `src/utils/config.py:86-101`
  - failing tests in `tests/unit/test_config.py` (8 failures)
  - stale docs: `configs/README.md` still says `modulus + 1` in examples.
- Impact: CI noise, developer confusion, loss of trust in test suite.
- Fix: update tests/docs to match current tokenization design.
- Estimated effort: 0.5 day.

### P1-4: Quickstart/Makefile command surface is stale
- Issue: commands reference old script paths (e.g., `scripts/train_baseline.py`, `scripts/train_sae.py`) and missing config files.
- Evidence: `Makefile:84-106`, `run_training.sh:10`, `QUICK_START.md`.
- Impact: onboarding failure and false "it doesn't run" reports.
- Fix: rewrite docs and make targets to working module invocations.
- Estimated effort: 0.5 day.

## P2 (Important Hygiene and Scale Risks)

### P2-1: Many hardcoded absolute paths reduce portability
- Issue: multiple scripts hardcode `/Users/brightliu/School_Work/HUSAI`.
- Example evidence: `scripts/training/train_expanded_seeds.py:48` plus many files under `scripts/analysis` and `scripts/experiments`.
- Impact: breaks on CI/cluster/other machines.
- Fix: use repo-relative paths derived from file location or cwd args.
- Estimated effort: 1-2 days (batch cleanup).

### P2-2 (Resolved): CI workflow now configured
- Previous issue: no CI workflow was present.
- Current state: `.github/workflows/ci.yml` exists with smoke and quality jobs.
- Remaining work: expand incremental lint/typecheck scope to full-repo gates.
- Estimated effort for expansion: 0.5-1 day.

### P2-3: Environment is not fully locked
- Issue: dependency definitions are split and partially unconstrained (`pyproject.toml` ranges, separate conda/pip stories).
- Impact: drift and hard-to-reproduce failures.
- Fix: choose one primary env source and generate lockfiles.
- Estimated effort: 1 day.

### P2-4: OpenMP runtime conflict in local env
- Issue: import can fail without `KMP_DUPLICATE_LIB_OK=TRUE`.
- Impact: local instability and potential silent numerical risk with workaround.
- Fix: standardize env + document supported stack; avoid duplicate OpenMP libs.
- Estimated effort: 0.5-1 day.

## Suggested Fix Order

1. P0-1, P0-2 (path and import correctness)
2. P0-3 (SAELens compatibility)
3. P0-4 + P1-3 (repair tests as real guardrails)
4. P1-1/P1-2 (trainer correctness)
5. P1-4 + P2 series (docs/CI/repro hardening)

## Status Update (After Initial Fix Pass)

Resolved in current workspace:
- P0-1 script pathing for core training/extraction scripts
- P0-2 SAE CLI extraction import path
- P0-3 SAELens incompatibility by switching core wrapper path to local simple SAE implementations
- P0-4 e2e pipeline script compatibility
- P1-3 config test drift (unit tests now aligned and passing)
- P1-4 command surface drift in `Makefile` and `run_training.sh`

Still open:
- P2 absolute-path cleanup outside the core experiment/analysis paths (for example auxiliary training scripts)
- P2 environment lockfile strategy
- P2 OpenMP/TMPDIR environment fragility on this machine
