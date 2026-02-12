# Stabilizing HUSAI: From Fragile SAE Pipeline to Reproducible Baseline

## Why this project exists
HUSAI studies a core interpretability question: if two SAEs are trained on the same activations with different seeds, do they learn the same features, or different decompositions with similar reconstruction quality?

That question is scientifically important because many interpretability claims assume SAE features are stable enough to support robust conclusions.

## What we found first
Before running new large experiments, we audited execution reliability. The core training path had multiple blockers:
- script pathing/import breakages
- SAE wrapper incompatibility with current dependencies
- stale tests/docs/config assumptions

These issues made the repo hard to reproduce from the documented commands.

## What we changed
- fixed script path bootstrapping in core training/extraction scripts
- fixed SAE CLI import wiring
- replaced brittle SAELens coupling in `src/models/sae.py` with a stable local SAE wrapper path
- updated transformer checkpoint extras and model interface compatibility
- repaired pipeline smoke script and config tests
- updated Makefile/run script command surfaces
- added architecture/audit/runbook/literature/experiment-plan docs

## Current state
- `pytest tests -q` passes (`83 passed`)
- baseline transformer smoke training works
- activation extraction works
- SAE training smoke works and saves checkpoints
- end-to-end pipeline script now runs successfully

## Early technical signal (not final scientific claim)
On one-epoch smoke runs, SAE quality metrics are weak (expected). That confirms infrastructure works but does not yet constitute a final scientific result. The next step is multi-seed, multi-regime runs from `EXPERIMENT_PLAN.md` with full artifact logging.

## What comes next
1. run Phase 4a multi-seed reproduction (trained vs random baseline refresh)
2. run core ablations (`k` sweep and `d_sae` sweep)
3. integrate at least one external benchmark slice (SAEBench/CE-Bench style)
4. publish artifact-linked result tables and figures

## Reproduce this status quickly
See:
- `RUNBOOK.md`
- `EXPERIMENT_LOG.md`
- `AUDIT.md`
