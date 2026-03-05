# What HUSAI Actually Proved (And What It Did Not)

Date: 2026-03-05

HUSAI was built to answer a simple question with strict standards:

Can SAE improvements survive both stress tests and external benchmarks, not just internal metrics?

## What Worked

- We built a reproducible research pipeline with manifests, scripted queues, and strict release gates.
- Internal trained-vs-random consistency signal is positive.
- Stress controls (`random_model`, `transcoder`, `OOD`) pass in documented runs.

## What Did Not Work Yet

- External competitiveness is still not there under strict thresholds.
- Final strict decision remains `pass_all=false`.

## Important Evidence Integrity Update

The repo currently has two evidence tiers (`EVIDENCE_STATUS.md`):
- Local verified snapshot (2026-02-15) selects `topk_seed123` and still fails strict external gates.
- Remote-reported cycle-10 package (2026-02-18) reports `relu_seed42`, also with `pass_all=false`.

So the core scientific conclusion is stable even though final-candidate identity differs across tiers.

## What We Improved In This Pass

We fixed several reliability defects that could distort research results:
- TopK auxiliary loss is now included in optimization.
- Small datasets no longer crash the training loop.
- Optional `wandb` support is now truly optional.
- Feature-stability code handles single-model edge cases.
- Routed/assignment/benchmark wrappers now fail fast on invalid inputs.
- Official benchmark harness no longer executes commands via `shell=True`.

## Why This Matters

Negative results are useful when they are trustworthy.

HUSAI now has stronger claim discipline: it separates real internal progress from unresolved external transfer and keeps release claims blocked when evidence is insufficient.

## Next Research Move

Shift from internal-only optimization to explicit external-transfer objectives with seed-complete grouped evaluation and matched benchmark protocols.
