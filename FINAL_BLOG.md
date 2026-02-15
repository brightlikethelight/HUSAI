# What We Learned After Running the Full B200 HUSAI Program

Date: 2026-02-15

The original question was simple but important:

Can we make SAE features more consistent and also improve external interpretability benchmarks?

We ran the full high-impact program end-to-end and forced every claim through artifact-backed gates.

## What We Executed

1. Direct HUSAI-checkpoint CE-Bench adapter path with matched baselines.
2. Matched-budget architecture frontier (`topk`, `relu`, `batchtopk`, `jumprelu`) with multiseed external evals.
3. External scaling study across token budget, hook layer, and `d_sae`.
4. Assignment-aware consistency objective v2.
5. Stress-gated release policy with random-model, transcoder, OOD, and external gates.

Primary synthesis artifact:
- `docs/evidence/cycle3_queue_final/cycle3_final_synthesis_run_20260214T210734Z.md`

## What Held Up

- Internal consistency can be improved reliably.
- External benchmark stack is fully operational and reproducible.
- Stress-gated release policy blocks unsupported claims.

## What Did Not Hold Up

- External superiority claims.
- Any narrative that internal improvements alone imply external gains.

## Key Results

Frontier multiseed (`4 architectures x 5 seeds`):
- Best SAEBench delta among tested: `relu = -0.024691`.
- Best CE-Bench interpretability among tested: `topk = 7.726768`.
- CE-Bench deltas vs matched baseline stayed strongly negative.

Scaling multiseed (24 conditions):
- Layer 1 and larger width improve CE-Bench.
- Those same settings worsen SAEBench deltas.

Stress gates:
- `random_model = pass`
- `transcoder = fail` (`transcoder_delta = -0.002227966984113039`)
- `ood = pass` (`ood_drop = 0.01445406161520213`)
- `external = fail` (`external_delta = -0.017257680751151527`)
- `pass_all = false`

Gate evidence:
- `docs/evidence/cycle3_queue_final/release_policy_run_20260214T225029Z.json`

## Why This Is Still a Strong Research Outcome

This is a high-quality negative/nuanced result:
- We reduced uncertainty.
- We prevented overclaiming.
- We exposed the true frontier: internal consistency and external validity are not aligned by default.

That is exactly the kind of result that makes future progress real rather than cosmetic.

## What To Do Next

1. Add explicit multi-objective optimization/selection over internal + external metrics.
2. Add one newer architecture family under matched protocols.
3. Close known-ground-truth circuit recovery from the original proposal.
4. Gate all future summary claims directly on strict release results.

## Read Next

- `START_HERE.md`
- `EXECUTIVE_SUMMARY.md`
- `PROPOSAL_COMPLETENESS_REVIEW.md`
- `RUNBOOK.md`
- `EXPERIMENT_LOG.md`
