# Blog Outline

Date: 2026-03-05

## Working Title

Why Our SAE Project Stayed Honest: Strong Internal Gains, Failed External Gate

## Audience

Applied ML researchers and interpretability engineers.

## Structure

1. Problem framing
- Why internal metrics are insufficient.
- Why strict release gates matter.

2. What HUSAI built
- Reproduction pipeline
- External adapters (SAEBench/CE-Bench)
- Stress-gated policy

3. What we found
- Internal signal: positive
- Stress gates: pass
- External gates: fail
- Final decision: `pass_all=false`

4. Evidence integrity lesson
- Local vs remote evidence tiers (`EVIDENCE_STATUS.md`)
- Avoiding candidate/metric overclaims

5. Engineering fixes that mattered
- TopK aux loss wiring
- small-batch safety
- safer benchmark execution

6. Next experiments
- seed-complete grouped-LCB reruns
- external-aware objective branch
- official benchmark slice

7. Repro instructions
- minimal commands from `RUNBOOK.md`
- artifact locations

## Figures/Tables To Include

1. Gate result table (internal/stress/external).
2. Evidence-tier map (local verified vs remote-reported).
3. Experiment roadmap (phase4a-e).
