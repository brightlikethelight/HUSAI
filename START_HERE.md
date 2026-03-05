# Start Here

Updated: 2026-03-05

HUSAI is a reliability-first SAE research project. It evaluates whether SAE features satisfy strict release criteria: internal consistency, stress robustness, and external benchmark competitiveness.

**Result: `pass_all=false`.** Internal and stress gates pass; external benchmarks do not meet strict thresholds.

## Reading Order

1. `EVIDENCE_STATUS.md` -- what is locally verified vs remote-reported
2. `EXECUTIVE_SUMMARY.md` -- detailed status, gate outcomes, evidence paths
3. `paper/sae_stability_paper.md` -- the paper (PWMCC = random baseline finding)
4. `RUNBOOK.md` -- how to reproduce everything
5. `EXPERIMENT_LOG.md` -- run-by-run history

## If You Are Extending the Project

- `HIGH_IMPACT_FOLLOWUPS_REPORT.md` -- ranked next steps
- `NOVEL_CONTRIBUTIONS.md` -- what is novel here
- `docs/04-Execution/EXPERIMENT_PLAN_2026_02_20.md` -- experiment roadmap
- `LIT_REVIEW.md` -- literature and competitive landscape
- `scripts/experiments/run_all_followup_experiments.sh` -- run all 7 follow-up experiments (Section 4.11 of the paper)

## Quick Validation

```bash
pytest tests -q
make smoke
```
