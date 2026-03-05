# Paper Outline

Date: 2026-03-05

## Title

Reliability-First SAE Evaluation Under Strict Stress-Gated Release Criteria

## 1. Abstract
- Problem
- Method
- Core result (`pass_all=false`)
- Contribution (claim discipline + reproducible pipeline)

## 2. Introduction
- Motivation for strict release policy
- Failure mode of internal-only claims
- Contributions list

## 3. Related Work
- SAE instability literature
- SAEBench/CE-Bench evaluation standards
- Route/Nested/TopK architecture directions
- Transcoder comparator line

## 4. Method
- Dataset and activation extraction
- SAE families evaluated
- Candidate selection protocol
- Stress-gated release policy
- Evidence-tier reporting policy

## 5. Experiments
- Reproduction baseline (phase4a)
- Baseline suite + external calibrations (phase4b)
- Ablations (phase4c)
- SOTA-chasing variants (phase4d)
- Stress tests (phase4e)

## 6. Results
- Gate outcomes table
- External deltas with confidence bounds
- Tier1 vs Tier2 evidence reconciliation note

## 7. Limitations
- External deltas still negative
- Remote package mirroring gap
- Benchmark environment coupling constraints

## 8. Broader Impact
- Positive: overclaim reduction and stronger reproducibility
- Risk: false confidence from partial metric improvements

## 9. Reproducibility Appendix
- Exact commands
- configs and hashes
- artifact map and checklist
