# Cycle 4 High-Impact Execution Plan (B200)

Date: 2026-02-15
Owner: HUSAI core team
Mode: planning-first, fail-fast, artifact-gated

## 0) Planning Note on Subagents

Attempted live subagent spawning in this environment, but blocked by active thread limit (`max 6`).
This plan uses simulated subagent tracks with explicit scopes and handoffs:
- Track A: Evidence auditor
- Track B: Method scout (literature + implementation fit)
- Track C: Experiment strategist (compute + decisions)

## 1) Cycle 4 Objective

Primary objective:
- Produce at least one candidate that improves internal consistency and external metrics enough to pass strict release gates.

Hard constraints:
- Reproducibility and traceability remain first-class.
- No claim upgrades without `pass_all=true`.

## 2) Baseline (Start Point)

From cycle-3 final artifacts:
- `random_model = pass`
- `transcoder = fail`
- `ood = pass`
- `external = fail`
- `pass_all = false`

Main bottlenecks:
1. External gate negative.
2. Transcoder gate negative.
3. Candidate-selection/gating policy not explicitly multi-objective.

## 3) Ranked High-Impact Next Steps

## Step 1 (P0): External Gate Definition + Candidate Selection Policy

### Hypothesis
A better candidate-selection policy (across frontier + scaling, multi-metric) can materially improve gate outcomes without changing base training code.

### Implementation
1. Add a selector script:
- New file: `scripts/experiments/select_release_candidate.py`
- Input: frontier/scaling result JSONs + stress summaries.
- Output: chosen candidate JSON + Pareto table.

2. Extend release gate evaluator:
- File: `scripts/experiments/run_stress_gated_release_policy.py`
- Add explicit external metric mode:
  - `--external-mode saebench|cebench|joint`
  - `--min-saebench-delta`
  - `--min-cebench-delta`
  - `--joint-policy` (e.g., both must pass).

3. Update queue script:
- File: `scripts/experiments/run_b200_high_impact_queue.sh`
- Replace SAEBench-only best-candidate selection with selector output.

### Acceptance Criteria
- Release gate input references selector output artifact (not ad hoc summary path).
- Gate report includes both SAEBench and CE-Bench deltas when `joint` mode is selected.
- One-click queue run produces a candidate table and chosen rationale.

### Estimate
- Engineering: 1-2 days.
- Compute: negligible (<1 GPUh).

---

## Step 2 (P0): Matched Multi-Objective External Frontier (Expansion)

### Hypothesis
Newer sparse-routing architectures can improve external Pareto frontier beyond current TopK/ReLU/BatchTopK/JumpReLU tradeoff.

### Candidate methods
- RouteSAE (routing-based specialization)
- Hierarchical Top-K SAE (coarse-to-fine sparsity)
- Matryoshka SAE (nested capacity control)

### Implementation
1. Extend architecture support:
- File: `scripts/experiments/husai_custom_sae_adapter.py`
- File: `scripts/experiments/run_architecture_frontier_external.py`
- Add architecture aliases and constructor mapping.

2. Add safe fallback path:
- If direct class unavailable in local deps, use compatibility wrappers and log explicit capability flags.

3. Add architecture-level metadata schema:
- Ensure summary JSON stores architecture family/version/config hash.

### Experiment Matrix
- Architectures: existing 4 + 1 new family (minimum), target 2 new families.
- Seeds: `42,123,456,789,1011`.
- Keep matched activation budget + CE-Bench max rows fixed.

### Acceptance Criteria
- New architecture family evaluated with full multiseed external metrics.
- At least one candidate improves one external metric without regressing the other beyond pre-defined tolerance.
- If no improvement: explicit negative result with CI and failure analysis.

### Estimate
- Engineering: 2-4 days.
- Compute: 20-40 GPUh depending on architecture count.

---

## Step 3 (P0): Assignment-Aware Objective v3 with External-Aware Selection

### Hypothesis
Internal consistency objective can be retained while reducing external regressions via selection and calibration changes (not necessarily end-to-end differentiable external loss).

### Implementation
1. Add v3 runner:
- New file: `scripts/experiments/run_assignment_consistency_v3.py`
- Keep internal objective family, add external-aware checkpoint selection stage.

2. Selection logic:
- Pareto score over:
  - internal delta LCB,
  - EV drop,
  - SAEBench delta,
  - CE-Bench delta.

3. Calibration sweeps:
- small hypergrid around v2 best lambda and sparsity settings.

### Experiment Matrix
- Lambdas: around 0.2 baseline (`0.05,0.1,0.2,0.3`).
- `k` values: top 2 from prior internal sweeps.
- Seeds: 3 first (fail-fast), then 5 for finalists.

### Acceptance Criteria
- Candidate achieves non-negative change on internal metrics and improves external joint score relative to cycle-3 best candidate.
- Gate readiness report generated for top 3 candidates.

### Estimate
- Engineering: 2-3 days.
- Compute: 20-35 GPUh.

---

## Step 4 (P1): Transcoder Stress Fix Program

### Hypothesis
Current transcoder gate failure may be addressable through matched-capacity/transcoder-training calibration rather than indicating a fundamental ceiling.

### Implementation
1. Review and expand transcoder eval settings:
- File: `scripts/experiments/run_transcoder_stress_eval.py`
- Add sweeps for transcoder capacity, epochs, and regularization.

2. Add stress-summary comparability table:
- `delta`, confidence intervals, and per-seed variance.

3. Integrate best transcoder setting into release gate by selector.

### Acceptance Criteria
- Transcoder delta confidence interval overlaps or exceeds zero for at least one matched setting.
- Stress artifacts include reproducibility metadata and per-seed outputs.

### Estimate
- Engineering: 1-2 days.
- Compute: 10-20 GPUh.

---

## Step 5 (P1): Proposal Closure Track - Known-Circuit Recovery

### Hypothesis
Ground-truth circuit recovery experiments will clarify whether current non-transfer is benchmark mismatch or representational failure.

### Implementation
1. Add dedicated known-circuit suite:
- New folder: `scripts/experiments/known_circuit_recovery/`
- Include matched trained/random controls and cross-seed feature recovery metrics.

2. Add closure artifact:
- `results/experiments/known_circuit_recovery/<run_id>/summary.json`

3. Integrate into claim audit:
- File: `scripts/analysis/verify_experiment_consistency.py`
- Add proposal-closure check IDs.

### Acceptance Criteria
- Report includes recovery-above-random with CI (or explicit null result).
- Proposal completeness document can mark this section closed.

### Estimate
- Engineering: 2-3 days.
- Compute: 10-25 GPUh.

## 4) Two-Week B200 Schedule

## Week 1

Day 1-2:
- Step 1 policy/gating refactor.
- Sanity tests and dry-run queue.

Day 3-5:
- Step 2 architecture expansion and first multiseed frontier batch.
- Early stop if all new candidates are Pareto-dominated after 3 seeds.

## Week 2

Day 6-8:
- Step 3 objective v3 sweeps + external-aware selection.

Day 9-10:
- Step 4 transcoder stress sweeps for gate recovery attempts.

Day 11-14:
- Step 5 known-circuit recovery closure.
- Final strict gate and writeup refresh.

## 5) Compute Budget (Expected)

- Step 1: <1 GPUh
- Step 2: 20-40 GPUh
- Step 3: 20-35 GPUh
- Step 4: 10-20 GPUh
- Step 5: 10-25 GPUh
- Total planned: 60-120 GPUh (fits prior expected scale)

## 6) Release/Claim Gates for Cycle 4

Minimum for claim upgrade:
1. `pass_all=true` on strict release policy.
2. External improvement confirmed on selected candidate with multiseed uncertainty.
3. Candidate selection rationale artifact exists and is reproducible.
4. Consistency audit passes with updated checks.

## 7) Fail-Fast Rules

1. Stop any branch if 3-seed pilot shows both SAEBench and CE-Bench regressions vs cycle-3 best.
2. Stop any architecture branch if training instability produces >20% failed runs.
3. Stop objective branch if EV drop exceeds threshold without external gains.

## 8) Engineering Hygiene Requirements (Non-Negotiable)

- Determinism: set and log all seeds; add CUDA determinism envs in run scripts.
- Manifest discipline: command, config hash, git commit, dataset slice.
- Logging: add optional W&B integration with run IDs mapped to artifacts.
- CI: add selector/gate unit tests.

## 9) Deliverables

1. `ADVISOR_BRIEF.md` updated after cycle-4 completion.
2. `EXECUTIVE_SUMMARY.md` with gate status and external deltas.
3. `EXPERIMENT_LOG.md` entries for all cycle-4 runs.
4. Final candidate dossier:
- selector output,
- strict gate report,
- external and stress summaries,
- claim-safe conclusion.
