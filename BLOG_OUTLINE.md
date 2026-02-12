# Blog Outline

Working title:
- *When SAE Features Reconstruct Well but Fail to Reproduce*

## 1) Why this problem matters
- SAEs are used as interpretability tools.
- If features are not reproducible across seeds, conclusions are fragile.

## 2) What we set out to test
- Do SAEs trained on the same data converge to similar features?
- How does consistency relate to reconstruction quality?

## 3) Experimental setup
- Modular arithmetic transformer testbed.
- Multi-seed SAE runs.
- Metrics: PWMCC, random baseline, EV/MSE, sparsity stats.

## 4) Core findings
- Trained-vs-random stability gap is small in current baseline regime.
- Reconstruction can be strong while consistency remains low.
- Stability behavior depends on parameterization and regime.

## 5) What broke and what we fixed
- Script execution pathing bugs.
- Import/API drift in SAE path.
- Test and doc drift that masked reliability issues.

## 6) What we changed in engineering hygiene
- Repro runbook and audit.
- Clear critical path docs.
- Experiment logging standards and manifest plan.

## 7) What we tried next
- Baseline suite.
- Parameter sweeps and ablations.
- Candidate SOTA variants and stress tests.

## 8) What worked, what did not
- Include both positive and negative outcomes.
- Discuss random baseline controls and causal checks.

## 9) Practical recommendations
- Always run multi-seed.
- Always report random baselines.
- Always tie claims to run manifests and artifact paths.

## 10) Reproduce in one page
- minimal commands
- expected artifacts
- how to validate outputs
