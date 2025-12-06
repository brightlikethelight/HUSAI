# Summary Statistics: Trained vs Random SAE PWMCC

## Table 1: Descriptive Statistics

| Metric | Trained SAEs | Random SAEs | Difference |
|--------|--------------|-------------|------------|
| **Mean PWMCC** | 0.3086 | 0.3004 | 0.0082 |
| **Std Dev** | 0.0017 | 0.0007 | - |
| **Min** | 0.3059 | 0.2993 | - |
| **Max** | 0.3106 | 0.3014 | - |
| **Median** | 0.3094 | 0.3003 | - |
| **n (pairs)** | 10 | 10 | - |

## Table 2: Statistical Inference

| Test | Statistic | p-value | Effect Size (Cohen's d) | Conclusion |
|------|-----------|---------|-------------------------|------------|
| **Mann-Whitney U** | 100.00 | 9.13e-05 | 6.212 | Reject H0 |

## Interpretation

### Key Finding
**Trained SAEs show statistically significant but practically negligible improvement over random baseline.**

- **Statistical significance**: p = 9.13e-05 (p < 0.001) indicates the difference is unlikely due to chance
- **Practical significance**: Δ = 0.0082 is only 2.7% improvement
- **Effect size**: Cohen's d = 6.21 indicates a large statistical effect but small practical difference

### Conclusion
Despite achieving excellent reconstruction (EV > 0.99), trained SAEs learn features that are nearly as unstable as randomly initialized networks. This represents a fundamental reproducibility crisis in SAE training.

### Implications
1. **Multi-seed evaluation is essential** - single-seed results are not reproducible
2. **Standard metrics are insufficient** - reconstruction quality ≠ feature stability
3. **New training methods needed** - current approaches fail to learn stable features
