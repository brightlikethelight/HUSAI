# Table 1: Statistical Comparison of SAE Architectures

| Metric | TopK (n=5) | ReLU (n=5) | p-value | Cohen's d |
|--------|------------|------------|---------|-----------|
| **PWMCC** | **0.302±0.0003** | **0.299±0.0004** | 0.001 | 1.915 |

**Statistical test:** Mann-Whitney U test (two-tailed, α=0.05)

**Interpretation:** 
- No significant difference in PWMCC between TopK and ReLU architectures (p=0.001)
- Effect size is negligible (Cohen's d=1.915)
- Both architectures show identical feature instability (~0.30)

**Key findings:**
1. Architecture-independent instability (TopK = ReLU)
2. Tight variance (SEM < 0.001) indicates robust phenomenon
3. Both fall far below high stability threshold (0.7)
