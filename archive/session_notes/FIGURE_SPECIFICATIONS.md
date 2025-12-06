# Figure Specifications for Paper

**Date:** November 4, 2025  
**Purpose:** Exact specifications for publication figures

---

## Figure 1: PWMCC Overlap Matrices

**Layout:** 2 subplots side-by-side

**Data:**
- TopK: `results/analysis/topk_stability_analysis.json`
- ReLU: `results/analysis/relu_stability_analysis.json`

**Visual specs:**
- 5×5 heatmap per subplot
- Colormap: RdYlGn (red=low, yellow=medium, green=high)
- Range: 0.0-1.0
- Annotate cells with values (2 decimals)
- Diagonal = 1.0 (self-similarity)
- Seeds: 42, 123, 456, 789, 1011

**Caption:**
```
Feature stability via PWMCC across 5 seeds. Both TopK (k=32) and ReLU (L1=1e-3) 
show systematic instability (mean~0.30, std~0.001) despite different sparsity 
mechanisms.
```

---

## Figure 2: Reconstruction-Stability Scatter

**Purpose:** Show decoupling of metrics

**Data:** 10 points (5 TopK circles, 5 ReLU triangles)
- X: Explained variance (0.90-1.0)
- Y: Mean PWMCC (~0.30)

**Visual:**
- Horizontal line at y=0.7 (stability threshold)
- Vertical line at x=0.95 (good reconstruction)
- Label quadrants

**Caption:**
```
All SAEs achieve excellent reconstruction (EV>0.92) but low stability 
(PWMCC~0.30), revealing decoupling between standard metrics and feature 
consistency.
```

---

## Table 1: Architecture Comparison

| Metric | TopK | ReLU | p-value | Cohen's d |
|--------|------|------|---------|-----------|
| PWMCC | 0.302±0.001 | 0.300±0.001 | >0.05 | 0.02 |
| EV | 0.923±0.002 | 0.980±0.002 | <0.001 | 28.5 |
| L0 | 32±0 | 427±18 | <0.001 | - |
| Dead % | 0.4±0.1 | 15.6±2.9 | <0.001 | 6.5 |

**Caption:**
```
Despite different reconstruction quality and sparsity, both architectures show 
identical instability (p>0.05). Mean±SEM, n=5 per group.
```
