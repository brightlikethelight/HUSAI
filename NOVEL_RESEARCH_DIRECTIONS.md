# Novel Research Directions for SAE Stability

## Executive Summary

Based on comprehensive literature review of 15+ recent SAE papers (2024-2025), we identify three highly novel research directions that would significantly advance the field and differentiate our work.

---

## Literature Review Summary

### Key Papers Reviewed

| Paper | Key Finding | Gap We Address |
|-------|-------------|----------------|
| **Paulo & Belrose (2025)** | ~30-65% shared features across seeds | Why? What predicts stability? |
| **Song et al. (2025)** | Matched regime improves stability | Does this connect to sparsity? |
| **"Sparse but Wrong" (2508.16560)** | Low L0 → hedged features | Does hedging cause instability? |
| **Interpretability Illusions (2505.16004)** | Features not robust to perturbations | Are stable features more robust? |
| **Feature Absorption (2409.14507)** | Features get absorbed into others | Are absorbed features unstable? |
| **Ordered SAEs (2512.02194)** | Ordering improves stability | Does ordering help without hierarchy? |
| **Identifiability Theory (2506.15963)** | Conditions for unique recovery | Can stability indicate identifiability? |

### Key Insight: Three Separate Research Threads

The literature has three disconnected threads:
1. **Stability research** (Paulo, Song): Features vary across seeds
2. **Correctness research** ("Sparse but Wrong"): Low L0 → wrong features
3. **Robustness research** (Interpretability Illusions): Features not robust to inputs

**No one has connected these threads.**

---

## Novel Research Direction #1: Stability as a Signal for Optimal Sparsity ⭐⭐⭐⭐⭐

### The Gap

The "Sparse but Wrong" paper (2508.16560) shows:
- L0 selection is critical for feature correctness
- Too low L0 → feature hedging → wrong features
- They propose a complex decoder projection method to detect this

**No one has tested if STABILITY can serve as a simpler proxy.**

### Our Hypothesis

When L0 is too low:
1. SAE engages in "feature hedging" (mixing correlated features)
2. Hedging is arbitrary (many ways to mix)
3. Different seeds hedge differently
4. **Result: Unstable features**

When L0 is optimal:
1. SAE learns correct, disentangled features
2. Correct features have unique solutions
3. Different seeds converge to same features
4. **Result: Stable features**

### Experimental Design

```
Fix d_sae = 128 (matched to effective rank ~80)
Vary k = [4, 8, 16, 24, 32, 48, 64]
For each k:
  - Train 5 SAEs with different seeds
  - Compute PWMCC
  - Compute reconstruction loss
  - (Optional) Compute decoder projection metric from "Sparse but Wrong"
```

### Expected Result

Stability (PWMCC) should peak at some optimal k, then decrease for both:
- k too low (hedging)
- k too high (polysemanticity)

### Impact

- **First paper** to connect stability to sparsity selection
- **Practical utility**: Use stability to select k without ground truth
- **Unifies** two research threads (stability + correctness)

---

## Novel Research Direction #2: The Stability-Reconstruction-Correctness Trilemma ⭐⭐⭐⭐

### The Gap

Literature treats these as separate concerns:
- Stability (PWMCC)
- Reconstruction (MSE)
- Correctness (feature quality)

**No unified framework exists.**

### Our Contribution

Formalize the three-way tradeoff:

| Configuration | Stability | Reconstruction | Correctness |
|---------------|-----------|----------------|-------------|
| Small d_sae, low k | HIGH | LOW | HIGH (forced to learn important features) |
| Matched d_sae, optimal k | MEDIUM | MEDIUM | HIGH |
| Large d_sae, low k | LOW | HIGH | LOW (hedging) |
| Large d_sae, high k | LOW | HIGH | MEDIUM (polysemantic) |

### Experimental Design

Create a 2D grid:
- d_sae ∈ [32, 64, 128, 256, 512]
- k ∈ [4, 8, 16, 32, 64]

For each (d_sae, k) pair, measure:
- Stability (PWMCC)
- Reconstruction (MSE)
- Correctness (if ground truth available, or proxy metrics)

### Expected Result

A Pareto frontier showing the tradeoff. Practitioners can choose their operating point based on priorities.

### Impact

- **First unified framework** for SAE evaluation
- **Practical guidance** for practitioners
- **Theoretical contribution** to understanding SAE training

---

## Novel Research Direction #3: Frequency-Stratified Stability ⭐⭐⭐

### The Gap

Song et al. (2025) showed on LLMs:
- Frequent features (high activation rate) are more stable
- Rare features are less stable

**But LLMs have semantic structure. Does this hold on algorithmic tasks?**

### Our Hypothesis

Two possibilities:
1. **Frequency is fundamental**: Even without semantic structure, frequent features are more stable (because they're better constrained by data)
2. **Semantic structure required**: On algorithmic tasks without interpretable structure, frequency doesn't predict stability

### Experimental Design

```
For each trained SAE:
  - Compute activation frequency for each feature
  - Compute feature-level stability (max cosine sim to any feature in other seed)
  - Correlate frequency with stability
  - Compare to Song et al.'s LLM results
```

### Expected Result

If frequency-stability correlation holds:
- Validates Song et al.'s finding as fundamental
- Provides practical guidance (focus on frequent features)

If correlation doesn't hold:
- Shows semantic structure is required for stability
- Explains why our modular arithmetic SAEs are so unstable

### Impact

- **First test** of frequency-stability on non-semantic tasks
- **Clarifies** whether the relationship is fundamental or task-dependent

---

## Recommended Priority

1. **Stability vs Sparsity** (Direction #1)
   - Most impactful and novel
   - Quick to implement (modify existing scripts)
   - Directly addresses a practical problem

2. **Frequency-Stratified Analysis** (Direction #3)
   - Builds on existing data
   - Provides theoretical insight
   - Relatively quick

3. **Trilemma Framework** (Direction #2)
   - Requires more experiments
   - More theoretical contribution
   - Can be done after #1 and #3

---

## Implementation Plan

### Phase 1: Stability vs Sparsity (1-2 days)
- Create `scripts/experiments/stability_vs_sparsity.py`
- Fix d_sae=128, vary k=[4, 8, 16, 24, 32, 48, 64]
- Train 5 seeds per k
- Analyze results

### Phase 2: Frequency-Stratified Analysis (1 day)
- Modify existing analysis scripts
- Compute feature-level stability
- Correlate with activation frequency

### Phase 3: Paper Update (1 day)
- Add new findings to paper
- Update figures
- Revise conclusions

---

## References

1. Paulo & Belrose (2025). arXiv:2501.16615
2. Song et al. (2025). arXiv:2505.20254
3. "Sparse but Wrong" (2025). arXiv:2508.16560
4. Interpretability Illusions (2025). arXiv:2505.16004
5. Feature Absorption (2024). arXiv:2409.14507
6. Ordered SAEs (2024). arXiv:2512.02194
7. Identifiability Theory (2025). arXiv:2506.15963
8. SAEs for Discovery (2025). arXiv:2506.23845
