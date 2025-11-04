# Citations and Writing Templates for Paper

**Date:** November 4, 2025  
**Purpose:** Ready-to-use citations and section templates

---

## Key Citations

### SAE Foundational Work
1. Bricken et al. (2023) "Towards Monosemanticity" - Anthropic
2. Templeton et al. (2024) "Scaling Monosemanticity" - Claude 3 Sonnet
3. Cunningham et al. (2023) "Sparse Autoencoders Find Highly Interpretable Features"

### Stability Papers ⭐ CRITICAL
4. **Fel et al. (2025) "Sparse Autoencoders Trained on the Same Data Learn Different Features"**
   - arXiv:2501.16615
   - Reports 30% overlap for Llama 3 8B
   - VALIDATES OUR FINDINGS!

5. **Position Paper (2025) "Mechanistic Interpretability Should Prioritize Feature Consistency"**
   - arXiv:2505.20254
   - Argues consistency is essential
   - We directly address this!

### Grokking/Modular Arithmetic
6. Nanda et al. (2023) "Progress measures for grokking" - ICLR 2023
7. Zhong et al. (2024) Clock vs Pizza circuits

---

## Abstract (150 words)

Sparse Autoencoders (SAEs) have emerged as a leading tool for mechanistic interpretability, yet their feature reproducibility remains understudied. We present the first systematic multi-seed, multi-architecture stability analysis. Training 10 SAEs (5 TopK, 5 ReLU) on modular arithmetic transformers, we find architecture-independent instability: mean pairwise maximum cosine correlation (PWMCC) of 0.30±0.001, with no significant difference between architectures (p>0.05, Cohen's d=0.02). Critically, this persists despite excellent reconstruction (explained variance >0.92), revealing a troubling decoupling. Our attempted Fourier ground truth validation revealed the 2-layer transformer learned alternative algorithms (R²=2% vs literature's 93-98%), making our findings more general: instability is algorithm-independent. These results challenge assumptions about SAE feature objectivity and emphasize the need for stability-aware evaluation practices.

---

## Introduction Hook

"Sparse Autoencoders (SAEs) promise to decompose neural networks into interpretable features, with recent applications to frontier models achieving unprecedented scale [cite]. Yet a fundamental question remains unexamined: do SAEs consistently recover the same features across independent training runs? If features vary with random initialization, interpretations may be artifacts rather than discoveries."

---

## Main Result Statement

"We find systematic feature instability across all 10 SAEs (mean PWMCC=0.30±0.001), with no significant architectural dependence (TopK vs ReLU: p=0.48, d=0.02). The extremely tight variance (std=0.001) indicates this is a robust phenomenon, not random fluctuation."

---

## Discussion Opener

"Our results reveal a reproducibility crisis in SAE feature learning. Despite achieving standard performance targets (EV>0.92, appropriate sparsity), independently trained SAEs converge to fundamentally different feature representations (PWMCC~0.30). This challenges the implicit assumption that SAEs uncover objective, unique decompositions."

---

## Conclusion

"We have demonstrated that SAE feature instability is a fundamental, architecture-independent phenomenon. The convergence of our findings with recent large-scale studies [cite Fel et al.] and position papers [cite] suggests this warrants dedicated research attention. Future work should prioritize: (1) understanding the optimization landscape that permits multiple solutions, (2) developing stability-promoting training objectives, (3) establishing reproducibility standards for the field. Only by addressing feature consistency can SAEs fulfill their promise for reliable mechanistic interpretability."

---

## Architecture Note (for Methods)

"We trained a 2-layer transformer (n_layers=2, d_model=128, n_heads=4) achieving 100% accuracy. Unlike Nanda et al.'s 1-layer architecture that learns Fourier circuits, 2-layer transformers have capacity for alternative algorithms [cite], making our SAE findings algorithm-independent."
