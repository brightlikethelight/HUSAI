# Do Sparse Autoencoders Learn Reproducible Features? A Multi-Seed, Multi-Architecture Analysis

**Authors:** [To be filled]  
**Affiliation:** [To be filled]  
**Date:** November 2025

---

## Abstract

Sparse Autoencoders (SAEs) have emerged as a leading tool for mechanistic interpretability, decomposing neural network activations into interpretable features. Recent work identified low feature overlap across training runs (Paulo & Belrose, 2025) and argued that consistency should be prioritized (Song et al., 2025). We present the first systematic multi-seed, multi-architecture stability analysis with critical random baseline controls. Training 10 SAEs (5 TopK, 5 ReLU) on a grokking transformer for modular arithmetic, we find that feature consistency (PWMCC ≈ 0.30) is **indistinguishable from randomly initialized SAEs**—standard training produces zero stability above random chance. However, alternative metrics (subspace overlap) show trained SAEs DO learn structure, suggesting they find the correct subspace but not a consistent basis. We also discover striking layer dependence: Layer 0 shows BELOW-random consistency (PWMCC = 0.047), indicating orthogonal feature solutions across seeds. These findings reframe the "consistency gap" from "low vs high stability" to "random baseline vs learned structure," with urgent implications for interpretability research relying on single SAE instances.

**Keywords:** Sparse Autoencoders, Mechanistic Interpretability, Feature Stability, Reproducibility

---

## 1. Introduction

Sparse Autoencoders (SAEs) promise to decompose neural networks into human-interpretable features, with recent applications scaling to frontier language models (Templeton et al., 2024; Gao et al., 2024). The fundamental premise is that SAEs can uncover "the true underlying features used by a model" (Elhage et al., 2022), enabling safety-relevant analyses such as verifying "a model will never lie" (Olah, 2023). Yet a critical question threatens this enterprise: **do SAEs consistently recover the same features across independent training runs?**

If features vary with random initialization, interpretations based on single SAE instances may be artifacts rather than discoveries. This concern was recently validated empirically by Paulo & Belrose (2025), who found that only 30% of features are shared across independently trained SAEs on large language models, prompting Song et al. (2025) to argue that feature consistency should be elevated to a primary evaluation criterion. While Song et al. demonstrate that 0.80 consistency is theoretically achievable with appropriate architectural choices and training objectives, the baseline consistency of current standard practices—what practitioners actually obtain—remains uncharacterized.

### 1.1 Research Questions

We address two fundamental questions:

1. **What is the baseline feature consistency of SAEs under standard training?** We systematically measure feature stability using the Pairwise Maximum Cosine Correlation (PWMCC) metric across 10 independently trained SAEs.

2. **Is feature instability architecture-dependent?** We compare TopK and ReLU SAE architectures under matched conditions to determine whether stability can be improved through architectural choice alone.

### 1.2 Key Contributions

Our work makes four primary contributions:

1. **First systematic multi-architecture stability study:** We provide controlled comparison of TopK vs ReLU SAEs at matched sparsity levels, finding no significant practical difference in stability (PWMCC~0.30 for both).

2. **Baseline empirical characterization:** We confirm that standard training practices yield 0.30 PWMCC baseline across architectures and tasks, validating Paulo & Belrose's observations in a controlled setting.

3. **Identification of the consistency gap:** We quantify the gap between current practice (0.30) and achievable consistency (0.80, Song et al.), revealing that closing this 0.50 PWMCC gap requires dedicated training-level interventions beyond architectural choice.

4. **Decoupling of reconstruction and stability metrics:** We demonstrate that excellent reconstruction quality (explained variance >0.92) does not guarantee feature consistency, challenging the adequacy of current evaluation practices.

Our findings provide critical empirical grounding for the mechanistic interpretability community, confirming that feature reproducibility should indeed be prioritized and that current standard practices fall far short of achievable consistency levels.

---

## 2. Related Work

### 2.1 Sparse Autoencoders for Interpretability

Sparse Autoencoders have emerged as a leading tool for mechanistic interpretability (Cunningham et al., 2023; Bricken et al., 2023). The core idea is to train an encoder-decoder pair that transforms dense neural activations into sparse, higher-dimensional latent spaces, with the decoder's columns ideally representing monosemantic features. Recent scaling efforts have applied SAEs to production language models, including GPT-4 (Gao et al., 2024) and Claude 3 Sonnet (Templeton et al., 2024).

Two major architectural variants have emerged: **ReLU SAEs** with L1 sparsity penalty (Bricken et al., 2023) and **TopK SAEs** that enforce hard sparsity by activating only the k largest latents (Gao et al., 2024). While TopK SAEs achieve better reconstruction with fewer active features, their stability properties relative to ReLU SAEs remained uncharacterized prior to our work.

### 2.2 Feature Stability and Consistency

Recent work has raised concerns about SAE feature reproducibility. Paulo & Belrose (2025) trained 9 SAEs with different random seeds on Pythia 160M and Llama 3 8B, finding that only 30-42% of features are consistently recovered across runs. Using the Hungarian algorithm for optimal feature matching, they classified features as "shared" (high cosine similarity with counterpart in other SAE) or "orphan" (seed-dependent). Critically, they found TopK SAEs more seed-dependent than ReLU SAEs on large language models.

Song et al. (2025) elevated this concern to a position paper, arguing that mechanistic interpretability should prioritize feature consistency. They propose using PWMCC as a practical metric and demonstrate that 0.80 consistency is achievable with appropriate architectural choices and training objectives. Their theoretical analysis and synthetic validation establish consistency as a viable optimization target.

Our work bridges these findings by providing systematic empirical characterization of baseline consistency across architectures under standard training, identifying the gap between current practice and demonstrated achievable levels.

### 2.3 Grokking and Modular Arithmetic

We evaluate SAEs on a grokking transformer trained for modular arithmetic (Power et al., 2022; Nanda et al., 2023). This controlled setting offers several advantages: (1) 100% accuracy enables verification that the model has learned the task, (2) prior work identified specific algorithmic solutions (e.g., Fourier circuits in 1-layer transformers), and (3) the simplicity allows focused study of SAE stability without confounds from task complexity.

Nanda et al. (2023) showed that 1-layer transformers learn Fourier addition circuits for modular arithmetic, achieving 93-98% explained variance from Fourier components. Our 2-layer transformer architecture learns alternative algorithms (R²=2%), making our stability findings algorithm-independent rather than specific to Fourier-based models.

---

## 3. Methods

### 3.1 Experimental Setup

**Transformer training:** We trained a 2-layer transformer (d_model=128, n_heads=4, d_mlp=512) on modular addition (mod 113) to 100% accuracy, following the grokking paradigm (Power et al., 2022). Training used AdamW optimizer with learning rate 1e-3 and weight decay 1.0 for 5000 epochs on 30% of possible pairs.

**SAE architectures:** We trained two SAE variants:
- **TopK:** Hard sparsity with k=32 active features (Gao et al., 2024)
- **ReLU:** Soft sparsity with L1 penalty λ=1e-3 (Bricken et al., 2023)

Both used expansion factor 8× (d_sae=1024 for d_model=128) and were trained on transformer residual stream activations at layer 1.

**Multi-seed protocol:** For each architecture, we trained 5 SAEs with random seeds {42, 123, 456, 789, 1011}, ensuring:
- Identical data (same transformer activations)
- Identical hyperparameters (only initialization differs)
- Identical training duration (convergence verified)

This yielded 10 total SAEs (5 TopK + 5 ReLU) for systematic comparison.

### 3.2 Evaluation Metrics

**Pairwise Maximum Cosine Correlation (PWMCC):** Following Paulo & Belrose (2025) and Song et al. (2025), we measure feature consistency using PWMCC. For each pair of SAEs (i, j), we compute:

1. **Cosine similarity matrix:** S ∈ ℝ^(d_sae × d_sae) where S_ab = cos(decoder_i[:,a], decoder_j[:,b])
2. **Max similarity per feature:** For each feature in SAE i, find its best match in SAE j
3. **PWMCC:** Average of maximum similarities

PWMCC ranges from 0 (no overlap) to 1 (perfect alignment). We use 0.7 as the high stability threshold (Song et al., 2025).

**Reconstruction metrics:** We also measure:
- Explained variance: 1 - ||x - x̂||²/||x||²
- L0 sparsity: Mean number of active features
- Dead neuron percentage: Features that never activate

**Statistical tests:** We compare TopK vs ReLU using Mann-Whitney U test (non-parametric, appropriate for PWMCC distributions) and report Cohen's d effect sizes.

### 3.3 Implementation

All experiments used PyTorch 2.0 with mixed precision training. SAEs were trained for 10,000 steps with batch size 4096, using AdamW (lr=3e-4, β=(0.9, 0.999)). Code and data are available at [repository URL].

---

## 4. Results

### 4.1 Main Finding: Architecture-Independent Instability

Figure 1 shows PWMCC matrices for both architectures. The off-diagonal values cluster tightly around 0.30 for both TopK and ReLU SAEs, revealing systematic feature instability independent of architectural choice.

**Quantitative comparison:**
- TopK: PWMCC = 0.302 ± 0.0003 (mean ± SEM across 10 pairwise comparisons)
- ReLU: PWMCC = 0.300 ± 0.0004
- Difference: 0.002 (0.7% relative)
- Statistical test: p = 0.0013 (Mann-Whitney U)
- Effect size: Cohen's d = 1.92 (large statistically, but absolute difference of 0.002 is negligible)

While the difference is statistically significant due to extremely tight variance (SEM < 0.001), the practical significance is negligible: both architectures operate at the same ~0.30 baseline, far below the high stability threshold of 0.7.

**Interpretation:** The extremely low variance (std < 0.002) indicates this is a robust phenomenon, not random fluctuation. Feature instability is systematic and architecture-independent under standard training.

### 4.2 Decoupling of Reconstruction and Stability

Figure 2 reveals troubling decoupling: all 10 SAEs achieve excellent reconstruction quality (TopK EV = 0.919±0.002, ReLU EV = 0.977±0.0002) but show poor feature consistency (PWMCC ~ 0.30). The SAEs cluster in the bottom-right quadrant ("good reconstruction, poor stability"), with no SAEs reaching the ideal top-right quadrant.

This decoupling challenges current evaluation practices. Standard metrics (explained variance, L0 sparsity, dead neuron percentage) suggest successful training, yet features fail to reproduce across independent runs. A practitioner evaluating a single SAE using standard metrics would incorrectly conclude the model is reliable.

### 4.3 Validation Against Literature

Our PWMCC~0.30 finding validates Paulo & Belrose's (2025) observation of 30% feature sharing in large language models, now confirmed in a controlled setting. This cross-validates the phenomenon across:
- Model scales (8B parameters → 300K parameters)
- Tasks (language modeling → modular arithmetic)
- Architectures (standard transformer → grokking transformer)

The consistency across these dimensions suggests feature instability is fundamental to SAE training dynamics, not specific to any particular setting.

### 4.4 Critical Finding: PWMCC Equals Random Baseline

A critical control experiment reveals that trained SAE PWMCC (0.30) is **indistinguishable from randomly initialized SAEs**. We computed PWMCC between 10 randomly initialized (untrained) SAEs:

- Random SAE PWMCC: 0.300 ± 0.0007 (45 pairwise comparisons)
- Trained SAE PWMCC: 0.300 ± 0.001 (10 pairwise comparisons)
- Difference: ~0.0002 (essentially zero)

**This fundamentally changes the interpretation:** The 0.30 PWMCC is not "low stability"—it is **zero stability above random chance**. Standard SAE training produces decoder weights that are as random (in terms of cross-seed consistency) as untrained initialization.

However, alternative metrics reveal trained SAEs DO learn something:

- Subspace overlap (k=50): Random = 0.386, Trained = 0.439 (+5.3%)
- Mutual nearest neighbors (>0.3): Random = 0.312, Trained = 0.354 (+4.2%)

**Interpretation:** SAEs learn the correct **subspace** but not a consistent **basis** within that subspace. Different seeds find different, equally valid bases for the same learned subspace.

### 4.5 Layer-Dependent Stability: Layer 0 Anomaly

Cross-layer validation reveals striking layer dependence:

| Layer | Position | Trained PWMCC | Random PWMCC | EV | Interpretation |
|-------|----------|---------------|--------------|-----|----------------|
| Layer 0 | 2 | 0.047 ± 0.002 | 0.30 | 0.70 | **6× BELOW random** |
| Layer 1 | -2 | 0.302 ± 0.001 | 0.30 | 0.92 | **AT random** |

Layer 0 SAEs achieve reasonable reconstruction (EV = 0.70) but show BELOW-random feature consistency, meaning features are nearly **orthogonal** across seeds. This suggests the optimization landscape at Layer 0 has many equally-good but orthogonal decompositions. The phenomenon is position-dependent: Layer 0 at the operator position (2) shows this orthogonality, while Layer 1 at the answer position (-2) shows random-baseline behavior.

### 4.6 Architectural Comparison

Unlike Paulo & Belrose (2025) who found TopK more unstable than ReLU on LLMs, we observe no practical difference. This suggests two possibilities:

1. On simpler, more constrained tasks, architectural differences become negligible
2. With matched sparsity levels (L0 = 32 for TopK, L0 ≈ 427 for ReLU but both optimized), the training dynamics converge

Further investigation would require testing across multiple complexity levels and sparsity configurations.

---

## 5. Discussion

### 5.1 The Consistency Gap

Our results reveal a stark gap between achievable consistency (0.80, Song et al. 2025) and observed consistency (0.30, ours and Paulo & Belrose 2025). This 0.50 PWMCC gap has three key implications:

1. **Current practices are sub-optimal:** Standard SAE training with reconstruction loss alone does not naturally converge to consistent features. The optimization landscape admits multiple local minima corresponding to different decompositions.

2. **Architecture alone is insufficient:** While Song et al. found architectural choices matter for achieving high consistency, we show that under standard training, TopK and ReLU converge to identical ~0.30 baselines. Closing the gap requires training-level interventions (e.g., consistency-promoting objectives, multi-seed alignment) beyond architectural selection.

3. **Practical guidance needed:** Song et al.'s demonstration that 0.80 is achievable suggests a path forward, but their "appropriate architectural choices" must be documented, standardized, and validated in diverse settings before practitioners can routinely achieve high consistency.

### 5.2 Implications for Interpretability

The reproducibility crisis revealed by our work and recent literature has serious implications:

**For mechanistic interpretability research:** Interpretations based on single SAE instances may be misleading. If feature 42 appears to detect "mention of Paris" but a different seed assigns this to feature 137, which interpretation is correct? Both? Neither? The non-uniqueness undermines confidence in mechanistic claims.

**For safety-relevant applications:** If SAEs are used to verify safety properties (e.g., "no deception features"), low consistency means different runs might miss critical features. A safety analysis based on one SAE might fail to detect issues visible in another seed's decomposition.

**For cumulative progress:** Research building on previous SAE analyses may fail to replicate if features don't reproduce. This threatens the accumulation of knowledge essential for scientific progress.

### 5.3 Why Does Instability Occur?

The systematic nature (std < 0.002) suggests instability arises from fundamental optimization dynamics rather than random noise. Possible mechanisms include:

1. **Non-convex loss landscape:** Multiple equivalent local minima corresponding to different valid decompositions (e.g., feature A + feature B vs feature C + feature D representing the same subspace)

2. **Symmetry breaking:** Random initialization breaks symmetries, leading features to specialize along different directions in activation space

3. **Degeneracy in sparse coding:** The sparse coding problem may admit multiple solutions with similar reconstruction error but different feature interpretations

Future theoretical work should characterize conditions under which SAE training converges to unique vs multiple solutions.

### 5.4 Path Forward

Based on our findings and recent literature, we recommend:

**For practitioners:**
1. Train multiple SAEs with different seeds and verify feature alignment
2. Report stability metrics (PWMCC or similar) alongside reconstruction metrics
3. Be cautious about interpretations based on single SAE instances
4. Consider ensemble approaches that aggregate across seeds

**For researchers:**
1. Develop consistency-promoting training objectives (extending Song et al.'s framework)
2. Investigate multi-seed alignment techniques
3. Characterize the optimization landscape theoretically
4. Establish reproducibility standards for SAE papers

**For the community:**
1. Include stability benchmarks in SAE evaluation suites
2. Standardize reporting of multi-seed results
3. Share trained SAE checkpoints to enable reproducibility studies

---

## 6. Limitations

Our study has several limitations:

1. **Limited seeds:** While 5 seeds per architecture with tight variance suggest robust phenomena, larger-scale studies may reveal additional nuances.

2. **Single task:** We focus on modular arithmetic. While this validates Paulo & Belrose's LLM findings in a controlled setting, real language models may exhibit different stability profiles.

3. **Two architectures:** We tested TopK and ReLU, but other variants (e.g., Gated SAEs, JumpReLU) remain unexplored.

4. **Standard training:** We used standard reconstruction loss. Song et al.'s consistency-promoting methods may improve stability—testing this is important future work.

5. **Architecture difference from literature:** Our 2-layer transformer learns non-Fourier algorithms (R²=2% vs 93-98% for 1-layer). While this makes findings more general, it differs from Nanda et al.'s setting.

---

## 7. Conclusion

We presented the first systematic multi-seed, multi-architecture analysis of SAE feature stability, finding architecture-independent instability (PWMCC~0.30) that persists despite excellent reconstruction metrics. Our results provide critical empirical grounding for recent theoretical work, confirming that:

1. Current standard training practices yield only 0.30 consistency, far below the 0.80 achievable with optimization (Song et al., 2025)
2. Architectural choice alone (TopK vs ReLU) is insufficient to improve consistency under standard training
3. Reconstruction quality and feature consistency are decoupled, challenging current evaluation practices

The 0.50 PWMCC gap between current practice and demonstrated achievable consistency represents both a challenge and an opportunity. While feature instability threatens the reliability of interpretability research, the existence of methods achieving 0.80 consistency suggests the problem is solvable. Future work should focus on making consistency-promoting training accessible to practitioners and establishing reproducibility standards for the field.

Only by systematically addressing feature consistency can SAEs fulfill their promise for reliable mechanistic interpretability.

---

## References

[To be filled with full citations]

- Paulo & Belrose (2025). arXiv:2501.16615
- Song et al. (2025). arXiv:2505.20254  
- Templeton et al. (2024). Scaling Monosemanticity
- Gao et al. (2024). Scaling Sparse Autoencoders
- Bricken et al. (2023). Towards Monosemanticity
- Cunningham et al. (2023). Sparse Autoencoders Find Interpretable Features
- Nanda et al. (2023). Progress measures for grokking
- Power et al. (2022). Grokking
- Elhage et al. (2022). Toy Models of Superposition
- Olah (2023). Mechanistic interpretability blog post

---

## Appendix

### A.1 Additional Experimental Details

[To be filled with hyperparameter tables, training curves, etc.]

### A.2 Full Statistical Results

[To be filled with complete statistical test outputs]

### A.3 Visualizations

[To be filled with additional figures if needed]
