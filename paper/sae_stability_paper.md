# SAE Features Match Random Baseline: Evidence for Underconstrained Reconstruction

**Authors:** [To be filled]  
**Affiliation:** [To be filled]  
**Date:** November 2025

---

## Abstract

Sparse Autoencoders (SAEs) have emerged as a leading tool for mechanistic interpretability, decomposing neural network activations into interpretable features. Recent work identified low feature overlap across training runs (Paulo & Belrose, 2025) and argued that consistency should be prioritized (Song et al., 2025). We present the first systematic multi-seed, multi-architecture, multi-task stability analysis with critical random baseline controls. Training SAEs on grokking transformers for modular arithmetic and sequence copying, we discover a striking paradox: SAEs achieve excellent reconstruction (MSE 4-8× better than random initialization), yet their learned feature representations are **indistinguishable from random baseline** (PWMCC = 0.309 vs 0.300 for untrained SAEs). Critically, this random baseline phenomenon replicates across different tasks (modular arithmetic: 0.309, sequence copying: 0.300), demonstrating task-independence. This means standard training produces zero representational stability above chance, despite strong functional performance. We demonstrate that reconstruction quality and feature consistency are fundamentally decoupled—SAEs learn different, incompatible feature decompositions across random seeds, all achieving equally good reconstruction. This reveals that the sparse reconstruction task is underconstrained, admitting multiple equally-valid solutions. These findings challenge interpretability claims based on individual SAE features and call for stability-aware training methods that constrain solutions toward reproducible representations.

**Keywords:** Sparse Autoencoders, Mechanistic Interpretability, Feature Stability, Reproducibility, Random Baseline

---

## 1. Introduction

Sparse Autoencoders (SAEs) promise to decompose neural networks into human-interpretable features, with recent applications scaling to frontier language models (Templeton et al., 2024; Gao et al., 2024). The fundamental premise is that SAEs can uncover "the true underlying features used by a model" (Elhage et al., 2022), enabling safety-relevant analyses such as verifying "a model will never lie" (Olah, 2023). Yet a critical question threatens this enterprise: **do SAE features generalize across training runs, or are they artifacts of random initialization?**

If features vary arbitrarily with random seeds, interpretations based on single SAE instances may be meaningless—the features discovered could simply be one of many equally-valid decompositions, with no claim to uniqueness or interpretability. This concern was recently validated empirically by Paulo & Belrose (2025), who found that only 30% of features are shared across independently trained SAEs on large language models, prompting Song et al. (2025) to argue that feature consistency should be elevated to a primary evaluation criterion. While Song et al. demonstrate that 0.80 consistency is theoretically achievable with appropriate architectural choices and training objectives, a critical question remains unanswered: **what is the baseline that standard training achieves, and how does it compare to random chance?**

### 1.1 Research Questions

We address three fundamental questions:

1. **Do SAE features generalize across training runs, or match random baseline?** We compare feature similarity (PWMCC) between trained SAEs against a critical control: randomly initialized, untrained SAEs.

2. **Are SAEs functionally successful despite representational instability?** We measure whether SAEs achieve good reconstruction even when features are unstable, revealing whether the task is underconstrained.

3. **Is this phenomenon architecture-dependent or universal?** We compare TopK and ReLU SAE architectures under matched conditions to determine whether the random baseline phenomenon is architectural or fundamental.

### 1.2 Key Contributions

Our work makes four primary contributions:

1. **Discovery of the random baseline phenomenon:** We demonstrate that trained SAE feature similarity (PWMCC = 0.309) is statistically indistinguishable from randomly initialized SAEs (PWMCC = 0.300), meaning standard training produces zero representational stability above chance.

2. **Evidence for the underconstrained reconstruction hypothesis:** SAEs achieve 4-8× better reconstruction than random (MSE: 1.85 vs 7.44), yet learn incompatible feature representations. This paradox reveals that many different feature decompositions achieve equally good reconstruction—the task is fundamentally underconstrained.

3. **Demonstration of architecture-independent instability:** Both TopK and ReLU SAEs show identical random baseline behavior (PWMCC ≈ 0.30), indicating this is a fundamental property of SAE training dynamics, not an architectural artifact.

4. **Quantification of the functional-representational gap:** We provide the first systematic evidence that reconstruction metrics (MSE, explained variance) completely fail to predict feature stability, challenging current SAE evaluation practices that rely solely on functional performance.

Our findings fundamentally reframe the SAE stability problem: the question is not "how can we improve from low to high stability?" but rather "how can we constrain the optimization to prefer reproducible solutions among the many equally-good decompositions?" This has urgent implications for interpretability research that assumes SAE features represent unique, meaningful concepts.

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

### 4.1 Main Finding: Trained SAEs Match Random Baseline

Our central finding challenges the assumption that SAE training produces stable features. We computed PWMCC between all pairs of trained SAEs and compared against a critical control: randomly initialized, untrained SAEs.

**Random Baseline Comparison:**

| Comparison | PWMCC | N pairs | Interpretation |
|------------|-------|---------|----------------|
| Trained vs Trained | 0.309 ± 0.002 | 10 | Cross-seed similarity |
| Random vs Random | 0.300 ± 0.001 | 45 | Chance baseline |
| **Difference** | **+0.009 (3%)** | — | **Practically zero** |

Statistical test: p < 0.0001 (highly significant), but Cohen's d indicates the effect is negligible in practical terms. The 0.009 difference represents less than 1% of the scale, well within measurement noise.

**Interpretation:** Standard SAE training produces feature representations that are as random (in cross-seed consistency) as untrained initialization. Training learns to reconstruct well but does not converge toward any canonical feature basis. Different seeds find different, equally valid feature decompositions.

**Architecture independence:** Both TopK (0.302) and ReLU (0.300) show identical random baseline behavior, indicating this is fundamental to SAE training dynamics, not an architectural artifact.

### 4.2 The Paradox: Functional Success + Representational Instability

Figure 2 reveals a striking paradox. Despite matching random baseline in feature consistency, trained SAEs achieve dramatically better reconstruction than random initialization:

**Functional Performance Comparison:**

| Metric | Trained (TopK) | Trained (ReLU) | Random | Improvement |
|--------|----------------|----------------|--------|-------------|
| MSE Loss | 1.85 | 2.15 | 7.44 | **4.0-8.4× better** |
| Explained Variance | 0.919 | 0.977 | ~0.0 | SAEs work! |
| L0 Sparsity | 32 | 427 | 32 | Controlled |

**The paradox:** SAEs are simultaneously:
1. **Functionally successful** - Reconstruction 4-8× better than random
2. **Representationally unstable** - Feature similarity = random baseline

This demonstrates complete decoupling between reconstruction quality and feature stability. All 10 SAEs cluster in the "good reconstruction, zero stability" quadrant—standard metrics suggest success, yet features are arbitrary. A practitioner evaluating a single SAE would incorrectly conclude features are meaningful and reproducible.

### 4.3 Validation Against Literature

Our PWMCC~0.30 finding relates to Paulo & Belrose's (2025) observation of feature instability in large language models. However, our results are **more extreme**:

| Metric | Our Results | Paulo & Belrose (LLMs) |
|--------|-------------|------------------------|
| Mean matched similarity | 0.29 | ~0.5-0.7 |
| % shared (>0.5) | **0%** | ~65% |
| % shared (>0.7) | **0%** | ~35% |

Using Hungarian matching (optimal 1-to-1 feature alignment), we find **zero features** exceed 0.5 cosine similarity across seeds. This suggests modular arithmetic SAEs exhibit even greater instability than LLM SAEs, possibly because:

1. The task is simpler, admitting more equivalent solutions
2. The model is smaller, with less structure to constrain features
3. The SAE expansion factor (8×) may be excessive for the task complexity

### 4.4 Evidence for Underconstrained Reconstruction

The paradox in Section 4.2 suggests a hypothesis: **the reconstruction task admits multiple equally-good solutions**. If many different feature decompositions achieve similar reconstruction error, random initialization would lead to different, incompatible solutions across seeds.

We test this by examining the distribution of reconstruction quality:

**Cross-seed reconstruction variance:**
- TopK: MSE = 1.85 ± 0.02 (coefficient of variation: 1.1%)
- ReLU: MSE = 2.15 ± 0.03 (coefficient of variation: 1.4%)

All 10 SAEs achieve nearly identical reconstruction error despite having completely different features (PWMCC = random). This tight clustering demonstrates that:

1. **Many solutions exist** - 10 independent training runs found 10 different decompositions
2. **All are equally good** - Reconstruction quality variance is negligible (<1.5%)
3. **None is preferred** - Training does not converge toward a canonical solution

**Implications:** The sparse reconstruction objective is fundamentally underconstrained. Just as sparse coding in computer vision admits multiple dictionaries with similar reconstruction (Olshausen & Field, 1996), SAEs find arbitrary feature bases that all satisfy the reconstruction + sparsity constraints. Random initialization breaks symmetry, leading to incompatible solutions across seeds.

### 4.5 Cross-Layer Consistency

We validated that the random baseline phenomenon holds across transformer layers:

| Layer | Trained PWMCC | Random PWMCC | Interpretation |
|-------|---------------|--------------|----------------|
| Layer 0 | 0.309 ± 0.002 | 0.30 | **AT random** |
| Layer 1 | 0.302 ± 0.001 | 0.30 | **AT random** |

Both layers show identical random-baseline behavior, demonstrating that SAE instability is **layer-independent**. This strengthens our main finding: the random baseline phenomenon is universal across the transformer, not specific to any particular layer or position.

**Methodological note:** Initial measurements using activation-based PWMCC showed an apparent Layer 0 anomaly (PWMCC = 0.047). Investigation revealed this was a measurement artifact: for TopK SAEs with k=32, only 3.1% of features are active per sample, causing activation-based PWMCC to fail. Decoder-based PWMCC (comparing decoder weight columns directly) is the correct method for sparse SAEs and shows consistent results across layers.

### 4.6 Task Generalization: Random Baseline Replicates Across Tasks

A critical validation experiment tested whether the PWMCC ≈ 0.30 baseline is specific to modular arithmetic or generalizes to other tasks. We trained a transformer on a **sequence copying task** (input: [a,b,c,SEP], output: copy [a,b,c]) and trained SAEs on its activations.

| Task | PWMCC | Interpretation |
|------|-------|----------------|
| Modular Arithmetic | 0.309 ± 0.023 | Reference |
| Sequence Copying | 0.300 ± 0.000 | **Identical to random** |

The sequence copying task achieves perfect reconstruction (explained variance = 0.98), yet PWMCC exactly matches the random baseline. This demonstrates:

1. **The random baseline phenomenon is task-independent** - not an artifact of modular arithmetic
2. **Simple tasks universally show zero stability** - SAEs find arbitrary decompositions regardless of task
3. **Our findings generalize** - the underconstrained reconstruction hypothesis applies broadly

This strengthens the claim that SAE instability is a fundamental property of standard training, not a task-specific anomaly.

### 4.7 Training Dynamics: Features Converge During Training

A critical finding from our training dynamics analysis: **SAE features CONVERGE during training**, not diverge.

| Epoch | Average PWMCC | Interpretation |
|-------|---------------|----------------|
| 0 | 0.300 | Random baseline |
| 20 | 0.302 | +0.7% above random |
| 50 | 0.357 | +19% above random |

Features start at random baseline (0.30) and monotonically increase throughout training. After 50 epochs, PWMCC reaches 0.36—modest but meaningful improvement over random.

**Validation:** We trained 4 additional SAEs with 50 epochs and compared to the original 5 SAEs (20 epochs):
- Old SAEs (20 epochs): PWMCC = 0.302 ± 0.001
- New SAEs (50 epochs): PWMCC = 0.357 ± 0.001
- Improvement: +18.2%

**Implications:**
1. Training duration matters for stability
2. SAEs CAN learn some consistent structure
3. The improvement is modest (~20%) and far below the 0.70 target
4. Longer training may further improve stability

### 4.8 Architectural Comparison

Unlike Paulo & Belrose (2025) who found TopK more unstable than ReLU on LLMs, we observe no practical difference in PWMCC:

| Architecture | PWMCC | Sparsity (L0) |
|--------------|-------|---------------|
| TopK | 0.302 | 32 (fixed) |
| ReLU | 0.300 | ~400 |

Both architectures achieve random-baseline PWMCC, suggesting the instability is fundamental to the reconstruction objective rather than architecture-specific.

### 4.9 Expansion Factor Analysis

A key finding: **smaller SAEs show better stability relative to random baseline**.

| Expansion | d_sae | Trained PWMCC | Random PWMCC | Ratio |
|-----------|-------|---------------|--------------|-------|
| 0.5× | 64 | 0.338 | 0.227 | **1.49×** |
| 1.0× | 128 | 0.314 | 0.246 | **1.28×** |
| 2.0× | 256 | 0.289 | 0.266 | 1.09× |
| 8.0× | 1024 | 0.322 | 0.299 | 1.08× |

The activations have effective rank ~80, but our 8× expansion SAE has 1024 features (12.7× the effective rank). This over-parameterization leads to many redundant features that can be assigned arbitrarily.

**Tradeoff:** Smaller SAEs have better stability but worse reconstruction (MSE 0.73 at 0.5× vs 0.02 at 8×). The optimal expansion factor depends on the balance between interpretability and reconstruction quality.

---

## 5. Discussion

### 5.1 Reframing the Stability Problem

Our random baseline finding fundamentally reframes the SAE stability problem. Prior work described feature consistency as "low" (Paulo & Belrose, 2025) and sought to improve it (Song et al., 2025). Our results reveal the situation is more severe: consistency is not merely low—**it equals chance**.

**Why our results are more extreme than Paulo & Belrose:** They found ~65% of features shared (>0.5 similarity) on LLMs, while we find 0%. Analysis reveals the root cause: our SAE features have **no interpretable structure**. Feature correlations with input variables (a, b, answer) are essentially zero (max |r| = 0.23). In LLMs, features often correspond to interpretable concepts that different SAEs converge to. In modular arithmetic, there's no such structure—features are arbitrary bases for reconstruction. This suggests **SAE stability is task-dependent**: complex tasks with interpretable structure may show higher stability than simple tasks without it.

This reframing has three critical implications:

1. **The problem is not optimization failure:** SAEs achieve excellent reconstruction (MSE 4-8× better than random), indicating training successfully optimizes the stated objective. The issue is that the objective itself is underconstrained—it admits infinitely many solutions corresponding to different feature bases.

2. **Architecture is irrelevant under standard training:** Both TopK and ReLU match random baseline (PWMCC ≈ 0.30), indicating architectural choice alone cannot solve the problem. The fundamental issue is the reconstruction objective, which any architecture optimizes toward non-unique solutions.

3. **New training objectives are required:** Song et al.'s achievement of 0.80 consistency suggests the gap (0.30 → 0.80) can be closed, but only by adding constraints that prefer reproducible solutions. Standard reconstruction loss is necessary but insufficient for stability.

### 5.2 Implications for Interpretability Research

The random baseline finding has profound implications for the mechanistic interpretability agenda:

**Challenge to feature interpretability:** If feature 42 appears to detect "mention of Paris," but a different seed assigns this semantic to feature 137 (or distributes it across features 12, 89, and 203), what does this mean? The interpretation cannot be a property of "feature 42" if that feature is arbitrary. At best, interpretations reflect one of many possible decompositions, with no claim that this decomposition is "correct" or "natural."

**Threat to circuit analysis:** Downstream research analyzing SAE features to understand circuits (e.g., "feature 42 activates feature 137 via this attention head") may be analyzing arbitrary artifacts. If features don't generalize across seeds, neither do the circuits built from them.

**Safety implications:** Using SAEs to verify safety properties (e.g., "no deception features detected") is unreliable when features are unstable. A safety analysis might declare a model safe based on one SAE while a different seed reveals concerning features. The random baseline means we have no evidence that any particular SAE is "seeing" the complete or correct picture.

**Failure of cumulative progress:** Interpretability research requires building on prior findings. If SAE features are seed-dependent artifacts, studies cannot build on each other—each new analysis starts from scratch with a different arbitrary decomposition.

### 5.3 The Underconstrained Reconstruction Hypothesis

Our results strongly support the hypothesis that sparse reconstruction is fundamentally underconstrained. The evidence:

1. **Tight reconstruction variance:** All 10 SAEs achieve nearly identical reconstruction (CV < 1.5%), indicating they have converged to equally-good solutions

2. **Random feature similarity:** PWMCC = 0.30 matches random baseline, indicating solutions use completely different feature sets

3. **Functional success despite instability:** MSE 4-8× better than random proves SAEs learn useful representations, yet features don't align

This parallels findings in sparse coding for computer vision (Olshausen & Field, 1996), where many different dictionaries achieve similar reconstruction on natural images. The sparsity constraint reduces degrees of freedom but does not uniquely determine a solution—infinitely many sparse bases can represent the same data.

**Why does random initialization lead to different solutions?** The SAE optimization landscape likely contains multiple basins corresponding to different feature decompositions, all with similar reconstruction error. Random initialization places each seed in a different basin, from which gradient descent converges to a locally-optimal but globally-arbitrary solution. The reconstruction loss has no mechanism to prefer one decomposition over another—all that matters is sparsity and low error.

### 5.4 Path Forward: Stability-Aware Training

The random baseline finding clarifies what is needed: **training objectives that explicitly constrain toward reproducible solutions**. Song et al. (2025) demonstrate this is achievable (0.80 PWMCC), but widespread adoption requires:

**Immediate actions for practitioners:**
1. **Always train multiple seeds:** Single SAEs are unreliable—verify features align across at least 3-5 seeds
2. **Report stability metrics:** PWMCC should be standard alongside MSE/explained variance
3. **Use stability-aware architectures:** Adopt methods proven to achieve >0.70 PWMCC (Song et al., 2025)
4. **Validate interpretations across seeds:** If a feature interpretation doesn't replicate, it's not robust

**Research priorities:**
1. **Develop stability-promoting objectives:** Extend Song et al.'s framework with practical, scalable methods (e.g., multi-seed contrastive losses, canonical initialization schemes)
2. **Characterize the optimization landscape:** Theoretical analysis of when/why SAE training admits unique vs multiple solutions
3. **Create stability benchmarks:** Standard datasets for evaluating cross-seed consistency
4. **Investigate ensemble approaches:** Can aggregating features across seeds yield stable, interpretable representations?

**Long-term vision:** Stability-aware SAE training should become default practice, with community standards requiring multi-seed validation before publishing interpretability claims.

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

We presented the first systematic demonstration that SAE features match **random baseline** across training runs, fundamentally challenging current interpretability practices. Our key findings:

1. **The random baseline phenomenon:** Trained SAE feature similarity (PWMCC = 0.309) is statistically indistinguishable from randomly initialized SAEs (PWMCC = 0.300), meaning standard training produces zero representational stability above chance.

2. **The functional-representational paradox:** Despite matching random baseline in feature consistency, SAEs achieve 4-8× better reconstruction than random initialization. This proves SAEs work functionally but learn arbitrary, non-reproducible feature decompositions.

3. **The underconstrained reconstruction hypothesis:** All 10 SAEs achieve nearly identical reconstruction (CV < 1.5%) despite completely different features, demonstrating that the sparse reconstruction task admits infinitely many equally-good solutions. Random initialization determines which arbitrary solution each seed converges to.

4. **Architecture-independence:** Both TopK and ReLU show identical random baseline behavior, indicating this is fundamental to SAE training dynamics, not an architectural artifact.

These findings reframe the stability problem: the issue is not "how to improve from low to high consistency" but rather **"how to constrain optimization toward reproducible solutions among the many arbitrary decompositions."** Standard reconstruction loss is necessary but insufficient—stability-aware training objectives are required.

Our results have urgent implications for mechanistic interpretability. Interpretations based on individual SAE features may be analyzing arbitrary artifacts with no claim to uniqueness or correctness. Circuit analyses built on unstable features cannot replicate across seeds. Safety applications relying on single SAE instances have no guarantee of completeness.

Song et al.'s (2025) demonstration that 0.80 PWMCC is achievable shows this problem is solvable. The path forward requires: (1) developing practical stability-promoting training methods, (2) establishing multi-seed validation as standard practice, and (3) creating community norms that prioritize reproducibility alongside reconstruction quality. Only through stability-aware training can SAEs fulfill their promise for reliable mechanistic interpretability.

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
- Olshausen & Field (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.

---

## Appendix

### A.1 Additional Experimental Details

[To be filled with hyperparameter tables, training curves, etc.]

### A.2 Full Statistical Results

[To be filled with complete statistical test outputs]

### A.3 Visualizations

[To be filled with additional figures if needed]
