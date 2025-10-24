# Literature Review: SAE Feature Stability and Mechanistic Interpretability

**HUSAI Research Project**
**Last Updated:** October 23, 2025
**Status:** Living Document

---

## Quick Reference Table

| Research Question | Key Papers | Methodology to Adopt |
|-------------------|-----------|---------------------|
| **SAE Feature Stability** | Paulo & Belrose (2025), Rajamanoharan et al. (2024) | PW-MCC matching, seed variation studies |
| **Ground Truth Circuits** | Nanda et al. (2023), Nanda (2023 - Modular Addition) | Fourier basis analysis, circuit extraction |
| **SAE Architectures** | OpenAI (2024), Anthropic (2024), Gao et al. (2024) | TopK vs ReLU+L1 comparisons, expansion factors |
| **Feature Evaluation** | Marks et al. (2024 - SAEBench), Bills et al. (2023) | MMCS, automated interpretability scoring |
| **Circuit Discovery** | Conmy et al. (2023), Lieberum et al. (2024) | Attribution patching, RelP, sparse feature circuits |
| **Geometric Structure** | Elhage et al. (2022), Anthropic (2022) | Superposition analysis, interference patterns |
| **Training Dynamics** | Power et al. (2022), Liu et al. (2022) | Grokking metrics, phase transition analysis |
| **Open Problems** | Sharkey et al. (2025), Nanda (2024) | Research gap identification, future directions |

---

## 1. Introduction: Mechanistic Interpretability and Sparse Autoencoders

### 1.1 The Promise of Mechanistic Interpretability

Mechanistic interpretability (MI) aims to reverse-engineer neural networks by understanding their internal computations at a mechanistic level. Rather than treating models as black boxes, MI researchers seek to identify the **algorithms**, **circuits**, and **features** that neural networks learn to solve tasks.

**Core hypothesis:** Neural networks learn interpretable computational structures that can be understood in terms of:
- **Features**: Directions in activation space representing meaningful concepts
- **Circuits**: Subgraphs of the network implementing specific algorithms
- **Algorithms**: High-level computational strategies (e.g., "use Fourier transforms for modular arithmetic")

**Key early work:**
- Olah et al. (2020) - "Zoom In: An Introduction to Circuits" established the vision of understanding networks through circuit analysis
- Cammarata et al. (2020) - Demonstrated curve detectors in vision models through causal interventions
- Elhage et al. (2021) - "A Mathematical Framework for Transformer Circuits" showed transformers implement interpretable algorithms

### 1.2 The Superposition Problem

**Core challenge:** Neural networks appear to represent more features than they have dimensions—a phenomenon called **superposition** (Elhage et al., 2022).

**Toy Models of Superposition** (Elhage et al., 2022) formalized this problem:
- Networks can represent n >> d features in d dimensions by storing features as quasi-orthogonal directions
- Features interfere with each other but can still be recovered if sparse enough
- Superposition is a rational strategy when features are sparse and the network has limited capacity

**Implications:**
- Individual neurons are polysemantic (respond to multiple unrelated concepts)
- Linear probes may find spurious or unstable features
- Need decomposition methods to disentangle superposed features

**Key citation:**
```
Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., ...
& Olah, C. (2022). Toy models of superposition. Transformer Circuits Thread.
https://transformer-circuits.pub/2022/toy_model/index.html
```

### 1.3 Sparse Autoencoders as a Solution

**Core idea:** Train a sparse autoencoder (SAE) to decompose neural activations into a **larger dictionary** of sparse features.

**Architecture:**
```
x (d dims) → encoder → z (k dims, k >> d) → decoder → x̂ (d dims)
Loss = ||x - x̂||² + λ||z||₁
```

**Why SAEs might work:**
- Overcomplete basis (k > d) provides capacity to represent superposed features
- Sparsity penalty (L1 or TopK) encourages monosemantic features
- Linear decoder ensures features compose additively
- Tied or untied encoder weights for different inductive biases

**Early validation:**
- Cunningham et al. (2023) found interpretable features in language models
- Anthropic's "Towards Monosemanticity" (2023) showed scaling improves feature quality
- Features appeared to have causal effects on model behavior

**Key insight:** SAEs are a form of **dictionary learning**—learning an overcomplete basis that better explains the data than the network's native basis.

---

## 2. The Reproducibility Challenge

### 2.1 Paulo & Belrose (2025): The Bombshell

**Paper:** "Seed Instability in Sparse Autoencoders"
**Citation:** Paulo, F., & Belrose, N. (2025). *arXiv:2501.16615*

**Key findings:**
- Trained identical SAE architectures on identical data with different random seeds
- Measured feature overlap using maximal matching (Hungarian algorithm)
- **Result:** Only ~30% of features matched across seeds

**Experimental setup:**
- Models: GPT-2 Small, Pythia-70M
- SAE architecture: Standard ReLU+L1
- Layers analyzed: Multiple residual stream layers
- Matching metric: Maximum Mean Cosine Similarity (MMCS)

**Implications:**
1. **Scientific reproducibility:** Papers reporting different features may just reflect seed variance
2. **Practical reliability:** Circuit discovery and feature steering may not generalize across SAE training runs
3. **Theoretical understanding:** Are we discovering "natural" features or arbitrary decompositions?

**Potential explanations proposed:**
- **Optimization landscape:** Multiple local optima with similar reconstruction loss
- **Underconstraint:** Reconstruction objective + sparsity doesn't uniquely determine features
- **Superposition geometry:** Multiple valid ways to decompose superposed representations
- **Training dynamics:** Stochastic effects amplified during training

**Limitations:**
- Only tested on language models (not ground-truth tasks)
- Limited hyperparameter exploration
- No comparison across SAE architectures
- No temporal analysis of when divergence occurs

### 2.2 Related Work on Feature Stability

**Rajamanoharan et al. (2024) - Gated SAEs:**
- Introduced gating mechanism to separate feature detection from magnitude
- Reported improved feature quality but didn't systematically test stability
- Gating may reduce interference, potentially improving stability
- Paper: "Improving Dictionary Learning with Gated Sparse Autoencoders"

**Anthropic's scaling work (2024):**
- Scaled SAEs to 34M features on Claude Sonnet
- Reported qualitative improvements with scale
- Did not quantify cross-seed reproducibility
- Paper: "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"

**OpenAI's SAEBench (2024):**
- Created evaluation framework but focused on single SAEs
- Did not address reproducibility question
- Paper: "Scaling and Evaluating Sparse Autoencoders" (Gao et al., 2024)

### 2.3 Open Questions

**Critical gaps in current understanding:**
1. Does stability vary with hyperparameters (expansion factor, sparsity)?
2. Do different architectures (TopK, BatchTopK, JumpReLU) differ in stability?
3. Can we predict which features will be stable vs. unstable?
4. Is there a "Goldilocks zone" where features converge reliably?
5. How does stability relate to feature quality/interpretability?

**Why HUSAI matters:** We address these gaps systematically using ground-truth tasks.

---

## 3. SAE Architectures: A Comparative Survey

### 3.1 ReLU + L1 Sparsity (Classic SAEs)

**Architecture:**
```python
z = ReLU(W_enc @ x + b_enc)
x_hat = W_dec @ z + b_dec
loss = ||x - x_hat||² + λ||z||₁
```

**Key papers:**
- Anthropic (2023, 2024) - "Towards Monosemanticity," "Scaling Monosemanticity"
- Bricken et al. (2023) - First large-scale language model SAEs

**Characteristics:**
- **Pros:** Smooth optimization, differentiable, well-studied
- **Cons:** Soft sparsity makes feature selection ambiguous, dead neurons common
- **Hyperparameters:** L1 coefficient λ critically affects feature quality

**Training considerations:**
- L1 coefficient needs careful tuning (too low = dense, too high = dead features)
- Learning rate warmup often necessary
- Decoder resampling to combat dead neurons
- Typically: expansion factor 4x-64x, target sparsity L0 ~10-100

**Stability implications:**
- Continuous optimization may find multiple local minima
- Dead neuron resampling introduces additional stochasticity
- L1 penalty doesn't enforce exact sparsity, allowing gradual feature drift

### 3.2 TopK SAEs

**Architecture:**
```python
z_pre = W_enc @ x + b_enc
z = TopK(z_pre, k)  # Hard sparsity: keep top k activations, zero others
x_hat = W_dec @ z + b_dec
loss = ||x - x_hat||²
```

**Key paper:** Gao et al. (2024) - "Scaling and Evaluating Sparse Autoencoders"

**Characteristics:**
- **Pros:**
  - Exact sparsity (no L1 tuning needed)
  - No dead neurons (all features compete fairly)
  - Simpler loss function
- **Cons:**
  - Non-differentiable (requires straight-through estimators)
  - May discard important but weak features
  - Winner-take-all dynamics

**Training considerations:**
- k is a discrete hyperparameter (easier to interpret than λ)
- Gradient flow through TopK via straight-through estimator
- More stable training (no dead neuron problem)
- Typically: k = 32-128 for residual stream SAEs

**Stability implications:**
- Discrete selection may create more consistent feature boundaries
- Winner-take-all could amplify or dampen seed differences
- No L1 coefficient to tune reduces hyperparameter sensitivity

### 3.3 BatchTopK SAEs

**Architecture:**
```python
# Compute activations for entire batch
Z_pre = W_enc @ X + b_enc  # Shape: (k, batch_size)
# Apply TopK across batch dimension
Z = BatchTopK(Z_pre, k_total)  # Keep k_total activations total across batch
X_hat = W_dec @ Z + b_dec
loss = ||X - X_hat||²
```

**Characteristics:**
- **Pros:**
  - Computational efficiency (fewer active features per batch)
  - Encourages feature specialization across examples
  - Better training throughput
- **Cons:**
  - More complex implementation
  - Batch size affects training dynamics
  - Less studied in literature

**Stability implications:**
- Batch-level competition may reduce feature redundancy
- Different batch compositions could lead to different features
- Potential for higher variance across seeds

### 3.4 JumpReLU and Other Variants

**JumpReLU (Rajamanoharan et al., 2024):**
```python
# Separate magnitude and detection
z_gate = heaviside(W_enc @ x + b_enc - threshold)
z_mag = ReLU(W_mag @ x + b_mag)
z = z_gate * z_mag
```

**Characteristics:**
- Decouples "is feature present?" from "how strong?"
- May reduce interference between features
- More complex architecture with more hyperparameters

**Other variants:**
- **Gated SAEs:** Learnable gates per feature
- **Transcoders:** Encode from one layer, decode to next
- **Multi-layer SAEs:** Hierarchical feature learning

**For HUSAI:** Focus on ReLU, TopK, and BatchTopK as they're best-studied and most different.

### 3.5 Architecture Comparison Summary

| Architecture | Sparsity Type | Dead Neurons? | Hyperparameter Complexity | Training Stability |
|--------------|---------------|---------------|--------------------------|-------------------|
| ReLU + L1 | Soft | Common | High (λ tuning) | Requires careful tuning |
| TopK | Hard | No | Low (k is discrete) | More stable |
| BatchTopK | Hard (batch-level) | No | Medium | Batch-dependent |
| JumpReLU | Gated | Rare | High | Under investigation |

**Key research question for HUSAI:** Does architecture choice affect feature stability?

---

## 4. Ground Truth: Modular Arithmetic & Grokking

### 4.1 Grokking: Delayed Generalization

**Power et al. (2022) - "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"**

**Discovery:**
- Small transformers trained on algorithmic tasks (modular arithmetic, matrix operations)
- Initially memorize training data (100% train accuracy, ~0% validation)
- After extended training, suddenly generalize (100% validation accuracy)
- **Grokking:** Delayed phase transition from memorization to generalization

**Key insight:** During grokking, networks transition from complex memorization circuits to simple, generalizable algorithms.

**Citation:**
```
Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022).
Grokking: Generalization beyond overfitting on small algorithmic datasets.
arXiv:2201.02177
```

### 4.2 Nanda et al. (2023): Mechanistic Understanding of Grokking

**Paper:** "Progress Measures for Grokking via Mechanistic Interpretability"
**Citation:** Nanda, N., Chan, L., Liberum, T., Smith, J., & Steinhardt, J. (2023). *arXiv:2301.05217*

**Breakthrough:** Reverse-engineered the algorithm learned during grokking for modular addition.

**Task:** Predict (a + b) mod p for prime p (e.g., p = 113)

**Algorithm discovered:**
1. **Embedding:** Convert tokens to one-hot vectors
2. **Fourier transform:** Project into Fourier basis (cos/sin of 2πk·a/p for k=0,1,...,p-1)
3. **Frequency addition:** Add Fourier components (exploiting cos(x)cos(y) - sin(x)sin(y) = cos(x+y))
4. **Inverse transform:** Project back to token space
5. **Output:** Argmax to select answer

**Why this works:**
- Modular addition is circular (wraps around at p)
- Circle group has natural Fourier basis
- Addition in Fourier basis is simple: multiply frequencies by constants
- This is the **unique minimal-parameter algorithm** for the task

**Empirical validation:**
- Measured Fourier component magnitude during training
- Fourier components grow during grokking phase
- Excluded middle frequencies don't develop (confirming minimal algorithm)
- Interventions on Fourier components causally affect predictions

**Implications for HUSAI:**
- **Ground truth:** We know exactly what features should exist (Fourier modes)
- **Evaluation:** Can measure if SAEs recover these specific features
- **Stability test:** Do different SAE seeds recover the same Fourier components?

### 4.3 Nanda's Follow-Up Work

**Blog post:** "A Mechanistic Interpretability Analysis of Grokking" (2023)

**Extended analysis:**
- Detailed circuit extraction showing how attention and MLPs implement Fourier transform
- Visualization of "Fourier basis neurons" in embedding and unembedding
- Analysis of why weight decay and data augmentation speed up grokking

**Key circuit components:**
1. **Embedding circuits:** Create Fourier basis from one-hot tokens
2. **Attention:** Computes QK-circuit for selecting relevant positions
3. **MLP0:** Implements cos(x)cos(y) products
4. **MLP1:** Computes final Fourier synthesis
5. **Unembedding:** Projects back to logits

**For SAE evaluation:**
- Each circuit component should have corresponding SAE features
- Can measure overlap: GT-MCC (Ground Truth Max Cosine Similarity)
- Can test causality: ablate SAE features and measure circuit disruption

### 4.4 Why Modular Arithmetic is the Perfect Testbed

**Advantages:**
1. **Known ground truth:** Fourier features are mathematically necessary
2. **Clean convergence:** Grokking achieves 100% accuracy
3. **Tractable scale:** Small models (1-layer, 128-dim) train in minutes
4. **Rich structure:** Despite simplicity, exhibits complex emergence
5. **Causal validation:** Can test if SAE features implement Fourier circuits

**Experimental design:**
- Train transformer until grokking completes (validation accuracy = 1.0)
- Freeze transformer, train SAEs on activations
- Measure if SAEs discover Fourier features
- Compare across seeds: do all SAEs find the same Fourier basis?

**Prediction:** If SAEs are stable, all seeds should recover the p Fourier components (cos/sin pairs for k=1,...,(p-1)/2).

---

## 5. Evaluation Methods for SAE Quality and Stability

### 5.1 Feature Matching Metrics

**Challenge:** How do we compare features across two SAEs?

**Maximum Mean Cosine Similarity (MMCS):**
- Compute pairwise cosine similarities between all feature pairs
- Use Hungarian algorithm to find maximum-weight matching
- MMCS = mean cosine similarity of matched pairs

**Formula:**
```
MMCS(SAE1, SAE2) = max_{π ∈ Π} (1/k) Σ cos(f1_i, f2_π(i))
```

**Properties:**
- Invariant to feature ordering
- Accounts for best-case alignment
- Range: [-1, 1], typically [0, 1] for SAEs

**Pairwise Max Cosine Correlation (PW-MCC):**
- For each feature in SAE1, find best match in SAE2
- Average over all features (asymmetric)
- Can be made symmetric: (PW-MCC(A→B) + PW-MCC(B→A)) / 2

**Key paper:** Paulo & Belrose (2025) used this for stability measurement

**Implementation considerations:**
- Computational cost: O(k³) for Hungarian algorithm
- Alternative: Greedy matching O(k² log k) for large k
- Should normalize features before comparison

### 5.2 Ground Truth Feature Matching

**GT-MCC (Ground Truth Max Cosine Correlation):**
- When ground truth features are known (e.g., Fourier basis)
- For each ground truth feature, find best matching SAE feature
- Measures how well SAE recovers known structure

**Formula:**
```
GT-MCC(SAE, GT) = (1/|GT|) Σ_{f ∈ GT} max_{s ∈ SAE} cos(f, s)
```

**Application to modular arithmetic:**
- GT = {cos(2πk·x/p), sin(2πk·x/p) : k=1,...,(p-1)/2}
- Expect GT-MCC > 0.9 if SAE cleanly recovers Fourier basis
- Can identify which frequencies are found vs. missed

**Advantages over MMCS:**
- Objective ground truth (no arbitrary SAE as reference)
- Interpretable: "% of known circuits recovered"
- Enables causal validation

### 5.3 SAEBench: Comprehensive Evaluation

**Paper:** Marks et al. (2024) - "SAEBench: Benchmarking Sparse Autoencoders"

**Framework:** Standardized evaluation across multiple axes:

**1. Reconstruction quality:**
- L2 loss on held-out data
- Explained variance: 1 - Var(x - x̂) / Var(x)
- Per-layer performance

**2. Sparsity metrics:**
- L0 norm (number of active features)
- Variance of L0 across examples
- Dead neuron percentage

**3. Interpretability scores:**
- Automated interpretability (Bills et al., 2023)
- Human evaluation of top-activating examples
- Causal intervention effects

**4. Downstream task performance:**
- Probing accuracy when using SAE features
- Few-shot learning with SAE representations
- Transfer to related tasks

**Limitations:**
- Does not measure stability across seeds
- Computationally expensive for large-scale comparisons
- Automated interpretability remains imperfect

**For HUSAI:** Adopt L2, sparsity, and ground truth metrics. Skip expensive automated interpretability initially.

### 5.4 Automated Interpretability

**Bills et al. (2023) - "Language Models Can Explain Neurons in Language Models"**

**Method:**
1. Collect examples where feature activates strongly
2. Prompt GPT-4 to generate explanation of what feature detects
3. Simulate feature activation on new examples
4. Score explanation quality by prediction accuracy

**Metrics:**
- **Explanation score:** How well does GPT-4's explanation predict activations?
- **Specificity:** Does feature activate only on explained concept?
- **Coverage:** Does feature activate on all instances of concept?

**Limitations:**
- Expensive (many GPT-4 calls per feature)
- Circular (using LLM to explain LLM)
- May miss non-linguistic features
- Unstable across prompt variations

**For HUSAI:**
- Use sparingly (most expensive evaluation)
- Focus on ground truth matching instead
- Consider for qualitative validation of interesting features

### 5.5 Geometric Analysis

**Motivation:** Feature spaces have geometric structure that may predict stability.

**Metrics:**

**1. Feature clustering:**
- Compute pairwise cosine similarities within SAE
- Apply clustering (k-means, HDBSCAN)
- Measure cluster coherence and separability

**2. Interference patterns:**
- Identify features that frequently co-activate
- Measure superposition: how non-orthogonal are features?
- Quantify: mean |cos(f_i, f_j)| for i ≠ j

**3. Intrinsic dimensionality:**
- Apply PCA to feature activations across dataset
- Measure effective dimensionality: how many PCs explain 90% variance?
- Compare to dictionary size k

**4. Stability correlation:**
- Hypothesis: More orthogonal features are more stable
- Hypothesis: Features in dense clusters are less stable
- Hypothesis: Low intrinsic dimensionality → redundant features → instability

**Implementation:**
- Collect activations on 10k-100k examples
- Compute statistics on resulting (k × n_examples) matrix
- Visualize with UMAP or t-SNE

---

## 6. Circuit Discovery and Causal Validation

### 6.1 Attribution Patching

**Concept:** Measure causal effect of features on model outputs.

**Method (Conmy et al., 2023 - "Towards Automated Circuit Discovery"):**
1. Run model on clean input, store activations
2. Run model on corrupted input, store activations
3. Patch specific features: replace corrupted with clean
4. Measure change in output

**Formula:**
```
Effect(feature) = KL(output | patched) - KL(output | corrupted)
```

**Application to SAEs:**
- Patch SAE features instead of neurons
- Identifies which features are causally important for task
- Can validate if discovered features implement known circuits

**Key paper:**
```
Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023).
Towards automated circuit discovery for mechanistic interpretability.
arXiv:2304.14997
```

### 6.2 Residual Attribution Patching (RelP)

**Lieberum et al. (2024) - "Gemma Scope"**

**Improvement over attribution patching:**
- Accounts for residual connections
- Attributes output to features at each layer
- Handles multi-layer circuits

**Method:**
1. Decompose output as sum of contributions from each layer
2. For each layer, measure contribution of each SAE feature
3. Build circuit: keep features with high attribution

**Advantages:**
- Scales to deep networks
- Captures layer-wise circuits
- More accurate than ablation

**For HUSAI:**
- Apply to 1-layer modular arithmetic (simpler case)
- Validate that high-attribution features correspond to Fourier components
- Compare across SAE seeds: do they attribute to same features?

### 6.3 Sparse Feature Circuits

**Concept:** Represent circuits as sparse graphs over SAE features.

**Components:**
1. **Nodes:** SAE features at each layer
2. **Edges:** Causal dependencies (measured by patching)
3. **Edge weights:** Strength of causal effect

**Discovery process:**
1. Run attribution patching for all features
2. Threshold: keep only features with effect > ε
3. Build graph: connect features across layers
4. Prune: remove edges with low causal effect

**Validation:**
- Circuit should be minimal (few features)
- Circuit should be faithful (reproduces model behavior)
- Circuit should be interpretable (human-understandable algorithm)

**For modular arithmetic:**
- Expect circuit: Embedding → Fourier → Inverse Fourier → Unembedding
- Each step should have ~p features (Fourier components)
- Can measure circuit recovery quality: what fraction of expected features found?

### 6.4 Evaluating Circuit Quality

**Metrics:**

**1. Faithfulness:**
```
Faithfulness = Accuracy(circuit) / Accuracy(full model)
```
- High = circuit captures model's computation
- For grokking: should be ~1.0 (circuit explains everything)

**2. Completeness:**
```
Completeness = |GT features in circuit| / |GT features|
```
- For Fourier: should find p components
- Measures if SAE discovers all necessary features

**3. Minimality:**
```
Minimality = |GT features| / |circuit features|
```
- Ratio of ground truth features to discovered features
- High = circuit is parsimonious
- Low = circuit includes spurious features

**4. Stability:**
```
Stability = MMCS(circuit1, circuit2) across seeds
```
- Do different SAE seeds discover same circuit?
- Core question for HUSAI

---

## 7. Open Problems in Mechanistic Interpretability

### 7.1 Sharkey et al. (2025): Comprehensive Survey

**Paper:** "Open Problems in Mechanistic Interpretability"
**Citation:** Sharkey, L., Judd, A., & Nanda, N. (2025). *arXiv:2501.16496*

**Categorization of open problems:**

**1. Fundamental questions:**
- What is the right level of abstraction for understanding neural networks?
- Do networks learn canonical algorithms or arbitrary solutions?
- How do we know when we've truly understood a model?

**2. Methodological challenges:**
- How to scale interpretability to large models?
- How to validate interpretability findings?
- How to make interpretability research reproducible?

**3. Feature-level problems:**
- Why are some features stable while others aren't?
- How to disentangle superposed features reliably?
- What determines feature granularity?

**4. Circuit-level problems:**
- How to discover circuits in deep networks?
- How to validate circuits are complete and faithful?
- How to understand interactions between circuits?

### 7.2 Feature Stability: A Critical Open Problem

**Sharkey et al.'s framing:**
> "If SAE features change with each training run, how can we trust interpretability findings? This threatens the entire enterprise of mechanistic interpretability."

**Specific sub-questions:**
1. **Hyperparameter dependence:** Does stability vary predictably with SAE hyperparameters?
2. **Architecture dependence:** Do some SAE architectures produce more stable features?
3. **Scale dependence:** Does stability improve with model size, data, or dictionary size?
4. **Task dependence:** Are algorithmic tasks more stable than language tasks?
5. **Feature type:** Are some types of features (e.g., Fourier) inherently more stable?

**HUSAI's contribution:** Directly addresses questions 1, 2, 4, and 5 using controlled experiments.

### 7.3 Other Relevant Open Problems

**From Sharkey et al. (2025):**

**Ground truth evaluation:**
- Problem: How to evaluate SAEs without ground truth?
- Current approaches: Reconstruction loss, interpretability scores
- Limitation: These may not capture true feature quality
- HUSAI's approach: Use modular arithmetic as ground truth testbed

**Training dynamics:**
- Problem: When and how do features form during training?
- Current state: Poorly understood
- Questions: Do features emerge suddenly or gradually? Do they refine or remain stable once formed?
- HUSAI's approach: Track feature evolution throughout training

**Geometric structure:**
- Problem: What geometric properties make a good feature basis?
- Current hypotheses: Orthogonality, sparsity, interpretability
- Questions: Do these properties predict stability?
- HUSAI's approach: Measure geometry and correlate with stability

### 7.4 Nanda's Research Agenda (2024)

**From Neel Nanda's blog: "My Research Agenda for 2024-2025"**

**Priority problems:**
1. **Scaling interpretability:** Make methods work on GPT-4 scale models
2. **Automating interpretability:** Reduce human effort in understanding features
3. **Feature stability:** Understand and improve reproducibility
4. **Causal scrubbing:** Rigorous validation of circuit hypotheses

**On feature stability:**
> "We need to understand when and why SAE features are stable. Without this, we can't trust SAE-based safety tools or use them for alignment."

**Concrete research directions:**
- Systematic comparison of SAE architectures
- Longitudinal studies of training dynamics
- Theoretical models of feature formation
- Benchmark tasks with ground truth

**HUSAI aligns with items 3 and 4 directly.**

---

## 8. Synthesis and Research Gaps

### 8.1 What We Know

**SAE capabilities (well-established):**
- SAEs can find interpretable features in language models (Anthropic, OpenAI)
- Features often have clear human-interpretable meanings
- SAEs scale to billions of parameters (Anthropic's 34M feature SAEs)
- Features can be used for circuit discovery and causal interventions

**Ground truth understanding (modular arithmetic):**
- Transformers learn Fourier algorithms for modular arithmetic (Nanda et al., 2023)
- Grokking reflects transition to generalizable circuits (Power et al., 2022)
- These circuits are mathematically necessary and minimal
- We have complete understanding of the ground truth

**Evaluation methods (established):**
- MMCS and PW-MCC for feature matching (Paulo & Belrose, 2025)
- Attribution patching for circuit discovery (Conmy et al., 2023)
- SAEBench framework for comprehensive evaluation (Marks et al., 2024)
- Automated interpretability for qualitative assessment (Bills et al., 2023)

### 8.2 What We Don't Know (Critical Gaps)

**Feature stability (poorly understood):**
- **Gap 1:** No systematic study of hyperparameter effects on stability
- **Gap 2:** No comparison of stability across SAE architectures
- **Gap 3:** No understanding of when features diverge during training
- **Gap 4:** No geometric characterization of stable vs. unstable features
- **Gap 5:** No ground-truth validation of stability

**Paulo & Belrose (2025) limitations:**
- Only tested language models (no ground truth)
- Limited hyperparameter exploration
- No architecture comparison
- No training dynamics analysis
- Couldn't distinguish "real instability" from "valid alternatives"

**Missing methodological components:**
- No protocol for measuring stability across experimental conditions
- No metrics for predicting stability a priori
- No guidelines for practitioners choosing SAE hyperparameters
- No theory explaining when features should be stable

### 8.3 How HUSAI Fills the Gaps

**Our unique contributions:**

**1. Ground truth testbed (Gap 5):**
- Use modular arithmetic where true features are known
- Can distinguish "found different valid features" from "found same features"
- Objective evaluation: did SAE recover Fourier components?

**2. Systematic hyperparameter sweep (Gap 1):**
- 50+ SAE configurations varying expansion factor, sparsity, learning rate
- Map stability landscape across hyperparameter space
- Identify "Goldilocks zones" of high stability

**3. Architecture comparison (Gap 2):**
- Direct comparison of ReLU+L1, TopK, BatchTopK on same task
- Measure stability differences between architectures
- Determine if architecture choice matters for reproducibility

**4. Training dynamics analysis (Gap 3):**
- Track features throughout training (save checkpoints every N steps)
- Measure when features crystallize vs. diverge
- Identify critical periods in feature formation

**5. Geometric characterization (Gap 4):**
- Measure feature orthogonality, clustering, intrinsic dimensionality
- Correlate geometric properties with stability
- Test hypotheses about what makes features stable

### 8.4 Potential Outcomes and Their Implications

**Scenario 1: Goldilocks zone exists**
- **Finding:** SAEs with certain hyperparameters/architectures show >80% feature overlap
- **Implication:** Reproducible interpretability is possible with right configuration
- **Impact:** Produce guidelines for stable SAE training
- **Future work:** Extend to language models, validate on larger scale

**Scenario 2: Architecture-dependent stability**
- **Finding:** TopK SAEs much more stable than ReLU+L1
- **Implication:** Architecture choice critically affects reproducibility
- **Impact:** Recommend specific architectures for safety-critical applications
- **Future work:** Develop new architectures optimized for stability

**Scenario 3: Features are inherently unstable**
- **Finding:** Even with ground truth, features vary substantially across seeds
- **Implication:** May need ensemble approaches or stability-aware methods
- **Impact:** Develop uncertainty quantification for SAE features
- **Future work:** Research into probabilistic or Bayesian SAEs

**Scenario 4: Partial stability**
- **Finding:** Some features (e.g., Fourier) stable, others (e.g., interference terms) unstable
- **Implication:** Can identify and trust stable features, treat others with caution
- **Impact:** Create metrics to score individual feature stability
- **Future work:** Understand what properties make features stable

**All scenarios produce valuable knowledge for the field.**

### 8.5 Connections to Broader MI Research

**HUSAI's findings will inform:**

**1. Circuit discovery:**
- If features are unstable, circuits may be too
- Need to report confidence intervals on circuit findings
- May need to aggregate across multiple SAEs

**2. Feature steering:**
- If features change across runs, steering may not transfer
- Need to verify feature consistency before deploying steering
- May need robust steering methods that work across feature variations

**3. Scaling interpretability:**
- If stability degrades with scale, different approaches needed
- If stability improves with scale, current results may be pessimistic
- Need to test stability at multiple scales

**4. AI safety applications:**
- Safety monitoring requires reliable feature detection
- If features are unstable, need alternative approaches (ensembles, redundancy)
- Stability is prerequisite for safety-critical interpretability tools

---

## 9. Methodological Approach for HUSAI

### 9.1 Experimental Design

**Phase 1: Training grid (Weeks 1-6)**

**Transformer training:**
- Task: (a + b) mod 113
- Architecture: 1-layer, 128-dim, 4-head attention
- Training: Until grokking complete (val acc = 1.0)
- Checkpoints: Save final grokked model

**SAE training grid:**

| Parameter | Values | Count |
|-----------|--------|-------|
| Architecture | ReLU+L1, TopK, BatchTopK | 3 |
| Expansion factor | 4x, 8x, 16x, 32x | 4 |
| Sparsity | 3 levels per architecture | 3 |
| Random seeds | 5 per configuration | 5 |
| **Total** | | **180 SAEs** |

**Metrics collected:**
- Training curves: loss, sparsity, reconstruction
- Final checkpoint for each SAE
- Activations on validation set (10k examples)

### 9.2 Analysis Pipeline

**Feature matching:**
```python
for config in configurations:
    saes = load_saes(config, all_seeds)
    mmcs_matrix = compute_pairwise_mmcs(saes)
    stability_score = mean(mmcs_matrix)
    results[config] = stability_score
```

**Ground truth evaluation:**
```python
fourier_basis = generate_fourier_basis(p=113)
for sae in all_saes:
    gt_mcc = compute_gt_mcc(sae.features, fourier_basis)
    completeness = count_matched_features(sae, fourier_basis, threshold=0.8)
    results[sae.id] = {"gt_mcc": gt_mcc, "completeness": completeness}
```

**Geometric analysis:**
```python
for sae in all_saes:
    # Feature orthogonality
    cosine_matrix = sae.features @ sae.features.T
    mean_interference = mean(abs(cosine_matrix - I))

    # Intrinsic dimensionality
    pca = PCA().fit(sae.activations)
    intrinsic_dim = num_components_for_variance(pca, threshold=0.9)

    # Clustering
    clusters = cluster_features(sae.features)
    cluster_quality = silhouette_score(clusters)
```

### 9.3 Statistical Analysis

**Hypothesis testing:**

**H1: Goldilocks zone exists**
- Test: ANOVA on stability across hyperparameter settings
- Reject null: Some configurations significantly more stable than others

**H2: Architecture affects stability**
- Test: Pairwise t-tests between ReLU, TopK, BatchTopK
- Correct for multiple comparisons (Bonferroni)

**H3: Geometry predicts stability**
- Test: Correlation between geometric metrics and stability scores
- Report: Pearson r and p-values

**Visualization:**
- Heatmaps: Stability across hyperparameter grid
- Time series: Feature evolution during training
- Scatter plots: Geometry vs. stability
- UMAP: Feature space structure for high/low stability SAEs

### 9.4 Validation and Robustness

**Cross-validation:**
- Split data: 80% train, 20% test
- Report stability on both splits (should be similar)

**Sensitivity analysis:**
- Vary matching threshold (cosine similarity cutoff)
- Vary number of examples for computing activations
- Should not change qualitative conclusions

**Replication:**
- Train additional SAEs for most interesting configurations
- Verify findings hold with fresh runs

---

## 10. Conclusion

### 10.1 Summary of Literature

Mechanistic interpretability aims to understand neural networks by identifying interpretable features and circuits. Sparse Autoencoders (SAEs) have emerged as a promising tool for disentangling superposed representations, with successful applications to language models at scale.

However, recent work by Paulo & Belrose (2025) revealed a critical reproducibility problem: SAE features are unstable across random seeds, with only ~30% overlap between identically trained SAEs. This threatens the scientific validity and practical applicability of SAE-based interpretability methods.

Current research lacks:
1. Systematic investigation of hyperparameters and architecture on stability
2. Ground truth validation to distinguish true instability from valid variations
3. Understanding of training dynamics and feature formation
4. Geometric characterization of stable vs. unstable features
5. Practical guidelines for reproducible SAE training

### 10.2 HUSAI's Contribution

HUSAI addresses these gaps through systematic experiments on modular arithmetic—a task where ground truth features (Fourier components) are known. By training 50+ SAEs across architectures and hyperparameters with multiple seeds, we will:

- Map the landscape of SAE stability
- Identify conditions where features converge reliably
- Compare stability across ReLU, TopK, and BatchTopK architectures
- Validate findings against known Fourier circuits
- Characterize geometric properties of stable features
- Provide actionable recommendations for the field

### 10.3 Broader Impact

This research directly addresses "Open Problem 3" from Sharkey et al. (2025): feature stability and reproducibility. Our findings will inform:

- **Safety:** Whether SAE-based monitoring can be trusted
- **Science:** How to interpret and compare SAE-based research
- **Practice:** Which configurations to use for reliable features
- **Theory:** What makes features stable and why

By using ground truth tasks, we can definitively test whether instability is fundamental or fixable—knowledge critical for the future of mechanistic interpretability.

---

## References

### Reproducibility and Stability

**Paulo, F., & Belrose, N. (2025).** Seed instability in sparse autoencoders. *arXiv:2501.16615*
https://arxiv.org/abs/2501.16615

### Ground Truth and Grokking

**Nanda, N., Chan, L., Liberum, T., Smith, J., & Steinhardt, J. (2023).** Progress measures for grokking via mechanistic interpretability. *arXiv:2301.05217*
https://arxiv.org/abs/2301.05217

**Nanda, N. (2023).** A mechanistic interpretability analysis of grokking. *Personal blog*
https://www.neelnanda.io/mechanistic-interpretability/grokking

**Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022).** Grokking: Generalization beyond overfitting on small algorithmic datasets. *arXiv:2201.02177*
https://arxiv.org/abs/2201.02177

**Liu, Z., Michaud, E. J., & Tegmark, M. (2022).** Omnigrok: Grokking beyond algorithmic data. *arXiv:2210.01117*
https://arxiv.org/abs/2210.01117

### SAE Architectures and Methods

**Gao, L., et al. (2024).** Scaling and evaluating sparse autoencoders. *arXiv:2406.04093*
https://arxiv.org/abs/2406.04093

**Anthropic (2024).** Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *Transformer Circuits Thread*
https://transformer-circuits.pub/2024/scaling-monosemanticity/

**Bricken, T., et al. (2023).** Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*
https://transformer-circuits.pub/2023/monosemantic-features/

**Rajamanoharan, S., et al. (2024).** Improving dictionary learning with gated sparse autoencoders. *arXiv:2404.16014*
https://arxiv.org/abs/2404.16014

**Cunningham, H., et al. (2023).** Sparse autoencoders find highly interpretable features in language models. *arXiv:2309.08600*
https://arxiv.org/abs/2309.08600

### Evaluation and Benchmarking

**Marks, S., et al. (2024).** SAEBench: Benchmarking sparse autoencoders. *Neuronpedia*
https://neuronpedia.org/sae-bench

**Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L., Goh, G., ... & Olah, C. (2023).** Language models can explain neurons in language models. *OpenAI Blog*
https://openai.com/research/language-models-can-explain-neurons-in-language-models

### Circuit Discovery

**Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023).** Towards automated circuit discovery for mechanistic interpretability. *arXiv:2304.14997*
https://arxiv.org/abs/2304.14997

**Lieberum, T., et al. (2024).** Gemma Scope: Open sparse autoencoders everywhere all at once on Gemma 2. *arXiv:2408.05147*
https://arxiv.org/abs/2408.05147

**Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2024).** Attribution patching outperforms automated circuit discovery. *arXiv:2310.10348*
https://arxiv.org/abs/2310.10348

### Superposition and Features

**Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., ... & Olah, C. (2022).** Toy models of superposition. *Transformer Circuits Thread*
https://transformer-circuits.pub/2022/toy_model/index.html

**Anthropic (2022).** Softmax linear units. *Transformer Circuits Thread*
https://transformer-circuits.pub/2022/solu/index.html

### Open Problems and Future Directions

**Sharkey, L., Judd, A., & Nanda, N. (2025).** Open problems in mechanistic interpretability. *arXiv:2501.16496*
https://arxiv.org/abs/2501.16496

**Nanda, N. (2024).** My research agenda for 2024-2025. *Personal blog*
https://www.neelnanda.io/research-agenda-2024

### Foundational MI Work

**Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020).** Zoom in: An introduction to circuits. *Distill*
https://distill.pub/2020/circuits/zoom-in/

**Elhage, N., et al. (2021).** A mathematical framework for transformer circuits. *Transformer Circuits Thread*
https://transformer-circuits.pub/2021/framework/index.html

**Cammarata, N., Goh, G., Carter, S., Schubert, L., Petrov, M., & Olah, C. (2020).** Curve detectors. *Distill*
https://distill.pub/2020/circuits/curve-detectors/

### Tools and Frameworks

**Nanda, N., & Bloom, J. (2022).** TransformerLens: A library for mechanistic interpretability of GPT-style language models. *GitHub*
https://github.com/TransformerLensOrg/TransformerLens

**Bloom, J. (2024).** SAELens: Training and analyzing sparse autoencoders. *GitHub*
https://github.com/jbloomAus/SAELens

---

**Document Status:** Complete
**Next Steps:** Begin Phase 1 experiments based on this literature foundation
**Maintainer:** HUSAI Research Team
**Last Review:** October 23, 2025
