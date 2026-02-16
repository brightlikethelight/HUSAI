# HUSAI: Hunting for Stable AI Features

**Tagline**: *Uncovering the reproducibility crisis in sparse autoencoders and finding the path to stable, interpretable AI*

---

## The Problem Statement

### The January 2025 Bombshell

In January 2025, Paulo and Belrose dropped a bombshell on the mechanistic interpretability community: **SAE features are unstable**. When you train the same sparse autoencoder (SAE) architecture twice with different random seeds, only ~30% of features overlap between runs.

Think about what this means. The entire premise of mechanistic interpretability research relies on SAEs discovering "natural" features in neural networks—the building blocks of computation that we can understand and analyze. But if these features change drastically with every training run, what are we actually discovering? Are we uncovering genuine structure in neural networks, or just finding one of many equally valid decompositions?

### Why This Matters

This isn't just an academic curiosity. The stability of SAE features has profound implications for:

**AI Safety**: If we can't reliably identify features across training runs, how can we monitor for dangerous capabilities? How can we build robust interpretability tools that work consistently?

**Scientific Validity**: Can we trust results from mechanistic interpretability papers if features aren't reproducible? Are we building a house of cards?

**Practical Applications**: Circuit discovery, feature steering, and mechanistic understanding all depend on having stable, meaningful features. Without stability, these applications become unreliable.

### Current State of the Field

The interpretability community is at a crossroads. We have powerful tools (SAEs, dictionary learning) and exciting results (finding circuits, understanding features), but we lack fundamental knowledge about when these tools actually work. Key open questions include:

- Do SAEs converge to stable features under certain conditions?
- How do different SAE architectures (ReLU, TopK, BatchTopK) compare in stability?
- What role does the training dynamics play in feature formation?
- Can we predict which features will be stable vs. unstable?

We need systematic, controlled experiments to answer these questions. That's where HUSAI comes in.

---

## Our Research Questions

HUSAI investigates five core questions about SAE feature stability:

### 1. The Goldilocks Zone Hypothesis
**Does there exist a "sweet spot" in hyperparameter space where SAEs converge to stable, reproducible features?**

We hypothesize that feature stability isn't binary—it exists on a spectrum controlled by factors like:
- Dictionary size (expansion factor)
- Sparsity coefficient (L1 penalty strength)
- Learning rate and training duration
- Model capacity vs. data complexity

By systematically varying these parameters, we aim to map the landscape of feature stability and identify regions where reproducibility is high.

### 2. Architecture Comparison
**How do different SAE architectures compare in terms of feature stability and quality?**

We'll compare three major architecture families:
- **ReLU-based SAEs**: The classic approach with L1 sparsity
- **TopK SAEs**: Hard sparsity via top-k activation selection
- **BatchTopK SAEs**: Batch-level top-k for training efficiency

Each architecture makes different trade-offs between sparsity, reconstruction quality, and computational efficiency. We'll measure how these trade-offs affect stability.

### 3. Circuit Recovery as Ground Truth
**Can we reliably recover known computational circuits (specifically, Fourier features in modular arithmetic)?**

Using Nanda et al.'s grokking framework, we know that transformers learn interpretable Fourier features when solving modular arithmetic (e.g., a + b mod p). This gives us ground truth: we *know* what features should exist.

By training SAEs on grokked models, we can ask:
- Do different SAE seeds discover the same Fourier features?
- How clean is the recovery (one feature per Fourier component)?
- Does stability correlate with circuit clarity?

### 4. Geometric Feature Space Structure
**What is the geometric organization of SAE feature spaces, and how does it relate to stability?**

Features don't exist in isolation—they form a geometric structure in activation space. We'll investigate:
- **Feature clustering**: Do stable features cluster together?
- **Interference patterns**: How do overlapping features affect stability?
- **Dimensionality**: What's the intrinsic dimensionality of stable vs. unstable feature manifolds?
- **Conservation laws**: Are certain geometric properties preserved across seeds?

### 5. Temporal Dynamics of Feature Formation
**When during SAE training do features crystallize into stable patterns vs. diverge across seeds?**

We'll track feature evolution throughout training to understand:
- Do features start similar and diverge, or start different and sometimes converge?
- Are there critical periods where stability is determined?
- Can we detect early warning signs of instability?
- Does feature formation mirror neural network training dynamics (e.g., progressive differentiation)?

---

## Our Approach

### Ground Truth Testbed: Modular Arithmetic

Rather than diving into the complexity of language models, we start with a **controllable, well-understood domain**: transformers trained on modular arithmetic tasks.

**Why modular arithmetic?**
- **Known ground truth**: We know exactly what features should exist (Fourier components)
- **Perfect grokking**: Models achieve 100% accuracy and learn interpretable circuits
- **Tractable scale**: Small models (1-layer, 128d) that train in minutes
- **Rich structure**: Despite simplicity, exhibits complex phase transitions and emergence

This testbed lets us ask clean questions with definitive answers before tackling messier domains.

### Systematic Experimentation: 50+ SAE Training Runs

We'll train **50+ SAE variants** with systematic parameter sweeps:

**Architecture variations:**
- 3 SAE types (ReLU, TopK, BatchTopK)
- 3-5 expansion factors (4x, 8x, 16x, 32x, 64x)
- 3-5 sparsity levels (L1 coefficients or k values)

**Training variations:**
- 5+ random seeds per configuration
- Multiple training durations (early stopping vs. convergence)
- Different learning rates and schedules

**Measurement protocol:**
- Track training metrics (loss, sparsity, reconstruction)
- Compute pairwise feature overlap across seeds
- Measure Fourier feature recovery quality
- Analyze geometric properties of feature spaces

### Three-Phase Methodology

#### Phase 1: Controlled Experiments (Weeks 1-6)
**Goal**: Map the stability landscape

- Train SAE grid across architecture × hyperparameter space
- Collect comprehensive metrics on all runs
- Identify high-stability and low-stability regions
- Generate initial hypotheses about stability factors

**Deliverables**:
- 50+ trained SAE checkpoints
- Stability heatmaps across parameter space
- Initial feature overlap statistics

#### Phase 2: Deep Analysis (Weeks 7-12)
**Goal**: Understand mechanisms of stability/instability

- Perform detailed geometric analysis of feature spaces
- Track temporal dynamics of feature formation
- Compare Fourier feature recovery across conditions
- Investigate failure modes and edge cases

**Deliverables**:
- Feature clustering visualizations
- Training dynamics plots
- Circuit recovery quality metrics
- Mechanistic hypotheses for stability

#### Phase 3: Synthesis & Validation (Weeks 13-16)
**Goal**: Consolidate findings and create actionable guidelines

- Synthesize results into coherent narrative
- Validate key findings with targeted experiments
- Develop practical recommendations for SAE training
- Document limitations and future directions

**Deliverables**:
- Final research report
- Open-source analysis toolkit
- Training best practices guide
- Conference-ready presentation

---

## Success Criteria

### Minimum Viable Success (Must Achieve)
✅ **Reproducibility**: Successfully train 50+ SAEs with comprehensive metric logging
✅ **Baseline measurement**: Quantify feature overlap across seeds for all architectures
✅ **Ground truth validation**: Demonstrate SAEs can recover Fourier features in at least one configuration
✅ **Team learning**: All members gain hands-on experience with SAE training and analysis
✅ **Open source release**: Publish clean, documented codebase and dataset

### Target Success (Primary Goals)
🎯 **Goldilocks zone identification**: Find hyperparameter regions with >60% feature overlap
🎯 **Architecture ranking**: Definitively compare stability across ReLU/TopK/BatchTopK
🎯 **Mechanistic insight**: Identify at least 2-3 clear factors that predict stability
🎯 **Temporal understanding**: Characterize feature formation dynamics across training
🎯 **Practical guidelines**: Produce actionable recommendations for choosing SAE hyperparameters

### Stretch Goals (Reach Objectives)
🚀 **Novel theoretical insight**: Develop mathematical framework for predicting stability
🚀 **Beyond modular arithmetic**: Validate findings on a second domain (e.g., tiny language models)
🚀 **Causal interventions**: Experimentally manipulate stability factors to test hypotheses
🚀 **Publication potential**: Results worthy of ML conference workshop or interpretability venue
🚀 **Tool impact**: Create stability analysis tools adopted by interpretability community

---

## Team Structure

### Core Team (3 Members)

**Research Lead**
- Responsibilities: Experimental design, hypothesis generation, paper writing
- Time commitment: 10-15 hours/week
- Skills needed: ML research experience, interpretability background, scientific writing

**Infrastructure Engineer**
- Responsibilities: Training pipeline, experiment tracking, cloud compute management
- Time commitment: 10-15 hours/week
- Skills needed: Python, PyTorch, distributed training, MLOps

**Analysis Specialist**
- Responsibilities: Data analysis, visualization, geometric analysis, metrics
- Time commitment: 10-15 hours/week
- Skills needed: Python, data science, linear algebra, plotting libraries

### Collaboration Model
- **Weekly sync**: 1-hour team meeting to review progress and plan next steps
- **Pair programming**: Infrastructure + Analysis work together on pipeline
- **Research reviews**: Lead reviews all analysis before conclusions
- **Rotating documentation**: All members contribute to writing and documentation

### Timeline: 16 Weeks (1 Semester)

**Weeks 1-2**: Setup & pilot experiments
**Weeks 3-6**: Phase 1 experiments (controlled grid search)
**Weeks 7-9**: Initial analysis & hypothesis refinement
**Weeks 10-12**: Phase 2 experiments (deep dives)
**Weeks 13-14**: Final analysis & synthesis
**Weeks 15-16**: Documentation, presentation, open-source release

### Resource Requirements

**Compute**:
- Cloud provider: AWS or GCP
- Estimated cost: $200-500 for semester
- GPU requirements: T4 or V100 level (modular arithmetic trains fast)
- Storage: ~50GB for checkpoints and results

**Software**:
- PyTorch, TransformerLens, SAELens
- Weights & Biases for experiment tracking
- Jupyter for analysis
- GitHub for version control

**Knowledge**:
- Papers: Paulo & Belrose 2025, Nanda grokking, SAE literature
- Tutorials: TransformerLens documentation, SAE training guides
- Community: EleutherAI, Alignment Forum, interpretability Slack channels

---

## Expected Impact

### Learning Outcomes (Team Development)

**Technical Skills**:
- Deep understanding of SAE architectures and training dynamics
- Hands-on experience with transformer mechanistic interpretability
- Proficiency in experiment design and scientific computing
- Data analysis and visualization for high-dimensional spaces

**Research Skills**:
- Formulating testable hypotheses from open-ended questions
- Systematic experimentation and ablation studies
- Reproducing and extending published research
- Scientific writing and presentation

**Domain Expertise**:
- Mechanistic interpretability methods and challenges
- Neural network training dynamics and optimization
- Sparse coding and dictionary learning theory
- AI safety considerations in interpretability

### Contributions to the Field

**Empirical Evidence**:
- First systematic study of SAE stability across architecture and hyperparameter space
- Quantitative benchmarks for feature reproducibility
- Validation (or refutation) of the "Goldilocks zone" hypothesis

**Methodological Advances**:
- Established protocols for measuring feature stability
- Ground-truth testing framework for SAE evaluation
- Metrics for feature quality beyond reconstruction loss

**Practical Value**:
- Actionable guidelines for researchers training SAEs
- Identification of stable vs. unstable parameter regimes
- Tool for predicting whether an SAE configuration will yield reproducible features

### Open-Source Deliverables

**Code Release**:
- `husai-training`: SAE training pipeline with multiple architectures
- `husai-analysis`: Feature stability analysis toolkit
- `husai-viz`: Visualization tools for high-dimensional feature spaces

**Datasets**:
- Trained SAE checkpoints across parameter space
- Feature overlap matrices and stability metrics
- Grokked transformer models for reproducibility

**Documentation**:
- Comprehensive experimental protocol
- Best practices guide for SAE training
- Tutorial notebooks for replicating analysis

**Knowledge Sharing**:
- Blog posts explaining key findings
- Open-source everything with MIT license
- Presentation materials for sharing with community

---

## Key References

### Foundational Papers

**SAE Stability & Reproducibility**:
- Paulo & Belrose (2025). "Seed Instability in Sparse Autoencoders" - *The paper that started it all*
- Cunningham et al. (2023). "Sparse Autoencoders Find Highly Interpretable Features in Language Models"

**Modular Arithmetic & Grokking**:
- Nanda et al. (2023). "Progress Measures for Grokking via Mechanistic Interpretability" - *Our ground truth*
- Power et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"

**SAE Architectures**:
- Gao et al. (2024). "Scaling and Evaluating Sparse Autoencoders" (TopK SAEs)
- Rajamanoharan et al. (2024). "Improving Dictionary Learning with Gated Sparse Autoencoders"
- Anthropic (2024). "Scaling Monosemanticity: Extracting Interpretable Features" (ReLU SAEs)

**Mechanistic Interpretability Background**:
- Elhage et al. (2022). "Toy Models of Superposition" - *Why we need SAEs*
- Olah et al. (2020). "Zoom In: An Introduction to Circuits" - *The vision for interpretability*
- Nanda & Bloom (2022). "TransformerLens: A Library for Mechanistic Interpretability" - *Our primary tool*

### Tools & Frameworks

- **TransformerLens**: Easy access to transformer internals
- **SAELens**: SAE training and analysis library
- **Weights & Biases**: Experiment tracking and visualization
- **PyTorch**: Deep learning framework

### Community Resources

- **Alignment Forum**: Discussion of interpretability research
- **EleutherAI Discord**: Active interpretability research community
- **Neel Nanda's Blog**: Tutorials and insights on mech interp
- **Anthropic Research**: Leading work on SAEs and feature interpretation

---

## Vision: Where We're Going

Imagine a future where mechanistic interpretability is a **reliable, reproducible science**. Where we can:

- Train an SAE and trust that key features will appear consistently
- Compare results across papers knowing they used comparable methods
- Build safety tools knowing they'll work on tomorrow's models, not just today's
- Understand the fundamental structure of neural network representations

HUSAI is a small but crucial step toward that future. By understanding when and why SAE features stabilize, we're laying the groundwork for interpretability research that scales—not just to bigger models, but to bigger dreams.

This semester, we're not just running experiments. We're building the foundation for trustworthy AI understanding.

**Let's hunt for stable features. Let's make interpretability reproducible. Let's build the future of AI safety.**

---

*HUSAI: Hunting for Stable AI Features*
*Harvard Undergraduate AI Safety Initiative*
*Spring 2025*

---

## Getting Started

New to the team? Start here:

1. **Read this document** - You just did! ✓
2. **Review the research plan**: `docs/04-Execution/research-plan.md`
3. **Set up your environment**: `docs/04-Execution/setup-guide.md`
4. **Run the tutorial**: `notebooks/00-getting-started.ipynb`
5. **Join the team meeting**: Check calendar for weekly sync time

Questions? Reach out to the Research Lead or post in the team channel.

**Welcome to HUSAI. Let's make AI interpretable together.**
