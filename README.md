# HUSAI: Hunting for Stable AI Features

> **Investigating the reproducibility crisis in sparse autoencoders and finding the path to stable, interpretable AI**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Foundation_Complete_(20%25)-yellow)
![Phase](https://img.shields.io/badge/Phase-Implementation_Starting-blue)

---

## 🎯 The Problem

In January 2025, a bombshell paper revealed that **Sparse Autoencoders (SAEs) trained on identical data with different random seeds learn entirely different features** — with only ~30% overlap. This is like taking two identical MRI machines, scanning the same brain, and getting completely different images.

If the features we're finding are just artifacts of our measurement process rather than real properties of neural networks, then the entire field of mechanistic interpretability might be built on shaky ground.

**Our mission:** Determine whether there's a "Goldilocks zone" where SAEs actually converge to stable, meaningful features.

---

## 🔬 Our Approach

We're taking a systematic, ground-truth-based approach:

### Phase 0: Foundation ✅ (Week 1 - COMPLETE)
- Production-ready modular arithmetic dataset with 2 token formats
- Pydantic configuration system with validation
- 85 passing tests
- Comprehensive documentation

### Phase 1: Implementation 🔄 (Weeks 2-4 - IN PROGRESS)
- Transformer model + training loop
- SAE architectures (ReLU, TopK, BatchTopK)
- W&B experiment tracking
- Initial validation experiments

### Phase 2: Controlled Experiments 📋 (Weeks 5-10 - PLANNED)
- Train **50+ SAEs** on modular arithmetic tasks where we know the "right answer" (Fourier transforms)
- Systematically vary: random seeds, architectures (ReLU, TopK, BatchTopK), sparsity levels, widths
- Track complete training trajectories — when do features crystallize vs diverge?

### Phase 3: Deep Analysis 📋 (Weeks 11-16 - PLANNED)
- Measure feature consistency across seeds using state-of-the-art matching algorithms (PW-MCC, MMCS)
- Test whether SAEs recover known Fourier circuits
- Explore geometric structure of learned feature spaces

### Phase 4: Building Better Tools 📋 (Weeks 17-20 - PLANNED)
- If Goldilocks zone exists: create guidelines for reproducible SAE training
- If features remain unstable: develop metrics to characterize uncertainty
- Open-source everything with clean, reusable code

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11
- CUDA 11.8 or 12.1 (for GPU support)
- conda/miniconda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/brightlikethelight/HUSAI.git
   cd HUSAI
   ```

2. **Create conda environment** (recommended)
   ```bash
   conda env create -f environment.yml
   conda activate husai
   ```

   Or manually install dependencies:
   ```bash
   # Create environment
   conda create -n husai python=3.11
   conda activate husai

   # Install PyTorch with CUDA 11.8
   conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia

   # Install remaining dependencies
   pip install -r requirements.txt
   ```

3. **Set up development tools**
   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   python -c "import transformer_lens; import sae_lens; print('TransformerLens and SAELens loaded successfully')"
   ```

---

## ✅ Current Capabilities (What Works Now)

**You can currently:**
- ✅ Generate modular arithmetic datasets (`src/data/modular_arithmetic.py`)
  ```python
  from src.data.modular_arithmetic import create_dataloaders
  train_loader, test_loader = create_dataloaders(modulus=113, batch_size=512)
  ```
- ✅ Create and validate experiment configurations (`src/utils/config.py`)
  ```python
  from src.utils.config import ExperimentConfig
  config = ExperimentConfig.from_yaml("configs/examples/baseline_relu.yaml")
  ```
- ✅ Run 85 passing tests on implemented modules
  ```bash
  make test  # All tests pass!
  ```

**Coming in Weeks 2-4 (Implementation Phase):**
- 🔄 Transformer model + training loop
- 🔄 SAE architectures (ReLU, TopK, BatchTopK)
- 🔄 Training scripts and experiment pipelines
- 🔄 Feature analysis and matching tools

See [`IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md) for detailed timeline.

---

## 📁 Project Structure

```
HUSAI/
├── README.md                 # You are here
├── docs/                     # Comprehensive documentation
│   ├── 00-Foundations/      # Mission, vision, research questions
│   ├── 01-Strategy/         # Research strategy
│   ├── 02-Product/          # Technical specifications
│   ├── 03-Go-To-Market/     # Dissemination plans
│   ├── 04-Execution/        # Implementation details
│   └── ADRs/                # Architectural Decision Records
├── src/                      # Source code
│   ├── data/                # Dataset generation and loading
│   ├── models/              # Model architectures
│   ├── training/            # Training loops and SAE implementations
│   ├── analysis/            # Feature matching and evaluation
│   └── utils/               # Utility functions
├── tests/                    # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── e2e/                 # End-to-end tests
├── notebooks/                # Jupyter notebooks for exploration
├── scripts/                  # Utility scripts
├── data/                     # Data directory (gitignored)
├── results/                  # Results directory (gitignored)
├── checkpoints/              # Model checkpoints (gitignored)
├── environment.yml           # Conda environment specification
├── requirements.txt          # Core Python dependencies
├── requirements-dev.txt      # Development dependencies
├── pyproject.toml            # Python project configuration
├── Makefile                  # Common commands
└── .pre-commit-config.yaml   # Pre-commit hooks configuration
```

---

## 🎓 Key Research Questions

1. **Goldilocks Zone Hypothesis:** Does a sweet spot exist where SAEs reliably converge to stable features?

2. **Architecture Comparison:** How do ReLU, TopK, and BatchTopK SAEs compare in stability and interpretability?

3. **Circuit Recovery as Ground Truth:** Can SAEs recover the known Fourier circuits in modular arithmetic?

4. **Geometric Structure:** What is the geometric organization of feature spaces, and does it predict stability?

5. **Temporal Dynamics:** When during training do features crystallize versus diverge?

---

## 🛠️ Technical Stack

### Core Libraries
- **[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)** (v2.16.1) - Model internals access
- **[SAELens](https://github.com/jbloomAus/SAELens)** (v6.17.0) - SAE training and analysis
- **PyTorch** (v2.5.1) - Deep learning framework
- **Weights & Biases** - Experiment tracking

### Analysis & Visualization
- NumPy, SciPy, pandas - Numerical computing
- scikit-learn - Matching algorithms (Hungarian)
- matplotlib, seaborn, plotly - Visualization
- UMAP - Dimensionality reduction

### Development
- pytest - Testing framework
- black, isort, flake8, mypy - Code quality
- pre-commit - Git hooks
- JupyterLab - Interactive development

---

## 📚 Key References

### Foundational Papers
- **Paulo & Belrose (2025)** - "SAEs Trained on the Same Data Learn Different Features" ([arXiv:2501.16615](https://arxiv.org/abs/2501.16615))
- **Nanda et al. (2023)** - "Progress measures for grokking via mechanistic interpretability" ([arXiv:2301.05217](https://arxiv.org/abs/2301.05217))
- **Sharkey et al. (2025)** - "Open Problems in Mechanistic Interpretability" ([arXiv:2501.16496](https://arxiv.org/abs/2501.16496))

### SAE Methods
- **OpenAI (2024)** - "Scaling and evaluating sparse autoencoders" ([arXiv:2406.04093](https://arxiv.org/abs/2406.04093))
- **Anthropic (2024)** - "Scaling Monosemanticity" ([transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/))
- **DeepMind (2024)** - "Gemma Scope" ([arXiv:2408.05147](https://arxiv.org/abs/2408.05147))

See [`docs/00-Foundations/mission.md`](docs/00-Foundations/mission.md) for complete reference list.

---

## 👥 Team Structure

### Roles
- **Research Lead** - Overall coordination, architecture decisions, integration
- **Infrastructure Engineer** - Training pipeline, cloud resources, experiment tracking
- **Analysis Specialist** - Feature matching, statistics, visualization

### Timeline
- **Phase 1** (Weeks 1-8): Controlled experiments, 50+ SAE training runs
- **Phase 2** (Weeks 9-14): Deep analysis, consistency measurement, geometric structure
- **Phase 3** (Weeks 15-20): Synthesis, open-source toolkit, presentation

---

## 🧪 Development Workflow

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_modular_arithmetic.py
```

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Training SAEs
```bash
# Train single SAE
python scripts/train_sae.py --config configs/relu_sae.yaml

# Run full experiment grid
python scripts/run_experiments.py --experiment modular_arithmetic_sweep
```

### Experiment Tracking
```bash
# Login to W&B
wandb login

# View experiments
wandb sync
```

---

## 📊 Expected Deliverables

### Minimum Viable Success
✅ Baseline transformer learning modular arithmetic
✅ 15+ trained SAEs (3 architectures × 5 seeds)
✅ Feature consistency measurement (PW-MCC)
✅ Fourier ground truth comparison (GT-MCC)
✅ Comprehensive documentation

### Target Success
🎯 50+ trained SAEs with hyperparameter sweep
🎯 Architecture-specific convergence patterns identified
🎯 Geometric structure characterization
🎯 Actionable guidelines for practitioners
🎯 Clean open-source release

### Stretch Goals
🚀 Extension to other algorithmic tasks
🚀 Application to language models (GPT-2 Small)
🚀 State-of-the-art circuit discovery implementation
🚀 Workshop/conference submission

---

## 🤝 Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

### Getting Started as a Contributor
1. Read [`docs/00-Foundations/mission.md`](docs/00-Foundations/mission.md)
2. Review open issues on GitHub
3. Set up development environment (see Quick Start)
4. Run tests to verify setup
5. Pick an issue or propose a new feature

---

## 📖 Documentation

- **Mission & Vision:** [`docs/00-Foundations/mission.md`](docs/00-Foundations/mission.md)
- **Research Strategy:** [`docs/01-Strategy/`](docs/01-Strategy/)
- **Technical Specifications:** [`docs/02-Product/`](docs/02-Product/)
- **Architectural Decisions:** [`docs/ADRs/`](docs/ADRs/)

---

## 🎓 Learning Resources

### Mechanistic Interpretability
- [TransformerLens Tutorials](https://transformerlensorg.github.io/TransformerLens/)
- [ARENA MI Track](https://www.arena.education/) - Comprehensive course
- [Neel Nanda's Blog](https://www.neelnanda.io/mechanistic-interpretability/)
- [Anthropic Interpretability Research](https://transformer-circuits.pub/)

### Sparse Autoencoders
- [SAELens Documentation](https://jbloomaus.github.io/SAELens/)
- [Neuronpedia](https://www.neuronpedia.org/) - Explore existing SAE features
- [SAEBench](https://neuronpedia.org/sae-bench) - Evaluation framework

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This project builds on foundational work by:
- Neel Nanda, Joseph Bloom, and the TransformerLens/SAELens teams
- Anthropic's interpretability research team
- OpenAI and DeepMind interpretability groups
- The broader mechanistic interpretability community

Special thanks to researchers who made their code and data publicly available.

---

## 📬 Contact

**Project Lead:** Bright Liu (brightliu@college.harvard.edu)
**GitHub:** [HUSAI Repository](https://github.com/brightlikethelight/HUSAI)

---

## 🔍 Status

**Current Phase:** Week 1 - Foundation & Setup
**Last Updated:** October 23, 2025

See [GitHub Issues](https://github.com/brightlikethelight/HUSAI/issues) for current tasks and progress.

---

<div align="center">

**Building reproducible, interpretable AI — one feature at a time.**

</div>
