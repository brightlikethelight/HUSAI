# ADR-001: Project Architecture & Technology Stack

**Status:** Accepted
**Date:** 2025-10-23
**Authors:** Bright Liu
**Reviewers:** Team

---

## Context

We're building HUSAI to systematically investigate SAE feature stability across random seeds using modular arithmetic as ground truth. The project requires:

1. Training 50+ SAEs with different architectures, seeds, and hyperparameters
2. Analyzing feature consistency using sophisticated matching algorithms
3. Validating against known Fourier circuits
4. Managing experiments, data, and results across cloud infrastructure
5. Supporting collaboration among 3 team members
6. Producing publication-quality research and open-source tools

We need to make foundational decisions about:
- Programming language and deep learning framework
- Mechanistic interpretability libraries
- Experiment tracking and management
- Code structure and organization
- Development practices and quality assurance

---

## Decision

### Technology Stack

#### Core Framework
- **Python 3.11** as primary language
  - **Rationale:** Industry standard for ML research, extensive library ecosystem
  - **Alternative considered:** Python 3.10 (rejected: 3.11 has better performance and type hints)

- **PyTorch 2.5.1** as deep learning framework
  - **Rationale:** TransformerLens/SAELens built on PyTorch; excellent for research
  - **CUDA 11.8** for GPU support (widely compatible, stable)
  - **Alternative considered:** JAX (rejected: fewer mechanistic interpretability tools)
  - **Note:** Explicitly avoiding PyTorch 2.6.0 due to breaking `torch.load` changes

#### Mechanistic Interpretability

- **TransformerLens 2.16.1** for model internals access
  - **Rationale:** De facto standard; used by Neel Nanda, ARENA, research community
  - **Features:** Hook-based activation access, 50+ pretrained models, clean transformer implementation

- **SAELens 6.17.0** for SAE training and analysis
  - **Rationale:** Purpose-built for SAE research, actively maintained, HuggingFace integration
  - **Features:** Multiple architectures (ReLU, TopK, BatchTopK), pretrained SAEs, visualization
  - **Alternative considered:** Custom implementation (rejected: reinventing wheel, slower)

- **CircuitsVis 1.43.3** for visualization
  - **Rationale:** React-based interactive visualizations created by Alan Cooney & Neel Nanda
  - **Optional but recommended**

#### Experiment Management

- **Weights & Biases (wandb) 0.22.2** for experiment tracking
  - **Rationale:** Industry standard, free tier sufficient, excellent visualization
  - **Features:** Hyperparameter tracking, metric logging, artifact management, collaboration
  - **Alternative considered:** MLflow (rejected: less feature-rich for our use case)

- **YAML configuration files** for experiment specification
  - **Rationale:** Human-readable, version-controllable, easy to template
  - **Location:** `configs/` directory

#### Development Tools

- **pytest 8.4.2** for testing
  - **Rationale:** Python standard, extensive plugin ecosystem
  - **Coverage:** pytest-cov for code coverage reports

- **black 25.9.0 + isort 7.0.0** for code formatting
  - **Rationale:** Eliminate formatting debates, consistent style
  - **Configuration:** 100-character line length (balances readability and horizontal space)

- **mypy 1.18.2** for type checking
  - **Rationale:** Catch bugs early, improve code documentation
  - **Strategy:** Gradual adoption (not strict mode initially)

- **pre-commit 4.3.0** for git hooks
  - **Rationale:** Enforce quality checks before commits
  - **Hooks:** formatting, linting, security checks (bandit), notebook cleaning

- **JupyterLab 4.4.10** for interactive development
  - **Rationale:** Essential for exploratory analysis, visualization
  - **Convention:** Notebooks in `notebooks/`, numbered sequentially

### Project Structure

```
HUSAI/
├── src/                      # Source code (package: import src.data)
│   ├── data/                # Dataset generation, loading
│   ├── models/              # Model architectures (transformers, SAEs)
│   ├── training/            # Training loops, SAE training logic
│   ├── analysis/            # Feature matching, evaluation
│   └── utils/               # Shared utilities
├── tests/                    # Test suite
│   ├── unit/                # Unit tests (fast, isolated)
│   ├── integration/         # Integration tests
│   └── e2e/                 # End-to-end tests
├── notebooks/                # Jupyter notebooks
│   ├── 01_*.ipynb           # Numbered for order
│   ├── 02_*.ipynb
│   └── exploratory/         # Ad-hoc exploration (not numbered)
├── scripts/                  # Standalone scripts
│   ├── train_*.py           # Training scripts
│   └── analyze_*.py         # Analysis scripts
├── configs/                  # Configuration files (YAML)
│   ├── base.yaml            # Base configuration
│   ├── relu_sae.yaml        # Architecture-specific
│   └── experiments/         # Full experiment specs
├── docs/                     # Documentation
│   ├── 00-Foundations/
│   ├── 01-Strategy/
│   ├── 02-Product/
│   ├── 03-Go-To-Market/
│   ├── 04-Execution/
│   └── ADRs/
├── data/                     # Data (gitignored)
│   ├── raw/
│   ├── processed/
│   └── activations/
├── results/                  # Results (gitignored)
│   ├── trained_saes/
│   ├── analyses/
│   └── figures/
├── checkpoints/              # Model checkpoints (gitignored)
└── wandb/                    # W&B logs (gitignored)
```

**Rationale for structure:**
- **Separation of concerns:** Clear boundaries between data, models, training, analysis
- **Testability:** `src/` as package allows `import src.data` in tests
- **Reproducibility:** `configs/` centralizes experimental parameters
- **Collaboration:** Clear conventions for where things go

### Coding Conventions

1. **Naming:**
   - Modules/packages: `snake_case`
   - Classes: `PascalCase`
   - Functions/variables: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`

2. **Type hints:**
   - All public functions must have type hints
   - Return types required
   - Example: `def train_sae(config: SAEConfig, model: HookedTransformer) -> SAE:`

3. **Docstrings:**
   - Google-style docstrings for all public functions/classes
   - Include: Args, Returns, Raises, Examples
   - Example:
     ```python
     def compute_pw_mcc(sae1: SAE, sae2: SAE) -> float:
         """Compute Pairwise Dictionary Mean Correlation Coefficient.

         Args:
             sae1: First sparse autoencoder
             sae2: Second sparse autoencoder

         Returns:
             PW-MCC score between 0 and 1

         Raises:
             ValueError: If SAEs have different dimensions
         """
     ```

4. **Testing:**
   - Minimum 70% code coverage for `src/`
   - All new features must have tests
   - Use fixtures for common test setup
   - Mark slow tests with `@pytest.mark.slow`
   - Mark GPU tests with `@pytest.mark.gpu`

5. **Logging:**
   - Use Python `logging` module (not print statements)
   - Levels: DEBUG (training details), INFO (progress), WARNING (issues), ERROR (failures)
   - Rich library for beautiful terminal output

### Development Workflow

1. **Branching:**
   - `main` branch: stable, tested code
   - Feature branches: `feature/feature-name`
   - Bugfix branches: `fix/issue-description`

2. **Pull Requests:**
   - All changes via PR (no direct commits to main)
   - PR must pass: tests, linting, type checking
   - At least one review required

3. **Commits:**
   - Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
   - Example: `feat(training): add BatchTopK SAE implementation`

4. **Experiment Tracking:**
   - Every training run logged to W&B
   - Config file saved as artifact
   - Random seed fixed and logged
   - Tag runs with: architecture, seed, hyperparameters

---

## Consequences

### Positive

1. **Reproducibility:** Fixed versions, clear configs, logged experiments
2. **Collaboration:** Clear structure, code quality enforced, shared tools
3. **Velocity:** Established libraries (TransformerLens, SAELens) accelerate development
4. **Quality:** Testing, type checking, linting catch bugs early
5. **Community alignment:** Using standard tools enables knowledge sharing

### Negative

1. **Learning curve:** Team must learn TransformerLens, SAELens APIs
2. **Dependency risk:** Relying on external libraries (mitigated by pinned versions)
3. **PyTorch 2.6 compatibility:** Must eventually migrate (planned for Phase 3)
4. **Cloud costs:** W&B artifact storage, GPU compute (mitigated by budget monitoring)

### Neutral

1. **Python over alternatives:** Standard choice, no strong opinion
2. **YAML configs:** Simple but less powerful than Python-based configs (Hydra)
3. **JupyterLab:** Essential for research but can lead to messy notebooks (mitigated by numbering convention and nbstripout)

---

## Implementation Plan

### Week 1 (Current)
- ✅ Set up project structure
- ✅ Create environment files
- ✅ Install pre-commit hooks
- ✅ Write this ADR

### Week 2
- Create `src/` package structure
- Implement modular arithmetic dataset (`src/data/modular_arithmetic.py`)
- Write first tests
- Set up W&B project

### Week 3-4
- Implement SAE architectures using SAELens
- Create training pipeline
- Add experiment configs

---

## Alternatives Considered

### Alternative 1: JAX + Flax
**Rejected because:**
- TransformerLens is PyTorch-only
- SAELens is PyTorch-only
- Would require reimplementing significant tooling
- Team more familiar with PyTorch

**If chosen:** Better performance via JIT compilation, but much higher development cost

### Alternative 2: Custom SAE Implementation
**Rejected because:**
- SAELens already implements ReLU, TopK, BatchTopK
- Reinventing wheel delays experiments
- Community-validated code more trustworthy

**If chosen:** More control, but slower and more bug-prone

### Alternative 3: MLflow for Experiment Tracking
**Rejected because:**
- W&B has better UI and visualization
- W&B free tier sufficient
- Better integration with notebooks

**If chosen:** More self-hostable, but less feature-rich

### Alternative 4: Strict Typing from Day 1
**Rejected because:**
- Would slow down initial exploration
- Many libraries have incomplete type stubs

**Gradual adoption instead:** Add types incrementally as code stabilizes

---

## Future Decisions

### To Be Decided Later

1. **Model registry** (if releasing 50+ SAEs)
   - HuggingFace Hub or custom solution?
   - Decision point: Week 15

2. **Documentation framework** (if publishing)
   - MkDocs Material vs Sphinx?
   - Decision point: Week 17

3. **Cloud provider** (AWS vs GCP)
   - Depends on team's existing accounts/credits
   - Decision point: Week 3

4. **Parallel training framework**
   - Ray, Slurm, or simple bash scripts?
   - Decision point: Week 5 (based on scale)

---

## References

- [TransformerLens Docs](https://transformerlensorg.github.io/TransformerLens/)
- [SAELens Docs](https://jbloomaus.github.io/SAELens/)
- [Weights & Biases Best Practices](https://docs.wandb.ai/guides)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

## Changelog

- **2025-10-23:** Initial ADR created
- **Future:** This section will track major changes to architecture decisions

---

**Review Status:** ✅ Accepted

**Next Review:** Week 8 (mid-project checkpoint)
