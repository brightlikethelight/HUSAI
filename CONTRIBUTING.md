# Contributing to HUSAI

Thank you for your interest in contributing to HUSAI! This document provides guidelines and instructions for contributing.

## 🎯 Quick Start

1. **Read the mission:** Start with [`docs/00-Foundations/mission.md`](docs/00-Foundations/mission.md)
2. **Set up environment:** Follow the setup instructions in [`README.md`](README.md)
3. **Pick an issue:** Browse [open issues](https://github.com/yourusername/HUSAI/issues)
4. **Join the discussion:** Reach out to the team before starting major work

## 🔄 Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/yourusername/HUSAI.git
cd HUSAI
```

### 2. Create a Branch

```bash
# Feature branch
git checkout -b feature/your-feature-name

# Bugfix branch
git checkout -b fix/issue-description
```

### 3. Make Changes

- Write code following our style guide (see below)
- Add tests for new functionality
- Update documentation as needed
- Keep commits atomic and well-described

### 4. Run Quality Checks

```bash
# Run all checks
make quality

# Or individually:
make format    # Auto-format code
make lint      # Check code style
make typecheck # Type checking
make test      # Run tests
```

### 5. Commit

```bash
# Use conventional commits
git add .
git commit -m "feat(training): add BatchTopK SAE implementation"
```

**Commit types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

### 6. Push and Create PR

```bash
git push origin your-branch-name
```

Then create a Pull Request on GitHub with:
- Clear title following conventional commit format
- Description of what changed and why
- Link to related issues
- Screenshots/plots if relevant

## 📝 Code Style Guide

### Python Style

We follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with some modifications:

- **Line length:** 100 characters (not 80)
- **Formatting:** Use `black` and `isort` (enforced by pre-commit)
- **Type hints:** Required for all public functions
- **Docstrings:** Google-style docstrings required

**Example:**

```python
from typing import Tuple

import torch
from transformer_lens import HookedTransformer


def train_sae(
    model: HookedTransformer,
    config: SAEConfig,
    seed: int = 42,
) -> Tuple[SAE, dict]:
    """Train a sparse autoencoder on model activations.

    Args:
        model: Pre-trained transformer to extract activations from
        config: SAE configuration specifying architecture and hyperparameters
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
            - Trained SAE model
            - Dictionary of training metrics (loss, L0, etc.)

    Raises:
        ValueError: If config.architecture not in ['relu', 'topk', 'batchtopk']

    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> config = SAEConfig(architecture="topk", expansion_factor=4)
        >>> sae, metrics = train_sae(model, config)
    """
    # Implementation here
    pass
```

### File Organization

```python
# 1. Future imports
from __future__ import annotations

# 2. Standard library
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# 3. Third-party
import numpy as np
import torch
from transformer_lens import HookedTransformer

# 4. Local imports
from src.data import load_dataset
from src.utils import set_seed

# 5. Module-level constants
DEFAULT_BATCH_SIZE = 32
LOGGER = logging.getLogger(__name__)
```

### Naming Conventions

- **Modules/packages:** `snake_case` (e.g., `modular_arithmetic.py`)
- **Classes:** `PascalCase` (e.g., `SAETrainer`)
- **Functions/variables:** `snake_case` (e.g., `train_sae`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `DEFAULT_LEARNING_RATE`)
- **Private methods:** `_leading_underscore` (e.g., `_compute_loss`)

## 🧪 Testing Guidelines

### Test Structure

```python
import pytest
from src.models.sae import ReLUSAE


class TestReLUSAE:
    """Tests for ReLU SAE implementation."""

    @pytest.fixture
    def sae(self):
        """Create SAE for testing."""
        return ReLUSAE(input_dim=512, hidden_dim=2048)

    def test_forward_pass_shape(self, sae):
        """Test that forward pass returns correct shape."""
        x = torch.randn(32, 512)
        x_recon, features = sae(x)

        assert x_recon.shape == (32, 512)
        assert features.shape == (32, 2048)

    def test_sparsity(self, sae):
        """Test that SAE produces sparse features."""
        x = torch.randn(32, 512)
        _, features = sae(x)

        l0 = (features != 0).float().mean()
        assert l0 < 0.5  # At least 50% sparse

    @pytest.mark.slow
    def test_training_reduces_loss(self, sae):
        """Test that training decreases reconstruction loss."""
        # Longer test marked as slow
        pass

    @pytest.mark.gpu
    def test_gpu_training(self, sae):
        """Test training on GPU."""
        # GPU-specific test
        pass
```

### Test Coverage

- Aim for **70%+ coverage** for `src/`
- All public APIs must have tests
- Test edge cases and error conditions
- Use fixtures for common setup

### Running Tests

```bash
# All tests
pytest

# Fast tests only
make test-fast

# With coverage
make test-cov

# Specific file
pytest tests/unit/test_sae.py

# Specific test
pytest tests/unit/test_sae.py::TestReLUSAE::test_forward_pass_shape
```

## 📚 Documentation

### Code Documentation

- All public modules, classes, and functions need docstrings
- Use Google-style docstrings
- Include examples for complex functions
- Keep docstrings up-to-date with code changes

### Project Documentation

When adding features, update:
- `README.md` if it affects setup/usage
- Relevant docs in `docs/` directory
- Add ADR if making architectural decision

### Notebooks

- Number notebooks sequentially: `01_baseline.ipynb`, `02_training.ipynb`
- Add markdown cells explaining what you're doing
- Clear outputs before committing (handled by pre-commit)
- Move polished notebooks to main directory, keep exploratory in `exploratory/`

## 🐛 Reporting Bugs

### Before Reporting

1. Check if issue already exists
2. Verify it's a bug (not expected behavior)
3. Try to reproduce in clean environment

### Bug Report Template

```markdown
**Description**
Clear description of the bug.

**To Reproduce**
1. Step 1
2. Step 2
3. See error

**Expected Behavior**
What should happen.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.11]
- PyTorch version: [e.g., 2.5.1]
- CUDA version: [e.g., 11.8]

**Additional Context**
- Error traceback
- Relevant logs
- Screenshots if applicable
```

## 💡 Suggesting Features

### Feature Request Template

```markdown
**Problem Statement**
What problem does this solve?

**Proposed Solution**
How would you solve it?

**Alternatives Considered**
Other approaches you've thought about.

**Additional Context**
Examples, mockups, references to papers, etc.
```

## 🎯 Good First Issues

Look for issues tagged `good-first-issue` or `help-wanted`:
- Documentation improvements
- Adding tests
- Fixing small bugs
- Adding examples/tutorials

## 🤝 Code Review

### As an Author

- Keep PRs focused and reasonably sized
- Respond to feedback constructively
- Update PR based on review comments
- Don't take criticism personally—we're all learning!

### As a Reviewer

- Be kind and constructive
- Explain *why* changes are needed
- Suggest concrete improvements
- Approve when code meets standards

### Review Checklist

- [ ] Code follows style guide
- [ ] Tests added for new functionality
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Commit messages follow conventional commits
- [ ] No merge conflicts

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License (see [`LICENSE`](LICENSE)).

## 🙏 Recognition

All contributors will be recognized in:
- GitHub contributors page
- Project documentation
- Academic papers (if applicable)

## 📬 Questions?

- **Email:** brightliu@college.harvard.edu
- **GitHub Issues:** [Open an issue](https://github.com/yourusername/HUSAI/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/HUSAI/discussions)

## 🎓 Learning Resources

New to mechanistic interpretability? Check out:
- [TransformerLens Tutorials](https://transformerlensorg.github.io/TransformerLens/)
- [ARENA MI Track](https://www.arena.education/)
- [Neel Nanda's Blog](https://www.neelnanda.io/)
- Our [`docs/00-Foundations/literature-review.md`](docs/00-Foundations/literature-review.md)

---

**Thank you for contributing to HUSAI!** 🚀
