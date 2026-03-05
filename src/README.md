# HUSAI Source Code

## Modules

| Module | Description |
|--------|-------------|
| **data/** | Modular arithmetic dataset generation |
| **utils/** | Pydantic configuration system |
| **models/** | Transformer and SAE architectures (`simple_sae.py`) |
| **training/** | SAE training loop (`train_sae.py`) |
| **analysis/** | Feature matching, stability metrics (`feature_matching.py`) |

## Key Entry Points

- Dataset: `data/modular_arithmetic.py`
- SAE model: `models/simple_sae.py`
- Training: `training/train_sae.py`
- Stability analysis: `analysis/feature_matching.py`
- Config: `utils/config.py`

## Running Tests

```bash
pytest tests -q
```

## Code Quality

```bash
black src/ && isort src/
mypy src/
flake8 src/
```

## Contact

- See `RUNBOOK.md` for development and reproduction guidelines
- Email: brightliu@college.harvard.edu
