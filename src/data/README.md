# Data Module

This module contains dataset generators and data utilities for the HUSAI project.

## Contents

### `modular_arithmetic.py`

Modular arithmetic dataset generator for grokking research.

**Task:** Predict c = (a + b) mod p

**Key Features:**
- Full or sampled dataset generation (up to p² examples)
- Two token formats: sequence `[BOS, a, +, b, =, c, EOS]` and tuple `[a, b, c]`
- Deterministic generation with seed control
- PyTorch Dataset interface
- Helper functions for DataLoaders, visualization, and validation

**Quick Start:**

```python
from src.data.modular_arithmetic import create_dataloaders

# Create train/test DataLoaders
train_loader, test_loader = create_dataloaders(
    modulus=113,
    batch_size=512,
    train_fraction=0.7,
    seed=42,
)

# Use in training loop
for tokens, labels in train_loader:
    # tokens: [batch_size, 7] (sequence format)
    # labels: [batch_size]
    logits = model(tokens)
    loss = criterion(logits, labels)
    # ...
```

**Documentation:** See `/docs/02-Product/modular-arithmetic-dataset.md`

**Tests:** See `/tests/unit/test_modular_arithmetic.py`

**Examples:** See `/notebooks/modular_arithmetic_example.py`

## Usage Examples

### Example 1: Basic Dataset

```python
from src.data.modular_arithmetic import ModularArithmeticDataset

dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42)
print(f"Dataset size: {len(dataset)}")  # 12,769 examples

tokens, label = dataset[0]
print(f"Tokens shape: {tokens.shape}")  # [7]
print(f"Label: {label.item()}")
```

### Example 2: DataLoaders for Training

```python
from src.data.modular_arithmetic import create_dataloaders

train_loader, test_loader = create_dataloaders(
    modulus=113,
    batch_size=512,
    train_fraction=0.7,
)

print(f"Train examples: {len(train_loader.dataset)}")  # 8,938
print(f"Test examples: {len(test_loader.dataset)}")    # 3,831
```

### Example 3: Tuple Format (More Efficient)

```python
from src.data.modular_arithmetic import create_dataloaders

train_loader, test_loader = create_dataloaders(
    modulus=113,
    format="tuple",  # [a, b, c] instead of [BOS, a, +, b, =, c, EOS]
    batch_size=1024,  # Can use larger batches
)

for tokens, labels in train_loader:
    # tokens: [batch_size, 3] (more compact)
    pass
```

### Example 4: Visualization

```python
from src.data.modular_arithmetic import (
    ModularArithmeticDataset,
    visualize_samples,
)

dataset = ModularArithmeticDataset(modulus=113)
visualize_samples(dataset, n=5)

# Output:
# Sample 0: [BOS] 47 [+] 39 [=] 86 [EOS] -> 86
# Sample 1: [BOS] 94 [+] 36 [=] 17 [EOS] -> 17
# ...
```

### Example 5: Multiple Seeds for Reproducibility Studies

```python
from src.data.modular_arithmetic import create_dataloaders

seeds = [42, 123, 456, 789, 1011]

for seed in seeds:
    train_loader, test_loader = create_dataloaders(
        modulus=113,
        seed=seed,
        batch_size=512,
    )
    # Each seed produces different train/test split
    # but same underlying full dataset
```

## API Reference

### Main Classes

- **`ModularArithmeticDataset`**: PyTorch Dataset for modular arithmetic
- **`ModularArithmeticConfig`**: Configuration dataclass

### Functions

- **`create_dataloaders()`**: Create train/test DataLoaders
- **`get_vocab_size()`**: Get vocabulary size for given modulus
- **`visualize_samples()`**: Print sample examples
- **`get_statistics()`**: Compute dataset statistics
- **`validate_dataset()`**: Validate dataset correctness

## Testing

Run tests:

```bash
# All tests
pytest tests/unit/test_modular_arithmetic.py -v

# With coverage
pytest tests/unit/test_modular_arithmetic.py --cov=src.data.modular_arithmetic

# Run example
PYTHONPATH=. python notebooks/modular_arithmetic_example.py
```

## Design Choices

1. **Two formats:** Sequence (interpretable) vs Tuple (efficient)
2. **Deterministic:** All randomness controlled by seed
3. **Full dataset by default:** p²=12,769 examples for p=113 is manageable
4. **PyTorch interface:** Seamless integration with training loops
5. **In-memory generation:** Fast, no disk I/O needed

## Performance

- **Memory:** ~0.7 MB for full dataset (p=113, sequence format)
- **Generation time:** <100ms for p=113
- **Training speed:** Tuple format ~2.3× faster than sequence

## Research Applications

- Grokking studies (following Nanda et al. 2023)
- SAE feature stability experiments
- Fourier circuit recovery validation
- Architecture comparison (ReLU vs TopK vs BatchTopK SAEs)

## References

1. Nanda et al. (2023) - Progress measures for grokking via mechanistic interpretability
2. Power et al. (2022) - Grokking: Generalization beyond overfitting on small algorithmic datasets

---

**Last Updated:** October 24, 2025
**Maintainer:** HUSAI Research Team
