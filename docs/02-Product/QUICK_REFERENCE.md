# Modular Arithmetic Dataset - Quick Reference

**One-page cheat sheet for common operations**

---

## 🚀 Quick Start (30 seconds)

```python
from src.data.modular_arithmetic import create_dataloaders

# Create DataLoaders
train_loader, test_loader = create_dataloaders(modulus=113, batch_size=512)

# Use in training
for tokens, labels in train_loader:
    logits = model(tokens)  # tokens: [batch_size, 7]
    loss = criterion(logits, labels)
    # ...
```

---

## 📊 Common Patterns

### Pattern 1: Basic Training Setup

```python
from src.data.modular_arithmetic import create_dataloaders

train_loader, test_loader = create_dataloaders(
    modulus=113,           # Prime modulus
    batch_size=512,        # Batch size
    train_fraction=0.7,    # 70% train, 30% test
    seed=42,               # For reproducibility
)
```

### Pattern 2: Multiple Seeds for Reproducibility Studies

```python
from src.data.modular_arithmetic import create_dataloaders

seeds = [42, 123, 456, 789, 1011]
all_loaders = []

for seed in seeds:
    train_loader, test_loader = create_dataloaders(modulus=113, seed=seed)
    all_loaders.append((train_loader, test_loader))
```

### Pattern 3: Efficient Training with Tuple Format

```python
from src.data.modular_arithmetic import create_dataloaders

# Tuple format: [a, b, c] instead of [BOS, a, +, b, =, c, EOS]
train_loader, test_loader = create_dataloaders(
    modulus=113,
    format="tuple",      # 3 tokens instead of 7
    batch_size=1024,     # Can use larger batches
)
```

### Pattern 4: Small Dataset for Debugging

```python
from src.data.modular_arithmetic import create_dataloaders

# Use small modulus for quick tests
train_loader, test_loader = create_dataloaders(modulus=5, batch_size=10)
# Total: 25 examples (5²)
```

### Pattern 5: Partial Dataset for Quick Experiments

```python
from src.data.modular_arithmetic import create_dataloaders

# Use only 10% of data
train_loader, test_loader = create_dataloaders(
    modulus=113,
    fraction=0.1,  # 10% of 12,769 = ~1,277 examples
)
```

---

## 🔍 Inspection and Debugging

### Visualize Samples

```python
from src.data.modular_arithmetic import ModularArithmeticDataset, visualize_samples

dataset = ModularArithmeticDataset(modulus=113)
visualize_samples(dataset, n=5)

# Output:
# Sample 0: [BOS] 47 [+] 39 [=] 86 [EOS] -> 86
# Sample 1: [BOS] 94 [+] 36 [=] 17 [EOS] -> 17
# ...
```

### Get Statistics

```python
from src.data.modular_arithmetic import ModularArithmeticDataset, get_statistics

dataset = ModularArithmeticDataset(modulus=113)
stats = get_statistics(dataset)

print(f"Total examples: {stats['total_examples']}")
print(f"Vocabulary size: {stats['vocab_size']}")
print(f"Sequence length: {stats['sequence_length']}")
```

### Validate Dataset

```python
from src.data.modular_arithmetic import ModularArithmeticDataset, validate_dataset

dataset = ModularArithmeticDataset(modulus=113)
is_valid = validate_dataset(dataset)
print(f"Dataset valid: {is_valid}")  # Should be True
```

---

## 📐 Key Parameters

| Parameter | Default | Description | Common Values |
|-----------|---------|-------------|---------------|
| `modulus` | 113 | Prime modulus p | 5, 59, 97, 113 |
| `fraction` | 1.0 | Fraction of p² examples | 0.1, 0.5, 1.0 |
| `train_fraction` | 0.7 | Train/test split | 0.6, 0.7, 0.8 |
| `batch_size` | 512 | Batch size | 128, 256, 512, 1024 |
| `seed` | 42 | Random seed | 42, 123, 456, ... |
| `format` | "sequence" | Token format | "sequence", "tuple" |

---

## 🎯 Token Formats Comparison

| Aspect | Sequence Format | Tuple Format |
|--------|----------------|--------------|
| **Tokens** | `[BOS, a, +, b, =, c, EOS]` | `[a, b, c]` |
| **Length** | 7 | 3 |
| **Vocab Size** | p + 4 | p |
| **Speed** | Baseline | 2.3× faster |
| **Use Case** | Interpretability | Efficiency |

---

## 📊 Dataset Sizes

| Modulus (p) | Full Dataset (p²) | 50% Sample | 10% Sample |
|-------------|------------------|------------|------------|
| 5 | 25 | 12 | 2 |
| 59 | 3,481 | 1,740 | 348 |
| 97 | 9,409 | 4,704 | 940 |
| **113** | **12,769** | **6,384** | **1,276** |

---

## 🧪 Testing Commands

```bash
# Run all tests
pytest tests/unit/test_modular_arithmetic.py -v

# Run with coverage
pytest tests/unit/test_modular_arithmetic.py --cov=src.data.modular_arithmetic

# Run example script
PYTHONPATH=. python notebooks/modular_arithmetic_example.py

# Type checking
python -m mypy src/data/modular_arithmetic.py --strict

# Code formatting
python -m black src/data/modular_arithmetic.py
```

---

## 🐛 Common Issues

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution:** Run from project root or set PYTHONPATH:
```bash
cd /path/to/HUSAI
python your_script.py

# OR
PYTHONPATH=/path/to/HUSAI python your_script.py
```

### Issue: Different results across runs

**Solution:** Use same seed:
```python
train_loader, test_loader = create_dataloaders(seed=42)  # Always same
```

### Issue: CUDA out of memory

**Solution:** Reduce batch size or use CPU:
```python
train_loader, test_loader = create_dataloaders(batch_size=256, device="cpu")
```

---

## 📚 Function Reference

### Main Functions

```python
# Create DataLoaders (most common)
create_dataloaders(modulus, fraction, train_fraction, batch_size, seed, format, num_workers, device)

# Get vocabulary size
get_vocab_size(modulus, format)

# Visualize samples
visualize_samples(dataset, n, indices)

# Get statistics
get_statistics(dataset)

# Validate dataset
validate_dataset(dataset)
```

### Main Classes

```python
# Dataset class
ModularArithmeticDataset(modulus, fraction, seed, format)

# Config dataclass
ModularArithmeticConfig(modulus, fraction, train_fraction, seed, format, batch_size, num_workers, device)
```

---

## 🔗 File Locations

- **Implementation:** `src/data/modular_arithmetic.py`
- **Tests:** `tests/unit/test_modular_arithmetic.py`
- **Examples:** `notebooks/modular_arithmetic_example.py`
- **Full Docs:** `docs/02-Product/modular-arithmetic-dataset.md`
- **Quick Ref:** `docs/02-Product/QUICK_REFERENCE.md` (this file)

---

## 📖 Research References

1. **Nanda et al. (2023)** - Progress measures for grokking via mechanistic interpretability
   - arXiv:2301.05217
   - https://arxiv.org/abs/2301.05217

2. **Power et al. (2022)** - Grokking: Generalization beyond overfitting
   - arXiv:2201.02177
   - https://arxiv.org/abs/2201.02177

---

**Last Updated:** October 24, 2025
**Contact:** brightliu@college.harvard.edu
