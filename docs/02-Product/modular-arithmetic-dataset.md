# Modular Arithmetic Dataset Documentation

**HUSAI Research Project**
**Component:** Data Generation
**Status:** Implemented
**Last Updated:** October 24, 2025

---

## Overview

The Modular Arithmetic Dataset Generator creates datasets for the task: **c = (a + b) mod p**, where p is a prime modulus. This implementation follows the grokking research by Nanda et al. (2023) and Power et al. (2022).

**Key features:**
- Full or sampled dataset generation
- Two tokenization formats (sequence and tuple)
- Deterministic with seed control
- PyTorch Dataset interface
- Helper functions for DataLoaders, visualization, and validation

---

## Quick Start

```python
from src.data.modular_arithmetic import (
    ModularArithmeticDataset,
    create_dataloaders,
    visualize_samples,
)

# Create dataset
dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42)

# Or create ready-to-use DataLoaders
train_loader, test_loader = create_dataloaders(
    modulus=113,
    batch_size=512,
    train_fraction=0.7,
    seed=42
)

# Visualize examples
visualize_samples(dataset, n=5)
```

---

## Task Description

### Mathematical Definition

Given a prime modulus **p**, the task is to predict:

```
c = (a + b) mod p
```

where:
- `a, b ∈ {0, 1, ..., p-1}` (operands)
- `c ∈ {0, 1, ..., p-1}` (result)

### Dataset Size

For a given modulus p:
- **Full dataset:** p² examples (all possible pairs)
- **Example for p=113:** 113² = 12,769 examples
- **Fractional dataset:** Randomly sampled subset

### Why Modular Arithmetic?

Following Nanda et al. (2023), this task is ideal for grokking research because:

1. **Known ground truth:** Networks learn Fourier transform algorithms
2. **Clean convergence:** Achieves 100% accuracy after grokking
3. **Tractable scale:** Small models (1-2 layers) suffice
4. **Rich structure:** Exhibits phase transitions and emergent behavior
5. **Causal validation:** Can verify learned circuits match theory

---

## Token Formats

### Sequence Format (Default)

**Format:** `[BOS, a, +, b, =, c, EOS]`

**Token mapping:**
- Digits `0` to `p-1`: Represent themselves
- `BOS` (Begin of Sequence): `p`
- `EOS` (End of Sequence): `p + 1`
- `=` (Equals): `p + 2`
- `+` (Plus): `p + 3`

**Vocabulary size:** `p + 4`

**Example (p=113):**
```
Input: 47 + 39 = 86
Tokens: [113, 47, 116, 39, 115, 86, 114]
Decoded: "[BOS] 47 [+] 39 [=] 86 [EOS]"
```

**Advantages:**
- More interpretable (matches natural language)
- Explicit markers for sequence boundaries
- Easier to visualize and debug
- Follows Nanda et al. (2023) format

### Tuple Format

**Format:** `[a, b, c]`

**Token mapping:**
- Digits `0` to `p-1`: Represent themselves

**Vocabulary size:** `p`

**Example (p=113):**
```
Input: 47 + 39 = 86
Tokens: [47, 39, 86]
Decoded: "47 39 86"
```

**Advantages:**
- More compact (3 tokens vs 7)
- Faster training (fewer tokens to process)
- Lower memory usage
- Simpler vocabulary

---

## API Reference

### ModularArithmeticDataset

```python
class ModularArithmeticDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for modular arithmetic tasks."""

    def __init__(
        self,
        modulus: int = 113,
        fraction: float = 1.0,
        seed: int = 42,
        format: Literal["sequence", "tuple"] = "sequence",
    ) -> None:
        """Initialize dataset.

        Args:
            modulus: Prime modulus p for (a + b) mod p
            fraction: Fraction of p² examples to use (1.0 = all)
            seed: Random seed for reproducible sampling
            format: Token format ('sequence' or 'tuple')
        """
```

**Key methods:**

```python
# Get dataset size
len(dataset)  # Returns number of examples

# Get a single example
tokens, label = dataset[idx]  # tokens: [7] or [3], label: scalar

# Decode tokens to string
decoded = dataset.decode_tokens(tokens)

# Get vocabulary mapping
vocab = dataset.get_vocab_mapping()  # Dict[int, str]
```

### create_dataloaders

```python
def create_dataloaders(
    modulus: int = 113,
    fraction: float = 1.0,
    train_fraction: float = 0.7,
    batch_size: int = 512,
    seed: int = 42,
    format: Literal["sequence", "tuple"] = "sequence",
    num_workers: int = 0,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders.

    Returns:
        (train_loader, test_loader)
    """
```

### Helper Functions

```python
# Get vocabulary size
vocab_size = get_vocab_size(modulus=113, format="sequence")  # Returns 117

# Visualize samples
visualize_samples(dataset, n=5)  # Prints 5 random examples

# Get statistics
stats = get_statistics(dataset)  # Returns Dict with statistics

# Validate dataset
is_valid = validate_dataset(dataset)  # Returns True if valid
```

---

## Usage Examples

### Example 1: Basic Training Setup

```python
from src.data.modular_arithmetic import create_dataloaders

# Create DataLoaders for training
train_loader, test_loader = create_dataloaders(
    modulus=113,
    batch_size=512,
    train_fraction=0.7,
    seed=42,
)

# Training loop
for epoch in range(num_epochs):
    for tokens, labels in train_loader:
        # tokens: [batch_size, 7] for sequence format
        # labels: [batch_size]

        # Forward pass
        logits = model(tokens)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
```

### Example 2: Multiple Seeds for Reproducibility Study

```python
from src.data.modular_arithmetic import create_dataloaders

seeds = [42, 123, 456, 789, 1011]
dataloaders = []

for seed in seeds:
    train_loader, test_loader = create_dataloaders(
        modulus=113,
        seed=seed,
        batch_size=512,
    )
    dataloaders.append((train_loader, test_loader))

# Each seed produces different train/test split
# but same underlying full dataset
```

### Example 3: Partial Dataset for Quick Experiments

```python
from src.data.modular_arithmetic import ModularArithmeticDataset

# Use 10% of data for fast prototyping
dataset = ModularArithmeticDataset(
    modulus=113,
    fraction=0.1,  # Only 10% of 12,769 = ~1,277 examples
    seed=42,
)

print(f"Dataset size: {len(dataset)}")
# Output: Dataset size: 1276
```

### Example 4: Tuple Format for Efficiency

```python
from src.data.modular_arithmetic import create_dataloaders

# Use tuple format for faster training
train_loader, test_loader = create_dataloaders(
    modulus=113,
    format="tuple",  # 3 tokens instead of 7
    batch_size=1024,  # Can use larger batches
)

for tokens, labels in train_loader:
    # tokens: [batch_size, 3]
    # More compact, faster to process
    pass
```

### Example 5: Dataset Validation

```python
from src.data.modular_arithmetic import (
    ModularArithmeticDataset,
    validate_dataset,
    get_statistics,
)

dataset = ModularArithmeticDataset(modulus=113)

# Validate correctness
if validate_dataset(dataset):
    print("Dataset is valid!")

# Get statistics
stats = get_statistics(dataset)
print(f"Total examples: {stats['total_examples']}")
print(f"Vocabulary size: {stats['vocab_size']}")
print(f"Answer distribution: {stats['answer_distribution']}")
```

---

## Configuration

### ModularArithmeticConfig

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModularArithmeticConfig:
    """Configuration for modular arithmetic dataset."""

    modulus: int = 113  # Prime modulus
    fraction: float = 1.0  # Fraction of dataset to use
    train_fraction: float = 0.7  # Train/test split
    seed: int = 42  # Random seed
    format: Literal["sequence", "tuple"] = "sequence"  # Token format
    batch_size: int = 512  # Batch size
    num_workers: int = 0  # DataLoader workers
    device: str = "cpu"  # Device ('cpu', 'cuda', 'mps')
```

**Usage:**

```python
from src.data.modular_arithmetic import ModularArithmeticConfig, create_dataloaders

config = ModularArithmeticConfig(
    modulus=113,
    batch_size=512,
    train_fraction=0.7,
    seed=42,
)

# Unpack config into create_dataloaders
train_loader, test_loader = create_dataloaders(
    modulus=config.modulus,
    fraction=config.fraction,
    train_fraction=config.train_fraction,
    batch_size=config.batch_size,
    seed=config.seed,
    format=config.format,
    num_workers=config.num_workers,
    device=config.device,
)
```

---

## Design Decisions

### 1. Why Two Formats?

**Sequence format:**
- Follows Nanda et al. (2023) exactly
- Better for interpretability research
- Explicit structure makes debugging easier
- Recommended for initial experiments

**Tuple format:**
- More efficient for large-scale training
- Reduces memory and compute requirements
- Suitable for production training
- Recommended when training many models

### 2. Deterministic Sampling

All randomness is controlled by `seed`:
- Dataset sampling (if `fraction < 1.0`)
- Train/test split
- DataLoader shuffling (implicitly via PyTorch)

**Benefit:** Reproducible experiments across runs and machines.

### 3. Full Dataset by Default

Default `fraction=1.0` generates all p² examples because:
- For p=113: 12,769 examples is manageable
- Ensures complete coverage of the task
- Matches Nanda et al. (2023) experimental setup
- Avoids sampling bias

### 4. PyTorch Dataset Interface

Inherits from `torch.utils.data.Dataset` for:
- Seamless integration with PyTorch training loops
- Automatic batching via DataLoader
- Multi-process loading support
- Standard API familiar to ML practitioners

---

## Performance Considerations

### Memory Usage

**Sequence format (p=113):**
- Dataset size: 12,769 examples
- Tokens per example: 7
- Memory: ~89,000 int64 tokens ≈ 0.7 MB
- Negligible memory footprint

**Tuple format (p=113):**
- Dataset size: 12,769 examples
- Tokens per example: 3
- Memory: ~38,000 int64 tokens ≈ 0.3 MB
- Even more negligible

**Conclusion:** Memory is not a concern for modular arithmetic datasets.

### Training Speed

**Factors affecting speed:**
1. **Token format:** Tuple (3 tokens) is ~2.3× faster than sequence (7 tokens)
2. **Batch size:** Larger batches improve GPU utilization
3. **num_workers:** Multi-process loading (set to 2-4 for speedup)

**Recommendations:**
- Use `num_workers=2` or `4` on multi-core machines
- Use `format="tuple"` for faster training
- Batch size 512-1024 works well for p=113

### Disk I/O

**Note:** Dataset is generated in-memory, no disk I/O.
- Full dataset generation: <100ms for p=113
- No need to cache or save to disk
- Regenerated fresh for each experiment

---

## Testing

### Validation Checks

The `validate_dataset()` function verifies:

1. **Mathematical correctness:** `c = (a + b) % p` for all examples
2. **Token ranges:** All tokens within `[0, vocab_size-1]`
3. **Sequence lengths:** Correct length (7 or 3) for all examples
4. **Label consistency:** Labels match example answers

### Test Suite

Run comprehensive tests:

```bash
# All tests
pytest tests/unit/test_modular_arithmetic.py -v

# Specific test class
pytest tests/unit/test_modular_arithmetic.py::TestModularArithmeticDataset -v

# With coverage
pytest tests/unit/test_modular_arithmetic.py --cov=src.data.modular_arithmetic
```

**Test coverage:** 43 tests covering:
- Config validation
- Dataset generation
- Tokenization (both formats)
- DataLoader creation
- Helper functions
- Edge cases
- Integration scenarios

---

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Run from project root or set PYTHONPATH:
```bash
cd /path/to/HUSAI
python your_script.py

# OR
PYTHONPATH=/path/to/HUSAI python your_script.py
```

### Issue 2: Different Results Across Runs

**Problem:** Getting different train/test splits or samples

**Solution:** Ensure same seed:
```python
# Use same seed for reproducibility
train_loader, test_loader = create_dataloaders(seed=42)  # Always same
```

### Issue 3: Batch Size Too Large

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:** Reduce batch size or use CPU:
```python
# Reduce batch size
train_loader, test_loader = create_dataloaders(batch_size=256)

# Or use CPU
train_loader, test_loader = create_dataloaders(device="cpu")
```

### Issue 4: Type Errors with mypy

**Problem:** mypy complains about `DataLoader` types

**Solution:** Already handled with proper type annotations and `# type: ignore` where needed.

---

## Research Applications

### 1. Grokking Studies

```python
# Train transformer until grokking
train_loader, test_loader = create_dataloaders(modulus=113)

# Track training and validation accuracy
# Expect: train acc → 100% fast, val acc → 100% slow (grokking)
```

### 2. SAE Feature Stability

```python
# Train multiple SAEs on same grokked transformer
seeds = [42, 123, 456, 789, 1011]
saes = []

for seed in seeds:
    # Each SAE uses same data but different initialization
    sae = train_sae(model_activations, seed=seed)
    saes.append(sae)

# Measure feature overlap between SAEs
```

### 3. Fourier Circuit Recovery

```python
# Train SAE on grokked model
dataset = ModularArithmeticDataset(modulus=113)
sae = train_sae(model)

# Check if SAE features recover Fourier components
fourier_features = generate_fourier_basis(p=113)
gt_mcc = compute_gt_mcc(sae.features, fourier_features)
print(f"Ground truth recovery: {gt_mcc:.2%}")
```

### 4. Architecture Comparison

```python
# Compare ReLU vs TopK vs BatchTopK SAEs
for architecture in ["relu", "topk", "batchtopk"]:
    sae = train_sae(model, architecture=architecture)
    stability = measure_stability(sae, seeds=[1, 2, 3, 4, 5])
    print(f"{architecture} stability: {stability:.2%}")
```

---

## References

1. **Nanda, N., Chan, L., Liberum, T., Smith, J., & Steinhardt, J. (2023).** Progress measures for grokking via mechanistic interpretability. *arXiv:2301.05217*

2. **Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022).** Grokking: Generalization beyond overfitting on small algorithmic datasets. *arXiv:2201.02177*

3. **Nanda, N. (2023).** A mechanistic interpretability analysis of grokking. *Blog post*: https://www.neelnanda.io/mechanistic-interpretability/grokking

---

## File Locations

- **Implementation:** `/Users/brightliu/School_Work/HUSAI/src/data/modular_arithmetic.py`
- **Tests:** `/Users/brightliu/School_Work/HUSAI/tests/unit/test_modular_arithmetic.py`
- **Examples:** `/Users/brightliu/School_Work/HUSAI/notebooks/modular_arithmetic_example.py`
- **Documentation:** `/Users/brightliu/School_Work/HUSAI/docs/02-Product/modular-arithmetic-dataset.md`

---

**Document Status:** Complete
**Maintainer:** HUSAI Research Team
**Contact:** brightliu@college.harvard.edu
**Last Review:** October 24, 2025
