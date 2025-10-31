# Configuration System Summary

## Overview

The HUSAI configuration system provides type-safe, validated, and serializable configuration management using **Pydantic v2**. This document summarizes the design decisions and implementation.

## Architecture Decision: Pydantic v2

**Chosen:** Pydantic v2
**Alternative Considered:** Python dataclasses

### Rationale

| Feature | Pydantic v2 | Dataclasses |
|---------|-------------|-------------|
| **Validation** | ✅ Built-in with clear errors | ❌ Manual implementation required |
| **YAML Serialization** | ✅ Native support | ❌ Manual serialization |
| **Type Coercion** | ✅ Automatic (e.g., "0.001" → 0.001) | ❌ Manual conversion |
| **Field Constraints** | ✅ `Field(gt=0, le=1.0)` | ❌ Requires custom validators |
| **Ecosystem** | ✅ SAELens, Transformers use it | - Stdlib |
| **W&B Integration** | ✅ `.model_dump()` for dicts | ❌ Manual conversion |
| **Dependency Cost** | ~2MB (negligible for ML) | None |

**Conclusion:** Pydantic v2's validation and serialization features are essential for ML experiment configuration. The dependency cost is negligible compared to PyTorch (1GB+).

## Implementation

### File Structure

```
src/utils/config.py          # Main configuration classes
tests/unit/test_config.py    # Comprehensive unit tests (33 tests, 100% pass)
configs/examples/            # Example YAML configurations
examples/config_usage.py     # Usage demonstration script
```

### Configuration Classes

#### 1. ModularArithmeticConfig
Dataset generation parameters.

```python
config = ModularArithmeticConfig(
    modulus=113,         # Prime modulus (determines vocab_size)
    num_samples=50_000,  # Total samples to generate
    train_split=0.9,     # 90% train, 10% validation
    seed=42             # Random seed
)
# Derived properties:
config.vocab_size  # 114 (modulus + 1)
config.num_train   # 45,000
config.num_val     # 5,000
```

**Validation:**
- `modulus > 1`
- `num_samples > 0`
- `0 < train_split < 1`
- `seed >= 0`

#### 2. TransformerConfig
Transformer architecture specification.

```python
config = TransformerConfig(
    n_layers=2,          # Number of layers
    d_model=128,         # Residual stream dimension
    n_heads=4,           # Attention heads (d_model must be divisible)
    d_mlp=512,           # MLP hidden dimension
    vocab_size=114,      # Token vocabulary size
    max_seq_len=3,       # Maximum sequence length
    activation="gelu"    # Activation function
)
# Derived properties:
config.d_head  # 32 (d_model / n_heads)
```

**Validation:**
- `d_model % n_heads == 0` (cross-field validation)
- `activation` must be one of: `["relu", "gelu", "gelu_new", "silu", "gelu_fast"]`

#### 3. SAEConfig
Sparse autoencoder architecture and training.

```python
# ReLU SAE
config = SAEConfig(
    architecture="relu",
    input_dim=128,
    expansion_factor=8,
    sparsity_level=1e-3,
    learning_rate=3e-4,
    batch_size=256,
    num_epochs=10,
    l1_coefficient=1e-3,  # Required for ReLU
    seed=42
)

# TopK SAE
config = SAEConfig(
    architecture="topk",
    input_dim=128,
    expansion_factor=16,
    sparsity_level=32,
    learning_rate=3e-4,
    batch_size=512,
    num_epochs=20,
    k=32,  # Required for TopK
    seed=42
)
```

**Validation:**
- ReLU SAE: requires `l1_coefficient`, must NOT have `k`
- TopK/BatchTopK SAE: requires `k`, must NOT have `l1_coefficient`
- `sparsity_level` can be numeric or `"auto"`

**Derived properties:**
- `hidden_dim` = `input_dim * expansion_factor`
- `num_features` (alias for `hidden_dim`)

#### 4. ExperimentConfig
Top-level experiment orchestration.

```python
config = ExperimentConfig(
    experiment_name="baseline_relu_seed42",
    wandb_project="husai-sae-stability",
    save_dir=Path("results/baseline_relu_seed42"),
    checkpoint_frequency=5,
    log_frequency=100,
    dataset=ModularArithmeticConfig(...),
    transformer=TransformerConfig(...),
    sae=SAEConfig(...)
)
```

**Cross-Config Validation:**
- `transformer.vocab_size == dataset.vocab_size`
- `sae.input_dim == transformer.d_model`

## Key Features

### 1. YAML Serialization

```python
# Save
config.save_yaml(Path("configs/my_experiment.yaml"))

# Load
config = ExperimentConfig.from_yaml("configs/my_experiment.yaml")
```

### 2. W&B Integration

```python
import wandb
wandb.init(
    project=config.wandb_project,
    name=config.experiment_name,
    config=config.to_dict()  # Converts to JSON-serializable dict
)
```

### 3. Comprehensive Validation

```python
try:
    config = ExperimentConfig.from_yaml("invalid.yaml")
except ValidationError as e:
    print(e)
    # Clear error: "transformer vocab_size (100) must match dataset vocab_size (114)"
```

### 4. Helper Functions

```python
# Create from nested dicts (useful for sweeps)
config = create_experiment_config_from_dict(
    experiment_name="sweep_1",
    wandb_project="husai",
    save_dir="results/sweep_1",
    dataset_kwargs={...},
    transformer_kwargs={...},
    sae_kwargs={...}
)

# Load with error handling
config = load_and_validate_config("configs/experiment.yaml")
```

## Example Configurations

### baseline_relu.yaml
Standard ReLU SAE with 8x expansion (1024 features).

```yaml
experiment_name: baseline_relu_seed42
wandb_project: husai-sae-stability
save_dir: results/baseline_relu_seed42
checkpoint_frequency: 5
log_frequency: 100

dataset:
  modulus: 113
  num_samples: 50000
  train_split: 0.9
  seed: 42

transformer:
  n_layers: 2
  d_model: 128
  n_heads: 4
  d_mlp: 512
  vocab_size: 114
  max_seq_len: 3
  activation: relu

sae:
  architecture: relu
  input_dim: 128
  expansion_factor: 8
  sparsity_level: 0.001
  learning_rate: 0.0003
  batch_size: 256
  num_epochs: 10
  l1_coefficient: 0.001
  seed: 42
```

### topk_16x.yaml
TopK SAE with 16x expansion (2048 features), k=64.

### batchtopk_32x.yaml
BatchTopK SAE with 32x expansion (4096 features), k=128.

## Testing

**Test Suite:** 33 unit tests, 100% pass rate

Coverage:
- ✅ Valid config creation for all classes
- ✅ Field-level validation (ranges, types, constraints)
- ✅ Cross-config validation (consistency checks)
- ✅ Architecture-specific parameter validation
- ✅ YAML serialization/deserialization round-trips
- ✅ Error handling with clear messages
- ✅ Derived property calculations
- ✅ Helper function correctness

Run tests:
```bash
pytest tests/unit/test_config.py -v
```

## Usage Examples

See `/examples/config_usage.py` for comprehensive examples:

```bash
python examples/config_usage.py
```

Examples cover:
1. Loading from YAML
2. Creating programmatically
3. Creating from dictionaries
4. Save/load round-trips
5. Validation error handling
6. W&B integration
7. Generating sweep configs

## Best Practices

### Naming Conventions
Use descriptive experiment names:
```
{architecture}_{expansion}x_seed{seed}
```
Examples: `relu_8x_seed42`, `topk_16x_seed123`

### Seed Management
Use different seeds for reproducibility studies:
- Baseline: 42
- Variations: 123, 456, 789, 999

### Expansion Factors
Common values:
- Small experiments: 4x, 8x
- Standard: 16x
- Large-scale: 32x, 64x

### Version Control
Always commit configs alongside code for full reproducibility:
```bash
git add configs/experiment_name.yaml
git commit -m "Add config for experiment_name"
```

## Dependencies

```toml
# pyproject.toml
dependencies = [
    "pydantic>=2.0",     # Core validation
    "pyyaml>=6.0",       # YAML serialization
    "typing-extensions"  # Self type hint for validators
]
```

## Future Enhancements

Potential additions (not currently needed):

1. **Config inheritance:** Base configs with overrides
2. **Hydra integration:** More sophisticated config composition
3. **Config versioning:** Track schema changes over time
4. **Auto-documentation:** Generate docs from Pydantic models
5. **CLI integration:** `python train.py --config configs/baseline.yaml`

## References

- Pydantic v2 Docs: https://docs.pydantic.dev/latest/
- ADR-001: Project Architecture
- `/configs/README.md`: Detailed configuration guide

---

**Status:** ✅ Implemented and tested
**Last Updated:** 2025-10-24
**Author:** Bright Liu
