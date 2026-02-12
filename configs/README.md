# Configuration Files

This directory contains YAML configuration files for HUSAI experiments.

## Directory Structure

```
configs/
├── README.md           # This file
└── examples/           # Example configurations
    ├── baseline_relu.yaml      # ReLU SAE with 8x expansion
    ├── topk_16x.yaml          # TopK SAE with 16x expansion
    └── batchtopk_32x.yaml     # BatchTopK SAE with 32x expansion
```

## Configuration Schema

Each experiment config is a YAML file with the following structure:

```yaml
experiment_name: "descriptive_name_seed42"
wandb_project: "husai-sae-stability"
save_dir: "results/experiment_name"
checkpoint_frequency: 5  # Save every N epochs (0 = only final)
log_frequency: 100  # Log every N steps

dataset:
  modulus: 113  # Prime modulus for modular arithmetic
  num_samples: 50000
  train_split: 0.9  # 90% train, 10% validation
  seed: 42

transformer:
  n_layers: 2
  d_model: 128  # Must be divisible by n_heads
  n_heads: 4
  d_mlp: 512  # Typically 4x d_model
  vocab_size: 117  # Must equal dataset.modulus + 4
  max_seq_len: 3
  activation: "relu"  # Options: relu, gelu, gelu_new, silu, gelu_fast

sae:
  architecture: "relu"  # Options: relu, topk, batchtopk
  input_dim: 128  # Must equal transformer.d_model
  expansion_factor: 8  # Hidden dim = input_dim * expansion_factor
  sparsity_level: 0.001  # For relu: L1 coeff; for topk: k value
  learning_rate: 0.0003
  batch_size: 256
  num_epochs: 10
  l1_coefficient: 0.001  # Required for relu SAE
  # k: 32  # Required for topk/batchtopk SAE
  seed: 42
```

## Architecture-Specific Parameters

### ReLU SAE
Requires:
- `l1_coefficient`: L1 penalty for sparsity (e.g., 0.001)

Example:
```yaml
sae:
  architecture: "relu"
  l1_coefficient: 0.001
  # Do NOT set k
```

### TopK SAE
Requires:
- `k`: Number of top features to keep (e.g., 32)

Example:
```yaml
sae:
  architecture: "topk"
  k: 32
  # Do NOT set l1_coefficient
```

### BatchTopK SAE
Requires:
- `k`: Number of top features across batch (e.g., 128)

Example:
```yaml
sae:
  architecture: "batchtopk"
  k: 128
  # Do NOT set l1_coefficient
```

## Usage

### Load Config in Python

```python
from pathlib import Path
from src.utils.config import ExperimentConfig

# Load config
config = ExperimentConfig.from_yaml("configs/examples/baseline_relu.yaml")

# Access nested configs
print(f"Modulus: {config.dataset.modulus}")
print(f"d_model: {config.transformer.d_model}")
print(f"SAE architecture: {config.sae.architecture}")

# Convert to dict for W&B
import wandb
wandb.init(
    project=config.wandb_project,
    name=config.experiment_name,
    config=config.to_dict()
)
```

### Create Config Programmatically

```python
from src.utils.config import create_experiment_config_from_dict
from pathlib import Path

config = create_experiment_config_from_dict(
    experiment_name="my_experiment",
    wandb_project="husai-sae-stability",
    save_dir=Path("results/my_experiment"),
    dataset_kwargs={
        "modulus": 113,
        "num_samples": 50000,
        "train_split": 0.9,
        "seed": 42
    },
    transformer_kwargs={
        "n_layers": 2,
        "d_model": 128,
        "n_heads": 4,
        "d_mlp": 512,
        "vocab_size": 117,
        "max_seq_len": 3
    },
    sae_kwargs={
        "architecture": "relu",
        "input_dim": 128,
        "expansion_factor": 8,
        "sparsity_level": 0.001,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "num_epochs": 10,
        "l1_coefficient": 0.001
    }
)

# Save to file
config.save_yaml(Path("configs/my_experiment.yaml"))
```

### Generate Sweep Configs

```python
from pathlib import Path
from src.utils.config import ExperimentConfig, ModularArithmeticConfig, TransformerConfig, SAEConfig

# Base config
base_dataset = ModularArithmeticConfig(modulus=113, num_samples=50000, train_split=0.9)
base_transformer = TransformerConfig(
    n_layers=2, d_model=128, n_heads=4, d_mlp=512, vocab_size=117, max_seq_len=3
)

# Generate configs for different seeds
for seed in [42, 123, 456, 789, 999]:
    config = ExperimentConfig(
        experiment_name=f"relu_8x_seed{seed}",
        wandb_project="husai-sae-stability",
        save_dir=Path(f"results/relu_8x_seed{seed}"),
        checkpoint_frequency=5,
        log_frequency=100,
        dataset=ModularArithmeticConfig(**{**base_dataset.model_dump(), "seed": seed}),
        transformer=base_transformer,
        sae=SAEConfig(
            architecture="relu",
            input_dim=128,
            expansion_factor=8,
            sparsity_level=0.001,
            learning_rate=3e-4,
            batch_size=256,
            num_epochs=10,
            l1_coefficient=0.001,
            seed=seed
        )
    )
    config.save_yaml(Path(f"configs/sweep/relu_8x_seed{seed}.yaml"))
```

## Validation

Configs are automatically validated on load with helpful error messages:

```python
# This will raise ValidationError with clear message
config = ExperimentConfig.from_yaml("configs/invalid.yaml")
# ValidationError: transformer vocab_size (100) must match dataset vocab_size (117)
```

Common validation errors:
- `vocab_size mismatch`: transformer.vocab_size must equal dataset.vocab_size (modulus + 4)
- `input_dim mismatch`: sae.input_dim must equal transformer.d_model
- `d_model not divisible by n_heads`: d_model must be evenly divisible by n_heads
- `l1_coefficient required for ReLU SAE`: ReLU architecture requires l1_coefficient
- `k required for TopK SAE`: TopK/BatchTopK architectures require k parameter

## Example Configs

### `baseline_relu.yaml`
Standard ReLU SAE with 8x expansion, suitable for quick experiments.

### `topk_16x.yaml`
TopK SAE with 16x expansion (2048 features), keeps top 64 active.

### `batchtopk_32x.yaml`
BatchTopK SAE with 32x expansion (4096 features), very sparse with k=128.

## Best Practices

1. **Naming**: Use descriptive names: `{architecture}_{expansion}x_seed{seed}`
2. **Seeds**: Use different seeds (42, 123, 456, etc.) for reproducibility studies
3. **Expansion factors**: Common values are 4x, 8x, 16x, 32x
4. **Validation**: Always load and validate configs before long training runs
5. **Version control**: Commit configs alongside code for full reproducibility
6. **Documentation**: Add comments in YAML for non-obvious choices

## See Also

- `/src/utils/config.py` - Configuration class definitions
- `/tests/unit/test_config.py` - Configuration validation tests
- `/docs/ADRs/ADR-001-project-architecture.md` - Architecture decisions
