#!/usr/bin/env python3
"""Example script demonstrating configuration usage.

This script shows how to:
1. Load configs from YAML
2. Create configs programmatically
3. Generate sweep configs
4. Validate and handle errors
5. Use configs with W&B (commented out)
"""

from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import (
    ExperimentConfig,
    ModularArithmeticConfig,
    TransformerConfig,
    SAEConfig,
    create_experiment_config_from_dict,
    load_and_validate_config,
)
from pydantic import ValidationError


def example_1_load_from_yaml() -> None:
    """Example 1: Load config from YAML file."""
    print("\n" + "=" * 60)
    print("Example 1: Load Config from YAML")
    print("=" * 60)

    config_path = Path("configs/examples/baseline_relu.yaml")
    config = ExperimentConfig.from_yaml(config_path)

    print(f"✓ Loaded: {config.experiment_name}")
    print(f"  Dataset: modulus={config.dataset.modulus}, "
          f"vocab_size={config.dataset.vocab_size}")
    print(f"  Transformer: {config.transformer.n_layers} layers, "
          f"d_model={config.transformer.d_model}")
    print(f"  SAE: {config.sae.architecture}, "
          f"{config.sae.expansion_factor}x expansion = {config.sae.num_features} features")


def example_2_create_programmatically() -> None:
    """Example 2: Create config programmatically."""
    print("\n" + "=" * 60)
    print("Example 2: Create Config Programmatically")
    print("=" * 60)

    config = ExperimentConfig(
        experiment_name="programmatic_test",
        wandb_project="husai-sae-stability",
        save_dir=Path("results/programmatic_test"),
        checkpoint_frequency=5,
        log_frequency=100,
        dataset=ModularArithmeticConfig(
            modulus=113,
            num_samples=10_000,
            train_split=0.9,
            seed=42,
        ),
        transformer=TransformerConfig(
            n_layers=2,
            d_model=128,
            n_heads=4,
            d_mlp=512,
            vocab_size=114,  # modulus + 1
            max_seq_len=3,
            activation="gelu",
        ),
        sae=SAEConfig(
            architecture="topk",
            input_dim=128,  # matches d_model
            expansion_factor=8,
            sparsity_level=32,
            learning_rate=3e-4,
            batch_size=256,
            num_epochs=10,
            k=32,  # TopK specific
            seed=42,
        ),
    )

    print(f"✓ Created: {config.experiment_name}")
    print(f"  SAE type: {config.sae.architecture} with k={config.sae.k}")
    print(f"  Features: {config.sae.num_features}")


def example_3_create_from_dict() -> None:
    """Example 3: Create config from nested dictionaries."""
    print("\n" + "=" * 60)
    print("Example 3: Create Config from Dictionaries")
    print("=" * 60)

    config = create_experiment_config_from_dict(
        experiment_name="dict_based_config",
        wandb_project="husai-sae-stability",
        save_dir="results/dict_based",
        dataset_kwargs={
            "modulus": 113,
            "num_samples": 50_000,
            "train_split": 0.9,
            "seed": 999,
        },
        transformer_kwargs={
            "n_layers": 2,
            "d_model": 128,
            "n_heads": 4,
            "d_mlp": 512,
            "vocab_size": 114,
            "max_seq_len": 3,
        },
        sae_kwargs={
            "architecture": "relu",
            "input_dim": 128,
            "expansion_factor": 16,
            "sparsity_level": 1e-3,
            "learning_rate": 3e-4,
            "batch_size": 256,
            "num_epochs": 10,
            "l1_coefficient": 1e-3,
            "seed": 999,
        },
    )

    print(f"✓ Created: {config.experiment_name}")
    print(f"  Expansion: {config.sae.expansion_factor}x = {config.sae.num_features} features")


def example_4_save_and_load() -> None:
    """Example 4: Save config to YAML and reload."""
    print("\n" + "=" * 60)
    print("Example 4: Save and Load Round-trip")
    print("=" * 60)

    # Create config
    config = ExperimentConfig(
        experiment_name="save_load_test",
        wandb_project="husai-sae-stability",
        save_dir=Path("results/save_load_test"),
        checkpoint_frequency=10,
        log_frequency=50,
        dataset=ModularArithmeticConfig(
            modulus=113, num_samples=10_000, train_split=0.8, seed=123
        ),
        transformer=TransformerConfig(
            n_layers=2,
            d_model=128,
            n_heads=4,
            d_mlp=512,
            vocab_size=114,
            max_seq_len=3,
        ),
        sae=SAEConfig(
            architecture="batchtopk",
            input_dim=128,
            expansion_factor=8,
            sparsity_level=64,
            learning_rate=1e-4,
            batch_size=512,
            num_epochs=20,
            k=64,
            seed=123,
        ),
    )

    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = Path(f.name)

    config.save_yaml(temp_path)
    print(f"✓ Saved to: {temp_path}")

    # Load it back
    loaded_config = ExperimentConfig.from_yaml(temp_path)
    print(f"✓ Loaded from: {temp_path}")
    print(f"  Match: {config.model_dump() == loaded_config.model_dump()}")

    # Clean up
    temp_path.unlink()


def example_5_validation_errors() -> None:
    """Example 5: Demonstrate validation errors."""
    print("\n" + "=" * 60)
    print("Example 5: Validation Error Handling")
    print("=" * 60)

    # Error 1: vocab_size mismatch
    print("\n[Test 1] vocab_size mismatch:")
    try:
        config = ExperimentConfig(
            experiment_name="error_test",
            wandb_project="test",
            save_dir=Path("results/test"),
            checkpoint_frequency=5,
            log_frequency=100,
            dataset=ModularArithmeticConfig(
                modulus=113, num_samples=1000, train_split=0.9  # vocab_size = 114
            ),
            transformer=TransformerConfig(
                n_layers=2,
                d_model=128,
                n_heads=4,
                d_mlp=512,
                vocab_size=100,  # WRONG! Should be 114
                max_seq_len=3,
            ),
            sae=SAEConfig(
                architecture="relu",
                input_dim=128,
                expansion_factor=8,
                sparsity_level=1e-3,
                learning_rate=3e-4,
                batch_size=256,
                num_epochs=10,
                l1_coefficient=1e-3,
            ),
        )
        print("  ✗ ERROR: Should have raised ValidationError!")
    except ValidationError as e:
        print(f"  ✓ Caught expected error: {str(e).splitlines()[0]}")

    # Error 2: d_model not divisible by n_heads
    print("\n[Test 2] d_model not divisible by n_heads:")
    try:
        config = TransformerConfig(
            n_layers=2,
            d_model=100,  # Not divisible by 7
            n_heads=7,
            d_mlp=512,
            vocab_size=114,
            max_seq_len=3,
        )
        print("  ✗ ERROR: Should have raised ValidationError!")
    except ValidationError as e:
        print(f"  ✓ Caught expected error: divisible validation failed")

    # Error 3: ReLU SAE missing l1_coefficient
    print("\n[Test 3] ReLU SAE missing l1_coefficient:")
    try:
        config = SAEConfig(
            architecture="relu",
            input_dim=128,
            expansion_factor=8,
            sparsity_level=1e-3,
            learning_rate=3e-4,
            batch_size=256,
            num_epochs=10,
            # Missing l1_coefficient!
        )
        print("  ✗ ERROR: Should have raised ValidationError!")
    except ValidationError as e:
        print(f"  ✓ Caught expected error: l1_coefficient required")

    # Error 4: TopK SAE with wrong parameter
    print("\n[Test 4] TopK SAE with l1_coefficient (should use k):")
    try:
        config = SAEConfig(
            architecture="topk",
            input_dim=128,
            expansion_factor=8,
            sparsity_level=32,
            learning_rate=3e-4,
            batch_size=256,
            num_epochs=10,
            l1_coefficient=1e-3,  # WRONG! Should use k for TopK
            k=32,
        )
        print("  ✗ ERROR: Should have raised ValidationError!")
    except ValidationError as e:
        print(f"  ✓ Caught expected error: l1_coefficient should not be set for topk")


def example_6_wandb_integration() -> None:
    """Example 6: W&B integration (commented out)."""
    print("\n" + "=" * 60)
    print("Example 6: W&B Integration (Demo Only)")
    print("=" * 60)

    config = ExperimentConfig.from_yaml("configs/examples/baseline_relu.yaml")

    print("\nTo use with W&B:")
    print("```python")
    print("import wandb")
    print("")
    print("# Initialize run with config")
    print("wandb.init(")
    print("    project=config.wandb_project,")
    print("    name=config.experiment_name,")
    print("    config=config.to_dict()")
    print(")")
    print("")
    print("# Access config values in W&B")
    print("print(f'Learning rate: {wandb.config.sae.learning_rate}')")
    print("```")


def example_7_generate_sweep() -> None:
    """Example 7: Generate multiple configs for hyperparameter sweep."""
    print("\n" + "=" * 60)
    print("Example 7: Generate Sweep Configs")
    print("=" * 60)

    # Base configurations
    base_dataset = ModularArithmeticConfig(
        modulus=113, num_samples=50_000, train_split=0.9
    )
    base_transformer = TransformerConfig(
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_mlp=512,
        vocab_size=114,
        max_seq_len=3,
    )

    # Generate configs for different expansion factors
    for expansion in [4, 8, 16, 32]:
        config = ExperimentConfig(
            experiment_name=f"relu_{expansion}x_seed42",
            wandb_project="husai-sae-stability",
            save_dir=Path(f"results/relu_{expansion}x_seed42"),
            checkpoint_frequency=5,
            log_frequency=100,
            dataset=base_dataset,
            transformer=base_transformer,
            sae=SAEConfig(
                architecture="relu",
                input_dim=128,
                expansion_factor=expansion,
                sparsity_level=1e-3,
                learning_rate=3e-4,
                batch_size=256,
                num_epochs=10,
                l1_coefficient=1e-3,
                seed=42,
            ),
        )
        print(f"  Generated: {config.experiment_name} ({config.sae.num_features} features)")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("HUSAI Configuration System Examples")
    print("=" * 60)

    example_1_load_from_yaml()
    example_2_create_programmatically()
    example_3_create_from_dict()
    example_4_save_and_load()
    example_5_validation_errors()
    example_6_wandb_integration()
    example_7_generate_sweep()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
