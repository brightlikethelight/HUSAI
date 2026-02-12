"""Unit tests for configuration classes.

Tests cover:
- Basic instantiation and validation
- Field constraints (ranges, types, etc.)
- Cross-config validation (consistency checks)
- YAML serialization/deserialization
- Error handling and messages
"""

import pytest
from pathlib import Path
from pydantic import ValidationError
import tempfile
import yaml

from src.utils.config import (
    ModularArithmeticConfig,
    TransformerConfig,
    SAEConfig,
    ExperimentConfig,
    create_experiment_config_from_dict,
    load_and_validate_config,
)


# ============================================================================
# ModularArithmeticConfig Tests
# ============================================================================


class TestModularArithmeticConfig:
    """Tests for ModularArithmeticConfig."""

    def test_valid_config(self) -> None:
        """Test creating valid config."""
        config = ModularArithmeticConfig(
            modulus=113, num_samples=10_000, train_split=0.9, seed=42
        )
        assert config.modulus == 113
        assert config.num_samples == 10_000
        assert config.train_split == 0.9
        assert config.seed == 42

    def test_vocab_size_property(self) -> None:
        """Test vocab_size derived property."""
        config = ModularArithmeticConfig(
            modulus=113, num_samples=10_000, train_split=0.9, seed=42
        )
        assert config.vocab_size == 117  # modulus + 4

    def test_num_train_val_properties(self) -> None:
        """Test train/val split properties."""
        config = ModularArithmeticConfig(
            modulus=113, num_samples=10_000, train_split=0.9, seed=42
        )
        assert config.num_train == 9_000
        assert config.num_val == 1_000
        assert config.num_train + config.num_val == config.num_samples

    def test_invalid_modulus(self) -> None:
        """Test that modulus must be > 1."""
        with pytest.raises(ValidationError) as exc_info:
            ModularArithmeticConfig(modulus=1, num_samples=10_000, train_split=0.9)
        assert "greater than 1" in str(exc_info.value).lower()

    def test_invalid_num_samples(self) -> None:
        """Test that num_samples must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            ModularArithmeticConfig(modulus=113, num_samples=0, train_split=0.9)
        assert "greater than 0" in str(exc_info.value).lower()

    def test_invalid_train_split_too_low(self) -> None:
        """Test that train_split must be > 0."""
        with pytest.raises(ValidationError) as exc_info:
            ModularArithmeticConfig(modulus=113, num_samples=10_000, train_split=0.0)
        assert "greater than 0" in str(exc_info.value).lower()

    def test_invalid_train_split_too_high(self) -> None:
        """Test that train_split must be < 1."""
        with pytest.raises(ValidationError) as exc_info:
            ModularArithmeticConfig(modulus=113, num_samples=10_000, train_split=1.0)
        assert "less than 1" in str(exc_info.value).lower()

    def test_default_seed(self) -> None:
        """Test that seed defaults to 42."""
        config = ModularArithmeticConfig(modulus=113, num_samples=10_000, train_split=0.9)
        assert config.seed == 42


# ============================================================================
# TransformerConfig Tests
# ============================================================================


class TestTransformerConfig:
    """Tests for TransformerConfig."""

    def test_valid_config(self) -> None:
        """Test creating valid config."""
        config = TransformerConfig(
            n_layers=2,
            d_model=128,
            n_heads=4,
            d_mlp=512,
            vocab_size=117,
            max_seq_len=3,
            activation="relu",
        )
        assert config.n_layers == 2
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.d_mlp == 512
        assert config.vocab_size == 117
        assert config.max_seq_len == 3
        assert config.activation == "relu"

    def test_d_head_property(self) -> None:
        """Test d_head derived property."""
        config = TransformerConfig(
            n_layers=2,
            d_model=128,
            n_heads=4,
            d_mlp=512,
            vocab_size=117,
            max_seq_len=3,
        )
        assert config.d_head == 32  # 128 / 4

    def test_invalid_d_model_not_divisible(self) -> None:
        """Test that d_model must be divisible by n_heads."""
        with pytest.raises(ValidationError) as exc_info:
            TransformerConfig(
                n_layers=2,
                d_model=100,  # Not divisible by 7
                n_heads=7,
                d_mlp=512,
                vocab_size=117,
                max_seq_len=3,
            )
        assert "divisible" in str(exc_info.value).lower()

    def test_valid_activation_types(self) -> None:
        """Test all valid activation types."""
        valid_activations = ["relu", "gelu", "gelu_new", "silu", "gelu_fast"]
        for activation in valid_activations:
            config = TransformerConfig(
                n_layers=2,
                d_model=128,
                n_heads=4,
                d_mlp=512,
                vocab_size=117,
                max_seq_len=3,
                activation=activation,
            )
            assert config.activation == activation

    def test_invalid_activation_type(self) -> None:
        """Test that invalid activation raises error."""
        with pytest.raises(ValidationError):
            TransformerConfig(
                n_layers=2,
                d_model=128,
                n_heads=4,
                d_mlp=512,
                vocab_size=117,
                max_seq_len=3,
                activation="invalid_activation",  # type: ignore
            )

    def test_default_activation(self) -> None:
        """Test that activation defaults to gelu."""
        config = TransformerConfig(
            n_layers=2, d_model=128, n_heads=4, d_mlp=512, vocab_size=117, max_seq_len=3
        )
        assert config.activation == "gelu"


# ============================================================================
# SAEConfig Tests
# ============================================================================


class TestSAEConfig:
    """Tests for SAEConfig."""

    def test_valid_relu_config(self) -> None:
        """Test creating valid ReLU SAE config."""
        config = SAEConfig(
            architecture="relu",
            input_dim=128,
            expansion_factor=8,
            sparsity_level=1e-3,
            learning_rate=3e-4,
            batch_size=256,
            num_epochs=10,
            l1_coefficient=1e-3,
            seed=42,
        )
        assert config.architecture == "relu"
        assert config.l1_coefficient == 1e-3
        assert config.k is None

    def test_valid_topk_config(self) -> None:
        """Test creating valid TopK SAE config."""
        config = SAEConfig(
            architecture="topk",
            input_dim=128,
            expansion_factor=8,
            sparsity_level=32,
            learning_rate=3e-4,
            batch_size=256,
            num_epochs=10,
            k=32,
            seed=42,
        )
        assert config.architecture == "topk"
        assert config.k == 32
        assert config.l1_coefficient is None

    def test_valid_batchtopk_config(self) -> None:
        """Test creating valid BatchTopK SAE config."""
        config = SAEConfig(
            architecture="batchtopk",
            input_dim=128,
            expansion_factor=8,
            sparsity_level=32,
            learning_rate=3e-4,
            batch_size=256,
            num_epochs=10,
            k=32,
            seed=42,
        )
        assert config.architecture == "batchtopk"
        assert config.k == 32
        assert config.l1_coefficient is None

    def test_hidden_dim_property(self) -> None:
        """Test hidden_dim derived property."""
        config = SAEConfig(
            architecture="relu",
            input_dim=128,
            expansion_factor=8,
            sparsity_level=1e-3,
            learning_rate=3e-4,
            batch_size=256,
            num_epochs=10,
            l1_coefficient=1e-3,
        )
        assert config.hidden_dim == 1024  # 128 * 8
        assert config.num_features == 1024  # Alias

    def test_relu_missing_l1_coefficient(self) -> None:
        """Test that ReLU SAE requires l1_coefficient."""
        with pytest.raises(ValidationError) as exc_info:
            SAEConfig(
                architecture="relu",
                input_dim=128,
                expansion_factor=8,
                sparsity_level=1e-3,
                learning_rate=3e-4,
                batch_size=256,
                num_epochs=10,
                # Missing l1_coefficient
            )
        assert "l1_coefficient required" in str(exc_info.value)

    def test_topk_missing_k(self) -> None:
        """Test that TopK SAE requires k."""
        with pytest.raises(ValidationError) as exc_info:
            SAEConfig(
                architecture="topk",
                input_dim=128,
                expansion_factor=8,
                sparsity_level=32,
                learning_rate=3e-4,
                batch_size=256,
                num_epochs=10,
                # Missing k
            )
        assert "k required" in str(exc_info.value)

    def test_relu_with_k_should_fail(self) -> None:
        """Test that ReLU SAE should not have k parameter."""
        with pytest.raises(ValidationError) as exc_info:
            SAEConfig(
                architecture="relu",
                input_dim=128,
                expansion_factor=8,
                sparsity_level=1e-3,
                learning_rate=3e-4,
                batch_size=256,
                num_epochs=10,
                l1_coefficient=1e-3,
                k=32,  # Should not be set
            )
        assert "k should not be set" in str(exc_info.value)

    def test_topk_with_l1_coefficient_should_fail(self) -> None:
        """Test that TopK SAE should not have l1_coefficient."""
        with pytest.raises(ValidationError) as exc_info:
            SAEConfig(
                architecture="topk",
                input_dim=128,
                expansion_factor=8,
                sparsity_level=32,
                learning_rate=3e-4,
                batch_size=256,
                num_epochs=10,
                k=32,
                l1_coefficient=1e-3,  # Should not be set
            )
        assert "l1_coefficient should not be set" in str(exc_info.value)

    def test_sparsity_level_auto(self) -> None:
        """Test that sparsity_level can be 'auto'."""
        config = SAEConfig(
            architecture="relu",
            input_dim=128,
            expansion_factor=8,
            sparsity_level="auto",
            learning_rate=3e-4,
            batch_size=256,
            num_epochs=10,
            l1_coefficient=1e-3,
        )
        assert config.sparsity_level == "auto"

    def test_sparsity_level_invalid_string(self) -> None:
        """Test that invalid sparsity_level string raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SAEConfig(
                architecture="relu",
                input_dim=128,
                expansion_factor=8,
                sparsity_level="invalid",  # Not 'auto'
                learning_rate=3e-4,
                batch_size=256,
                num_epochs=10,
                l1_coefficient=1e-3,
            )
        assert "auto" in str(exc_info.value).lower()


# ============================================================================
# ExperimentConfig Tests
# ============================================================================


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_valid_config(self) -> None:
        """Test creating valid experiment config."""
        config = ExperimentConfig(
            experiment_name="test_experiment",
            wandb_project="husai",
            save_dir=Path("results/test"),
            checkpoint_frequency=5,
            log_frequency=100,
            dataset=ModularArithmeticConfig(
                modulus=113, num_samples=10_000, train_split=0.9
            ),
            transformer=TransformerConfig(
                n_layers=2,
                d_model=128,
                n_heads=4,
                d_mlp=512,
                vocab_size=117,  # modulus + 4
                max_seq_len=3,
            ),
            sae=SAEConfig(
                architecture="relu",
                input_dim=128,  # Matches d_model
                expansion_factor=8,
                sparsity_level=1e-3,
                learning_rate=3e-4,
                batch_size=256,
                num_epochs=10,
                l1_coefficient=1e-3,
            ),
        )
        assert config.experiment_name == "test_experiment"

    def test_vocab_size_mismatch(self) -> None:
        """Test that transformer vocab_size must match dataset."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(
                experiment_name="test",
                wandb_project="husai",
                save_dir=Path("results/test"),
                checkpoint_frequency=5,
                log_frequency=100,
                dataset=ModularArithmeticConfig(
                    modulus=113, num_samples=10_000, train_split=0.9
                ),
                transformer=TransformerConfig(
                    n_layers=2,
                    d_model=128,
                    n_heads=4,
                    d_mlp=512,
                    vocab_size=100,  # Wrong! Should be 117
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
        assert "vocab_size" in str(exc_info.value).lower()

    def test_input_dim_mismatch(self) -> None:
        """Test that SAE input_dim must match transformer d_model."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(
                experiment_name="test",
                wandb_project="husai",
                save_dir=Path("results/test"),
                checkpoint_frequency=5,
                log_frequency=100,
                dataset=ModularArithmeticConfig(
                    modulus=113, num_samples=10_000, train_split=0.9
                ),
                transformer=TransformerConfig(
                    n_layers=2,
                    d_model=128,
                    n_heads=4,
                    d_mlp=512,
                    vocab_size=117,
                    max_seq_len=3,
                ),
                sae=SAEConfig(
                    architecture="relu",
                    input_dim=256,  # Wrong! Should be 128
                    expansion_factor=8,
                    sparsity_level=1e-3,
                    learning_rate=3e-4,
                    batch_size=256,
                    num_epochs=10,
                    l1_coefficient=1e-3,
                ),
            )
        assert "input_dim" in str(exc_info.value).lower()

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = ExperimentConfig(
            experiment_name="test",
            wandb_project="husai",
            save_dir=Path("results/test"),
            checkpoint_frequency=5,
            log_frequency=100,
            dataset=ModularArithmeticConfig(
                modulus=113, num_samples=10_000, train_split=0.9
            ),
            transformer=TransformerConfig(
                n_layers=2,
                d_model=128,
                n_heads=4,
                d_mlp=512,
                vocab_size=117,
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
        d = config.to_dict()
        assert d["experiment_name"] == "test"
        assert d["dataset"]["modulus"] == 113
        assert d["transformer"]["d_model"] == 128
        assert d["sae"]["architecture"] == "relu"
        assert isinstance(d["save_dir"], str)  # Path converted to string


# ============================================================================
# YAML Serialization Tests
# ============================================================================


class TestYAMLSerialization:
    """Tests for YAML save/load functionality."""

    def test_save_and_load_yaml(self) -> None:
        """Test saving and loading config from YAML."""
        config = ExperimentConfig(
            experiment_name="test",
            wandb_project="husai",
            save_dir=Path("results/test"),
            checkpoint_frequency=5,
            log_frequency=100,
            dataset=ModularArithmeticConfig(
                modulus=113, num_samples=10_000, train_split=0.9, seed=42
            ),
            transformer=TransformerConfig(
                n_layers=2,
                d_model=128,
                n_heads=4,
                d_mlp=512,
                vocab_size=117,
                max_seq_len=3,
                activation="relu",
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
                seed=42,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"

            # Save
            config.save_yaml(yaml_path)
            assert yaml_path.exists()

            # Load
            loaded_config = ExperimentConfig.from_yaml(yaml_path)

            # Verify
            assert loaded_config.experiment_name == config.experiment_name
            assert loaded_config.dataset.modulus == config.dataset.modulus
            assert loaded_config.transformer.d_model == config.transformer.d_model
            assert loaded_config.sae.architecture == config.sae.architecture

    def test_from_yaml_file_not_found(self) -> None:
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            ExperimentConfig.from_yaml("nonexistent.yaml")

    def test_yaml_roundtrip_preserves_values(self) -> None:
        """Test that save/load preserves all values exactly."""
        config = ExperimentConfig(
            experiment_name="test",
            wandb_project="husai",
            save_dir=Path("results/test"),
            checkpoint_frequency=5,
            log_frequency=100,
            dataset=ModularArithmeticConfig(
                modulus=113, num_samples=10_000, train_split=0.9, seed=42
            ),
            transformer=TransformerConfig(
                n_layers=2,
                d_model=128,
                n_heads=4,
                d_mlp=512,
                vocab_size=117,
                max_seq_len=3,
                activation="relu",
            ),
            sae=SAEConfig(
                architecture="topk",
                input_dim=128,
                expansion_factor=8,
                sparsity_level=32,
                learning_rate=3e-4,
                batch_size=256,
                num_epochs=10,
                k=32,
                seed=42,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.save_yaml(yaml_path)
            loaded_config = ExperimentConfig.from_yaml(yaml_path)

            # Deep equality check
            assert loaded_config.model_dump() == config.model_dump()


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_experiment_config_from_dict(self) -> None:
        """Test creating config from dictionaries."""
        config = create_experiment_config_from_dict(
            experiment_name="test",
            wandb_project="husai",
            save_dir="results/test",
            dataset_kwargs={"modulus": 113, "num_samples": 10_000, "train_split": 0.9},
            transformer_kwargs={
                "n_layers": 2,
                "d_model": 128,
                "n_heads": 4,
                "d_mlp": 512,
                "vocab_size": 117,
                "max_seq_len": 3,
            },
            sae_kwargs={
                "architecture": "relu",
                "input_dim": 128,
                "expansion_factor": 8,
                "sparsity_level": 1e-3,
                "learning_rate": 3e-4,
                "batch_size": 256,
                "num_epochs": 10,
                "l1_coefficient": 1e-3,
            },
        )
        assert config.experiment_name == "test"
        assert config.dataset.modulus == 113

    def test_load_and_validate_config(self) -> None:
        """Test load_and_validate_config helper."""
        config = ExperimentConfig(
            experiment_name="test",
            wandb_project="husai",
            save_dir=Path("results/test"),
            checkpoint_frequency=5,
            log_frequency=100,
            dataset=ModularArithmeticConfig(
                modulus=113, num_samples=10_000, train_split=0.9
            ),
            transformer=TransformerConfig(
                n_layers=2,
                d_model=128,
                n_heads=4,
                d_mlp=512,
                vocab_size=117,
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

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.save_yaml(yaml_path)

            loaded_config = load_and_validate_config(yaml_path)
            assert loaded_config.experiment_name == "test"
