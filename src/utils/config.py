"""Configuration classes for HUSAI experiments.

This module defines Pydantic models for configuring:
- Modular arithmetic dataset generation
- Transformer architecture
- Sparse autoencoder (SAE) training and architecture
- Full experiment specification

All configs support:
- YAML serialization/deserialization
- Validation with clear error messages
- Conversion to dict for W&B logging
- Type safety and autocompletion

Example:
    >>> from pathlib import Path
    >>> config = ExperimentConfig.from_yaml("configs/baseline.yaml")
    >>> config.save_yaml(Path("outputs/config.yaml"))
    >>> wandb.config.update(config.to_dict())
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Literal, Any
from typing_extensions import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class ModularArithmeticConfig(BaseModel):
    """Configuration for modular arithmetic dataset generation.

    This dataset is used for mechanistic interpretability research because:
    1. Known ground truth (Fourier circuits)
    2. Controllable complexity via modulus
    3. Deterministic and reproducible

    Attributes:
        modulus: Prime modulus for arithmetic operations (e.g., 113).
            Determines vocab size and circuit complexity.
        num_samples: Total number of (a, b, c) samples to generate where c = (a + b) % modulus.
            Typical values: 10_000 for quick experiments, 100_000+ for full training.
        train_split: Fraction of samples used for training vs validation.
            Must be in (0, 1). Common: 0.8 or 0.9.
        seed: Random seed for reproducible dataset generation.
            Different seeds produce different sample orderings but same distribution.

    Example:
        >>> config = ModularArithmeticConfig(
        ...     modulus=113,
        ...     num_samples=50_000,
        ...     train_split=0.9,
        ...     seed=42
        ... )
        >>> config.vocab_size
        117  # modulus + 4 (BOS, EOS, EQUALS, PLUS tokens)
    """

    modulus: int = Field(
        ...,
        description="Prime modulus for arithmetic (determines vocab size)",
        gt=1,  # Must be > 1
        examples=[113, 97, 59],
    )
    num_samples: int = Field(
        ...,
        description="Total number of samples to generate",
        gt=0,
        examples=[10_000, 50_000, 100_000],
    )
    train_split: float = Field(
        ...,
        description="Fraction of samples for training (rest for validation)",
        gt=0.0,
        lt=1.0,
        examples=[0.8, 0.9],
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
        ge=0,
    )

    @property
    def vocab_size(self) -> int:
        """Vocabulary size for sequence format tokenization.

        Tokens: 0 to p-1 (digit tokens) + 4 special tokens:
        - BOS (beginning of sequence)
        - EOS (end of sequence)
        - EQUALS (=)
        - PLUS (+)

        Total: p + 4 tokens

        Note: For tuple format ([a, b, c]), only p tokens are needed,
        but we use sequence format as the default.
        """
        return self.modulus + 4

    @property
    def num_train(self) -> int:
        """Number of training samples."""
        return int(self.num_samples * self.train_split)

    @property
    def num_val(self) -> int:
        """Number of validation samples."""
        return self.num_samples - self.num_train

    model_config = {"frozen": False}  # Allow mutation for experimentation


class TransformerConfig(BaseModel):
    """Configuration for baseline transformer architecture.

    This is the model we'll train SAEs on. Uses standard decoder-only
    transformer architecture (similar to GPT).

    Attributes:
        n_layers: Number of transformer layers (depth).
            More layers = more representational capacity but slower training.
        d_model: Dimension of residual stream (hidden size).
            Core dimension that flows through the model.
        n_heads: Number of attention heads per layer.
            d_model must be divisible by n_heads.
        d_mlp: Dimension of MLP hidden layer.
            Typically 4 * d_model (standard transformer ratio).
        vocab_size: Size of token vocabulary.
            Usually set automatically from ModularArithmeticConfig.vocab_size.
        max_seq_len: Maximum sequence length the model can handle.
            For modular arithmetic: typically 3 (a, b, c) or 4 (with separator).
        activation: Activation function for MLP layers.
            Options: "relu", "gelu", "gelu_new", "silu", "gelu_fast".

    Example:
        >>> config = TransformerConfig(
        ...     n_layers=2,
        ...     d_model=128,
        ...     n_heads=4,
        ...     d_mlp=512,
        ...     vocab_size=114,
        ...     max_seq_len=3,
        ...     activation="relu"
        ... )
    """

    n_layers: int = Field(
        ...,
        description="Number of transformer layers",
        gt=0,
        le=12,  # Sanity check for small models
        examples=[1, 2, 4],
    )
    d_model: int = Field(
        ...,
        description="Dimension of residual stream",
        gt=0,
        examples=[64, 128, 256, 512],
    )
    n_heads: int = Field(
        ...,
        description="Number of attention heads",
        gt=0,
        examples=[1, 2, 4, 8],
    )
    d_mlp: int = Field(
        ...,
        description="Dimension of MLP hidden layer",
        gt=0,
        examples=[256, 512, 1024, 2048],
    )
    vocab_size: int = Field(
        ...,
        description="Size of token vocabulary (from dataset)",
        gt=0,
    )
    max_seq_len: int = Field(
        ...,
        description="Maximum sequence length",
        gt=0,
        le=1024,  # Sanity check
        examples=[3, 4, 5],
    )
    activation: Literal["relu", "gelu", "gelu_new", "silu", "gelu_fast"] = Field(
        default="gelu",
        description="Activation function for MLP",
    )

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        """Ensure d_model is divisible by n_heads."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self

    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    model_config = {"frozen": False}


class SAEConfig(BaseModel):
    """Configuration for Sparse Autoencoder (SAE) architecture and training.

    SAEs are trained to reconstruct transformer activations while enforcing
    sparsity. Different architectures achieve sparsity differently:
    - ReLU: L1 penalty on activations
    - TopK: Keep only top-k activations per sample
    - BatchTopK: Keep top-k across entire batch (more efficient)

    Attributes:
        architecture: SAE architecture type.
            Options: "relu", "topk", "batchtopk".
        input_dim: Dimension of input activations (typically d_model from transformer).
        expansion_factor: Hidden dimension multiplier (expansion_factor * input_dim).
            Larger = more features but sparser. Common: 4, 8, 16, 32.
        sparsity_level: Target sparsity (architecture-dependent).
            - For ReLU: L1 coefficient (float, e.g., 1e-3)
            - For TopK/BatchTopK: k value (int, e.g., 32) or "auto"
        learning_rate: Learning rate for Adam optimizer.
        batch_size: Batch size for training.
        num_epochs: Number of training epochs.
        l1_coefficient: L1 penalty coefficient (ReLU SAE only).
            Controls sparsity vs reconstruction trade-off.
            Higher = sparser but worse reconstruction.
        k: Number of active features (TopK/BatchTopK only).
            Lower = sparser representations.
        seed: Random seed for reproducibility.

    Example (ReLU SAE):
        >>> config = SAEConfig(
        ...     architecture="relu",
        ...     input_dim=128,
        ...     expansion_factor=8,
        ...     sparsity_level=1e-3,
        ...     learning_rate=3e-4,
        ...     batch_size=256,
        ...     num_epochs=10,
        ...     l1_coefficient=1e-3,
        ...     seed=42
        ... )

    Example (TopK SAE):
        >>> config = SAEConfig(
        ...     architecture="topk",
        ...     input_dim=128,
        ...     expansion_factor=8,
        ...     sparsity_level=32,
        ...     learning_rate=3e-4,
        ...     batch_size=256,
        ...     num_epochs=10,
        ...     k=32,
        ...     seed=42
        ... )
    """

    architecture: Literal["relu", "topk", "batchtopk"] = Field(
        ...,
        description="SAE architecture type",
    )
    input_dim: int = Field(
        ...,
        description="Dimension of input activations (e.g., d_model)",
        gt=0,
    )
    expansion_factor: int = Field(
        ...,
        description="Hidden dimension multiplier (hidden_dim = expansion_factor * input_dim)",
        gt=1,
        examples=[4, 8, 16, 32],
    )
    sparsity_level: str | float = Field(
        ...,
        description="Target sparsity (L1 coeff for ReLU, k for TopK, or 'auto')",
    )
    learning_rate: float = Field(
        ...,
        description="Learning rate for Adam optimizer",
        gt=0.0,
        examples=[1e-4, 3e-4, 1e-3],
    )
    batch_size: int = Field(
        ...,
        description="Batch size for training",
        gt=0,
        examples=[128, 256, 512, 1024],
    )
    num_epochs: int = Field(
        ...,
        description="Number of training epochs",
        gt=0,
        examples=[5, 10, 20, 50],
    )
    l1_coefficient: float | None = Field(
        default=None,
        description="L1 penalty coefficient (ReLU SAE only)",
        gt=0.0,
    )
    k: int | None = Field(
        default=None,
        description="Number of active features (TopK/BatchTopK only)",
        gt=0,
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
        ge=0,
    )

    @model_validator(mode="after")
    def validate_architecture_params(self) -> Self:
        """Ensure architecture-specific parameters are provided."""
        if self.architecture == "relu":
            if self.l1_coefficient is None:
                raise ValueError("l1_coefficient required for ReLU SAE")
            if self.k is not None:
                raise ValueError("k should not be set for ReLU SAE")
        elif self.architecture in ["topk", "batchtopk"]:
            if self.k is None:
                raise ValueError(f"k required for {self.architecture} SAE")
            if self.l1_coefficient is not None:
                raise ValueError(f"l1_coefficient should not be set for {self.architecture} SAE")
        return self

    @field_validator("sparsity_level")
    @classmethod
    def validate_sparsity_level(cls, v: str | float) -> str | float:
        """Ensure sparsity_level is valid."""
        if isinstance(v, str):
            if v != "auto":
                raise ValueError("sparsity_level string must be 'auto'")
        elif isinstance(v, (int, float)):
            if v <= 0:
                raise ValueError("sparsity_level must be positive")
        else:
            raise ValueError("sparsity_level must be str or numeric")
        return v

    @property
    def hidden_dim(self) -> int:
        """SAE hidden dimension (number of learned features)."""
        return self.input_dim * self.expansion_factor

    @property
    def num_features(self) -> int:
        """Alias for hidden_dim (number of SAE features)."""
        return self.hidden_dim

    model_config = {"frozen": False}


class ExperimentConfig(BaseModel):
    """Full experiment specification combining all sub-configs.

    This is the top-level config that orchestrates an entire experiment:
    dataset generation, transformer training, SAE training, and logging.

    Attributes:
        experiment_name: Human-readable experiment name.
            Use descriptive names like "baseline_relu_seed42" for clarity.
        wandb_project: Weights & Biases project name.
            All runs from this project will be logged here.
        save_dir: Directory for saving checkpoints, results, and artifacts.
            Will be created if it doesn't exist.
        checkpoint_frequency: Save checkpoint every N epochs.
            Set to 0 to disable checkpointing (only save final model).
        log_frequency: Log metrics every N training steps.
            Lower = more detailed logs but slower and more W&B traffic.
        dataset: Configuration for dataset generation.
        transformer: Configuration for transformer architecture.
        sae: Configuration for SAE training.

    Example:
        >>> config = ExperimentConfig(
        ...     experiment_name="baseline_relu_seed42",
        ...     wandb_project="husai-sae-stability",
        ...     save_dir=Path("results/baseline"),
        ...     checkpoint_frequency=5,
        ...     log_frequency=100,
        ...     dataset=ModularArithmeticConfig(modulus=113, ...),
        ...     transformer=TransformerConfig(n_layers=2, ...),
        ...     sae=SAEConfig(architecture="relu", ...)
        ... )
        >>> config.save_yaml(Path("configs/baseline_relu_seed42.yaml"))
    """

    experiment_name: str = Field(
        ...,
        description="Human-readable experiment name",
        min_length=1,
        examples=["baseline_relu_seed42", "topk_8x_seed123"],
    )
    wandb_project: str = Field(
        ...,
        description="Weights & Biases project name",
        min_length=1,
        examples=["husai-sae-stability"],
    )
    save_dir: Path = Field(
        ...,
        description="Directory for saving checkpoints and results",
    )
    checkpoint_frequency: int = Field(
        ...,
        description="Save checkpoint every N epochs (0 = only final)",
        ge=0,
        examples=[0, 5, 10],
    )
    log_frequency: int = Field(
        ...,
        description="Log metrics every N training steps",
        gt=0,
        examples=[10, 50, 100, 500],
    )
    dataset: ModularArithmeticConfig = Field(
        ...,
        description="Dataset generation configuration",
    )
    transformer: TransformerConfig = Field(
        ...,
        description="Transformer architecture configuration",
    )
    sae: SAEConfig = Field(
        ...,
        description="SAE training configuration",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Ensure configs are consistent with each other."""
        # Transformer vocab_size should match dataset
        if self.transformer.vocab_size != self.dataset.vocab_size:
            raise ValueError(
                f"Transformer vocab_size ({self.transformer.vocab_size}) must match "
                f"dataset vocab_size ({self.dataset.vocab_size})"
            )

        # SAE input_dim should match transformer d_model
        if self.sae.input_dim != self.transformer.d_model:
            raise ValueError(
                f"SAE input_dim ({self.sae.input_dim}) must match "
                f"transformer d_model ({self.transformer.d_model})"
            )

        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for W&B logging.

        Returns:
            Nested dictionary with all configuration values.
            Paths are converted to strings for JSON serialization.

        Example:
            >>> config = ExperimentConfig.from_yaml("configs/baseline.yaml")
            >>> wandb.init(project=config.wandb_project, config=config.to_dict())
        """
        data = self.model_dump()
        # Convert Path to string for JSON serialization
        data["save_dir"] = str(data["save_dir"])
        return data

    def save_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file.

        Example:
            >>> config.save_yaml(Path("outputs/experiment_config.yaml"))
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # Convert to dict and handle Path serialization
            data = self.model_dump()
            data["save_dir"] = str(data["save_dir"])
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            Loaded ExperimentConfig instance.

        Raises:
            FileNotFoundError: If YAML file doesn't exist.
            ValidationError: If YAML contains invalid configuration.

        Example:
            >>> config = ExperimentConfig.from_yaml("configs/baseline.yaml")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Convert string paths back to Path objects
        if "save_dir" in data:
            data["save_dir"] = Path(data["save_dir"])

        return cls(**data)

    model_config = {"frozen": False, "arbitrary_types_allowed": True}


# ============================================================================
# Helper Functions
# ============================================================================


def create_experiment_config_from_dict(
    experiment_name: str,
    wandb_project: str,
    save_dir: Path | str,
    dataset_kwargs: dict[str, Any],
    transformer_kwargs: dict[str, Any],
    sae_kwargs: dict[str, Any],
    checkpoint_frequency: int = 5,
    log_frequency: int = 100,
) -> ExperimentConfig:
    """Helper function to create ExperimentConfig from nested dictionaries.

    Useful for programmatic config generation (e.g., hyperparameter sweeps).

    Args:
        experiment_name: Experiment name.
        wandb_project: W&B project name.
        save_dir: Save directory path.
        dataset_kwargs: Dict of ModularArithmeticConfig parameters.
        transformer_kwargs: Dict of TransformerConfig parameters.
        sae_kwargs: Dict of SAEConfig parameters.
        checkpoint_frequency: Checkpoint save frequency.
        log_frequency: Logging frequency.

    Returns:
        Fully constructed and validated ExperimentConfig.

    Example:
        >>> config = create_experiment_config_from_dict(
        ...     experiment_name="test",
        ...     wandb_project="husai",
        ...     save_dir="results/test",
        ...     dataset_kwargs={"modulus": 113, "num_samples": 10000, "train_split": 0.9},
        ...     transformer_kwargs={"n_layers": 2, "d_model": 128, "n_heads": 4,
        ...                         "d_mlp": 512, "vocab_size": 114, "max_seq_len": 3},
        ...     sae_kwargs={"architecture": "relu", "input_dim": 128, "expansion_factor": 8,
        ...                 "sparsity_level": 1e-3, "learning_rate": 3e-4,
        ...                 "batch_size": 256, "num_epochs": 10, "l1_coefficient": 1e-3}
        ... )
    """
    dataset_config = ModularArithmeticConfig(**dataset_kwargs)
    transformer_config = TransformerConfig(**transformer_kwargs)
    sae_config = SAEConfig(**sae_kwargs)

    return ExperimentConfig(
        experiment_name=experiment_name,
        wandb_project=wandb_project,
        save_dir=Path(save_dir),
        checkpoint_frequency=checkpoint_frequency,
        log_frequency=log_frequency,
        dataset=dataset_config,
        transformer=transformer_config,
        sae=sae_config,
    )


def load_and_validate_config(yaml_path: Path | str) -> ExperimentConfig:
    """Load config from YAML with comprehensive error handling.

    Args:
        yaml_path: Path to YAML config file.

    Returns:
        Validated ExperimentConfig.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config is invalid (with detailed error message).

    Example:
        >>> try:
        ...     config = load_and_validate_config("configs/baseline.yaml")
        ... except ValidationError as e:
        ...     print(f"Invalid config: {e}")
    """
    try:
        config = ExperimentConfig.from_yaml(yaml_path)
        return config
    except Exception as e:
        # Re-raise with more context
        raise type(e)(f"Failed to load config from {yaml_path}: {e}") from e
