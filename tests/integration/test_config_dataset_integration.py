"""Integration tests for config-dataset compatibility.

This module tests that the configuration system properly integrates with
the modular arithmetic dataset, ensuring no vocabulary size mismatches
or other integration issues that could cause runtime crashes.

These tests catch bugs that unit tests might miss, such as:
- Vocab size mismatches between config and dataset
- Embedding layer dimension errors
- Token index out of bounds errors
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from src.utils.config import ExperimentConfig
from src.data.modular_arithmetic import create_dataloaders


class TestConfigDatasetIntegration:
    """Integration tests ensuring config and dataset work together."""

    def test_vocab_size_consistency_sequence_format(self):
        """Test that config vocab_size matches dataset vocab_size for sequence format."""
        # Load a config file
        config_path = Path("configs/examples/baseline_relu.yaml")
        config = ExperimentConfig.from_yaml(config_path)

        # Create dataloaders using dataset params
        train_loader, _ = create_dataloaders(
            modulus=config.dataset.modulus,
            fraction=1.0,
            train_fraction=config.dataset.train_split,
            batch_size=32,
            seed=config.dataset.seed,
            format="sequence",  # Sequence format uses p + 4 tokens
        )

        # Get actual dataset vocab size
        # For sequence format: BOS, EOS, EQUALS, PLUS + digits 0 to p-1
        expected_vocab_size = config.dataset.modulus + 4

        # Check config matches
        assert config.transformer.vocab_size == expected_vocab_size, (
            f"Config vocab_size ({config.transformer.vocab_size}) "
            f"doesn't match expected ({expected_vocab_size}) for modulus={config.dataset.modulus}"
        )

        # Verify dataset uses correct vocab size property
        assert config.dataset.vocab_size == expected_vocab_size

    def test_embedding_doesnt_crash_with_actual_tokens(self):
        """Test that embedding layer doesn't crash when used with actual dataset tokens."""
        # Load config
        config_path = Path("configs/examples/baseline_relu.yaml")
        config = ExperimentConfig.from_yaml(config_path)

        # Create embedding layer with config vocab_size
        embedding = nn.Embedding(
            num_embeddings=config.transformer.vocab_size,
            embedding_dim=config.transformer.d_model,
        )

        # Create dataloaders
        train_loader, _ = create_dataloaders(
            modulus=config.dataset.modulus,
            fraction=0.1,  # Small fraction for fast test
            train_fraction=0.8,
            batch_size=16,
            seed=42,
            format="sequence",
        )

        # Get a batch
        tokens, labels = next(iter(train_loader))

        # Verify token shapes
        assert tokens.shape == (16, 7), f"Expected (16, 7), got {tokens.shape}"

        # Verify all token indices are within valid range
        max_token_idx = tokens.max().item()
        assert max_token_idx < config.transformer.vocab_size, (
            f"Token index {max_token_idx} exceeds vocab_size {config.transformer.vocab_size}"
        )

        # This should NOT crash - the critical test
        try:
            embedded = embedding(tokens)
            assert embedded.shape == (16, 7, config.transformer.d_model)
        except IndexError as e:
            pytest.fail(
                f"Embedding crashed with IndexError: {e}. "
                f"Token indices: {tokens.unique().tolist()}, "
                f"Vocab size: {config.transformer.vocab_size}"
            )

    def test_all_config_files_have_correct_vocab_size(self):
        """Test that all example config files have correct vocab_size."""
        config_dir = Path("configs/examples")
        config_files = list(config_dir.glob("*.yaml"))

        assert len(config_files) >= 3, "Expected at least 3 example configs"

        for config_path in config_files:
            config = ExperimentConfig.from_yaml(config_path)

            # Expected vocab size for sequence format
            expected = config.dataset.modulus + 4

            assert config.transformer.vocab_size == expected, (
                f"Config {config_path.name} has vocab_size={config.transformer.vocab_size}, "
                f"expected {expected} for modulus={config.dataset.modulus}"
            )

            # Verify consistency between dataset and transformer configs
            assert config.dataset.vocab_size == config.transformer.vocab_size

    def test_max_seq_len_matches_sequence_format(self):
        """Test that max_seq_len is correct for sequence format (7 tokens)."""
        config_path = Path("configs/examples/baseline_relu.yaml")
        config = ExperimentConfig.from_yaml(config_path)

        # Sequence format: [BOS, a, +, b, =, c, EOS] = 7 tokens
        assert config.transformer.max_seq_len == 7, (
            f"max_seq_len should be 7 for sequence format, got {config.transformer.max_seq_len}"
        )

        # Verify with actual dataset
        train_loader, _ = create_dataloaders(
            modulus=config.dataset.modulus,
            fraction=0.01,
            batch_size=8,
            format="sequence",
        )

        tokens, _ = next(iter(train_loader))
        seq_len = tokens.shape[1]

        assert seq_len == 7, f"Dataset produces {seq_len} tokens, config expects {config.transformer.max_seq_len}"
        assert seq_len <= config.transformer.max_seq_len, (
            f"Dataset sequence length ({seq_len}) exceeds model max_seq_len ({config.transformer.max_seq_len})"
        )

    def test_config_validation_catches_mismatch(self):
        """Test that ExperimentConfig validation catches vocab_size mismatches."""
        from src.utils.config import ModularArithmeticConfig, TransformerConfig, SAEConfig
        from pydantic import ValidationError

        # Create configs with intentional mismatch
        dataset_config = ModularArithmeticConfig(
            modulus=113,
            num_samples=1000,
            train_split=0.9,
        )

        transformer_config = TransformerConfig(
            n_layers=2,
            d_model=128,
            n_heads=4,
            d_mlp=512,
            vocab_size=999,  # WRONG! Should be 117
            max_seq_len=7,
        )

        sae_config = SAEConfig(
            architecture="relu",
            input_dim=128,
            expansion_factor=4,
            sparsity_level=1e-3,
            learning_rate=3e-4,
            batch_size=256,
            num_epochs=10,
            l1_coefficient=1e-3,
        )

        # Creating ExperimentConfig should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(
                experiment_name="test",
                wandb_project="test",
                save_dir=Path("test"),
                checkpoint_frequency=0,
                log_frequency=100,
                dataset=dataset_config,
                transformer=transformer_config,
                sae=sae_config,
            )

        # Check error message mentions vocab_size mismatch
        error_msg = str(exc_info.value)
        assert "vocab_size" in error_msg.lower(), f"Expected vocab_size error, got: {error_msg}"

    def test_tuple_format_would_use_different_vocab_size(self):
        """Document that tuple format ([a, b, c]) would use modulus tokens only.

        Note: Current configs all use sequence format. This test documents the
        alternative if we switch to tuple format in the future.
        """
        modulus = 113

        # Sequence format: p + 4
        sequence_vocab = modulus + 4  # 117

        # Tuple format would only need: p (just the digits 0 to p-1)
        tuple_vocab = modulus  # 113

        assert sequence_vocab == 117
        assert tuple_vocab == 113
        assert sequence_vocab > tuple_vocab

        # Current config uses sequence format
        config_path = Path("configs/examples/baseline_relu.yaml")
        config = ExperimentConfig.from_yaml(config_path)
        assert config.transformer.vocab_size == sequence_vocab, "Configs should use sequence format"

    def test_special_token_indices_are_valid(self):
        """Test that special token indices (BOS, EOS, EQUALS, PLUS) are within vocab."""
        modulus = 113
        vocab_size = modulus + 4  # 117

        # Special tokens should be: p, p+1, p+2, p+3
        bos = modulus          # 113
        eos = modulus + 1      # 114
        equals = modulus + 2   # 115
        plus = modulus + 3     # 116

        # All should be less than vocab_size
        assert bos < vocab_size, f"BOS={bos} >= vocab_size={vocab_size}"
        assert eos < vocab_size, f"EOS={eos} >= vocab_size={vocab_size}"
        assert equals < vocab_size, f"EQUALS={equals} >= vocab_size={vocab_size}"
        assert plus < vocab_size, f"PLUS={plus} >= vocab_size={vocab_size}"

        # Highest token index
        max_token = plus  # 116
        assert max_token == vocab_size - 1, "Highest token should be vocab_size - 1"
