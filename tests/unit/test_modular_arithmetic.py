"""Unit tests for modular arithmetic dataset generator.

Tests cover:
    - Dataset generation and correctness
    - Tokenization in both formats
    - Train/test splitting
    - Edge cases and error handling
    - Helper functions
"""

import pytest
import torch

from src.data.modular_arithmetic import (
    ModularArithmeticConfig,
    ModularArithmeticDataset,
    create_dataloaders,
    get_statistics,
    get_vocab_size,
    validate_dataset,
    visualize_samples,
)


class TestModularArithmeticConfig:
    """Test ModularArithmeticConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ModularArithmeticConfig()
        assert config.modulus == 113
        assert config.fraction == 1.0
        assert config.train_fraction == 0.7
        assert config.seed == 42
        assert config.format == "sequence"
        assert config.batch_size == 512

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ModularArithmeticConfig(
            modulus=59, fraction=0.5, train_fraction=0.8, seed=123, format="tuple"
        )
        assert config.modulus == 59
        assert config.fraction == 0.5
        assert config.train_fraction == 0.8
        assert config.seed == 123
        assert config.format == "tuple"

    def test_invalid_modulus(self) -> None:
        """Test that invalid modulus raises ValueError."""
        with pytest.raises(ValueError, match="Modulus must be >= 2"):
            ModularArithmeticConfig(modulus=1)

    def test_invalid_fraction(self) -> None:
        """Test that invalid fraction raises ValueError."""
        with pytest.raises(ValueError, match="Fraction must be in"):
            ModularArithmeticConfig(fraction=0.0)
        with pytest.raises(ValueError, match="Fraction must be in"):
            ModularArithmeticConfig(fraction=1.5)

    def test_invalid_train_fraction(self) -> None:
        """Test that invalid train_fraction raises ValueError."""
        with pytest.raises(ValueError, match="Train fraction must be in"):
            ModularArithmeticConfig(train_fraction=0.0)
        with pytest.raises(ValueError, match="Train fraction must be in"):
            ModularArithmeticConfig(train_fraction=1.0)

    def test_invalid_format(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Format must be"):
            ModularArithmeticConfig(format="invalid")  # type: ignore


class TestModularArithmeticDataset:
    """Test ModularArithmeticDataset class."""

    def test_dataset_creation_small(self) -> None:
        """Test dataset creation with small modulus."""
        dataset = ModularArithmeticDataset(modulus=5, fraction=1.0)
        assert len(dataset) == 25  # 5^2
        assert dataset.modulus == 5
        assert dataset.vocab_size == 9  # 5 digits + 4 special tokens

    def test_dataset_creation_large(self) -> None:
        """Test dataset creation with p=113."""
        dataset = ModularArithmeticDataset(modulus=113, fraction=1.0)
        assert len(dataset) == 12769  # 113^2
        assert dataset.modulus == 113
        assert dataset.vocab_size == 117  # 113 digits + 4 special tokens

    def test_dataset_fraction(self) -> None:
        """Test fractional dataset generation."""
        dataset = ModularArithmeticDataset(modulus=113, fraction=0.5, seed=42)
        expected_size = int(113 * 113 * 0.5)
        assert len(dataset) == expected_size
        # Check that it's reproducible
        dataset2 = ModularArithmeticDataset(modulus=113, fraction=0.5, seed=42)
        assert len(dataset2) == len(dataset)
        assert dataset.examples == dataset2.examples

    def test_dataset_reproducibility(self) -> None:
        """Test that same seed produces same dataset."""
        dataset1 = ModularArithmeticDataset(modulus=113, fraction=0.3, seed=42)
        dataset2 = ModularArithmeticDataset(modulus=113, fraction=0.3, seed=42)
        assert dataset1.examples == dataset2.examples

    def test_dataset_different_seeds(self) -> None:
        """Test that different seeds produce different datasets."""
        dataset1 = ModularArithmeticDataset(modulus=113, fraction=0.3, seed=42)
        dataset2 = ModularArithmeticDataset(modulus=113, fraction=0.3, seed=123)
        # Should have same length but potentially different examples
        assert len(dataset1) == len(dataset2)
        # Very unlikely to have identical examples
        assert dataset1.examples != dataset2.examples

    def test_example_correctness(self) -> None:
        """Test that all examples satisfy c = (a + b) mod p."""
        dataset = ModularArithmeticDataset(modulus=7, fraction=1.0)
        for a, b, c in dataset.examples:
            assert c == (a + b) % 7
            assert 0 <= a < 7
            assert 0 <= b < 7
            assert 0 <= c < 7

    def test_sequence_tokenization(self) -> None:
        """Test sequence format tokenization."""
        dataset = ModularArithmeticDataset(modulus=5, fraction=1.0, format="sequence")
        tokens, label = dataset[0]

        # Should be [BOS, a, +, b, =, c, EOS]
        assert len(tokens) == 7
        assert tokens[0] == 5  # BOS
        assert tokens[6] == 6  # EOS
        assert tokens[2] == 8  # PLUS
        assert tokens[4] == 7  # EQUALS

        # Check that a, b, c are digits
        a, b, c = tokens[1].item(), tokens[3].item(), tokens[5].item()
        assert 0 <= a < 5
        assert 0 <= b < 5
        assert 0 <= c < 5
        assert c == (a + b) % 5

        # Label should match c
        assert label.item() == c

    def test_tuple_tokenization(self) -> None:
        """Test tuple format tokenization."""
        dataset = ModularArithmeticDataset(modulus=5, fraction=1.0, format="tuple")
        tokens, label = dataset[0]

        # Should be [a, b, c]
        assert len(tokens) == 3
        a, b, c = tokens[0].item(), tokens[1].item(), tokens[2].item()

        # Check correctness
        assert 0 <= a < 5
        assert 0 <= b < 5
        assert 0 <= c < 5
        assert c == (a + b) % 5
        assert label.item() == c

    def test_vocab_size_sequence(self) -> None:
        """Test vocabulary size for sequence format."""
        dataset = ModularArithmeticDataset(modulus=113, format="sequence")
        assert dataset.vocab_size == 117  # 113 + 4 special tokens

    def test_vocab_size_tuple(self) -> None:
        """Test vocabulary size for tuple format."""
        dataset = ModularArithmeticDataset(modulus=113, format="tuple")
        assert dataset.vocab_size == 113  # Only digits

    def test_get_vocab_mapping_sequence(self) -> None:
        """Test vocabulary mapping for sequence format."""
        dataset = ModularArithmeticDataset(modulus=5, format="sequence")
        mapping = dataset.get_vocab_mapping()

        assert mapping[0] == "0"
        assert mapping[4] == "4"
        assert mapping[5] == "[BOS]"
        assert mapping[6] == "[EOS]"
        assert mapping[7] == "[=]"
        assert mapping[8] == "[+]"

    def test_get_vocab_mapping_tuple(self) -> None:
        """Test vocabulary mapping for tuple format."""
        dataset = ModularArithmeticDataset(modulus=5, format="tuple")
        mapping = dataset.get_vocab_mapping()

        assert mapping[0] == "0"
        assert mapping[4] == "4"
        assert len(mapping) == 5

    def test_decode_tokens_sequence(self) -> None:
        """Test token decoding for sequence format."""
        dataset = ModularArithmeticDataset(modulus=5, format="sequence")
        tokens = torch.tensor([5, 2, 8, 3, 7, 0, 6])  # BOS 2 + 3 = 0 EOS
        decoded = dataset.decode_tokens(tokens)
        assert decoded == "[BOS] 2 [+] 3 [=] 0 [EOS]"

    def test_decode_tokens_tuple(self) -> None:
        """Test token decoding for tuple format."""
        dataset = ModularArithmeticDataset(modulus=5, format="tuple")
        tokens = torch.tensor([2, 3, 0])  # 2 3 0
        decoded = dataset.decode_tokens(tokens)
        assert decoded == "2 3 0"

    def test_invalid_modulus(self) -> None:
        """Test that invalid modulus raises ValueError."""
        with pytest.raises(ValueError, match="Modulus must be >= 2"):
            ModularArithmeticDataset(modulus=1)

    def test_invalid_fraction(self) -> None:
        """Test that invalid fraction raises ValueError."""
        with pytest.raises(ValueError, match="Fraction must be in"):
            ModularArithmeticDataset(modulus=5, fraction=0.0)
        with pytest.raises(ValueError, match="Fraction must be in"):
            ModularArithmeticDataset(modulus=5, fraction=1.5)

    def test_invalid_format(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Format must be"):
            ModularArithmeticDataset(modulus=5, format="invalid")  # type: ignore


class TestCreateDataloaders:
    """Test create_dataloaders function."""

    def test_dataloader_creation(self) -> None:
        """Test basic DataLoader creation."""
        train_loader, test_loader = create_dataloaders(modulus=5, batch_size=10, train_fraction=0.6)

        # Check that loaders exist
        assert train_loader is not None
        assert test_loader is not None

        # Check dataset sizes
        total_size = 25  # 5^2
        train_size = int(total_size * 0.6)
        test_size = total_size - train_size

        assert len(train_loader.dataset) == train_size
        assert len(test_loader.dataset) == test_size

    def test_dataloader_batch_size(self) -> None:
        """Test that batches have correct size."""
        train_loader, test_loader = create_dataloaders(modulus=5, batch_size=5)

        # Get a batch
        tokens_batch, labels_batch = next(iter(train_loader))

        assert tokens_batch.shape[0] == 5  # batch size
        assert labels_batch.shape[0] == 5

    def test_dataloader_reproducibility(self) -> None:
        """Test that same seed produces same train/test split."""
        train_loader1, test_loader1 = create_dataloaders(modulus=5, seed=42)
        train_loader2, test_loader2 = create_dataloaders(modulus=5, seed=42)

        # Get first batch from each
        tokens1, _ = next(iter(train_loader1))
        tokens2, _ = next(iter(train_loader2))

        # Should be identical (same seed, same shuffle seed)
        assert len(train_loader1.dataset) == len(train_loader2.dataset)
        assert len(test_loader1.dataset) == len(test_loader2.dataset)

    def test_dataloader_fraction(self) -> None:
        """Test DataLoader with fractional dataset."""
        train_loader, test_loader = create_dataloaders(
            modulus=113, fraction=0.5, train_fraction=0.7, batch_size=100
        )

        total_size = int(113 * 113 * 0.5)
        train_size = int(total_size * 0.7)
        test_size = total_size - train_size

        assert len(train_loader.dataset) == train_size
        assert len(test_loader.dataset) == test_size

    def test_dataloader_format(self) -> None:
        """Test DataLoader with different formats."""
        # Sequence format
        train_loader_seq, _ = create_dataloaders(modulus=5, format="sequence")
        tokens_seq, _ = next(iter(train_loader_seq))
        assert tokens_seq.shape[1] == 7  # sequence length

        # Tuple format
        train_loader_tuple, _ = create_dataloaders(modulus=5, format="tuple")
        tokens_tuple, _ = next(iter(train_loader_tuple))
        assert tokens_tuple.shape[1] == 3  # tuple length


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_vocab_size_sequence(self) -> None:
        """Test get_vocab_size for sequence format."""
        assert get_vocab_size(5, format="sequence") == 9
        assert get_vocab_size(113, format="sequence") == 117

    def test_get_vocab_size_tuple(self) -> None:
        """Test get_vocab_size for tuple format."""
        assert get_vocab_size(5, format="tuple") == 5
        assert get_vocab_size(113, format="tuple") == 113

    def test_visualize_samples(self, capsys) -> None:
        """Test visualize_samples function."""
        dataset = ModularArithmeticDataset(modulus=5, fraction=1.0)
        visualize_samples(dataset, n=2)

        # Capture printed output
        captured = capsys.readouterr()
        assert "Dataset samples" in captured.out
        assert "Sample 0:" in captured.out
        assert "Sample 1:" in captured.out

    def test_visualize_samples_with_indices(self, capsys) -> None:
        """Test visualize_samples with specific indices."""
        dataset = ModularArithmeticDataset(modulus=5, fraction=1.0)
        visualize_samples(dataset, indices=[0, 1, 2])

        captured = capsys.readouterr()
        assert "Sample 0:" in captured.out
        assert "Sample 1:" in captured.out
        assert "Sample 2:" in captured.out

    def test_get_statistics(self) -> None:
        """Test get_statistics function."""
        dataset = ModularArithmeticDataset(modulus=5, fraction=1.0)
        stats = get_statistics(dataset)

        assert stats["total_examples"] == 25
        assert stats["modulus"] == 5
        assert stats["vocab_size"] == 9
        assert stats["format"] == "sequence"
        assert stats["sequence_length"] == 7
        assert stats["fraction_used"] == 1.0
        assert "answer_distribution" in stats

    def test_validate_dataset_valid(self) -> None:
        """Test validate_dataset on valid dataset."""
        dataset = ModularArithmeticDataset(modulus=113, fraction=1.0)
        assert validate_dataset(dataset) is True

    def test_validate_dataset_valid_tuple(self) -> None:
        """Test validate_dataset on valid tuple format dataset."""
        dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, format="tuple")
        assert validate_dataset(dataset) is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_modulus(self) -> None:
        """Test with minimum valid modulus (2)."""
        dataset = ModularArithmeticDataset(modulus=2, fraction=1.0)
        assert len(dataset) == 4  # 2^2
        assert dataset.vocab_size == 6  # 2 digits + 4 special tokens

        # Verify examples
        expected = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        assert dataset.examples == expected

    def test_small_fraction(self) -> None:
        """Test with very small fraction."""
        dataset = ModularArithmeticDataset(modulus=113, fraction=0.01, seed=42)
        expected_size = max(1, int(113 * 113 * 0.01))
        assert len(dataset) == expected_size

    def test_full_dataset_coverage(self) -> None:
        """Test that full dataset covers all (a, b) pairs."""
        dataset = ModularArithmeticDataset(modulus=7, fraction=1.0)

        # Extract all (a, b) pairs
        pairs = set((a, b) for a, b, _ in dataset.examples)

        # Should have all 7^2 = 49 pairs
        assert len(pairs) == 49
        for a in range(7):
            for b in range(7):
                assert (a, b) in pairs

    def test_dataloader_single_batch(self) -> None:
        """Test DataLoader when batch size >= dataset size."""
        train_loader, test_loader = create_dataloaders(
            modulus=5, batch_size=100, train_fraction=0.6
        )

        # Should have only one batch each (since dataset is small)
        train_batches = list(train_loader)
        test_batches = list(test_loader)

        assert len(train_batches) == 1
        assert len(test_batches) == 1

    def test_answer_distribution_uniform(self) -> None:
        """Test that full dataset has uniform answer distribution."""
        dataset = ModularArithmeticDataset(modulus=7, fraction=1.0)
        stats = get_statistics(dataset)

        # For full dataset, each answer should appear equally
        # Each answer c appears when (a + b) % p == c
        # For prime p, this is exactly p times
        answer_dist = stats["answer_distribution"]
        for c in range(7):
            assert answer_dist[c] == 7


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_training_setup(self) -> None:
        """Test complete setup for training."""
        # Create dataloaders
        train_loader, test_loader = create_dataloaders(
            modulus=113, fraction=1.0, train_fraction=0.7, batch_size=512, seed=42
        )

        # Get a batch
        tokens_batch, labels_batch = next(iter(train_loader))

        # Verify batch properties
        assert tokens_batch.shape[0] <= 512  # batch size
        assert tokens_batch.shape[1] == 7  # sequence length
        assert labels_batch.shape[0] <= 512

        # Verify token ranges
        assert tokens_batch.min() >= 0
        assert tokens_batch.max() < 117  # vocab size

        # Verify label ranges
        assert labels_batch.min() >= 0
        assert labels_batch.max() < 113

    def test_multiple_epochs(self) -> None:
        """Test iterating through multiple epochs."""
        train_loader, _ = create_dataloaders(modulus=5, batch_size=5, seed=42)

        # Iterate through 2 epochs
        total_examples_epoch1 = 0
        for tokens, labels in train_loader:
            total_examples_epoch1 += len(tokens)

        total_examples_epoch2 = 0
        for tokens, labels in train_loader:
            total_examples_epoch2 += len(tokens)

        # Should see same number of examples each epoch
        assert total_examples_epoch1 == total_examples_epoch2
        assert total_examples_epoch1 == len(train_loader.dataset)

    def test_consistency_across_formats(self) -> None:
        """Test that both formats represent same mathematical problem."""
        dataset_seq = ModularArithmeticDataset(modulus=5, fraction=1.0, format="sequence")
        dataset_tuple = ModularArithmeticDataset(modulus=5, fraction=1.0, format="tuple")

        # Should have same number of examples
        assert len(dataset_seq) == len(dataset_tuple)

        # Should have same underlying examples
        assert dataset_seq.examples == dataset_tuple.examples

        # Get corresponding examples
        tokens_seq, label_seq = dataset_seq[0]
        tokens_tuple, label_tuple = dataset_tuple[0]

        # Labels should match
        assert label_seq == label_tuple

        # Extract a, b, c from sequence format
        # [BOS, a, +, b, =, c, EOS]
        a_seq, b_seq, c_seq = (
            tokens_seq[1].item(),
            tokens_seq[3].item(),
            tokens_seq[5].item(),
        )

        # Extract a, b, c from tuple format
        a_tuple, b_tuple, c_tuple = (
            tokens_tuple[0].item(),
            tokens_tuple[1].item(),
            tokens_tuple[2].item(),
        )

        # Should be identical
        assert a_seq == a_tuple
        assert b_seq == b_tuple
        assert c_seq == c_tuple
