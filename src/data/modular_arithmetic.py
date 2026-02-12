"""Modular Arithmetic Dataset Generator for Grokking Research.

This module implements dataset generation for modular addition tasks, following
Nanda et al. (2023) "Progress Measures for Grokking via Mechanistic Interpretability".

The task is to predict: c = (a + b) mod p, where p is prime.

Key features:
- Deterministic generation with seed control
- Full dataset or random sampling modes
- PyTorch Dataset interface
- Flexible tokenization (sequence or tuple format)
- Train/test splitting with configurable ratios

Example usage:
    >>> from src.data.modular_arithmetic import ModularArithmeticDataset
    >>> dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42)
    >>> train_loader, test_loader = create_dataloaders(modulus=113, batch_size=512)
    >>> print(f"Vocabulary size: {get_vocab_size(113)}")
    >>> visualize_samples(dataset, n=5)

References:
    - Nanda et al. (2023): https://arxiv.org/abs/2301.05217
    - Power et al. (2022): https://arxiv.org/abs/2201.02177
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class ModularArithmeticConfig:
    """Configuration for modular arithmetic dataset generation.

    Attributes:
        modulus: Prime modulus p for modular arithmetic (a + b) mod p
        fraction: Fraction of total dataset to use (1.0 = all p^2 examples)
        train_fraction: Fraction of dataset to use for training
        seed: Random seed for reproducibility
        format: Token sequence format ('sequence' or 'tuple')
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        device: Device to load data on ('cpu', 'cuda', or 'mps')
    """

    modulus: int = 113
    fraction: float = 1.0
    train_fraction: float = 0.7
    seed: int = 42
    format: Literal["sequence", "tuple"] = "sequence"
    batch_size: int = 512
    num_workers: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.modulus < 2:
            raise ValueError(f"Modulus must be >= 2, got {self.modulus}")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"Fraction must be in (0, 1], got {self.fraction}")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError(f"Train fraction must be in (0, 1), got {self.train_fraction}")
        if self.format not in ["sequence", "tuple"]:
            raise ValueError(f"Format must be 'sequence' or 'tuple', got {self.format}")


class ModularArithmeticDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for modular arithmetic tasks.

    Generates all possible (a, b, c) tuples where c = (a + b) mod p, or a random
    subset if fraction < 1.0.

    The dataset supports two token formats:
        - 'sequence': [BOS, a, +, b, =, c, EOS] (length 7)
        - 'tuple': [a, b, c] (length 3)

    Attributes:
        modulus: Prime modulus p
        fraction: Fraction of total dataset to use
        seed: Random seed for reproducibility
        format: Token sequence format
        examples: List of (a, b, c) tuples
        vocab_size: Total vocabulary size including special tokens

    Special tokens (sequence format only):
        - BOS (Begin of Sequence): modulus
        - EOS (End of Sequence): modulus + 1
        - =: modulus + 2
        - +: modulus + 3
        - Digits 0 to p-1: themselves

    Example:
        >>> dataset = ModularArithmeticDataset(modulus=5, fraction=1.0, seed=42)
        >>> len(dataset)
        25
        >>> tokens, label = dataset[0]
        >>> tokens.shape
        torch.Size([7])
    """

    # Special token indices (for sequence format)
    SPECIAL_TOKENS = {
        "BOS": "modulus",
        "EOS": "modulus + 1",
        "EQUALS": "modulus + 2",
        "PLUS": "modulus + 3",
    }

    def __init__(
        self,
        modulus: int = 113,
        fraction: float = 1.0,
        seed: int = 42,
        format: Literal["sequence", "tuple"] = "sequence",
    ) -> None:
        """Initialize modular arithmetic dataset.

        Args:
            modulus: Prime modulus p for (a + b) mod p
            fraction: Fraction of total p^2 examples to use (1.0 = all)
            seed: Random seed for reproducible sampling
            format: Token sequence format ('sequence' or 'tuple')

        Raises:
            ValueError: If modulus < 2, fraction not in (0, 1], or invalid format
        """
        if modulus < 2:
            raise ValueError(f"Modulus must be >= 2, got {modulus}")
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Fraction must be in (0, 1], got {fraction}")
        if format not in ["sequence", "tuple"]:
            raise ValueError(f"Format must be 'sequence' or 'tuple', got {format}")

        self.modulus = modulus
        self.fraction = fraction
        self.seed = seed
        self.format = format

        # Generate dataset
        self.examples = self._generate_examples()

        # Compute vocabulary size
        self.vocab_size = self._compute_vocab_size()

    def _generate_examples(self) -> List[Tuple[int, int, int]]:
        """Generate all (a, b, c) examples where c = (a + b) mod p.

        Returns:
            List of (a, b, c) tuples

        Example:
            >>> dataset = ModularArithmeticDataset(modulus=3, fraction=1.0)
            >>> dataset.examples
            [(0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 0, 1), (1, 1, 2),
             (1, 2, 0), (2, 0, 2), (2, 1, 0), (2, 2, 1)]
        """
        # Generate all possible pairs
        all_examples = [
            (a, b, (a + b) % self.modulus) for a in range(self.modulus) for b in range(self.modulus)
        ]

        # Sample if fraction < 1.0
        if self.fraction < 1.0:
            rng = np.random.RandomState(self.seed)
            n_samples = max(1, int(len(all_examples) * self.fraction))
            indices = rng.choice(len(all_examples), size=n_samples, replace=False)
            all_examples = [all_examples[i] for i in sorted(indices.tolist())]

        return all_examples

    def _compute_vocab_size(self) -> int:
        """Compute vocabulary size based on format.

        Returns:
            Vocabulary size (number of unique tokens)

        Example:
            >>> dataset = ModularArithmeticDataset(modulus=5, format='sequence')
            >>> dataset.vocab_size
            9  # 0-4 (digits) + BOS + EOS + EQUALS + PLUS
        """
        if self.format == "sequence":
            # Digits (0 to p-1) + BOS + EOS + EQUALS + PLUS
            return self.modulus + 4
        else:  # tuple
            # Only digits (0 to p-1)
            return self.modulus

    def _tokenize_sequence(self, a: int, b: int, c: int) -> torch.Tensor:
        """Tokenize example in sequence format: [BOS, a, +, b, =, c, EOS].

        Args:
            a: First operand
            b: Second operand
            c: Result (a + b) mod p

        Returns:
            Token sequence of length 7

        Example:
            >>> dataset = ModularArithmeticDataset(modulus=5, format='sequence')
            >>> tokens = dataset._tokenize_sequence(2, 3, 0)
            >>> tokens.tolist()
            [5, 2, 8, 3, 7, 0, 6]  # [BOS, 2, +, 3, =, 0, EOS]
        """
        bos = self.modulus
        eos = self.modulus + 1
        equals = self.modulus + 2
        plus = self.modulus + 3

        return torch.tensor([bos, a, plus, b, equals, c, eos], dtype=torch.long)

    def _tokenize_tuple(self, a: int, b: int, c: int) -> torch.Tensor:
        """Tokenize example in tuple format: [a, b, c].

        Args:
            a: First operand
            b: Second operand
            c: Result (a + b) mod p

        Returns:
            Token sequence of length 3

        Example:
            >>> dataset = ModularArithmeticDataset(modulus=5, format='tuple')
            >>> tokens = dataset._tokenize_tuple(2, 3, 0)
            >>> tokens.tolist()
            [2, 3, 0]
        """
        return torch.tensor([a, b, c], dtype=torch.long)

    def __len__(self) -> int:
        """Return number of examples in dataset.

        Returns:
            Number of examples

        Example:
            >>> dataset = ModularArithmeticDataset(modulus=113, fraction=0.5)
            >>> len(dataset)
            6384  # Approximately 113^2 * 0.5
        """
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single example from the dataset.

        Args:
            idx: Index of example to retrieve

        Returns:
            Tuple of (tokens, label) where:
                - tokens: Input token sequence (length 7 or 3 depending on format)
                - label: Target answer (scalar)

        Example:
            >>> dataset = ModularArithmeticDataset(modulus=5, format='sequence')
            >>> tokens, label = dataset[0]
            >>> tokens.shape, label.shape
            (torch.Size([7]), torch.Size([]))
        """
        a, b, c = self.examples[idx]

        # Tokenize based on format
        if self.format == "sequence":
            tokens = self._tokenize_sequence(a, b, c)
        else:  # tuple
            tokens = self._tokenize_tuple(a, b, c)

        # Label is the answer (c)
        label = torch.tensor(c, dtype=torch.long)

        return tokens, label

    def get_vocab_mapping(self) -> Dict[int, str]:
        """Get mapping from token indices to human-readable strings.

        Returns:
            Dictionary mapping token index to string representation

        Example:
            >>> dataset = ModularArithmeticDataset(modulus=5, format='sequence')
            >>> mapping = dataset.get_vocab_mapping()
            >>> mapping[5], mapping[6], mapping[7], mapping[8]
            ('[BOS]', '[EOS]', '[=]', '[+]')
        """
        if self.format == "sequence":
            mapping = {i: str(i) for i in range(self.modulus)}
            mapping[self.modulus] = "[BOS]"
            mapping[self.modulus + 1] = "[EOS]"
            mapping[self.modulus + 2] = "[=]"
            mapping[self.modulus + 3] = "[+]"
            return mapping
        else:  # tuple
            return {i: str(i) for i in range(self.modulus)}

    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """Decode token sequence to human-readable string.

        Args:
            tokens: Token sequence (1D tensor)

        Returns:
            Human-readable string representation

        Example:
            >>> dataset = ModularArithmeticDataset(modulus=5, format='sequence')
            >>> tokens = torch.tensor([5, 2, 8, 3, 7, 0, 6])
            >>> dataset.decode_tokens(tokens)
            '[BOS] 2 [+] 3 [=] 0 [EOS]'
        """
        mapping = self.get_vocab_mapping()
        return " ".join(mapping[int(t)] for t in tokens)


def create_dataloaders(
    modulus: int = 113,
    fraction: float = 1.0,
    train_fraction: float = 0.7,
    batch_size: int = 512,
    seed: int = 42,
    format: Literal["sequence", "tuple"] = "sequence",
    num_workers: int = 0,
    device: str = "cpu",
) -> Tuple[
    DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    DataLoader[Tuple[torch.Tensor, torch.Tensor]],
]:
    """Create train and test DataLoaders for modular arithmetic.

    Args:
        modulus: Prime modulus p for (a + b) mod p
        fraction: Fraction of total p^2 examples to use
        train_fraction: Fraction of dataset to use for training
        batch_size: Batch size for both train and test loaders
        seed: Random seed for train/test split
        format: Token sequence format ('sequence' or 'tuple')
        num_workers: Number of DataLoader workers
        device: Device to load data on ('cpu', 'cuda', or 'mps')

    Returns:
        Tuple of (train_loader, test_loader)

    Example:
        >>> train_loader, test_loader = create_dataloaders(
        ...     modulus=113, batch_size=512, train_fraction=0.7
        ... )
        >>> len(train_loader.dataset), len(test_loader.dataset)
        (8938, 3831)  # 70% and 30% of 113^2 = 12769
    """
    # Create full dataset
    dataset = ModularArithmeticDataset(modulus=modulus, fraction=fraction, seed=seed, format=format)

    # Split into train/test
    train_size = int(len(dataset) * train_fraction)
    test_size = len(dataset) - train_size

    # Use generator for reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device in ["cuda", "mps"]),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device in ["cuda", "mps"]),
    )

    return train_loader, test_loader


def get_vocab_size(modulus: int, format: Literal["sequence", "tuple"] = "sequence") -> int:
    """Get vocabulary size for given modulus and format.

    Args:
        modulus: Prime modulus p
        format: Token sequence format ('sequence' or 'tuple')

    Returns:
        Vocabulary size (number of unique tokens)

    Example:
        >>> get_vocab_size(113, format='sequence')
        117  # 113 digits + 4 special tokens
        >>> get_vocab_size(113, format='tuple')
        113  # Only digits
    """
    if format == "sequence":
        return modulus + 4  # digits + BOS + EOS + EQUALS + PLUS
    else:  # tuple
        return modulus


def visualize_samples(
    dataset: ModularArithmeticDataset, n: int = 5, indices: Optional[List[int]] = None
) -> None:
    """Print n random samples from dataset in human-readable format.

    Args:
        dataset: ModularArithmeticDataset to visualize
        n: Number of samples to display
        indices: Specific indices to display (overrides n if provided)

    Example:
        >>> dataset = ModularArithmeticDataset(modulus=113, fraction=1.0)
        >>> visualize_samples(dataset, n=3)
        Sample 0: [BOS] 42 [+] 71 [=] 0 [EOS] -> 0
        Sample 1: [BOS] 15 [+] 98 [=] 0 [EOS] -> 0
        Sample 2: [BOS] 67 [+] 23 [=] 90 [EOS] -> 90
    """
    if indices is None:
        # Sample random indices
        rng = np.random.RandomState(42)
        indices = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False).tolist()

    print(f"Dataset samples (modulus={dataset.modulus}, format={dataset.format}):")
    print("-" * 60)

    for i, idx in enumerate(indices):
        tokens, label = dataset[idx]
        decoded = dataset.decode_tokens(tokens)
        print(f"Sample {i}: {decoded} -> {label.item()}")

        # Also show the raw example for reference
        a, b, c = dataset.examples[idx]
        print(f"         (Raw: {a} + {b} = {c} mod {dataset.modulus})")
        print()


def get_statistics(dataset: ModularArithmeticDataset) -> Dict[str, Any]:
    """Compute dataset statistics.

    Args:
        dataset: ModularArithmeticDataset to analyze

    Returns:
        Dictionary with statistics including:
            - total_examples: Total number of examples
            - modulus: Prime modulus used
            - vocab_size: Vocabulary size
            - format: Token format
            - sequence_length: Length of token sequences
            - answer_distribution: Distribution of answer values

    Example:
        >>> dataset = ModularArithmeticDataset(modulus=5, fraction=1.0)
        >>> stats = get_statistics(dataset)
        >>> stats['total_examples']
        25
        >>> stats['vocab_size']
        9
    """
    # Compute answer distribution
    answers = [c for _, _, c in dataset.examples]
    answer_counts = {i: answers.count(i) for i in range(dataset.modulus)}

    sequence_length = 7 if dataset.format == "sequence" else 3

    return {
        "total_examples": len(dataset),
        "modulus": dataset.modulus,
        "vocab_size": dataset.vocab_size,
        "format": dataset.format,
        "sequence_length": sequence_length,
        "answer_distribution": answer_counts,
        "fraction_used": dataset.fraction,
    }


def validate_dataset(dataset: ModularArithmeticDataset) -> bool:
    """Validate that dataset is correctly generated.

    Checks:
        1. All examples satisfy c = (a + b) mod p
        2. Vocabulary size is correct
        3. Token sequences have correct length
        4. All tokens are in valid range

    Args:
        dataset: ModularArithmeticDataset to validate

    Returns:
        True if dataset is valid, False otherwise

    Example:
        >>> dataset = ModularArithmeticDataset(modulus=113)
        >>> validate_dataset(dataset)
        True
    """
    try:
        # Check all examples
        for a, b, c in dataset.examples:
            # Validate equation
            if c != (a + b) % dataset.modulus:
                print(f"Invalid example: {a} + {b} = {c} (expected {(a + b) % dataset.modulus})")
                return False

            # Validate ranges
            if not (0 <= a < dataset.modulus and 0 <= b < dataset.modulus):
                print(f"Out of range operands: a={a}, b={b}")
                return False

        # Check tokenization
        for idx in range(min(100, len(dataset))):  # Sample first 100
            tokens, label = dataset[idx]

            # Check sequence length
            expected_len = 7 if dataset.format == "sequence" else 3
            if len(tokens) != expected_len:
                print(f"Invalid token length: {len(tokens)} (expected {expected_len})")
                return False

            # Check token ranges
            max_token = dataset.vocab_size - 1
            if tokens.max() > max_token or tokens.min() < 0:
                print(f"Token out of range: {tokens}")
                return False

            # Check label
            if label.item() != dataset.examples[idx][2]:
                print(f"Label mismatch at index {idx}")
                return False

        return True

    except Exception as e:
        print(f"Validation error: {e}")
        return False


# Example usage for documentation
if __name__ == "__main__":
    print("Modular Arithmetic Dataset Generator")
    print("=" * 60)

    # Example 1: Basic usage with sequence format
    print("\n1. Creating dataset with sequence format (p=113):")
    dataset_seq = ModularArithmeticDataset(modulus=113, fraction=1.0, format="sequence")
    print(f"   Total examples: {len(dataset_seq)}")
    print(f"   Vocabulary size: {dataset_seq.vocab_size}")
    visualize_samples(dataset_seq, n=3)

    # Example 2: Tuple format
    print("\n2. Creating dataset with tuple format (p=113):")
    dataset_tuple = ModularArithmeticDataset(modulus=113, fraction=1.0, format="tuple")
    print(f"   Total examples: {len(dataset_tuple)}")
    print(f"   Vocabulary size: {dataset_tuple.vocab_size}")
    visualize_samples(dataset_tuple, n=3)

    # Example 3: Partial dataset
    print("\n3. Creating partial dataset (50% of data, p=113):")
    dataset_partial = ModularArithmeticDataset(modulus=113, fraction=0.5, seed=42)
    print(f"   Total examples: {len(dataset_partial)}")

    # Example 4: Create DataLoaders
    print("\n4. Creating DataLoaders:")
    train_loader, test_loader = create_dataloaders(modulus=113, batch_size=512, train_fraction=0.7)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print(f"   Train examples: {len(train_loader.dataset)}")  # type: ignore[arg-type]
    print(f"   Test examples: {len(test_loader.dataset)}")  # type: ignore[arg-type]

    # Example 5: Get a batch
    print("\n5. Sample batch:")
    tokens_batch, labels_batch = next(iter(train_loader))
    print(f"   Batch shape: {tokens_batch.shape}")
    print(f"   Labels shape: {labels_batch.shape}")

    # Example 6: Dataset validation
    print("\n6. Validating dataset:")
    is_valid = validate_dataset(dataset_seq)
    print(f"   Dataset valid: {is_valid}")

    # Example 7: Statistics
    print("\n7. Dataset statistics:")
    stats = get_statistics(dataset_seq)
    for key, value in stats.items():
        if key != "answer_distribution":
            print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
