"""Example usage of ModularArithmeticDataset for grokking research.

This script demonstrates how to:
1. Create modular arithmetic datasets
2. Generate train/test DataLoaders
3. Visualize samples
4. Inspect dataset properties
5. Iterate through batches for training

Run this file to see examples of the dataset in action.
"""

import torch

from src.data.modular_arithmetic import (
    ModularArithmeticDataset,
    create_dataloaders,
    get_statistics,
    get_vocab_size,
    validate_dataset,
    visualize_samples,
)


def main() -> None:
    """Run example usage of modular arithmetic dataset."""
    print("=" * 80)
    print("MODULAR ARITHMETIC DATASET EXAMPLES")
    print("=" * 80)

    # Example 1: Basic dataset creation
    print("\n" + "=" * 80)
    print("Example 1: Creating a basic dataset (p=113, full dataset)")
    print("=" * 80)
    dataset = ModularArithmeticDataset(modulus=113, fraction=1.0, seed=42, format="sequence")
    print(f"Total examples: {len(dataset)}")
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Format: {dataset.format}")
    print("\nValidating dataset...")
    is_valid = validate_dataset(dataset)
    print(f"Dataset is valid: {is_valid}")

    # Example 2: Visualize samples
    print("\n" + "=" * 80)
    print("Example 2: Visualizing sample examples")
    print("=" * 80)
    visualize_samples(dataset, n=5)

    # Example 3: Dataset statistics
    print("\n" + "=" * 80)
    print("Example 3: Dataset statistics")
    print("=" * 80)
    stats = get_statistics(dataset)
    for key, value in stats.items():
        if key != "answer_distribution":
            print(f"{key}: {value}")

    # Example 4: Tuple format
    print("\n" + "=" * 80)
    print("Example 4: Using tuple format (more compact)")
    print("=" * 80)
    dataset_tuple = ModularArithmeticDataset(modulus=113, fraction=1.0, format="tuple")
    print(f"Total examples: {len(dataset_tuple)}")
    print(f"Vocabulary size: {dataset_tuple.vocab_size}")
    visualize_samples(dataset_tuple, n=3)

    # Example 5: Partial dataset
    print("\n" + "=" * 80)
    print("Example 5: Creating a partial dataset (30% of data)")
    print("=" * 80)
    dataset_partial = ModularArithmeticDataset(modulus=113, fraction=0.3, seed=123)
    print(f"Total examples: {len(dataset_partial)}")
    print(f"Fraction of full dataset: {len(dataset_partial) / (113 * 113):.2%}")

    # Example 6: Create DataLoaders
    print("\n" + "=" * 80)
    print("Example 6: Creating train/test DataLoaders")
    print("=" * 80)
    train_loader, test_loader = create_dataloaders(
        modulus=113,
        fraction=1.0,
        train_fraction=0.7,
        batch_size=512,
        seed=42,
        format="sequence",
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Train examples: {len(train_loader.dataset)}")  # type: ignore[arg-type]
    print(f"Test examples: {len(test_loader.dataset)}")  # type: ignore[arg-type]

    # Example 7: Inspect a batch
    print("\n" + "=" * 80)
    print("Example 7: Inspecting a training batch")
    print("=" * 80)
    tokens_batch, labels_batch = next(iter(train_loader))
    print(f"Batch tokens shape: {tokens_batch.shape}")
    print(f"Batch labels shape: {labels_batch.shape}")
    print(f"\nFirst 3 examples in batch:")
    for i in range(3):
        tokens = tokens_batch[i]
        label = labels_batch[i]
        decoded = dataset.decode_tokens(tokens)
        print(f"  Example {i}: {decoded} -> {label.item()}")

    # Example 8: Token ranges
    print("\n" + "=" * 80)
    print("Example 8: Token value ranges")
    print("=" * 80)
    print(f"Min token value in batch: {tokens_batch.min().item()}")
    print(f"Max token value in batch: {tokens_batch.max().item()}")
    print(f"Expected max (vocab_size - 1): {dataset.vocab_size - 1}")

    # Example 9: Vocabulary mapping
    print("\n" + "=" * 80)
    print("Example 9: Vocabulary mapping")
    print("=" * 80)
    vocab = dataset.get_vocab_mapping()
    print("Special tokens:")
    print(f"  BOS (Begin of Sequence): {dataset.modulus}")
    print(f"  EOS (End of Sequence): {dataset.modulus + 1}")
    print(f"  EQUALS: {dataset.modulus + 2}")
    print(f"  PLUS: {dataset.modulus + 3}")
    print(f"\nFirst few digit tokens:")
    for i in range(min(5, dataset.modulus)):
        print(f"  {i}: {vocab[i]}")

    # Example 10: Small modulus for debugging
    print("\n" + "=" * 80)
    print("Example 10: Small modulus (p=5) for debugging")
    print("=" * 80)
    dataset_small = ModularArithmeticDataset(modulus=5, fraction=1.0, format="sequence")
    print(f"Total examples: {len(dataset_small)}")
    print(f"\nAll examples (first 10):")
    for i in range(min(10, len(dataset_small))):
        tokens, label = dataset_small[i]
        a, b, c = dataset_small.examples[i]
        decoded = dataset_small.decode_tokens(tokens)
        print(f"  {a} + {b} = {c} mod 5: {decoded}")

    # Example 11: Iterating through an epoch
    print("\n" + "=" * 80)
    print("Example 11: Simulating training epoch")
    print("=" * 80)
    print("Creating small dataset and loader...")
    small_train_loader, _ = create_dataloaders(modulus=5, batch_size=10, train_fraction=0.6)
    print(f"Batches per epoch: {len(small_train_loader)}")
    print("\nIterating through epoch:")
    for batch_idx, (tokens, labels) in enumerate(small_train_loader):
        print(f"  Batch {batch_idx}: {tokens.shape[0]} examples")

    # Example 12: Reproducibility test
    print("\n" + "=" * 80)
    print("Example 12: Testing reproducibility")
    print("=" * 80)
    dataset1 = ModularArithmeticDataset(modulus=113, fraction=0.5, seed=42)
    dataset2 = ModularArithmeticDataset(modulus=113, fraction=0.5, seed=42)
    print(f"Dataset 1 length: {len(dataset1)}")
    print(f"Dataset 2 length: {len(dataset2)}")
    print(f"First examples match: {dataset1.examples[0] == dataset2.examples[0]}")
    print(f"Last examples match: {dataset1.examples[-1] == dataset2.examples[-1]}")
    print(f"All examples match: {dataset1.examples == dataset2.examples}")

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
