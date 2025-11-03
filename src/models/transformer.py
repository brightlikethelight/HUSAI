"""Transformer model for modular arithmetic using TransformerLens.

This module provides a wrapper around TransformerLens's HookedTransformer
for training on modular arithmetic tasks. It handles model creation,
activation extraction, and checkpoint management.

Example:
    >>> from src.utils.config import TransformerConfig
    >>> from src.models.transformer import ModularArithmeticTransformer
    >>>
    >>> config = TransformerConfig(
    ...     n_layers=2,
    ...     d_model=128,
    ...     n_heads=4,
    ...     d_mlp=512,
    ...     vocab_size=117,
    ...     max_seq_len=7
    ... )
    >>> model = ModularArithmeticTransformer(config)
    >>> logits = model(tokens)  # Forward pass
    >>> activations = model.get_activations(tokens, layer=1)  # Extract activations
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
from transformer_lens import HookedTransformer

from src.utils.config import TransformerConfig


class ModularArithmeticTransformer(nn.Module):
    """Transformer model for modular arithmetic using TransformerLens.

    This class wraps TransformerLens's HookedTransformer with a clean interface
    for modular arithmetic tasks. It provides methods for training, activation
    extraction, and checkpoint management.

    Attributes:
        config: TransformerConfig with model hyperparameters
        model: Underlying HookedTransformer from TransformerLens
        device: Device to run model on (cuda/cpu)

    Example:
        >>> model = ModularArithmeticTransformer(config)
        >>> model.to('cuda')
        >>> logits = model(tokens)
        >>> loss = nn.CrossEntropyLoss()(logits[:, -2, :], labels)
    """

    def __init__(
        self,
        config: TransformerConfig,
        device: Optional[str] = None
    ):
        """Initialize transformer model from config.

        Args:
            config: TransformerConfig with model architecture parameters
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__()
        self.config = config

        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Create HookedTransformer config
        tl_config = {
            'n_layers': config.n_layers,
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'd_head': config.d_model // config.n_heads,
            'd_mlp': config.d_mlp,
            'd_vocab': config.vocab_size,
            'n_ctx': config.max_seq_len,
            'act_fn': config.activation,
            'normalization_type': 'LN',  # LayerNorm
            'device': self.device,
        }

        # Create model
        self.model = HookedTransformer(tl_config)

    def forward(
        self,
        tokens: torch.Tensor,
        return_type: str = "logits"
    ) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            tokens: Input token indices [batch, seq_len]
            return_type: What to return ("logits" or "loss")

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        return self.model(tokens, return_type=return_type)

    def get_activations(
        self,
        tokens: torch.Tensor,
        layer: Optional[int] = None,
        activation_name: str = "resid_post"
    ) -> torch.Tensor:
        """Extract activations from a specific layer.

        This is the primary method for extracting activations to train SAEs on.

        Args:
            tokens: Input token indices [batch, seq_len]
            layer: Which layer to extract from (None = last layer)
            activation_name: Name of activation to extract
                - "resid_post": Residual stream after layer (default)
                - "mlp_out": MLP output
                - "attn_out": Attention output

        Returns:
            activations: Activations [batch, seq_len, d_model]

        Example:
            >>> tokens = torch.randint(0, 117, (32, 7))
            >>> activations = model.get_activations(tokens, layer=1)
            >>> activations.shape
            torch.Size([32, 7, 128])
        """
        if layer is None:
            layer = self.config.n_layers - 1

        # Run model with caching
        _, cache = self.model.run_with_cache(tokens)

        # Extract activation using TransformerLens naming convention
        # TransformerLens uses format: "blocks.{layer}.hook_{activation_name}"
        activation_key = f"blocks.{layer}.hook_{activation_name}"
        activations = cache[activation_key]

        return activations

    def get_all_layer_activations(
        self,
        tokens: torch.Tensor,
        activation_name: str = "resid_post"
    ) -> Dict[int, torch.Tensor]:
        """Extract activations from all layers.

        Args:
            tokens: Input token indices [batch, seq_len]
            activation_name: Name of activation to extract

        Returns:
            dict mapping layer_idx -> activations [batch, seq_len, d_model]
        """
        _, cache = self.model.run_with_cache(tokens)

        activations_dict = {}
        for layer in range(self.config.n_layers):
            # TransformerLens uses format: "blocks.{layer}.hook_{activation_name}"
            activation_key = f"blocks.{layer}.hook_{activation_name}"
            activations_dict[layer] = cache[activation_key]

        return activations_dict

    def save_checkpoint(
        self,
        path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict] = None
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer state to save
            epoch: Optional epoch number
            metrics: Optional metrics dict to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.model_dump(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        device: Optional[str] = None,
        load_optimizer: bool = False
    ) -> Tuple['ModularArithmeticTransformer', Optional[Dict]]:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model on
            load_optimizer: Whether to return optimizer state

        Returns:
            model: Loaded ModularArithmeticTransformer
            checkpoint_extras: Dict with optimizer state, epoch, metrics

        Example:
            >>> model, extras = ModularArithmeticTransformer.load_checkpoint(
            ...     'results/transformer.pt'
            ... )
            >>> print(f"Loaded model from epoch {extras['epoch']}")
        """
        checkpoint = torch.load(path, map_location=device or 'cpu')

        # Reconstruct config
        config = TransformerConfig(**checkpoint['config'])

        # Create model
        model = cls(config, device=device)
        model.model.load_state_dict(checkpoint['model_state_dict'])

        # Prepare extras
        extras = {}
        if 'optimizer_state_dict' in checkpoint and load_optimizer:
            extras['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
        if 'epoch' in checkpoint:
            extras['epoch'] = checkpoint['epoch']
        if 'metrics' in checkpoint:
            extras['metrics'] = checkpoint['metrics']

        return model, extras

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def to(self, device: str) -> 'ModularArithmeticTransformer':
        """Move model to device.

        Args:
            device: Device to move to ('cuda' or 'cpu')

        Returns:
            self for chaining
        """
        self.device = device
        self.model = self.model.to(device)
        return self

    def __repr__(self) -> str:
        """String representation of model."""
        params = self.count_parameters()
        return (
            f"ModularArithmeticTransformer(\n"
            f"  n_layers={self.config.n_layers},\n"
            f"  d_model={self.config.d_model},\n"
            f"  n_heads={self.config.n_heads},\n"
            f"  d_mlp={self.config.d_mlp},\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  parameters={params:,},\n"
            f"  device={self.device}\n"
            f")"
        )
