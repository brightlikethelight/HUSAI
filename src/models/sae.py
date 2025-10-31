"""Sparse Autoencoder (SAE) models using SAELens.

This module provides wrappers around SAELens for training SAEs on transformer
activations. It supports three architectures:
- ReLU SAE (with L1 penalty)
- TopK SAE (top-k activations per sample)
- BatchTopK SAE (top-k activations across batch)

Example:
    >>> from src.utils.config import SAEConfig
    >>> from src.models.sae import create_sae
    >>>
    >>> config = SAEConfig(
    ...     architecture="relu",
    ...     input_dim=128,
    ...     expansion_factor=4,
    ...     l1_coefficient=1e-3
    ... )
    >>> sae = create_sae(config)
    >>> reconstructed = sae(activations)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Literal
from sae_lens import SAE
from sae_lens.config import LanguageModelSAERunnerConfig

from src.utils.config import SAEConfig


def create_sae(
    config: SAEConfig,
    device: Optional[str] = None
) -> SAE:
    """Create SAE from config using SAELens.

    Args:
        config: SAEConfig with architecture parameters
        device: Device to create SAE on ('cuda', 'cpu', or None for auto-detect)

    Returns:
        SAE model from SAELens

    Example:
        >>> config = SAEConfig(
        ...     architecture="relu",
        ...     input_dim=128,
        ...     expansion_factor=4,
        ...     l1_coefficient=1e-3
        ... )
        >>> sae = create_sae(config)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Calculate SAE dimension
    d_sae = config.input_dim * config.expansion_factor

    # Create SAELens config based on architecture
    if config.architecture == "relu":
        # ReLU SAE with L1 penalty
        sae_config = LanguageModelSAERunnerConfig(
            architecture="standard",
            d_in=config.input_dim,
            d_sae=d_sae,
            l1_coefficient=config.l1_coefficient,
            lr=config.learning_rate,
            train_batch_size_tokens=config.batch_size,
            seed=config.seed if config.seed is not None else 42,
        )

    elif config.architecture == "topk":
        # TopK SAE (per-sample top-k)
        sae_config = LanguageModelSAERunnerConfig(
            architecture="topk",
            d_in=config.input_dim,
            d_sae=d_sae,
            activation_fn_kwargs={"k": config.k},
            lr=config.learning_rate,
            train_batch_size_tokens=config.batch_size,
            seed=config.seed if config.seed is not None else 42,
        )

    elif config.architecture == "batchtopk":
        # BatchTopK SAE (batch-level top-k)
        # Note: SAELens may not have native BatchTopK, we may need custom implementation
        # For now, use TopK as approximation
        sae_config = LanguageModelSAERunnerConfig(
            architecture="topk",
            d_in=config.input_dim,
            d_sae=d_sae,
            activation_fn_kwargs={"k": config.k},
            lr=config.learning_rate,
            train_batch_size_tokens=config.batch_size,
            seed=config.seed if config.seed is not None else 42,
        )

    else:
        raise ValueError(
            f"Unknown SAE architecture: {config.architecture}. "
            f"Must be one of: relu, topk, batchtopk"
        )

    # Create SAE
    sae = SAE(sae_config)
    sae = sae.to(device)

    return sae


def save_sae_checkpoint(
    sae: SAE,
    path: Path,
    config: SAEConfig,
    metrics: Optional[Dict] = None
) -> None:
    """Save SAE checkpoint.

    Args:
        sae: SAE model to save
        path: Path to save checkpoint
        config: SAEConfig used to create the SAE
        metrics: Optional metrics dict to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'sae_state_dict': sae.state_dict(),
        'config': config.model_dump(),
    }

    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, path)


def load_sae_checkpoint(
    path: Path,
    device: Optional[str] = None
) -> tuple[SAE, SAEConfig, Optional[Dict]]:
    """Load SAE from checkpoint.

    Args:
        path: Path to checkpoint file
        device: Device to load SAE on

    Returns:
        sae: Loaded SAE model
        config: SAEConfig used to create the SAE
        metrics: Optional metrics dict

    Example:
        >>> sae, config, metrics = load_sae_checkpoint('results/sae.pt')
        >>> print(f"Loaded SAE with {config.expansion_factor}x expansion")
    """
    checkpoint = torch.load(path, map_location=device or 'cpu')

    # Reconstruct config
    config = SAEConfig(**checkpoint['config'])

    # Create SAE
    sae = create_sae(config, device=device)
    sae.load_state_dict(checkpoint['sae_state_dict'])

    # Get metrics if available
    metrics = checkpoint.get('metrics', None)

    return sae, config, metrics


class SAEWrapper(nn.Module):
    """Wrapper around SAELens SAE with additional utilities.

    This class provides a clean PyTorch nn.Module interface around SAELens
    SAE models, with methods for activation extraction, reconstruction,
    and sparsity analysis.

    Attributes:
        sae: Underlying SAELens SAE model
        config: SAEConfig with architecture parameters
        device: Device to run SAE on

    Example:
        >>> wrapper = SAEWrapper(config)
        >>> reconstructed, latents = wrapper(activations)
        >>> sparsity = wrapper.compute_sparsity(latents)
    """

    def __init__(
        self,
        config: SAEConfig,
        device: Optional[str] = None
    ):
        """Initialize SAE wrapper.

        Args:
            config: SAEConfig with architecture parameters
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__()
        self.config = config

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Create SAE
        self.sae = create_sae(config, device=self.device)

    def forward(
        self,
        activations: torch.Tensor,
        return_latents: bool = True
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through SAE.

        Args:
            activations: Input activations [batch, seq, d_model]
            return_latents: Whether to return latent activations

        Returns:
            reconstructed: Reconstructed activations [batch, seq, d_model]
            latents: Latent activations [batch, seq, d_sae] (if return_latents=True)
        """
        # SAELens expects [batch * seq, d_model]
        batch, seq, d_model = activations.shape
        activations_flat = activations.view(-1, d_model)

        # Forward pass
        reconstructed_flat = self.sae(activations_flat)

        # Reshape back
        reconstructed = reconstructed_flat.view(batch, seq, d_model)

        if return_latents:
            # Get latent activations
            latents_flat = self.sae.encode(activations_flat)
            latents = latents_flat.view(batch, seq, -1)
            return reconstructed, latents
        else:
            return reconstructed, None

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Encode activations to latent space.

        Args:
            activations: Input activations [batch, seq, d_model]

        Returns:
            latents: Latent activations [batch, seq, d_sae]
        """
        batch, seq, d_model = activations.shape
        activations_flat = activations.view(-1, d_model)
        latents_flat = self.sae.encode(activations_flat)
        return latents_flat.view(batch, seq, -1)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to activation space.

        Args:
            latents: Latent activations [batch, seq, d_sae]

        Returns:
            reconstructed: Reconstructed activations [batch, seq, d_model]
        """
        batch, seq, d_sae = latents.shape
        latents_flat = latents.view(-1, d_sae)
        reconstructed_flat = self.sae.decode(latents_flat)
        return reconstructed_flat.view(batch, seq, -1)

    @torch.no_grad()
    def compute_sparsity(self, latents: torch.Tensor) -> Dict[str, float]:
        """Compute sparsity metrics for latent activations.

        Args:
            latents: Latent activations [batch, seq, d_sae]

        Returns:
            dict with sparsity metrics:
                - l0: Average number of active features per sample
                - l1: Average L1 norm
                - frac_active: Fraction of features ever active
        """
        # L0 sparsity (number of non-zero features)
        l0 = (latents != 0).float().sum(dim=-1).mean().item()

        # L1 norm
        l1 = latents.abs().sum(dim=-1).mean().item()

        # Fraction of features ever active
        active_features = (latents != 0).any(dim=(0, 1))
        frac_active = active_features.float().mean().item()

        return {
            'l0': l0,
            'l1': l1,
            'frac_active': frac_active,
        }

    def to(self, device: str) -> 'SAEWrapper':
        """Move SAE to device.

        Args:
            device: Device to move to ('cuda' or 'cpu')

        Returns:
            self for chaining
        """
        self.device = device
        self.sae = self.sae.to(device)
        return self

    def save(
        self,
        path: Path,
        metrics: Optional[Dict] = None
    ) -> None:
        """Save SAE checkpoint.

        Args:
            path: Path to save checkpoint
            metrics: Optional metrics dict to save
        """
        save_sae_checkpoint(self.sae, path, self.config, metrics)

    @classmethod
    def load(
        cls,
        path: Path,
        device: Optional[str] = None
    ) -> 'SAEWrapper':
        """Load SAE from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load on

        Returns:
            Loaded SAEWrapper

        Example:
            >>> wrapper = SAEWrapper.load('results/sae.pt')
        """
        sae, config, metrics = load_sae_checkpoint(path, device)
        wrapper = cls.__new__(cls)
        super(SAEWrapper, wrapper).__init__()
        wrapper.config = config
        wrapper.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        wrapper.sae = sae
        return wrapper

    def __repr__(self) -> str:
        """String representation of SAE."""
        return (
            f"SAEWrapper(\n"
            f"  architecture={self.config.architecture},\n"
            f"  input_dim={self.config.input_dim},\n"
            f"  expansion_factor={self.config.expansion_factor},\n"
            f"  d_sae={self.config.input_dim * self.config.expansion_factor},\n"
            f"  device={self.device}\n"
            f")"
        )
