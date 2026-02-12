"""Sparse Autoencoder wrappers used by the HUSAI training pipeline.

This module provides a stable wrapper API for SAE training and analysis.
It currently uses the local simple SAE implementations (`TopKSAE`, `ReLUSAE`)
so the core pipeline is decoupled from external SAELens API churn.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.simple_sae import ReLUSAE, TopKSAE
from src.utils.config import SAEConfig


def create_sae(config: SAEConfig, device: Optional[str] = None) -> nn.Module:
    """Create an SAE model from config.

    Args:
        config: SAE configuration.
        device: Device string; auto-detected if None.

    Returns:
        Instantiated SAE module.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    d_sae = config.input_dim * config.expansion_factor

    if config.architecture == "topk":
        if config.k is None:
            raise ValueError("TopK SAE requires config.k")
        sae = TopKSAE(d_model=config.input_dim, d_sae=d_sae, k=config.k)

    elif config.architecture == "relu":
        if config.l1_coefficient is None:
            raise ValueError("ReLU SAE requires config.l1_coefficient")
        sae = ReLUSAE(d_model=config.input_dim, d_sae=d_sae, l1_coef=config.l1_coefficient)

    elif config.architecture == "batchtopk":
        if config.k is None:
            raise ValueError("BatchTopK SAE requires config.k")
        warnings.warn(
            "BatchTopK uses TopKSAE fallback in current implementation.",
            stacklevel=2,
        )
        sae = TopKSAE(d_model=config.input_dim, d_sae=d_sae, k=config.k)

    else:
        raise ValueError(
            f"Unknown SAE architecture: {config.architecture}. "
            "Must be one of: relu, topk, batchtopk"
        )

    return sae.to(device)


def save_sae_checkpoint(
    sae: nn.Module,
    path: Path,
    config: SAEConfig,
    metrics: Optional[Dict] = None,
) -> None:
    """Save SAE checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, object] = {
        "sae_state_dict": sae.state_dict(),
        "config": config.model_dump(),
    }
    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, path)


def load_sae_checkpoint(path: Path, device: Optional[str] = None) -> tuple[nn.Module, SAEConfig, Optional[Dict]]:
    """Load SAE from checkpoint."""
    checkpoint = torch.load(path, map_location=device or "cpu")

    config = SAEConfig(**checkpoint["config"])
    sae = create_sae(config, device=device)
    sae.load_state_dict(checkpoint["sae_state_dict"])

    metrics = checkpoint.get("metrics")
    return sae, config, metrics  # type: ignore[return-value]


class SAEWrapper(nn.Module):
    """Wrapper around local SAE models with a stable interface."""

    def __init__(self, config: SAEConfig, device: Optional[str] = None):
        super().__init__()
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sae = create_sae(config, device=self.device)

    def _forward_2d(
        self,
        activations: torch.Tensor,
        return_latents: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward helper for [batch, d_model] activations."""
        outputs = self.sae(activations)

        if isinstance(outputs, tuple):
            reconstructed = outputs[0]
            latents = outputs[1] if len(outputs) > 1 else None
        else:
            reconstructed = outputs
            latents = None

        if return_latents and latents is None:
            latents = self.sae.encode(activations)

        if not return_latents:
            latents = None

        return reconstructed, latents

    def forward(
        self,
        activations: torch.Tensor,
        return_latents: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for 2D or 3D activation tensors.

        Accepted inputs:
        - [batch, d_model]
        - [batch, seq, d_model]
        """
        if activations.ndim == 2:
            return self._forward_2d(activations, return_latents=return_latents)

        if activations.ndim != 3:
            raise ValueError(
                f"Expected 2D or 3D activations, got shape {tuple(activations.shape)}"
            )

        batch, seq, d_model = activations.shape
        flat = activations.reshape(-1, d_model)
        recon_flat, latents_flat = self._forward_2d(flat, return_latents=return_latents)

        reconstructed = recon_flat.reshape(batch, seq, d_model)
        if not return_latents or latents_flat is None:
            return reconstructed, None

        latents = latents_flat.reshape(batch, seq, -1)
        return reconstructed, latents

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Encode activations into SAE latent space."""
        if activations.ndim == 2:
            return self.sae.encode(activations)

        if activations.ndim == 3:
            batch, seq, d_model = activations.shape
            flat = activations.reshape(-1, d_model)
            latents = self.sae.encode(flat)
            return latents.reshape(batch, seq, -1)

        raise ValueError(f"Expected 2D or 3D activations, got shape {tuple(activations.shape)}")

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode SAE latents back to activation space."""
        if latents.ndim == 2:
            return self.sae.decode(latents)

        if latents.ndim == 3:
            batch, seq, d_sae = latents.shape
            flat = latents.reshape(-1, d_sae)
            decoded = self.sae.decode(flat)
            return decoded.reshape(batch, seq, -1)

        raise ValueError(f"Expected 2D or 3D latents, got shape {tuple(latents.shape)}")

    @torch.no_grad()
    def compute_sparsity(self, latents: torch.Tensor) -> Dict[str, float]:
        """Compute simple sparsity metrics."""
        l0 = (latents != 0).float().sum(dim=-1).mean().item()
        l1 = latents.abs().sum(dim=-1).mean().item()

        if latents.ndim == 2:
            active_features = (latents != 0).any(dim=0)
        elif latents.ndim == 3:
            active_features = (latents != 0).any(dim=(0, 1))
        else:
            raise ValueError(f"Expected 2D or 3D latents, got shape {tuple(latents.shape)}")

        frac_active = active_features.float().mean().item()
        return {
            "l0": l0,
            "l1": l1,
            "frac_active": frac_active,
        }

    def to(self, device: str) -> "SAEWrapper":
        """Move wrapped SAE to target device."""
        self.device = device
        self.sae = self.sae.to(device)
        return self

    def save(self, path: Path, metrics: Optional[Dict] = None) -> None:
        """Save wrapper checkpoint."""
        save_sae_checkpoint(self.sae, path, self.config, metrics)

    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> "SAEWrapper":
        """Load wrapper from checkpoint."""
        sae, config, _ = load_sae_checkpoint(path, device)
        wrapper = cls.__new__(cls)
        super(SAEWrapper, wrapper).__init__()
        wrapper.config = config
        wrapper.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        wrapper.sae = sae
        return wrapper

    def __repr__(self) -> str:
        d_sae = self.config.input_dim * self.config.expansion_factor
        return (
            "SAEWrapper(\n"
            f"  architecture={self.config.architecture},\n"
            f"  input_dim={self.config.input_dim},\n"
            f"  d_sae={d_sae},\n"
            f"  device={self.device}\n"
            ")"
        )
