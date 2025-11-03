"""Simple, clean SAE implementation for research.

This module provides a straightforward implementation of Sparse Autoencoders
without external dependencies (except PyTorch). Designed for research transparency
and full control over training dynamics.

Architectures supported:
- TopK SAE: Keep top-k activations, zero others
- ReLU SAE: ReLU activation with L1 penalty

Reference implementations:
- Anthropic (2024): Scaling Monosemanticity
- Google DeepMind (2024): Gemma Scope
- OpenAI (2024): Sparse Autoencoders

Example:
    >>> sae = TopKSAE(d_model=128, d_sae=1024, k=32)
    >>> reconstruction, latents, aux_loss = sae(activations)
    >>> loss = F.mse_loss(reconstruction, activations) + aux_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from pathlib import Path


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder.
    
    Architecture:
        x → encoder (W_enc, b_enc) → pre_activation
        → TopK(pre_activation, k) → latents  
        → decoder (W_dec) → reconstruction
    
    Key features:
    - Explicit sparsity: exactly k active features
    - Decoder normalization: ||W_dec[:, i]|| = 1 for all i
    - Auxiliary loss: for dead neuron revival
    
    Args:
        d_model: Input dimension (e.g., transformer d_model)
        d_sae: SAE hidden dimension (expansion_factor * d_model)
        k: Number of top activations to keep
        aux_loss_coef: Coefficient for auxiliary loss (default: 1/k)
        dead_threshold: Threshold for dead neuron detection
    """
    
    def __init__(
        self,
        d_model: int,
        d_sae: int,
        k: int,
        aux_loss_coef: Optional[float] = None,
        dead_threshold: int = 10
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.aux_loss_coef = aux_loss_coef if aux_loss_coef is not None else (1.0 / k)
        self.dead_threshold = dead_threshold
        
        # Encoder: linear transformation + bias
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        
        # Decoder: linear transformation (no bias)
        # Will be unit-normalized after each training step
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        
        # Initialize decoder with unit norm columns
        self.reset_parameters()
        
        # Track feature activation counts (for dead neuron detection)
        self.register_buffer('feature_counts', torch.zeros(d_sae))
        
    def reset_parameters(self):
        """Initialize weights with proper scaling."""
        # Encoder: Kaiming initialization
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        # Decoder: random initialization + normalization
        nn.init.kaiming_uniform_(self.decoder.weight)
        self.normalize_decoder()
    
    def normalize_decoder(self):
        """Normalize decoder weights to unit norm.
        
        CRITICAL: This must be called after every optimizer step!
        Following Anthropic/Google best practices.
        """
        with torch.no_grad():
            # Normalize columns (each feature direction)
            self.decoder.weight.data = F.normalize(
                self.decoder.weight.data, dim=0
            )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to pre-activation SAE features.
        
        Args:
            x: Input activations [batch, d_model]
            
        Returns:
            pre_activation: Pre-TopK activations [batch, d_sae]
        """
        return self.encoder(x)
    
    def topk_activation(
        self, 
        pre_activation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply TopK activation function.
        
        Keeps top-k values, zeros others. This enforces exact sparsity.
        
        Args:
            pre_activation: [batch, d_sae]
            
        Returns:
            latents: TopK activated features [batch, d_sae]
            mask: Boolean mask of kept features [batch, d_sae]
        """
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(
            pre_activation, 
            k=self.k, 
            dim=-1, 
            sorted=False
        )
        
        # Create sparse latent representation
        latents = torch.zeros_like(pre_activation)
        latents.scatter_(dim=-1, index=topk_indices, src=topk_values)
        
        # Create mask for auxiliary loss
        mask = (latents != 0)
        
        return latents, mask
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to reconstruction.
        
        Args:
            latents: Sparse features [batch, d_sae]
            
        Returns:
            reconstruction: Reconstructed input [batch, d_model]
        """
        return self.decoder(latents)
    
    def forward(
        self, 
        x: torch.Tensor,
        compute_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through SAE.
        
        Args:
            x: Input activations [batch, d_model]
            compute_aux_loss: Whether to compute auxiliary loss
            
        Returns:
            reconstruction: Reconstructed input [batch, d_model]
            latents: Sparse SAE features [batch, d_sae]
            aux_loss: Auxiliary loss for dead neuron revival (scalar)
        """
        # Encode
        pre_activation = self.encode(x)
        
        # Apply TopK
        latents, mask = self.topk_activation(pre_activation)
        
        # Decode
        reconstruction = self.decode(latents)
        
        # Auxiliary loss (for dead neuron revival)
        # Following Anthropic's approach: encourage dead neurons to activate
        aux_loss = torch.tensor(0.0, device=x.device)
        if compute_aux_loss and self.training:
            # Compute reconstruction error magnitude (scalar per sample)
            error_magnitude = (x - reconstruction).pow(2).sum(dim=-1, keepdim=True)  # [batch, 1]
            
            # For features NOT in top-k, compute auxiliary loss
            # This encourages dead features to become useful
            dead_feature_act = pre_activation.abs() * (~mask).float()  # [batch, d_sae]
            aux_loss = (error_magnitude * dead_feature_act).topk(self.k, dim=-1)[0].sum(-1).mean()
            aux_loss = self.aux_loss_coef * aux_loss
        
        # Update feature counts (for dead neuron tracking)
        if self.training:
            with torch.no_grad():
                self.feature_counts += mask.float().sum(dim=0)
        
        return reconstruction, latents, aux_loss
    
    def get_dead_neurons(self) -> torch.Tensor:
        """Get indices of dead neurons (rarely/never activated).
        
        Returns:
            dead_indices: Indices of dead features
        """
        return (self.feature_counts < self.dead_threshold).nonzero(as_tuple=True)[0]
    
    def reset_feature_counts(self):
        """Reset feature activation counts (call at start of each epoch)."""
        self.feature_counts.zero_()
    
    def get_l0(self, latents: torch.Tensor) -> float:
        """Compute L0 sparsity (average number of active features).
        
        Args:
            latents: Sparse features [batch, d_sae]
            
        Returns:
            l0: Average number of non-zero features per sample
        """
        return (latents != 0).float().sum(dim=-1).mean().item()
    
    def save(self, path: Path):
        """Save SAE checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'd_model': self.d_model,
            'd_sae': self.d_sae,
            'k': self.k,
            'aux_loss_coef': self.aux_loss_coef,
            'dead_threshold': self.dead_threshold,
            'feature_counts': self.feature_counts,
        }, path)
    
    @classmethod
    def load(cls, path: Path, device: str = 'cpu'):
        """Load SAE checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load to
            
        Returns:
            sae: Loaded SAE model
        """
        checkpoint = torch.load(path, map_location=device)
        
        sae = cls(
            d_model=checkpoint['d_model'],
            d_sae=checkpoint['d_sae'],
            k=checkpoint['k'],
            aux_loss_coef=checkpoint['aux_loss_coef'],
            dead_threshold=checkpoint['dead_threshold']
        )
        
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.feature_counts = checkpoint['feature_counts']
        
        return sae.to(device)


class ReLUSAE(nn.Module):
    """ReLU Sparse Autoencoder.
    
    Architecture:
        x → encoder (W_enc, b_enc) → ReLU → latents
        → decoder (W_dec) → reconstruction
    
    Sparsity via L1 penalty on latent activations.
    
    Args:
        d_model: Input dimension
        d_sae: SAE hidden dimension
        l1_coef: L1 penalty coefficient (tune this!)
        dead_threshold: Threshold for dead neuron detection
    """
    
    def __init__(
        self,
        d_model: int,
        d_sae: int,
        l1_coef: float = 1e-3,
        dead_threshold: int = 10
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_sae = d_sae
        self.l1_coef = l1_coef
        self.dead_threshold = dead_threshold
        
        # Encoder
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        
        # Decoder (unit-normalized)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        
        # Initialize
        self.reset_parameters()
        
        # Track activations
        self.register_buffer('feature_counts', torch.zeros(d_sae))
    
    def reset_parameters(self):
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self.normalize_decoder()
    
    def normalize_decoder(self):
        """Normalize decoder weights to unit norm."""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(
                self.decoder.weight.data, dim=0
            )
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns:
            reconstruction, latents, l1_loss
        """
        # Encode + ReLU
        latents = F.relu(self.encoder(x))
        
        # Decode
        reconstruction = self.decoder(latents)
        
        # L1 sparsity loss
        l1_loss = self.l1_coef * latents.abs().sum(dim=-1).mean()
        
        # Track activations
        if self.training:
            with torch.no_grad():
                self.feature_counts += (latents > 0).float().sum(dim=0)
        
        return reconstruction, latents, l1_loss
    
    def get_dead_neurons(self) -> torch.Tensor:
        """Get dead neuron indices."""
        return (self.feature_counts < self.dead_threshold).nonzero(as_tuple=True)[0]
    
    def reset_feature_counts(self):
        """Reset counts."""
        self.feature_counts.zero_()
    
    def get_l0(self, latents: torch.Tensor) -> float:
        """Get L0 sparsity."""
        return (latents > 0).float().sum(dim=-1).mean().item()
    
    def save(self, path: Path):
        """Save checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'd_model': self.d_model,
            'd_sae': self.d_sae,
            'l1_coef': self.l1_coef,
            'dead_threshold': self.dead_threshold,
            'feature_counts': self.feature_counts,
        }, path)
    
    @classmethod
    def load(cls, path: Path, device: str = 'cpu'):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        sae = cls(
            d_model=checkpoint['d_model'],
            d_sae=checkpoint['d_sae'],
            l1_coef=checkpoint['l1_coef'],
            dead_threshold=checkpoint['dead_threshold']
        )
        
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.feature_counts = checkpoint['feature_counts']
        
        return sae.to(device)


if __name__ == "__main__":
    print("Simple SAE implementation for research")
    print("\nExample usage:")
    print("  from src.models.simple_sae import TopKSAE")
    print("  sae = TopKSAE(d_model=128, d_sae=1024, k=32)")
    print("  reconstruction, latents, aux_loss = sae(activations)")
