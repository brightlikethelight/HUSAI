"""SAE training loop implementation.

This module provides the core training loop for Sparse Autoencoders (SAEs),
including loss computation, dead neuron tracking, and metrics logging.

Example:
    >>> from src.models.sae import SAEWrapper
    >>> from src.training.train_sae import train_sae
    >>>
    >>> sae = SAEWrapper(config)
    >>> metrics = train_sae(
    ...     sae=sae,
    ...     activations=activations,
    ...     config=config,
    ...     use_wandb=True
    ... )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import wandb
from tqdm import tqdm

from src.utils.config import SAEConfig


@dataclass
class SAETrainingMetrics:
    """Container for SAE training metrics.

    Attributes:
        loss: Total loss per epoch
        mse_loss: MSE reconstruction loss per epoch
        sparsity_loss: Sparsity penalty per epoch (ReLU only)
        l0: Average number of active features per sample
        l1: Average L1 norm of latent activations
        explained_variance: Fraction of activation variance explained
        dead_neuron_count: Number of dead neurons
        dead_neuron_pct: Percentage of dead neurons
    """
    loss: List[float] = field(default_factory=list)
    mse_loss: List[float] = field(default_factory=list)
    sparsity_loss: List[float] = field(default_factory=list)
    l0: List[float] = field(default_factory=list)
    l1: List[float] = field(default_factory=list)
    explained_variance: List[float] = field(default_factory=list)
    dead_neuron_count: List[int] = field(default_factory=list)
    dead_neuron_pct: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'loss': self.loss,
            'mse_loss': self.mse_loss,
            'sparsity_loss': self.sparsity_loss,
            'l0': self.l0,
            'l1': self.l1,
            'explained_variance': self.explained_variance,
            'dead_neuron_count': self.dead_neuron_count,
            'dead_neuron_pct': self.dead_neuron_pct
        }


def normalize_decoder_weights(sae, method: str = "column_norm"):
    """Normalize SAE decoder weights.

    Critical for stable SAE training. Should be called after every optimizer step.

    Args:
        sae: SAE model (SAEWrapper or SAELens SAE)
        method: Normalization method ("column_norm" or "none")

    Note:
        This is a critical operation recommended by Anthropic, Google, and OpenAI.
        Prevents decoder norm collapse and improves feature learning.
    """
    if method == "none":
        return

    # Access decoder weights (handle both SAEWrapper and raw SAE)
    if hasattr(sae, 'sae'):
        decoder = sae.sae.decoder
    else:
        decoder = sae.decoder

    # Column normalization (normalize each feature direction)
    with torch.no_grad():
        decoder.weight.data = F.normalize(decoder.weight.data, dim=0)


def compute_sparsity_loss(
    latents: torch.Tensor,
    architecture: str,
    l1_coefficient: float = 1e-3,
    k: Optional[int] = None
) -> torch.Tensor:
    """Compute sparsity penalty for SAE latents.

    Args:
        latents: Latent activations [batch, d_sae]
        architecture: SAE architecture ("relu", "topk", "batchtopk")
        l1_coefficient: L1 penalty coefficient (for ReLU only)
        k: TopK parameter (for TopK/BatchTopK)

    Returns:
        sparsity_loss: Scalar loss term

    Note:
        - ReLU: L1 penalty on activations
        - TopK/BatchTopK: No explicit sparsity loss (enforced by activation function)
    """
    if architecture == "relu":
        # L1 penalty on activations
        return l1_coefficient * latents.abs().sum(dim=-1).mean()

    elif architecture in ["topk", "batchtopk"]:
        # No explicit sparsity loss for TopK
        # Sparsity is enforced by the activation function
        return torch.tensor(0.0, device=latents.device)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def compute_dead_neurons(
    activation_counts: torch.Tensor,
    threshold: int = 10
) -> Tuple[int, float]:
    """Compute number of dead neurons.

    A neuron is "dead" if it activates in fewer than `threshold` samples
    across an entire epoch.

    Args:
        activation_counts: Number of activations per neuron [d_sae]
        threshold: Minimum activations to be considered "alive"

    Returns:
        dead_count: Number of dead neurons
        dead_pct: Percentage of dead neurons
    """
    dead_mask = activation_counts < threshold
    dead_count = dead_mask.sum().item()
    dead_pct = 100.0 * dead_count / len(activation_counts)

    return dead_count, dead_pct


def train_sae(
    sae,  # SAEWrapper or SAELens SAE
    activations: torch.Tensor,
    config: SAEConfig,
    use_wandb: bool = True,
    wandb_project: str = "husai-sae-stability",
    wandb_run_name: Optional[str] = None,
    device: str = "cuda",
    checkpoint_dir: Optional[Path] = None,
    checkpoint_freq: int = 5,
    verbose: bool = True
) -> SAETrainingMetrics:
    """Train Sparse Autoencoder on activations.

    This is the core training loop for SAEs. It handles:
    - MSE reconstruction loss
    - Sparsity penalties (architecture-dependent)
    - Decoder normalization (critical!)
    - Dead neuron tracking
    - W&B logging
    - Checkpoint management

    Args:
        sae: SAE model (SAEWrapper or SAELens SAE)
        activations: Pre-extracted activations [num_samples, d_model]
        config: SAEConfig with training hyperparameters
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        wandb_run_name: Optional run name
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: Save checkpoint every N epochs
        verbose: Whether to print progress

    Returns:
        metrics: SAETrainingMetrics with training history

    Example:
        >>> metrics = train_sae(
        ...     sae=sae,
        ...     activations=activations,
        ...     config=config,
        ...     use_wandb=True
        ... )
        >>> print(f"Final L0: {metrics.l0[-1]:.2f}")
    """
    # Move SAE to device
    if hasattr(sae, 'to'):
        sae = sae.to(device)

    # Move activations to device
    activations = activations.to(device)

    # Create dataloader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True  # Drop last incomplete batch
    )

    # Create optimizer
    if hasattr(sae, 'sae'):
        params = sae.sae.parameters()
    else:
        params = sae.parameters()

    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    # Initialize W&B
    if use_wandb:
        if wandb_run_name is None:
            wandb_run_name = f"sae_{config.architecture}_seed{config.seed}"

        try:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=config.model_dump(),
                tags=[config.architecture, f"seed{config.seed}", "sae_training"]
            )
            if verbose:
                print(f"W&B initialized: {wandb.run.get_url()}")
        except Exception as e:
            if verbose:
                print(f"Warning: W&B initialization failed: {e}")
            use_wandb = False

    # Initialize metrics
    metrics = SAETrainingMetrics()

    # Compute activation variance (for explained variance)
    act_var = activations.var(dim=0).sum().item()

    # Initialize dead neuron tracking
    d_sae = config.input_dim * config.expansion_factor
    total_activation_counts = torch.zeros(d_sae, device=device)

    if verbose:
        print(f"\nStarting SAE training...")
        print(f"  Architecture: {config.architecture}")
        print(f"  Input dim: {config.input_dim}")
        print(f"  SAE dim: {d_sae} ({config.expansion_factor}x)")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Num epochs: {config.num_epochs}")
        print(f"  Device: {device}")

    # Training loop
    for epoch in range(config.num_epochs):
        sae.train()

        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_sparsity = 0.0
        epoch_l0 = 0.0
        epoch_l1 = 0.0
        num_batches = 0

        # Reset activation counts for this epoch
        epoch_activation_counts = torch.zeros(d_sae, device=device)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}",
                    disable=not verbose)

        for (batch_acts,) in pbar:
            batch_acts = batch_acts.to(device)
            # Forward pass (supports wrapper and raw SAE modules)
            try:
                reconstructed, latents = sae(batch_acts, return_latents=True)
            except TypeError:
                outputs = sae(batch_acts)
                if isinstance(outputs, tuple):
                    reconstructed = outputs[0]
                    latents = outputs[1] if len(outputs) > 1 else sae.encode(batch_acts)
                else:
                    reconstructed = outputs
                    latents = sae.encode(batch_acts)

            # Compute MSE loss
            mse_loss = F.mse_loss(reconstructed, batch_acts)

            # Compute sparsity loss
            sparsity_loss = compute_sparsity_loss(
                latents,
                architecture=config.architecture,
                l1_coefficient=config.l1_coefficient,
                k=config.k
            )

            # Total loss
            loss = mse_loss + sparsity_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder weights (CRITICAL!)
            normalize_decoder_weights(sae, method="column_norm")

            # Track metrics
            with torch.no_grad():
                # L0 sparsity (average number of active features)
                l0 = (latents != 0).float().sum(dim=-1).mean().item()

                # L1 norm
                l1 = latents.abs().sum(dim=-1).mean().item()

                # Track activation counts (for dead neuron detection)
                batch_counts = (latents.abs() > 1e-8).float().sum(dim=0)
                epoch_activation_counts += batch_counts
                total_activation_counts += batch_counts

                # Accumulate epoch metrics
                epoch_loss += loss.item()
                epoch_mse += mse_loss.item()
                epoch_sparsity += sparsity_loss.item()
                epoch_l0 += l0
                epoch_l1 += l1
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'l0': f'{l0:.1f}',
                    'mse': f'{mse_loss.item():.4f}'
                })

        # Compute epoch averages
        avg_loss = epoch_loss / num_batches
        avg_mse = epoch_mse / num_batches
        avg_sparsity = epoch_sparsity / num_batches
        avg_l0 = epoch_l0 / num_batches
        avg_l1 = epoch_l1 / num_batches

        # Compute explained variance
        with torch.no_grad():
            # Sample a batch for EV computation
            sample_acts = activations[:min(1000, len(activations))]
            try:
                sample_recon, _ = sae(sample_acts, return_latents=True)
            except TypeError:
                sample_outputs = sae(sample_acts)
                sample_recon = sample_outputs[0] if isinstance(sample_outputs, tuple) else sample_outputs
            reconstruction_mse = F.mse_loss(sample_recon, sample_acts, reduction='sum').item()
            explained_var = 1.0 - (reconstruction_mse / (act_var * len(sample_acts)))

        # Compute dead neurons
        dead_count, dead_pct = compute_dead_neurons(
            epoch_activation_counts,
            threshold=10
        )

        # Record metrics
        metrics.loss.append(avg_loss)
        metrics.mse_loss.append(avg_mse)
        metrics.sparsity_loss.append(avg_sparsity)
        metrics.l0.append(avg_l0)
        metrics.l1.append(avg_l1)
        metrics.explained_variance.append(explained_var)
        metrics.dead_neuron_count.append(dead_count)
        metrics.dead_neuron_pct.append(dead_pct)

        # Log to W&B
        if use_wandb:
            wandb.log({
                'train/loss': avg_loss,
                'train/mse_loss': avg_mse,
                'train/sparsity_loss': avg_sparsity,
                'train/l0': avg_l0,
                'train/l1': avg_l1,
                'train/explained_variance': explained_var,
                'train/dead_neurons': dead_count,
                'train/dead_neuron_pct': dead_pct,
                'epoch': epoch + 1
            }, step=epoch + 1)

        # Print epoch summary
        if verbose:
            print(f"\nEpoch {epoch+1}/{config.num_epochs}:")
            print(f"  Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, Sparsity: {avg_sparsity:.6f})")
            print(f"  L0: {avg_l0:.2f}, L1: {avg_l1:.4f}")
            print(f"  Explained variance: {explained_var:.4f}")
            print(f"  Dead neurons: {dead_count}/{d_sae} ({dead_pct:.2f}%)")

        # Save checkpoint
        if checkpoint_dir is not None and (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = checkpoint_dir / f"sae_epoch_{epoch+1}.pt"
            if hasattr(sae, 'save'):
                sae.save(checkpoint_path, metrics=metrics.to_dict())
            else:
                torch.save({
                    'sae_state_dict': sae.state_dict(),
                    'config': config.model_dump(),
                    'metrics': metrics.to_dict(),
                    'epoch': epoch + 1
                }, checkpoint_path)
            if verbose:
                print(f"  Checkpoint saved: {checkpoint_path}")

    # Save final checkpoint
    if checkpoint_dir is not None:
        final_path = checkpoint_dir / "sae_final.pt"
        if hasattr(sae, 'save'):
            sae.save(final_path, metrics=metrics.to_dict())
        else:
            torch.save({
                'sae_state_dict': sae.state_dict(),
                'config': config.model_dump(),
                'metrics': metrics.to_dict(),
                'epoch': config.num_epochs
            }, final_path)
        if verbose:
            print(f"\nFinal checkpoint saved: {final_path}")

    # Finish W&B
    if use_wandb:
        wandb.finish()

    if verbose:
        print("\n✓ SAE training complete!")
        print(f"Final metrics:")
        print(f"  L0: {metrics.l0[-1]:.2f}")
        print(f"  Explained variance: {metrics.explained_variance[-1]:.4f}")
        print(f"  Dead neurons: {metrics.dead_neuron_count[-1]}/{d_sae} ({metrics.dead_neuron_pct[-1]:.2f}%)")

    return metrics
