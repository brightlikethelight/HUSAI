#!/usr/bin/env python3
"""Train SAE with simple custom implementation.

This script uses our clean SimpleSAE implementation for full control.
No external SAE library dependencies - just PyTorch!

Usage:
    # Quick test (1 epoch)
    python scripts/train_simple_sae.py \
        --transformer results/transformer_5000ep/transformer_best.pt \
        --epochs 1 \
        --test-run

    # Full training
    python scripts/train_simple_sae.py \
        --transformer results/transformer_5000ep/transformer_best.pt \
        --architecture topk \
        --k 32 \
        --expansion 8 \
        --epochs 20 \
        --seed 42
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.transformer import ModularArithmeticTransformer
from src.models.simple_sae import TopKSAE, ReLUSAE
from src.analysis.fourier_validation import get_fourier_basis, compute_fourier_overlap

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available - training without logging")


def extract_activations(model, modulus=113, layer=1, max_samples=50000):
    """Extract activations from trained transformer."""
    print(f"Extracting activations from layer {layer}...")
    
    # Import here to avoid .gitignore issues
    sys.path.insert(0, str(project_root / "src"))
    from data.modular_arithmetic import create_dataloaders
    
    train_loader, _ = create_dataloaders(
        modulus=modulus,
        batch_size=256,
        train_fraction=0.9,
        seed=42
    )
    
    activations = []
    model.eval()
    
    with torch.no_grad():
        for batch, _ in tqdm(train_loader, desc="Extracting"):
            if len(activations) * 256 >= max_samples:
                break
            
            # Get activations from specified layer
            act = model.get_activations(batch, layer=layer, activation_name="resid_post")
            activations.append(act[:, -2, :])  # Answer position
    
    activations = torch.cat(activations, dim=0)
    print(f"✅ Extracted {activations.shape[0]} activations: {activations.shape}")
    print(f"   Mean: {activations.mean():.4f}, Std: {activations.std():.4f}")
    
    return activations


def train_sae(
    sae,
    activations,
    epochs=20,
    batch_size=256,
    lr=3e-4,
    use_wandb=False,
    save_dir=None
):
    """Train SAE on activations."""
    
    # Create dataloader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    # Training loop
    print(f"\nTraining {sae.__class__.__name__} for {epochs} epochs...")
    
    for epoch in range(epochs):
        sae.train()
        sae.reset_feature_counts()
        
        epoch_loss = 0
        epoch_aux_loss = 0
        epoch_l0 = 0
        epoch_mse = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for (batch,) in pbar:
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, latents, aux_loss = sae(batch)
            
            # MSE loss
            mse_loss = F.mse_loss(reconstruction, batch)
            
            # Total loss (MSE + auxiliary/L1 loss)
            loss = mse_loss + aux_loss
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # CRITICAL: Normalize decoder after every step
            sae.normalize_decoder()
            
            # Metrics
            l0 = sae.get_l0(latents)
            epoch_loss += loss.item()
            epoch_aux_loss += aux_loss.item()
            epoch_l0 += l0
            epoch_mse += mse_loss.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'l0': f'{l0:.1f}',
                'mse': f'{mse_loss.item():.4f}'
            })
        
        # Epoch metrics
        avg_loss = epoch_loss / n_batches
        avg_aux = epoch_aux_loss / n_batches
        avg_l0 = epoch_l0 / n_batches
        avg_mse = epoch_mse / n_batches
        
        # Dead neurons
        dead_neurons = sae.get_dead_neurons()
        dead_pct = len(dead_neurons) / sae.d_sae * 100
        
        # Explained variance
        with torch.no_grad():
            sae.eval()
            all_recon = []
            for (batch,) in DataLoader(dataset, batch_size=1024):
                recon, _, _ = sae(batch)
                all_recon.append(recon)
            all_recon = torch.cat(all_recon, dim=0)
            
            data_var = activations.var()
            error_var = (activations - all_recon).var()
            explained_var = 1 - (error_var / data_var)
            explained_var = explained_var.item()
        
        print(f"\n Epoch {epoch+1}/{epochs}:")
        print(f"   Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, Aux/L1: {avg_aux:.4f})")
        print(f"   L0: {avg_l0:.2f}")
        print(f"   Explained variance: {explained_var:.4f}")
        print(f"   Dead neurons: {len(dead_neurons)}/{sae.d_sae} ({dead_pct:.1f}%)")
        
        # W&B logging
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': avg_loss,
                'train/mse_loss': avg_mse,
                'train/aux_loss': avg_aux,
                'train/l0': avg_l0,
                'train/explained_variance': explained_var,
                'train/dead_neurons': len(dead_neurons),
                'train/dead_neuron_pct': dead_pct,
            })
    
    print("\n✅ Training complete!")
    
    # Save if requested
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "sae_final.pt"
        sae.save(save_path)
        print(f"✅ Saved to {save_path}")
    
    return {
        'loss': avg_loss,
        'mse': avg_mse,
        'l0': avg_l0,
        'explained_variance': explained_var,
        'dead_neuron_pct': dead_pct
    }


def main():
    parser = argparse.ArgumentParser(description="Train simple SAE")
    
    # Model args
    parser.add_argument('--transformer', type=Path, required=True,
                       help='Path to trained transformer')
    parser.add_argument('--architecture', type=str, default='topk',
                       choices=['topk', 'relu'], help='SAE architecture')
    parser.add_argument('--expansion', type=int, default=8,
                       help='Expansion factor (d_sae = expansion * d_model)')
    parser.add_argument('--k', type=int, default=32,
                       help='TopK k parameter (TopK only)')
    parser.add_argument('--l1-coef', type=float, default=1e-3,
                       help='L1 coefficient (ReLU only)')
    
    # Training args
    parser.add_argument('--layer', type=int, default=1,
                       help='Layer to extract activations from')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output args
    parser.add_argument('--save-dir', type=Path,
                       help='Directory to save SAE')
    parser.add_argument('--wandb', action='store_true',
                       help='Use W&B logging')
    parser.add_argument('--wandb-project', type=str,
                       default='husai-sae-stability',
                       help='W&B project name')
    parser.add_argument('--test-run', action='store_true',
                       help='Quick test run (1 epoch, small data)')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Test run overrides
    if args.test_run:
        print("🧪 TEST RUN MODE")
        args.epochs = 1
        args.wandb = False
    
    # Load transformer
    print(f"Loading transformer from {args.transformer}...")
    model, extras = ModularArithmeticTransformer.load_checkpoint(args.transformer)
    config = extras.get('config')
    modulus = config.dataset.modulus if config else 113
    d_model = model.model.cfg.d_model  # Get from underlying HookedTransformer
    print(f"✅ Loaded: d_model={d_model}, modulus={modulus}")
    
    # Extract activations
    max_samples = 10000 if args.test_run else 50000
    activations = extract_activations(model, modulus=modulus, layer=args.layer, max_samples=max_samples)
    
    # Create SAE
    d_sae = args.expansion * d_model
    print(f"\nCreating {args.architecture.upper()} SAE:")
    print(f"  d_model: {d_model}")
    print(f"  d_sae: {d_sae} ({args.expansion}x expansion)")
    
    if args.architecture == 'topk':
        print(f"  k: {args.k}")
        sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=args.k)
    else:
        print(f"  L1 coef: {args.l1_coef}")
        sae = ReLUSAE(d_model=d_model, d_sae=d_sae, l1_coef=args.l1_coef)
    
    # W&B init
    if args.wandb and WANDB_AVAILABLE:
        run_name = f"{args.architecture}_layer{args.layer}_seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
    
    # Train
    metrics = train_sae(
        sae,
        activations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_wandb=args.wandb and WANDB_AVAILABLE,
        save_dir=args.save_dir
    )
    
    # Fourier validation
    print("\n" + "="*60)
    print("FOURIER VALIDATION")
    print("="*60)
    fourier_basis = get_fourier_basis(modulus=modulus)
    overlap = compute_fourier_overlap(sae.decoder.weight.data.T, fourier_basis)
    print(f"Fourier overlap: {overlap:.3f}")
    
    if overlap > 0.7:
        print("🌟 EXCELLENT! SAE recovered Fourier circuits strongly!")
    elif overlap > 0.5:
        print("✅ GOOD! SAE shows clear Fourier structure recovery.")
    elif overlap > 0.3:
        print("⚠️  MODERATE. Some Fourier structure, but could be better.")
    else:
        print("❌ LOW. SAE did not recover Fourier circuits well.")
    
    if args.wandb and WANDB_AVAILABLE:
        wandb.log({'eval/fourier_overlap': overlap})
        wandb.finish()
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Architecture: {args.architecture.upper()}")
    print(f"Final loss: {metrics['loss']:.4f}")
    print(f"L0: {metrics['l0']:.2f}")
    print(f"Explained variance: {metrics['explained_variance']:.4f}")
    print(f"Dead neurons: {metrics['dead_neuron_pct']:.1f}%")
    print(f"Fourier overlap: {overlap:.3f}")
    
    # Quality assessment
    score = 0
    if metrics['l0'] < 40:  # Reasonable sparsity
        score += 1
    if metrics['explained_variance'] > 0.85:
        score += 1
    if metrics['dead_neuron_pct'] < 20:
        score += 1
    if overlap > 0.6:
        score += 2  # Double weight for ground truth!
    
    print(f"\nQuality: {'⭐' * score}/⭐⭐⭐⭐⭐")
    
    if score >= 4:
        print("🎉 EXCELLENT SAE! Ready for multi-seed experiments!")
    elif score >= 3:
        print("✅ GOOD SAE! Proceed with experiments.")
    else:
        print("⚠️  SAE needs tuning. Consider adjusting hyperparameters.")
    
    print("="*60)


if __name__ == "__main__":
    main()
