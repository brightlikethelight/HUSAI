#!/usr/bin/env python3
"""Task Complexity Experiment: Does task complexity affect SAE stability?

Hypothesis: More complex tasks have more interpretable structure,
leading to higher SAE stability.

We test this by comparing:
1. Simple task: Modular addition (a + b mod 113)
2. Medium task: Modular multiplication (a * b mod 113)
3. Complex task: Combined operations (a + b * c mod 113)

If our hypothesis is correct:
- Simple task → low stability (no interpretable structure)
- Complex task → higher stability (more structure to discover)

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/task_complexity_experiment.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader, TensorDataset, Dataset

# Paths
BASE_DIR = Path('/Users/brightliu/School_Work/HUSAI')
RESULTS_DIR = BASE_DIR / 'results'
OUTPUT_DIR = RESULTS_DIR / 'task_complexity'
FIGURES_DIR = BASE_DIR / 'figures'

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = 'cpu'
MODULUS = 113


class ModularArithmeticDataset(Dataset):
    """Dataset for various modular arithmetic tasks."""
    
    def __init__(self, task: str, modulus: int = 113, seed: int = 42):
        """
        Args:
            task: One of 'addition', 'multiplication', 'combined'
            modulus: Prime modulus
            seed: Random seed
        """
        self.task = task
        self.modulus = modulus
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate all valid (a, b) pairs
        if task == 'addition':
            # a + b mod p
            self.data = []
            for a in range(modulus):
                for b in range(modulus):
                    c = (a + b) % modulus
                    self.data.append((a, b, c))
        
        elif task == 'multiplication':
            # a * b mod p
            self.data = []
            for a in range(modulus):
                for b in range(modulus):
                    c = (a * b) % modulus
                    self.data.append((a, b, c))
        
        elif task == 'combined':
            # (a + b) * c mod p - more complex
            self.data = []
            # Sample subset to keep manageable
            for _ in range(modulus * modulus):
                a = np.random.randint(0, modulus)
                b = np.random.randint(0, modulus)
                c = np.random.randint(0, modulus)
                result = ((a + b) * c) % modulus
                self.data.append((a, b, c, result))
        
        else:
            raise ValueError(f"Unknown task: {task}")
        
        self.data = np.array(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


class SimpleTransformer(nn.Module):
    """Simple transformer for modular arithmetic."""
    
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, 
                 vocab_size: int = 120, max_seq_len: int = 10):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x)
        return self.output(x)
    
    def get_activations(self, x: torch.Tensor, layer: int = -1) -> torch.Tensor:
        """Get intermediate activations."""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        x = self.embedding(x) + self.pos_embedding(positions)
        
        # Get activations from specified layer
        for i, layer_module in enumerate(self.transformer.layers):
            x = layer_module(x)
            if i == layer or (layer == -1 and i == len(self.transformer.layers) - 1):
                return x
        
        return x


class TopKSAE(nn.Module):
    """TopK SAE with proper decoder normalization."""
    
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        self.normalize_decoder()
    
    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_act = self.encoder(x)
        topk_values, topk_indices = torch.topk(pre_act, k=self.k, dim=-1)
        latents = torch.zeros_like(pre_act)
        latents.scatter_(-1, topk_indices, topk_values)
        return latents
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        recon = self.decoder(latents)
        return recon, latents


def compute_pwmcc(d1: torch.Tensor, d2: torch.Tensor) -> float:
    """Compute PWMCC between two decoder matrices."""
    d1_norm = F.normalize(d1, dim=0)
    d2_norm = F.normalize(d2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    return (max_1to2 + max_2to1) / 2


def compute_random_baseline(d_model: int, d_sae: int, n_trials: int = 10) -> float:
    """Compute random PWMCC baseline."""
    pwmcc_values = []
    for _ in range(n_trials):
        d1 = torch.randn(d_model, d_sae)
        d2 = torch.randn(d_model, d_sae)
        pwmcc_values.append(compute_pwmcc(d1, d2))
    return np.mean(pwmcc_values)


def train_transformer(task: str, epochs: int = 500) -> SimpleTransformer:
    """Train transformer on given task."""
    print(f"  Training transformer for {task}...")
    
    dataset = ModularArithmeticDataset(task, MODULUS)
    
    # Prepare data
    if task in ['addition', 'multiplication']:
        # Format: [a, op, b, eq] -> predict c
        data = dataset.data
        inputs = []
        targets = []
        for row in data:
            a, b, c = row
            # Use special tokens: 114=+, 115=*, 116==
            op_token = 114 if task == 'addition' else 115
            inputs.append([a, op_token, b, 116])
            targets.append(c)
        
        inputs = torch.tensor(inputs, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
    
    else:  # combined
        data = dataset.data
        inputs = []
        targets = []
        for row in data:
            a, b, c, result = row
            # Format: [a, +, b, *, c, =]
            inputs.append([a, 114, b, 115, c, 116])
            targets.append(result)
        
        inputs = torch.tensor(inputs, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
    
    # Create dataloader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Create model
    model = SimpleTransformer(d_model=128, n_heads=4, n_layers=2, vocab_size=120)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_inputs, batch_targets in dataloader:
            logits = model(batch_inputs)
            # Predict from last position
            pred_logits = logits[:, -1, :]
            
            loss = F.cross_entropy(pred_logits, batch_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = pred_logits.argmax(dim=-1)
            correct += (preds == batch_targets).sum().item()
            total += len(batch_targets)
        
        if (epoch + 1) % 100 == 0:
            acc = correct / total
            print(f"    Epoch {epoch+1}: loss={total_loss/len(dataloader):.4f}, acc={acc:.4f}")
    
    return model


def extract_activations(model: SimpleTransformer, task: str) -> torch.Tensor:
    """Extract activations from trained model."""
    dataset = ModularArithmeticDataset(task, MODULUS)
    
    if task in ['addition', 'multiplication']:
        data = dataset.data
        inputs = []
        for row in data:
            a, b, c = row
            op_token = 114 if task == 'addition' else 115
            inputs.append([a, op_token, b, 116])
        inputs = torch.tensor(inputs, dtype=torch.long)
    else:
        data = dataset.data
        inputs = []
        for row in data:
            a, b, c, result = row
            inputs.append([a, 114, b, 115, c, 116])
        inputs = torch.tensor(inputs, dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        acts = model.get_activations(inputs, layer=-1)
        # Take last position
        acts = acts[:, -1, :]
    
    return acts


def train_sae(acts: torch.Tensor, d_sae: int, k: int, seed: int, epochs: int = 20) -> TopKSAE:
    """Train SAE on activations."""
    torch.manual_seed(seed)
    
    sae = TopKSAE(d_model=128, d_sae=d_sae, k=k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)
    
    dataset = TensorDataset(acts)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    for epoch in range(epochs):
        for (batch,) in dataloader:
            recon, _ = sae(batch)
            loss = F.mse_loss(recon, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
    
    return sae


def run_task_experiment(task: str, d_sae: int = 128, k: int = 16, n_seeds: int = 5) -> Dict:
    """Run stability experiment for a given task."""
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"{'='*60}")
    
    # Train transformer
    model = train_transformer(task, epochs=500)
    
    # Extract activations
    acts = extract_activations(model, task)
    print(f"  Extracted {len(acts)} activation vectors")
    
    # Compute effective rank
    centered = acts - acts.mean(dim=0, keepdim=True)
    _, S, _ = torch.svd(centered)
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-10]
    entropy = -(S_norm * torch.log(S_norm)).sum()
    eff_rank = torch.exp(entropy).item()
    print(f"  Effective rank: {eff_rank:.1f}")
    
    # Train SAEs with different seeds
    saes = []
    for seed in range(n_seeds):
        sae = train_sae(acts, d_sae, k, seed, epochs=20)
        saes.append(sae)
    
    # Compute pairwise PWMCC
    pwmcc_values = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            pwmcc = compute_pwmcc(
                saes[i].decoder.weight.data,
                saes[j].decoder.weight.data
            )
            pwmcc_values.append(pwmcc)
    
    # Random baseline
    random_baseline = compute_random_baseline(128, d_sae)
    
    result = {
        'task': task,
        'effective_rank': eff_rank,
        'pwmcc_mean': np.mean(pwmcc_values),
        'pwmcc_std': np.std(pwmcc_values),
        'random_baseline': random_baseline,
        'ratio': np.mean(pwmcc_values) / random_baseline,
    }
    
    print(f"  PWMCC: {result['pwmcc_mean']:.4f} ± {result['pwmcc_std']:.4f}")
    print(f"  Random: {result['random_baseline']:.4f}")
    print(f"  Ratio: {result['ratio']:.2f}×")
    
    return result


def main():
    print("=" * 70)
    print("TASK COMPLEXITY EXPERIMENT")
    print("Testing if task complexity affects SAE stability")
    print("=" * 70)
    
    # Run experiments for each task
    tasks = ['addition', 'multiplication', 'combined']
    results = {}
    
    for task in tasks:
        results[task] = run_task_experiment(task, d_sae=128, k=16, n_seeds=5)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    task_names = list(results.keys())
    pwmcc_means = [results[t]['pwmcc_mean'] for t in task_names]
    pwmcc_stds = [results[t]['pwmcc_std'] for t in task_names]
    random_baselines = [results[t]['random_baseline'] for t in task_names]
    ratios = [results[t]['ratio'] for t in task_names]
    
    x = np.arange(len(task_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pwmcc_means, width, yerr=pwmcc_stds, 
                   label='Trained PWMCC', color='steelblue', capsize=5)
    bars2 = ax.bar(x + width/2, random_baselines, width, 
                   label='Random Baseline', color='gray', alpha=0.7)
    
    # Add ratio labels
    for i, (bar, ratio) in enumerate(zip(bars1, ratios)):
        ax.annotate(f'{ratio:.2f}×',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('PWMCC', fontsize=12)
    ax.set_title('SAE Stability by Task Complexity', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in task_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / 'task_complexity_experiment.pdf'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure to {fig_path}")
    
    # Save results
    output_path = OUTPUT_DIR / 'task_complexity_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Task Complexity vs SAE Stability:")
    print("-" * 50)
    for task in task_names:
        r = results[task]
        print(f"  {task:15s}: PWMCC={r['pwmcc_mean']:.4f}, eff_rank={r['effective_rank']:.1f}, ratio={r['ratio']:.2f}×")
    print()
    
    # Interpret results
    if results['combined']['ratio'] > results['addition']['ratio']:
        print("✓ HYPOTHESIS SUPPORTED: More complex tasks show higher stability!")
    else:
        print("✗ HYPOTHESIS NOT SUPPORTED: Task complexity doesn't improve stability")


if __name__ == '__main__':
    main()
