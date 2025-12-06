"""Detailed investigation of transformer algorithm for modular addition."""

import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import ModularArithmeticDataset, create_dataloaders

# Suppress OpenMP warnings
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def test_generalization(model, dataset, modulus, device, n_samples=1000):
    """Test if model generalizes beyond training data."""
    print("\n" + "="*80)
    print("TESTING GENERALIZATION")
    print("="*80)

    # Test on random samples
    correct = 0
    total = 0

    # Sample uniformly
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in indices:
            tokens, label = dataset[int(idx)]
            tokens = tokens.unsqueeze(0).to(device)

            logits = model(tokens)
            pred = logits[0, -2, :modulus].argmax()

            if pred.item() == label.item():
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"Accuracy on {total} samples: {accuracy:.4f} ({correct}/{total})")

    # Test edge cases
    print("\nTesting edge cases:")
    edge_cases = [
        (0, 0),
        (0, 1),
        (1, 0),
        (modulus-1, modulus-1),
        (modulus-1, 1),
        (1, modulus-1),
        (modulus//2, modulus//2),
    ]

    edge_correct = 0
    with torch.no_grad():
        for a, b in edge_cases:
            c = (a + b) % modulus
            idx = a * modulus + b
            tokens, label = dataset[idx]
            tokens = tokens.unsqueeze(0).to(device)

            logits = model(tokens)
            pred = logits[0, -2, :modulus].argmax()

            is_correct = pred.item() == label.item()
            edge_correct += is_correct

            print(f"  {a} + {b} = {c}: {'✓' if is_correct else '✗'} (predicted {pred.item()})")

    print(f"\nEdge case accuracy: {edge_correct}/{len(edge_cases)}")

    return {
        'overall_accuracy': accuracy,
        'edge_case_accuracy': edge_correct / len(edge_cases),
        'generalizes': accuracy > 0.95
    }


def analyze_mlp_structure(model, modulus):
    """Analyze MLP weights for lookup table structure."""
    print("\n" + "="*80)
    print("ANALYZING MLP STRUCTURE")
    print("="*80)

    results = {}

    # Analyze each layer
    for layer_idx in range(model.config.n_layers):
        print(f"\nLayer {layer_idx}:")

        # Get MLP weights
        W_in = model.model.blocks[layer_idx].mlp.W_in.detach().cpu().numpy()
        W_out = model.model.blocks[layer_idx].mlp.W_out.detach().cpu().numpy()

        print(f"  W_in shape: {W_in.shape}")
        print(f"  W_out shape: {W_out.shape}")

        # Check sparsity
        sparsity_01 = (np.abs(W_out) < 0.1).sum() / W_out.size
        sparsity_05 = (np.abs(W_out) < 0.5).sum() / W_out.size

        print(f"  Sparsity (|w| < 0.1): {sparsity_01:.3f}")
        print(f"  Sparsity (|w| < 0.5): {sparsity_05:.3f}")

        # Check effective rank (low rank suggests structured computation)
        s = np.linalg.svd(W_out, compute_uv=False)
        total_energy = (s ** 2).sum()
        cumsum = np.cumsum(s ** 2)
        n_components_90 = np.argmax(cumsum / total_energy >= 0.9) + 1

        print(f"  Effective rank (90% energy): {n_components_90} / {min(W_out.shape)}")

        # Check for neuron specialization
        # If lookup table, some neurons should be highly specialized
        neuron_norms = np.linalg.norm(W_out, axis=1)  # [d_mlp]
        max_norm = neuron_norms.max()
        mean_norm = neuron_norms.mean()

        print(f"  Neuron norm ratio (max/mean): {max_norm / mean_norm:.2f}")

        results[f'layer_{layer_idx}'] = {
            'sparsity_0.1': sparsity_01,
            'sparsity_0.5': sparsity_05,
            'effective_rank': n_components_90,
            'max_rank': min(W_out.shape),
            'neuron_norm_ratio': max_norm / mean_norm,
        }

    return results


def analyze_activation_patterns(model, dataset, modulus, device, n_samples=200):
    """Analyze activation patterns for different inputs."""
    print("\n" + "="*80)
    print("ANALYZING ACTIVATION PATTERNS")
    print("="*80)

    # Collect activations for different input types
    activations_by_answer = {i: [] for i in range(modulus)}

    # Sample diverse inputs
    indices = np.linspace(0, len(dataset)-1, n_samples, dtype=int)

    with torch.no_grad():
        for idx in indices:
            tokens, label = dataset[int(idx)]
            tokens = tokens.unsqueeze(0).to(device)

            # Get activation from final layer at answer position
            act = model.get_activations(tokens, layer=model.config.n_layers-1)
            act_vector = act[0, -2, :].cpu().numpy()  # Position -2 is '='

            activations_by_answer[label.item()].append(act_vector)

    # Compute statistics
    print(f"\nAnalyzing {n_samples} activation patterns...")

    # Check if activations cluster by answer
    # Compute within-cluster vs between-cluster distances
    within_distances = []
    between_distances = []

    for answer in range(min(10, modulus)):  # Check first 10 answers
        acts = activations_by_answer[answer]
        if len(acts) < 2:
            continue

        # Within-cluster distance
        for i in range(len(acts)):
            for j in range(i+1, len(acts)):
                dist = np.linalg.norm(acts[i] - acts[j])
                within_distances.append(dist)

        # Between-cluster distance (compare to next answer)
        other_answer = (answer + 1) % modulus
        other_acts = activations_by_answer[other_answer]
        if len(other_acts) > 0:
            for act1 in acts:
                for act2 in other_acts[:len(acts)]:  # Sample same number
                    dist = np.linalg.norm(act1 - act2)
                    between_distances.append(dist)

    if within_distances and between_distances:
        within_mean = np.mean(within_distances)
        between_mean = np.mean(between_distances)
        separation_ratio = between_mean / within_mean

        print(f"\nActivation clustering:")
        print(f"  Within-cluster distance: {within_mean:.3f}")
        print(f"  Between-cluster distance: {between_mean:.3f}")
        print(f"  Separation ratio: {separation_ratio:.3f}")
        print(f"  Interpretation: >1.5 indicates strong clustering by answer")

        return {
            'within_distance': within_mean,
            'between_distance': between_mean,
            'separation_ratio': separation_ratio,
            'clusters_by_answer': separation_ratio > 1.5
        }
    else:
        print("Not enough samples to analyze clustering")
        return {}


def test_intermediate_representations(model, dataset, modulus, device):
    """Test if model computes intermediate values (a, b, or specific patterns)."""
    print("\n" + "="*80)
    print("TESTING FOR INTERMEDIATE REPRESENTATIONS")
    print("="*80)

    # Test if early layers represent 'a' and 'b' separately
    # Get activations for inputs with same 'a' but different 'b'

    test_cases = []
    # Fix a=10, vary b
    for b in range(min(10, modulus)):
        a = 10
        c = (a + b) % modulus
        idx = a * modulus + b
        tokens, label = dataset[idx]
        test_cases.append((tokens, a, b, c))

    # Get activations at position 1 (where 'a' is) and position 3 (where 'b' is)
    with torch.no_grad():
        # Get activations from layer 0
        acts_at_a = []
        acts_at_b = []

        for tokens, a, b, c in test_cases:
            tokens = tokens.unsqueeze(0).to(device)
            act = model.get_activations(tokens, layer=0)

            # Position 1 is 'a', position 3 is 'b'
            acts_at_a.append(act[0, 1, :].cpu().numpy())
            acts_at_b.append(act[0, 3, :].cpu().numpy())

    acts_at_a = np.array(acts_at_a)
    acts_at_b = np.array(acts_at_b)

    # Check if activations at 'a' position are similar (should be, since a is fixed)
    a_similarity = []
    for i in range(len(acts_at_a)):
        for j in range(i+1, len(acts_at_a)):
            sim = np.dot(acts_at_a[i], acts_at_a[j]) / (
                np.linalg.norm(acts_at_a[i]) * np.linalg.norm(acts_at_a[j]) + 1e-10
            )
            a_similarity.append(sim)

    # Check if activations at 'b' position vary (should vary, since b changes)
    b_similarity = []
    for i in range(len(acts_at_b)):
        for j in range(i+1, len(acts_at_b)):
            sim = np.dot(acts_at_b[i], acts_at_b[j]) / (
                np.linalg.norm(acts_at_b[i]) * np.linalg.norm(acts_at_b[j]) + 1e-10
            )
            b_similarity.append(sim)

    a_sim_mean = np.mean(a_similarity) if a_similarity else 0
    b_sim_mean = np.mean(b_similarity) if b_similarity else 0

    print(f"\nPosition-wise activation analysis (Layer 0):")
    print(f"  Same 'a' (different 'b') - similarity at 'a' position: {a_sim_mean:.3f}")
    print(f"  Same 'a' (different 'b') - similarity at 'b' position: {b_sim_mean:.3f}")
    print(f"\nInterpretation:")
    print(f"  High similarity at 'a' pos, low at 'b' pos → position-wise processing")
    print(f"  Low similarity at both → token embeddings matter more than position")

    return {
        'a_position_similarity': a_sim_mean,
        'b_position_similarity': b_sim_mean,
        'position_aware': a_sim_mean > b_sim_mean + 0.1
    }


def compare_to_fourier_baseline(model, dataset, modulus, device):
    """Compare model to expected Fourier circuit behavior."""
    print("\n" + "="*80)
    print("COMPARING TO FOURIER BASELINE")
    print("="*80)

    # Get embeddings
    embed = model.model.embed.W_E.detach().cpu().numpy()
    digit_embeds = embed[:modulus]

    print(f"\nEmbedding analysis:")
    print(f"  Shape: {digit_embeds.shape}")

    # Compute DFT basis
    # True Fourier circuit should have embeddings that are rows of DFT matrix
    # DFT matrix: W[k,n] = exp(-2πi*k*n/p) / sqrt(p)

    def dft_matrix(p):
        """Compute DFT matrix for modulus p."""
        W = np.zeros((p, p), dtype=complex)
        for k in range(p):
            for n in range(p):
                W[k, n] = np.exp(-2j * np.pi * k * n / p)
        return W / np.sqrt(p)

    dft = dft_matrix(modulus)

    # True Fourier circuit would use complex embeddings or split into real/imag
    # Check if embeddings align with DFT basis (real and imaginary parts)

    # Project embeddings onto DFT basis (real and imaginary parts)
    dft_real = dft.real
    dft_imag = dft.imag

    # Compute alignment for each embedding dimension
    alignments_real = []
    alignments_imag = []

    for dim in range(digit_embeds.shape[1]):
        emb_vec = digit_embeds[:, dim]

        # Best alignment with any DFT frequency (real part)
        corr_real = []
        for freq in range(modulus):
            c = np.corrcoef(emb_vec, dft_real[freq])[0, 1]
            corr_real.append(abs(c))

        # Best alignment with any DFT frequency (imag part)
        corr_imag = []
        for freq in range(modulus):
            c = np.corrcoef(emb_vec, dft_imag[freq])[0, 1]
            corr_imag.append(abs(c))

        alignments_real.append(max(corr_real))
        alignments_imag.append(max(corr_imag))

    alignments_real = np.array(alignments_real)
    alignments_imag = np.array(alignments_imag)

    # Strong Fourier would have many dimensions with high alignment
    n_high_alignment = ((alignments_real > 0.7) | (alignments_imag > 0.7)).sum()
    frac_high_alignment = n_high_alignment / digit_embeds.shape[1]

    print(f"\nAlignment with DFT basis:")
    print(f"  Dimensions with >0.7 correlation to DFT: {n_high_alignment} / {digit_embeds.shape[1]}")
    print(f"  Fraction: {frac_high_alignment:.3f}")
    print(f"  Mean alignment (real): {alignments_real.mean():.3f}")
    print(f"  Mean alignment (imag): {alignments_imag.mean():.3f}")
    print(f"\nInterpretation:")
    print(f"  >0.5 fraction → Strong Fourier structure")
    print(f"  0.2-0.5 → Weak Fourier structure")
    print(f"  <0.2 → No Fourier structure")

    return {
        'dft_alignment_frac': frac_high_alignment,
        'dft_alignment_mean_real': alignments_real.mean(),
        'dft_alignment_mean_imag': alignments_imag.mean(),
        'is_fourier_circuit': frac_high_alignment > 0.5
    }


def main():
    """Run comprehensive investigation."""
    print("="*80)
    print("COMPREHENSIVE ALGORITHM INVESTIGATION")
    print("="*80)

    # Setup
    model_path = Path(__file__).parent.parent / 'results' / 'transformer_5000ep' / 'transformer_final.pt'
    modulus = 113
    device = 'cpu'

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Load model
    print("\nLoading model...")
    model, extras = ModularArithmeticTransformer.load_checkpoint(model_path, device=device)
    model.eval()

    print(f"Model: {model.config}")
    print(f"Trained for: {extras.get('epoch', 'unknown')} epochs")
    if 'metrics' in extras:
        print(f"Final metrics: {extras['metrics']}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = ModularArithmeticDataset(modulus=modulus, fraction=1.0, seed=42)

    # Run all analyses
    all_results = {}

    # 1. Test generalization
    gen_results = test_generalization(model, dataset, modulus, device)
    all_results['generalization'] = gen_results

    # 2. Compare to Fourier baseline
    fourier_results = compare_to_fourier_baseline(model, dataset, modulus, device)
    all_results['fourier'] = fourier_results

    # 3. Analyze MLP structure
    mlp_results = analyze_mlp_structure(model, modulus)
    all_results['mlp'] = mlp_results

    # 4. Analyze activation patterns
    act_results = analyze_activation_patterns(model, dataset, modulus, device)
    all_results['activations'] = act_results

    # 5. Test intermediate representations
    inter_results = test_intermediate_representations(model, dataset, modulus, device)
    all_results['intermediate'] = inter_results

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print("\n1. GENERALIZATION:")
    print(f"   Overall accuracy: {gen_results['overall_accuracy']:.4f}")
    print(f"   Edge case accuracy: {gen_results['edge_case_accuracy']:.4f}")
    print(f"   Generalizes: {'YES ✓' if gen_results['generalizes'] else 'NO ✗'}")

    print("\n2. ALGORITHM TYPE:")

    # Fourier?
    is_fourier = fourier_results['is_fourier_circuit']
    print(f"   Fourier circuits: {'YES ✓' if is_fourier else 'NO ✗'}")
    print(f"     (DFT alignment: {fourier_results['dft_alignment_frac']:.3f})")

    # Lookup table?
    layer1_sparsity = mlp_results.get('layer_1', {}).get('sparsity_0.1', 0)
    layer1_rank_ratio = mlp_results.get('layer_1', {}).get('effective_rank', 1) / mlp_results.get('layer_1', {}).get('max_rank', 1)
    is_lookup = layer1_sparsity > 0.5 or layer1_rank_ratio < 0.3
    print(f"   Lookup table: {'YES ✓' if is_lookup else 'NO ✗'}")
    print(f"     (Sparsity: {layer1_sparsity:.3f}, Rank ratio: {layer1_rank_ratio:.3f})")

    # Structured activations?
    has_clusters = act_results.get('clusters_by_answer', False)
    print(f"   Clustered activations: {'YES ✓' if has_clusters else 'NO ✗'}")
    if 'separation_ratio' in act_results:
        print(f"     (Separation ratio: {act_results['separation_ratio']:.3f})")

    # Position-aware?
    is_position_aware = inter_results.get('position_aware', False)
    print(f"   Position-aware: {'YES ✓' if is_position_aware else 'NO ✗'}")

    print("\n3. MOST LIKELY HYPOTHESIS:")

    if is_fourier:
        hypothesis = "FOURIER CIRCUITS (discrete Fourier transform)"
        confidence = "High"
    elif is_lookup:
        hypothesis = "LOOKUP TABLE (sparse, learned compression)"
        confidence = "High"
    elif has_clusters:
        hypothesis = "STRUCTURED REPRESENTATION (non-Fourier algebraic structure)"
        confidence = "Medium"
    else:
        hypothesis = "DENSE DISTRIBUTED REPRESENTATION (no clear mathematical structure)"
        confidence = "Low"

    print(f"   {hypothesis}")
    print(f"   Confidence: {confidence}")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'algorithm_investigation'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'detailed_results.txt'
    with open(results_file, 'w') as f:
        f.write("TRANSFORMER ALGORITHM INVESTIGATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model.config}\n\n")

        for section, results in all_results.items():
            f.write(f"\n{section.upper()}:\n")
            f.write("-"*40 + "\n")
            if isinstance(results, dict):
                for k, v in results.items():
                    f.write(f"  {k}: {v}\n")

        f.write(f"\n\nFINAL HYPOTHESIS: {hypothesis}\n")
        f.write(f"CONFIDENCE: {confidence}\n")

    print(f"\nResults saved to: {results_file}")

    # Save as torch file
    torch.save(all_results, output_dir / 'detailed_results.pt')
    print(f"Detailed results saved to: {output_dir / 'detailed_results.pt'}")


if __name__ == '__main__':
    main()
