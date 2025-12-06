"""Investigate what algorithm the transformer learned for modular addition.

This script performs comprehensive analysis to determine the algorithm used by
the transformer, given that it's NOT using Fourier circuits as hypothesized.

Analysis steps:
1. Weight pattern analysis (embeddings, attention, MLP)
2. Algorithmic hypothesis testing (memorization, lookup, polynomial, etc.)
3. Probing analysis (activation clustering and structure)
4. Generalization validation (test on held-out data)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.stats as stats

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import ModularArithmeticTransformer
from src.data.modular_arithmetic import create_dataloaders, ModularArithmeticDataset


class AlgorithmInvestigator:
    """Investigates what algorithm a trained transformer learned."""

    def __init__(
        self,
        model_path: Path,
        modulus: int = 113,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize investigator with trained model.

        Args:
            model_path: Path to trained model checkpoint
            modulus: Modulus for modular arithmetic (default: 113)
            device: Device to run on
        """
        self.modulus = modulus
        self.device = device

        # Load model
        print(f"Loading model from {model_path}...")
        self.model, self.extras = ModularArithmeticTransformer.load_checkpoint(
            model_path, device=device
        )
        self.model.eval()

        # Create data loaders
        print("Creating data loaders...")
        self.train_loader, self.test_loader = create_dataloaders(
            modulus=modulus,
            batch_size=512,
            train_fraction=0.7,
            seed=42,
            device=device
        )

        # Create full dataset for analysis
        self.full_dataset = ModularArithmeticDataset(
            modulus=modulus, fraction=1.0, seed=42
        )

        self.results = {}

    def analyze_embedding_weights(self) -> Dict:
        """Analyze embedding matrix for patterns.

        Returns:
            Dictionary with embedding analysis results
        """
        print("\n" + "="*80)
        print("ANALYZING EMBEDDING WEIGHTS")
        print("="*80)

        # Get embedding weights
        embed = self.model.model.embed.W_E.detach().cpu().numpy()  # [vocab_size, d_model]

        results = {
            'shape': embed.shape,
            'mean': float(embed.mean()),
            'std': float(embed.std()),
            'max': float(embed.max()),
            'min': float(embed.min()),
        }

        # Analyze sparsity
        sparsity = (np.abs(embed) < 0.01).sum() / embed.size
        results['sparsity'] = float(sparsity)

        # Analyze structure in digit embeddings (0 to modulus-1)
        digit_embeds = embed[:self.modulus]  # Only digit embeddings

        # Check for periodic/Fourier-like patterns
        # Compute FFT for each embedding dimension
        fft_magnitudes = []
        for dim in range(digit_embeds.shape[1]):
            fft = np.fft.fft(digit_embeds[:, dim])
            fft_mag = np.abs(fft)
            fft_magnitudes.append(fft_mag)

        fft_magnitudes = np.array(fft_magnitudes)  # [d_model, modulus]

        # Check if FFT has strong peaks (indicating Fourier structure)
        # Exclude DC component (first frequency)
        fft_non_dc = fft_magnitudes[:, 1:]
        max_fft = fft_non_dc.max(axis=1)
        mean_fft = fft_non_dc.mean(axis=1)

        # Ratio of max to mean indicates peakiness
        peak_ratio = max_fft / (mean_fft + 1e-10)
        results['fft_peak_ratio_mean'] = float(peak_ratio.mean())
        results['fft_peak_ratio_max'] = float(peak_ratio.max())
        results['fft_peak_ratio_std'] = float(peak_ratio.std())

        # Check for polynomial patterns
        # If embeddings follow polynomial basis, they should be correlated with powers
        positions = np.arange(self.modulus)
        polynomial_correlations = []
        for power in range(1, 6):  # Check powers 1-5
            poly_basis = positions ** power
            corrs = []
            for dim in range(digit_embeds.shape[1]):
                corr = np.corrcoef(digit_embeds[:, dim], poly_basis)[0, 1]
                corrs.append(abs(corr))
            polynomial_correlations.append(max(corrs))

        results['max_polynomial_corr'] = [float(x) for x in polynomial_correlations]

        # Check for structure in similarity matrix
        similarity = digit_embeds @ digit_embeds.T  # [modulus, modulus]

        # Check if similar values have similar embeddings
        # For modular addition, (a+b) mod p structure might appear
        same_sum_similarities = []
        for target_sum in range(self.modulus):
            indices = [(a, b) for a in range(self.modulus)
                      for b in range(self.modulus)
                      if (a + b) % self.modulus == target_sum]
            if len(indices) >= 2:
                sims = [similarity[a, b] for a, b in indices]
                same_sum_similarities.append(np.mean(sims))

        results['same_sum_similarity_mean'] = float(np.mean(same_sum_similarities))
        results['same_sum_similarity_std'] = float(np.std(same_sum_similarities))

        # Print results
        print(f"\nEmbedding shape: {results['shape']}")
        print(f"Sparsity (|w| < 0.01): {results['sparsity']:.3f}")
        print(f"\nFourier structure analysis:")
        print(f"  FFT peak ratio (mean): {results['fft_peak_ratio_mean']:.3f}")
        print(f"  FFT peak ratio (max): {results['fft_peak_ratio_max']:.3f}")
        print(f"  FFT peak ratio (std): {results['fft_peak_ratio_std']:.3f}")
        print(f"  Interpretation: <2.0 = no strong Fourier, 2-5 = weak, >5 = strong")
        print(f"\nPolynomial correlations (max per power):")
        for i, corr in enumerate(results['max_polynomial_corr'], 1):
            print(f"  x^{i}: {corr:.3f}")
        print(f"\nModular structure:")
        print(f"  Same-sum similarity: {results['same_sum_similarity_mean']:.3f} ± {results['same_sum_similarity_std']:.3f}")

        self.results['embedding'] = results
        return results

    def analyze_attention_patterns(self) -> Dict:
        """Analyze attention patterns for structure.

        Returns:
            Dictionary with attention analysis results
        """
        print("\n" + "="*80)
        print("ANALYZING ATTENTION PATTERNS")
        print("="*80)

        # Sample some inputs
        sample_inputs = []
        for a in [0, 1, self.modulus//4, self.modulus//2, self.modulus-1]:
            for b in [0, 1, self.modulus//4, self.modulus//2, self.modulus-1]:
                c = (a + b) % self.modulus
                tokens, _ = self.full_dataset[a * self.modulus + b]
                sample_inputs.append(tokens)

        sample_inputs = torch.stack(sample_inputs).to(self.device)  # [n_samples, seq_len]

        # Run model and get attention patterns
        with torch.no_grad():
            _, cache = self.model.model.run_with_cache(sample_inputs)

        results = {}

        # Analyze each layer
        for layer in range(self.model.config.n_layers):
            layer_results = {}

            # Get attention pattern [batch, n_heads, seq_len, seq_len]
            attn_pattern = cache[f'blocks.{layer}.attn.hook_pattern']
            attn_np = attn_pattern.cpu().numpy()

            # Average over batch and heads
            avg_attn = attn_np.mean(axis=(0, 1))  # [seq_len, seq_len]

            # Check sparsity (how focused is attention?)
            entropy = stats.entropy(attn_np.reshape(-1, attn_np.shape[-1]).T)
            layer_results['attention_entropy_mean'] = float(entropy.mean())
            layer_results['attention_entropy_std'] = float(entropy.std())

            # Check which positions attend to which
            # For sequence: [BOS, a, +, b, =, c, EOS]
            # Positions:    [0,   1, 2, 3, 4, 5, 6]
            # Key positions: a=1, b=3, c=5

            # How much does 'c' position attend to 'a' and 'b'?
            c_to_a = avg_attn[5, 1]  # position 5 (c) attending to position 1 (a)
            c_to_b = avg_attn[5, 3]  # position 5 (c) attending to position 3 (b)
            c_to_ops = avg_attn[5, [2, 4]].sum()  # attending to operators

            layer_results['c_attends_to_a'] = float(c_to_a)
            layer_results['c_attends_to_b'] = float(c_to_b)
            layer_results['c_attends_to_ops'] = float(c_to_ops)

            results[f'layer_{layer}'] = layer_results

            print(f"\nLayer {layer}:")
            print(f"  Attention entropy: {layer_results['attention_entropy_mean']:.3f} ± {layer_results['attention_entropy_std']:.3f}")
            print(f"  Answer position attends to:")
            print(f"    a: {layer_results['c_attends_to_a']:.3f}")
            print(f"    b: {layer_results['c_attends_to_b']:.3f}")
            print(f"    operators: {layer_results['c_attends_to_ops']:.3f}")

        self.results['attention'] = results
        return results

    def test_memorization_hypothesis(self) -> Dict:
        """Test if model is just memorizing training examples.

        Returns:
            Dictionary with memorization test results
        """
        print("\n" + "="*80)
        print("TESTING MEMORIZATION HYPOTHESIS")
        print("="*80)

        # Collect train and test indices
        train_indices = set()
        test_indices = set()

        # Get train indices
        for batch_idx, (tokens, _) in enumerate(self.train_loader):
            for token_seq in tokens:
                # Extract a, b from sequence [BOS, a, +, b, =, c, EOS]
                a = token_seq[1].item()
                b = token_seq[3].item()
                idx = a * self.modulus + b
                train_indices.add(idx)

        # Get test indices
        for batch_idx, (tokens, _) in enumerate(self.test_loader):
            for token_seq in tokens:
                a = token_seq[1].item()
                b = token_seq[3].item()
                idx = a * self.modulus + b
                test_indices.add(idx)

        # Evaluate on train and test
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            # Evaluate on train examples
            for idx in list(train_indices)[:1000]:  # Sample for speed
                tokens, label = self.full_dataset[idx]
                tokens = tokens.unsqueeze(0).to(self.device)
                logits = self.model(tokens)
                pred = logits[0, -2, :self.modulus].argmax()  # Predict at '=' position
                if pred.item() == label.item():
                    train_correct += 1
                train_total += 1

            # Evaluate on test examples
            for idx in list(test_indices):
                tokens, label = self.full_dataset[idx]
                tokens = tokens.unsqueeze(0).to(self.device)
                logits = self.model(tokens)
                pred = logits[0, -2, :self.modulus].argmax()
                if pred.item() == label.item():
                    test_correct += 1
                test_total += 1

        train_acc = train_correct / train_total if train_total > 0 else 0
        test_acc = test_correct / test_total if test_total > 0 else 0

        results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_examples': train_total,
            'test_examples': test_total,
            'generalizes': test_acc > 0.9,  # If test acc > 90%, it generalizes
        }

        print(f"\nTrain accuracy: {train_acc:.4f} ({train_correct}/{train_total})")
        print(f"Test accuracy: {test_acc:.4f} ({test_correct}/{test_total})")
        print(f"\nConclusion: {'GENERALIZES - not pure memorization' if results['generalizes'] else 'MEMORIZATION suspected'}")

        self.results['memorization'] = results
        return results

    def test_lookup_table_hypothesis(self) -> Dict:
        """Test if model uses a sparse lookup table structure.

        Returns:
            Dictionary with lookup table test results
        """
        print("\n" + "="*80)
        print("TESTING LOOKUP TABLE HYPOTHESIS")
        print("="*80)

        results = {}

        # Analyze MLP weights for sparsity
        # Get final MLP weights (most likely location for lookup)
        final_layer = self.model.config.n_layers - 1

        # W_in: [d_model, d_mlp]
        W_in = self.model.model.blocks[final_layer].mlp.W_in.detach().cpu().numpy()
        # W_out: [d_mlp, d_model]
        W_out = self.model.model.blocks[final_layer].mlp.W_out.detach().cpu().numpy()

        # Check sparsity at different thresholds
        for threshold in [0.01, 0.05, 0.1]:
            sparsity_in = (np.abs(W_in) < threshold).sum() / W_in.size
            sparsity_out = (np.abs(W_out) < threshold).sum() / W_out.size
            results[f'sparsity_{threshold}_in'] = float(sparsity_in)
            results[f'sparsity_{threshold}_out'] = float(sparsity_out)

        # Check rank of weight matrices (low rank = more structured)
        # Use SVD to compute effective rank
        s_in = np.linalg.svd(W_in, compute_uv=False)
        s_out = np.linalg.svd(W_out, compute_uv=False)

        # Effective rank: number of singular values > 5% of max
        eff_rank_in = (s_in > 0.05 * s_in.max()).sum()
        eff_rank_out = (s_out > 0.05 * s_out.max()).sum()

        results['effective_rank_in'] = int(eff_rank_in)
        results['effective_rank_out'] = int(eff_rank_out)
        results['max_rank_in'] = min(W_in.shape)
        results['max_rank_out'] = min(W_out.shape)

        # Check for block structure (suggests lookup table)
        # Compute correlation matrix of neurons
        neuron_corr = np.corrcoef(W_out)  # [d_mlp, d_mlp]
        avg_corr = np.abs(neuron_corr[np.triu_indices_from(neuron_corr, k=1)]).mean()
        results['neuron_correlation'] = float(avg_corr)

        print(f"\nMLP weight sparsity:")
        for threshold in [0.01, 0.05, 0.1]:
            print(f"  Threshold {threshold}:")
            print(f"    W_in: {results[f'sparsity_{threshold}_in']:.3f}")
            print(f"    W_out: {results[f'sparsity_{threshold}_out']:.3f}")

        print(f"\nWeight matrix rank:")
        print(f"  W_in: {results['effective_rank_in']} / {results['max_rank_in']}")
        print(f"  W_out: {results['effective_rank_out']} / {results['max_rank_out']}")
        print(f"  (Lower effective rank = more structured)")

        print(f"\nNeuron correlation: {results['neuron_correlation']:.3f}")
        print(f"  (Higher = more redundancy/structure)")

        # Heuristic: lookup table if high sparsity OR low rank
        is_lookup = (results['sparsity_0.1_out'] > 0.5 or
                    results['effective_rank_out'] < 0.2 * results['max_rank_out'])
        results['is_lookup_table'] = is_lookup

        print(f"\nConclusion: {'LOOKUP TABLE structure detected' if is_lookup else 'Not a simple lookup table'}")

        self.results['lookup_table'] = results
        return results

    def analyze_activation_space(self, n_samples: int = 500) -> Dict:
        """Analyze structure in activation space using dimensionality reduction.

        Args:
            n_samples: Number of samples to analyze

        Returns:
            Dictionary with activation space analysis results
        """
        print("\n" + "="*80)
        print("ANALYZING ACTIVATION SPACE STRUCTURE")
        print("="*80)

        # Sample diverse inputs
        indices = np.linspace(0, len(self.full_dataset)-1, n_samples, dtype=int)

        activations_list = []
        labels_list = []
        a_values = []
        b_values = []

        with torch.no_grad():
            for idx in indices:
                tokens, label = self.full_dataset[idx]
                a, b, c = self.full_dataset.examples[idx]

                tokens = tokens.unsqueeze(0).to(self.device)

                # Get activations from final layer at answer position
                activations = self.model.get_activations(tokens, layer=self.model.config.n_layers-1)
                # Get activation at position 5 (where answer is predicted)
                act = activations[0, -2, :].cpu().numpy()  # Position -2 is '='

                activations_list.append(act)
                labels_list.append(c)
                a_values.append(a)
                b_values.append(b)

        activations = np.array(activations_list)  # [n_samples, d_model]
        labels = np.array(labels_list)
        a_values = np.array(a_values)
        b_values = np.array(b_values)

        results = {
            'n_samples': n_samples,
            'd_model': activations.shape[1],
        }

        # Check if activations cluster by answer
        # Use k-means with k=modulus
        kmeans = KMeans(n_clusters=self.modulus, random_state=42)
        cluster_labels = kmeans.fit_predict(activations)

        # Check if clusters correspond to answers
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(labels, cluster_labels)
        nmi = normalized_mutual_info_score(labels, cluster_labels)

        results['cluster_ari'] = float(ari)
        results['cluster_nmi'] = float(nmi)

        # PCA analysis
        pca = PCA(n_components=min(50, activations.shape[1]))
        pca_transform = pca.fit_transform(activations)

        # How many components to explain 90% variance?
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components_90 = np.argmax(cumsum >= 0.9) + 1
        results['pca_components_90pct'] = int(n_components_90)
        results['pca_variance_ratio_top10'] = [float(x) for x in pca.explained_variance_ratio_[:10]]

        print(f"\nActivation space dimensionality:")
        print(f"  Original: {results['d_model']}")
        print(f"  PCA components for 90% variance: {results['pca_components_90pct']}")
        print(f"  Top 10 variance ratios: {[f'{x:.3f}' for x in results['pca_variance_ratio_top10'][:5]]}...")

        print(f"\nClustering by answer value:")
        print(f"  Adjusted Rand Index: {results['cluster_ari']:.3f}")
        print(f"  Normalized Mutual Info: {results['cluster_nmi']:.3f}")
        print(f"  (Higher = activations cluster by answer)")

        # Check for linear separability
        # Train a simple linear classifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            activations, labels, test_size=0.2, random_state=42
        )

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        linear_acc = clf.score(X_test, y_test)

        results['linear_separability'] = float(linear_acc)
        print(f"\nLinear separability: {results['linear_separability']:.3f}")
        print(f"  (Accuracy of linear classifier on activations)")

        self.results['activation_space'] = results
        return results

    def run_full_investigation(self, output_dir: Optional[Path] = None):
        """Run all investigation methods and save results.

        Args:
            output_dir: Directory to save results (default: same as model)
        """
        print("\n" + "="*80)
        print("TRANSFORMER ALGORITHM INVESTIGATION")
        print("="*80)
        print(f"Model: {self.model}")
        print(f"Modulus: {self.modulus}")
        print(f"Device: {self.device}")

        # Run all analyses
        self.analyze_embedding_weights()
        self.analyze_attention_patterns()
        self.test_memorization_hypothesis()
        self.test_lookup_table_hypothesis()
        self.analyze_activation_space()

        # Generate final report
        print("\n" + "="*80)
        print("INVESTIGATION SUMMARY")
        print("="*80)

        print("\n1. GENERALIZATION:")
        mem_results = self.results['memorization']
        print(f"   Test accuracy: {mem_results['test_accuracy']:.4f}")
        print(f"   Generalizes: {mem_results['generalizes']}")

        print("\n2. ALGORITHM TYPE:")

        # Check for Fourier structure
        emb_results = self.results['embedding']
        has_fourier = emb_results['fft_peak_ratio_mean'] > 3.0
        print(f"   Fourier circuits: {'YES' if has_fourier else 'NO'}")
        print(f"     (FFT peak ratio: {emb_results['fft_peak_ratio_mean']:.2f})")

        # Check for polynomial structure
        has_polynomial = max(emb_results['max_polynomial_corr']) > 0.7
        print(f"   Polynomial basis: {'YES' if has_polynomial else 'NO'}")
        print(f"     (Max correlation: {max(emb_results['max_polynomial_corr']):.2f})")

        # Check for lookup table
        lookup_results = self.results['lookup_table']
        print(f"   Lookup table: {'YES' if lookup_results['is_lookup_table'] else 'NO'}")

        # Check activation structure
        act_results = self.results['activation_space']
        has_structured_acts = act_results['cluster_nmi'] > 0.5
        print(f"   Structured activations: {'YES' if has_structured_acts else 'NO'}")
        print(f"     (NMI: {act_results['cluster_nmi']:.3f})")

        print("\n3. HYPOTHESIS:")
        if has_fourier:
            hypothesis = "Fourier circuits (discrete Fourier transform)"
        elif has_polynomial:
            hypothesis = "Polynomial basis representation"
        elif lookup_results['is_lookup_table']:
            hypothesis = "Sparse lookup table (memorization with compression)"
        elif has_structured_acts:
            hypothesis = "Learned algebraic structure (non-Fourier)"
        else:
            hypothesis = "Dense distributed representation (no clear structure)"

        print(f"   Most likely algorithm: {hypothesis}")

        # Save results
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'results' / 'algorithm_investigation'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / 'investigation_results.txt'
        with open(results_file, 'w') as f:
            f.write("TRANSFORMER ALGORITHM INVESTIGATION RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Modulus: {self.modulus}\n\n")

            for key, value in self.results.items():
                f.write(f"\n{key.upper()}:\n")
                f.write("-"*40 + "\n")
                if isinstance(value, dict):
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {value}\n")

            f.write(f"\n\nFINAL HYPOTHESIS: {hypothesis}\n")

        print(f"\nResults saved to: {results_file}")

        # Save results as torch file for later analysis
        torch.save(self.results, output_dir / 'investigation_results.pt')
        print(f"Detailed results saved to: {output_dir / 'investigation_results.pt'}")

        return self.results


def main():
    """Run investigation on trained transformer."""
    # Path to trained model
    model_path = Path(__file__).parent.parent / 'results' / 'transformer_5000ep' / 'transformer_final.pt'

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using scripts/train_baseline.py")
        return

    # Create investigator
    investigator = AlgorithmInvestigator(
        model_path=model_path,
        modulus=113,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Run full investigation
    results = investigator.run_full_investigation()

    print("\nInvestigation complete!")


if __name__ == '__main__':
    main()
