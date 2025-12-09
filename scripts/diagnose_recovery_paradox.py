"""
Diagnose the Ground Truth Recovery Paradox

Paradox: SAEs recover 8.8/10 true features (88%) with high similarity,
but have only 14% subspace overlap. How is this possible?

Hypotheses to test:
1. Different SAEs recover DIFFERENT subsets (e.g., SAE1 gets f1-f9, SAE2 gets f2-f10)
2. Recovered features are linear combinations, not exact matches
3. The 10th singular value is weak (σ_10 ≈ 0.2), so effective subspace is 9D
"""

import torch
import torch.nn.functional as F
import json
from pathlib import Path
import numpy as np
from src.models.simple_sae import TopKSAE

def load_sae(seed, results_dir):
    """Load trained SAE"""
    sae_path = results_dir / f"sae_seed_{seed}.pt"
    checkpoint = torch.load(sae_path, map_location='cpu')

    # Reconstruct SAE
    sae = TopKSAE(
        d_model=checkpoint['d_model'],
        d_sae=checkpoint['d_sae'],
        k=checkpoint['k']
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    return sae

def analyze_recovery_patterns(results_dir):
    """Analyze which true features each SAE recovers"""

    # Load true features
    results_path = results_dir / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    # Generate same true features (deterministic with seed)
    torch.manual_seed(42)
    d_model = 128
    n_true = 10

    # Orthonormal true features
    true_features = torch.randn(d_model, n_true)
    Q, _ = torch.linalg.qr(true_features)
    true_features = Q

    # Load all SAEs
    seeds = [42, 123, 456, 789, 1011]
    recovery_matrix = np.zeros((len(seeds), n_true))  # [n_saes, n_true]
    similarity_matrix = np.zeros((len(seeds), n_true))  # [n_saes, n_true]

    print("\n" + "="*80)
    print("GROUND TRUTH RECOVERY PATTERN ANALYSIS")
    print("="*80)

    for i, seed in enumerate(seeds):
        sae = load_sae(seed, results_dir)

        # Get normalized decoder
        decoder = sae.decoder.weight.data  # [d_model, d_sae]
        decoder = F.normalize(decoder, dim=0)  # Corrected normalization

        # Compute similarities: [n_true, d_sae]
        cos_sim = true_features.T @ decoder

        # For each true feature, find best match
        max_sims, best_matches = cos_sim.abs().max(dim=1)

        # Record which features are recovered (>0.9 threshold)
        recovered = (max_sims > 0.9).numpy()
        recovery_matrix[i] = recovered
        similarity_matrix[i] = max_sims.numpy()

        print(f"\nSeed {seed}:")
        print(f"  Recovered: {recovered.sum()}/10 features")
        print(f"  Feature indices recovered: {np.where(recovered)[0].tolist()}")
        print(f"  Similarities: {max_sims.numpy()}")

    # Analyze overlap patterns
    print("\n" + "="*80)
    print("RECOVERY OVERLAP ANALYSIS")
    print("="*80)

    # Which features are recovered by ALL SAEs?
    always_recovered = recovery_matrix.all(axis=0)
    print(f"\nFeatures recovered by ALL 5 SAEs: {np.where(always_recovered)[0].tolist()}")
    print(f"  Count: {always_recovered.sum()}/10")

    # Which features are NEVER recovered?
    never_recovered = ~recovery_matrix.any(axis=0)
    print(f"\nFeatures NEVER recovered: {np.where(never_recovered)[0].tolist()}")
    print(f"  Count: {never_recovered.sum()}/10")

    # Recovery consistency
    recovery_consistency = recovery_matrix.mean(axis=0)
    print(f"\nRecovery consistency per feature:")
    for feat_idx, consistency in enumerate(recovery_consistency):
        print(f"  Feature {feat_idx}: {consistency*100:.1f}% of SAEs")

    # Check if different SAEs recover different features
    print("\n" + "="*80)
    print("PAIRWISE RECOVERY OVERLAP")
    print("="*80)

    for i in range(len(seeds)):
        for j in range(i+1, len(seeds)):
            # How many features are recovered by BOTH SAEs?
            both_recovered = (recovery_matrix[i].astype(bool) & recovery_matrix[j].astype(bool)).sum()
            # Union
            either_recovered = (recovery_matrix[i].astype(bool) | recovery_matrix[j].astype(bool)).sum()
            # Jaccard similarity
            jaccard = both_recovered / either_recovered if either_recovered > 0 else 0

            print(f"Seeds {seeds[i]} vs {seeds[j]}: "
                  f"Both={both_recovered}/10, "
                  f"Either={either_recovered}/10, "
                  f"Jaccard={jaccard:.3f}")

    return recovery_matrix, similarity_matrix, true_features


def analyze_decoder_subspaces(results_dir):
    """Analyze decoder subspaces in detail"""

    print("\n" + "="*80)
    print("DECODER SUBSPACE ANALYSIS")
    print("="*80)

    seeds = [42, 123, 456, 789, 1011]

    # Load all decoders
    decoders = []
    for seed in seeds:
        sae = load_sae(seed, results_dir)
        decoder = sae.decoder.weight.data  # [d_model, d_sae]
        decoders.append(decoder)

    # Compute SVD for each
    print("\nSingular value profiles:")
    for i, (seed, decoder) in enumerate(zip(seeds, decoders)):
        U, S, Vh = torch.svd(decoder)
        print(f"\nSeed {seed}:")
        print(f"  Top 10 singular values: {S[:10].numpy()}")
        print(f"  σ_9 / σ_10 ratio: {(S[8] / S[9]).item():.2f}")

        # Effective rank
        total_var = S.pow(2).sum()
        cumsum = S.pow(2).cumsum(0) / total_var
        eff_rank_95 = (cumsum < 0.95).sum().item() + 1
        print(f"  Effective rank (95% variance): {eff_rank_95}")

    # Pairwise subspace overlap (top-k)
    print("\n" + "="*80)
    print("PAIRWISE SUBSPACE OVERLAP (varying k)")
    print("="*80)

    for k in [9, 10]:
        print(f"\nk = {k}:")
        overlaps = []

        for i in range(len(seeds)):
            for j in range(i+1, len(seeds)):
                U1, S1, _ = torch.svd(decoders[i])
                U2, S2, _ = torch.svd(decoders[j])

                U1_k = U1[:, :k]
                U2_k = U2[:, :k]

                overlap = (U1_k.T @ U2_k).pow(2).sum().item() / k
                overlaps.append(overlap)

                print(f"  Seeds {seeds[i]} vs {seeds[j]}: {overlap:.4f}")

        print(f"  Mean overlap: {np.mean(overlaps):.4f} ± {np.std(overlaps):.4f}")


def analyze_feature_alignment(results_dir, recovery_matrix, true_features):
    """Analyze how recovered features align with true features"""

    print("\n" + "="*80)
    print("FEATURE ALIGNMENT ANALYSIS")
    print("="*80)

    seeds = [42, 123, 456, 789, 1011]

    # For features recovered by multiple SAEs, check if they're the SAME SAE features
    print("\nFor consistently recovered true features:")
    print("Do different SAEs use the SAME learned feature to recover them?")

    # Focus on feature 0 (if recovered by multiple SAEs)
    for true_feat_idx in range(10):
        n_recovered = recovery_matrix[:, true_feat_idx].sum()
        if n_recovered < 2:
            continue

        print(f"\n--- True Feature {true_feat_idx} (recovered by {int(n_recovered)}/5 SAEs) ---")

        # Find which SAE features match this true feature
        sae_feature_indices = []

        for i, seed in enumerate(seeds):
            if not recovery_matrix[i, true_feat_idx]:
                continue

            sae = load_sae(seed, results_dir)
            decoder = F.normalize(sae.decoder.weight.data, dim=0)

            # Which SAE feature matches best?
            cos_sim = true_features[:, true_feat_idx] @ decoder
            best_idx = cos_sim.abs().argmax().item()
            best_sim = cos_sim[best_idx].item()

            sae_feature_indices.append(best_idx)
            print(f"  Seed {seed}: SAE feature #{best_idx} (similarity={best_sim:.4f})")

        # Check if they use different SAE feature indices
        unique_indices = len(set(sae_feature_indices))
        print(f"  Unique SAE feature indices used: {unique_indices}/{len(sae_feature_indices)}")
        print(f"  Interpretation: {'Different SAE features' if unique_indices > 1 else 'Same SAE feature index'}")


def main():
    results_dir = Path("results/synthetic_sparse_exact")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Please run the experiment first.")
        return

    # Run analyses
    recovery_matrix, similarity_matrix, true_features = analyze_recovery_patterns(results_dir)
    analyze_decoder_subspaces(results_dir)
    analyze_feature_alignment(results_dir, recovery_matrix, true_features)

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)

    print("\nKey findings:")
    print("1. Recovery pattern consistency - which features are always/never recovered?")
    print("2. Pairwise recovery overlap - do SAEs recover the same subset?")
    print("3. Subspace overlap at k=9 vs k=10 - is weak σ_10 the issue?")
    print("4. Feature alignment - do SAEs use different features to recover the same ground truth?")

    # Save detailed results
    output = {
        'recovery_matrix': recovery_matrix.tolist(),
        'similarity_matrix': similarity_matrix.tolist(),
        'recovery_consistency': recovery_matrix.mean(axis=0).tolist(),
        'always_recovered': int(recovery_matrix.all(axis=0).sum()),
        'never_recovered': int((~recovery_matrix.any(axis=0)).sum())
    }

    output_path = results_dir / "recovery_paradox_diagnosis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
