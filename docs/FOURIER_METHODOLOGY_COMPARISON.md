# Fourier Methodology Comparison: Literature vs Our Approach

**Date:** November 4, 2025
**Purpose:** Document differences between Nanda et al.'s Fourier measurement and ours

---

## Executive Summary

**CRITICAL FINDING:** We are measuring different things than the literature!

- **Nanda et al.**: Analyze **weight matrices** (embedding WE, neuron-logit map WL)
- **Our approach**: Analyze **activations** at specific layer/position

This explains why our Fourier overlap (~0.26) is much lower than expected (0.6-0.8).

---

## Literature Approach (Nanda et al. 2023)

### Paper: "Progress measures for grokking via mechanistic interpretability"
- **Authors:** Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt
- **Published:** ICLR 2023 (Spotlight)
- **arXiv:** 2301.05217

### Their Methodology

#### 1. **Embedding Matrix Analysis**

**What they analyze:**
- **Embedding matrix WE**: Dimensions (d_model × vocab_size) = (128 × 113)
- Applied to **numerical tokens** (0 to 112) only

**How they compute Fourier structure:**
```python
# Conceptual pseudocode from paper
W_E = model.embed.weight  # [128, 113]

# Apply DFT along input dimension (vocab dimension)
W_E_fourier = DFT(W_E, dim=1)  # Transform along vocab axis

# Compute L2 norm along model dimension
fourier_norms = torch.norm(W_E_fourier, dim=0)  # [113]

# Sparse structure at key frequencies wₖ = 2kπ/113
```

**Result:** Sparse structure concentrated at specific "key frequencies"

**Metric:** Not explicit overlap score, but **visual inspection** + **ablation studies**

---

#### 2. **Neuron-Logit Map Analysis**

**What they analyze:**
- **Neuron-logit map WL**: Computed as WU · Wout
- Dimensions: (vocab_size × n_neurons) = (113 × 512)

**How they compute Fourier structure:**
```python
# Compute neuron-logit map
W_L = W_U @ W_out  # [113, 512]

# Apply DFT on logit axis
W_L_fourier = DFT(W_L, dim=0)  # Transform along vocab axis

# Extract key frequency components
# Look at sin(wₖ) and cos(wₖ) directions
```

**Result:** Clear Fourier structure in key frequencies

---

#### 3. **Variance Explained (FVE) Metric**

**What they measure:**
- Fit learned weight projections to **theoretical trigonometric functions**
- Compute R² (coefficient of determination)

**Formula:**
```python
# For each Fourier component k:
# Theoretical: f_theory(x) = cos(2πkx/113) or sin(2πkx/113)
# Learned: f_learned(x) = <W_E[x], direction_k>

# Measure R² between theoretical and learned
R_squared = variance_explained(f_learned, f_theory)
```

**Their results:** R² = 93.2% - 98.2% for key frequencies

---

#### 4. **2D Fourier Analysis on Logits**

**What they analyze:**
- Model logits for all input pairs (a, b)
- Creates 113 × 113 grid

**How they compute:**
```python
# Get logits for all (a, b) pairs
logits = model(all_pairs)  # [113*113, 113]

# Reshape to 2D grid
logits_2d = logits.reshape(113, 113, 113)

# Apply 2D DFT over (a, b) dimensions
logits_fourier = DFT_2D(logits_2d, dims=[0,1])

# Ablate specific frequency components
# Measure performance degradation
```

**Result:** Performance depends on key frequencies

---

## Our Approach (HUSAI Project)

### What We Analyze

**Activations at specific layer/position:**
- **Layer:** 1 (residual stream post)
- **Position:** -2 (answer token)
- **Data:** Activation vectors from all training examples
- **Dimensions:** (n_examples, d_model) = (~11500, 128)

### How We Compute Fourier Overlap

**From `src/analysis/fourier_validation.py`:**

```python
def get_fourier_basis(modulus=113):
    """Generate Fourier basis for modular addition.

    Returns:
        fourier_basis: [2*modulus, modulus] tensor
            Rows are [cos(2π·0·k/p), sin(2π·0·k/p), ..., cos(2π·112·k/p), sin(2π·112·k/p)]
            for each input k
    """
    freqs = torch.arange(modulus).float()
    angles = 2 * torch.pi * freqs.unsqueeze(0) * freqs.unsqueeze(1) / modulus

    cos_basis = torch.cos(angles)  # [113, 113]
    sin_basis = torch.sin(angles)  # [113, 113]

    # Stack: [cos0, sin0, cos1, sin1, ..., cos112, sin112]
    fourier_basis = torch.stack([cos_basis, sin_basis], dim=1)
    fourier_basis = fourier_basis.reshape(2 * modulus, modulus)

    return fourier_basis


def compute_fourier_overlap(sae_features, fourier_basis):
    """Compute maximum cosine similarity between SAE features and Fourier basis.

    Args:
        sae_features: [d_model, d_sae] - SAE decoder weights OR
                      [d_sae, d_model] - SAE features (our current approach)
        fourier_basis: [2*modulus, modulus] - Fourier basis vectors

    Returns:
        overlap: float - mean max cosine similarity
    """
    # Normalize features
    sae_norm = F.normalize(sae_features, dim=1)  # [d_sae, d_model]

    # Normalize Fourier basis
    fourier_norm = F.normalize(fourier_basis, dim=1)  # [226, 113]

    # Compute cosine similarities
    # For each SAE feature, find max similarity to any Fourier component
    similarities = sae_norm @ fourier_norm.T  # [d_sae, 226]
    max_sims = similarities.abs().max(dim=1)[0]  # [d_sae]

    return max_sims.mean().item()
```

### Our Results

- **Transformer activations:** Fourier overlap = 0.2497 - 0.2573
- **Trained SAE features:** Fourier overlap = 0.2534
- **Random SAE features:** Fourier overlap = 0.2539

**Interpretation:** Activations at this layer/position don't show Fourier structure

---

## KEY DIFFERENCES

| Aspect | Nanda et al. (Literature) | Our Approach (HUSAI) |
|--------|--------------------------|----------------------|
| **What is analyzed** | Weight matrices (WE, WL) | Activations at layer/position |
| **Extraction point** | Embedding layer, output layer | Layer 1, position -2 |
| **Fourier transform** | DFT on weight matrices | Cosine similarity on activations |
| **Dimensions** | Along vocab axis (113 dims) | Along model axis (128 dims) |
| **Metric** | Variance explained (R²) | Mean max cosine similarity |
| **Expected result** | Sparse key frequencies | Dense representation |
| **Their results** | 93-98% variance explained | — |
| **Our results** | — | 25-26% overlap |

---

## Why This Matters

### The Disconnect

**Nanda et al. show:** Weights encode Fourier structure
- Embedding matrix maps inputs to sin/cos at key frequencies
- Neuron-logit map combines these frequencies

**We show:** Activations don't have Fourier structure
- Residual stream at layer 1, position -2 doesn't show Fourier pattern
- Only 26% overlap with Fourier basis

### Possible Explanations

#### 1. **Different Extraction Points** (Most Likely)

Nanda et al. analyze:
- **Embedding layer** (input representation)
- **Output layers** (logit computation)

We analyze:
- **Middle layer activations** (intermediate computation)

**Hypothesis:** Fourier structure exists in input/output but not intermediate layers

**Test:** Analyze our transformer's embedding matrix (WE) using their method

---

#### 2. **Weight vs Activation Semantics**

**Weights (their approach):**
- Encode the ALGORITHM (how to compute)
- Fourier structure in weights = "model uses Fourier-based computation"

**Activations (our approach):**
- Encode the DATA (current computation state)
- Fourier structure in activations = "current data is Fourier-like"

**Implication:** A model can use Fourier-based weights WITHOUT having Fourier-structured activations at all positions!

---

#### 3. **Different Fourier Bases**

**Nanda et al.:**
- 1D Fourier basis over vocab dimension
- Frequencies: wₖ = 2kπ/113

**Our approach:**
- 2D Fourier basis (cos/sin pairs) over model dimension
- Different dimensionality (113 vs 128)

**Possible issue:** Dimension mismatch (113 vs 128)

---

## Next Steps

### Option A: Replicate Literature Method (RECOMMENDED)

**Goal:** Measure Fourier overlap using Nanda et al.'s exact approach

**Implementation:**
1. Extract embedding matrix WE from our transformer
2. Apply DFT along vocab dimension
3. Compute variance explained for key frequencies
4. Measure R² against theoretical sin/cos

**Expected outcome:**
- **If R² > 0.9**: Our transformer DID learn Fourier (weights), we were measuring wrong thing
- **If R² < 0.4**: Our transformer truly didn't learn Fourier

**Time:** 1-2 hours

---

### Option B: Multi-Layer Activation Analysis

**Goal:** Check if Fourier structure appears in activations at OTHER layers/positions

**Implementation:**
1. Extract activations from all layers (0, 1) × all positions
2. Compute Fourier overlap for each
3. Identify where (if anywhere) Fourier structure appears

**Expected outcome:** May find Fourier structure at embedding or output positions

**Time:** 30 minutes

---

### Option C: Proceed with Current Narrative

**Goal:** Accept that we measured activations, not weights

**Narrative:**
- Our finding: "SAE activations don't show Fourier structure"
- Literature finding: "Transformer weights encode Fourier circuits"
- Both can be true! Different measurement targets

**Implication:** Our research is valid but answers a different question

---

## Recommendation

**STRONGLY RECOMMEND: Option A**

**Rationale:**
1. **Windsurf identified 70% likelihood** this is a measurement methodology issue
2. **Fast to implement** (1-2 hours)
3. **Decisive test** - will definitively answer if transformer learned Fourier
4. **Aligns with literature** - uses their validated method

**If Option A shows R² > 0.9:**
- ✅ Transformer DID grok to Fourier circuits
- ✅ Our SAE Fourier validation becomes meaningful again
- ✅ Can return to original research narrative

**If Option A shows R² < 0.4:**
- ✅ Confirms transformer didn't learn Fourier
- ✅ Validates our revised narrative
- ✅ Can proceed with paper on SAE instability

---

## Implementation Plan (Option A)

### Script: `scripts/fourier_validation_literature.py`

```python
#!/usr/bin/env python3
"""Validate Fourier learning using Nanda et al.'s exact methodology.

This script:
1. Extracts embedding matrix WE from transformer
2. Applies DFT along vocab dimension
3. Computes variance explained (R²) for key frequencies
4. Compares to theoretical Fourier basis
"""

import torch
import torch.fft as fft
import numpy as np
from pathlib import Path
from src.models.transformer import ModularArithmeticTransformer

def get_theoretical_fourier_embedding(modulus=113, d_model=128, key_freqs=[1, 5]):
    """Generate theoretical Fourier embedding.

    For each frequency k in key_freqs:
        embedding[i, :] should align with cos(2π·k·i/modulus) and sin(2π·k·i/modulus)

    Returns:
        theoretical: [modulus, 2*len(key_freqs)] - expected embedding directions
    """
    theoretical = []

    for k in key_freqs:
        angles = 2 * torch.pi * k * torch.arange(modulus) / modulus
        cos_component = torch.cos(angles).unsqueeze(1)  # [113, 1]
        sin_component = torch.sin(angles).unsqueeze(1)  # [113, 1]
        theoretical.append(cos_component)
        theoretical.append(sin_component)

    theoretical = torch.cat(theoretical, dim=1)  # [113, 2*len(key_freqs)]

    return theoretical


def compute_variance_explained(learned, theoretical):
    """Compute R² between learned and theoretical embeddings.

    Args:
        learned: [modulus, d_model] - learned embedding
        theoretical: [modulus, n_components] - theoretical Fourier components

    Returns:
        r_squared: float - variance explained by theoretical components
    """
    # Project learned onto theoretical basis
    projection = theoretical @ theoretical.T @ learned  # [113, 128]

    # Compute R²
    ss_tot = torch.sum((learned - learned.mean(dim=0))**2)
    ss_res = torch.sum((learned - projection)**2)
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared.item()


def analyze_embedding_fourier(model, modulus=113):
    """Analyze embedding matrix for Fourier structure (Nanda et al. method).

    Args:
        model: Trained transformer
        modulus: Vocabulary size

    Returns:
        dict with Fourier analysis results
    """
    # Extract embedding matrix
    W_E = model.model.embed.W_E.data  # [vocab_size, d_model]
    W_E_numbers = W_E[:modulus, :]  # [113, 128] - only numerical tokens

    print(f"Embedding matrix shape: {W_E_numbers.shape}")

    # Apply DFT along vocab dimension
    W_E_fourier = torch.fft.fft(W_E_numbers, dim=0)  # [113, 128] complex
    W_E_fourier_norms = torch.abs(W_E_fourier)  # [113, 128]

    # Average norm across model dimension
    freq_norms = W_E_fourier_norms.mean(dim=1)  # [113]

    # Identify key frequencies (top 5)
    top_k = 5
    top_freqs = torch.argsort(freq_norms, descending=True)[:top_k]

    print(f"\nTop {top_k} frequencies:")
    for i, freq_idx in enumerate(top_freqs):
        print(f"  Frequency {freq_idx.item()}: norm = {freq_norms[freq_idx].item():.4f}")

    # Compute variance explained by key frequencies
    theoretical = get_theoretical_fourier_embedding(
        modulus=modulus,
        d_model=W_E_numbers.shape[1],
        key_freqs=top_freqs.tolist()[:2]  # Use top 2 frequencies
    )

    r_squared = compute_variance_explained(W_E_numbers, theoretical)

    print(f"\nVariance explained (R²): {r_squared:.4f}")

    if r_squared > 0.9:
        print("✅ EXCELLENT: Strong Fourier structure (matches Nanda et al.)")
    elif r_squared > 0.6:
        print("⚠️  MODERATE: Partial Fourier structure")
    else:
        print("❌ WEAK: No clear Fourier structure")

    return {
        'top_frequencies': top_freqs.tolist(),
        'frequency_norms': freq_norms.tolist(),
        'r_squared': r_squared,
        'embedding_shape': W_E_numbers.shape
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer', type=Path, required=True)
    parser.add_argument('--modulus', type=int, default=113)
    args = parser.parse_args()

    print("="*60)
    print("FOURIER VALIDATION - LITERATURE METHOD")
    print("="*60)
    print(f"\nTransformer: {args.transformer}")
    print(f"Modulus: {args.modulus}")

    # Load model
    print("\nLoading model...")
    model, _ = ModularArithmeticTransformer.load_checkpoint(args.transformer)
    model.eval()

    # Analyze embedding
    results = analyze_embedding_fourier(model, args.modulus)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    return results


if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python scripts/fourier_validation_literature.py \
    --transformer results/transformer_5000ep/transformer_final.pt \
    --modulus 113
```

---

## Summary

**What we learned:**
1. ✅ Nanda et al. analyze **weights** (embedding WE, neuron-logit WL)
2. ✅ We analyze **activations** (layer 1, position -2)
3. ✅ These measure DIFFERENT aspects of Fourier structure
4. ✅ Both can be valid, but answer different questions

**Critical question:**
- Does our transformer's **embedding matrix** show Fourier structure?
- If YES → we were measuring the wrong thing
- If NO → transformer truly didn't learn Fourier

**Next action:** Implement and run Option A (literature method on our transformer)

---

**Status:** Ready to implement literature-based validation
**Expected time:** 1-2 hours
**Decision impact:** CRITICAL - determines entire research direction
