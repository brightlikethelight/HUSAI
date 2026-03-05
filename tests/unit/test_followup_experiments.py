"""Tests for follow-up experiment utilities.

Tests the core computational functions used across experiments 1.1-2.4
without requiring trained transformers or GPU access.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.simple_sae import TopKSAE, ReLUSAE


# ── PWMCC tests ──


def _compute_pwmcc(d1: torch.Tensor, d2: torch.Tensor) -> float:
    """Reference PWMCC implementation for testing."""
    d1_norm = F.normalize(d1, dim=0)
    d2_norm = F.normalize(d2, dim=0)
    cos_sim = d1_norm.T @ d2_norm
    max_1to2 = cos_sim.abs().max(dim=1)[0].mean().item()
    max_2to1 = cos_sim.abs().max(dim=0)[0].mean().item()
    return (max_1to2 + max_2to1) / 2


def test_pwmcc_identical_decoders():
    """PWMCC of identical decoders should be 1.0."""
    d = torch.randn(64, 128)
    d = F.normalize(d, dim=0)
    assert abs(_compute_pwmcc(d, d) - 1.0) < 1e-5


def test_pwmcc_random_decoders_bounded():
    """PWMCC of random decoders should be in (0, 1)."""
    d1 = torch.randn(64, 256)
    d2 = torch.randn(64, 256)
    p = _compute_pwmcc(d1, d2)
    assert 0 < p < 1, f"PWMCC={p} out of bounds"


def test_pwmcc_symmetric():
    """PWMCC should be symmetric."""
    d1 = torch.randn(64, 128)
    d2 = torch.randn(64, 128)
    assert abs(_compute_pwmcc(d1, d2) - _compute_pwmcc(d2, d1)) < 1e-5


def test_pwmcc_permuted_columns():
    """Permuting columns should not change PWMCC (max is over all columns)."""
    d1 = torch.randn(64, 128)
    perm = torch.randperm(128)
    d2 = d1[:, perm]
    p = _compute_pwmcc(d1, d2)
    assert abs(p - 1.0) < 1e-4, f"Permuted PWMCC={p}, expected ~1.0"


# ── Effective rank tests ──


def _compute_effective_rank(acts: torch.Tensor) -> float:
    centered = acts - acts.mean(dim=0, keepdim=True)
    _, S, _ = torch.svd(centered)
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-10]
    entropy = -(S_norm * torch.log(S_norm)).sum()
    return torch.exp(entropy).item()


def test_effective_rank_identity_covariance():
    """Identity covariance (equal singular values) should give rank = d."""
    d = 16
    acts = torch.randn(1000, d)
    er = _compute_effective_rank(acts)
    # Should be close to d (within margin due to finite samples)
    assert d * 0.6 < er <= d, f"Effective rank={er:.1f}, expected ~{d}"


def test_effective_rank_rank1():
    """Rank-1 data should have effective rank close to 1."""
    direction = F.normalize(torch.randn(1, 32), dim=1)
    acts = torch.randn(500, 1) * direction + 0.001 * torch.randn(500, 32)
    er = _compute_effective_rank(acts)
    assert er < 3, f"Effective rank={er:.1f}, expected ~1 for rank-1 data"


def test_effective_rank_positive():
    """Effective rank should always be positive."""
    acts = torch.randn(100, 8)
    er = _compute_effective_rank(acts)
    assert er > 0


# ── Principal angles / subspace overlap tests ──


def _compute_principal_angles(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
    M = U1.T @ U2
    _, S, _ = torch.svd(M)
    S = S.clamp(0.0, 1.0)
    return torch.acos(S)


def _compute_subspace_overlap(U1: torch.Tensor, U2: torch.Tensor) -> float:
    angles = _compute_principal_angles(U1, U2)
    return float(torch.cos(angles).pow(2).mean().item())


def test_subspace_overlap_identical():
    """Identical subspaces should have overlap 1.0."""
    U = torch.linalg.qr(torch.randn(32, 8))[0]
    overlap = _compute_subspace_overlap(U, U)
    assert abs(overlap - 1.0) < 1e-4, f"Overlap={overlap}, expected 1.0"


def test_subspace_overlap_orthogonal():
    """Orthogonal subspaces should have overlap 0.0."""
    Q = torch.linalg.qr(torch.randn(32, 16))[0]
    U1 = Q[:, :8]
    U2 = Q[:, 8:16]
    overlap = _compute_subspace_overlap(U1, U2)
    assert overlap < 0.01, f"Overlap={overlap}, expected ~0 for orthogonal subspaces"


def test_subspace_overlap_bounded():
    """Overlap should be in [0, 1]."""
    U1 = torch.linalg.qr(torch.randn(32, 8))[0]
    U2 = torch.linalg.qr(torch.randn(32, 8))[0]
    overlap = _compute_subspace_overlap(U1, U2)
    assert 0 <= overlap <= 1 + 1e-5, f"Overlap={overlap} out of bounds"


def test_principal_angles_dimensions():
    """Principal angles should have length = min(k1, k2)."""
    U1 = torch.linalg.qr(torch.randn(32, 4))[0]
    U2 = torch.linalg.qr(torch.randn(32, 8))[0]
    angles = _compute_principal_angles(U1, U2)
    assert len(angles) == 4


# ── Contrastive alignment loss tests ──


def _compute_alignment_loss(d_a: torch.Tensor, d_b: torch.Tensor) -> float:
    w_a = F.normalize(d_a, dim=0)
    w_b = F.normalize(d_b, dim=0)
    cos_sim = w_a.T @ w_b
    max_a_to_b = cos_sim.abs().max(dim=1)[0]
    max_b_to_a = cos_sim.abs().max(dim=0)[0]
    return float(2.0 - max_a_to_b.mean() - max_b_to_a.mean())


def test_alignment_loss_identical():
    """Identical decoders should have alignment loss ~0."""
    d = F.normalize(torch.randn(64, 128), dim=0)
    loss = _compute_alignment_loss(d, d)
    assert loss < 0.01, f"Alignment loss={loss}, expected ~0"


def test_alignment_loss_random_positive():
    """Random decoders should have positive alignment loss."""
    d1 = torch.randn(64, 128)
    d2 = torch.randn(64, 128)
    loss = _compute_alignment_loss(d1, d2)
    assert loss > 0, f"Alignment loss should be positive for random decoders"


def test_alignment_loss_bounded():
    """Alignment loss should be in [0, 2]."""
    d1 = torch.randn(64, 256)
    d2 = torch.randn(64, 256)
    loss = _compute_alignment_loss(d1, d2)
    assert 0 <= loss <= 2 + 1e-5, f"Loss={loss} out of [0,2]"


# ── SAE pinning tests ──


def test_pinned_sae_preserves_columns():
    """Freezing decoder columns should preserve them through training."""
    torch.manual_seed(42)
    d_model, d_sae, k = 32, 64, 8
    acts = torch.randn(200, d_model)

    # Create reference SAE
    ref_sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)

    # Create new SAE with pinned columns
    pinned_indices = [0, 1, 2, 3]
    new_sae = TopKSAE(d_model=d_model, d_sae=d_sae, k=k)

    # Copy pinned columns
    with torch.no_grad():
        for idx in pinned_indices:
            new_sae.decoder.weight.data[:, idx] = ref_sae.decoder.weight.data[:, idx]

    ref_cols_before = ref_sae.decoder.weight.data[:, pinned_indices].clone()

    # Train with frozen pinned columns
    optimizer = torch.optim.Adam(new_sae.parameters(), lr=1e-3)
    pinned_mask = torch.zeros(d_sae, dtype=torch.bool)
    pinned_mask[pinned_indices] = True

    for _ in range(5):
        recon, latents, aux = new_sae(acts)
        loss = F.mse_loss(recon, acts) + aux
        optimizer.zero_grad()
        loss.backward()
        if new_sae.decoder.weight.grad is not None:
            new_sae.decoder.weight.grad[:, pinned_mask] = 0.0
        optimizer.step()
        new_sae.normalize_decoder()

    # Check pinned columns haven't changed
    pinned_cols_after = new_sae.decoder.weight.data[:, pinned_indices]
    ref_cols_normalized = F.normalize(ref_cols_before, dim=0)
    pinned_normalized = F.normalize(pinned_cols_after, dim=0)

    # Should be identical up to normalization
    cos_sims = (ref_cols_normalized * pinned_normalized).sum(dim=0)
    assert cos_sims.min().item() > 0.99, (
        f"Pinned columns drifted! Min cosine sim = {cos_sims.min().item():.4f}"
    )


# ── Random baseline tests ──


def test_random_baseline_below_trained():
    """Random PWMCC should be lower than 'trained' (correlated) decoders."""
    # Simulate "trained" decoders that share structure
    shared = torch.randn(64, 128)
    d1 = shared + 0.3 * torch.randn(64, 128)
    d2 = shared + 0.3 * torch.randn(64, 128)
    trained_pwmcc = _compute_pwmcc(d1, d2)

    # Random baseline
    r1 = torch.randn(64, 128)
    r2 = torch.randn(64, 128)
    random_pwmcc = _compute_pwmcc(r1, r2)

    assert trained_pwmcc > random_pwmcc, (
        f"Trained PWMCC ({trained_pwmcc:.3f}) should exceed random ({random_pwmcc:.3f})"
    )


def test_random_baseline_scales_with_dimensions():
    """Random PWMCC should increase with d_sae (more chances for matches)."""
    d_model = 64
    pwmcc_small = np.mean([
        _compute_pwmcc(torch.randn(d_model, 32), torch.randn(d_model, 32))
        for _ in range(10)
    ])
    pwmcc_large = np.mean([
        _compute_pwmcc(torch.randn(d_model, 512), torch.randn(d_model, 512))
        for _ in range(10)
    ])
    assert pwmcc_large > pwmcc_small, (
        f"Larger d_sae ({pwmcc_large:.3f}) should have higher random PWMCC "
        f"than smaller ({pwmcc_small:.3f})"
    )


# ── TopK SAE basic tests ──


def test_topk_sae_output_shapes():
    sae = TopKSAE(d_model=64, d_sae=256, k=16)
    x = torch.randn(32, 64)
    recon, latents, aux = sae(x)
    assert recon.shape == (32, 64)
    assert latents.shape == (32, 256)
    assert aux.ndim == 0


def test_topk_sae_sparsity():
    """TopK should produce exactly k nonzero features per sample."""
    sae = TopKSAE(d_model=64, d_sae=256, k=16)
    sae.eval()
    x = torch.randn(10, 64)
    with torch.no_grad():
        _, latents, _ = sae(x)
    l0 = (latents != 0).float().sum(dim=-1)
    assert (l0 == 16).all(), f"Expected L0=16, got {l0.tolist()}"


def test_relu_sae_output_shapes():
    sae = ReLUSAE(d_model=64, d_sae=256, l1_coef=1e-3)
    x = torch.randn(32, 64)
    recon, latents, l1_loss = sae(x)
    assert recon.shape == (32, 64)
    assert latents.shape == (32, 256)
    assert l1_loss.ndim == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
