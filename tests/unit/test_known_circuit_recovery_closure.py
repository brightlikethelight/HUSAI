from __future__ import annotations

import torch
import pytest

from scripts.experiments.run_known_circuit_recovery_closure import (
    build_fourier_basis,
    project_token_fourier_to_model_space,
)


def test_project_token_fourier_to_model_space_shapes_and_norms() -> None:
    modulus = 11
    embedding = torch.randn(modulus, 7, dtype=torch.float32)
    token_basis = build_fourier_basis(modulus, [1, 2, 3])

    model_basis, meta = project_token_fourier_to_model_space(embedding, token_basis)

    assert model_basis.shape[0] == 7
    assert model_basis.shape[1] > 0
    assert meta["token_basis_dim"] == token_basis.shape[1]
    assert meta["model_basis_dim"] == model_basis.shape[1]

    norms = torch.linalg.vector_norm(model_basis, dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_project_token_fourier_to_model_space_mismatch_raises() -> None:
    embedding = torch.randn(13, 5, dtype=torch.float32)
    token_basis = torch.randn(11, 4, dtype=torch.float32)

    with pytest.raises(ValueError):
        project_token_fourier_to_model_space(embedding, token_basis)
