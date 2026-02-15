from __future__ import annotations

import torch
import pytest

from scripts.experiments.husai_custom_sae_adapter import (
    normalize_architecture,
    repair_decoder_rows_and_mask_encoder,
)


def test_normalize_architecture_matryoshka_aliases() -> None:
    assert normalize_architecture("matryoshka") == "matryoshka"
    assert normalize_architecture("matryoshka_topk") == "matryoshka"
    assert normalize_architecture("matryoshka-topk") == "matryoshka"


def test_normalize_architecture_routed_aliases() -> None:
    assert normalize_architecture("routed_topk") == "routed_topk"
    assert normalize_architecture("routed-topk") == "routed_topk"
    assert normalize_architecture("routesae") == "routed_topk"


def test_normalize_architecture_unsupported() -> None:
    with pytest.raises(ValueError):
        normalize_architecture("unknown_arch")


def test_repair_decoder_rows_and_mask_encoder_repairs_dead_features() -> None:
    mapped = {
        "W_dec": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "W_enc": torch.ones((3, 2), dtype=torch.float32),
        "b_enc": torch.zeros(2, dtype=torch.float32),
    }

    info = repair_decoder_rows_and_mask_encoder(mapped_state=mapped, d_model=3, d_sae=2)

    assert info["dead_features_repaired"] == 1
    assert info["dead_feature_indices"] == [0]

    # Decoder rows are unit-normalized after repair.
    norms = torch.linalg.vector_norm(mapped["W_dec"], dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    # Dead feature is masked out in encoder state.
    assert torch.allclose(mapped["W_enc"][:, 0], torch.zeros(3))
    assert mapped["b_enc"][0] < -1e5

    # Non-dead feature stays active.
    assert torch.allclose(mapped["W_enc"][:, 1], torch.ones(3))
    assert mapped["b_enc"][1] == 0.0


def test_repair_decoder_rows_and_mask_encoder_no_dead_features() -> None:
    mapped = {
        "W_dec": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "W_enc": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "b_enc": torch.tensor([0.2, -0.1], dtype=torch.float32),
    }
    old_w_enc = mapped["W_enc"].clone()
    old_b_enc = mapped["b_enc"].clone()

    info = repair_decoder_rows_and_mask_encoder(mapped_state=mapped, d_model=3, d_sae=2)

    assert info["dead_features_repaired"] == 0
    assert info["dead_feature_indices"] == []
    assert torch.allclose(mapped["W_enc"], old_w_enc)
    assert torch.allclose(mapped["b_enc"], old_b_enc)

    norms = torch.linalg.vector_norm(mapped["W_dec"], dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
