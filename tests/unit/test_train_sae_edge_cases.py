from __future__ import annotations

import builtins

import torch

from src.models.simple_sae import ReLUSAE, TopKSAE
from src.training.train_sae import train_sae
from src.utils.config import SAEConfig


def _topk_config(batch_size: int) -> SAEConfig:
    return SAEConfig(
        architecture="topk",
        input_dim=4,
        expansion_factor=2,
        sparsity_level=2,
        learning_rate=1e-3,
        batch_size=batch_size,
        num_epochs=1,
        k=2,
        seed=0,
    )


def test_topk_training_includes_aux_loss_in_total_loss() -> None:
    torch.manual_seed(0)
    model = TopKSAE(d_model=4, d_sae=8, k=2)
    activations = torch.randn(16, 4)
    config = _topk_config(batch_size=8)

    metrics = train_sae(
        sae=model,
        activations=activations,
        config=config,
        use_wandb=False,
        device="cpu",
        verbose=False,
    )

    assert len(metrics.loss) == 1
    assert len(metrics.mse_loss) == 1
    assert metrics.loss[0] > metrics.mse_loss[0]


def test_train_sae_handles_dataset_smaller_than_batch_size() -> None:
    torch.manual_seed(1)
    model = TopKSAE(d_model=4, d_sae=8, k=2)
    activations = torch.randn(2, 4)
    config = _topk_config(batch_size=16)

    metrics = train_sae(
        sae=model,
        activations=activations,
        config=config,
        use_wandb=False,
        device="cpu",
        verbose=False,
    )

    assert len(metrics.loss) == 1
    assert metrics.loss[0] == metrics.loss[0]


def test_train_sae_no_wandb_dependency_when_import_missing(monkeypatch) -> None:
    torch.manual_seed(2)
    model = ReLUSAE(d_model=4, d_sae=8, l1_coef=1e-3)
    activations = torch.randn(4, 4)
    config = SAEConfig(
        architecture="relu",
        input_dim=4,
        expansion_factor=2,
        sparsity_level=1e-3,
        learning_rate=1e-3,
        batch_size=2,
        num_epochs=1,
        l1_coefficient=1e-3,
        seed=0,
    )

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "wandb":
            raise ImportError("wandb missing in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    metrics = train_sae(
        sae=model,
        activations=activations,
        config=config,
        use_wandb=True,
        device="cpu",
        verbose=False,
    )

    assert len(metrics.loss) == 1
