from __future__ import annotations

import math

import pytest
import torch

from scripts.experiments import run_assignment_consistency_v2 as assign_v2


def test_assignment_update_interval_reduces_hungarian_solves(monkeypatch) -> None:
    torch.manual_seed(0)
    activations = torch.randn(8, 6)
    ref_decoder = torch.randn(6, 4)

    calls = {"n": 0}
    real_linear_sum_assignment = assign_v2.linear_sum_assignment

    def wrapped_linear_sum_assignment(cost):
        calls["n"] += 1
        return real_linear_sum_assignment(cost)

    monkeypatch.setattr(assign_v2, "linear_sum_assignment", wrapped_linear_sum_assignment)

    batch_size = 2
    epochs = 2
    interval = 3

    assign_v2.train_topk_assignment_v2(
        activations=activations,
        d_sae=4,
        k=2,
        seed=123,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        device="cpu",
        lambda_consistency=0.1,
        ref_decoder=ref_decoder,
        assignment_update_interval=interval,
    )

    total_steps = math.ceil(activations.shape[0] / batch_size) * epochs
    expected_solves = math.ceil(total_steps / interval)
    assert calls["n"] == expected_solves


def test_assignment_update_interval_must_be_positive() -> None:
    activations = torch.randn(4, 3)

    with pytest.raises(ValueError, match="assignment_update_interval"):
        assign_v2.train_topk_assignment_v2(
            activations=activations,
            d_sae=3,
            k=2,
            seed=0,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            device="cpu",
            lambda_consistency=0.0,
            ref_decoder=None,
            assignment_update_interval=0,
        )
