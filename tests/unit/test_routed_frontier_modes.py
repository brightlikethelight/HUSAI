from __future__ import annotations

import torch
from torch import nn

from scripts.experiments.run_routed_frontier_external import (
    build_expert_slices,
    routed_topk_features,
)


def _router_for_two_experts() -> nn.Linear:
    router = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        router.weight.zero_()
        router.weight[0, 0] = 1.0
        router.weight[1, 0] = -1.0
    return router


def test_expert_topk_keeps_k_features_within_routed_expert() -> None:
    batch = torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    pre = torch.tensor(
        [
            [9.0, 8.0, 7.0, 6.0, 100.0, 99.0, 98.0, 97.0],
            [100.0, 99.0, 98.0, 97.0, 9.0, 8.0, 7.0, 6.0],
        ]
    )
    slices = build_expert_slices(d_sae=8, num_experts=2)
    router = _router_for_two_experts()

    feats, _ = routed_topk_features(
        pre=pre,
        batch=batch,
        router=router,
        d_sae=8,
        expert_slices=slices,
        k=2,
        mode="expert_topk",
    )

    nz0 = torch.nonzero(feats[0]).flatten().tolist()
    nz1 = torch.nonzero(feats[1]).flatten().tolist()

    assert len(nz0) == 2
    assert len(nz1) == 2
    assert all(i < 4 for i in nz0)
    assert all(i >= 4 for i in nz1)


def test_global_mask_can_reduce_effective_l0_after_routing() -> None:
    batch = torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    pre = torch.tensor(
        [
            [9.0, 8.0, 7.0, 6.0, 100.0, 99.0, 98.0, 97.0],
            [100.0, 99.0, 98.0, 97.0, 9.0, 8.0, 7.0, 6.0],
        ]
    )
    slices = build_expert_slices(d_sae=8, num_experts=2)
    router = _router_for_two_experts()

    feats, _ = routed_topk_features(
        pre=pre,
        batch=batch,
        router=router,
        d_sae=8,
        expert_slices=slices,
        k=2,
        mode="global_mask",
    )

    # Global top-k is dominated by the opposite expert, then masked out.
    l0 = (feats != 0).float().sum(dim=-1).tolist()
    assert l0[0] < 2.0
    assert l0[1] < 2.0
