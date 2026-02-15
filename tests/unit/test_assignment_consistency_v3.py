from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch

from scripts.experiments import run_assignment_consistency_v3 as assign_v3


def _base_args(tmp_path: Path) -> Namespace:
    return Namespace(
        activation_cache_dir=None,
        activation_glob="*_blocks.0.hook_resid_pre.pt",
        max_files=10,
        max_rows_per_file=1024,
        max_total_rows=10000,
        source_cache_seed=None,
        seed_ref=42,
        activation_cache=tmp_path / "acts.pt",
        transformer_checkpoint=tmp_path / "transformer.pt",
        layer=1,
        batch_size=8,
        device="cpu",
        modulus=113,
    )


def test_load_training_activations_external_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensor = torch.randn(13, 6)
    torch.save(tensor, cache_dir / "toy_blocks.0.hook_resid_pre.pt")

    args = _base_args(tmp_path)
    args.activation_cache_dir = cache_dir
    args.max_files = 1
    args.max_rows_per_file = 13
    args.max_total_rows = 13

    acts, source = assign_v3.load_training_activations(args)

    assert isinstance(acts, torch.Tensor)
    assert tuple(acts.shape) == (13, 6)
    assert source["source"] == "external_cache"
    assert source["data_meta"]["d_model"] == 6
    assert source["data_meta"]["num_files_used"] == 1


def test_load_training_activations_modular_path(monkeypatch, tmp_path: Path) -> None:
    expected = torch.ones(7, 5)
    called: dict[str, bool] = {"value": False}

    def fake_loader(*, cache_path, transformer_checkpoint, layer, batch_size, device, modulus, seed):
        called["value"] = True
        assert cache_path == tmp_path / "acts.pt"
        assert transformer_checkpoint == tmp_path / "transformer.pt"
        assert layer == 1
        assert batch_size == 8
        assert device == "cpu"
        assert modulus == 113
        assert seed == 42
        return expected

    monkeypatch.setattr(assign_v3, "load_or_extract_activations", fake_loader)

    args = _base_args(tmp_path)
    acts, source = assign_v3.load_training_activations(args)

    assert called["value"]
    assert torch.equal(acts, expected.float())
    assert source["source"] == "modular_assignment"
    assert source["layer"] == 1
    assert source["modulus"] == 113
