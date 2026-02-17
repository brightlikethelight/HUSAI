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
        supervised_proxy_mode="none",
        supervised_proxy_weight=0.0,
        supervised_proxy_num_classes=0,
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

    acts, source, labels, supervised_meta = assign_v3.load_training_activations(args)

    assert isinstance(acts, torch.Tensor)
    assert tuple(acts.shape) == (13, 6)
    assert source["source"] == "external_cache"
    assert source["data_meta"]["d_model"] == 6
    assert source["data_meta"]["num_files_used"] == 1
    assert labels is None
    assert supervised_meta is None


def test_load_training_activations_external_cache_file_id_labels(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.randn(7, 4), cache_dir / "datasetA_blocks.0.hook_resid_pre.pt")
    torch.save(torch.randn(9, 4), cache_dir / "datasetB_blocks.0.hook_resid_pre.pt")

    args = _base_args(tmp_path)
    args.activation_cache_dir = cache_dir
    args.max_files = 2
    args.max_rows_per_file = 5
    args.max_total_rows = 20
    args.supervised_proxy_mode = "file_id"

    acts, source, labels, supervised_meta = assign_v3.load_training_activations(args)

    assert isinstance(acts, torch.Tensor)
    assert labels is not None
    assert supervised_meta is not None
    assert acts.shape[0] == labels.shape[0]
    assert set(labels.tolist()) == {0, 1}
    assert supervised_meta["mode"] == "file_id"
    assert supervised_meta["num_classes"] == 2
    assert source["source"] == "external_cache"


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
    acts, source, labels, supervised_meta = assign_v3.load_training_activations(args)

    assert called["value"]
    assert torch.equal(acts, expected.float())
    assert source["source"] == "modular_assignment"
    assert source["layer"] == 1
    assert source["modulus"] == 113
    assert labels is None
    assert supervised_meta is None


def test_build_external_checkpoint_pool_excludes_ref_and_sorts() -> None:
    rec = {
        "seed_ref": 42,
        "per_seed_metrics": [
            {"seed": 42, "checkpoint": "ckpt_ref.pt", "alignment_to_ref": 1.0, "explained_variance": 0.8},
            {"seed": 123, "checkpoint": "ckpt_a.pt", "alignment_to_ref": 0.7, "explained_variance": 0.6},
            {"seed": 456, "checkpoint": "ckpt_b.pt", "alignment_to_ref": 0.9, "explained_variance": 0.5},
            {"seed": 789, "checkpoint": "ckpt_c.pt", "alignment_to_ref": 0.9, "explained_variance": 0.7},
        ],
    }

    pool = assign_v3.build_external_checkpoint_pool(rec, include_ref=False, max_candidates=2)

    assert len(pool) == 2
    assert [row["seed"] for row in pool] == [789, 456]
    assert all(row["seed"] != 42 for row in pool)


def test_select_external_candidate_external_score_prefers_floor_passing() -> None:
    args = Namespace(
        external_checkpoint_policy="external_score",
        external_candidate_weight_saebench=0.45,
        external_candidate_weight_cebench=0.45,
        external_candidate_weight_alignment=0.05,
        external_candidate_weight_ev=0.05,
        external_candidate_min_saebench_delta=-0.02,
        external_candidate_min_cebench_delta=-10.0,
        external_candidate_require_both=True,
        run_saebench=True,
        run_cebench=True,
    )

    candidate_evals = [
        {
            "seed": 123,
            "checkpoint": "ckpt_bad_external.pt",
            "alignment_to_ref": 0.95,
            "explained_variance": 0.80,
            "saebench_delta": -0.08,
            "cebench_delta": -20.0,
            "external_eval": {"saebench_returncode": 0, "cebench_returncode": 0},
        },
        {
            "seed": 456,
            "checkpoint": "ckpt_good_external.pt",
            "alignment_to_ref": 0.75,
            "explained_variance": 0.60,
            "saebench_delta": -0.01,
            "cebench_delta": -5.0,
            "external_eval": {"saebench_returncode": 0, "cebench_returncode": 0},
        },
    ]

    selected, ranked = assign_v3.select_external_candidate(candidate_evals, args=args)

    assert selected["checkpoint"] == "ckpt_good_external.pt"
    assert ranked[0]["selection"]["passes_external_floor"] is True
    assert ranked[1]["selection"]["passes_external_floor"] is False


def test_finite_or_default_preserves_zero_and_filters_non_finite() -> None:
    assert assign_v3.finite_or_default(0.0, float("-inf")) == 0.0
    assert assign_v3.finite_or_default(float("nan"), -7.0) == -7.0
    assert assign_v3.finite_or_default(float("inf"), -3.0) == -3.0


def test_normalize_ignores_non_finite_values() -> None:
    norm = assign_v3.normalize([0.0, float("nan"), float("inf"), -1.0])
    assert set(norm.keys()) == {0, 3}
    assert norm[0] == 1.0
    assert norm[3] == 0.0
