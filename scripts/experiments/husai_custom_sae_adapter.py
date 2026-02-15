#!/usr/bin/env python3
"""Utilities to load HUSAI checkpoints into SAEBench custom SAE objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype string: {name}")
    return mapping[name]


def normalize_architecture(architecture: str | None) -> str | None:
    if architecture is None:
        return None
    key = architecture.strip().lower()
    aliases = {
        "topk": "topk",
        "relu": "relu",
        "batchtopk": "batchtopk",
        "batch_topk": "batchtopk",
        "batch-topk": "batchtopk",
        "jumprelu": "jumprelu",
        "jump_relu": "jumprelu",
        "jump-relu": "jumprelu",
        "matryoshka": "matryoshka",
        "matryoshka_topk": "matryoshka",
        "matryoshka-topk": "matryoshka",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return aliases[key]


def _to_state_dict(obj: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(obj, dict):
        state = obj.get("model_state_dict") or obj.get("sae_state_dict") or obj
        metadata = obj
    else:
        state = obj
        metadata = {}
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint state type: {type(state)}")
    return state, metadata


def _infer_dims(
    state: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[int, int]:
    d_model = metadata.get("d_model")
    d_sae = metadata.get("d_sae")

    if "decoder.weight" in state:
        dec = state["decoder.weight"]
        inferred_d_model = int(dec.shape[0])
        inferred_d_sae = int(dec.shape[1])
    elif "W_dec" in state:
        dec = state["W_dec"]
        inferred_d_sae = int(dec.shape[0])
        inferred_d_model = int(dec.shape[1])
    else:
        raise KeyError("Missing decoder weight in checkpoint (expected decoder.weight or W_dec)")

    d_model_final = int(d_model) if d_model is not None else inferred_d_model
    d_sae_final = int(d_sae) if d_sae is not None else inferred_d_sae
    return d_model_final, d_sae_final


def _infer_architecture(
    state: dict[str, Any],
    metadata: dict[str, Any],
    architecture_override: str | None,
) -> str:
    if architecture_override is not None:
        return normalize_architecture(architecture_override) or "topk"

    meta_arch = metadata.get("architecture")
    if isinstance(meta_arch, str):
        return normalize_architecture(meta_arch) or "topk"

    if "k" in state:
        if "threshold" in state and torch.as_tensor(state["threshold"]).ndim == 0:
            return "batchtopk"
        return "topk"
    if "threshold" in state:
        return "jumprelu"
    return "relu"


def _extract_k(state: dict[str, Any], metadata: dict[str, Any]) -> int | None:
    candidate = metadata.get("k", None)
    if candidate is None and "k" in state:
        candidate = state["k"]
    if candidate is None:
        return None
    if isinstance(candidate, torch.Tensor):
        return int(candidate.item())
    return int(candidate)


def repair_decoder_rows_and_mask_encoder(
    *,
    mapped_state: dict[str, torch.Tensor],
    d_model: int,
    d_sae: int,
    dead_norm_epsilon: float = 1e-12,
    dead_bias_value: float = -1e6,
) -> dict[str, Any]:
    """Repair zero-norm decoder rows and safely normalize decoder rows.

    Some checkpoints can include dead features with exactly-zero decoder rows.
    SAEBench expects unit-norm decoder rows, so we repair dead rows
    deterministically and mask corresponding encoder features to keep them
    effectively inactive during top-k selection.
    """
    if "W_dec" not in mapped_state:
        raise KeyError("mapped_state missing W_dec")

    w_dec = torch.as_tensor(mapped_state["W_dec"]).detach().cpu().clone()
    if w_dec.ndim != 2:
        raise ValueError(f"W_dec must be 2D, got shape={tuple(w_dec.shape)}")
    if w_dec.shape[0] != d_sae or w_dec.shape[1] != d_model:
        raise ValueError(
            "W_dec shape mismatch: "
            f"expected ({d_sae}, {d_model}), got {tuple(w_dec.shape)}"
        )

    row_norms = torch.linalg.vector_norm(w_dec, dim=1)
    dead_idx = (row_norms <= dead_norm_epsilon).nonzero(as_tuple=True)[0]

    if dead_idx.numel() > 0:
        # Deterministic one-hot rows avoid NaN normalization and keep traceability.
        for idx in dead_idx.tolist():
            basis_col = int(idx % d_model)
            w_dec[idx].zero_()
            w_dec[idx, basis_col] = 1.0

        if "W_enc" in mapped_state:
            w_enc = torch.as_tensor(mapped_state["W_enc"]).detach().cpu().clone()
            if w_enc.ndim == 2 and w_enc.shape[0] == d_model and w_enc.shape[1] == d_sae:
                w_enc[:, dead_idx] = 0.0
                mapped_state["W_enc"] = w_enc.contiguous()

        if "b_enc" in mapped_state:
            b_enc = torch.as_tensor(mapped_state["b_enc"]).detach().cpu().clone()
            if b_enc.ndim == 1 and b_enc.shape[0] == d_sae:
                b_enc[dead_idx] = float(dead_bias_value)
                mapped_state["b_enc"] = b_enc.contiguous()

    w_dec = F.normalize(w_dec, dim=1)
    mapped_state["W_dec"] = w_dec.contiguous()

    post_norms = torch.linalg.vector_norm(mapped_state["W_dec"], dim=1)
    max_norm_deviation = float(torch.max(torch.abs(post_norms - 1.0)).item())

    return {
        "dead_features_repaired": int(dead_idx.numel()),
        "dead_feature_indices": [int(i) for i in dead_idx.tolist()],
        "decoder_norm_max_deviation": max_norm_deviation,
    }


def _map_state_to_custom(
    *,
    state: dict[str, Any],
    metadata: dict[str, Any],
    architecture: str,
    d_model: int,
    d_sae: int,
) -> tuple[dict[str, torch.Tensor], int | None]:
    mapped: dict[str, torch.Tensor] = {}

    if "W_enc" in state and "W_dec" in state and "b_enc" in state:
        mapped["W_enc"] = state["W_enc"].detach().cpu().contiguous()
        mapped["W_dec"] = state["W_dec"].detach().cpu().contiguous()
        mapped["b_enc"] = state["b_enc"].detach().cpu().contiguous()
        mapped["b_dec"] = state.get("b_dec", torch.zeros(d_model)).detach().cpu().contiguous()
    else:
        required = ["encoder.weight", "encoder.bias", "decoder.weight"]
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(f"Missing expected keys in checkpoint: {missing}")

        encoder_w = state["encoder.weight"].detach().cpu()
        encoder_b = state["encoder.bias"].detach().cpu()
        decoder_w = state["decoder.weight"].detach().cpu()
        decoder_b = state.get("decoder.bias")

        mapped["W_enc"] = encoder_w.T.contiguous()
        mapped["W_dec"] = decoder_w.T.contiguous()
        mapped["b_enc"] = encoder_b.contiguous()
        if decoder_b is not None:
            mapped["b_dec"] = decoder_b.detach().cpu().contiguous()
        else:
            mapped["b_dec"] = torch.zeros(d_model, dtype=encoder_w.dtype)

    k_value = _extract_k(state, metadata)

    if architecture in {"topk", "batchtopk", "matryoshka"}:
        if k_value is None:
            raise KeyError(f"Checkpoint missing `k` for architecture={architecture}")
        mapped["k"] = torch.tensor(k_value, dtype=torch.int32)

    if architecture == "batchtopk":
        threshold = state.get("threshold", torch.tensor(-1.0))
        mapped["threshold"] = torch.as_tensor(threshold).detach().cpu().to(dtype=torch.float32)

    if architecture == "jumprelu":
        if "threshold" in state:
            threshold = torch.as_tensor(state["threshold"]).detach().cpu()
        elif "log_threshold" in state:
            threshold = torch.as_tensor(state["log_threshold"]).detach().cpu().exp()
        else:
            threshold = torch.zeros(d_sae, dtype=torch.float32)
        mapped["threshold"] = threshold.to(dtype=torch.float32)

    return mapped, k_value


def build_custom_sae_from_checkpoint(
    *,
    checkpoint_path: Path,
    model_name: str,
    hook_layer: int,
    hook_name: str,
    device: str,
    dtype: torch.dtype,
    architecture_override: str | None = None,
):
    """Load HUSAI checkpoint and return a SAEBench-compatible custom SAE object."""
    try:
        from sae_bench.custom_saes.batch_topk_sae import BatchTopKSAE
        from sae_bench.custom_saes.jumprelu_sae import JumpReluSAE
        from sae_bench.custom_saes.relu_sae import ReluSAE
        from sae_bench.custom_saes.topk_sae import TopKSAE
    except Exception as exc:
        raise RuntimeError("SAEBench is required for custom SAE loading (`pip install sae-bench`).") from exc

    obj = torch.load(checkpoint_path, map_location="cpu")
    state, metadata = _to_state_dict(obj)

    architecture = _infer_architecture(state, metadata, architecture_override)
    d_model, d_sae = _infer_dims(state, metadata)
    mapped_state, k_value = _map_state_to_custom(
        state=state,
        metadata=metadata,
        architecture=architecture,
        d_model=d_model,
        d_sae=d_sae,
    )
    repair_info = repair_decoder_rows_and_mask_encoder(
        mapped_state=mapped_state,
        d_model=d_model,
        d_sae=d_sae,
    )

    ctor_kwargs = {
        "d_in": d_model,
        "d_sae": d_sae,
        "model_name": model_name,
        "hook_layer": hook_layer,
        "hook_name": hook_name,
        "device": torch.device(device),
        "dtype": dtype,
    }

    eval_architecture = "topk" if architecture == "matryoshka" else architecture

    if architecture in {"topk", "matryoshka"}:
        sae = TopKSAE(k=int(k_value), **ctor_kwargs)
    elif architecture == "batchtopk":
        sae = BatchTopKSAE(k=int(k_value), **ctor_kwargs)
    elif architecture == "jumprelu":
        sae = JumpReluSAE(**ctor_kwargs)
    elif architecture == "relu":
        sae = ReluSAE(**ctor_kwargs)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    sae.load_state_dict(mapped_state, strict=False)
    sae.cfg.architecture = eval_architecture
    sae.to(device=torch.device(device), dtype=dtype)

    # Keep decoder rows unit-normalized; SAEBench eval expects this.
    if hasattr(sae, "W_dec"):
        with torch.no_grad():
            sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)

    if hasattr(sae, "check_decoder_norms") and not sae.check_decoder_norms():
        raise ValueError(
            "Failed to normalize decoder rows for custom SAE "
            f"(repaired_dead={repair_info['dead_features_repaired']}, "
            f"max_deviation={repair_info['decoder_norm_max_deviation']})"
        )

    metadata_out = {
        "architecture": architecture,
        "eval_architecture": eval_architecture,
        "d_model": d_model,
        "d_sae": d_sae,
        "k": k_value,
        "dead_features_repaired": repair_info["dead_features_repaired"],
        "decoder_norm_max_deviation": repair_info["decoder_norm_max_deviation"],
    }
    return sae, metadata_out
