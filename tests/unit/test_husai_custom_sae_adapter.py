from __future__ import annotations

import pytest

from scripts.experiments.husai_custom_sae_adapter import normalize_architecture


def test_normalize_architecture_matryoshka_aliases() -> None:
    assert normalize_architecture("matryoshka") == "matryoshka"
    assert normalize_architecture("matryoshka_topk") == "matryoshka"
    assert normalize_architecture("matryoshka-topk") == "matryoshka"


def test_normalize_architecture_unsupported() -> None:
    with pytest.raises(ValueError):
        normalize_architecture("routesae")
