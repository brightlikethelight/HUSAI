"""Tests for CE-Bench baseline mapping in external scaling study."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.run_external_metric_scaling_study import (
    load_baseline_map,
    select_baseline_for_condition,
)


def test_select_baseline_prefers_hook_name_then_layer_then_default() -> None:
    baseline_map = {
        "default": "/tmp/default.json",
        "1": "/tmp/layer1.json",
        "blocks.1.hook_resid_pre": "/tmp/hook1.json",
    }

    selected = select_baseline_for_condition(
        hook_layer=1,
        hook_name="blocks.1.hook_resid_pre",
        default_baseline=Path("/tmp/fallback.json"),
        baseline_map=baseline_map,
    )
    assert selected == Path("/tmp/hook1.json")

    selected = select_baseline_for_condition(
        hook_layer=1,
        hook_name="blocks.9.hook_resid_pre",
        default_baseline=Path("/tmp/fallback.json"),
        baseline_map=baseline_map,
    )
    assert selected == Path("/tmp/layer1.json")

    selected = select_baseline_for_condition(
        hook_layer=2,
        hook_name="blocks.9.hook_resid_pre",
        default_baseline=Path("/tmp/fallback.json"),
        baseline_map=baseline_map,
    )
    assert selected == Path("/tmp/default.json")


def test_load_baseline_map_ignores_non_string_values(tmp_path: Path) -> None:
    payload = {
        "default": "docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json",
        "1": 123,
        "bad": None,
    }
    mapping_path = tmp_path / "map.json"
    mapping_path.write_text(json.dumps(payload))

    out = load_baseline_map(mapping_path)
    assert "default" in out
    assert "1" not in out
    assert "bad" not in out
