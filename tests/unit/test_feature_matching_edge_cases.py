from __future__ import annotations

import math

import numpy as np

from src.analysis.feature_matching import compute_feature_statistics, visualize_overlap_matrix


def test_compute_feature_statistics_handles_single_model() -> None:
    overlap = np.array([[1.0]], dtype=np.float64)
    stats = compute_feature_statistics([object()], overlap_matrix=overlap)

    assert math.isnan(stats["mean_overlap"])
    assert math.isnan(stats["std_overlap"])
    assert math.isnan(stats["min_overlap"])
    assert math.isnan(stats["max_overlap"])
    assert math.isnan(stats["median_overlap"])
    assert math.isnan(stats["above_threshold"])


def test_visualize_overlap_matrix_handles_single_model() -> None:
    overlap = np.array([[1.0]], dtype=np.float64)
    fig = visualize_overlap_matrix(overlap_matrix=overlap, labels=["seed42"], title="single")
    try:
        assert fig is not None
    finally:
        fig.clf()
