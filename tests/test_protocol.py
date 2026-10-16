"""Metric, alignment, stereo conversion, and edge agreement."""

from __future__ import annotations

import numpy as np
import pytest

from depth_eval.align import least_squares_scale, median_scale
from depth_eval.edges import edge_f1
from depth_eval.generate import _blur
from depth_eval.masks import assert_same_frame
from depth_eval.metrics import compute_metrics
from depth_eval.stereo import disparity_to_depth


def test_abs_rel_rmse_and_delta():
    gt = np.ones((2, 2))
    pred = np.array([[1.0, 1.0], [1.0, 2.0]])
    metrics = compute_metrics(pred, gt)
    assert metrics["abs_rel"] == pytest.approx(0.25)
    assert metrics["rmse"] == pytest.approx(0.5)
    assert metrics["delta_1"] == pytest.approx(0.75)
    assert metrics["delta_3"] == pytest.approx(0.75)
    assert metrics["coverage"] == pytest.approx(1.0)


def test_empty_prediction_has_no_average_and_zero_coverage():
    gt = np.ones((4, 4))
    pred = np.full((4, 4), np.nan)
    metrics = compute_metrics(pred, gt)
    assert metrics["abs_rel"] is None
    assert metrics["coverage"] == 0.0
    assert metrics["rel_error_sum"] == pytest.approx(16.0)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="分辨率"):
        assert_same_frame(np.ones((4, 4)), np.ones((4, 5)))


def test_median_scale_ignores_a_single_outlier():
    gt = np.ones((8, 8))
    pred = np.full((8, 8), 0.5)
    pred[0, 0] = 0.01
    mask = np.ones_like(gt, dtype=bool)
    assert median_scale(pred, gt, mask) == pytest.approx(2.0)


def test_least_squares_scale_recovers_a_global_factor():
    gt = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
    pred = gt * 2
    mask = np.ones_like(gt, dtype=bool)
    assert least_squares_scale(pred, gt, mask) == pytest.approx(0.5)


def test_disparity_converts_with_calibration():
    disparity = np.full((3, 3), 5.0)
    depth = disparity_to_depth(disparity, fx=100.0, baseline_m=0.1)
    assert depth[0, 0] == pytest.approx(2.0)
    invalid = disparity_to_depth(np.array([[0.0, -1.0], [np.nan, 4.0]]), fx=100.0, baseline_m=0.1)
    assert np.isnan(invalid[0, 0])
    assert np.isnan(invalid[0, 1])
    assert invalid[1, 1] == pytest.approx(2.5)


def test_smooth_plane_has_no_edge_disagreement():
    gt = np.full((24, 24), 2.0)
    valid = np.ones_like(gt, dtype=bool)
    assert edge_f1(gt, gt, valid) == 1.0


def test_blur_removes_occlusion_edge():
    gt = np.ones((32, 40))
    gt[:, 20:] = 4.0
    valid = np.ones_like(gt, dtype=bool)
    blurred = _blur(gt, 8.0)
    sharp = edge_f1(gt, gt, valid)
    soft = edge_f1(blurred, gt, valid)
    assert sharp == pytest.approx(1.0)
    assert soft < sharp - 0.4
