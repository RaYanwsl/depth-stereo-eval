"""Depth-edge agreement. Threshold is meters per pixel, not a percentile."""

from __future__ import annotations

import numpy as np

from depth_eval.constants import EDGE_GRAD_M_PER_PX, EDGE_TOLERANCE_PX


def depth_gradient(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    filled = np.where(finite, depth, 0.0)
    gy, gx = np.gradient(filled.astype(np.float64))
    magnitude = np.hypot(gx, gy)
    magnitude[~finite] = 0.0
    return magnitude


def occlusion_edge_mask(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return (depth_gradient(depth) >= EDGE_GRAD_M_PER_PX) & valid


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    height, width = mask.shape
    shifts = [
        padded[radius + dy : radius + dy + height, radius + dx : radius + dx + width]
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    return np.logical_or.reduce(shifts)


def edge_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    tolerance_px: int = EDGE_TOLERANCE_PX,
) -> float:
    """F1 between occlusion edges, with a small spatial tolerance.

    When neither map has an edge inside the valid region, the score is 1:
    there is no boundary disagreement to penalize.
    """

    pred_edge = occlusion_edge_mask(pred, valid)
    gt_edge = occlusion_edge_mask(gt, valid)
    if not np.any(pred_edge) and not np.any(gt_edge):
        return 1.0
    gt_tolerant = _dilate(gt_edge, tolerance_px)
    pred_tolerant = _dilate(pred_edge, tolerance_px)
    true_positive = np.logical_and(pred_edge, gt_tolerant).sum()
    false_positive = np.logical_and(pred_edge, ~gt_tolerant).sum()
    false_negative = np.logical_and(gt_edge, ~pred_tolerant).sum()
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
