"""AbsRel, RMSE, delta thresholds, edge F1, and relative-error mass."""

from __future__ import annotations

import numpy as np

from depth_eval.constants import INVALID_REL_ERROR
from depth_eval.edges import edge_f1
from depth_eval.masks import eval_mask, gt_valid_mask, pred_valid_mask


def _mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def relative_error_mass(pred: np.ndarray, gt: np.ndarray, gt_region: np.ndarray) -> float:
    """Sum of |pred-gt|/gt. Missing predictions count as INVALID_REL_ERROR each."""

    mass = np.zeros(gt.shape, dtype=np.float64)
    covered = gt_region & pred_valid_mask(pred)
    missing = gt_region & ~pred_valid_mask(pred)
    mass[covered] = np.abs(pred[covered] - gt[covered]) / gt[covered]
    mass[missing] = INVALID_REL_ERROR
    return float(mass.sum())


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int | None]:
    region = gt_valid_mask(gt)
    mask = eval_mask(gt, pred)
    gt_count = int(region.sum())
    covered = int(mask.sum())
    coverage = float(covered / gt_count) if gt_count else 0.0
    if covered == 0:
        return {
            "abs_rel": None,
            "rmse": None,
            "delta_1": None,
            "delta_2": None,
            "delta_3": None,
            "edge_f1": None,
            "coverage": coverage,
            "rel_error_sum": relative_error_mass(pred, gt, region),
            "pixel_count": covered,
            "gt_pixel_count": gt_count,
        }

    difference = pred[mask].astype(np.float64) - gt[mask].astype(np.float64)
    abs_rel = _mean(np.abs(difference) / gt[mask])
    rmse = float(np.sqrt(np.mean(difference**2)))
    ratio = np.maximum(pred[mask] / gt[mask], gt[mask] / pred[mask])
    return {
        "abs_rel": abs_rel,
        "rmse": rmse,
        "delta_1": float(np.mean(ratio < 1.25)),
        "delta_2": float(np.mean(ratio < 1.25**2)),
        "delta_3": float(np.mean(ratio < 1.25**3)),
        "edge_f1": edge_f1(pred, gt, mask),
        "coverage": coverage,
        "rel_error_sum": relative_error_mass(pred, gt, region),
        "pixel_count": covered,
        "gt_pixel_count": gt_count,
    }
