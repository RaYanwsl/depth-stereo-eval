"""Scale alignment for scale-ambiguous monocular depth.

Stereo depth is already metric and must not be passed through this step.
Median scale is the default because a few broken pixels cannot move it.
Least squares is available when the prediction has no heavy outliers.
"""

from __future__ import annotations

import numpy as np


def median_scale(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 10:
        raise ValueError("有效像素少于 10，无法估计单目尺度。")
    ratio = gt[mask] / pred[mask]
    scale = float(np.median(ratio))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"中值尺度无效: {scale}")
    return scale


def least_squares_scale(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 10:
        raise ValueError("有效像素少于 10，无法估计单目尺度。")
    prediction = pred[mask].astype(np.float64)
    target = gt[mask].astype(np.float64)
    denom = float(np.dot(prediction, prediction))
    if denom <= 0:
        raise ValueError("预测深度能量为 0，无法做最小二乘尺度对齐。")
    scale = float(np.dot(prediction, target) / denom)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"最小二乘尺度无效: {scale}")
    return scale


def align_scale(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    method: str,
) -> tuple[np.ndarray, float]:
    if method == "median":
        scale = median_scale(pred, gt, mask)
    elif method == "least_squares":
        scale = least_squares_scale(pred, gt, mask)
    else:
        raise ValueError(f"未知对齐方式: {method}")
    return pred.astype(np.float64) * scale, scale
