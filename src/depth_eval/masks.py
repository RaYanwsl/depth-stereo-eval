"""Valid-region masks. Evaluation never resizes a depth map to invent overlap."""

from __future__ import annotations

import numpy as np

from depth_eval.constants import DEPTH_MAX_M, DEPTH_MIN_M


def assert_same_frame(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError(
            f"深度图必须是二维数组，收到 pred {pred.shape}、gt {gt.shape}。"
            "请先重投影到 GT 相机，再进入评测。"
        )
    if pred.shape != gt.shape:
        raise ValueError(
            f"预测分辨率 {pred.shape} 与 GT {gt.shape} 不一致。"
            "评测阶段禁止缩放深度图，请先对齐到同一相机平面。"
        )


def gt_valid_mask(gt: np.ndarray) -> np.ndarray:
    return np.isfinite(gt) & (gt >= DEPTH_MIN_M) & (gt <= DEPTH_MAX_M)


def pred_valid_mask(pred: np.ndarray) -> np.ndarray:
    return np.isfinite(pred) & (pred > 0)


def eval_mask(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    assert_same_frame(pred, gt)
    return gt_valid_mask(gt) & pred_valid_mask(pred)
