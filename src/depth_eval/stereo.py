"""Convert stereo disparity into metric depth with the GT camera calibration."""

from __future__ import annotations

import numpy as np


def disparity_to_depth(disparity: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
    """Z = fx * baseline / disparity.

    Non-positive or non-finite disparities stay NaN and lower coverage.
    They are not clamped into a fake depth before this conversion.
    """

    if fx <= 0 or baseline_m <= 0:
        raise ValueError(f"标定参数无效: fx={fx}, baseline={baseline_m}")
    disparity = np.asarray(disparity, dtype=np.float64)
    depth = np.full(disparity.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(disparity) & (disparity > 0)
    depth[valid] = (float(fx) * float(baseline_m)) / disparity[valid]
    return depth
