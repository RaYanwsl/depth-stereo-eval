"""Error heatmaps and side-by-side panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from depth_eval.masks import eval_mask, gt_valid_mask


def _depth_color(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    span = max(vmax - vmin, 1e-6)
    t = np.clip((depth - vmin) / span, 0, 1)
    near = np.array([255, 190, 60], dtype=np.float64)
    far = np.array([25, 55, 130], dtype=np.float64)
    rgb = near * (1 - t[..., None]) + far * t[..., None]
    rgb[~np.isfinite(depth)] = 20
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _error_color(error: np.ndarray, vmax: float) -> np.ndarray:
    t = np.clip(error / max(vmax, 1e-6), 0, 1)
    # White (low) to dark red (high).
    red = 255 * np.ones_like(t)
    green = 255 * (1 - t)
    blue = 255 * (1 - t) ** 2
    rgb = np.stack([red, green, blue], axis=-1)
    rgb[~np.isfinite(error)] = 45
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _panel(images: list[np.ndarray], labels: list[str]) -> Image.Image:
    strips: list[Image.Image] = []
    for array, label in zip(images, labels):
        image = Image.fromarray(np.ascontiguousarray(array), mode="RGB")
        canvas = Image.new("RGB", (image.width, image.height + 14), (255, 255, 255))
        canvas.paste(image, (0, 14))
        draw = ImageDraw.Draw(canvas)
        draw.text((2, 1), label, fill=(0, 0, 0))
        strips.append(canvas)
    width = sum(item.width for item in strips)
    height = strips[0].height
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for item in strips:
        sheet.paste(item, (x, 0))
        x += item.width
    return sheet


def save_comparison_panel(
    path: Path,
    rgb: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    title: str,
) -> None:
    region = gt_valid_mask(gt)
    if int(region.sum()) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(gt[region], 1))
        vmax = float(np.percentile(gt[region], 99))
    error = np.abs(pred - gt)
    mask = eval_mask(gt, pred)
    error_view = np.where(mask, error, np.nan)
    finite = error_view[np.isfinite(error_view)]
    err_vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
    sheet = _panel(
        [
            rgb,
            _depth_color(gt, vmin, vmax),
            _depth_color(pred, vmin, vmax),
            _error_color(error_view, err_vmax),
        ],
        ["RGB", "GT", title, f"AbsErr p95={err_vmax:.2f}m"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
