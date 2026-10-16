"""Deterministic benchmark fixtures.

The stored mono/stereo outputs are not trained models. They are fixed error
patterns so the protocol can be run without a GPU and still separates
scale error, metric stereo, and the failure modes named in the report.
Replace the npz files with real model dumps that use the same schema and
the same command evaluates them.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from depth_eval.scene import Scene

HEIGHT = 96
WIDTH = 128
FX = 100.0
FY = 100.0
CX = (WIDTH - 1) / 2
CY = (HEIGHT - 1) / 2
BASELINE = 0.10


def _rng(scene_id: str) -> np.random.Generator:
    seed = zlib.crc32(scene_id.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(sigma * 3)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.astype(np.float64)
    kernel = _gaussian_kernel(sigma)
    radius = len(kernel) // 2

    def along(values: np.ndarray, axis: int) -> np.ndarray:
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(values, pad, mode="reflect")
        out = np.zeros_like(values, dtype=np.float64)
        for index, weight in enumerate(kernel):
            if axis == 0:
                out += weight * padded[index : index + values.shape[0], :]
            else:
                out += weight * padded[:, index : index + values.shape[1]]
        return out

    blurred = along(image.astype(np.float64), 0)
    return along(blurred, 1)


def _dilate_max(values: np.ndarray, radius: int) -> np.ndarray:
    out = values.astype(np.float64).copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="edge")
        neighbors = [
            padded[0:-2, 0:-2],
            padded[0:-2, 1:-1],
            padded[0:-2, 2:],
            padded[1:-1, 0:-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, 0:-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]
        out = np.maximum.reduce(neighbors)
    return out


def _ramp(z00: float, z01: float, z10: float, z11: float) -> np.ndarray:
    y = np.linspace(0, 1, HEIGHT)[:, None]
    x = np.linspace(0, 1, WIDTH)[None, :]
    top = z00 * (1 - x) + z01 * x
    bottom = z10 * (1 - x) + z11 * x
    return top * (1 - y) + bottom * y


def _value_noise(rng: np.random.Generator, cell: int) -> np.ndarray:
    grid_h = HEIGHT // cell + 2
    grid_w = WIDTH // cell + 2
    grid = rng.random((grid_h, grid_w))
    y = np.arange(HEIGHT) / cell
    x = np.arange(WIDTH) / cell
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    ty = (y - y0)[:, None]
    tx = (x - x0)[None, :]
    ty = ty * ty * (3 - 2 * ty)
    tx = tx * tx * (3 - 2 * tx)
    v00 = grid[y0[:, None], x0[None, :]]
    v01 = grid[y0[:, None], x0[None, :] + 1]
    v10 = grid[(y0 + 1)[:, None], x0[None, :]]
    v11 = grid[(y0 + 1)[:, None], (x0 + 1)[None, :]]
    return v00 * (1 - tx) * (1 - ty) + v01 * tx * (1 - ty) + v10 * (1 - tx) * ty + v11 * tx * ty


def _rich_rgb(rng: np.random.Generator) -> np.ndarray:
    channels = []
    for cell in (6, 10, 14):
        channels.append(_value_noise(rng, cell))
    rgb = np.stack(channels, axis=-1)
    rgb = 30 + 200 * rgb
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _weak_rgb(rng: np.random.Generator) -> np.ndarray:
    base = 214 + rng.normal(0, 1.6, size=(HEIGHT, WIDTH, 3))
    return np.clip(base, 0, 255).astype(np.uint8)


def _stripe_rgb(period: int, rng: np.random.Generator) -> np.ndarray:
    band = (np.arange(WIDTH) // period) % 2
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    rgb[..., 0] = 30 + 170 * band
    rgb[..., 1] = 50 + 30 * band
    rgb[..., 2] = 80 + 20 * (1 - band)
    rgb += rng.normal(0, 2.0, size=rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _mono_from_geometry(
    geometry: np.ndarray,
    rng: np.random.Generator,
    scale: float,
    noise: float,
    blur: float,
) -> np.ndarray:
    blurred = _blur(geometry, blur)
    noisy = blurred * (1 + rng.normal(0, noise, size=blurred.shape))
    return np.clip(noisy * scale, 1e-3, None)


def _disparity(depth: np.ndarray) -> np.ndarray:
    return (FX * BASELINE) / depth


def _scene(
    scene_id: str,
    failure_mode: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    mono: np.ndarray,
    disparity: np.ndarray,
    transparent: np.ndarray | None = None,
) -> Scene:
    if transparent is None:
        transparent = np.zeros(gt.shape, dtype=bool)
    return Scene(
        scene_id=scene_id,
        failure_mode=failure_mode,
        rgb=rgb,
        gt_depth=gt.astype(np.float32),
        mono_depth=mono.astype(np.float32),
        stereo_disparity=disparity.astype(np.float32),
        transparent_mask=transparent.astype(bool),
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        baseline=BASELINE,
    )


def _plane_scene(
    scene_id: str,
    failure_mode: str,
    corners: tuple[float, float, float, float],
    texture: str,
    mono_scale: float,
    mono_noise: float,
    mono_blur: float,
    stereo: str,
    period: int = 6,
    stereo_sigma: float = 0.1,
    stereo_bias: float = 0.0,
) -> Scene:
    rng = _rng(scene_id)
    gt = _ramp(*corners)
    if texture == "rich":
        rgb = _rich_rgb(rng)
    elif texture == "weak":
        rgb = _weak_rgb(rng)
    elif texture == "stripe":
        rgb = _stripe_rgb(period, rng)
    else:
        raise ValueError(texture)
    mono = _mono_from_geometry(gt, rng, mono_scale, mono_noise, mono_blur)
    disparity = _disparity(gt)
    if stereo == "noise":
        disparity = disparity + rng.normal(0, stereo_sigma, size=disparity.shape)
    elif stereo == "bias":
        disparity = disparity + stereo_bias + rng.normal(0, 0.08, size=disparity.shape)
    elif stereo == "quantize":
        # Integer-pixel matching. Subpixel disparity at long range collapses,
        # and a rounded 0 is stored as the smallest positive bin (1 px).
        disparity = np.maximum(np.rint(disparity), 1.0)
    else:
        raise ValueError(stereo)
    return _scene(scene_id, failure_mode, rgb, gt, mono, disparity)


def _occlusion_scene(
    scene_id: str,
    box: tuple[int, int, int, int],
    mono_scale: float,
    mono_blur: float,
    fatten: int,
) -> Scene:
    rng = _rng(scene_id)
    background = _ramp(3.8, 4.2, 4.3, 4.8)
    foreground = _ramp(1.35, 1.5, 1.45, 1.65)
    gt = background.copy()
    y0, y1, x0, x1 = box
    gt[y0:y1, x0:x1] = foreground[y0:y1, x0:x1]
    rgb = _rich_rgb(rng)
    rgb[y0:y1, x0:x1] = np.clip(rgb[y0:y1, x0:x1].astype(np.int16) - 40, 0, 255).astype(np.uint8)
    mono = _mono_from_geometry(gt, rng, mono_scale, 0.015, mono_blur)
    disparity = _dilate_max(_disparity(gt), fatten)
    disparity = disparity + rng.normal(0, 0.08, size=disparity.shape)
    return _scene(scene_id, "occlusion_boundary", rgb, gt, mono, disparity)


def _glass_scene(
    scene_id: str,
    window: tuple[int, int, int, int],
    gt_z: float,
    bg_z: float,
    mono_scale: float,
) -> Scene:
    rng = _rng(scene_id)
    gt = _ramp(gt_z, gt_z + 0.15, gt_z + 0.05, gt_z + 0.2)
    y0, y1, x0, x1 = window
    transparent = np.zeros(gt.shape, dtype=bool)
    transparent[y0:y1, x0:x1] = True
    rgb = _rich_rgb(rng)
    rgb[transparent] = np.array([210, 225, 235], dtype=np.uint8)
    geometry = gt.copy()
    geometry[transparent] = 0.5 * gt[transparent] + 0.5 * bg_z
    mono = _mono_from_geometry(geometry, rng, mono_scale, 0.015, 0.2)
    disparity = _disparity(gt)
    disparity = disparity + rng.normal(0, 0.08, size=disparity.shape)
    disparity[transparent] = (FX * BASELINE) / bg_z
    return _scene(scene_id, "transparent", rgb, gt, mono, disparity, transparent)


def build_scenes() -> list[Scene]:
    return [
        _plane_scene("office_desk", "normal", (1.6, 2.1, 2.0, 2.7), "rich", 1.42, 0.015, 0.3, "noise", stereo_sigma=0.12),
        _plane_scene("hallway_slant", "normal", (1.8, 2.3, 2.5, 3.3), "rich", 1.55, 0.018, 0.3, "noise", stereo_sigma=0.14),
        _plane_scene("white_wall", "weak_texture", (1.9, 2.1, 2.2, 2.5), "weak", 1.38, 0.02, 0.2, "noise", stereo_sigma=1.35),
        _plane_scene("foggy_corridor", "weak_texture", (2.1, 2.5, 2.8, 3.4), "weak", 1.48, 0.02, 0.3, "noise", stereo_sigma=1.55),
        _plane_scene(
            "brick_repeat",
            "repeated_texture",
            (1.7, 2.0, 2.1, 2.6),
            "stripe",
            1.36,
            0.015,
            0.2,
            "bias",
            period=6,
            stereo_bias=3.6,
        ),
        _plane_scene(
            "fence_repeat",
            "repeated_texture",
            (1.5, 1.8, 1.9, 2.4),
            "stripe",
            1.50,
            0.015,
            0.2,
            "bias",
            period=8,
            stereo_bias=4.0,
        ),
        _occlusion_scene("doorway_edge", (0, HEIGHT, 30, 78), 1.40, 8.0, 2),
        _occlusion_scene("box_occluder", (18, 80, 40, 100), 1.33, 7.5, 2),
        _plane_scene("distant_building", "far", (14, 16, 18, 22), "rich", 1.46, 0.02, 0.3, "quantize"),
        _plane_scene("skyline_far", "far", (12, 15, 17, 21), "rich", 1.52, 0.02, 0.3, "quantize"),
        _glass_scene("glass_window", (16, 64, 24, 104), 2.2, 7.5, 1.40),
        _glass_scene("mirror_cabinet", (20, 72, 36, 108), 1.8, 6.8, 1.47),
    ]


def generate_dataset(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.npz"):
        old.unlink()
    scenes = build_scenes()
    manifest_scenes = []
    paths: list[Path] = []
    for scene in scenes:
        filename = f"{scene.scene_id}.npz"
        path = out_dir / filename
        scene.save(path)
        paths.append(path)
        manifest_scenes.append(
            {
                "scene_id": scene.scene_id,
                "failure_mode": scene.failure_mode,
                "file": filename,
            }
        )
    manifest = {
        "image_size": [HEIGHT, WIDTH],
        "depth_unit": "meter",
        "disparity_unit": "pixel",
        "intrinsics": {
            "fx": FX,
            "fy": FY,
            "cx": CX,
            "cy": CY,
            "baseline_m": BASELINE,
        },
        "note": "单目与立体预测是可复现的误差夹具，不是训练出来的模型权重。",
        "scenes": manifest_scenes,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths
