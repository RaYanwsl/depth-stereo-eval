"""Scene containers and npz IO. Predictions stay in the GT camera frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Scene:
    scene_id: str
    failure_mode: str
    rgb: np.ndarray
    gt_depth: np.ndarray
    mono_depth: np.ndarray
    stereo_disparity: np.ndarray
    transparent_mask: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            rgb=self.rgb,
            gt_depth=self.gt_depth.astype(np.float32),
            mono_depth=self.mono_depth.astype(np.float32),
            stereo_disparity=self.stereo_disparity.astype(np.float32),
            transparent_mask=self.transparent_mask.astype(np.uint8),
            fx=np.float32(self.fx),
            fy=np.float32(self.fy),
            cx=np.float32(self.cx),
            cy=np.float32(self.cy),
            baseline=np.float32(self.baseline),
            scene_id=np.array(self.scene_id),
            failure_mode=np.array(self.failure_mode),
        )


def _text(value: np.ndarray) -> str:
    if isinstance(value, np.ndarray):
        return str(value.item())
    return str(value)


def load_scene(path: Path) -> Scene:
    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        required = {
            "gt_depth",
            "mono_depth",
            "stereo_disparity",
            "fx",
            "fy",
            "cx",
            "cy",
            "baseline",
            "scene_id",
            "failure_mode",
        }
        missing = sorted(required - files)
        if missing:
            raise ValueError(f"{path.name} 缺少字段: {', '.join(missing)}")
        gt = data["gt_depth"].astype(np.float64)
        if "rgb" in files:
            rgb = data["rgb"]
        else:
            rgb = np.full((*gt.shape, 3), 128, dtype=np.uint8)
        if "transparent_mask" in files:
            transparent = data["transparent_mask"].astype(bool)
        else:
            transparent = np.zeros(gt.shape, dtype=bool)
        return Scene(
            scene_id=_text(data["scene_id"]),
            failure_mode=_text(data["failure_mode"]),
            rgb=rgb,
            gt_depth=gt,
            mono_depth=data["mono_depth"].astype(np.float64),
            stereo_disparity=data["stereo_disparity"].astype(np.float64),
            transparent_mask=transparent,
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            baseline=float(data["baseline"]),
        )


def list_scene_paths(data_dir: Path) -> list[Path]:
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [data_dir / item["file"] for item in manifest["scenes"]]
    return sorted(data_dir.glob("*.npz"))
