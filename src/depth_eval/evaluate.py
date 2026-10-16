"""Run the paired mono/stereo protocol and write the report."""

from __future__ import annotations

from pathlib import Path

from depth_eval.align import align_scale, least_squares_scale
from depth_eval.heatmap import save_comparison_panel
from depth_eval.masks import eval_mask
from depth_eval.metrics import compute_metrics
from depth_eval.report import aggregate, build_conclusion, write_report
from depth_eval.scene import Scene, list_scene_paths, load_scene
from depth_eval.stereo import disparity_to_depth


def evaluate_scene(scene: Scene, align: str) -> dict:
    gt = scene.gt_depth
    raw_metrics = compute_metrics(scene.mono_depth, gt)
    align_mask = eval_mask(gt, scene.mono_depth)
    aligned, scale = align_scale(scene.mono_depth, gt, align_mask, align)
    aligned_metrics = compute_metrics(aligned, gt)
    ls_scale = least_squares_scale(scene.mono_depth, gt, align_mask)
    stereo_depth = disparity_to_depth(scene.stereo_disparity, scene.fx, scene.baseline)
    stereo_metrics = compute_metrics(stereo_depth, gt)
    return {
        "scene_id": scene.scene_id,
        "failure_mode": scene.failure_mode,
        "mono_scale": scale,
        "least_squares_scale": ls_scale,
        "mono_raw": raw_metrics,
        "mono_aligned": aligned_metrics,
        "stereo": stereo_metrics,
        "_arrays": {
            "rgb": scene.rgb,
            "gt": gt,
            "mono_raw": scene.mono_depth,
            "mono_aligned": aligned,
            "stereo": stereo_depth,
        },
    }


def run_evaluation(data_dir: Path, out_dir: Path, align: str = "median") -> dict:
    paths = list_scene_paths(data_dir)
    if not paths:
        raise FileNotFoundError(f"{data_dir} 里没有场景 npz。先运行 generate。")
    scenes = []
    heatmap_dir = out_dir / "heatmaps"
    for path in paths:
        scene = load_scene(path)
        row = evaluate_scene(scene, align)
        arrays = row.pop("_arrays")
        save_comparison_panel(
            heatmap_dir / f"{scene.scene_id}__mono_raw.png",
            arrays["rgb"],
            arrays["gt"],
            arrays["mono_raw"],
            "MonoRaw",
        )
        save_comparison_panel(
            heatmap_dir / f"{scene.scene_id}__mono_aligned.png",
            arrays["rgb"],
            arrays["gt"],
            arrays["mono_aligned"],
            f"Mono x{row['mono_scale']:.2f}",
        )
        save_comparison_panel(
            heatmap_dir / f"{scene.scene_id}__stereo.png",
            arrays["rgb"],
            arrays["gt"],
            arrays["stereo"],
            "Stereo",
        )
        scenes.append(row)
    by_mode, overall = aggregate(scenes)
    result = {
        "align": align,
        "scene_count": len(scenes),
        "scenes": scenes,
        "by_mode": by_mode,
        "overall": overall,
        "conclusion": build_conclusion(by_mode, overall),
    }
    command = f"python -m depth_eval evaluate --data {data_dir} --out {out_dir} --align {align}"
    write_report(out_dir, result, command)
    return result
