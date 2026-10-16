"""Markdown, JSON, and CSV reports. Numbers in the prose come from the metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from depth_eval.constants import (
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    EDGE_GRAD_M_PER_PX,
    EDGE_TOLERANCE_PX,
    INVALID_REL_ERROR,
    MODE_LABELS,
    MODE_ORDER,
)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(value)


def _fmt(value: object, digits: int = 3) -> str:
    if not _finite(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _pct(value: object) -> str:
    if not _finite(value):
        return "—"
    return f"{100 * float(value):.1f}%"


def _label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def _mean_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = rows[0].keys()
    averaged: dict[str, float | None] = {}
    for key in keys:
        values = [row[key] for row in rows if _finite(row[key])]
        averaged[key] = float(np.mean(values)) if values else None
    return averaged


def _ordered_modes(scenes: list[dict]) -> list[str]:
    present = {scene["failure_mode"] for scene in scenes}
    ordered = [mode for mode in MODE_ORDER if mode in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def aggregate(scenes: list[dict]) -> tuple[list[dict], dict]:
    modes = _ordered_modes(scenes)
    total_stereo = sum(scene["stereo"]["rel_error_sum"] for scene in scenes) or 1.0
    total_mono = sum(scene["mono_aligned"]["rel_error_sum"] for scene in scenes) or 1.0
    by_mode = []
    for mode in modes:
        group = [scene for scene in scenes if scene["failure_mode"] == mode]
        stereo_mass = sum(scene["stereo"]["rel_error_sum"] for scene in group)
        mono_mass = sum(scene["mono_aligned"]["rel_error_sum"] for scene in group)
        by_mode.append(
            {
                "failure_mode": mode,
                "label": _label(mode),
                "count": len(group),
                "mono_raw": _mean_metrics([scene["mono_raw"] for scene in group]),
                "mono_aligned": _mean_metrics([scene["mono_aligned"] for scene in group]),
                "stereo": _mean_metrics([scene["stereo"] for scene in group]),
                "stereo_error_share": stereo_mass / total_stereo,
                "mono_aligned_error_share": mono_mass / total_mono,
            }
        )
    overall = {
        "count": len(scenes),
        "mono_raw": _mean_metrics([scene["mono_raw"] for scene in scenes]),
        "mono_aligned": _mean_metrics([scene["mono_aligned"] for scene in scenes]),
        "stereo": _mean_metrics([scene["stereo"] for scene in scenes]),
    }
    return by_mode, overall


def build_conclusion(by_mode: list[dict], overall: dict) -> list[str]:
    indexed = {item["failure_mode"]: item for item in by_mode}
    lines = [
        (
            "全部场景上，单目中值尺度对齐把 AbsRel 从 "
            f"{overall['mono_raw']['abs_rel']:.3f} 降到 {overall['mono_aligned']['abs_rel']:.3f}。"
            "对齐只使用有效区域内 GT 与预测的中值比，没有借用立体基线。"
        )
    ]
    if "normal" in indexed:
        normal = indexed["normal"]
        lines.append(
            "常规纹理场景里，立体匹配不经尺度校正的 AbsRel 为 "
            f"{normal['stereo']['abs_rel']:.3f}，对齐后的单目为 "
            f"{normal['mono_aligned']['abs_rel']:.3f}。"
            "有纹理时立体可以直接给出度量深度；单目的主要缺口是绝对尺度。"
        )
    for item in by_mode:
        if item["failure_mode"] == "normal":
            continue
        mono = item["mono_aligned"]["abs_rel"]
        stereo = item["stereo"]["abs_rel"]
        name = item["label"]
        both_bad = mono >= 0.15 and stereo >= 0.15
        if both_bad and abs(stereo - mono) <= 0.08:
            relation = "两侧都偏高，而且接近。"
        elif stereo > mono + 0.08 and both_bad:
            relation = "两侧都不可靠，立体偏差更大。"
        elif stereo > mono + 0.08:
            relation = "这一类里立体更不稳定。"
        elif mono > stereo + 0.08 and both_bad:
            relation = "两侧都不可靠，对齐后的单目偏差更大。"
        elif mono > stereo + 0.08:
            relation = "这一类里立体更稳。"
        else:
            relation = "两侧误差接近。"
        lines.append(
            f"{name}：对齐后单目 AbsRel {_fmt(mono)}，立体 AbsRel {_fmt(stereo)}。{relation}"
        )
        if item["failure_mode"] == "occlusion_boundary":
            lines.append(
                f"遮挡边界的边缘 F1：立体 {_fmt(item['stereo']['edge_f1'])}，"
                f"对齐后单目 {_fmt(item['mono_aligned']['edge_f1'])}。"
                "边缘是否完整要单独看，不能从 AbsRel 里读出来。"
            )
    counted = sum(item["count"] for item in by_mode) or 1
    top = max(by_mode, key=lambda item: item["stereo_error_share"])
    scene_share = top["count"] / counted
    if top["stereo_error_share"] > scene_share + 0.08:
        lines.append(
            f"{top['label']}占场景数的 {scene_share:.0%}，"
            f"却贡献了立体相对误差质量的 {top['stereo_error_share']:.0%}。"
            "只报全局平均会把这类失败抹平。"
        )
    worst = max(by_mode, key=lambda item: item["stereo"]["abs_rel"] or 0)
    if (worst["stereo"]["abs_rel"] or 0) > (overall["stereo"]["abs_rel"] or 0) + 0.12:
        lines.append(
            f"立体 AbsRel 的总平均是 {_fmt(overall['stereo']['abs_rel'])}，"
            f"但{worst['label']}单项是 {_fmt(worst['stereo']['abs_rel'])}。"
        )
    return lines


def _metric_row(name: str, count: int, mono_raw: dict, mono_aligned: dict, stereo: dict, share: str) -> str:
    return (
        f"| {name} | {count} | {_fmt(mono_raw['abs_rel'])} | {_fmt(mono_aligned['abs_rel'])} | "
        f"{_fmt(stereo['abs_rel'])} | {_fmt(stereo['rmse'])} | {_pct(stereo['delta_1'])} | "
        f"{_fmt(mono_aligned['edge_f1'])} | {_fmt(stereo['edge_f1'])} | {share} |"
    )


def render_markdown(result: dict, command: str) -> str:
    header = [
        "# 深度估计与立体匹配评测报告",
        "",
        f"复现命令：`{command}`",
        "",
        "## 协议",
        "",
        f"- 深度单位为米，有效范围 [{DEPTH_MIN_M}, {DEPTH_MAX_M}]。视差单位为像素。",
        "- 预测与 GT 必须已经在同一相机平面、同一分辨率。分辨率不一致会直接报错，不会在评测里缩放。",
        "- 单目先在有效区域做尺度对齐（默认中值比），再与 GT 比较。立体用 Z = fx * baseline / disparity 转成米，不再做尺度对齐。",
        f"- 遮挡边缘：深度梯度 ≥ {EDGE_GRAD_M_PER_PX} 米/像素。边缘 F1 的空间容差为 {EDGE_TOLERANCE_PX} 像素。",
        "- AbsRel、RMSE、δ 只在 GT 与预测都有效的像素上计算，并同时给出覆盖率。",
        f"- 相对误差质量把无效预测记为每像素 {INVALID_REL_ERROR:.0f}。热力图按单张图的 95 分位归一化，颜色不能跨场景比较。",
        "",
        "## 分场景",
        "",
        "| 类型 | 样本 | 单目原始 AbsRel | 单目对齐 AbsRel | 立体 AbsRel | 立体 RMSE | 立体 δ<1.25 | 单目边缘 F1 | 立体边缘 F1 | 立体误差占比 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result["by_mode"]:
        header.append(
            _metric_row(
                item["label"],
                item["count"],
                item["mono_raw"],
                item["mono_aligned"],
                item["stereo"],
                _pct(item["stereo_error_share"]),
            )
        )
    overall = result["overall"]
    header.append(
        _metric_row(
            "全部场景",
            overall["count"],
            overall["mono_raw"],
            overall["mono_aligned"],
            overall["stereo"],
            "100%",
        )
    )
    header.extend(["", "## 结论", ""])
    header.extend(f"- {line}" for line in result["conclusion"])
    header.extend(["", "## 逐场景", ""])
    header.append("| 场景 | 类型 | 单目尺度 | 单目原始 AbsRel | 单目对齐 AbsRel | 立体 AbsRel | 立体覆盖率 |")
    header.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for scene in result["scenes"]:
        header.append(
            "| {scene_id} | {label} | {scale} | {raw} | {aligned} | {stereo} | {coverage} |".format(
                scene_id=scene["scene_id"],
                label=_label(scene["failure_mode"]),
                scale=_fmt(scene["mono_scale"], 3),
                raw=_fmt(scene["mono_raw"]["abs_rel"]),
                aligned=_fmt(scene["mono_aligned"]["abs_rel"]),
                stereo=_fmt(scene["stereo"]["abs_rel"]),
                coverage=_pct(scene["stereo"]["coverage"]),
            )
        )
    header.append("")
    return "\n".join(header)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return round(number, 6)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def write_csv(path: Path, result: dict) -> None:
    fields = [
        "scene_id",
        "failure_mode",
        "mono_scale",
        "mono_raw_abs_rel",
        "mono_aligned_abs_rel",
        "mono_aligned_rmse",
        "mono_aligned_edge_f1",
        "stereo_abs_rel",
        "stereo_rmse",
        "stereo_delta_1",
        "stereo_edge_f1",
        "stereo_coverage",
        "stereo_rel_error_sum",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scene in result["scenes"]:
            writer.writerow(
                {
                    "scene_id": scene["scene_id"],
                    "failure_mode": scene["failure_mode"],
                    "mono_scale": scene["mono_scale"],
                    "mono_raw_abs_rel": scene["mono_raw"]["abs_rel"],
                    "mono_aligned_abs_rel": scene["mono_aligned"]["abs_rel"],
                    "mono_aligned_rmse": scene["mono_aligned"]["rmse"],
                    "mono_aligned_edge_f1": scene["mono_aligned"]["edge_f1"],
                    "stereo_abs_rel": scene["stereo"]["abs_rel"],
                    "stereo_rmse": scene["stereo"]["rmse"],
                    "stereo_delta_1": scene["stereo"]["delta_1"],
                    "stereo_edge_f1": scene["stereo"]["edge_f1"],
                    "stereo_coverage": scene["stereo"]["coverage"],
                    "stereo_rel_error_sum": scene["stereo"]["rel_error_sum"],
                }
            )


def write_report(out_dir: Path, result: dict, command: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(result)
    (out_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(render_markdown(result, command), encoding="utf-8")
    write_csv(out_dir / "per_scene.csv", result)
