"""The fixture set has to tell the protocol's story, not just run."""

from __future__ import annotations

from pathlib import Path

from depth_eval.evaluate import run_evaluation
from depth_eval.generate import build_scenes, generate_dataset

ROOT = Path(__file__).resolve().parents[1]


def _index(result):
    return {item["failure_mode"]: item for item in result["by_mode"]}


def test_failure_modes_separate_scale_edges_and_stereo(tmp_path):
    data = tmp_path / "scenes"
    out = tmp_path / "out"
    generate_dataset(data)
    result = run_evaluation(data, out, align="median")
    modes = _index(result)

    normal = modes["normal"]
    assert normal["mono_raw"]["abs_rel"] > normal["mono_aligned"]["abs_rel"] + 0.2
    assert normal["stereo"]["abs_rel"] < 0.08

    weak = modes["weak_texture"]
    assert weak["stereo"]["abs_rel"] > weak["mono_aligned"]["abs_rel"] + 0.15

    repeated = modes["repeated_texture"]
    assert repeated["stereo"]["abs_rel"] > repeated["mono_aligned"]["abs_rel"] + 0.15

    far = modes["far"]
    assert far["stereo"]["abs_rel"] > far["mono_aligned"]["abs_rel"] + 0.1

    glass = modes["transparent"]
    assert glass["mono_aligned"]["abs_rel"] > 0.12
    assert glass["stereo"]["abs_rel"] > glass["mono_aligned"]["abs_rel"]

    edge = modes["occlusion_boundary"]
    assert edge["stereo"]["edge_f1"] > edge["mono_aligned"]["edge_f1"] + 0.25

    shares = [item["stereo_error_share"] for item in result["by_mode"]]
    assert abs(sum(shares) - 1.0) < 1e-6
    assert (out / "report.md").exists()
    assert (out / "heatmaps" / "office_desk__stereo.png").exists()
    text = "\n".join(result["conclusion"])
    assert "平均" in text
    assert "弱纹理" in text


def test_generator_is_deterministic(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    generate_dataset(first)
    generate_dataset(second)
    import numpy as np

    for left in sorted(first.glob("*.npz")):
        a = np.load(left)
        b = np.load(second / left.name)
        assert np.array_equal(a["gt_depth"], b["gt_depth"])
        assert np.array_equal(a["mono_depth"], b["mono_depth"])
        assert np.array_equal(a["stereo_disparity"], b["stereo_disparity"])


def test_weak_texture_is_visually_flatter_than_normal():
    scenes = {scene.scene_id: scene for scene in build_scenes()}
    weak = scenes["white_wall"].rgb.astype("float32").std()
    normal = scenes["office_desk"].rgb.astype("float32").std()
    assert weak < normal * 0.25


def test_committed_fixtures_exist():
    files = list((ROOT / "data" / "scenes").glob("*.npz"))
    assert len(files) == 12
    assert (ROOT / "data" / "scenes" / "manifest.json").exists()
