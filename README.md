# 深度估计与立体匹配结果评测

同一批场景、同一套相机对齐和有效区域掩码下，比较单目深度估计与立体匹配。协议关心的是尺度、边缘和失败场景，而不是只报一个全局平均。

仓库里的 12 个场景是可复现的误差夹具，不是训练出来的模型权重。把真实模型导出成相同的 npz 后，评测命令不用改。

## 协议

- 深度单位是米，有效范围 0.1–80。视差单位是像素。
- 预测和 GT 必须已经在同一相机平面、同一分辨率。分辨率不一致会直接报错，评测阶段不会缩放深度图。
- 单目深度有尺度歧义。默认在有效区域用中值比做尺度对齐，也可以改成最小二乘。立体结果先按标定 `Z = fx * baseline / disparity` 转成米，不再做尺度对齐。
- 指标：AbsRel、RMSE、δ < 1.25 / 1.25² / 1.25³、边缘 F1、覆盖率。
- 遮挡边缘用绝对梯度（≥ 0.25 米/像素），边缘 F1 的空间容差是 2 像素。
- 无效预测不参与 AbsRel 平均，但在“相对误差质量”里按每像素 1.0 计入，用来看哪一类失败拉高了整体。
- 失败类型：常规纹理、弱纹理、重复纹理、遮挡边界、远距离、透明/反光。

## 运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m depth_eval generate --out data/scenes
python -m depth_eval evaluate --data data/scenes --out outputs/report
pytest -q
```

报告写在 `outputs/report/`：

- `report.md`：分场景表和结论
- `report.json`：完整指标
- `per_scene.csv`：逐场景
- `heatmaps/`：RGB、GT、预测、相对误差。热力图按单张图的 95 分位着色，颜色不能跨场景比较

对齐方式：

```bash
python -m depth_eval evaluate --data data/scenes --out outputs/ls --align least_squares
```

## 换成真实模型输出

每个场景一个 npz，字段：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `rgb` | H×W×3 uint8 | 可选，热力图用 |
| `gt_depth` | H×W float32 | 米 |
| `mono_depth` | H×W float32 | 单目原始输出，允许差一个全局尺度 |
| `stereo_disparity` | H×W float32 | 像素视差，不要预先转成深度 |
| `transparent_mask` | H×W uint8 | 可选 |
| `fx` `fy` `cx` `cy` `baseline` | 标量 | `baseline` 单位米 |
| `scene_id` `failure_mode` | 字符串 | `failure_mode` 用协议里的英文类型名 |

`data/scenes/manifest.json` 决定评测顺序。没有 manifest 时会读取目录下全部 npz。

## 这套夹具在说明什么

夹具把误差模式固定下来，方便在没有 GPU 的环境里检查协议本身：

- 常规纹理：立体可以直接给出度量深度，单目的主要缺口是绝对尺度。
- 弱纹理、重复纹理、远距离：立体更容易失配；远距离还叠加了整像素量化。
- 遮挡边界：单目被模糊后边缘 F1 下降，立体边缘更完整。
- 透明/反光：两侧都会看穿或看反，不能只看平均 AbsRel。
