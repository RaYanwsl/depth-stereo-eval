"""Thresholds shared by the evaluation protocol.

Depth is in meters. Disparity is in pixels. Edges use an absolute gradient
so a smooth slanted plane is not marked as an occlusion boundary.
"""

DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 80.0
EDGE_GRAD_M_PER_PX = 0.25
EDGE_TOLERANCE_PX = 2
# Pixels with no finite positive prediction still contribute this relative
# error when measuring which failure mode owns the error mass.
INVALID_REL_ERROR = 1.0

MODE_ORDER = (
    "normal",
    "weak_texture",
    "repeated_texture",
    "occlusion_boundary",
    "far",
    "transparent",
)

MODE_LABELS = {
    "normal": "常规纹理",
    "weak_texture": "弱纹理",
    "repeated_texture": "重复纹理",
    "occlusion_boundary": "遮挡边界",
    "far": "远距离",
    "transparent": "透明/反光",
}
