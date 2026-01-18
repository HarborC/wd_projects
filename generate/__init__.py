"""Generate module for novel view synthesis.

Includes SEVA (Stable Virtual Camera) inference and camera geometry tools.
"""

from generate.infer import run_seva_inference, run_img2img
from generate.geometry.camera_utils import (
    get_default_intrinsics,
    get_lookat,
    generate_interpolated_path,
    generate_spiral_path,
    viewmatrix,
    normalize,
    rt_to_mat4,
)

__all__ = [
    # Inference
    "run_seva_inference",
    "run_img2img",
    # Camera geometry
    "get_default_intrinsics",
    "get_lookat",
    "generate_interpolated_path",
    "generate_spiral_path",
    "viewmatrix",
    "normalize",
    "rt_to_mat4",
]
