"""Render module for 3D Gaussian Splatting."""

from milkianimator.render.train import train_gaussians
from milkianimator.render.render import (
    render_interpolated,
    render_set,
    save_interpolated_poses,
)

__all__ = [
    "train_gaussians",
    "render_interpolated",
    "render_set",
    "save_interpolated_poses",
]
