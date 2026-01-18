"""
Render package for 3D Gaussian Splatting rendering.

This package provides rendering functionality using InstantSplat,
wrapping the train.py and render.py workflow.
"""
from render.instant_splat_wrapper import InstantSplatWrapper

__all__ = ["InstantSplatWrapper"]
