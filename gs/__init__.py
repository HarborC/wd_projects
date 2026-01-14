"""
Gaussian Splatting (GS) training and rendering module.

This module provides functionality to train 3D Gaussian Splatting models
from reconstruction output and render novel views.
"""

from .train import train_gaussians
from .render import render_gaussians

__all__ = ['train_gaussians', 'render_gaussians']
