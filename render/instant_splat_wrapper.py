"""
InstantSplat wrapper for training and rendering.

This module wraps the InstantSplat train.py and render.py workflow
for integration with the pipeline.
"""
import sys
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class InstantSplatWrapper:
    """
    Wrapper for InstantSplat training and rendering.

    This class encapsulates the InstantSplat workflow:
    1. Train 3D Gaussian model from COLMAP sparse reconstruction
    2. Render interpolated video along camera trajectory
    """

    def __init__(self, config: dict):
        """
        Initialize the wrapper.

        Args:
            config: Configuration dictionary with keys:
                - iterations: Training iterations (default: 3000)
                - resolution: Output resolution [width, height]
                - sh_degree: Spherical harmonics degree (default: 0)
                - densify_until_iter: Densification iterations (default: 0)
        """
        self.config = config
        self.iterations = config.get("iterations", 3000)
        self.resolution = config.get("resolution", [1920, 1080])

    def train(self, colmap_dir: str, output_dir: str) -> str:
        """
        Train 3D Gaussian model from COLMAP reconstruction.

        Args:
            colmap_dir: Path to COLMAP sparse/0/ directory
            output_dir: Path to save trained model

        Returns:
            Path to trained model directory
        """
        from render.train import train_gaussians

        logger.info(f"Training 3DGS model from {colmap_dir}")

        return train_gaussians(
            colmap_dir=colmap_dir,
            output_dir=output_dir,
            iterations=self.iterations,
            resolution=self.resolution,
            sh_degree=self.config.get("sh_degree", 0),
            densify_until_iter=self.config.get("densify_until_iter", 0),
        )

    def render_interpolated_video(self, model_dir: str,
                                   frames_per_pair: int,
                                   output_dir: str) -> Dict[str, str]:
        """
        Render interpolated video from trained model.

        Args:
            model_dir: Path to trained model
            frames_per_pair: Number of interpolated frames between keyframes
            output_dir: Path to save rendered images and video

        Returns:
            Dictionary with output paths
        """
        from render.render import render_interpolated

        logger.info(f"Rendering video from {model_dir}")

        return render_interpolated(
            model_dir=model_dir,
            frames_per_pair=frames_per_pair,
            output_dir=output_dir
        )
