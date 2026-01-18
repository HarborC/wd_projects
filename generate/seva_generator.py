"""
SEVA (Stable Virtual Camera) generator wrapper.

This module wraps the SEVA diffusion model for novel view generation.
"""
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class SevaGenerator:
    """
    Wrapper for SEVA novel view generation.

    This class encapsulates the SEVA workflow for generating
    novel views along interpolated camera trajectories.
    """

    def __init__(self, config: dict):
        """
        Initialize the generator.

        Args:
            config: Configuration dictionary with keys:
                - model_path: Path to SEVA checkpoint
                - guidance_scale: Diffusion guidance scale
                - num_inference_steps: Number of diffusion steps
        """
        self.config = config
        self.model_path = config.get("model_path", "stabilityai/stable-virtual-camera")
        self.guidance_scale = config.get("guidance_scale", 2.0)
        self.num_inference_steps = config.get("num_inference_steps", 50)

    def generate(self, colmap_dir: str, frames_per_pair: int,
                 output_dir: str) -> Dict[str, str]:
        """
        Generate novel views along interpolated trajectory.

        Args:
            colmap_dir: Path to COLMAP sparse/0/ directory
            frames_per_pair: Number of interpolated frames between keyframes
            output_dir: Path to save generated images and video

        Returns:
            Dictionary with output paths
        """
        from generate.infer import run_seva_inference

        logger.info(f"Generating views from {colmap_dir}")

        return run_seva_inference(
            colmap_dir=colmap_dir,
            output_dir=output_dir,
            task="img2img",
            traj_prior="interpolated",
            frames_per_pair=frames_per_pair,
            model_path=self.model_path,
            guidance_scale=self.guidance_scale,
            num_steps=self.num_inference_steps
        )
