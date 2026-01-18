"""
Gen3R pipeline wrapper.

This module wraps the Gen3R inference pipeline for end-to-end
3D scene generation from images.
"""
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Gen3RPipelineWrapper:
    """
    Wrapper for Gen3R end-to-end pipeline.

    This class encapsulates the Gen3R workflow for processing
    images and generating 3D scenes with RGB video output.
    """

    def __init__(self, config: dict):
        """
        Initialize the wrapper.

        Args:
            config: Configuration dictionary with keys:
                - model_path: Path to Gen3R checkpoint
                - task: Task type (1view, 2view, allview)
                - device: Device to run on (cuda/cpu)
        """
        self.model_path = Path(config.get("model_path",
                                           "test_code/Gen3R/checkpoints"))
        self.task = config.get("task", "allview")
        self.device = config.get("device", "cuda")

    def process(self, input_dir: str, output_dir: str,
                prompts: str = "a beautiful scene",
                cameras: str = "free") -> dict:
        """
        Process images with Gen3R.

        Args:
            input_dir: Path to input images directory or video file
            output_dir: Path to save outputs
            prompts: Text prompt for generation
            cameras: Camera trajectory or JSON file path

        Returns:
            Dictionary with output paths (rgb.mp4, pcds.ply, cameras.json)
        """
        from gen3r.infer import run_gen3r_inference

        logger.info(f"Processing {input_dir} with Gen3R (task={self.task})")

        return run_gen3r_inference(
            frame_path=input_dir,
            prompts=prompts,
            output_dir=output_dir,
            task=self.task,
            cameras=cameras,
            pretrained_model_name_or_path=str(self.model_path),
            device=self.device
        )
