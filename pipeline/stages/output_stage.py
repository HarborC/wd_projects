"""
Output stage for the pipeline.

This module provides the OutputStage class for final output generation.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.base import BaseStage, StageOutput
from render.instant_splat_wrapper import InstantSplatWrapper
from generate.seva_generator import SevaGenerator


class OutputStage(BaseStage):
    """
    Stage for final output generation using various methods.

    Supported methods:
        - render: 3D Gaussian Splatting rendering via InstantSplat
        - generate: SEVA novel view synthesis
    """

    def __init__(self, config: dict):
        """
        Initialize the output stage.

        Args:
            config: Configuration dictionary with keys:
                - method (str): Output method ('render' or 'generate')
                - frames_per_pair (int): Interpolated frames between keyframes
                - Additional method-specific config options
        """
        super().__init__(config)
        self.outputter = self._create_outputter()

    def _create_outputter(self):
        """Create the outputter instance based on config."""
        method = self.config.get("method", "render")

        if method == "render":
            self.logger.info("Using InstantSplat render outputter")
            return InstantSplatWrapper(self.config)
        elif method == "generate":
            self.logger.info("Using SEVA generator outputter")
            return SevaGenerator(self.config)
        else:
            raise ValueError(f"Unknown output method: {method}")

    def run(self, input_dir: str, output_dir: str) -> StageOutput:
        """
        Run the output stage.

        Args:
            input_dir: Directory containing COLMAP reconstruction
            output_dir: Directory where output will be saved

        Returns:
            StageOutput with output directory and method info

        Raises:
            NotImplementedError: Output workflow not yet implemented
        """
        if not self.validate_input(input_dir):
            raise ValueError(f"Invalid input directory: {input_dir}")

        self.prepare_output(output_dir)

        method = self.config.get("method", "render")
        frames_per_pair = self.config.get("frames_per_pair", 10)

        self.logger.info(f"Processing reconstruction from: {input_dir}")
        self.logger.info(f"Generating {frames_per_pair} frames per pair")

        # Output workflow to be implemented based on method
        # This will depend on the specific render/generate pipeline
        raise NotImplementedError(f"{method} output workflow not yet implemented")
