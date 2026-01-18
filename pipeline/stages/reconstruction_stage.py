"""
Reconstruction stage for the pipeline.

This module provides the ReconstructionStage class for 3D reconstruction.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.base import BaseStage, StageOutput
from reconstruction.factory import ReconstructionFactory


class ReconstructionStage(BaseStage):
    """
    Stage for 3D reconstruction using various methods.

    Supported methods:
        - da3: Depth Anything 3
        - mast3r: MASt3R multi-view stereo
        - hunyuanworld: HunyuanWorld Mirror
        - vggtx: VGGT-X

    Output:
        Generates 3D reconstruction in COLMAP format and converts to SEVA format
        for downstream processing.
    """

    def __init__(self, config: dict):
        """
        Initialize the reconstruction stage.

        Args:
            config: Configuration dictionary with keys:
                - method (str): Reconstruction method
                - device (str, optional): Device to use ('cuda' or 'cpu')
                - seva_interpolate (int, optional): Number of interpolated poses per pair
        """
        super().__init__(config)
        self.seva_interpolate = self.config.get("seva_interpolate", 20)
        self.reconstructor = self._create_reconstructor()

    def _create_reconstructor(self):
        """Create the reconstructor instance based on config."""
        method = self.config.get("method", "da3")
        device = self.config.get("device")

        self.logger.info(f"Creating {method} reconstructor" + (f" on {device}" if device else ""))

        return ReconstructionFactory.create(
            backend=method,
            device=device
        )

    def _convert_to_seva(self, output_dir: str) -> None:
        """
        Convert reconstruction output to SEVA format.

        Args:
            output_dir: Directory containing reconstruction output
        """
        try:
            from scripts.recon_to_seva_converter import convert_reconstruction_to_seva

            seva_output_dir = Path(output_dir) / "seva"
            self.logger.info(f"Converting to SEVA format with interpolate={self.seva_interpolate}")

            convert_reconstruction_to_seva(
                recon_output_dir=Path(output_dir),
                seva_output_dir=seva_output_dir,
                interpolate=self.seva_interpolate
            )

            self.logger.info(f"SEVA format saved to: {seva_output_dir}")

        except Exception as e:
            self.logger.warning(f"Failed to convert to SEVA format: {e}")

    def run(self, input_dir: str, output_dir: str) -> StageOutput:
        """
        Run the reconstruction stage.

        Args:
            input_dir: Directory containing input images (usually undistorted)
            output_dir: Directory where reconstruction results will be saved

        Returns:
            StageOutput with output directory
        """
        if not self.validate_input(input_dir):
            raise ValueError(f"Invalid input directory: {input_dir}")

        self.prepare_output(output_dir)

        self.logger.info(f"Processing images from: {input_dir}")

        self.reconstructor.process_directory(input_dir, output_dir)

        # Convert to SEVA format
        self._convert_to_seva(output_dir)

        return StageOutput(
            output_dir=output_dir,
            metadata={
                "method": self.config.get("method"),
                "device": self.config.get("device"),
                "seva_interpolate": self.seva_interpolate
            }
        )
