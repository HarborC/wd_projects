"""
Undistortion stage for the pipeline.

This module provides the UndistortStage class for image undistortion.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.base import BaseStage, StageOutput
from undistortion.geocalib_undistorter import GeoCalibUndistorter
from undistortion.anycalib_undistorter import AnyCalibUndistorter


class UndistortStage(BaseStage):
    """
    Stage for undistorting images using various methods.

    Supported methods:
        - geocalib: GeoCalibUndistorter
        - anycalib: AnyCalibUndistorter
    """

    def __init__(self, config: dict):
        """
        Initialize the undistortion stage.

        Args:
            config: Configuration dictionary with keys:
                - method (str): Undistortion method ('geocalib' or 'anycalib')
                - intrinsics_path (str): Path to save intrinsics JSON
        """
        super().__init__(config)
        self.undistorter = self._create_undistorter()

    def _create_undistorter(self):
        """Create the undistorter instance based on config."""
        method = self.config.get("method", "geocalib")

        if method == "geocalib":
            self.logger.info("Using GeoCalib undistorter")
            return GeoCalibUndistorter()
        elif method == "anycalib":
            self.logger.info("Using AnyCalib undistorter")
            return AnyCalibUndistorter()
        else:
            raise ValueError(f"Unknown undistort method: {method}")

    def run(self, input_dir: str, output_dir: str) -> StageOutput:
        """
        Run the undistortion stage.

        Args:
            input_dir: Directory containing distorted input images
            output_dir: Directory where undistorted images will be saved

        Returns:
            StageOutput with output directory and intrinsics path
        """
        if not self.validate_input(input_dir):
            raise ValueError(f"Invalid input directory: {input_dir}")

        self.prepare_output(output_dir)

        intrinsics_path = self.config.get("intrinsics_path")
        if intrinsics_path is None:
            intrinsics_path = str(Path(output_dir) / "intrinsics.json")

        self.logger.info(f"Processing images from: {input_dir}")
        self.logger.info(f"Intrinsics will be saved to: {intrinsics_path}")

        self.undistorter.process_directory(input_dir, output_dir, intrinsics_path)

        return StageOutput(
            output_dir=output_dir,
            metadata={"intrinsics_path": intrinsics_path, "method": self.config.get("method")}
        )
