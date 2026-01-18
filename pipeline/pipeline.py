"""
Main Pipeline class for orchestrating multi-view reconstruction.

This module provides the Pipeline class that coordinates different stages
of the processing pipeline.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.base import BaseStage, StageOutput
from pipeline.stages import UndistortStage, ReconstructionStage


logger = logging.getLogger(__name__)


class Pipeline:
    """
    Multi-view reconstruction processing pipeline orchestrator.

    This class coordinates different processing stages (undistortion,
    reconstruction, etc.) to create a complete workflow.

    Configuration example:
        config = {
            "undistort": {
                "method": "geocalib",
                "output_dir": "undistorted",
                "intrinsics_path": "intrinsics.json"
            },
            "reconstruction": {
                "method": "hunyuanworld",
                "device": "cuda",
                "output_dir": "reconstruction"
            }
        }
    """

    # Standard stage execution order
    DEFAULT_STAGE_ORDER = ["undistort", "reconstruction"]

    def __init__(self, config: dict):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Configuration dictionary with stage configs as keys
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.stages = {}
        self.stage_outputs = {}

        self._build_stages()

    def _build_stages(self):
        """Build stage instances based on configuration."""
        if "undistort" in self.config:
            self.stages["undistort"] = UndistortStage(self.config["undistort"])
            self.logger.info(f"Built undistort stage: {self.config['undistort'].get('method')}")

        if "reconstruction" in self.config:
            self.stages["reconstruction"] = ReconstructionStage(
                self.config["reconstruction"]
            )
            self.logger.info(f"Built reconstruction stage: {self.config['reconstruction'].get('method')}")

    def run(self, input_dir: str, output_base_dir: str) -> dict:
        """
        Execute the complete pipeline.

        Args:
            input_dir: Directory containing input images
            output_base_dir: Base directory for all outputs

        Returns:
            Dictionary mapping stage names to their StageOutput

        Raises:
            Exception: If any stage fails
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Pipeline")
        self.logger.info("=" * 60)
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output base directory: {output_base_dir}")
        self.logger.info(f"Configured stages: {list(self.stages.keys())}")

        current_input = input_dir

        # Execute stages in order
        for stage_name in self.DEFAULT_STAGE_ORDER:
            if stage_name not in self.stages:
                self.logger.info(f"Skipping stage: {stage_name} (not configured)")
                continue

            stage = self.stages[stage_name]
            output_dir = str(Path(output_base_dir) / stage.config.get("output_dir", stage_name))

            self.logger.info("")
            self.logger.info(">>> " + "-" * 50)
            self.logger.info(f">>> Running stage: {stage_name}")
            self.logger.info(">>> " + "-" * 50)

            try:
                output = stage.run(current_input, output_dir)
                self.stage_outputs[stage_name] = output
                current_input = output_dir

                self.logger.info("")
                self.logger.info("<<< " + "-" * 50)
                self.logger.info(f"<<< Stage {stage_name} completed")
                self.logger.info(f"<<< Output: {output.output_dir}")
                self.logger.info("<<< " + "-" * 50)
            except Exception as e:
                self.logger.error(f"Stage {stage_name} failed: {e}")
                self.logger.exception("Exception details:")
                raise

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Pipeline completed successfully")
        self.logger.info("=" * 60)

        return self.stage_outputs

    @classmethod
    def from_config_file(cls, config_path: str) -> "Pipeline":
        """
        Create a Pipeline from a JSON configuration file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            Configured Pipeline instance
        """
        import json

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        logger.info(f"Loaded config from: {config_path}")
        return cls(config)

    def list_stages(self) -> list:
        """
        List all configured stages.

        Returns:
            List of stage names
        """
        return list(self.stages.keys())
