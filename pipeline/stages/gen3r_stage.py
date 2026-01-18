"""
Gen3R stage for the pipeline.

This module provides the Gen3RStage class for Gen3R end-to-end processing.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.base import BaseStage, StageOutput
from gen3r.gen3r_pipeline import Gen3RPipelineWrapper


class Gen3RStage(BaseStage):
    """
    Stage for Gen3R end-to-end 3D scene generation.

    Supported tasks:
        - 1view: Single-view reconstruction
        - 2view: Two-view reconstruction
        - allview: Multi-view reconstruction
    """

    def __init__(self, config: dict):
        """
        Initialize the Gen3R stage.

        Args:
            config: Configuration dictionary with keys:
                - task (str): Gen3R task type ('1view', '2view', or 'allview')
                - model_path (str): Path to Gen3R checkpoint
                - prompts (str): Text prompt for generation
                - device (str): Device to run on ('cuda' or 'cpu')
        """
        super().__init__(config)

        # Validate task
        self.task = self.config.get("task", "allview")
        valid_tasks = ["1view", "2view", "allview"]
        if self.task not in valid_tasks:
            raise ValueError(f"Invalid task: {self.task}. Must be one of {valid_tasks}")

        self.gen3r_pipeline = Gen3RPipelineWrapper(self.config)

    def run(self, input_dir: str, output_dir: str) -> StageOutput:
        """
        Run the Gen3R stage.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory where Gen3R outputs will be saved

        Returns:
            StageOutput with output directory and Gen3R metadata

        Raises:
            ValueError: If input directory is invalid
            NotImplementedError: Gen3R processing workflow not yet implemented
        """
        if not self.validate_input(input_dir):
            raise ValueError(f"Invalid input directory: {input_dir}")

        self.prepare_output(output_dir)

        self.logger.info(f"Processing Gen3R {self.task} task")
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")

        # Gen3R workflow to be implemented
        # This will call Gen3RPipelineWrapper.process() with appropriate parameters
        raise NotImplementedError("Gen3R processing workflow not yet implemented")
