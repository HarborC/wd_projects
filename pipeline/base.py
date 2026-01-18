"""
Pipeline base classes.

This module provides the base classes for all pipeline stages.
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class StageOutput:
    """
    Container for stage output data.

    Attributes:
        output_dir: Directory where stage outputs were written
        metadata: Optional dictionary of metadata about the stage execution
    """

    def __init__(self, output_dir: str, metadata: Optional[dict] = None):
        self.output_dir = output_dir
        self.metadata = metadata or {}

    def __repr__(self):
        return f"StageOutput(output_dir='{self.output_dir}', metadata_keys={list(self.metadata.keys())})"


class BaseStage(ABC):
    """
    Base class for all pipeline stages.

    Each stage represents a single processing step in the pipeline.
    Subclasses must implement the run() method.
    """

    def __init__(self, config: dict):
        """
        Initialize the stage with configuration.

        Args:
            config: Configuration dictionary for this stage
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self, input_dir: str, output_dir: str) -> StageOutput:
        """
        Execute the stage processing.

        Args:
            input_dir: Directory containing input data
            output_dir: Directory where outputs should be written

        Returns:
            StageOutput containing output directory and metadata
        """
        pass

    def validate_input(self, input_dir: str) -> bool:
        """
        Validate that the input directory exists and is accessible.

        Args:
            input_dir: Path to validate

        Returns:
            True if input is valid, False otherwise
        """
        path = Path(input_dir)
        if not path.exists():
            self.logger.error(f"Input directory does not exist: {input_dir}")
            return False
        if not path.is_dir():
            self.logger.error(f"Input path is not a directory: {input_dir}")
            return False
        return True

    def prepare_output(self, output_dir: str):
        """
        Create the output directory if it doesn't exist.

        Args:
            output_dir: Path to create
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {output_dir}")
