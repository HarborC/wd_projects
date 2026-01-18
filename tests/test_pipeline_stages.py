"""
Unit tests for pipeline stage classes.

This module tests the initialization and basic configuration handling
of the new pipeline stage classes (OutputStage and Gen3RStage).
"""
import sys
from pathlib import Path
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.stages.output_stage import OutputStage
from pipeline.stages.gen3r_stage import Gen3RStage


class TestOutputStage:
    """Tests for OutputStage initialization and configuration."""

    def test_initialization_with_render_method(self):
        """Test OutputStage initialization with render method."""
        config = {
            "method": "render",
            "render": {}
        }
        stage = OutputStage(config)
        assert stage.config["method"] == "render"
        assert stage.outputter is not None
        # Check that outputter is InstantSplatWrapper
        assert "InstantSplatWrapper" in str(type(stage.outputter))

    def test_initialization_with_generate_method(self):
        """Test OutputStage initialization with generate method."""
        config = {
            "method": "generate",
            "generate": {}
        }
        stage = OutputStage(config)
        assert stage.config["method"] == "generate"
        assert stage.outputter is not None
        # Check that outputter is SevaGenerator
        assert "SevaGenerator" in str(type(stage.outputter))

    def test_initialization_default_method(self):
        """Test OutputStage initialization with default method."""
        config = {}
        stage = OutputStage(config)
        assert stage.config.get("method", "render") == "render"

    def test_initialization_with_frames_per_pair(self):
        """Test OutputStage initialization with frames_per_pair config."""
        config = {
            "method": "render",
            "frames_per_pair": 15
        }
        stage = OutputStage(config)
        assert stage.config["frames_per_pair"] == 15

    def test_initialization_with_invalid_method_raises_error(self):
        """Test OutputStage initialization with invalid method raises ValueError."""
        config = {
            "method": "invalid_method"
        }
        with pytest.raises(ValueError, match="Unknown output method"):
            OutputStage(config)


class TestGen3RStage:
    """Tests for Gen3RStage initialization and configuration."""

    def test_initialization_with_valid_task_1view(self):
        """Test Gen3RStage initialization with valid '1view' task."""
        config = {
            "task": "1view"
        }
        stage = Gen3RStage(config)
        assert stage.task == "1view"
        assert stage.config["task"] == "1view"

    def test_initialization_with_valid_task_2view(self):
        """Test Gen3RStage initialization with valid '2view' task."""
        config = {
            "task": "2view"
        }
        stage = Gen3RStage(config)
        assert stage.task == "2view"
        assert stage.config["task"] == "2view"

    def test_initialization_with_valid_task_allview(self):
        """Test Gen3RStage initialization with valid 'allview' task."""
        config = {
            "task": "allview"
        }
        stage = Gen3RStage(config)
        assert stage.task == "allview"
        assert stage.config["task"] == "allview"

    def test_initialization_default_task(self):
        """Test Gen3RStage initialization with default task."""
        config = {}
        stage = Gen3RStage(config)
        assert stage.task == "allview"

    def test_initialization_with_invalid_task_raises_error(self):
        """Test Gen3RStage initialization with invalid task raises ValueError."""
        config = {
            "task": "invalid"
        }
        with pytest.raises(ValueError, match="Invalid task"):
            Gen3RStage(config)

    def test_initialization_with_additional_config(self):
        """Test Gen3RStage initialization with additional configuration."""
        config = {
            "task": "allview",
            "model_path": "/path/to/model",
            "prompts": "test prompt",
            "device": "cuda"
        }
        stage = Gen3RStage(config)
        assert stage.task == "allview"
        assert stage.config["model_path"] == "/path/to/model"
        assert stage.config["prompts"] == "test prompt"
        assert stage.config["device"] == "cuda"

    def test_initialization_with_invalid_task_another_variant(self):
        """Test Gen3RStage initialization with another invalid task variant."""
        config = {
            "task": "3view"
        }
        with pytest.raises(ValueError, match="Invalid task"):
            Gen3RStage(config)
