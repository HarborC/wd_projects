# Pipeline Parallel Branches Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a parallel-branch pipeline architecture supporting main branch (reconstruction + render/generate output) and Gen3R branch (end-to-end generation).

**Architecture:** Two independent branches selected by configuration - main branch uses existing reconstruction with pluggable output (3DGS render or SEVA generate), Gen3R branch provides standalone end-to-end generation. Core functionality moved from test_code/ to git-tracked directories (render/, generate/, gen3r/).

**Tech Stack:** Python 3.10+, PyTorch, COLMAP, InstantSplat (3DGS), SEVA (diffusion), Gen3R

---

## Task 1: Create render/ directory structure

**Files:**
- Create: `render/__init__.py`
- Create: `render/instant_splat_wrapper.py`

**Step 1: Create render package init**

```python
# render/__init__.py
"""
Render package for 3D Gaussian Splatting rendering.

This package provides rendering functionality using InstantSplat,
wrapping the train.py and render.py workflow.
"""
from render.instant_splat_wrapper import InstantSplatWrapper

__all__ = ["InstantSplatWrapper"]
```

Create file: `render/__init__.py`

**Step 2: Create InstantSplat wrapper skeleton**

```python
# render/instant_splat_wrapper.py
"""
InstantSplat wrapper for training and rendering.

This module wraps the InstantSplat train.py and render.py workflow
for integration with the pipeline.
"""
import sys
from pathlib import Path
from typing import Optional
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
                - instant_splat_path: Path to InstantSplat code
                - iterations: Training iterations (default: 3000)
                - resolution: Output resolution [width, height]
        """
        self.config = config
        self.instant_splat_path = Path(config.get("instant_splat_path",
                                                    "test_code/InstantSplat"))
        self.iterations = config.get("iterations", 3000)

    def train(self, colmap_dir: str, output_dir: str) -> str:
        """
        Train 3D Gaussian model from COLMAP reconstruction.

        Args:
            colmap_dir: Path to COLMAP sparse/0/ directory
            output_dir: Path to save trained model

        Returns:
            Path to trained model directory
        """
        # TODO: Implement training wrapper
        logger.info(f"Training 3DGS model from {colmap_dir}")
        raise NotImplementedError("Training not yet implemented")

    def render_interpolated_video(self, model_dir: str,
                                   frames_per_pair: int,
                                   output_dir: str) -> dict:
        """
        Render interpolated video from trained model.

        Args:
            model_dir: Path to trained model
            frames_per_pair: Number of interpolated frames between keyframes
            output_dir: Path to save rendered images and video

        Returns:
            Dictionary with output paths
        """
        # TODO: Implement rendering wrapper
        logger.info(f"Rendering video from {model_dir}")
        raise NotImplementedError("Rendering not yet implemented")
```

Create file: `render/instant_splat_wrapper.py`

**Step 3: Commit**

```bash
git add render/__init__.py render/instant_splat_wrapper.py
git commit -m "feat: add render package with InstantSplat wrapper skeleton"
```

---

## Task 2: Create generate/ directory structure

**Files:**
- Create: `generate/__init__.py`
- Create: `generate/seva_generator.py`

**Step 1: Create generate package init**

```python
# generate/__init__.py
"""
Generate package for SEVA-based novel view generation.

This package provides generation functionality using SEVA
(Stable Virtual Camera) diffusion model.
"""
from generate.seva_generator import SevaGenerator

__all__ = ["SevaGenerator"]
```

Create file: `generate/__init__.py`

**Step 2: Create SEVA generator skeleton**

```python
# generate/seva_generator.py
"""
SEVA (Stable Virtual Camera) generator wrapper.

This module wraps the SEVA diffusion model for novel view generation.
"""
import sys
from pathlib import Path
from typing import Optional, List
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
        self.model_path = Path(config.get("model_path",
                                           "test_code/stable-virtual-camera"))
        self.guidance_scale = config.get("guidance_scale", 5.0)

    def generate(self, colmap_dir: str, frames_per_pair: int,
                 output_dir: str) -> dict:
        """
        Generate novel views along interpolated trajectory.

        Args:
            colmap_dir: Path to COLMAP sparse/0/ directory
            frames_per_pair: Number of interpolated frames between keyframes
            output_dir: Path to save generated images and video

        Returns:
            Dictionary with output paths
        """
        # TODO: Implement generation wrapper
        logger.info(f"Generating views from {colmap_dir}")
        raise NotImplementedError("Generation not yet implemented")
```

Create file: `generate/seva_generator.py`

**Step 3: Commit**

```bash
git add generate/__init__.py generate/seva_generator.py
git commit -m "feat: add generate package with SEVA generator skeleton"
```

---

## Task 3: Create gen3r/ directory structure

**Files:**
- Create: `gen3r/__init__.py`
- Create: `gen3r/gen3r_pipeline.py`

**Step 1: Create gen3r package init**

```python
# gen3r/__init__.py
"""
Gen3R package for end-to-end 3D scene generation and reconstruction.

This package provides Gen3R functionality for processing images
to generate 3D scenes with RGB outputs.
"""
from gen3r.gen3r_pipeline import Gen3RPipelineWrapper

__all__ = ["Gen3RPipelineWrapper"]
```

Create file: `gen3r/__init__.py`

**Step 2: Create Gen3R pipeline wrapper skeleton**

```python
# gen3r/gen3r_pipeline.py
"""
Gen3R pipeline wrapper.

This module wraps the Gen3R inference pipeline for end-to-end
3D scene generation from images.
"""
import sys
from pathlib import Path
from typing import Optional
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
                - prompts: Text prompt for generation
                - device: Device to run on (cuda/cpu)
        """
        self.config = config
        self.model_path = Path(config.get("model_path",
                                           "test_code/Gen3R/checkpoints"))
        self.task = config.get("task", "allview")

    def process(self, input_dir: str, output_dir: str) -> dict:
        """
        Process images with Gen3R.

        Args:
            input_dir: Path to input images directory
            output_dir: Path to save outputs

        Returns:
            Dictionary with output paths (rgb.mp4, pcds.ply, cameras.json)
        """
        # TODO: Implement Gen3R processing wrapper
        logger.info(f"Processing {input_dir} with Gen3R")
        raise NotImplementedError("Gen3R processing not yet implemented")
```

Create file: `gen3r/gen3r_pipeline.py`

**Step 3: Commit**

```bash
git add gen3r/__init__.py gen3r/gen3r_pipeline.py
git commit -m "feat: add gen3r package with pipeline wrapper skeleton"
```

---

## Task 4: Create OutputStage

**Files:**
- Create: `pipeline/stages/output_stage.py`

**Step 1: Create OutputStage class**

```python
# pipeline/stages/output_stage.py
"""
Output stage for novel view rendering/generation.

This module provides the OutputStage which handles both
3DGS rendering and SEVA generation based on configuration.
"""
import sys
from pathlib import Path
import logging
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.base import BaseStage, StageOutput

logger = logging.getLogger(__name__)


class OutputStage(BaseStage):
    """
    Stage for outputting novel views via render or generate.

    Supports two methods:
        - render: 3D Gaussian Splatting rendering
        - generate: SEVA diffusion-based generation
    """

    def __init__(self, config: dict):
        """
        Initialize the output stage.

        Args:
            config: Configuration dictionary with keys:
                - method: "render" or "generate"
                - interpolation: {frames_per_pair: int}
                - render: config for render method
                - generate: config for generate method
        """
        super().__init__(config)
        self.method = config.get("method", "render")
        self.interpolation_config = config.get("interpolation", {})
        self._create_outputter()

    def _create_outputter(self):
        """Create render or generate outputter based on method."""
        if self.method == "render":
            from render.instant_splat_wrapper import InstantSplatWrapper
            logger.info("Using 3DGS rendering output")
            self.outputter = InstantSplatWrapper(
                self.config.get("render", {})
            )
        elif self.method == "generate":
            from generate.seva_generator import SevaGenerator
            logger.info("Using SEVA generation output")
            self.outputter = SevaGenerator(
                self.config.get("generate", {})
            )
        else:
            raise ValueError(f"Unknown output method: {self.method}")

    def run(self, input_dir: str, output_dir: str) -> StageOutput:
        """
        Run the output stage.

        Args:
            input_dir: Directory with reconstruction results
            output_dir: Directory to save outputs

        Returns:
            StageOutput with output directory
        """
        if not self.validate_input(input_dir):
            raise ValueError(f"Invalid input directory: {input_dir}")

        self.prepare_output(output_dir)

        # Get reconstruction sparse directory
        sparse_dir = Path(input_dir) / "sparse" / "0"
        if not sparse_dir.exists():
            sparse_dir = Path(input_dir) / "sparse"
        if not sparse_dir.exists():
            raise ValueError(f"COLMAP sparse directory not found in {input_dir}")

        frames_per_pair = self.interpolation_config.get("frames_per_pair", 20)

        # Call outputter to render/generate
        if self.method == "render":
            # TODO: Implement render workflow
            logger.info(f"Rendering with {frames_per_pair} frames per pair")
            raise NotImplementedError("Render workflow not yet implemented")
        else:  # generate
            # TODO: Implement generate workflow
            logger.info(f"Generating with {frames_per_pair} frames per pair")
            raise NotImplementedError("Generate workflow not yet implemented")

        return StageOutput(
            output_dir=output_dir,
            metadata={
                "method": self.method,
                "frames_per_pair": frames_per_pair
            }
        )
```

Create file: `pipeline/stages/output_stage.py`

**Step 2: Update pipeline stages __init__.py**

```python
# pipeline/stages/__init__.py
"""
Pipeline stages package.
"""
from pipeline.base import BaseStage, StageOutput
from pipeline.stages.undistort_stage import UndistortStage
from pipeline.stages.reconstruction_stage import ReconstructionStage
from pipeline.stages.output_stage import OutputStage

__all__ = [
    "BaseStage",
    "StageOutput",
    "UndistortStage",
    "ReconstructionStage",
    "OutputStage",
]
```

Modify file: `pipeline/stages/__init__.py`

**Step 3: Commit**

```bash
git add pipeline/stages/output_stage.py pipeline/stages/__init__.py
git commit -m "feat: add OutputStage for render/generate output"
```

---

## Task 5: Create Gen3RStage

**Files:**
- Create: `pipeline/stages/gen3r_stage.py`

**Step 1: Create Gen3RStage class**

```python
# pipeline/stages/gen3r_stage.py
"""
Gen3R stage for end-to-end 3D scene generation.

This module provides the Gen3RStage for Gen3R processing.
"""
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.base import BaseStage, StageOutput

logger = logging.getLogger(__name__)


class Gen3RStage(BaseStage):
    """
    Stage for Gen3R end-to-end processing.

    Supports tasks:
        - 1view: Single frame to 3D
        - 2view: First-last frames to 3D
        - allview: Multi-view 3D reconstruction
    """

    def __init__(self, config: dict):
        """
        Initialize the Gen3R stage.

        Args:
            config: Configuration dictionary with keys:
                - task: "1view", "2view", or "allview"
                - model_path: Path to Gen3R checkpoint
                - prompts: Text prompt for generation
                - device: "cuda" or "cpu"
        """
        super().__init__(config)
        self.task = config.get("task", "allview")

        from gen3r.gen3r_pipeline import Gen3RPipelineWrapper
        logger.info(f"Creating Gen3R pipeline for task: {self.task}")
        self.gen3r = Gen3RPipelineWrapper(config)

    def run(self, input_dir: str, output_dir: str) -> StageOutput:
        """
        Run the Gen3R stage.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs

        Returns:
            StageOutput with output directory
        """
        if not self.validate_input(input_dir):
            raise ValueError(f"Invalid input directory: {input_dir}")

        self.prepare_output(output_dir)

        self.logger.info(f"Processing images from: {input_dir}")

        # Call Gen3R processing
        # TODO: Implement Gen3R processing
        logger.info("Gen3R processing not yet implemented")
        raise NotImplementedError("Gen3R processing not yet implemented")

        return StageOutput(
            output_dir=output_dir,
            metadata={"task": self.task}
        )
```

Create file: `pipeline/stages/gen3r_stage.py`

**Step 2: Update pipeline stages __init__.py**

```python
# pipeline/stages/__init__.py
"""
Pipeline stages package.
"""
from pipeline.base import BaseStage, StageOutput
from pipeline.stages.undistort_stage import UndistortStage
from pipeline.stages.reconstruction_stage import ReconstructionStage
from pipeline.stages.output_stage import OutputStage
from pipeline.stages.gen3r_stage import Gen3RStage

__all__ = [
    "BaseStage",
    "StageOutput",
    "UndistortStage",
    "ReconstructionStage",
    "OutputStage",
    "Gen3RStage",
]
```

Modify file: `pipeline/stages/__init__.py`

**Step 3: Commit**

```bash
git add pipeline/stages/gen3r_stage.py pipeline/stages/__init__.py
git commit -m "feat: add Gen3RStage for Gen3R processing"
```

---

## Task 6: Modify Pipeline class for branch support

**Files:**
- Modify: `pipeline/pipeline.py`

**Step 1: Update Pipeline class to support branches**

Read current: `pipeline/pipeline.py`

Add branch support to Pipeline class:

```python
# In pipeline/pipeline.py, modify Pipeline class

class Pipeline:
    """
    Multi-view reconstruction processing pipeline orchestrator.

    Now supports parallel branches:
        - main: undistort -> reconstruction -> output (render/generate)
        - gen3r: undistort -> gen3r
    """

    # Stage configurations for each branch
    BRANCH_STAGES = {
        "main": ["undistort", "reconstruction", "output"],
        "gen3r": ["undistort", "gen3r"]
    }

    def __init__(self, config: dict):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Configuration dictionary with:
                - branch: "main" or "gen3r"
                - Stage configs as keys (undistort, reconstruction, output, gen3r)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.stages = {}
        self.stage_outputs = {}
        self.branch = config.get("branch", "main")

        self._build_stages()

    def _build_stages(self):
        """Build stage instances based on configuration and branch."""
        self.logger.info(f"Building pipeline for branch: {self.branch}")

        if self.branch == "main":
            self._build_main_branch()
        elif self.branch == "gen3r":
            self._build_gen3r_branch()
        else:
            raise ValueError(f"Unknown branch: {self.branch}")

    def _build_main_branch(self):
        """Build stages for main branch."""
        if "undistort" in self.config:
            from pipeline.stages import UndistortStage
            self.stages["undistort"] = UndistortStage(self.config["undistort"])
            self.logger.info(f"Built undistort stage: {self.config['undistort'].get('method')}")

        if "reconstruction" in self.config:
            from pipeline.stages import ReconstructionStage
            self.stages["reconstruction"] = ReconstructionStage(
                self.config["reconstruction"]
            )
            self.logger.info(f"Built reconstruction stage: {self.config['reconstruction'].get('method')}")

        if "output" in self.config:
            from pipeline.stages import OutputStage
            self.stages["output"] = OutputStage(self.config["output"])
            self.logger.info(f"Built output stage: {self.config['output'].get('method')}")

    def _build_gen3r_branch(self):
        """Build stages for Gen3R branch."""
        if "undistort" in self.config:
            from pipeline.stages import UndistortStage
            self.stages["undistort"] = UndistortStage(self.config["undistort"])
            self.logger.info(f"Built undistort stage: {self.config['undistort'].get('method')}")

        if "gen3r" in self.config:
            from pipeline.stages import Gen3RStage
            self.stages["gen3r"] = Gen3RStage(self.config["gen3r"])
            self.logger.info("Built gen3r stage")

    def run(self, input_dir: str, output_base_dir: str) -> dict:
        """
        Execute the complete pipeline.

        Args:
            input_dir: Directory containing input images
            output_base_dir: Base directory for all outputs

        Returns:
            Dictionary mapping stage names to their StageOutput
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Pipeline")
        self.logger.info(f"Branch: {self.branch}")
        self.logger.info("=" * 60)

        current_input = input_dir
        stage_order = self.BRANCH_STAGES.get(self.branch, [])

        for stage_name in stage_order:
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
```

Modify file: `pipeline/pipeline.py`

**Step 2: Commit**

```bash
git add pipeline/pipeline.py
git commit -m "feat: add branch support to Pipeline class"
```

---

## Task 7: Add SEVA conversion to ReconstructionStage

**Files:**
- Modify: `pipeline/stages/reconstruction_stage.py`

**Step 1: Add SEVA conversion to run method**

Read current: `pipeline/stages/reconstruction_stage.py`

Modify to include SEVA conversion:

```python
# In pipeline/stages/reconstruction_stage.py

class ReconstructionStage(BaseStage):
    """
    Stage for 3D reconstruction using various methods.

    Supported methods:
        - da3: Depth Anything 3
        - mast3r: MASt3R multi-view stereo
        - hunyuanworld: HunyuanWorld Mirror
        - vggtx: VGGT-X

    Outputs:
        - COLMAP format (sparse/0/)
        - SEVA format (seva/)
    """

    def __init__(self, config: dict):
        """
        Initialize the reconstruction stage.

        Args:
            config: Configuration dictionary with keys:
                - method (str): Reconstruction method
                - device (str, optional): Device to use ('cuda' or 'cpu')
                - seva_interpolate (int, optional): Frames per pair for SEVA (default: 20)
        """
        super().__init__(config)
        self.seva_interpolate = config.get("seva_interpolate", 20)
        self.reconstructor = self._create_reconstructor()

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

        # Run reconstruction (outputs COLMAP format)
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

    def _convert_to_seva(self, output_dir: str):
        """Convert COLMAP output to SEVA format."""
        import sys
        from pathlib import Path

        # Add scripts directory to path
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from recon_to_seva_converter import convert_reconstruction_to_seva

        seva_output_dir = Path(output_dir) / "seva"

        self.logger.info(f"Converting to SEVA format with {self.seva_interpolate} frames per pair")

        try:
            convert_reconstruction_to_seva(
                recon_output_dir=Path(output_dir),
                seva_output_dir=seva_output_dir,
                interpolate=self.seva_interpolate
            )
            self.logger.info(f"SEVA format saved to: {seva_output_dir}")
        except Exception as e:
            self.logger.warning(f"SEVA conversion failed: {e}")
            self.logger.warning("Continuing without SEVA output")
```

Modify file: `pipeline/stages/reconstruction_stage.py`

**Step 2: Commit**

```bash
git add pipeline/stages/reconstruction_stage.py
git commit -m "feat: add SEVA format conversion to ReconstructionStage"
```

---

## Task 8: Create configuration examples

**Files:**
- Create: `configs/main_branch_render.json`
- Create: `configs/main_branch_generate.json`
- Create: `configs/gen3r_branch.json`

**Step 1: Create main branch render config**

```json
{
    "branch": "main",
    "undistort": {
        "method": "geocalib",
        "output_dir": "undistorted",
        "intrinsics_path": "undistorted/intrinsics.json"
    },
    "reconstruction": {
        "method": "hunyuanworld",
        "device": "cuda",
        "output_dir": "reconstruction",
        "seva_interpolate": 20
    },
    "output": {
        "method": "render",
        "interpolation": {
            "frames_per_pair": 20
        },
        "output_dir": "output"
    },
    "render": {
        "instant_splat_path": "test_code/InstantSplat",
        "iterations": 3000,
        "resolution": [1920, 1080]
    }
}
```

Create file: `configs/main_branch_render.json`

**Step 2: Create main branch generate config**

```json
{
    "branch": "main",
    "undistort": {
        "method": "geocalib",
        "output_dir": "undistorted"
    },
    "reconstruction": {
        "method": "mast3r",
        "device": "cuda",
        "output_dir": "reconstruction",
        "seva_interpolate": 20
    },
    "output": {
        "method": "generate",
        "interpolation": {
            "frames_per_pair": 20
        },
        "output_dir": "output"
    },
    "generate": {
        "model_path": "test_code/stable-virtual-camera",
        "guidance_scale": 5.0,
        "num_inference_steps": 50
    }
}
```

Create file: `configs/main_branch_generate.json`

**Step 3: Create gen3r branch config**

```json
{
    "branch": "gen3r",
    "undistort": {
        "method": "anycalib",
        "output_dir": "undistorted"
    },
    "gen3r": {
        "task": "allview",
        "prompts": "a beautiful scene",
        "model_path": "test_code/Gen3R/checkpoints",
        "device": "cuda",
        "output_dir": "gen3r_output"
    }
}
```

Create file: `configs/gen3r_branch.json`

**Step 4: Commit**

```bash
git add configs/main_branch_render.json configs/main_branch_generate.json configs/gen3r_branch.json
git commit -m "feat: add configuration examples for pipeline branches"
```

---

## Task 9: Update pipeline.py CLI for branch support

**Files:**
- Modify: `pipeline.py`

**Step 1: Add branch option to CLI**

Read current: `pipeline.py`

Add branch argument:

```python
# In pipeline.py, add to argument parser

parser.add_argument(
    "--branch",
    type=str,
    default="main",
    choices=["main", "gen3r"],
    help="Pipeline branch to run"
)

# Update build_config_from_args to include branch
def build_config_from_args(args) -> dict:
    """Build configuration dictionary from command-line arguments."""
    config = {"branch": args.branch}

    if not args.skip_undistort:
        config["undistort"] = {
            "method": args.undistort_method,
            "output_dir": "undistorted",
            "intrinsics_path": "undistorted/intrinsics.json"
        }

    if args.branch == "main":
        config["reconstruction"] = {
            "method": args.reconstruct_method,
            "device": args.device,
            "output_dir": "reconstruction"
        }

        # Add output stage config (default to render for now)
        config["output"] = {
            "method": "render",
            "interpolation": {"frames_per_pair": 20},
            "output_dir": "output"
        }
    # Note: gen3r branch requires config file for full options

    return config
```

Modify file: `pipeline.py`

**Step 2: Commit**

```bash
git add pipeline.py
git commit -m "feat: add branch option to CLI"
```

---

## Task 10: Create basic tests

**Files:**
- Create: `tests/test_pipeline_stages.py`
- Create: `tests/__init__.py`

**Step 1: Create tests package**

```python
# tests/__init__.py
"""
Tests package for pipeline.
"""
```

Create file: `tests/__init__.py`

**Step 2: Create basic stage tests**

```python
# tests/test_pipeline_stages.py
"""
Tests for pipeline stages.
"""
import pytest
from pathlib import Path
from pipeline.base import StageOutput
from pipeline.stages import OutputStage


class TestOutputStage:
    """Test OutputStage functionality."""

    def test_output_stage_init_render(self):
        """Test OutputStage initialization with render method."""
        config = {
            "method": "render",
            "interpolation": {"frames_per_pair": 20},
            "render": {"instant_splat_path": "test_code/InstantSplat"}
        }
        stage = OutputStage(config)
        assert stage.method == "render"
        assert stage.interpolation_config == {"frames_per_pair": 20}

    def test_output_stage_init_generate(self):
        """Test OutputStage initialization with generate method."""
        config = {
            "method": "generate",
            "interpolation": {"frames_per_pair": 10},
            "generate": {"model_path": "test_code/stable-virtual-camera"}
        }
        stage = OutputStage(config)
        assert stage.method == "generate"
        assert stage.interpolation_config == {"frames_per_pair": 10}

    def test_output_stage_invalid_method(self):
        """Test OutputStage with invalid method raises error."""
        config = {
            "method": "invalid",
            "interpolation": {"frames_per_pair": 20}
        }
        with pytest.raises(ValueError, match="Unknown output method"):
            OutputStage(config)
```

Create file: `tests/test_pipeline_stages.py`

**Step 3: Run tests to verify structure**

```bash
cd /data/cjg/projects/wd_projects/.worktrees/pipeline-parallel-branches
python -m pytest tests/test_pipeline_stages.py -v
```

Expected: Tests should pass (testing initialization only)

**Step 4: Commit**

```bash
git add tests/__init__.py tests/test_pipeline_stages.py
git commit -m "test: add basic tests for pipeline stages"
```

---

## Summary

This implementation plan creates the foundation for the parallel-branch pipeline architecture:

1. ✅ New directories: `render/`, `generate/`, `gen3r/` with wrapper skeletons
2. ✅ New stages: `OutputStage`, `Gen3RStage`
3. ✅ Pipeline class updated for branch support
4. ✅ SEVA conversion integrated into ReconstructionStage
5. ✅ Configuration examples for both branches
6. ✅ CLI updated for branch selection
7. ✅ Basic tests structure

**Remaining TODOs (marked with NotImplementedError):**
- Implement InstantSplatWrapper.train() and render_interpolated_video()
- Implement SevaGenerator.generate()
- Implement Gen3RPipelineWrapper.process()
- Implement OutputStage.run() full workflow
- Implement Gen3RStage.run() full workflow

**Next Steps:**
1. Implement wrapper methods by copying/adapting code from test_code/ directories
2. Add interpolation logic from recon_to_seva_converter.py
3. Add comprehensive integration tests
4. Documentation updates
