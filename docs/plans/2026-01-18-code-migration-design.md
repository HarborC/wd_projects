# Code Migration Design: test_code/ to Tracked Packages

## Overview

This document describes the design for migrating core code from `test_code/` into git-tracked packages (`gen3r/`, `render/`, `generate/`). The migration follows a selective approach: main workflow + minimal dependencies, organized as subpackage structures.

## Architecture

### Package Structure

```
gen3r/                          # Gen3R end-to-end reconstruction
├── __init__.py
├── gen3r_pipeline.py           # Pipeline wrapper (existing, to be extended)
├── infer.py                    # Main inference entry (from test_code/Gen3R/infer.py)
├── models/                     # Model-related code
│   ├── __init__.py
│   └── vggt/
│       ├── __init__.py
│       └── utils/
│           └── pose_enc.py     # Pose encoding utilities
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── common_utils.py         # Common utilities (save videos, point clouds, etc.)
│   └── data_utils.py           # Data processing utilities

render/                         # InstantSplat 3DGS rendering
├── __init__.py
├── instant_splat_wrapper.py    # Wrapper (existing, to be extended)
├── train.py                    # Training entry (from test_code/InstantSplat/train.py)
├── render.py                   # Rendering entry (from test_code/InstantSplat/render.py)
├── scene/                      # Scene management
│   ├── __init__.py
│   └── dataset_readers.py      # COLMAP data loading
├── gaussian_renderer/          # Gaussian renderer
│   └── __init__.py
└── utils/
    ├── __init__.py
    └── camera_utils.py         # Camera utilities (interpolated path generation)

generate/                       # SEVA generation
├── __init__.py
├── seva_generator.py           # Generator wrapper (existing, to be extended)
├── infer.py                    # Inference entry (from test_code/stable-virtual-camera/demo.py)
├── data_io/                    # Data I/O
│   └── __init__.py
├── geometry/                   # Geometry utilities
│   ├── __init__.py
│   └── camera_utils.py         # Camera trajectory generation
└── model/                      # Model wrappers
    └── __init__.py
```

## Migration Strategy

### Gen3R Migration

**Source Analysis:**
- `test_code/Gen3R/infer.py` - Main inference script, ~300 lines
- Key dependencies:
  - `gen3r.pipeline.Gen3RPipeline` - Core Pipeline class
  - `gen3r.models.vggt.utils.pose_enc` - Pose encoding
  - `gen3r.utils.*` - Utility functions

**Migration Content:**

| Destination | Source | Description |
|-------------|--------|-------------|
| `gen3r/infer.py` | `test_code/Gen3R/infer.py` | Main inference entry, simplified to `run_gen3r_inference()` function |
| `gen3r/models/vggt/utils/pose_enc.py` | `test_code/Gen3R/gen3r/models/vggt/utils/pose_enc.py` | Pose encoding utilities |
| `gen3r/utils/common_utils.py` | `test_code/Gen3R/gen3r/utils/common_utils.py` | Save videos, point clouds, etc. |
| `gen3r/utils/data_utils.py` | `test_code/Gen3R/gen3r/utils/data_utils.py` | Data preprocessing |

**Simplification Strategy:**
- Remove argparse, convert to function parameters
- Keep core inference logic
- `gen3r_pipeline.py` calls `infer.run_gen3r_inference()`

### InstantSplat Migration

**Source Analysis:**
- `test_code/InstantSplat/train.py` - Training script, ~330 lines
- `test_code/InstantSplat/render.py` - Rendering script, ~270 lines
- Key dependencies: internal `arguments`, `scene`, `gaussian_renderer`, `utils`

**Migration Content:**

| Destination | Source | Description |
|-------------|--------|-------------|
| `render/train.py` | `test_code/InstantSplat/train.py` | Simplified to `train_gaussians()` function |
| `render/render.py` | `test_code/InstantSplat/render.py` | Simplified to `render_images()` function |
| `render/scene/__init__.py` | `test_code/InstantSplat/scene/__init__.py` | Scene management core |
| `render/utils/camera_utils.py` | `test_code/InstantSplat/utils/camera_utils.py` | Camera utilities with interpolation functions |

**Simplification Strategy:**
- Training: Accept COLMAP path, output model path
- Rendering: Accept model path + interpolation parameters, output image sequence
- Remove non-core features like tensorboard, GUI

### SEVA Migration

**Source Analysis:**
- `test_code/stable-virtual-camera/demo.py` - Main entry, ~415 lines
- Uses fire CLI, depends on internal `seva.*` modules

**Migration Content:**

| Destination | Source | Description |
|-------------|--------|-------------|
| `generate/infer.py` | `test_code/stable-virtual-camera/demo.py` | Simplified to `run_seva_inference()` function, using `img2img` task |
| `generate/geometry/camera_utils.py` | `test_code/stable-virtual-camera/seva/geometry/*.py` | Trajectory generation utilities |
| `generate/data_io/__init__.py` | `test_code/stable-virtual-camera/seva/data_io.py` | COLMAP format data loading |

**Simplification Strategy:**
- Remove fire CLI, convert to function calls
- Use `task="img2img"` + `traj_prior="interpolated"`
- Output: Image sequence + transforms.json
- Config parameters: guidance_scale, num_inference_steps, etc.

## Wrapper Integration Interface

### gen3r/gen3r_pipeline.py

```python
class Gen3RPipelineWrapper:
    def __init__(self, config: dict):
        self.model_path = Path(config.get("model_path", "test_code/Gen3R/checkpoints"))
        self.task = config.get("task", "allview")
        self.device = config.get("device", "cuda")

    def process(self, input_dir: str, output_dir: str, prompts: str,
                cameras: str = "free") -> dict:
        """
        Run Gen3R inference.

        Args:
            input_dir: Input image directory or video file
            output_dir: Output directory
            prompts: Text prompt
            cameras: Camera trajectory ("free", "zoom_in", "arc_left", etc.) or JSON file path

        Returns:
            dict: {"rgb_video": "path/to/rgb.mp4",
                   "pointcloud": "path/to/pcds.ply",
                   "cameras": "path/to/cameras.json"}
        """
        from gen3r.infer import run_gen3r_inference

        return run_gen3r_inference(
            frame_path=input_dir,
            prompts=prompts,
            output_dir=output_dir,
            task=self.task,
            cameras=cameras,
            pretrained_model_name_or_path=str(self.model_path),
            device=self.device
        )
```

### render/instant_splat_wrapper.py

```python
class InstantSplatWrapper:
    def __init__(self, config: dict):
        self.colmap_dir = config.get("colmap_dir")
        self.iterations = config.get("iterations", 3000)
        self.resolution = config.get("resolution", [1920, 1080])

    def train(self, colmap_dir: str, output_dir: str) -> str:
        """
        Train 3D Gaussian model.

        Returns:
            str: Model output directory
        """
        from render.train import train_gaussians

        return train_gaussians(
            colmap_dir=colmap_dir,
            output_dir=output_dir,
            iterations=self.iterations,
            resolution=self.resolution
        )

    def render_interpolated_video(self, model_dir: str, frames_per_pair: int,
                                  output_dir: str) -> dict:
        """
        Render interpolated video.

        Returns:
            dict: {"images": "path/to/images/", "video": "path/to/video.mp4"}
        """
        from render.render import render_interpolated

        return render_interpolated(
            model_dir=model_dir,
            frames_per_pair=frames_per_pair,
            output_dir=output_dir
        )
```

### generate/seva_generator.py

```python
class SevaGenerator:
    def __init__(self, config: dict):
        self.model_path = config.get("model_path", "stabilityai/stable-virtual-camera")
        self.guidance_scale = config.get("guidance_scale", 2.0)
        self.num_inference_steps = config.get("num_inference_steps", 50)

    def generate(self, colmap_dir: str, frames_per_pair: int,
                output_dir: str) -> dict:
        """
        Generate novel views using SEVA.

        Returns:
            dict: {"images": "path/to/images/", "transforms": "path/to/transforms.json"}
        """
        from generate.infer import run_seva_inference

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
```

## Implementation Plan

### Task Breakdown

| Task | Description | Files |
|------|-------------|-------|
| **T1** | Create gen3r/ subpackage structure | `gen3r/models/`, `gen3r/utils/`, `__init__.py` |
| **T2** | Migrate Gen3R core code | `gen3r/infer.py`, `gen3r/models/vggt/utils/pose_enc.py`, `gen3r/utils/*.py` |
| **T3** | Update Gen3RPipelineWrapper | `gen3r/gen3r_pipeline.py` integration |
| **T4** | Create render/ subpackage structure | `render/scene/`, `render/gaussian_renderer/`, `render/utils/`, `__init__.py` |
| **T5** | Migrate InstantSplat core code | `render/train.py`, `render/render.py`, `render/scene/dataset_readers.py`, `render/utils/camera_utils.py` |
| **T6** | Update InstantSplatWrapper | `render/instant_splat_wrapper.py` integration |
| **T7** | Create generate/ subpackage structure | `generate/data_io/`, `generate/geometry/`, `generate/model/`, `__init__.py` |
| **T8** | Migrate SEVA core code | `generate/infer.py`, `generate/geometry/camera_utils.py`, `generate/data_io/__init__.py` |
| **T9** | Update SevaGenerator | `generate/seva_generator.py` integration |
| **T10** | Update Stage integration | `pipeline/stages/output_stage.py`, `pipeline/stages/gen3r_stage.py` |
| **T11** | Testing validation | Run basic functionality tests for each module |

### Execution Order

```
T1-T3 (Gen3R) → T4-T6 (InstantSplat) → T7-T9 (SEVA) → T10-T11 (Integration)
```

Each module is completed independently before moving to the next, reducing risk.

## Risks and Considerations

1. **External Dependencies**: Gen3R and SEVA depend on HuggingFace models requiring network download
2. **CUDA Memory**: 3DGS training requires significant GPU memory
3. **Path Handling**: Ensure relative paths from test_code/ are correctly migrated
4. **Python Versions**: Some packages may have version requirements
5. **Model Checkpoints**: Gen3R model path needs to point to correct location
