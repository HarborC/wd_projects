# Code Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate core code from test_code/ into git-tracked packages (gen3r/, render/, generate/) with subpackage structure.

**Architecture:** Selective migration of main workflow + minimal dependencies. Each module (Gen3R, InstantSplat, SEVA) will have its own subpackage with core inference/training/rendering functions exposed through wrapper classes.

**Tech Stack:** Python 3.10+, PyTorch, CUDA (for GPU acceleration), COLMAP format, HuggingFace models

---

## Task 1: Create gen3r/ subpackage structure

**Files:**
- Create: `gen3r/models/__init__.py`
- Create: `gen3r/models/vggt/__init__.py`
- Create: `gen3r/models/vggt/utils/__init__.py`
- Create: `gen3r/utils/__init__.py`
- Test: `tests/test_gen3r_imports.py`

**Step 1: Write the failing test**

Create `tests/test_gen3r_imports.py`:

```python
"""Test gen3r subpackage imports."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_gen3r_models_import():
    """Test that gen3r.models can be imported."""
    from gen3r.models import __init__ as models_init
    assert models_init is not None

def test_gen3r_models_vggt_import():
    """Test that gen3r.models.vggt can be imported."""
    from gen3r.models.vggt import __init__ as vggt_init
    assert vggt_init is not None

def test_gen3r_utils_import():
    """Test that gen3r.utils can be imported."""
    from gen3r.utils import __init__ as utils_init
    assert utils_init is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_gen3r_imports.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'gen3r.models'"

**Step 3: Write minimal implementation**

Create `gen3r/models/__init__.py`:
```python
"""Gen3R model-related modules."""
```

Create `gen3r/models/vggt/__init__.py`:
```python
"""VGGT model components."""
```

Create `gen3r/models/vggt/utils/__init__.py`:
```python
"""VGGT utility functions."""
```

Create `gen3r/utils/__init__.py`:
```python
"""Gen3R utility functions."""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_gen3r_imports.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add tests/test_gen3r_imports.py gen3r/models/ gen3r/utils/
git commit -m "feat(gen3r): create subpackage structure

Add gen3r/models/ and gen3r/utils/ subpackages with __init__.py files.
Tests verify imports work correctly.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Migrate Gen3R core code - pose_enc.py

**Files:**
- Create: `gen3r/models/vggt/utils/pose_enc.py`
- Source: `test_code/Gen3R/gen3r/models/vggt/utils/pose_enc.py`
- Test: `tests/test_gen3r_pose_enc.py`

**Step 1: Read source file**

Read the source file content:
```bash
cat /data/cjg/projects/wd_projects/test_code/Gen3R/gen3r/models/vggt/utils/pose_enc.py
```

**Step 2: Write the test**

Create `tests/test_gen3r_pose_enc.py`:

```python
"""Test pose_enc module."""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gen3r.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri

def test_pose_encoding_to_extri_intri():
    """Test pose encoding conversion."""
    # Create dummy pose encoding
    B, F = 1, 4
    pose_enc = torch.randn(B, F, 64, 48, 48)  # Dummy encoding
    image_size = (560, 560)

    # Call function
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_size)

    # Verify output shapes
    assert extrinsic.shape == (B, F, 3, 4), f"Expected extrinsic shape (B,F,3,4), got {extrinsic.shape}"
    assert intrinsic.shape == (B, F, 3, 3), f"Expected intrinsic shape (B,F,3,3), got {intrinsic.shape}"
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_gen3r_pose_enc.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'pose_enc'"

**Step 4: Copy and adapt implementation**

Copy from source to `gen3r/models/vggt/utils/pose_enc.py`.

The file should contain:
- `pose_encoding_to_extri_intri()` function
- Any helper functions it depends on
- All necessary imports (torch, etc.)

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_gen3r_pose_enc.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add gen3r/models/vggt/utils/pose_enc.py tests/test_gen3r_pose_enc.py
git commit -m "feat(gen3r): migrate pose_enc utility module

Add pose_encoding_to_extri_intri function for converting pose
encodings to extrinsic and intrinsic camera matrices.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Migrate Gen3R core code - common_utils.py

**Files:**
- Create: `gen3r/utils/common_utils.py`
- Source: `test_code/Gen3R/gen3r/utils/common_utils.py`
- Test: `tests/test_gen3r_common_utils.py`

**Step 1: Read source file**

```bash
cat /data/cjg/projects/wd_projects/test_code/Gen3R/gen3r/utils/common_utils.py
```

**Step 2: Write the test**

Create `tests/test_gen3r_common_utils.py`:

```python
"""Test common_utils module."""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gen3r.utils.common_utils import colorize_depth_map

def test_colorize_depth_map():
    """Test depth map colorization."""
    # Create dummy depth map
    depth = torch.randn(1, 100, 100, 1)

    # Call function
    colored = colorize_depth_map(depth)

    # Verify output shape
    assert colored.shape[-1] == 3, f"Expected RGB output, got shape {colored.shape}"
    assert colored.shape[0:3] == depth.shape[0:3], "Spatial dimensions should match"
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_gen3r_common_utils.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'common_utils'"

**Step 4: Copy and adapt implementation**

Copy from source to `gen3r/utils/common_utils.py`.

Include functions:
- `colorize_depth_map()`
- `downsample_and_save_pointcloud()`
- `save_videos_grid()`

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_gen3r_common_utils.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add gen3r/utils/common_utils.py tests/test_gen3r_common_utils.py
git commit -m "feat(gen3r): migrate common_utils module

Add utilities for depth visualization, point cloud saving,
and video grid generation.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Migrate Gen3R core code - data_utils.py

**Files:**
- Create: `gen3r/utils/data_utils.py`
- Source: `test_code/Gen3R/gen3r/utils/data_utils.py`
- Test: `tests/test_gen3r_data_utils.py`

**Step 1: Read source file**

```bash
cat /data/cjg/projects/wd_projects/test_code/Gen3R/gen3r/utils/data_utils.py
```

**Step 2: Write the test**

Create `tests/test_gen3r_data_utils.py`:

```python
"""Test data_utils module."""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gen3r.utils.data_utils import center_crop, compute_rays, preprocess_poses

def test_center_crop():
    """Test center crop function."""
    # Create dummy image
    img = torch.randn(1, 3, 640, 480)

    # Crop to square
    cropped = center_crop(img, (560, 560))

    # Verify output shape
    assert cropped.shape == (1, 3, 560, 560), f"Expected shape (1,3,560,560), got {cropped.shape}"

def test_preprocess_poses():
    """Test pose preprocessing."""
    # Create dummy poses [B, 4, 4]
    poses = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)

    # Preprocess
    processed = preprocess_poses(poses)

    # Verify output shape
    assert processed.shape == (2, 4, 4), f"Expected shape (2,4,4), got {processed.shape}"
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_gen3r_data_utils.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'data_utils'"

**Step 4: Copy and adapt implementation**

Copy from source to `gen3r/utils/data_utils.py`.

Include functions:
- `center_crop()`
- `compute_rays()`
- `preprocess_poses()`

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_gen3r_data_utils.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add gen3r/utils/data_utils.py tests/test_gen3r_data_utils.py
git commit -m "feat(gen3r): migrate data_utils module

Add utilities for data preprocessing: center crop, ray computation,
and pose preprocessing.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Migrate Gen3R infer.py

**Files:**
- Create: `gen3r/infer.py`
- Source: `test_code/Gen3R/infer.py`
- Modify: `gen3r/gen3r_pipeline.py`
- Test: `tests/test_gen3r_infer.py`

**Step 1: Read source file**

```bash
cat /data/cjg/projects/wd_projects/test_code/Gen3R/infer.py
```

**Step 2: Write the test**

Create `tests/test_gen3r_infer.py`:

```python
"""Test Gen3R inference."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_run_gen3r_inference_signature():
    """Test that run_gen3r_inference has correct signature."""
    from gen3r.infer import run_gen3r_inference
    import inspect

    sig = inspect.signature(run_gen3r_inference)
    params = list(sig.parameters.keys())

    required_params = ['frame_path', 'prompts', 'output_dir', 'task', 'cameras',
                       'pretrained_model_name_or_path', 'device']
    for param in required_params:
        assert param in params, f"Missing required parameter: {param}"
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_gen3r_infer.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'gen3r.infer'"

**Step 4: Copy and adapt implementation**

Create `gen3r/infer.py` with `run_gen3r_inference()` function:

```python
"""Gen3R inference module."""
import os
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from einops import rearrange
from torchvision.transforms.functional import resize
from accelerate.utils import set_seed

# Import from migrated modules
from gen3r.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from gen3r.utils.data_utils import center_crop, compute_rays, preprocess_poses
from gen3r.utils.common_utils import colorize_depth_map, downsample_and_save_pointcloud, save_videos_grid

# Import Gen3R pipeline from test_code (will be migrated later if needed)
import sys
test_code_gen3r = Path(__file__).parent.parent.parent / "test_code" / "Gen3R"
if str(test_code_gen3r) not in sys.path:
    sys.path.insert(0, str(test_code_gen3r))

from gen3r.pipeline import Gen3RPipeline


def get_poses(cam_type='', num_frames=49, scene_scale=1.0):
    """Generate camera poses for built-in trajectories."""
    import math

    F = int(num_frames)
    if F <= 0:
        raise ValueError("num_frames must be > 0")

    poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(F, 1, 1)

    def rot_y(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32)

    def rot_x(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float32)

    if cam_type == 'zoom_in' or cam_type == 'zoom_out':
        end = 1.0 * float(scene_scale)
        if cam_type == 'zoom_out':
            end = -end
        zs = torch.linspace(0.0, end, steps=F, dtype=torch.float32)
        for i in range(F):
            poses[i, 2, 3] = zs[i]

    elif cam_type == 'arc_right' or cam_type == 'arc_left':
        end_angle = math.radians(15)
        if cam_type == 'arc_left':
            end_angle = -end_angle
        angles = torch.linspace(0.0, end_angle, steps=F, dtype=torch.float32)
        for i in range(F):
            R = rot_y(float(angles[i]))
            poses[i, :3, :3] = R

    elif cam_type == 'translate_up' or cam_type == 'translate_down':
        end_angle = math.radians(15.0)
        if cam_type == 'translate_down':
            end_angle = -end_angle
        angles = torch.linspace(0.0, end_angle, steps=F, dtype=torch.float32)
        for i in range(F):
            R = rot_x(float(angles[i]))
            poses[i, :3, :3] = R

    elif cam_type == 'free':
        pass
    else:
        raise NotImplementedError(f"Camera type {cam_type} not implemented.")

    return poses


def save_results(output_dir, results, prompts):
    """Save Gen3R results."""
    import json
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    for key, value in results.items():
        if key == "rgbs" and value is not None:
            rgb = rearrange(value, "b f h w c -> b c f h w").float().cpu()
            filename = os.path.join(output_dir, f"rgb.mp4")
            save_videos_grid(rgb, filename, rescale=False)

        elif key == "pcds":
            downsample_and_save_pointcloud(
                os.path.join(output_dir, f"pcds.ply"),
                value[0],
                results['rgbs'][0],
                voxel_size=0.005,
                filter_outliers=True,
                depth_percentile=0,
                remove_far_points=False,
            )

        elif key == 'cameras':
            filename = os.path.join(output_dir, f"cameras.json")
            extrinsics, Ks = value
            extrinsics = torch.cat(
                [extrinsics,
                 torch.tensor([0, 0, 0, 1], device=extrinsics.device).view(1, 1, 1, 4).repeat(
                     extrinsics.shape[0], extrinsics.shape[1], 1, 1)], dim=2)

            cameras = {
                'extrinsics': extrinsics[0].float().cpu().numpy().tolist(),
                'intrinsics': Ks[0].float().cpu().numpy().tolist(),
            }
            with open(filename, "w") as f:
                json.dump(cameras, f, indent=4)

    # Save prompts
    filename = os.path.join(output_dir, f"prompts.txt")
    with open(filename, "w") as f:
        f.write(prompts[0])


def run_gen3r_inference(frame_path, prompts, output_dir, task='allview',
                        cameras='free', pretrained_model_name_or_path='./checkpoints',
                        device='cuda'):
    """
    Run Gen3R inference.

    Args:
        frame_path: Path to input images or video
        prompts: Text prompt
        output_dir: Output directory
        task: Task type ('1view', '2view', 'allview')
        cameras: Camera trajectory or JSON file path
        pretrained_model_name_or_path: Path to model checkpoints
        device: Device to run on

    Returns:
        dict with paths to output files
    """
    import imageio
    import numpy as np
    from datetime import datetime

    BUILTIN_CAMERA_TRAJECTORIES = ["zoom_in", "zoom_out", "arc_left", "arc_right",
                                    "translate_up", "translate_down", "free"]

    # Create output directory
    output_dir = os.path.join(output_dir, f'{task}', f'{datetime.now().strftime("%Y-%m-%d-%H-%M-00")}')
    os.makedirs(output_dir, exist_ok=True)

    set_seed(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load pipeline
    pipeline = Gen3RPipeline.from_pretrained(pretrained_model_name_or_path)
    pipeline.to(device).to(torch.bfloat16)

    # Load control images
    control_images = []
    if not task == 'allview' or (task == 'allview' and os.path.isdir(frame_path)):
        if task == 'allview':
            frame_path = sorted([os.path.join(frame_path, f) for f in os.listdir(frame_path)
                                if f.endswith('.png') or f.endswith('.jpg')])
        for fp in frame_path if isinstance(frame_path, list) else [frame_path]:
            frame = torch.from_numpy(imageio.v2.imread(fp))[..., :3]
            control_images.append(frame)
        control_images = torch.stack(control_images).to(device, torch.bfloat16)
    else:
        control_images = torch.from_numpy(imageio.v3.imread(frame_path))

    if task == '1view':
        control_images = control_images[:1, ...]
    elif task == '2view':
        control_images = control_images[[0, -1], ...]
    else:
        control_images = control_images[:49, ...]

    control_images = control_images.permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
    frame_height, frame_width = control_images.shape[3], control_images.shape[4]
    scale = 560 / min(frame_height, frame_width)
    new_height, new_width = round(frame_height * scale), round(frame_width * scale)
    control_images = resize(control_images[0], [new_height, new_width])
    control_images = center_crop(control_images, (560, 560))[None, ...].to(device, torch.bfloat16)

    # Handle cameras
    if cameras in BUILTIN_CAMERA_TRAJECTORIES:
        print("Using built-in camera trajectory:", cameras)
        with torch.no_grad():
            aggregated_token_list, ps_idx = pipeline.vggt.aggregator(control_images)
            aggregated_token_list = [aggregated_token_list[i]
                                     for i in pipeline.vggt.depth_head.intermediate_layer_idx]
            pose_enc = pipeline.vggt.camera_head(aggregated_token_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, control_images.shape[-2:])
            depth_maps, _ = pipeline.vggt.depth_head(aggregated_token_list, control_images, ps_idx)

        Ks = intrinsic[:, :1].repeat(1, 49, 1, 1)
        scene_scale = 0.8 * torch.median(depth_maps).item()
        c2ws = get_poses(cam_type=cameras, num_frames=49, scene_scale=scene_scale).to(device)
    else:
        import json
        with open(cameras, "r") as f:
            cameras_data = json.load(f)
        extrinsics = torch.from_numpy(np.array(cameras_data['extrinsics']))[:49].to(device)
        Ks = torch.from_numpy(np.array(cameras_data['intrinsics']))[None, :49].to(device)
        c2ws = torch.linalg.inv(extrinsics)

    c2ws = preprocess_poses(c2ws)[None, ...]

    # Compute plucker embeddings
    plucker_embeddings_list = []
    for i in range(len(c2ws)):
        rays_o, rays_d = compute_rays(c2ws[i], Ks[i], h=560, w=560, device=device)
        o_cross_d = torch.cross(rays_o, rays_d, dim=1)
        plucker_embeddings = torch.cat([o_cross_d, rays_d], dim=1)
        plucker_embeddings_list.append(plucker_embeddings)
    plucker_embeddings = torch.stack(plucker_embeddings_list, dim=0)
    if cameras == 'free' or task == 'allview':
        plucker_embeddings = torch.zeros_like(plucker_embeddings)

    # Load prompts
    if os.path.isfile(prompts):
        with open(prompts, "r", encoding="utf-8") as file:
            prompts = file.readlines()[0]
    else:
        prompts = prompts

    # Run inference
    sample = pipeline(
        prompt=prompts,
        control_cameras=plucker_embeddings,
        control_images=control_images,
        num_frames=49,
        negative_prompt="bad detailed",
        height=560,
        width=560,
        guidance_scale=5,
        return_dict=True,
        min_max_depth_mask=True,
    )

    artifacts = {
        'rgbs': sample.rgbs,
        'depth_maps': sample.depth_maps,
        "pcds": sample.pcds,
        'point_masks': sample.point_masks,
        'cameras': sample.cameras,
    }

    save_results(output_dir, artifacts, [prompts])

    return {
        "rgb_video": os.path.join(output_dir, "rgb.mp4"),
        "pointcloud": os.path.join(output_dir, "pcds.ply"),
        "cameras": os.path.join(output_dir, "cameras.json"),
        "output_dir": output_dir,
    }
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_gen3r_infer.py -v`
Expected: PASS

**Step 6: Update Gen3RPipelineWrapper**

Modify `gen3r/gen3r_pipeline.py`:

```python
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
```

**Step 7: Commit**

```bash
git add gen3r/infer.py gen3r/gen3r_pipeline.py tests/test_gen3r_infer.py
git commit -m "feat(gen3r): migrate infer.py and update wrapper

Add run_gen3r_inference() function with support for:
- 1view, 2view, allview tasks
- Built-in camera trajectories or custom JSON
- RGB video, point cloud, and camera output

Update Gen3RPipelineWrapper.process() to call the migrated function.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6-11: InstantSplat and SEVA Migration

[Remaining tasks follow similar pattern - continue with T6-T11 for render/ and generate/ packages]

### Task 6: Create render/ subpackage structure

**Files:**
- Create: `render/scene/__init__.py`, `render/gaussian_renderer/__init__.py`, `render/utils/__init__.py`
- Test: `tests/test_render_imports.py`

[Follow same pattern as Task 1]

### Task 7: Migrate InstantSplat camera_utils.py

**Files:**
- Create: `render/utils/camera_utils.py`
- Source: `test_code/InstantSplat/utils/camera_utils.py`
- Test: `tests/test_render_camera_utils.py`

[Focus on `generate_interpolated_path()` function]

### Task 8: Migrate InstantSplat train.py

**Files:**
- Create: `render/train.py`
- Source: `test_code/InstantSplat/train.py`
- Test: `tests/test_render_train.py`

[Simplify to `train_gaussians()` function]

### Task 9: Migrate InstantSplat render.py

**Files:**
- Create: `render/render.py`
- Source: `test_code/InstantSplat/render.py`
- Test: `tests/test_render_render.py`

[Simplify to `render_interpolated()` function]

### Task 10: Create generate/ subpackage structure

**Files:**
- Create: `generate/data_io/__init__.py`, `generate/geometry/__init__.py`, `generate/model/__init__.py`
- Test: `tests/test_generate_imports.py`

### Task 11: Migrate SEVA code

**Files:**
- Create: `generate/infer.py`
- Create: `generate/geometry/camera_utils.py`
- Source: `test_code/stable-virtual-camera/`
- Test: `tests/test_generate_infer.py`

[Simplify to `run_seva_inference()` with img2img task]

---

## Summary

Total tasks: 11
- Gen3R: Tasks 1-5 (subpackage structure, utils, infer.py)
- InstantSplat: Tasks 6-9 (subpackage structure, utils, train.py, render.py)
- SEVA: Tasks 10-11 (subpackage structure, infer.py)

Each task follows TDD: test first, implement, verify, commit.
