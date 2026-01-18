# End-to-End Multi-View Reconstruction and Generation Pipeline

**Date:** 2026-01-18
**Status:** Design Approved
**Author:** Claude Code

---

## Overview

This document describes the design for a complete end-to-end pipeline that integrates multi-view image reconstruction and novel view generation. The pipeline orchestrates five modular stages with smart checkpointing, configurable rendering methods, and interpolation-based view generation.

### Goals

1. **End-to-end integration** - Single command from distorted input to rendered novel views
2. **Smart checkpointing** - Resume from any stage if outputs exist
3. **Configurable rendering** - Switch between Gaussian Splatting and diffusion-based synthesis
4. **Interpolation-based generation** - Generate novel views between existing camera poses
5. **Video output** - Compile rendered views into MP4 format

---

## Pipeline Architecture

### Stage Execution Order

```
input_images
    ↓
┌─────────────────┐
│  UndistortStage │ → undistorted/images + intrinsics.json
└─────────────────┘
    ↓
┌──────────────────────┐
│ ReconstructionStage  │ → COLMAP sparse + depth/normal maps
└──────────────────────┘
    ↓
┌─────────────────────────┐
│ GaussianSplattingStage │ → trained 3DGS model (.ply)
└─────────────────────────┘
    ↓
┌──────────────────────┐
│ ViewGenerationStage  │ → interpolated camera poses
└──────────────────────┘
    ↓
┌──────────────────┐
│  RenderingStage  │ → rendered images + output.mp4
└──────────────────┘
```

### Stage Output Data Structure

```python
class StageOutput:
    output_dir: str      # Directory containing stage outputs
    metadata: dict       # Stage-specific data for next stage
```

**Data flow between stages:**

| Stage | Input | Output | Key Metadata |
|-------|-------|--------|--------------|
| Undistort | Raw images | Undistorted images + intrinsics | `intrinsics_path`, `image_count` |
| Reconstruction | Undistorted images | COLMAP sparse + depth/normal maps | `method`, `camera_poses`, `image_size` |
| GaussianSplatting | COLMAP output | Trained 3DGS model (.ply) | `model_path`, `iteration_count` |
| ViewGeneration | Camera poses | Interpolated trajectory | `original_poses`, `new_poses`, `interp_factor` |
| Rendering | 3DGS model + poses | Rendered views + MP4 | `render_method`, `output_paths`, `video_path` |

---

## Stage Specifications

### 1. GaussianSplattingStage (New)

**Purpose:** Wrap existing `gs/train.py` functionality into a pipeline stage.

**Configuration:**
```json
{
  "gaussian_splatting": {
    "enabled": true,
    "output_dir": "gaussians",
    "iterations": 30000,
    "position_lr_init": 0.00016,
    "position_lr_final": 0.0000016,
    "feature_lr": 0.0025,
    "opacity_lr": 0.05,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
    "lambda_dssim": 0.2,
    "use_confidence": true,
    "save_interval": 10000
  }
}
```

**Responsibilities:**
- Load reconstruction output from `sparse/0/` and `results.pkl`
- Initialize Gaussian model from point cloud
- Train with confidence-weighted optimization if available
- Save final model to `{output_dir}/point_cloud/iteration_{N}/point_cloud.ply`
- Return metadata: `{"model_path": "...", "iteration_count": N}`

**Completion check:** `{output_dir}/point_cloud/iteration_{N}/point_cloud.ply` exists

**File:** `pipeline/stages/gaussian_splatting_stage.py`

---

### 2. ViewGenerationStage (New)

**Purpose:** Generate interpolated camera trajectories between existing views.

**Configuration:**
```json
{
  "view_generation": {
    "enabled": true,
    "output_dir": "novel_views",
    "interpolation_factor": 2,
    "trajectory_type": "linear",
    "smooth_poses": true
  }
}
```

**Responsibilities:**
- Read camera poses from COLMAP sparse output
- Sort poses by camera ID/timestamp
- Interpolate between consecutive poses:
  - Translation: linear interpolation
  - Rotation: SLERP (spherical linear interpolation)
  - Intrinsics: linear interpolation
- Write poses to JSON and COLMAP format

**Output structure:**
```
novel_views/
├── original_poses.json
├── interpolated_poses.json
└── transforms.json
```

**Completion check:** `{output_dir}/interpolated_poses.json` exists

**File:** `pipeline/stages/view_generation_stage.py`

---

### 3. RenderingStage (New)

**Purpose:** Render novel views using configurable method (3DGS or diffusion).

**Configuration:**
```json
{
  "rendering": {
    "enabled": true,
    "output_dir": "rendered_views",
    "method": "gaussian_splatting",
    "video": {
      "enabled": true,
      "fps": 30,
      "codec": "mp4v",
      "quality": "high"
    },
    "diffusion": {
      "model": "stable-virtual-camera",
      "steps": 50,
      "guidance_scale": 7.5
    },
    "gaussian_splatting": {
      "resolution": [512, 512],
      "render_background": true
    }
  }
}
```

**Method-specific behavior:**

*Gaussian Splatting mode:*
- Load trained 3DGS model from GaussianSplattingStage
- Render each interpolated pose using `gs.render.render()`
- Output to `{output_dir}/images/view_XXXX.jpg`

*Diffusion mode:*
- Use `test_code/stable-virtual-camera` (SEVA) pipeline
- Requires COLMAP format input
- Higher quality but slower rendering

**Video output:**
- Compile rendered images into `{output_dir}/output.mp4`
- Use OpenCV VideoWriter with specified FPS and codec

**Output structure:**
```
rendered_views/
├── images/
│   ├── view_0000.jpg
│   ├── view_0001.jpg
│   └── ...
├── output.mp4
└── metadata.json
```

**Completion check:** Image count matches expected view count + MP4 exists (if enabled)

**File:** `pipeline/stages/rendering_stage.py`

---

## Checkpoint and Resume System

### Resume Logic

```python
def run(self, input_dir: str, output_base_dir: str, resume: bool = True):
    for stage_name in self.DEFAULT_STAGE_ORDER:
        if stage_name not in self.stages:
            continue

        output_dir = Path(output_base_dir) / stage.config.get("output_dir", stage_name)

        # Check if stage output exists and is valid
        if resume and stage._check_complete(output_dir):
            self.logger.info(f"Skipping {stage_name}: outputs found at {output_dir}")
            self.stage_outputs[stage_name] = self._load_existing_output(output_dir)
            current_input = str(output_dir)
            continue

        # Run stage
        output = stage.run(current_input, str(output_dir))
        self.stage_outputs[stage_name] = output
        current_input = str(output_dir)
```

### Stage Completion Validation

| Stage | Completion Check |
|-------|------------------|
| Undistort | `undistorted/intrinsics.json` exists + N images match input |
| Reconstruction | `sparse/0/cameras.bin` + `results.pkl` exist |
| GaussianSplatting | `point_cloud/iteration_{N}/point_cloud.ply` exists |
| ViewGeneration | `interpolated_poses.json` exists with valid poses |
| Rendering | Output image count matches expected + MP4 exists (if enabled) |

---

## Configuration System

### Full Pipeline Configuration

**File:** `configs/full_pipeline.json`

```json
{
  "pipeline": {
    "resume": true,
    "force_stages": null,
    "log_level": "INFO"
  },
  "undistort": {
    "method": "geocalib",
    "output_dir": "undistorted",
    "intrinsics_path": "undistorted/intrinsics.json"
  },
  "reconstruction": {
    "method": "hunyuanworld",
    "device": "cuda",
    "output_dir": "reconstruction"
  },
  "gaussian_splatting": {
    "enabled": true,
    "output_dir": "gaussians",
    "iterations": 30000,
    "position_lr_init": 0.00016,
    "position_lr_final": 0.0000016,
    "use_confidence": true,
    "save_interval": 10000
  },
  "view_generation": {
    "enabled": true,
    "output_dir": "novel_views",
    "interpolation_factor": 2,
    "trajectory_type": "linear",
    "smooth_poses": true
  },
  "rendering": {
    "enabled": true,
    "output_dir": "rendered_views",
    "method": "gaussian_splatting",
    "video": {
      "enabled": true,
      "fps": 30,
      "codec": "mp4v",
      "quality": "high"
    }
  }
}
```

### Preset Configurations

**Fast preview** (`configs/fast.json`):
```json
{
  "reconstruction": {"method": "da3"},
  "gaussian_splatting": {"iterations": 5000},
  "rendering": {"method": "gaussian_splatting"}
}
```

**High quality** (`configs/high_quality.json`):
```json
{
  "reconstruction": {"method": "vggtx"},
  "gaussian_splatting": {"iterations": 60000},
  "rendering": {"method": "diffusion"}
}
```

**Dense views** (`configs/dense_view.json`):
```json
{
  "view_generation": {"interpolation_factor": 4},
  "rendering": {"method": "gaussian_splatting"}
}
```

---

## CLI Interface

### Usage Examples

```bash
# Full pipeline (with auto-resume)
python pipeline.py \
  --input-dir ./raw_images \
  --output-dir ./output \
  --config configs/full_pipeline.json

# Force re-run specific stages
python pipeline.py \
  --input-dir ./raw_images \
  --output-dir ./output \
  --resume \
  --force-stages gaussian_splatting,rendering

# Run only specific stages
python pipeline.py \
  --input-dir ./raw_images \
  --output-dir ./output \
  --stages view_generation,rendering

# Configure rendering method and interpolation
python pipeline.py \
  --input-dir ./raw_images \
  --output-dir ./output \
  --render-method diffusion \
  --interp-factor 3
```

### New CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--resume` | flag | Enable auto-resume from existing outputs |
| `--force-stages` | string | Comma-separated list of stages to force re-run |
| `--stages` | string | Comma-separated list of stages to run |
| `--render-method` | string | Rendering method: `gaussian_splatting` or `diffusion` |
| `--interp-factor` | int | Interpolation factor for view generation |
| `--video-fps` | int | FPS for output video (default: 30) |

---

## Output Directory Structure

```
output/
├── undistorted/
│   ├── images/
│   │   ├── 0000.jpg
│   │   └── ...
│   └── intrinsics.json
├── reconstruction/
│   ├── sparse/
│   │   └── 0/
│   │       ├── cameras.bin
│   │       ├── images.bin
│   │       ├── points3D.ply
│   │       └── confidence_dsp.npy
│   ├── images/
│   ├── depths/
│   ├── normals/
│   └── hunyuanworld/
│       └── results.pkl
├── gaussians/
│   ├── point_cloud/
│   │   └── iteration_30000/
│   │       └── point_cloud.ply
│   └── pose/
│       └── ours_30000/
│           └── pose_optimized.npy
├── novel_views/
│   ├── original_poses.json
│   ├── interpolated_poses.json
│   └── transforms.json
├── rendered_views/
│   ├── images/
│   │   ├── view_0000.jpg
│   │   ├── view_0001.jpg
│   │   └── ...
│   ├── output.mp4
│   └── metadata.json
└── pipeline.log
```

---

## Implementation Plan

### Phase 1: Core Pipeline Enhancements

- [ ] Update `Pipeline` class with resume/checkpoint logic
- [ ] Add `--resume`, `--force-stages`, `--stages` CLI arguments
- [ ] Create stage completion validation methods
- [ ] Update `BaseStage` with `_check_complete()` abstract method

**Files:**
- `pipeline/pipeline.py`
- `pipeline/base.py`
- `pipeline.py`

---

### Phase 2: GaussianSplattingStage

- [ ] Create `pipeline/stages/gaussian_splatting_stage.py`
- [ ] Wrap existing `gs/train.py` functionality
- [ ] Load from reconstruction output (COLMAP + results.pkl)
- [ ] Save trained model path in metadata
- [ ] Add to stage registry

**Files:**
- `pipeline/stages/gaussian_splatting_stage.py` (new)
- `pipeline/__init__.py`

---

### Phase 3: ViewGenerationStage

- [ ] Create `pipeline/stages/view_generation_stage.py`
- [ ] Load camera poses from COLMAP output
- [ ] Implement pose interpolation (SLERP + linear)
- [ ] Generate N interpolated poses
- [ ] Output poses in JSON format

**Files:**
- `pipeline/stages/view_generation_stage.py` (new)
- `pipeline/utils/pose_utils.py` (new)

---

### Phase 4: RenderingStage

- [ ] Create `pipeline/stages/rendering_stage.py`
- [ ] Implement Gaussian Splatting rendering mode
- [ ] Implement diffusion rendering mode
- [ ] Add video compilation (MP4 output)
- [ ] Save rendered images and metadata

**Files:**
- `pipeline/stages/rendering_stage.py` (new)
- `pipeline/utils/video_utils.py` (new)

---

### Phase 5: Configuration and Testing

- [ ] Create preset configurations
- [ ] Update `configs/README.md`
- [ ] End-to-end testing
- [ ] Documentation

**Files:**
- `configs/full_pipeline.json`
- `configs/fast.json`
- `configs/high_quality.json`
- `configs/dense_view.json`

---

## Dependencies

### New Dependencies

```txt
# Video processing (existing)
opencv-python>=4.8.0

# For pose interpolation
scipy>=1.11.0
```

### Existing Dependencies

No new major dependencies. The design leverages existing modules:
- `gs/` - Gaussian Splatting training and rendering
- `reconstruction/` - 3D reconstruction backends
- `test_code/stable-virtual-camera/` - Diffusion-based rendering

---

## Future Enhancements

Out of scope for this implementation but worth noting:

1. **Alternative trajectories** - Bezier curves, circular orbits, custom paths
2. **Parallel processing** - Multi-GPU training and rendering
3. **Quality metrics** - PSNR, LPIPS for rendered output validation
4. **Interactive viewer** - Real-time preview during rendering
5. **Sparse view support** - Integration with `InstantSplat` for few-view scenarios

---

## Appendix: Pose Interpolation Details

### SLERP for Rotation

```python
def slerp(q1, q2, t):
    """Spherical linear interpolation between quaternions."""
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        return (1 - t) * q1 + t * q2

    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    s1 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)
    s2 = np.sin(theta) / np.sin(theta_0)
    return s1 * q1 + s2 * q2
```

### Pose Interpolation Pipeline

```python
def interpolate_poses(poses, factor):
    """
    poses: List of (4x4) camera matrices
    factor: Number of views to insert between each pair
    Returns: List of interpolated (4x4) camera matrices
    """
    interpolated = []

    for i in range(len(poses) - 1):
        pose1, pose2 = poses[i], poses[i + 1]

        # Extract rotation and translation
        R1, t1 = pose1[:3, :3], pose1[:3, 3]
        R2, t2 = pose2[:3, :3], pose2[:3, 3]

        # Convert to quaternions
        q1 = RotationMatrixToQuaternion(R1)
        q2 = RotationMatrixToQuaternion(R2)

        # Insert original pose
        interpolated.append(pose1)

        # Generate interpolated poses
        for j in range(1, factor + 1):
            t = j / (factor + 1)
            q_interp = slerp(q1, q2, t)
            t_interp = (1 - t) * t1 + t * t2
            R_interp = QuaternionToRotationMatrix(q_interp)

            pose_interp = np.eye(4)
            pose_interp[:3, :3] = R_interp
            pose_interp[:3, 3] = t_interp
            interpolated.append(pose_interp)

    interpolated.append(poses[-1])
    return interpolated
```
