# 3D Vision Pipeline

A comprehensive 3D computer vision pipeline that combines image processing, 3D reconstruction, and novel view synthesis. This modular system enables end-to-end processing from distorted input images to rendered 3D scenes.

## Overview

This project provides a complete pipeline for:
1. **Image Undistortion** - Camera calibration and lens distortion correction
2. **3D Reconstruction** - Multiple state-of-the-art backends for scene geometry
3. **Gaussian Splatting** - High-quality real-time rendering
4. **Novel View Synthesis** - Generate new camera views using diffusion models

## Project Structure

```
wd_projects/
├── pipeline.py                    # Main workflow orchestrator
├── undistortion/                  # Camera calibration & undistortion
│   ├── geocalib_undistorter.py
│   ├── anycalib_undistorter.py
│   └── ...
├── reconstruction/                # 3D reconstruction engines
│   ├── factory.py                 # Backend factory
│   ├── base_reconstructor.py      # Abstract base class
│   ├── da3_reconstructor.py       # Depth Anything 3
│   ├── mast3r_reconstructor.py    # MASt3R multi-view stereo
│   ├── hunyuanworld_reconstructor.py  # HunyuanWorld Mirror
│   ├── depth_anything_3/          # DA3 implementation
│   ├── mast3r/                    # MASt3R implementation
│   └── HunyuanWorld-Mirror/       # HunyuanWorld implementation
├── render/                        # Gaussian Splatting pipeline
│   ├── pipeline_difix.py
│   ├── simple_deblur_difix.py
│   ├── scoring_model.py
│   ├── bad_gaussians/             # 3DGS implementation
│   └── pyproject.toml
├── recon_to_seva_converter.py     # COLMAP to SEVA converter
├── test_code/                     # Demo implementations
│   ├── stable-virtual-camera/     # SEVA diffusion model
│   └── InstantSplat/              # Sparse-view 3DGS
├── requirements.txt
└── RECON_TO_SEVA_README.md
```

## Features

### Multi-Backend Support

Choose from state-of-the-art 3D reconstruction methods:

| Backend | Description |
|---------|-------------|
| **DA3** | Depth Anything 3 - Deep learning-based depth estimation |
| **MASt3R** | Multi-view stereo with attention mechanisms |
| **HunyuanWorld** | Tencent's Mirror 3D reconstruction model |

### Pipeline Workflow

1. **Undistortion**: Correct lens distortion and extract camera intrinsics
2. **Reconstruction**: Generate 3D scene geometry
3. **Conversion**: Transform outputs to various formats (COLMAP, SEVA)
4. **Rendering**: Gaussian Splatting for novel view synthesis

## Installation

```bash
pip install -r requirements.txt
```

### Key Dependencies

- PyTorch
- OpenCV
- COLMAP
- Diffusers (for SEVA)

## Usage

### Basic Pipeline

```python
from pipeline import Pipeline

# Initialize pipeline with desired backend
pipeline = Pipeline(
    undistorter="geocalib",
    reconstructor="da3"  # or "mast3r", "hunyuanworld"
)

# Run pipeline
pipeline.run(input_images, output_dir)
```

### Reconstruction to SEVA

```bash
python recon_to_seva_converter.py \
    --colmap_dir /path/to/colmap \
    --output_dir /path/to/seva \
    --target_views 32
```

## Documentation

- [RECON_TO_SEVA_README.md](RECON_TO_SEVA_README.md) - Detailed SEVA conversion guide

## License

This project integrates multiple research implementations. Please refer to individual submodules for their specific licenses.
