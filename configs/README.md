# Pipeline Configuration Files

This directory contains example configuration files for the multi-view reconstruction pipeline.

## Available Configurations

| File | Undistort | Reconstruction | Description |
|------|-----------|----------------|-------------|
| `default.json` | geocalib | da3 | Default configuration |
| `hunyuanworld.json` | geocalib | hunyuanworld | HunyuanWorld Mirror |
| `mast3r.json` | geocalib | mast3r | MASt3R multi-view stereo |
| `vggtx.json` | anycalib | vggtx | VGGT-X with AnyCalib |

## Usage

```bash
# Run with specific config
python pipeline.py \
    --input-dir ./data/images \
    --output-dir ./output \
    --config configs/hunyuanworld.json
```

## Configuration Format

```json
{
  "undistort": {
    "method": "geocalib",  // or "anycalib"
    "output_dir": "undistorted",
    "intrinsics_path": "undistorted/intrinsics.json"
  },
  "reconstruction": {
    "method": "hunyuanworld",  // "da3", "mast3r", "vggtx"
    "device": "cuda",  // or "cpu"
    "output_dir": "reconstruction"
  }
}
```

## Supported Methods

### Undistortion
- `geocalib` - GeoCalib undistortion
- `anycalib` - AnyCalib undistortion

### Reconstruction
- `da3` - Depth Anything 3
- `mast3r` - MASt3R multi-view stereo
- `hunyuanworld` - HunyuanWorld Mirror
- `vggtx` - VGGT-X
