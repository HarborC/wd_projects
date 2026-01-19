#!/usr/bin/env python3
"""
Test InstantSplat train and render functionality.

This script tests the integration with InstantSplat using the
test_reconstruction_output data.
"""
import sys
from pathlib import Path
import os
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")

    # Check for required packages
    try:
        import torch
        print(f"  ✅ torch {torch.__version__}")
    except ImportError:
        print("  ❌ torch not found")
        return False

    try:
        import diff_gaussian_rasterization
        print("  ✅ diff-gaussian-rasterization")
    except ImportError:
        print("  ❌ diff-gaussian-rasterization not found")
        return False

    try:
        from gaussian_splat.gaussian_model import GaussianModel
        print("  ✅ InstantSplat (local)")
    except ImportError as e:
        print(f"  ❌ InstantSplat import failed: {e}")
        return False

    return True


def check_reconstruction_data():
    """Check if reconstruction data is available."""
    base_dir = Path("./test_reconstruction_output")
    required_files = [
        base_dir / "sparse" / "0" / "cameras.txt",
        base_dir / "sparse" / "0" / "images.txt",
        base_dir / "sparse" / "0" / "points3D.txt",
        base_dir / "images",
    ]

    missing = []
    for f in required_files:
        if not f.exists():
            missing.append(f)

    if missing:
        print(f"❌ Missing required files:")
        for f in missing:
            print(f"   - {f}")
        return False

    print(f"✅ Reconstruction data found at: {base_dir}")
    return True


def prepare_instantsplat_data(reconstruction_dir: Path, source_dir: Path, model_dir: Path):
    """
    Prepare data in the format expected by InstantSplat.
    """
    print(f"\nPreparing InstantSplat data format...")

    # Create source directories
    n_views = 6
    sparse_dir = source_dir / f"sparse_{n_views}" / "0"
    images_dir = source_dir / "images"

    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    src_images = reconstruction_dir / "images"
    if src_images.exists():
        for img in src_images.glob("*.png"):
            shutil.copy(img, images_dir / img.name)
        print(f"  Copied {len(list(images_dir.glob('*.png')))} images")

    # Copy COLMAP sparse files
    src_sparse = reconstruction_dir / "sparse" / "0"
    if src_sparse.exists():
        for file in ["cameras.txt", "images.txt", "points3D.txt"]:
            src_file = src_sparse / file
            if src_file.exists():
                shutil.copy(src_file, sparse_dir / file)
        print(f"  Copied COLMAP sparse files")

    # Create PLY file with normals
    src_ply = src_sparse / "points3D.ply"
    dst_ply = sparse_dir / "points3D.ply"

    # Use local dataset readers
    from gaussian_splat.utils.dataset_readers import read_points3D_text, storePly
    import numpy as np

    txt_path = src_sparse / "points3D.txt"
    xyz, rgb, _ = read_points3D_text(str(txt_path))
    storePly(str(dst_ply), xyz, rgb)
    print(f"  ✅ Created PLY file with normals ({len(xyz)} points)")

    # Check for confidence file
    confidence_src = reconstruction_dir / "sparse" / "0" / "confidence_dsp.npy"
    if confidence_src.exists():
        shutil.copy(confidence_src, sparse_dir / "confidence_dsp.npy")
        print(f"  ✅ Copied confidence file")
    else:
        # Create dummy confidence file
        dummy_confidence = np.zeros((512, 512), dtype=np.float32)
        np.save(sparse_dir / "confidence_dsp.npy", dummy_confidence)
        print(f"  ⚠️  Created dummy confidence file for testing")

    print(f"✅ Data prepared:")
    print(f"   Source: {source_dir}")
    print(f"   Model:  {model_dir}")
    return source_dir, model_dir


def test_train():
    """Test InstantSplat training."""
    print("\n" + "=" * 60)
    print("Test 1: InstantSplat Training")
    print("=" * 60)

    from gaussian_splat.train import train_gaussians

    # Prepare data
    reconstruction_dir = Path("./test_reconstruction_output")
    test_output_dir = Path("./test_instantsplat_output")
    source_dir = test_output_dir / "source"
    model_dir = test_output_dir / "model"

    source_path, model_path = prepare_instantsplat_data(reconstruction_dir, source_dir, model_dir)

    # Training parameters (small for testing)
    iterations = 1000

    print(f"\n🎯 Starting training...")
    print(f"  Source: {source_path}")
    print(f"  Model: {model_path}")
    print(f"  Iterations: {iterations}")

    try:
        source_path_abs = source_path.resolve()
        model_path_abs = model_path.resolve()

        result = train_gaussians(
            model_path=str(model_path_abs),
            source_path=str(source_path_abs),
            iterations=iterations,
            resolution=1,
            white_background=False,
            sh_degree=0,
            densify_until_iter=0,
            densify_from_iter=500,
            densification_interval=100,
            optim_pose=True,
            pp_optimizer=False,
            random_background=False,
            n_views=6,
            save_iterations=[iterations],
            test_iterations=[iterations],
            checkpoint_iterations=[],
            quiet=False,
        )

        print(f"\n✅ Training complete!")
        print(f"   Result: {result}")

        if (model_path / "point_cloud").exists():
            print(f"   ✅ Point cloud saved")
        if (model_path / "cfg_args").exists():
            print(f"   ✅ Config saved")

        return model_path

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_render(model_path: Path):
    """Test InstantSplat rendering."""
    print("\n" + "=" * 60)
    print("Test 2: InstantSplat Rendering")
    print("=" * 60)

    if model_path is None or not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("   Skipping render test")
        return None

    from gaussian_splat.render import render_interpolated

    print(f"\n🎯 Starting rendering...")
    print(f"  Model: {model_path}")

    try:
        result = render_interpolated(
            model_path=str(model_path),
            iteration=-1,
            n_views=6,
            n_interp_per_view=10,
            fps=30,
            skip_train=True,
            skip_test=False,
            optim_test_pose=False,
            quiet=False,
        )

        print(f"\n✅ Rendering complete!")
        print(f"   Result: {result}")

        if result.exists():
            print(f"   ✅ Video created: {result.name}")
            size_mb = result.stat().st_size / (1024 * 1024)
            print(f"   Size: {size_mb:.2f} MB")

        return result

    except Exception as e:
        print(f"\n❌ Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing InstantSplat Integration")
    print("=" * 60)

    # Check prerequisites
    if not check_dependencies():
        return 1

    if not check_reconstruction_data():
        return 1

    # Test 1: Training
    model_path = test_train()

    # Test 2: Rendering
    if model_path:
        test_render(model_path)

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("For full testing, ensure:")
    print("  1. All dependencies are installed")
    print("  2. Reconstruction data is complete")
    print("  3. Sufficient GPU memory is available")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
