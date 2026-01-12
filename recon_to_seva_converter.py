"""
Convert Reconstruction COLMAP output to SEVA format.

This script converts the output from Reconstruction module (MASt3R/DA3/HunyuanWorld)
to the format required by Stable Virtual Camera (SEVA).

Input (from Reconstruction):
    output_dir/
        images/           # Input images
        sparse/0/         # COLMAP format
            cameras.bin   # Camera intrinsics
            images.bin    # Camera extrinsics
        depths/           # Optional: depth maps
        normals/          # Optional: normal maps

Output (for SEVA):
    seva_output_dir/
        transforms.json           # Camera parameters
        train_test_split_1.json    # Single-view split
        train_test_split_3.json    # 3-view split
        train_test_split_6.json    # 6-view split
        train_test_split_9.json    # 9-view split
        images/                    # Symlink or copy of images

Usage:
    python recon_to_seva_converter.py \
        --recon_output /path/to/reconstruction/output \
        --seva_output /path/to/seva/output \
        --num_splits 1,3,6,9
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Try importing pycolmap for reading COLMAP format
try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False
    logging.warning("pycolmap not available, will try reading binary format directly")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_colmap_text_model(sparse_dir: Path):
    """
    Read COLMAP text model format.

    This reads the .txt format which is more compatible across COLMAP versions.

    Args:
        sparse_dir: Path to sparse directory (e.g., output/sparse/0)

    Returns:
        Tuple of (cameras dict, images dict)
    """
    logger.info(f"Reading COLMAP text model from {sparse_dir}")

    # Read cameras
    cameras_file = sparse_dir / "cameras.txt"
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = {
                'id': cam_id,
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    logger.info(f"Read {len(cameras)} cameras")

    # Read images
    images_file = sparse_dir / "images.txt"
    images = {}
    with open(images_file, 'r') as f:
        while True:
            # First line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            line1 = f.readline()
            if not line1:
                break
            line1 = line1.strip()
            if line1.startswith('#') or not line1:
                continue

            parts = line1.split()
            img_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cam_id = int(parts[8])
            name = parts[9]

            # Second line: POINTS2D (skip for our purpose)
            f.readline()

            # Convert quaternion to rotation matrix
            # COLMAP uses qw, qx, qy, qz format (world to camera)
            import roma
            import torch
            quat = torch.tensor([qw, qx, qy, qz])
            R = roma.unitquat_to_rotmat(quat).numpy()

            # Build 4x4 transformation matrix (world to camera)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = [tx, ty, tz]

            images[img_id] = {
                'id': img_id,
                'name': name,
                'cam_id': cam_id,
                'transform_matrix': transform_matrix
            }

    logger.info(f"Read {len(images)} images")
    return cameras, images


def extract_common_intrinsics(cameras: Dict) -> Dict:
    """
    Extract common intrinsics from cameras.

    Args:
        cameras: Dict of camera info from read_colmap_text_model

    Returns dict with: w, h, fl_x, fl_y, cx, cy, k1, k2, p1, p2
    """
    if not cameras:
        raise ValueError("No cameras found")

    # Get first camera as reference
    cam = next(iter(cameras.values()))

    intrinsics = {
        "w": cam['width'],
        "h": cam['height'],
        "fl_x": cam['params'][0],  # focal length x
        "fl_y": cam['params'][1],  # focal length y
        "cx": cam['params'][2],    # principal point x
        "cy": cam['params'][3],    # principal point y
    }

    # Add distortion parameters if available
    if len(cam['params']) > 4:
        intrinsics["k1"] = cam['params'][4]
    if len(cam['params']) > 5:
        intrinsics["k2"] = cam['params'][5]
    if len(cam['params']) > 6:
        intrinsics["p1"] = cam['params'][6]
    if len(cam['params']) > 7:
        intrinsics["p2"] = cam['params'][7]

    # Add camera model
    intrinsics["camera_model"] = cam['model']

    logger.info(f"Intrinsics: {intrinsics['w']}x{intrinsics['h']}, fx={intrinsics['fl_x']:.2f}, fy={intrinsics['fl_y']:.2f}")
    return intrinsics


def reconstruction_to_transforms(images: Dict, intrinsics: Dict) -> Dict:
    """
    Convert COLMAP data to SEVA transforms.json format.

    Args:
        images: Dict of image info from read_colmap_text_model
        intrinsics: Common intrinsics dict

    Returns:
        transforms dict compatible with SEVA
    """
    frames = []

    # Sort images by image_id to ensure consistent ordering
    sorted_images = sorted(images.items(), key=lambda x: x[0])

    for img_id, img_data in sorted_images:
        frame_data = {
            "file_path": f"images/{img_data['name']}",
            "transform_matrix": img_data['transform_matrix'].tolist(),
            "colmap_im_id": img_id
        }
        frames.append(frame_data)

    transforms = {
        **intrinsics,
        "frames": frames
    }

    return transforms


def interpolate_camera_poses(images: Dict, num_interp: int = 10) -> Tuple[Dict, List[int]]:
    """
    Interpolate camera poses between existing images.

    Args:
        images: Dict of image info from read_colmap_text_model
        num_interp: Number of interpolated poses per pair of consecutive images

    Returns:
        Tuple of (all_images dict including interpolated, test_ids list)
    """
    logger.info(f"Interpolating {num_interp} poses between each pair of consecutive images")

    # Sort images by ID
    sorted_images = sorted(images.items(), key=lambda x: x[0])

    all_images = {}
    test_ids = []

    next_id = max(images.keys()) + 1

    for i in range(len(sorted_images) - 1):
        img_id_1, img_data_1 = sorted_images[i]
        img_id_2, img_data_2 = sorted_images[i + 1]

        # Add first image to all_images
        all_images[img_id_1] = img_data_1

        # Interpolate between img_1 and img_2
        for j in range(1, num_interp + 1):
            t = j / (num_interp + 1)

            # Interpolate transform matrices
            transform_1 = img_data_1['transform_matrix']
            transform_2 = img_data_2['transform_matrix']

            # SLERP-like interpolation for rotation, linear for translation
            import roma
            import torch

            # Extract rotations and translations
            R1 = torch.from_numpy(transform_1[:3, :3]).float()
            R2 = torch.from_numpy(transform_2[:3, :3]).float()
            t1 = torch.from_numpy(transform_1[:3, 3]).float()
            t2 = torch.from_numpy(transform_2[:3, 3]).float()

            # Convert to rotation matrices to quaternions
            quat1 = roma.rotmat_to_unitquat(R1)
            quat2 = roma.rotmat_to_unitquat(R2)

            # SLERP interpolation for quaternions
            quat_interp = roma.unitquat_slerp(quat1, quat2, torch.tensor(t))

            # Convert back to rotation matrix
            R_interp = roma.unitquat_to_rotmat(quat_interp).numpy()

            # Linear interpolation for translation
            t_interp = (1 - t) * t1.numpy() + t * t2.numpy()

            # Build interpolated transform matrix
            transform_interp = np.eye(4)
            transform_interp[:3, :3] = R_interp
            transform_interp[:3, 3] = t_interp

            # Generate interpolated image name
            name_1 = img_data_1['name'].rsplit('.', 1)[0]
            name_2 = img_data_2['name'].rsplit('.', 1)[0]
            ext = img_data_1['name'].rsplit('.', 1)[1] if '.' in img_data_1['name'] else 'jpg'
            interp_name = f"{name_1}_to_{name_2}_interp_{j:03d}.{ext}"

            all_images[next_id] = {
                'id': next_id,
                'name': interp_name,
                'cam_id': img_data_1['cam_id'],
                'transform_matrix': transform_interp,
                'is_interpolated': True
            }
            test_ids.append(next_id)
            next_id += 1

    # Add last image
    all_images[sorted_images[-1][0]] = sorted_images[-1][1]

    logger.info(f"Generated {len(test_ids)} interpolated poses")
    return all_images, test_ids


def convert_reconstruction_to_seva(
    recon_output_dir: Path,
    seva_output_dir: Path,
    num_splits: List[int] = [1, 3, 6, 9],
    split_strategy: str = "uniform",
    copy_images: bool = True,
    symlink_images: bool = False,
    interpolate: bool = False,
    num_interp: int = 10
):
    """
    Main conversion function.

    Args:
        recon_output_dir: Reconstruction output directory
        seva_output_dir: SEVA format output directory
        num_splits: List of num_inputs for train/test splits
        split_strategy: Strategy for selecting train indices
        copy_images: Whether to copy images to output
        symlink_images: Whether to symlink images (overrides copy if True)
        interpolate: Whether to interpolate camera poses for test views
        num_interp: Number of interpolated poses per pair of images
    """
    recon_output_dir = Path(recon_output_dir)
    seva_output_dir = Path(seva_output_dir)

    # Validate input
    sparse_dir = recon_output_dir / "sparse" / "0"
    if not sparse_dir.exists():
        raise ValueError(f"Sparse directory not found: {sparse_dir}")

    images_dir = recon_output_dir / "images"
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    # Create output directory
    seva_output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Read COLMAP text model
    logger.info("=" * 80)
    logger.info("Step 1: Reading COLMAP text model")
    logger.info("=" * 80)
    cameras, images = read_colmap_text_model(sparse_dir)

    # Step 2: Extract intrinsics
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Extracting intrinsics")
    logger.info("=" * 80)
    intrinsics = extract_common_intrinsics(cameras)

    # Step 2.5: Interpolate camera poses if requested
    if interpolate:
        logger.info("\n" + "=" * 80)
        logger.info(f"Step 2.5: Interpolating {num_interp} poses between each image pair")
        logger.info("=" * 80)
        images, test_ids = interpolate_camera_poses(images, num_interp)
        # All original images are train
        train_ids = sorted([img_id for img_id, img_data in images.items()
                           if not img_data.get('is_interpolated', False)])

    # Step 3: Get sorted image names (for all images)
    image_names = sorted([img['name'] for _, img in sorted(images.items())])
    logger.info(f"Found {len(image_names)} images (including interpolated)" if interpolate else f"Found {len(image_names)} images")

    # Step 4: Convert to transforms.json
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Converting to transforms.json")
    logger.info("=" * 80)
    transforms = reconstruction_to_transforms(images, intrinsics)

    transforms_path = seva_output_dir / "transforms.json"
    with open(transforms_path, 'w') as f:
        json.dump(transforms, f, indent=5)
    logger.info(f"Saved transforms.json to {transforms_path}")

    # Step 5: Create train/test splits
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Creating train/test splits")
    logger.info("=" * 80)

    if interpolate:
        # In interpolation mode: create a single split with all original as train
        split = {
            "train_ids": train_ids,
            "test_ids": test_ids
        }
        split_path = seva_output_dir / "train_test_split.json"
        with open(split_path, 'w') as f:
            json.dump(split, f, indent=2)
        logger.info(f"Saved train_test_split.json")
        logger.info(f"  Train: {len(train_ids)} original images")
        logger.info(f"  Test: {len(test_ids)} interpolated poses")
    else:
        splits = create_train_test_splits(len(image_names), num_splits, split_strategy)

        for num_inputs, split in splits.items():
            split_path = seva_output_dir / f"train_test_split_{num_inputs}.json"
            with open(split_path, 'w') as f:
                json.dump(split, f, indent=2)
            logger.info(f"Saved train_test_split_{num_inputs}.json")

    # Step 6: Copy or symlink images
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Processing images")
    logger.info("=" * 80)

    seva_images_dir = seva_output_dir / "images"
    seva_images_dir.mkdir(parents=True, exist_ok=True)

    if interpolate:
        # Only copy/symlink original images (not interpolated)
        original_image_names = [img['name'] for _, img in sorted(images.items())
                                if not img.get('is_interpolated', False)]
        logger.info(f"Processing {len(original_image_names)} original images (interpolated poses have no images)")
    else:
        original_image_names = image_names

    if symlink_images:
        # Create symlinks
        for img_name in original_image_names:
            src = images_dir / img_name
            dst = seva_images_dir / img_name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        logger.info(f"Created {len(original_image_names)} symlinks in {seva_images_dir}")
    elif copy_images:
        # Copy images
        for img_name in original_image_names:
            src = images_dir / img_name
            dst = seva_images_dir / img_name
            if not dst.exists():
                shutil.copy2(src, dst)
        logger.info(f"Copied {len(original_image_names)} images to {seva_images_dir}")
    else:
        logger.info("Skipping image copying/symlinking")

    logger.info("\n" + "=" * 80)
    logger.info("Conversion Complete!")
    logger.info("=" * 80)
    logger.info(f"SEVA format output: {seva_output_dir}")
    logger.info(f"  - transforms.json ({len(transforms['frames'])} frames)")
    if interpolate:
        logger.info(f"  - train_test_split.json (train: {len(train_ids)}, test: {len(test_ids)})")
    else:
        logger.info(f"  - train_test_split_*.json ({len(num_splits)} files)")
    logger.info(f"  - images/ ({len(original_image_names)} images)")
    logger.info("=" * 80)

    return transforms, splits if not interpolate else None


def main():
    parser = argparse.ArgumentParser(
        description="Convert Reconstruction output to SEVA format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--recon-output",
        type=str,
        required=True,
        help="Path to Reconstruction output directory (contains sparse/0/, images/)"
    )

    parser.add_argument(
        "--seva-output",
        type=str,
        required=True,
        help="Path to SEVA format output directory"
    )

    parser.add_argument(
        "--num-splits",
        type=str,
        default="1,3,6,9",
        help="Comma-separated list of num_inputs for train/test splits (e.g., '1,3,6,9')"
    )

    parser.add_argument(
        "--split-strategy",
        type=str,
        default="uniform",
        choices=["uniform", "first", "random"],
        help="Strategy for selecting train indices"
    )

    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images to output directory"
    )

    parser.add_argument(
        "--symlink-images",
        action="store_true",
        help="Create symlinks instead of copying (Linux/Mac only, overrides --copy-images)"
    )

    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Generate interpolated camera poses as test views (all original images become train)"
    )

    parser.add_argument(
        "--num-interp",
        type=int,
        default=10,
        help="Number of interpolated poses between each pair of consecutive images"
    )

    args = parser.parse_args()

    # Parse num_splits
    num_splits = [int(x.strip()) for x in args.num_splits.split(",")]

    # Convert paths
    recon_output = Path(args.recon_output)
    seva_output = Path(args.seva_output)

    # Validate input directory
    if not recon_output.exists():
        logger.error(f"Reconstruction output directory not found: {recon_output}")
        return 1

    try:
        convert_reconstruction_to_seva(
            recon_output_dir=recon_output,
            seva_output_dir=seva_output,
            num_splits=num_splits,
            split_strategy=args.split_strategy,
            copy_images=args.copy_images,
            symlink_images=args.symlink_images,
            interpolate=args.interpolate,
            num_interp=args.num_interp
        )
        return 0

    except Exception as e:
        logger.error(f"\nError during conversion: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
