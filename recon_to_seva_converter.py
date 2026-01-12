"""
Convert Reconstruction COLMAP output to SEVA format with interpolation.

This script converts the output from Reconstruction module (MASt3R/DA3/HunyuanWorld)
to the format required by Stable Virtual Camera (SEVA), with interpolated poses as test set.

Input (from Reconstruction):
    output_dir/
        images/           # Input images
        sparse/0/         # COLMAP format
            cameras.txt   # Camera intrinsics
            images.txt    # Camera extrinsics

Output (for SEVA):
    seva_output_dir/
        transforms.json           # Camera parameters
        train_test_split_N.json    # Train/test splits (N = num original images)
        images/                    # Symlink of original images + black interpolated images

Usage:
    python recon_to_seva_converter.py \
        --recon_output /path/to/reconstruction/output \
        --seva_output /path/to/seva/output \
        --interpolate 1
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_colmap_text_model(sparse_dir: Path):
    """
    Read COLMAP text model format.

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
            quat = torch.tensor([qw, qx, qy, qz])
            R = roma.unitquat_to_rotmat(quat).numpy()

            # Build 4x4 transformation matrix (world to camera)
            w2c_matrix = np.eye(4)
            w2c_matrix[:3, :3] = R
            w2c_matrix[:3, 3] = [tx, ty, tz]

            # Convert to camera-to-world (SEVA expects camtoworld)
            c2w_matrix = np.linalg.inv(w2c_matrix)

            images[img_id] = {
                'id': img_id,
                'name': name,
                'cam_id': cam_id,
                'c2w_matrix': c2w_matrix
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


def interpolate_poses(images: Dict, num_interp: int = 1) -> List[Dict]:
    """
    Interpolate poses between consecutive images.

    Args:
        images: Dict of image info from read_colmap_text_model
        num_interp: Number of interpolated poses between each pair of images

    Returns:
        Tuple of (original_frames, interpolated_frames) with c2w_matrix (camera-to-world)
    """
    original_frames = []
    interpolated_frames = []

    # Sort images by image_id
    sorted_images = sorted(images.items(), key=lambda x: x[0])

    # Add all original images
    for _, img_data in sorted_images:
        original_frames.append({
            "c2w_matrix": img_data['c2w_matrix'].copy(),
        })

    # Generate interpolated poses between consecutive images
    for i in range(len(sorted_images) - 1):
        _, img_curr = sorted_images[i]
        _, img_next = sorted_images[i + 1]

        # Interpolate between current and next
        for j in range(1, num_interp + 1):
            t = j / (num_interp + 1)

            # Interpolate translation (camera position in world space)
            trans_curr = img_curr['c2w_matrix'][:3, 3]
            trans_next = img_next['c2w_matrix'][:3, 3]
            trans_interp = trans_curr * (1 - t) + trans_next * t

            # Interpolate rotation
            rot_curr = img_curr['c2w_matrix'][:3, :3]
            rot_next = img_next['c2w_matrix'][:3, :3]

            # Linear interpolation of rotation matrix, then orthonormalize
            rot_interp = rot_curr * (1 - t) + rot_next * t

            # Orthonormalize using SVD
            U, _, Vt = np.linalg.svd(rot_interp)
            rot_interp = U @ Vt

            # Build interpolated camera-to-world matrix
            c2w_interp = np.eye(4)
            c2w_interp[:3, :3] = rot_interp
            c2w_interp[:3, 3] = trans_interp

            interpolated_frames.append({
                "c2w_matrix": c2w_interp,
            })

    return original_frames, interpolated_frames


def convert_reconstruction_to_seva(
    recon_output_dir: Path,
    seva_output_dir: Path,
    interpolate: int = 1,
):
    """
    Main conversion function.

    Args:
        recon_output_dir: Reconstruction output directory
        seva_output_dir: SEVA format output directory
        interpolate: Number of interpolated poses between each pair of images
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

    # Step 3: Get sorted image names
    sorted_images = sorted(images.items(), key=lambda x: x[0])
    image_names = [img['name'] for _, img in sorted_images]
    logger.info(f"Found {len(image_names)} images")

    # Get image dimensions from first image
    first_image_path = images_dir / image_names[0]
    first_image = Image.open(first_image_path)
    img_width, img_height = first_image.size
    first_image.close()
    logger.info(f"Image size: {img_width}x{img_height}")

    # Step 4: Interpolate poses
    logger.info("\n" + "=" * 80)
    logger.info(f"Step 4: Interpolating poses ({interpolate} poses between each pair)")
    logger.info("=" * 80)

    original_frames, interpolated_frames = interpolate_poses(images, interpolate)
    logger.info(f"Generated {len(original_frames)} original + {len(interpolated_frames)} interpolated frames")

    # Step 5: Build transforms.json and train/test split
    # All original images -> train_ids (first N frames)
    # All interpolated poses -> test_ids (after original frames)
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Building transforms.json and train/test split")
    logger.info("=" * 80)

    train_ids = list(range(len(original_frames)))
    test_ids = list(range(len(original_frames), len(original_frames) + len(interpolated_frames)))

    json_frames = []

    # Add original frames (train)
    for i, frame_data in enumerate(original_frames):
        json_frames.append({
            "file_path": f"images/{image_names[i]}",
            "transform_matrix": frame_data['c2w_matrix'].tolist(),
        })

    # Add interpolated frames (test)
    for i, frame_data in enumerate(interpolated_frames):
        json_frames.append({
            "file_path": f"images/interp_{i:06d}.png",
            "transform_matrix": frame_data['c2w_matrix'].tolist(),
        })

    transforms = {
        **intrinsics,
        "frames": json_frames
    }

    # Create train/test split
    splits = {
        "train_ids": train_ids,
        "test_ids": test_ids
    }

    # Save transforms.json
    transforms_path = seva_output_dir / "transforms.json"
    with open(transforms_path, 'w') as f:
        json.dump(transforms, f, indent=5)
    logger.info(f"Saved transforms.json to {transforms_path}")

    # Save train_test split
    split_path = seva_output_dir / f"train_test_split_{len(train_ids)}.json"
    with open(split_path, 'w') as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved train_test_split_{len(train_ids)}.json")

    # Step 6: Create symlinks for original images and black images for interpolated
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Processing images")
    logger.info("=" * 80)

    seva_images_dir = seva_output_dir / "images"
    seva_images_dir.mkdir(parents=True, exist_ok=True)

    # Create symlinks for original images
    for img_name in image_names:
        src = images_dir / img_name
        dst = seva_images_dir / img_name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
    logger.info(f"Created {len(image_names)} symlinks for original images")

    # Create black images for interpolated poses
    black_image = Image.new('RGB', (img_width, img_height), (0, 0, 0))
    for frame in transforms['frames']:
        if 'interp_' in frame['file_path']:
            img_name = frame['file_path'].split('/')[-1]
            dst = seva_images_dir / img_name
            if not dst.exists():
                black_image.save(dst)
    logger.info(f"Created {len(test_ids)} black images for interpolated poses")

    logger.info("\n" + "=" * 80)
    logger.info("Conversion Complete!")
    logger.info("=" * 80)
    logger.info(f"SEVA format output: {seva_output_dir}")
    logger.info(f"  - transforms.json ({len(transforms['frames'])} frames)")
    logger.info(f"  - train_test_split_{len(train_ids)}.json ({len(train_ids)} train, {len(test_ids)} test)")
    logger.info(f"  - images/ ({len(train_ids)} original + {len(test_ids)} interpolated)")
    logger.info("=" * 80)

    return transforms, splits


def main():
    parser = argparse.ArgumentParser(
        description="Convert Reconstruction output to SEVA format with interpolation",
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
        "--interpolate",
        type=int,
        default=5,
        help="Number of interpolated poses between each pair of images"
    )

    args = parser.parse_args()

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
            interpolate=args.interpolate,
        )
        return 0

    except Exception as e:
        logger.error(f"\nError during conversion: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
