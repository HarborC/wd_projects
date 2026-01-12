"""
Convert Reconstruction COLMAP output to SEVA format with interpolation.

Fixes applied:
1. Removed dependency on 'roma', used 'scipy' exclusively.
2. Fixed quaternion order (COLMAP WXYZ -> Scipy XYZW).
3. Sorted images by filename instead of ID to ensure correct temporal order.
4. Used Scipy's Slerp for robust rotation interpolation.
5. Added check for multiple cameras.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_colmap_text_model(sparse_dir: Path):
    """
    Read COLMAP text model format.
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

            # Second line: POINTS2D (skip)
            f.readline()

            # Convert quaternion to rotation matrix
            # COLMAP uses (W, X, Y, Z), Scipy uses (X, Y, Z, W)
            r = R.from_quat([qx, qy, qz, qw])
            rot_mat = r.as_matrix()

            # Build 4x4 transformation matrix (world to camera)
            w2c_matrix = np.eye(4)
            w2c_matrix[:3, :3] = rot_mat
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
    Warns if multiple cameras are present.
    """
    if not cameras:
        raise ValueError("No cameras found")

    if len(cameras) > 1:
        logger.warning(f"Found {len(cameras)} different cameras! Using intrinsics from the first one. "
                       "This may cause issues if images have different resolutions or focal lengths.")

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

    # Add distortion parameters if available (Simple Radial / Pinhole)
    # Note: Parameter mapping depends on the COLMAP model (e.g. OPENCV vs PINHOLE).
    # This assumes a basic mapping compatible with PINHOLE/SIMPLE_RADIAL.
    if len(cam['params']) > 4:
        intrinsics["k1"] = cam['params'][4]
    if len(cam['params']) > 5:
        intrinsics["k2"] = cam['params'][5]
    if len(cam['params']) > 6:
        intrinsics["p1"] = cam['params'][6]
    if len(cam['params']) > 7:
        intrinsics["p2"] = cam['params'][7]

    intrinsics["camera_model"] = cam['model']

    logger.info(f"Intrinsics: {intrinsics['w']}x{intrinsics['h']}, fx={intrinsics['fl_x']:.2f}, fy={intrinsics['fl_y']:.2f}")
    return intrinsics


def interpolate_poses(images: Dict, num_interp: int = 1) -> List[Dict]:
    """
    Interpolate poses between consecutive images (sorted by filename).
    """
    all_frames = []

    # FIX: Sort images by NAME (filename) to ensure correct temporal sequence
    # COLMAP image_ids are not guaranteed to be sequential or temporal.
    sorted_images = sorted(images.items(), key=lambda x: x[1]['name'])

    logger.info("Interpolating poses...")
    
    for i in range(len(sorted_images) - 1):
        _, img_curr = sorted_images[i]
        _, img_next = sorted_images[i + 1]

        # Add current original frame
        all_frames.append({
            "c2w_matrix": img_curr['c2w_matrix'].copy(),
            "is_interpolated": False,
            "original_name": img_curr['name'] # Keep track for debugging
        })

        # Prepare for interpolation
        times = [0, 1]
        key_rots = R.from_matrix([
            img_curr['c2w_matrix'][:3, :3],
            img_next['c2w_matrix'][:3, :3]
        ])
        key_trans = np.array([
            img_curr['c2w_matrix'][:3, 3],
            img_next['c2w_matrix'][:3, 3]
        ])

        # Create Slerp object
        slerp = Slerp(times, key_rots)

        # Generate interpolated poses
        for j in range(1, num_interp + 1):
            t = j / (num_interp + 1)
            
            # Interpolate Rotation (SLERP)
            interp_rot = slerp([t]).as_matrix()[0]
            
            # Interpolate Translation (Linear)
            interp_trans = (1 - t) * key_trans[0] + t * key_trans[1]

            # Build matrix
            c2w_interp = np.eye(4)
            c2w_interp[:3, :3] = interp_rot
            c2w_interp[:3, 3] = interp_trans

            all_frames.append({
                "c2w_matrix": c2w_interp,
                "is_interpolated": True,
            })

    # Add the last original frame
    if sorted_images:
        _, img_last = sorted_images[-1]
        all_frames.append({
            "c2w_matrix": img_last['c2w_matrix'].copy(),
            "is_interpolated": False,
            "original_name": img_last['name']
        })

    return all_frames


def convert_reconstruction_to_seva(
    recon_output_dir: Path,
    seva_output_dir: Path,
    interpolate: int = 1,
):
    recon_output_dir = Path(recon_output_dir)
    seva_output_dir = Path(seva_output_dir)

    # Validate input
    sparse_dir = recon_output_dir / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = recon_output_dir / "sparse"
        if not (sparse_dir / "cameras.txt").exists():
             raise ValueError(f"Sparse directory (cameras.txt) not found in: {recon_output_dir}")

    images_dir = recon_output_dir / "images"
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    # Create output directory
    seva_output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Read COLMAP text model
    cameras, images = read_colmap_text_model(sparse_dir)

    # Step 2: Extract intrinsics
    intrinsics = extract_common_intrinsics(cameras)

    # Step 3: Get sorted image names
    sorted_images = sorted(images.items(), key=lambda x: x[1]['name'])
    image_names = [img['name'] for _, img in sorted_images]
    
    first_image_path = images_dir / image_names[0]
    if not first_image_path.exists():
         raise FileNotFoundError(f"Image not found: {first_image_path}")
         
    with Image.open(first_image_path) as first_image:
        img_width, img_height = first_image.size
    logger.info(f"Image size: {img_width}x{img_height}")

    # Step 4: Interpolate poses (Original COLMAP space)
    all_frames = interpolate_poses(images, interpolate)

    train_ids = [i for i, f in enumerate(all_frames) if not f['is_interpolated']]
    test_ids = [i for i, f in enumerate(all_frames) if f['is_interpolated']]

    logger.info(f"Generated {len(train_ids)} original + {len(test_ids)} interpolated frames")

    # Step 5: Build transforms.json with Coordinate System Flip
    json_frames = []
    
    for i, frame_data in enumerate(all_frames):
        # Copy matrix to avoid modifying original data accidentally
        c2w = frame_data['c2w_matrix'].copy()

        # === Apply Coordinate System Transformation ===
        # Flip Y and Z axes (OpenCV [Right, Down, Forward] -> OpenGL [Right, Up, Back])
        c2w[:, [1, 2]] *= -1 
        # ==============================================

        json_frames.append({
            "file_path": f"images/frame_{i:06d}.png",
            "transform_matrix": c2w.tolist(),
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

    # Save JSON files
    transforms_path = seva_output_dir / "transforms.json"
    with open(transforms_path, 'w') as f:
        json.dump(transforms, f, indent=4)
    logger.info(f"Saved transforms.json")

    split_path = seva_output_dir / f"train_test_split_{len(train_ids)}.json"
    with open(split_path, 'w') as f:
        json.dump(splits, f, indent=4)
    logger.info(f"Saved train_test_split_{len(train_ids)}.json")

    # Step 6: Copy images
    logger.info("Processing images...")
    seva_images_dir = seva_output_dir / "images"
    seva_images_dir.mkdir(parents=True, exist_ok=True)

    black_image = Image.new('RGB', (img_width, img_height), (0, 0, 0))

    original_img_idx = 0
    for i, frame_data in enumerate(all_frames):
        dst = seva_images_dir / f"frame_{i:06d}.png"
        
        if frame_data['is_interpolated']:
            black_image.save(dst)
        else:
            src = images_dir / image_names[original_img_idx]
            shutil.copy(src, dst)
            original_img_idx += 1

    logger.info("Conversion Complete!")
    return transforms, splits
# ... (main function remains mostly same, just standard boilerplate)

def main():
    parser = argparse.ArgumentParser(
        description="Convert Reconstruction output to SEVA format with interpolation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--recon-output", type=str, required=True, help="Path to Reconstruction output")
    parser.add_argument("--seva-output", type=str, required=True, help="Path to SEVA output")
    parser.add_argument("--interpolate", type=int, default=20, help="Interpolated poses per pair")

    args = parser.parse_args()
    
    try:
        convert_reconstruction_to_seva(args.recon_output, args.seva_output, args.interpolate)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())