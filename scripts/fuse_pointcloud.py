"""
Fuse depth maps and camera poses to generate a colored point cloud.

This script reads:
- Depth maps from depths/ directory (.npy files)
- Camera poses from sparse/0/images.bin
- Camera intrinsics from sparse/0/cameras.bin

Output:
- Unified colored point cloud in PLY format
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
import logging
import argparse

import pycolmap

logger = logging.getLogger(__name__)


def read_colmap_binary(sparse_dir: Path) -> Tuple[dict, dict, dict]:
    """
    Read COLMAP binary files (cameras.bin, images.bin, points3D.bin).

    Args:
        sparse_dir: Path to sparse directory (e.g., sparse/0/)

    Returns:
        cameras: dict {camera_id: camera}
        images: dict {image_id: image}
        points3D: dict {point3D_id: point3D}
    """
    sparse_dir = Path(sparse_dir)

    try:
        reconstruction = pycolmap.Reconstruction(str(sparse_dir))
        print(reconstruction.cameras)
        return reconstruction.cameras, reconstruction.images, reconstruction.points3D
    except Exception as e:
        logger.warning(f"Failed to read with pycolmap: {e}")

    raise RuntimeError("Cannot read COLMAP files. Please install pycolmap.")


def get_camera_matrix(camera) -> np.ndarray:
    """
    Get 3x3 camera intrinsic matrix from COLMAP camera.

    Args:
        camera: COLMAP camera object

    Returns:
        K: 3x3 intrinsic matrix
    """
    # Convert camera.model to string and handle enum/int cases
    model_name = str(camera.model)

    # Handle both string and enum representations
    if "SIMPLE_PINHOLE" in model_name or camera.model == 0:
        # params: f, cx, cy
        f, cx, cy = camera.params
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float64)
    elif "PINHOLE" in model_name or camera.model == 1:
        # params: fx, fy, cx, cy
        fx, fy, cx, cy = camera.params
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported camera model: {camera.model} (type: {type(camera.model)})")

    return K


def depth_to_point_cloud(
    depth_map: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray,
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    depth_threshold: float = 0.0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Back-project depth map to 3D points in world coordinates.

    Args:
        depth_map: Depth map (H, W)
        K: 3x3 camera intrinsic matrix
        pose: 4x4 world-to-camera transformation matrix (extrinsics)
        image: RGB image (H, W, 3) for coloring points
        mask: Optional mask (H, W) to filter valid pixels
        depth_threshold: Minimum depth threshold

    Returns:
        points: (N, 3) array of 3D points in world coordinates
        colors: (N, 3) array of RGB colors, or None if no image provided
    """
    H, W = depth_map.shape

    # Create pixel coordinates grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.reshape(-1).astype(np.float64)
    v = v.reshape(-1).astype(np.float64)
    depth = depth_map.reshape(-1)

    # Filter by depth threshold
    valid_mask = depth > depth_threshold
    if mask is not None:
        valid_mask &= mask.reshape(-1) > 0

    u = u[valid_mask]
    v = v[valid_mask]
    depth = depth[valid_mask]

    if len(depth) == 0:
        return np.empty((0, 3)), None

    # Back-project to camera coordinates
    # Z = depth
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    X_cam = (u - cx) * depth / fx
    Y_cam = (v - cy) * depth / fy
    Z_cam = depth

    points_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)  # (N, 3)

    # Transform to world coordinates
    # pose is world-to-camera (w2c), so we need camera-to-world (c2w)
    # c2w = inv(w2c)
    c2w = np.linalg.inv(pose)

    # Add homogeneous coordinate
    points_cam_h = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
    points_world = (c2w @ points_cam_h.T).T[:, :3]

    # Extract colors if image provided
    colors = None
    if image is not None:
        image_flat = image.reshape(-1, 3)
        colors = image_flat[valid_mask]

    return points_world, colors


def fuse_point_clouds(
    reconstruction_dir: Path,
    depth_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    depth_threshold: float = 0.0,
    stride: int = 1,
    max_distance: float = 100.0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fuse multiple depth maps into a unified point cloud.

    Args:
        reconstruction_dir: Root reconstruction directory containing sparse/ and depths/
        depth_dir: Directory containing depth maps (.npy files). If None, uses reconstruction_dir/depths/
        output_path: Path to save output PLY file. If None, saves to reconstruction_dir/fused.ply
        image_dir: Directory containing RGB images. If None, uses reconstruction_dir/images/
        depth_threshold: Minimum depth threshold
        stride: Subsample stride (1 = use all pixels, 2 = use every 2nd pixel, etc.)
        max_distance: Maximum distance to include points

    Returns:
        points: (N, 3) array of fused 3D points
        colors: (N, 3) array of RGB colors, or None if no images
    """
    reconstruction_dir = Path(reconstruction_dir)

    # Setup paths
    sparse_dir = reconstruction_dir / "sparse" / "0"
    if depth_dir is None:
        depth_dir = reconstruction_dir / "depths"
    if image_dir is None:
        image_dir = reconstruction_dir / "images"
    if output_path is None:
        output_path = reconstruction_dir / "fused.ply"

    depth_dir = Path(depth_dir)
    image_dir = Path(image_dir)
    output_path = Path(output_path)

    # Check directories exist
    if not sparse_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse directory not found: {sparse_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    # Read COLMAP data
    logger.info(f"Reading COLMAP data from {sparse_dir}...")
    cameras, images, points3D = read_colmap_binary(sparse_dir)
    logger.info(f"  Found {len(cameras)} cameras, {len(images)} images")

    # Process each image
    all_points = []
    all_colors = []

    for img_id, img in images.items():
        logger.info(f"Processing image {img_id}: {img.name}")

        # Get camera
        cam = cameras[img.camera_id]

        # Get intrinsic matrix
        K = get_camera_matrix(cam)
        logger.debug(f"  Camera K:\n{K}")

        # Get pose (world-to-camera)
        # pycolmap returns Rigid3d object with rotation and translation
        rigid3d = img.cam_from_world()

        # Convert Rigid3d to 4x4 matrix
        pose_w2c = np.eye(4)
        pose_w2c[:3, :3] = rigid3d.rotation.matrix()  # 3x3 rotation matrix
        pose_w2c[:3, 3] = rigid3d.translation          # 3x1 translation vector
        logger.debug(f"  Pose w2c:\n{pose_w2c}")

        # Load depth map
        depth_path = depth_dir / f"{Path(img.name).stem}.npy"
        if not depth_path.exists():
            logger.warning(f"  Depth map not found: {depth_path}")
            continue

        depth = np.load(depth_path)
        logger.debug(f"  Loaded depth: {depth.shape}, range [{depth.min():.3f}, {depth.max():.3f}]")

        # Load image for coloring
        color_image = None
        if image_dir.exists():
            image_path = image_dir / img.name
            if image_path.exists():
                color_image = cv2.imread(str(image_path))
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                logger.info(f"  Loaded color image: {color_image.shape}")

                # Resize depth to match image if needed
                if depth.shape[:2] != color_image.shape[:2]:
                    orig_h, orig_w = color_image.shape[:2]
                    depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        # Convert depth to point cloud
        points, colors = depth_to_point_cloud(
            depth,
            K,
            pose_w2c,
            color_image,
            depth_threshold=depth_threshold
        )

        # Subsample for efficiency
        if stride > 1:
            points = points[::stride,]
            colors = colors[::stride,]

        logger.debug(f"  Generated {len(points)} points")
        if colors is not None:
            logger.info(f"  Generated {len(colors)} colors")
        else:
            logger.warning(f"  No colors generated!")

        # Filter by max distance
        if max_distance > 0:
            distances = np.linalg.norm(points, axis=1)
            valid = distances < max_distance
            points = points[valid]
            if colors is not None:
                colors = colors[valid]

        all_points.append(points)
        if colors is not None:
            all_colors.append(colors)

    # Concatenate all points
    logger.info("Fusing point clouds...")
    all_points = np.concatenate(all_points, axis=0)
    logger.info(f"Total points: {len(all_points)}")

    if all_colors and len(all_colors) > 0:
        all_colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
        logger.info(f"Total colors: {len(all_colors)}")
    else:
        all_colors = None

    # Save to PLY
    save_ply_file(output_path, all_points, all_colors, binary=True)
    logger.info(f"Saved fused point cloud to {output_path}")

    return all_points, all_colors


def save_ply_file(filepath: Path, points: np.ndarray, colors: Optional[np.ndarray] = None, binary: bool = True):
    """
    Save point cloud to PLY file.

    Args:
        filepath: Output PLY file path
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors (0-255), optional
        binary: If True, save in binary format (smaller file size)
    """
    n_points = len(points)

    with open(filepath, 'wb') if binary else open(filepath, 'w') as f:
        if binary:
            # Write header
            header = "ply\n"
            header += "format binary_little_endian 1.0\n"
            header += f"element vertex {n_points}\n"
            header += "property float x\n"
            header += "property float y\n"
            header += "property float z\n"

            if colors is not None:
                header += "property uchar red\n"
                header += "property uchar green\n"
                header += "property uchar blue\n"

            header += "end_header\n"
            f.write(header.encode('ascii'))

            # Write vertex data
            for i in range(n_points):
                x, y, z = points[i]
                f.write(np.float32(x).tobytes())
                f.write(np.float32(y).tobytes())
                f.write(np.float32(z).tobytes())
                if colors is not None:
                    r, g, b = colors[i]
                    f.write(np.uint8(r).tobytes())
                    f.write(np.uint8(g).tobytes())
                    f.write(np.uint8(b).tobytes())
        else:
            # Write ASCII header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")

            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

            f.write("end_header\n")

            # Write vertex data
            for i in range(n_points):
                x, y, z = points[i]
                if colors is not None:
                    r, g, b = colors[i]
                    f.write(f"{x} {y} {z} {r} {g} {b}\n")
                else:
                    f.write(f"{x} {y} {z}\n")

    logger.info(f"Saved {n_points} points to {filepath} (binary={binary})")


def main():
    parser = argparse.ArgumentParser(
        description="Fuse depth maps and camera poses to generate colored point cloud",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "reconstruction_dir",
        type=str,
        help="Root reconstruction directory containing sparse/ and depths/"
    )
    parser.add_argument(
        "--depth-threshold",
        type=float,
        default=0.0,
        help="Minimum depth threshold (meters)"
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=100.0,
        help="Maximum distance to include points (meters)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Subsample stride (1=use all pixels, 2=every 2nd pixel, etc.)"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("=" * 80)
    logger.info("Point Cloud Fusion")
    logger.info("=" * 80)
    logger.info(f"Reconstruction dir: {args.reconstruction_dir}")
    logger.info(f"Depth threshold: {args.depth_threshold}")
    logger.info(f"Max distance: {args.max_distance}")
    logger.info(f"Stride: {args.stride}")
    logger.info("=" * 80)

    try:
        points, colors = fuse_point_clouds(
            reconstruction_dir=args.reconstruction_dir,
            depth_threshold=args.depth_threshold,
            stride=args.stride,
            max_distance=args.max_distance
        )

        logger.info("=" * 80)
        logger.info("Fusion Complete!")
        logger.info(f"Points: {len(points)}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
