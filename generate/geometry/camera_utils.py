"""Camera utilities for SEVA geometry processing.

Migrated from stable-virtual-camera/seva/geometry.py
Focus: Camera pose generation, interpolation, and transformations
"""

import numpy as np
from typing import Literal, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Default FOV in radians (54 degrees)
DEFAULT_FOV_RAD = 0.9424777960769379


def get_default_intrinsics(
    fov_rad: float = DEFAULT_FOV_RAD,
    aspect_ratio: float = 1.0,
) -> np.ndarray:
    """Get default camera intrinsics matrix.
    
    Args:
        fov_rad: Field of view in radians
        aspect_ratio: Width/height aspect ratio
    
    Returns:
        3x3 intrinsics matrix
    """
    if aspect_ratio >= 1.0:  # W >= H
        focal_y = 0.5 / np.tan(0.5 * fov_rad)
        focal_x = focal_y * aspect_ratio
    else:  # W < H
        focal_x = 0.5 / np.tan(0.5 * fov_rad)
        focal_y = focal_x / aspect_ratio
    
    intrinsics = np.eye(3)
    intrinsics[0, 0] = focal_x
    intrinsics[1, 1] = focal_y
    intrinsics[0, 2] = 0.5
    intrinsics[1, 2] = 0.5
    
    return intrinsics


def get_lookat(
    positions: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Get lookat point from camera positions and directions.
    
    Args:
        positions: Camera positions (N, 3)
        directions: Camera directions (N, 3)
    
    Returns:
        Lookat point (3,)
    """
    # Simple average intersection of rays
    # For more robust results, use least squares intersection
    return positions.mean(axis=0)


def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
    endpoint: bool = False,
) -> np.ndarray:
    """Generate interpolated camera path using B-splines.
    
    Args:
        poses: Input poses (N, 3, 4) or (N, 4, 4)
        n_interp: Number of interpolation steps
        spline_degree: B-spline degree
        smoothness: Smoothing factor (0 = exact interpolation)
        rot_weight: Weight for rotation vs translation
        endpoint: Whether to include endpoint
    
    Returns:
        Interpolated poses (M, 3, 4)
    """
    import scipy.interpolate
    
    def poses_to_points(poses, dist):
        """Convert poses to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)
    
    def points_to_poses(points):
        """Convert (position, lookat, up) to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])
    
    def interp(points, n, k, s):
        """B-spline interpolation."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=endpoint)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        return np.reshape(new_points.T, (n, sh[1], sh[2]))
    
    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points, n_interp, spline_degree, smoothness)
    return points_to_poses(new_points)


def generate_spiral_path(
    poses: np.ndarray,
    bounds: np.ndarray,
    n_frames: int = 60,
    n_rots: int = 2,
    zrate: float = 0.5,
    radii: list = [0.5, 0.5, 0.2],
    endpoint: bool = False,
) -> np.ndarray:
    """Generate spiral camera path.
    
    Args:
        poses: Input poses for reference
        bounds: Scene bounds (N, 3) or (2, 3)
        n_frames: Number of output frames
        n_rots: Number of rotations
        zrate: Vertical movement rate
        radii: Spiral radii [x, y, z]
        endpoint: Whether to include endpoint
    
    Returns:
        Spiral path poses (n_frames, 3, 4)
    """
    # Get center and radius from poses
    center = poses[:, :3, 3].mean(axis=0)
    
    # Generate spiral
    theta = np.linspace(0, n_rots * 2 * np.pi, n_frames, endpoint=endpoint)
    z = np.linspace(0, zrate, n_frames, endpoint=endpoint)
    
    x = center[0] + radii[0] * np.cos(theta)
    y = center[1] + radii[1] * np.sin(theta)
    z = center[2] + z * radii[2]
    
    # Create poses looking at center
    positions = np.stack([x, y, z], axis=1)
    poses_out = []
    
    for pos in positions:
        # Look at center with up direction
        lookat = center - pos
        lookat = lookat / np.linalg.norm(lookat)
        right = np.cross(lookat, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, lookat)
        
        R = np.stack([right, up, -lookat], axis=1)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = pos
        poses_out.append(pose[:3, :4])
    
    return np.array(poses_out)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix.
    
    Args:
        lookdir: Look direction vector
        up: Up direction vector
        position: Camera position
    
    Returns:
        3x4 view matrix
    """
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize vector.
    
    Args:
        x: Input vector
    
    Returns:
        Normalized vector
    """
    return x / np.linalg.norm(x)


def rt_to_mat4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Convert rotation and translation to 4x4 matrix.
    
    Args:
        R: Rotation matrix (..., 3, 3)
        t: Translation vector (..., 3)
    
    Returns:
        4x4 transformation matrix
    """
    # Handle both (3,3) and (N,3,3) cases
    if R.ndim == 2:
        mat34 = np.concatenate([R, t[:, None]], axis=-1)
        bottom = np.array([[0, 0, 0, 1]])
        return np.concatenate([mat34, bottom], axis=-2)
    else:
        mat34 = np.concatenate([R, t[..., None]], axis=-1)
        bottom = np.zeros((R.shape[0], 1, 4))
        bottom[:, 0, 3] = 1.0
        return np.concatenate([mat34, bottom], axis=-2)
