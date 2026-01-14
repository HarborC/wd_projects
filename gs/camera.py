"""
Camera class for Gaussian Splatting.

This module implements the Camera class used for rendering.
"""

import torch
import numpy as np
from typing import Optional
from PIL import Image


class Camera:
    """
    Camera class for rendering.

    Represents a camera with intrinsics and extrinsics for rendering
    Gaussian splats.
    """

    def __init__(
        self,
        uid: int,
        colmap_id: int,
        R: np.ndarray,
        T: np.ndarray,
        FoVx: float,
        FoVy: float,
        image: Optional[np.ndarray],
        image_path: str,
        image_name: str,
        width: int,
        height: int,
        device: str = "cuda"
    ):
        """
        Initialize Camera.

        Args:
            uid: Unique identifier
            colmap_id: COLMAP camera ID
            R: 3x3 rotation matrix (world to camera)
            T: 3x1 translation vector
            FoVx: Horizontal field of view in radians
            FoVy: Vertical field of view in radians
            image: Image as numpy array (H, W, 3)
            image_path: Path to image file
            image_name: Name of image
            width: Image width
            height: Image height
            device: Device to use
        """
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.image_width = width
        self.image_height = height

        self.device = device

        # Build projection matrix
        self.projection_matrix = self.get_projection_matrix()

        # Build world view transform
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).to(device)

        # Camera center
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # Original image
        if image is not None:
            self.original_image = torch.from_numpy(image).float().to(device) / 255.0
            if self.original_image.shape[0] == 3:
                self.original_image = self.original_image.permute(1, 2, 0)  # H, W, C
            self.original_image = self.original_image.permute(2, 0, 1)  # C, H, W
        else:
            self.original_image = None

    @property
    def width(self):
        return self.image_width

    @property
    def height(self):
        return self.image_height

    def get_projection_matrix(self):
        """Get projection matrix."""
        near = 0.01
        far = 100.0

        top = near * np.tan(self.FoVy / 2.0)
        bottom = -top
        right = near * np.tan(self.FoVx / 2.0)
        left = -right

        P = torch.zeros(4, 4, dtype=torch.float32)

        # Perspective projection matrix
        P[0, 0] = 2.0 * near / (right - left)
        P[0, 2] = (right + left) / (right - left)
        P[1, 1] = 2.0 * near / (top - bottom)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[2, 2] = -(far + near) / (far - near)
        P[2, 3] = -2.0 * far * near / (far - near)
        P[3, 2] = -1.0

        return P.to(self.device)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """Get world to view matrix."""
    Rt = np.eye(4)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    Rt[:3, 3] = -(R.transpose() @ (scale * cam_center + translate))
    return Rt
