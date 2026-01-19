"""
AnyCalib undistorter module.

This module provides undistortion using the AnyCalib library.
"""
import sys
import logging
from pathlib import Path
from typing import Dict

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

# Import base class first
from undistortion.base_undistorter import BaseUndistorter

logger = logging.getLogger(__name__)


class AnyCalibUndistorter(BaseUndistorter):
    """
    Undistorter implementation using AnyCalib backend.

    This class provides camera calibration and image undistortion using
    the AnyCalib library with support for various camera models.
    """

    def __init__(
        self,
        device: str = None,
        model_id: str = "anycalib_gen",
        camera_model: str = "kb",
        undistort_scale: float = -1.0,
        center_principal_point: bool = True
    ):
        """
        Initialize the AnyCalibUndistorter.

        Args:
            device (str, optional): Device to run the model on ('cuda' or 'cpu').
                                    Defaults to auto-detect.
            model_id (str): AnyCalib model ID. Defaults to "anycalib_gen".
            camera_model (str): Camera model to use for undistortion.
                               Defaults to "simple_radial".
            undistort_scale (float): Scale factor for undistortion.
                                    < 0 for adaptive (default), > 0 for fixed scale.
            center_principal_point (bool): If True, force the principal point to
                                          be at the image center.
        """
        super().__init__(device)
        self.model_id = model_id
        self.camera_model = camera_model
        self.undistort_scale = undistort_scale
        self.adaptive_scale = True if self.undistort_scale < 0.0 else False
        self.center_principal_point = center_principal_point

        self._load_model()

    def _load_model(self):
        """Load the AnyCalib model."""
        # Add AnyCalib to path if needed
        anycalib_path = Path(__file__).parent / "AnyCalib"
        if anycalib_path.exists() and str(anycalib_path) not in sys.path:
            sys.path.append(str(anycalib_path))

        logger.info(f"Initializing AnyCalib backend on {self.device} with model_id='{self.model_id}'")

        try:
            from anycalib.model import AnyCalib
            self.model = AnyCalib(model_id=self.model_id).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load AnyCalib model: {e}")
            raise

    def _calibrate(self, image: torch.Tensor) -> Dict:
        """
        Calibrate a single image using AnyCalib.

        Args:
            image (torch.Tensor): Input image tensor [C, H, W].

        Returns:
            Dict: Calibration results containing intrinsics.
        """
        with torch.no_grad():
            res = self.model.predict(image, cam_id=self.camera_model)

        return res

    def _undistort_image(self, image: torch.Tensor, calibration_result: Dict) -> tuple:
        """
        Undistort a single image using AnyCalib calibration results.

        Args:
            image (torch.Tensor): Input image tensor [C, H, W].
            calibration_result (Dict): Calibration result from _calibrate.

        Returns:
            tuple: (undistorted_image, camera_info_dict)
        """
        intrinsics = calibration_result['intrinsics']

        # Create camera object
        from anycalib.cameras import CameraFactory
        cam = CameraFactory.create_from_params(self.camera_model, intrinsics)

        # Prepare image and intrinsics
        image_batch = image.unsqueeze(0)

        if intrinsics.ndim == 1:
            intrinsics_batch = intrinsics.unsqueeze(0)
        else:
            intrinsics_batch = intrinsics

        # Determine scale
        scale = self.undistort_scale
        if self.adaptive_scale:
            h, w = image.shape[-2:]
            scale = self._compute_optimal_scale(cam, intrinsics_batch, h, w)
            logger.info(f"Adaptive scale computed: {scale:.4f}")

        # Undistort
        with torch.no_grad():
            if self.center_principal_point:
                undistorted_img = self._custom_undistort(
                    cam, image_batch, intrinsics_batch, scale=scale, outside_value=0.0
                )
            else:
                undistorted_img = cam.undistort_image(
                    image_batch, intrinsics_batch, scale=scale, outside_value=0.0
                )

            # Create mask to find valid region
            mask = torch.ones_like(image_batch[:, :1, ...])
            if self.center_principal_point:
                undistorted_mask = self._custom_undistort(
                    cam, mask, intrinsics_batch, scale=scale, outside_value=0.0, interp_mode="nearest"
                )
            else:
                undistorted_mask = cam.undistort_image(
                    mask, intrinsics_batch, scale=scale, outside_value=0.0
                )

            # Binarize
            undistorted_mask = (undistorted_mask > 0.5).float()

            # Find largest centered rectangle
            x, y, w_rect, h_rect = self._get_largest_centered_rectangle(undistorted_mask)

            # Crop image
            undistorted_img = undistorted_img[..., y:y+h_rect, x:x+w_rect]

        # Extract camera info
        cam_info = self._extract_camera_info(intrinsics, cam, image.shape[-2:], scale)

        # Update camera info for crop
        for params in cam_info['undistorted_pinhole']['params']:
            # params is [fx, fy, cx, cy]
            params[2] = w_rect / 2.0 # cx
            params[3] = h_rect / 2.0 # cy
        
        # Add image size to info
        cam_info['undistorted_pinhole']['width'] = int(w_rect)
        cam_info['undistorted_pinhole']['height'] = int(h_rect)

        # Cleanup
        del image_batch

        return undistorted_img, cam_info

    def _extract_camera_info(
        self,
        intrinsics: torch.Tensor,
        cam,
        image_size: tuple,
        scale: float
    ) -> Dict:
        """
        Extract serializable camera info.

        Args:
            intrinsics (torch.Tensor): Predicted intrinsics.
            cam: Camera object.
            image_size (tuple): Image size (H, W).
            scale (float): Scale factor used for undistortion.

        Returns:
            Dict: Serializable camera information.
        """
        h, w = image_size

        # Prepare intrinsics for serialization
        if intrinsics.ndim == 1:
            intr_batch = intrinsics.unsqueeze(0)
        else:
            intr_batch = intrinsics

        orig_intrinsics_list = intr_batch.cpu().numpy().tolist()

        # Build undistorted pinhole intrinsics
        num_f_pred = cam.NUM_F
        f_pred = intr_batch[:, :num_f_pred]
        
        if scale != 1.0:
            f_pred = f_pred * scale

        if num_f_pred == 1:
            fx = f_pred
            fy = f_pred
        else:
            fx = f_pred[:, 0:1]
            fy = f_pred[:, 1:2]

        cx_out = torch.full_like(fx, w / 2.0)
        cy_out = torch.full_like(fy, h / 2.0)
        
        # pinhole model expects: fx, fy, cx, cy
        pinhole_params = torch.cat([fx, fy, cx_out, cy_out], dim=1)
        pinhole_list = pinhole_params.cpu().numpy().tolist()

        return {
            'predicted': {
                'model': self.camera_model,
                'params': orig_intrinsics_list,
            },
            'undistorted_pinhole': {
                'model': 'pinhole',
                'params': pinhole_list,
            },
        }

    def _custom_undistort(
        self,
        cam,
        im: torch.Tensor,
        params: torch.Tensor,
        scale: float = 1.0,
        outside_value: float = 0.0,
        interp_mode: str = "bicubic"
    ) -> torch.Tensor:
        """
        Custom undistortion that forces principal point to image center.

        Args:
            cam: Camera object.
            im (torch.Tensor): Input image batch.
            params (torch.Tensor): Camera parameters.
            scale (float): Scale factor.
            outside_value (float): Value for outside regions.
            interp_mode (str): Interpolation mode.

        Returns:
            torch.Tensor: Undistorted image.
        """
        b, _, h, w = im.shape
        num_f = cam.NUM_F
        f = params[..., None, :num_f]

        # Force target principal point to center
        # Use f directly (broadcasts if needed) to preserve aspect ratio
        f_target = f 
        c_target = torch.tensor([w / 2.0, h / 2.0], device=im.device, dtype=params.dtype)
        c_target = c_target.expand(b, 1, 2)

        # Generate grid for target image
        grid = cam.pixel_grid_coords(h, w, params, 0.0).reshape(-1, 2)
        grid = grid.expand(b, -1, -1)

        # Normalized image coordinates for target
        im_n = (grid - c_target) / (f_target * scale)

        r = torch.linalg.norm(im_n, dim=-1)

        target_proj = "perspective"
        theta = cam.ideal_unprojection(r, target_proj)
        phi = torch.atan2(im_n[..., 1], im_n[..., 0])
        R = torch.sin(theta)
        rays = torch.stack(
            (R * torch.cos(phi), R * torch.sin(phi), torch.cos(theta)), dim=-1
        )

        # Project rays back to SOURCE image
        map_xy, valid = cam.project(params, rays)

        # Normalize coords to [-1, 1] for grid_sample
        map_xy = 2 * map_xy.reshape(b, h, w, 2) / map_xy.new_tensor((w, h)) - 1

        im_undist = outside_value + F.grid_sample(
            im - outside_value,
            map_xy,
            mode=interp_mode,
            padding_mode="zeros",
            align_corners=False,
        )

        return im_undist

    def _get_largest_centered_rectangle(self, mask: torch.Tensor) -> tuple:
        """
        Find the largest centered rectangle of 1s in a binary mask.

        Args:
            mask (torch.Tensor): Binary mask [H, W] or [1, H, W].

        Returns:
            tuple: (x, y, w, h) of the largest centered rectangle.
        """
        if mask.ndim == 4:
            mask = mask.squeeze(0).squeeze(0)
        elif mask.ndim == 3:
            mask = mask.squeeze(0)

        h, w = mask.shape
        device = mask.device

        # Identify invalid pixels
        invalid = (mask == 0)
        indices = torch.arange(w, device=device).expand(h, w)

        # Left side invalid pixels (indices < w/2)
        # We need crop_x > index, so crop_x >= index + 1
        left_mask = invalid & (indices < w / 2)
        # If no invalid pixels on left, max returns -1 (if we set default)
        # We use a trick: set valid pixels to -1
        left_indices = torch.where(left_mask, indices, torch.tensor(-1, device=device))
        req_k_left = left_indices.max(dim=1).values + 1

        # Right side invalid pixels (indices >= w/2)
        # We need w - 1 - crop_x < index, so crop_x > w - 1 - index => crop_x >= w - index
        right_mask = invalid & (indices >= w / 2)
        # Set valid pixels to w
        right_indices = torch.where(right_mask, indices, torch.tensor(w, device=device))
        req_k_right = w - right_indices.min(dim=1).values

        # Required horizontal crop for each row
        k_row = torch.max(req_k_left, req_k_right)

        # Now consider vertical symmetry
        # We want to choose y_crop such that we keep rows [y_crop, h - 1 - y_crop]
        # The required x_crop for a given y_crop is max(k_row[y_crop : h - y_crop])

        # Fold k_row to consider symmetry
        # k_sym[y] = max(k_row[y], k_row[h - 1 - y]) for y in 0..h//2
        half_h = h // 2
        
        # Create symmetric k profile
        k_sym = torch.zeros(half_h + (h % 2), device=device, dtype=torch.long)
        k_sym[:half_h] = torch.max(k_row[:half_h], k_row[h - half_h:].flip(0))
        if h % 2:
            k_sym[half_h] = k_row[half_h]

        best_area = -1
        best_rect = (0, 0, 0, 0)

        curr_max_x_crop = 0

        # Iterate from center (y_crop = largest) to top (y_crop = 0)
        for y_crop in range(len(k_sym) - 1, -1, -1):
            curr_max_x_crop = max(curr_max_x_crop, k_sym[y_crop].item())

            current_h = h - 2 * y_crop
            current_w = w - 2 * curr_max_x_crop

            if current_w > 0 and current_h > 0:
                area = current_w * current_h
                if area > best_area:
                    best_area = area
                    best_rect = (curr_max_x_crop, y_crop, current_w, current_h)

        return best_rect

    def _compute_optimal_scale(
        self,
        cam,
        intrinsics: torch.Tensor,
        h: int,
        w: int,
        n_points: int = 20
    ) -> float:
        """
        Compute optimal scale to fit all image boundary points.

        Args:
            cam: Camera object.
            intrinsics (torch.Tensor): Camera intrinsics.
            h (int): Image height.
            w (int): Image width.
            n_points (int): Number of sample points per edge.

        Returns:
            float: Optimal scale factor.
        """
        device = intrinsics.device

        # Generate boundary points
        x = torch.linspace(0, w - 1, n_points, device=device)
        y = torch.linspace(0, h - 1, n_points, device=device)

        # Top, Bottom, Left, Right
        pts_top = torch.stack([x, torch.zeros_like(x)], dim=1)
        pts_bot = torch.stack([x, torch.full_like(x, h - 1)], dim=1)
        pts_left = torch.stack([torch.zeros_like(y), y], dim=1)
        pts_right = torch.stack([torch.full_like(y, w - 1), y], dim=1)

        points = torch.cat([pts_top, pts_bot, pts_left, pts_right], dim=0)

        # Add batch dim if needed
        if intrinsics.ndim == 2:
            points = points.unsqueeze(0).expand(intrinsics.shape[0], -1, -1)

        # Unproject
        bearings, valid = cam.unproject(intrinsics, points)

        if valid is not None:
            pass

        # Assuming batch size 1
        bearings = bearings[0]
        if valid is not None:
            valid = valid[0]
            bearings = bearings[valid]

        # Filter rays with z <= 0
        mask = bearings[..., 2] > 1e-6
        if not mask.any():
            return 1.0

        bearings = bearings[mask]

        # Project to ideal pinhole
        x_proj = bearings[..., 0] / bearings[..., 2]
        y_proj = bearings[..., 1] / bearings[..., 2]

        # Extract f and c
        num_f = cam.NUM_F
        if intrinsics.ndim == 2:
            params = intrinsics[0]
        else:
            params = intrinsics

        f = params[:num_f]
        if num_f == 1:
            fx = fy = f[0]
        else:
            fx, fy = f[0], f[1]

        cx = params[num_f]
        cy = params[num_f + 1]

        scales = []

        # X constraints
        mask_pos = x_proj > 1e-6
        mask_neg = x_proj < -1e-6

        if mask_pos.any():
            s_max = (w - 1 - cx) / (fx * x_proj[mask_pos])
            scales.append(s_max.min())

        if mask_neg.any():
            s_max = -cx / (fx * x_proj[mask_neg])
            scales.append(s_max.min())

        # Y constraints
        mask_pos = y_proj > 1e-6
        mask_neg = y_proj < -1e-6

        if mask_pos.any():
            s_max = (h - 1 - cy) / (fy * y_proj[mask_pos])
            scales.append(s_max.min())

        if mask_neg.any():
            s_max = -cy / (fy * y_proj[mask_neg])
            scales.append(s_max.min())

        if not scales:
            return 1.0

        optimal_scale = torch.stack(scales).min().item()

        return optimal_scale


# Import the base class
from undistortion.base_undistorter import BaseUndistorter
