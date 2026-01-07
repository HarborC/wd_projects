"""
GeoCalib undistorter module.

This module provides undistortion using the GeoCalib library.
"""
import sys
import logging
from pathlib import Path
from typing import Dict

import torch
import torchvision

# Import base class first
from undistortion.base_undistorter import BaseUndistorter

logger = logging.getLogger(__name__)


class GeoCalibUndistorter(BaseUndistorter):
    """
    Undistorter implementation using GeoCalib backend.

    This class provides camera calibration and image undistortion using
    the GeoCalib library.
    """

    def __init__(self, device: str = None, weights: str = "distorted"):
        """
        Initialize the GeoCalibUndistorter.

        Args:
            device (str, optional): Device to run the model on ('cuda' or 'cpu').
                                    Defaults to auto-detect.
            weights (str): Weights to load for GeoCalib. Defaults to "distorted".
        """
        super().__init__(device)
        self.weights = weights
        self._load_model()

    def _load_model(self):
        """Load the GeoCalib model."""
        # Add GeoCalib to path
        geocalib_path = Path(__file__).parent.parent / "GeoCalib"
        if geocalib_path.exists() and str(geocalib_path) not in sys.path:
            sys.path.append(str(geocalib_path))

        logger.info(f"Initializing GeoCalib backend on {self.device} with weights='{self.weights}'")

        try:
            from geocalib import GeoCalib
            self.model = GeoCalib(weights=self.weights).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load GeoCalib model: {e}")
            raise

    def _calibrate(self, image: torch.Tensor) -> Dict:
        """
        Calibrate a single image using GeoCalib.

        Args:
            image (torch.Tensor): Input image tensor [C, H, W].

        Returns:
            Dict: Calibration results containing 'camera' and 'gravity' objects.
        """
        with torch.no_grad():
            # Load and calibrate
            image_batch = self.model.load_image_tensor(image).to(self.device)
            res = self.model.calibrate(image_batch, camera_model="simple_divisional")

        return res

    def _undistort_image(self, image: torch.Tensor, calibration_result: Dict) -> tuple:
        """
        Undistort a single image using GeoCalib calibration results.

        Args:
            image (torch.Tensor): Input image tensor [C, H, W].
            calibration_result (Dict): Calibration result from _calibrate.

        Returns:
            tuple: (undistorted_image, camera_info_dict)
        """
        cam = calibration_result.get('camera')
        grav = calibration_result.get('gravity')

        if cam is None:
            raise ValueError("Camera calibration failed")

        # Undistort
        image_batch = image.unsqueeze(0)
        with torch.no_grad():
            undistorted_img = cam.undistort_image(image_batch)

        # Extract camera info
        cam_info = self._extract_camera_info(cam)

        # Cleanup
        del image_batch

        grav_info = None
        if grav is not None:
            grav_info = self._extract_gravity_info(grav)

        # Add gravity info to camera info if available
        if grav_info is not None:
            cam_info['gravity'] = grav_info

        return undistorted_img, cam_info

    def _extract_camera_info(self, cam) -> Dict:
        """
        Extract serializable camera info from GeoCalib camera object.

        Args:
            cam: GeoCalib camera object.

        Returns:
            Dict: Serializable camera information.
        """
        return {
            'model': cam.__class__.__name__,
            'K': cam.K.cpu().numpy().tolist() if hasattr(cam, 'K') else None,
            'f': cam.f.cpu().numpy().tolist() if hasattr(cam, 'f') else None,
            'c': cam.c.cpu().numpy().tolist() if hasattr(cam, 'c') else None,
            'size': cam.size.cpu().numpy().tolist() if hasattr(cam, 'size') else None,
            'dist': cam.dist.cpu().numpy().tolist() if hasattr(cam, 'dist') else None,
            'k1': float(cam.k1.cpu().numpy()) if hasattr(cam, 'k1') else None,
            'k2': float(cam.k2.cpu().numpy()) if hasattr(cam, 'k2') else None,
            'vfov': float(cam.vfov.cpu().numpy()) if hasattr(cam, 'vfov') else None,
        }

    def _extract_gravity_info(self, grav) -> Dict:
        """
        Extract serializable gravity info from GeoCalib gravity object.

        Args:
            grav: GeoCalib gravity object.

        Returns:
            Dict: Serializable gravity information.
        """
        return {
            'vec3d': grav.vec3d.cpu().numpy().tolist() if hasattr(grav, 'vec3d') else None,
            'rp': grav.rp.cpu().numpy().tolist() if hasattr(grav, 'rp') else None,
            'roll': float(grav.roll.cpu().numpy()) if hasattr(grav, 'roll') else None,
            'pitch': float(grav.pitch.cpu().numpy()) if hasattr(grav, 'pitch') else None,
        }

from undistortion.base_undistorter import BaseUndistorter