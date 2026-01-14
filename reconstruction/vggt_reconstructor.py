"""
VGGT-X reconstructor module.

This module implements the BaseReconstructor interface using VGGT-X
for multi-view stereo 3D reconstruction.
"""
import logging
import numpy as np
import torch
import torch.nn.functional as F
import sys
import pickle
from pathlib import Path
from typing import Optional, Union
from PIL import Image

from reconstruction.base_reconstructor import BaseReconstructor

logger = logging.getLogger(__name__)

class VGGTReconstructor(BaseReconstructor):
    """
    VGGT-X-based reconstructor.

    This class uses VGGT-X for multi-view stereo matching and 3D reconstruction.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "facebook/VGGT-1B",
        chunk_size: int = 256,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the VGGT-X reconstructor.

        Args:
            device (str, optional): Device to run the model on.
            model_name (str): VGGT model name (for HuggingFace).
            chunk_size (int): Chunk size for frame-wise operation in VGGT.
            model_path (str, optional): Local path to model weights. If provided, will load from this path instead of downloading.
        """
        super().__init__(device)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the VGGT-X model."""

        vggt_path = Path(__file__).parent / "VGGT-X"
        sys.path.insert(0, str(vggt_path))

        from vggt.models.vggt import VGGT

        logger.info(f"Loading VGGT-X model: {self.model_name}")
        self.model = VGGT(chunk_size=self.chunk_size)

        # Load weights
        try:
            logger.info(f"Loading model weights from local path: {self.model_path}")
            state_dict = torch.load(self.model_path, map_location='cpu')
        except Exception:
            # Download and load weights from HuggingFace
            logger.info("Downloading VGGT-X model weights from HuggingFace...")
            _URL = "https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt"
            state_dict = torch.hub.load_state_dict_from_url(_URL)

        self.model.load_state_dict(state_dict)

        # Determine the dtype to use
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float16

        self.model.eval()
        self.model = self.model.to(self.device).to(dtype)
        self.model.track_head = None  # we do not need tracking head for reconstruction
        logger.info(f"VGGT-X model loaded successfully (dtype={dtype})")

    def _process(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        image_files: Optional[list] = None,
        **kwargs
    ):
        """
        Process images using VGGT-X reconstruction.

        Args:
            input_dir (str or Path): Directory containing input images.
            output_dir (str or Path): Directory to save reconstruction results.
            image_files (list, optional): List of image file paths to process.
            **kwargs: Additional parameters.
        """
        from vggt.utils.load_fn import load_and_preprocess_images_ratio
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.helper import create_pixel_coordinate_grid

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Setup backend-specific directory
        vggt_output_dir = output_path / 'vggtx'
        vggt_output_dir.mkdir(parents=True, exist_ok=True)

        # Gather images
        if image_files is None:
            image_files = self._load_imagefiles(input_path)

        if not image_files:
            logger.warning(f"No images found in {input_path}")
            return None

        # Load and preprocess images
        # VGGT works with 5x518 resolution
        img_load_resolution = 518
        images, original_coords = load_and_preprocess_images_ratio(image_files, img_load_resolution)
        logger.info(f"Loaded {len(images)} images")

        # Get original image size from first image
        img = Image.open(image_files[0])
        raw_image_size = (img.height, img.width)

        # Run VGGT inference
        logger.info("Running VGGT-X inference...")

        # Determine the dtype to use (same as in _load_model)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float16

        with torch.no_grad():
            # VGGT expects images in [0, 1] range and specific dtype
            images_tensor = images.to(self.device).to(dtype)

            # Get predictions
            predictions = self.model(images_tensor, verbose=False)

            # Extract pose encoding and convert to extrinsics/intrinsics
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions['pose_enc'],
                images_tensor.shape[-2:]
            )

            # Move to CPU and convert to numpy
            extrinsic = extrinsic.squeeze(0).cpu().numpy()  # (S, 3, 4)
            intrinsic = intrinsic.squeeze(0).cpu().numpy()  # (S, 3, 3)
            depth_map = predictions['depth'].squeeze(0).cpu().numpy()  # (S, H, W, 1)
            depth_conf = predictions['depth_conf'].squeeze(0).cpu().numpy()  # (S, H, W)

            # Check if world points are available
            if 'world_points' in predictions:
                pts3d = predictions['world_points'].squeeze(0).cpu().numpy()  # (S, H, W, 3)
                pts3d_conf = predictions['world_points_conf'].squeeze(0).cpu().numpy()  # (S, H, W)
            else:
                # Unproject depth to get 3D points
                pts3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
                pts3d_conf = depth_conf

        logger.info("Inference completed. Processing results...")

        # Apply scale factor c=2.5 for better reconstruction (as in VGGT demo)
        c = 2.5
        extrinsic[:, :3, 3] *= c
        depth_map *= c
        pts3d *= c

        # Confidence threshold
        conf_thres_value = np.percentile(depth_conf, 0.5)
        logger.info(f"Using confidence threshold: {conf_thres_value}")

        # Create mask
        final_mask = depth_conf >= conf_thres_value  # (S, H, W)

        # Convert images to uint8 (use original images in float32 for better precision)
        images_8u = (images.permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

        # Get processed image size
        S, H, W = depth_map.shape[0], depth_map.shape[1], depth_map.shape[2]
        processed_image_size = (H, W)

        results = {
            "image_num": len(image_files),
            "raw_image_size": raw_image_size,
            "image_names": [Path(f).name for f in image_files],
            "forward_results": {
                'imgs': images_8u,  # (S, H, W, 3)
                'poses': extrinsic,  # (S, 3, 4) - w2c extrinsics
                'intrs': intrinsic,  # (S, 3, 3)
                'depths': depth_map,  # (S, H, W, 1)
                'depths_conf': depth_conf,  # (S, H, W)
                'normals': None,
                'pts3d': pts3d,  # (S, H, W, 3)
                'pts3d_conf': pts3d_conf,  # (S, H, W)
                'masks': final_mask,  # (S, H, W)
            },
        }

        # Save results to pickle file
        results_path = vggt_output_dir / "results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Saved results to {results_path}")

        # Print results shapes for debugging
        for key, value in results["forward_results"].items():
            if isinstance(value, np.ndarray):
                logger.info(f"{key}: {value.shape}")
            else:
                logger.info(f"{key}: {type(value)}")

        logger.info(f"VGGT-X Reconstruction completed. Results saved to {output_path}")

        return results
