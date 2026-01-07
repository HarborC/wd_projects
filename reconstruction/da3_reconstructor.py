"""
DA3 (Depth Anything V3) reconstructor module.

This module implements the BaseReconstructor interface using Depth Anything V3
for 3D reconstruction.
"""
import logging
import shutil
import cv2
import sys
import pickle
import numpy as np
import pycolmap
from pathlib import Path
from typing import Optional, Union
from PIL import Image

from reconstruction.base_reconstructor import BaseReconstructor

logger = logging.getLogger(__name__)

def get_conf_thresh(
    conf_pixels: np.ndarray,
    conf_thresh: float = 1.05,
    conf_thresh_percentile: float = 10.0,
    ensure_thresh_percentile: float = 90.0,
):
    lower = np.percentile(conf_pixels, conf_thresh_percentile)
    upper = np.percentile(conf_pixels, ensure_thresh_percentile)
    conf_thresh = min(max(conf_thresh, lower), upper)
    return conf_thresh

class DA3Reconstructor(BaseReconstructor):
    """
    Depth Anything V3-based reconstructor.

    This class uses DA3 for monocular depth estimation and 3D reconstruction.
    """

    def __init__(self, device: Optional[str] = None, model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE"):
        """
        Initialize the DA3 reconstructor.

        Args:
            device (str, optional): Device to run the model on.
            model_name (str): DA3 model name.
        """
        super().__init__(device)
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load the DA3 model."""
        da3_path = Path(__file__).parent
        sys.path.insert(0, str(da3_path))

        from depth_anything_3.api import DepthAnything3
        logger.info(f"Loading DA3 model: {self.model_name}")
        self.model = DepthAnything3.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        logger.info("DA3 model loaded successfully")
   
    def _process(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        image_files: Optional[list] = None,
        **kwargs
    ):
        """
        Process images using DA3 reconstruction.

        Args:
            input_dir (str or Path): Directory containing input images.
            output_dir (str or Path): Directory to save reconstruction results.
            **kwargs: Additional parameters.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # DA3 outputs are already in output_path / "da3" directory
        da3_output_path = output_path / "da3"
        da3_output_path.mkdir(parents=True, exist_ok=True)

        # Gather images
        if image_files is None:
            image_files = self._load_imagefiles(input_path)

        if not image_files:
            logger.warning(f"No images found in {input_path}")
            return

        # Run inference without automatic export
        prediction = self.model.inference(
            image_files,
            export_dir=str(da3_output_path),
            export_format="npz-glb-gs_ply-gs_video-colmap",
            infer_gs=True,
        )

        conf_thresh = get_conf_thresh(conf_pixels=prediction.conf)
        final_mask = prediction.conf >= conf_thresh  # [S, H, W]

        poses_4x4 = np.zeros((prediction.extrinsics.shape[0], 4, 4))
        poses_4x4[:, :3, :4] = prediction.extrinsics
        poses_4x4[:, 3, 3] = 1.0

        img = Image.open(image_files[0])
        raw_image_size = (img.height, img.width)

        results = {
          "image_num": len(image_files),
          "raw_image_size": raw_image_size,
          "image_names": [Path(f).name for f in image_files],
          "forward_results": {
            'imgs': prediction.processed_images,
            'poses': poses_4x4,  
            'intrs': prediction.intrinsics, 
            'depths': prediction.depth[..., None], 
            'depths_conf': prediction.conf,
            'normals': None,
            'pts3d': None,
            'pts3d_conf': None,
            'masks': final_mask,
          },
        }

        # save results to output_path / "results.pkl"
        results_path = da3_output_path / "results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Saved results to {results_path}")

        # print results shapes
        for key, value in results["forward_results"].items():
            if isinstance(value, np.ndarray):
                logger.info(f"{key}: {value.shape}")
            else:
                logger.info(f"{key}: {type(value)}")

        logger.info(f"DA3 Reconstruction completed. Results saved to {output_path}")

        return results
