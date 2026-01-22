"""
Base undistorter module.

This module defines the abstract base class for all undistortion implementations.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union
import logging
import json

logger = logging.getLogger(__name__)


class BaseUndistorter(ABC):
    """
    Abstract base class for undistortion implementations.

    All undistorter implementations should inherit from this class and implement
    the required methods.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the base undistorter.

        Args:
            device (str, optional): Device to run the model on ('cuda' or 'cpu').
                                    Defaults to auto-detect.
        """
        import torch
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def _load_model(self):
        """
        Load the calibration model for this backend.

        This method should set self.model with the loaded model.
        """
        pass

    @abstractmethod
    def _calibrate(self, image) -> Dict:
        """
        Calibrate a single image.

        Args:
            image: Input image tensor.

        Returns:
            Dict: Calibration results containing camera parameters.
        """
        pass

    @abstractmethod
    def _undistort_image(self, image, calibration_result: Dict) -> tuple:
        """
        Undistort a single image.

        Args:
            image: Input image tensor.
            calibration_result (Dict): Calibration result from _calibrate.

        Returns:
            tuple: (undistorted_image, camera_info_dict)
        """
        pass

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        intrinsics_output_path: Optional[Union[str, Path]] = None,
        target_resolution: Optional[tuple] = None
    ) -> Dict:
        """
        Process all images in a directory: calibrate, undistort, and save.

        Args:
            input_dir (str or Path): Directory containing input images.
            output_dir (str or Path): Directory to save undistorted images.
            intrinsics_output_path (str or Path, optional): Path to save the intrinsics JSON.
            target_resolution (tuple, optional): Target (width, height) to resize all images to.
                                                 If None, automatically use the most common resolution.

        Returns:
            Dict: A dictionary containing calibration results for each image.
        """
        from PIL import Image
        from collections import Counter

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in image_extensions])

        if not image_files:
            logger.warning(f"No images found in {input_path}")
            return {}

        logger.info(f"Found {len(image_files)} images in {input_path}")

        # Auto-detect most common resolution if target_resolution not specified
        if target_resolution is None:
            resolutions = []
            for img_path in image_files:
                with Image.open(img_path) as img:
                    resolutions.append((img.width, img.height))
            resolution_counts = Counter(resolutions)
            most_common = resolution_counts.most_common(1)
            if most_common:
                target_resolution = most_common[0][0]
                logger.info(f"Auto-detected target resolution: {target_resolution[0]}x{target_resolution[1]} "
                           f"({resolution_counts[target_resolution]}/{len(resolutions)} images)")
            else:
                target_resolution = resolutions[0] if resolutions else (1280, 720)
                logger.info(f"Using default target resolution: {target_resolution[0]}x{target_resolution[1]}")

        results = {}

        from tqdm import tqdm
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                result = self.process_single_image(img_path, output_path, target_resolution)
                results[str(img_path)] = result
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
                results[str(img_path)] = {'error': str(e)}

        if intrinsics_output_path:
            self._save_intrinsics(results, intrinsics_output_path)

        return results

    def process_single_image(self, img_path: Path, output_dir: Path, target_resolution: Optional[tuple] = None) -> Dict:
        """
        Process a single image: calibrate, undistort, and save.

        Args:
            img_path (Path): Path to the input image.
            output_dir (Path): Directory to save the undistorted image.
            target_resolution (tuple, optional): Target (width, height) to resize to.
                                                If None, keep original resolution.

        Returns:
            Dict: Processing result containing camera info and output path.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import torchvision.transforms.functional as TF

        # Load image
        pil_img = Image.open(img_path).convert('RGB')
        image = TF.to_tensor(pil_img).to(self.device)

        # Keep track of original size
        orig_h, orig_w = image.shape[-2:]

        # Calibrate
        calibration_result = self._calibrate(image)

        # Undistort
        undistorted_img, camera_info = self._undistort_image(image, calibration_result)

        # Resize back to original resolution
        # Ensure 4D for interpolate
        if undistorted_img.ndim == 3:
            undistorted_img = undistorted_img.unsqueeze(0)

        h_new, w_new = undistorted_img.shape[-2:]

        # Determine target size: either specified target_resolution or original size
        if target_resolution is not None:
            target_w, target_h = target_resolution
        else:
            target_w, target_h = orig_w, orig_h

        if h_new != target_h or w_new != target_w:
            undistorted_img = F.interpolate(
                undistorted_img,
                size=(target_h, target_w),
                mode='bicubic',
                align_corners=False
            )

            # Update intrinsics
            scale_x = target_w / w_new
            scale_y = target_h / h_new

            if 'undistorted_pinhole' in camera_info:
                params_list = camera_info['undistorted_pinhole']['params']
                for params in params_list:
                    # params: [fx, fy, cx, cy]
                    params[0] *= scale_x
                    params[1] *= scale_y
                    params[2] *= scale_x
                    params[3] *= scale_y

                camera_info['undistorted_pinhole']['width'] = target_w
                camera_info['undistorted_pinhole']['height'] = target_h

        # Save
        save_path = output_dir / f"{img_path.stem}_undistorted{img_path.suffix}"
        import torchvision
        torchvision.utils.save_image(undistorted_img, save_path)

        # Cleanup
        del image, undistorted_img
        torch.cuda.empty_cache()

        return {
            'camera': camera_info,
            'output_path': str(save_path)
        }

    def _save_intrinsics(self, results: Dict, output_path: Union[str, Path]):
        """
        Save results to JSON.

        Args:
            results (Dict): Results dictionary from process_directory.
            output_path (str or Path): Path to save the JSON file.
        """
        out_path = Path(output_path)
        try:
            out_path.write_text(json.dumps(results, indent=2))
            logger.info(f"Saved intrinsics to {out_path}")
        except Exception as e:
            logger.error(f"Failed to save intrinsics to {out_path}: {e}")
