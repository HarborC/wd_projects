"""
HunyuanWorld-Mirror reconstructor module.

This module implements the BaseReconstructor interface using HunyuanWorld-Mirror
for universal 3D reconstruction.
"""
import logging
import os
import sys
import glob
import pickle
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from reconstruction.base_reconstructor import BaseReconstructor

logger = logging.getLogger(__name__)

def as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")

def create_filter_mask(
    pts3d_conf: np.ndarray,
    depth_preds: np.ndarray, 
    normal_preds: np.ndarray,
    sky_mask: np.ndarray,
    confidence_percentile: float = 10.0,
    edge_normal_threshold: float = 5.0,
    edge_depth_threshold: float = 0.03,
    apply_confidence_mask: bool = True,
    apply_edge_mask: bool = True,
    apply_sky_mask: bool = False,
) -> np.ndarray:
    """
    Create comprehensive filter mask based on confidence, edges, and sky segmentation.
    This follows the same logic as app.py for consistent mask computation.
    
    Args:
        pts3d_conf: Point confidence scores [S, H, W]
        depth_preds: Depth predictions [S, H, W, 1]
        normal_preds: Normal predictions [S, H, W, 3]
        sky_mask: Sky segmentation mask [S, H, W]
        confidence_percentile: Percentile threshold for confidence filtering (0-100)
        edge_normal_threshold: Normal angle threshold in degrees for edge detection
        edge_depth_threshold: Relative depth threshold for edge detection
        apply_confidence_mask: Whether to apply confidence-based filtering
        apply_edge_mask: Whether to apply edge-based filtering
        apply_sky_mask: Whether to apply sky mask filtering
    
    Returns:
        final_mask: Boolean mask array [S, H, W] for filtering points
    """

    from src.utils.geometry import depth_edge, normals_edge

    S, H, W = pts3d_conf.shape[:3]
    final_mask_list = []
    
    for i in range(S):
        final_mask = None
        
        if apply_confidence_mask:
            # Compute confidence mask based on the pointmap confidence
            confidences = pts3d_conf[i, :, :]  # [H, W]
            percentile_threshold = np.quantile(confidences, confidence_percentile / 100.0)
            conf_mask = confidences >= percentile_threshold
            if final_mask is None:
                final_mask = conf_mask
            else:
                final_mask = final_mask & conf_mask
        
        if apply_edge_mask:
            # Compute edge mask based on the normalmap
            normal_pred = normal_preds[i]  # [H, W, 3]
            normal_edges = normals_edge(
                normal_pred, tol=edge_normal_threshold, mask=final_mask
            )
            # Compute depth mask based on the depthmap
            depth_pred = depth_preds[i, :, :, 0]  # [H, W]
            depth_edges = depth_edge(
                depth_pred, rtol=edge_depth_threshold, mask=final_mask
            )
            edge_mask = ~(depth_edges & normal_edges)
            if final_mask is None:
                final_mask = edge_mask
            else:
                final_mask = final_mask & edge_mask
        
        if apply_sky_mask:
            # Apply sky mask filtering (sky_mask is already inverted: True = non-sky)
            sky_mask_frame = sky_mask[i]  # [H, W]
            if final_mask is None:
                final_mask = sky_mask_frame
            else:
                final_mask = final_mask & sky_mask_frame
        
        final_mask_list.append(final_mask)
    
    # Stack all frame masks
    if final_mask_list[0] is not None:
        final_mask = np.stack(final_mask_list, axis=0)  # [S, H, W]
    else:
        final_mask = np.ones(pts3d_conf.shape[:3], dtype=bool)  # [S, H, W]
    
    return final_mask

class HunyuanWorldReconstructor(BaseReconstructor):
    """
    HunyuanWorld-Mirror-based reconstructor.

    This class uses HunyuanWorld-Mirror for universal 3D geometric prediction,
    supporting multi-modal priors (camera poses, intrinsics, depth maps) and
    generating various 3D representations (point clouds, depths, normals, 3D Gaussians).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_path: str = "tencent/HunyuanWorld-Mirror"
    ):
        """
        Initialize the HunyuanWorld-Mirror reconstructor.

        Args:
            device (str, optional): Device to run the model on.
            model_path (str): Path or HuggingFace ID for HunyuanWorld-Mirror model.
        """
        super().__init__(device)
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the HunyuanWorld-Mirror model."""
        # Set rank_zero_only rank for distributed training compatibility
        from lightning.fabric.utilities.rank_zero import rank_zero_only
        rank_zero_only.rank = 0

        hyw_path = Path(__file__).parent / "HunyuanWorld-Mirror"
        sys.path.insert(0, str(hyw_path))

        from src.models.models.worldmirror import WorldMirror

        logger.info(f"Loading HunyuanWorld-Mirror model: {self.model_path}")
        self.model = WorldMirror.from_pretrained(self.model_path).to(self.device)
        self.model.eval()
        logger.info("HunyuanWorld-Mirror model loaded successfully")

    def _process(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        image_files: Optional[list] = None,
        **kwargs
    ):
        """
        Process images using HunyuanWorld-Mirror reconstruction.

        Args:
            input_dir (str or Path): Directory containing input images.
            output_dir (str or Path): Directory to save reconstruction results.
            image_files (list, optional): List of image file paths to process.
            **kwargs: Additional parameters.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Setup backend-specific directory
        hunyuanworld_output_dir = output_path / 'hunyuanworld'
        hunyuanworld_output_dir.mkdir(parents=True, exist_ok=True)

        from src.utils.inference_utils import prepare_images_to_tensor
        from src.utils.save_utils import save_depth_png, save_depth_npy, save_normal_png
        from src.utils.save_utils import save_scene_ply, save_gs_ply
        from src.utils.build_pycolmap_recon import build_pycolmap_reconstruction2
        from src.utils.save_utils import save_points_ply
        from src.models.utils.camera_utils import vector_to_camera_matrices
        from src.models.utils.geometry import create_pixel_coordinate_grid, depth_to_world_coords_points
        from src.utils.render_utils import render_interpolated_video

        # --- Load Images ---
        if image_files is None:
            image_files = self._load_imagefiles(input_path)

        if not image_files:
            logger.warning(f"No images found in {input_path}")
            return

        # --- Preprocess Images ---
        target_size = 518
        imgs = prepare_images_to_tensor(image_files, target_size=target_size, resize_strategy="crop").to(self.device)

        B, S, C, H, W = imgs.shape
        logger.info(f"Loaded {S} images with shape {imgs.shape}")

        # --- Inference ---
        logger.info("Running HunyuanWorld-Mirror inference...")
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                predictions = self.model(views={"img": imgs}, cond_flags=[0, 0, 0])

        logger.info("Inference completed. Saving results...")

        hunyuanworld_gs_dir = hunyuanworld_output_dir / "gs"
        hunyuanworld_gs_dir.mkdir(parents=True, exist_ok=True)
        # Save Gaussians PLY and render video
        if "splats" in predictions:
            # Get Gaussian parameters (already filtered by GaussianSplatRenderer)
            means = predictions["splats"]["means"][0].reshape(-1, 3)
            scales = predictions["splats"]["scales"][0].reshape(-1, 3)
            quats = predictions["splats"]["quats"][0].reshape(-1, 4)
            colors = (predictions["splats"]["sh"][0] if "sh" in predictions["splats"] else predictions["splats"]["colors"][0]).reshape(-1, 3)
            opacities = predictions["splats"]["opacities"][0].reshape(-1)
            
            # Save Gaussian PLY
            ply_path = hunyuanworld_gs_dir / "gaussians.ply"
            save_gs_ply(ply_path, means, scales, quats, colors, opacities)

            ply_path = output_path / "gaussians.ply"
            save_gs_ply(ply_path, means, scales, quats, colors, opacities)

            # Render video using the same filtered splats from predictions
            num_views = S
            e4x4 = predictions['camera_poses']
            k3x3 = predictions['camera_intrs']
            render_interpolated_video(self.model.gs_renderer, predictions["splats"], e4x4, k3x3, (H, W), hunyuanworld_gs_dir / "rendered", interp_per_pair=15, loop_reverse=num_views==1)

        final_mask = create_filter_mask(
            pts3d_conf=predictions["pts3d_conf"][0].detach().cpu().numpy(),
            depth_preds=predictions["depth"][0].detach().cpu().numpy(),
            normal_preds=predictions["normals"][0].detach().cpu().numpy(),
            sky_mask=np.ones((S, H, W), dtype=bool)
        )  # [S, H, W]

        if True:
            pts_list = []
            pts_colors_list = []
            
            for i in range(S):
                pts = predictions["pts3d"][0, i]  # [H,W,3]
                img_colors = imgs[0, i].permute(1, 2, 0)  # [H, W, 3]
                img_colors = (img_colors * 255).to(torch.uint8)
                
                pts_list.append(pts.reshape(-1, 3))
                pts_colors_list.append(img_colors.reshape(-1, 3))

            all_pts = torch.cat(pts_list, dim=0)
            all_colors = torch.cat(pts_colors_list, dim=0)

            final_mask_flat = final_mask.reshape(-1)  # Flatten to [S*H*W]
            final_mask_torch = torch.from_numpy(final_mask_flat).to(all_pts.device)
            filtered_pts = all_pts[final_mask_torch]
            filtered_colors = all_colors[final_mask_torch]
            
            save_scene_ply(hunyuanworld_output_dir / "pts_from_pointmap.ply", filtered_pts, filtered_colors)

        img = Image.open(image_files[0])
        raw_image_size = (img.height, img.width)
        images_8u = (imgs[0].permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()  # [S, H, W, 3]

        E4x4_cws = predictions['camera_poses'][0].detach().cpu().numpy()
        E4x4_wcs = np.zeros_like(E4x4_cws)
        S = E4x4_cws.shape[0]
        for i in range(S):
            E4x4_wc = np.linalg.inv(E4x4_cws[i])
            E4x4_wcs[i] = E4x4_wc

        results = {
          "image_num": len(image_files),
          "raw_image_size": raw_image_size,
          "image_names": [Path(f).name for f in image_files],
          "forward_results": {
            'imgs': images_8u,
            'poses': E4x4_wcs, 
            'intrs': predictions['camera_intrs'][0].detach().cpu().numpy(), 
            'depths': predictions['depth'][0].detach().cpu().numpy(), 
            'depths_conf': predictions["pts3d_conf"][0].detach().cpu().numpy(),
            'normals': predictions["normals"][0].detach().cpu().numpy(),
            'pts3d': predictions["pts3d"][0].detach().cpu().numpy(),
            'pts3d_conf': predictions["pts3d_conf"][0].detach().cpu().numpy(),
            'masks': final_mask,
          },
        }

        # save results to output_path / "results.pkl"
        results_path = hunyuanworld_output_dir / "results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Saved results to {results_path}")

        # print results shapes
        for key, value in results["forward_results"].items():
            if isinstance(value, np.ndarray):
                logger.info(f"{key}: {value.shape}")
            else:
                logger.info(f"{key}: {type(value)}")

        logger.info(f"HunyuanWorld-Mirror Reconstruction completed. Results saved to {output_path}")

        return results