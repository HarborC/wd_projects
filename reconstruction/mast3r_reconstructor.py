"""
MASt3R reconstructor module.

This module implements the BaseReconstructor interface using MASt3R
for multi-view stereo 3D reconstruction.
"""
import logging
import shutil
import cv2
import numpy as np
import torch
import sys
from pathlib import Path
from typing import Optional, Union
from PIL import Image
from time import time
import os

from reconstruction.base_reconstructor import BaseReconstructor

logger = logging.getLogger(__name__)

class MASt3RReconstructor(BaseReconstructor):
    """
    MASt3R-based reconstructor.

    This class uses MASt3R for multi-view stereo matching and 3D reconstruction.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    ):
        """
        Initialize the MASt3R reconstructor.

        Args:
            device (str, optional): Device to run the model on.
            model_name (str): Path to MASt3R checkpoint.
        """
        super().__init__(device)
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load the MASt3R model."""

        mast3r_path = Path(__file__).parent / "mast3r"
        sys.path.insert(0, str(mast3r_path))

        from mast3r.model import AsymmetricMASt3R

        if not Path(self.model_name).exists():
            os.system('mkdir -p mast3r && wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/')

        logger.info(f"Loading MASt3R model from: {self.model_name}")
        self.model = AsymmetricMASt3R.from_pretrained(self.model_name).to(self.device)
        logger.info("MASt3R model loaded successfully")

    def _process(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        image_files: Optional[list] = None,
        image_size=512,
        n_views=None,
        schedule='cosine',
        lr=0.01,
        niter=300,
        min_conf_thr=5,
        llffhold=8,
        co_vis_dsp=False,
        depth_thre=0.01,
        conf_aware_ranking=False,
        focal_avg=False,
        infer_video=False,
        **kwargs
    ):
        """
        Process images using MASt3R reconstruction.

        Args:
            input_dir (str or Path): Directory containing input images.
            output_dir (str or Path): Directory to save reconstruction results.
            image_size (int): Size to resize images to.
            n_views (int, optional): Number of views to use.
            schedule (str): Learning rate schedule.
            lr (float): Learning rate.
            niter (int): Number of iterations.
            min_conf_thr (float): Minimum confidence threshold.
            llffhold (int): LLFF hold parameter.
            co_vis_dsp (bool): Use co-visibility masking.
            depth_thre (float): Depth threshold for co-visibility.
            conf_aware_ranking (bool): Use confidence-aware ranking.
            focal_avg (bool): Use average focal length.
            infer_video (bool): Infer video.
            **kwargs: Additional parameters.
        """
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.utils.device import to_numpy
        from dust3r.utils.geometry import inv
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        from utils.sfm_utils import (save_intrinsics, save_extrinsic, save_points3D,
                                     get_sorted_image_files, load_images, compute_co_vis_masks)

        # TODO
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Setup backend-specific directory
        mast3r_output_dir = output_path / 'mast3r'
        mast3r_output_dir.mkdir(parents=True, exist_ok=True)

        # Load images
        if image_files is None:
            image_files = self._load_imagefiles(input_path)

        if not image_files:
            logger.warning(f"No images found in {input_path}")
            return

        # Setup paths for backend-specific sparse output
        sparse_0_path = mast3r_output_dir / 'sparse' / '0'
        sparse_0_path.mkdir(parents=True, exist_ok=True)

        # Use all images for reconstruction
        train_img_files = image_files
        image_suffix = Path(image_files[0]).suffix

        images, org_imgs_shape = load_images(train_img_files, size=image_size)

        start_time = time()
        logger.info(f'>> Making pairs...')
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        logger.info(f'>> Inference...')
        output = inference(pairs, self.model, self.device, batch_size=1, verbose=True)
        logger.info(f'>> Global alignment...')
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr, focal_avg=focal_avg)

        # Extract scene information
        extrinsics_w2c = inv(to_numpy(scene.get_im_poses()))
        intrinsics = to_numpy(scene.get_intrinsics())
        focals = to_numpy(scene.get_focals())
        imgs = np.array(scene.imgs)
        pts3d = np.array(to_numpy(scene.get_pts3d()))
        depthmaps = to_numpy(scene.im_depthmaps.detach().cpu().numpy())
        imshapes = to_numpy(torch.tensor(scene.imshapes))
        confs = np.array([param.detach().cpu().numpy() for param in scene.im_conf])

        if conf_aware_ranking:
            logger.info(f'>> Confiden-aware Ranking...')
            avg_conf_scores = confs.mean(axis=(1, 2))
            sorted_conf_indices = np.argsort(avg_conf_scores)[::-1]
        else:
            sorted_conf_indices = np.arange(len(images))

        # Calculate the co-visibility mask
        logger.info(f'>> Calculate the co-visibility mask...')
        if depth_thre > 0:
            overlapping_masks = compute_co_vis_masks(sorted_conf_indices, depthmaps, pts3d, intrinsics, extrinsics_w2c, imgs.shape, depth_threshold=depth_thre)
            overlapping_masks = ~overlapping_masks
        else:
            co_vis_dsp = False
            overlapping_masks = None

        end_time = time()
        Train_Time = end_time - start_time
        logger.info(f"Time taken: {Train_Time} seconds")

        # Save results to mast3r/ directory
        focals = np.repeat(focals[0], len(images))
        logger.info(f'>> Saving results to {sparse_0_path}...')

        save_extrinsic(sparse_0_path, extrinsics_w2c, train_img_files, image_suffix)
        save_intrinsics(sparse_0_path, focals, org_imgs_shape, imgs.shape, save_focals=True)
        pts_num = save_points3D(sparse_0_path, imgs, pts3d, confs.reshape(pts3d.shape[0], -1), overlapping_masks, use_masks=co_vis_dsp, save_all_pts=True, save_txt_path=mast3r_output_dir, depth_threshold=depth_thre)

        # Copy images to unified images/ directory using base class method
        logger.info(f"Copying images to {output_path / 'images'}")
        self._copy_images(train_img_files, output_path)

        # Save depth maps to both mast3r/depths and unified depths/
        mast3r_depths_dir = mast3r_output_dir / "depths"
        mast3r_depths_dir.mkdir(parents=True, exist_ok=True)

        depths_dir = output_path / "depths"
        depths_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving depth maps to {depths_dir}")

        for i, (depth, imshape) in enumerate(zip(depthmaps, imshapes)):
            # upsample depth to original image size
            orig_w, orig_h = Image.open(train_img_files[i]).size
            depth = depth.reshape(imshape[0], imshape[1])
            depth_raw = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

            img_name = Path(train_img_files[i]).stem

            # Save to both directories using unified method
            self._save_depth_map(depth_raw, mast3r_depths_dir / img_name)
            self._save_depth_map(depth_raw, depths_dir / img_name)

        # Copy only COLMAP standard files to unified sparse/0 directory
        unified_sparse_dir = output_path / "sparse" / "0"
        unified_sparse_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Copying COLMAP files to {unified_sparse_dir}")

        colmap_files = ['cameras.txt', 'images.txt', 'points3D.txt',
                       'cameras.bin', 'images.bin', 'points3D.bin']

        for filename in colmap_files:
            src_file = sparse_0_path / filename
            if src_file.exists():
                shutil.copy(src_file, unified_sparse_dir / filename)

        # Copy confidence_dsp.npy for InstantSplat compatibility
        conf_dsp_src = sparse_0_path / 'confidence_dsp.npy'
        if conf_dsp_src.exists():
            shutil.copy(conf_dsp_src, unified_sparse_dir / 'confidence_dsp.npy')
            logger.info(f"Copied confidence_dsp.npy to {unified_sparse_dir}")

        logger.info(f'[INFO] MASt3R Reconstruction completed.')
