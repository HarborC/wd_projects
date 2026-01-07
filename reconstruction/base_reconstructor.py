"""
Base reconstructor module.

This module defines the abstract base class for all 3D reconstruction implementations.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union
import logging
import json
import numpy as np
import cv2
import shutil
from PIL import Image
from plyfile import PlyData, PlyElement
import os

logger = logging.getLogger(__name__)

def save_points_ply(path: Path, pts_np: np.ndarray, cols_np: np.ndarray) -> None:
    """Save point cloud to PLY format from numpy arrays"""
    vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), 
                    ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertex_elements = np.empty(len(pts_np), dtype=vertex_dtype)
    vertex_elements["x"] = pts_np[:, 0]
    vertex_elements["y"] = pts_np[:, 1]
    vertex_elements["z"] = pts_np[:, 2]
    vertex_elements["red"] = cols_np[:, 0]
    vertex_elements["green"] = cols_np[:, 1]
    vertex_elements["blue"] = cols_np[:, 2]
    
    # Write PLY file
    PlyData([PlyElement.describe(vertex_elements, "vertex")]).write(str(path))

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

class BaseReconstructor(ABC):
    """
    Abstract base class for 3D reconstruction implementations.

    All reconstructor implementations should inherit from this class and implement
    the required methods.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the base reconstructor.

        Args:
            device (str, optional): Device to run the model on ('cuda' or 'cpu').
                                    Defaults to auto-detect.
        """
        import torch
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def _load_model(self):
        """
        Load the reconstruction model for this backend.

        This method should set self.model with the loaded model.
        """
        pass

    @abstractmethod
    def _process(self, input_dir: Union[str, Path], output_dir: Union[str, Path], image_files: Optional[list] = None, **kwargs):
        """
        Process images for 3D reconstruction.

        Args:
            input_dir (str or Path): Directory containing input images.
            output_dir (str or Path): Directory to save reconstruction results.
            image_files (list, optional): List of image file paths to process.
            **kwargs: Additional backend-specific parameters.
        """
        pass

    def process_directory(self, input_dir: Union[str, Path], output_dir: Union[str, Path], **kwargs):
        """
        Process all images in a directory for 3D reconstruction.

        Args:
            input_dir (str or Path): Directory containing input images.
            output_dir (str or Path): Directory to save reconstruction results.
            **kwargs: Additional backend-specific parameters.
        """
        image_files = self._load_imagefiles(Path(input_dir))
        results = self._process(input_dir, output_dir, image_files, **kwargs)
        self._copy_images(image_files, output_dir)
        self._save_depth_maps(results, output_dir)
        self._save_normal_maps(results, output_dir)
        self._save_as_colmap_format(results, output_dir)

    def _copy_images(self, image_paths: list, output_path: Union[str, Path]):
        """
        Copy images to the output directory.

        Args:
            image_paths (list): List of image file paths.
            output_path (str or Path): Directory to copy images to.
        """
        output_path = Path(output_path)
        images_output_path = output_path / "images"
        images_output_path.mkdir(parents=True, exist_ok=True)

        for img_path in image_paths:
            img_name = Path(img_path).name
            shutil.copy(img_path, images_output_path / img_name)

    def _save_depth_maps(self, results: Dict, output_path: Union[str, Path]):
        """
        Save depth map in both PNG (visualization) and NPY (raw data) formats.

        Args:
            depth (np.ndarray): Depth map array (H, W).
            output_path (str or Path): Base output path (without extension).
            save_npy (bool): Whether to save .npy file with raw depth values.
            save_png (bool): Whether to save .png file with visualization.
            confidence (np.ndarray, optional): Confidence map (H, W).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        depth_output_path = output_path / "depths"
        depth_output_path.mkdir(parents=True, exist_ok=True)

        image_num = results['image_num']
        raw_height, raw_width = results['raw_image_size']
        for i in range(image_num):
            img_name = Path(results['image_names'][i])
            depth = results['forward_results']['depths'][i]    
            depth_upsampled = cv2.resize(depth, (raw_width, raw_height), interpolation=cv2.INTER_CUBIC)
            np.save(depth_output_path / f"{img_name.stem}.npy", depth_upsampled)
        
            depth_norm = (depth_upsampled - depth_upsampled.min()) / (depth_upsampled.max() - depth_upsampled.min() + 1e-6)
            depth_vis = (depth_norm * 255).astype(np.uint8)
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(depth_output_path / f"{img_name.stem}.png"), depth_vis_color)

            if results['forward_results']['depths_conf'] is not None:
                confidence = results['forward_results']['depths_conf'][i]
                confidence_upsampled = cv2.resize(confidence, (raw_width, raw_height), interpolation=cv2.INTER_CUBIC)
                np.save(depth_output_path / f"{img_name.stem}_conf.npy", confidence_upsampled)

    def _save_normal_maps(self, results: Dict, output_path: Union[str, Path]):
        if results['forward_results']['normals'] is None:
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        normal_output_path = output_path / "normals"
        normal_output_path.mkdir(parents=True, exist_ok=True)

        image_num = results['image_num']
        raw_height, raw_width = results['raw_image_size']
        for i in range(image_num):
            img_name = Path(results['image_names'][i])

            # Get normal map at processed resolution
            normal_np = results['forward_results']['normals'][i]  # (H, W, 3)
            normal_upsampled = cv2.resize(normal_np, (raw_width, raw_height), interpolation=cv2.INTER_CUBIC)

            # Convert back to tensor and save
            np.save(normal_output_path / f"{img_name.stem}.npy", normal_upsampled)
            img = (np.clip((normal_upsampled + 1.0) * 0.5, 0, 1) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(str(normal_output_path / f"{img_name.stem}.png"))

    def _load_imagefiles(self, input_path: Path) -> list:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted([str(p) for p in input_path.iterdir() if p.suffix.lower() in image_extensions])
        logger.info(f"Found {len(image_files)} images in {input_path}.")
        return image_files

    def _load_image(self, input_path: Path):
        image = np.array(Image.open(img_path).convert("RGB"))
        return image

    def _save_as_colmap_format(self, results: Dict, output_path: Union[str, Path]):
        """
        Save reconstruction results in COLMAP format.

        Args:
            output_path (str or Path): Directory to save COLMAP format files.
            **kwargs: Additional parameters for COLMAP format saving.
        """
        output_path = Path(output_path)
        sparse_dir = output_path / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        points, colors = self._depths_to_world_points_with_colors(
            results['forward_results']['depths'],
            results['forward_results']['intrs'],
            results['forward_results']['poses'],
            results['forward_results']['imgs'],
            results['forward_results']['masks'][..., None],
        )

        num_points = len(points)
        logger.info(f"Exporting to COLMAP with {num_points} points")
        num_frames = results['image_num']
        h, w = results['forward_results']['imgs'].shape[1:3]
        points_xyf = self._create_xyf(num_frames, h, w)
        points_xyf = points_xyf[results['forward_results']['masks']]
        points_xyf = points_xyf.reshape(-1, 3).astype(np.float64, copy=False)
        orig_h, orig_w = results['raw_image_size']
        image_paths = results['image_names']

        # 2. Set Reconstruction
        import pycolmap
        reconstruction = pycolmap.Reconstruction()

        point3d_ids = []
        for vidx in range(num_points):
            point3d_id = reconstruction.add_point3D(points[vidx], pycolmap.Track(), colors[vidx])
            point3d_ids.append(point3d_id)

        for fidx in range(num_frames):
            sx = float(orig_w) / float(w)
            sy = float(orig_h) / float(h)

            intrinsic = results['forward_results']['intrs'][fidx].copy()
            intrinsic[:1] *= sx
            intrinsic[1:2] *= sy

            pycolmap_intri = np.array(
                [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
            )

            extrinsic = results['forward_results']['poses'][fidx]
            cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(extrinsic[:3, :3]), extrinsic[:3, 3])

            # set and add camera
            camera = pycolmap.Camera()
            camera.camera_id = fidx + 1
            camera.model = pycolmap.CameraModelId.PINHOLE
            camera.width = orig_w
            camera.height = orig_h
            camera.params = pycolmap_intri
            reconstruction.add_camera(camera)

            # set and add rig (from camera)
            rig = pycolmap.Rig()
            rig.rig_id = camera.camera_id
            rig.add_ref_sensor(camera.sensor_id)
            reconstruction.add_rig(rig)

            # set image
            image = pycolmap.Image()
            image.image_id = fidx + 1
            image.camera_id = camera.camera_id

            # set and add frame (from image)
            frame = pycolmap.Frame()
            frame.frame_id = image.image_id
            frame.rig_id = camera.camera_id
            frame.add_data_id(image.data_id)
            frame.rig_from_world = cam_from_world
            reconstruction.add_frame(frame)

            # set point2d and update track
            point2d_list = []
            points_in_frame = points_xyf[:, 2].astype(np.int32) == fidx
            for vidx in np.where(points_in_frame)[0]:
                point2d = points_xyf[vidx][:2]
                point2d[0] *= sx
                point2d[1] *= sy
                point3d_id = point3d_ids[vidx]
                point2d_list.append(pycolmap.Point2D(point2d, point3d_id))
                reconstruction.point3D(point3d_id).track.add_element(
                    image.image_id, len(point2d_list) - 1
                )

            # set and add image
            image.frame_id = image.image_id
            image.name = os.path.basename(image_paths[fidx])
            image.points2D = pycolmap.Point2DList(point2d_list)
            reconstruction.add_image(image)

        # 3. Export
        reconstruction.write(sparse_dir)
        reconstruction.write_text(sparse_dir)

        # Save points3D.ply
        save_points_ply(sparse_dir / "points3D.ply", points, colors)
        logger.info(f"Saved COLMAP reconstruction to {sparse_dir}")

    def _depths_to_world_points_with_colors(self,
        depth: np.ndarray,
        K: np.ndarray,
        ext_w2c: np.ndarray,
        images_u8: np.ndarray,
        mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        For each frame, transform (u,v,1) through K^{-1} to get rays,
        multiply by depth to camera frame, then use (w2c)^{-1} to transform to world frame.
        Simultaneously extract colors.
        """
        N, H, W, _ = depth.shape
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        ones = np.ones_like(us)
        pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

        pts_all, col_all = [], []

        for i in range(N):
            d = depth[i]  # (H,W)
            valid = np.isfinite(d) & (d > 0)
            if mask is not None:
                valid &= mask[i]
            if not np.any(valid):
                continue

            d_flat = d.reshape(-1)
            vidx = np.flatnonzero(valid.reshape(-1))

            K_inv = np.linalg.inv(K[i])  # (3,3)
            c2w = np.linalg.inv(as_homogeneous44(ext_w2c[i]))  # (4,4)

            rays = K_inv @ pix[vidx].T  # (3,M)
            Xc = rays * d_flat[vidx][None, :]  # (3,M)
            Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
            Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)

            cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M,3)

            pts_all.append(Xw)
            col_all.append(cols)

        if len(pts_all) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)

    def _create_xyf(self, num_frames, height, width):
        """
        Creates a grid of pixel coordinates and frame indices (fidx) for all frames.
        """
        # Create coordinate grids for a single frame
        y_grid, x_grid = np.indices((height, width), dtype=np.int32)
        x_grid = x_grid[np.newaxis, :, :]
        y_grid = y_grid[np.newaxis, :, :]

        # Broadcast to all frames
        x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
        y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

        # Create frame indices and broadcast
        f_idx = np.arange(num_frames, dtype=np.int32)[:, np.newaxis, np.newaxis]
        f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

        # Stack coordinates and frame indices
        points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

        return points_xyf