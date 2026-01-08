# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional, Dict, Tuple

import cv2
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm
from rich.console import Console

import datasets.colmap_parsing_utils as colmap_utils
from datasets.colmap_utils import (
    parse_colmap_camera_params,
    auto_orient_and_center_poses,
)

MAX_AUTO_RESOLUTION = 1600
CONSOLE = Console(width=120)


def status(msg: str, spinner: str = "bouncingBall", verbose: bool = False):
    """A context manager for logging status updates."""
    if verbose:
        return nullcontext()
    return CONSOLE.status(msg, spinner=spinner)


def _resize_depth_worker(args):
    """Worker function for parallel depth map resizing (NPY)."""
    path_in, path_out, factor, mode, interp = args
    if path_out.exists():
        return
    
    try:
        # Handle NPY depth maps (float data)
        img = np.load(path_in)
        h, w = img.shape[:2]
        
        if mode == "floor":
            new_w, new_h = math.floor(w / factor), math.floor(h / factor)
        elif mode == "round":
            new_w, new_h = round(w / factor), round(h / factor)
        elif mode == "ceil":
            new_w, new_h = math.ceil(w / factor), math.ceil(h / factor)
        else:
            new_w, new_h = w // factor, h // factor

        # Use NEAREST for depth to avoid interpolation artifacts (e.g. smoothing over edges)
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        path_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(path_out, resized)
    except Exception as e:
        print(f"Error processing {path_in}: {e}")


def _resize_image_worker(args):
    """Worker function for parallel image resizing."""
    path_in, path_out, factor, mode, interp = args
    if path_out.exists():
        return
    
    try:
        # Use PIL for logic consistency with mode (requires size)
        img = Image.open(path_in)
        w, h = img.size
        
        if mode == "floor":
            new_w, new_h = math.floor(w / factor), math.floor(h / factor)
        elif mode == "round":
            new_w, new_h = round(w / factor), round(h / factor)
        elif mode == "ceil":
            new_w, new_h = math.ceil(w / factor), math.ceil(h / factor)
        else:
            new_w, new_h = w // factor, h // factor

        img = img.resize((new_w, new_h), interp)
        path_out.parent.mkdir(parents=True, exist_ok=True)
        img.save(path_out, quality=95)
    except Exception as e:
        print(f"Error processing {path_in}: {e}")


class ColmapParser:
    """An adapted version of nerfstudio ColmapDataParser with cleaner structure and parallel processing."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        scale_factor: float = 1.0,
        orientation_method: Literal["pca", "up", "vertical", "none"] = "up",
        center_method: Literal["poses", "focus", "none"] = "poses",
        auto_scale_poses: bool = True,
        test_every: int = 8,
        downscale_rounding_mode: Literal["floor", "round", "ceil"] = "floor",
        adjust_scene: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.scale_factor = scale_factor
        self.downscale_rounding_mode = downscale_rounding_mode
        self.orientation_method = orientation_method
        self.center_method = center_method
        self.auto_scale_poses = auto_scale_poses
        self.adjust_scene = adjust_scene
        
        # Initialize containers
        self.cameras = {}
        self.camera_ids = []
        self.image_names = []
        self.image_paths = []
        self.depth_paths = None
        self.mask_paths = None
        self.Ks_dict = {}
        self.params_dict = {}
        self.imsize_dict = {}
        self.camtoworlds = None
        self.camtoworlds_gt = None
        self.points = None
        self.points_err = None
        self.points_rgb = None
        self.transform = np.eye(4)
        self.scene_scale = 1.0

        # Load Data
        self._load_colmap_data()
        self._process_poses_and_intrinsics()
        self._process_images()
        self._process_depths()
        self._process_points()
        self._undistort_cameras()
        self._validate_and_fix_camera_centers()

    def _get_colmap_dir(self) -> Path:
        colmap_dir = self.data_dir / "sparse/0/"
        if not colmap_dir.exists():
            colmap_dir = self.data_dir / "sparse"
        if not colmap_dir.exists():
            raise FileNotFoundError(f"COLMAP directory not found in {self.data_dir}")
        return colmap_dir

    def _load_colmap_data(self):
        colmap_dir = self._get_colmap_dir()
        
        # Load Cameras
        if (colmap_dir / "cameras.txt").exists():
            self.raw_cameras = colmap_utils.read_cameras_text(colmap_dir / "cameras.txt")
            self.raw_images = colmap_utils.read_images_text(colmap_dir / "images.txt")
        elif (colmap_dir / "cameras.bin").exists():
            self.raw_cameras = colmap_utils.read_cameras_binary(colmap_dir / "cameras.bin")
            self.raw_images = colmap_utils.read_images_binary(colmap_dir / "images.bin")
        else:
            raise ValueError(f"Could not find cameras.txt/bin in {colmap_dir}")

        # Load GT Poses (Optional)
        self.raw_images_gt = None
        if (colmap_dir / "images_gt.bin").exists():
            self.raw_images_gt = colmap_utils.read_images_binary(colmap_dir / "images_gt.bin")
        elif (colmap_dir / "images_gt.txt").exists():
            self.raw_images_gt = colmap_utils.read_images_text(colmap_dir / "images_gt.txt")

    def _process_poses_and_intrinsics(self):
        # Sort images by ID
        ordered_im_id = sorted(self.raw_images.keys())
        
        # Parse camera parameters
        parsed_cameras = {
            cid: parse_colmap_camera_params(cdata) 
            for cid, cdata in self.raw_cameras.items()
        }

        w2c_mats = []
        w2c_mats_gt = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        for im_id in ordered_im_id:
            im = self.raw_images[im_id]
            
            # W2C extraction
            rot = colmap_utils.qvec2rotmat(im.qvec)
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # GT W2C extraction
            if self.raw_images_gt is not None:
                im_gt = self.raw_images_gt.get(im_id + 1) # Note: +1 hack preserved
                if im_gt:
                    rot_gt = colmap_utils.qvec2rotmat(im_gt.qvec)
                    trans_gt = im_gt.tvec.reshape(3, 1)
                    w2c_gt_mat = np.concatenate([np.concatenate([rot_gt, trans_gt], 1), bottom], axis=0)
                    w2c_mats_gt.append(w2c_gt_mat)

            # Intrinsics
            self.camera_ids.append(im.camera_id)
            cam = parsed_cameras[im.camera_id]
            self._parse_camera_intrinsics(im.camera_id, cam)

        print(f"[Parser] {len(self.raw_images)} images, taken by {len(set(self.camera_ids))} cameras.")
        
        if not w2c_mats:
            raise ValueError("No valid images found.")

        # C2W conversion
        w2c_mats = np.stack(w2c_mats, axis=0)
        self.camtoworlds = np.linalg.inv(w2c_mats)
        
        # Sort based on image names to ensure consistency
        self.image_names = [self.raw_images[k].name for k in ordered_im_id]
        inds = np.argsort(self.image_names)
        
        self.image_names = [self.image_names[i] for i in inds]
        self.camtoworlds = self.camtoworlds[inds]
        self.camera_ids = [self.camera_ids[i] for i in inds]
        
        # Process GT poses
        if w2c_mats_gt:
            w2c_mats_gt = np.stack(w2c_mats_gt, axis=0)
            self.camtoworlds_gt = np.linalg.inv(w2c_mats_gt)
            self.camtoworlds_gt = self.camtoworlds_gt[inds]

        # Auto orientation and scaling
        self._align_and_scale_poses()

    def _parse_camera_intrinsics(self, camera_id, cam):
        fx, fy, cx, cy = cam["fl_x"], cam["fl_y"], cam["cx"], cam["cy"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K[:2, :] /= self.factor
        self.Ks_dict[camera_id] = K
        self.imsize_dict[camera_id] = (cam["w"] // self.factor, cam["h"] // self.factor)

        # Distortion params
        params = np.empty(0, dtype=np.float32)
        model = cam["model"]
        if model in ["SIMPLE_RADIAL"]:
            params = np.array([cam["k1"]], dtype=np.float32)
        elif model in ["RADIAL"]:
            params = np.array([cam["k1"], cam["k2"], 0, 0], dtype=np.float32)
        elif model in ["OPENCV"]:
            params = np.array([cam["k1"], cam["k2"], cam["p1"], cam["p2"]], dtype=np.float32)
        elif model in ["OPENCV_FISHEYE"]:
            params = np.array([cam["k1"], cam["k2"], cam["k3"], cam["k4"]], dtype=np.float32)
        
        self.params_dict[camera_id] = params

    def _align_and_scale_poses(self):
        scale_factor = 1.0
        transform_matrix = np.eye(4)
        
        if self.adjust_scene:
            self.camtoworlds, transform_matrix = auto_orient_and_center_poses(
                self.camtoworlds, method=self.orientation_method, center_method=self.center_method
            )
            if self.auto_scale_poses:
                scale_factor /= float(np.max(np.abs(self.camtoworlds[:, :3, 3])))
            scale_factor *= self.scale_factor
        
        self.camtoworlds[:, :3, 3] *= scale_factor
        self.transform = transform_matrix
        self.final_scale_factor = scale_factor # Store for points usage

        # Apply same transform to GT
        if self.camtoworlds_gt is not None:
             # Re-align GT separately (as in original code) or use same transform?
             # Original code realigns GT independently which is quirky but we preserve it.
             self.camtoworlds_gt, _ = auto_orient_and_center_poses(
                self.camtoworlds_gt, method=self.orientation_method, center_method=self.center_method
            )
             # Note: GT scaling logic in original was slightly duplicated, simplifying here
             gt_scale = 1.0
             if self.auto_scale_poses:
                 gt_scale /= float(np.max(np.abs(self.camtoworlds_gt[:, :3, 3])))
             gt_scale *= self.scale_factor
             self.camtoworlds_gt[:, :3, 3] *= gt_scale

    def _process_images(self):
        """Handle image downscaling and path resolution."""
        # Load extended metadata
        self.extconf = {"spiral_radius_scale": 0.1, "no_factor_suffix": False}
        extconf_file = self.data_dir / "ext_metadata.json"
        if extconf_file.exists():
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Determine folders
        suffix = f"_{self.factor}" if self.factor > 1 and not self.extconf["no_factor_suffix"] else ""
        
        colmap_img_dir = self.data_dir / "images"
        target_img_dir = self.data_dir / ("images" + suffix)
        
        # Get all image files relative to root
        all_colmap_images = sorted([f.relative_to(colmap_img_dir) for f in colmap_img_dir.rglob("*") if f.is_file()])
        
        if not target_img_dir.exists():
            print(f"Target image dir {target_img_dir} does not exist. Downscal, interpolation=Image.LANCZOS")
            
        # Also handle test images if present
        test_img_dir = self.data_dir / "images_test"
        target_test_img_dir = self.data_dir / ("images_test" + suffix)
        if test_img_dir.exists() and not target_test_img_dir.exists():
             test_files = sorted([f.relative_to(test_img_dir) for f in test_img_dir.rglob("*") if f.is_file()])
             self._downscale_images_parallel(
                 test_img_dir, target_test_img_dir, test_files, self.factor, interpolation=Image.LANCZOS
             )

        # Map current sorted image names to actual paths
        # Note: image_names are sorted by name earlier
        self.image_paths = [str(target_img_dir / name) for name in self.image_names]
        
        # Masks
        mask_dir = self.data_dir / "masks"
        if mask_dir.exists():
            self.mask_paths = [str(mask_dir / name) for name in self.image_names]
            print("Got masks path!")

    def _process_depths(self):
        """Handle depth maps downscaling and path resolution."""
        # Check if we need to load depths
        suffix = f"_{self.factor}" if self.factor > 1 and not self.extconf.get("no_factor_suffix", False) else ""
        
        colmap_depth_dir = self.data_dir / "depths"
        target_depth_dir = self.data_dir / ("depths" + suffix)

        if colmap_depth_dir.exists():
            if not target_depth_dir.exists():
                 print(f"Target depth dir {target_depth_dir} does not exist. Downscaling...")
                 # Only look for .npy files as requested
                 all_depths = sorted([f.relative_to(colmap_depth_dir) for f in colmap_depth_dir.rglob("*.npy") if f.is_file()])
                 self._downscale_depths_parallel(
                     colmap_depth_dir, target_depth_dir, all_depths, self.factor
                 )
            
            # Map depth paths matching image names
            self.depth_paths = []
            for name in self.image_names:
                # name is relative path from images dir (e.g. point_0000.jpg or sub/image.jpg)
                stem = Path(name).stem
                parent = Path(name).parent
                
                # Force mapping to .npy
                p_npy = target_depth_dir / parent / f"{stem}.npy"
                self.depth_paths.append(str(p_npy))
                    
            print("Got depth paths!")

    def _downscale_depths_parallel(self, src_dir: Path, dst_dir: Path, file_list: List[Path], factor: int):
        """Downscale depth maps using parallel processing."""
        
        # Prepare tasks
        worker_tasks = []
        for rel_path in file_list:
            src = src_dir / rel_path
            dst = dst_dir / rel_path
            worker_tasks.append((src, dst, factor, self.downscale_rounding_mode, cv2.INTER_NEAREST))

        with status(msg=f"[bold yellow]Downscaling {len(file_list)} depths (x{factor})...", spinner="growVertical"):
            # Use top-level worker to ensure pickling works on all platforms
            with ProcessPoolExecutor() as executor:
                list(tqdm(executor.map(_resize_depth_worker, worker_tasks), total=len(worker_tasks)))

    def _downscale_images_parallel(self, src_dir: Path, dst_dir: Path, file_list: List[Path], factor: int, interpolation=Image.LANCZOS):
        """Downscale images using parallel processing."""
        
        # Prepare tasks
        worker_tasks = []
        for rel_path in file_list:
            src = src_dir / rel_path
            dst = dst_dir / rel_path
            worker_tasks.append((src, dst, factor, self.downscale_rounding_mode, interpolation))

        with status(msg=f"[bold yellow]Downscaling {len(file_list)} images (x{factor})...", spinner="growVertical"):
            # Use top-level worker to ensure pickling works on all platforms
            with ProcessPoolExecutor() as executor:
                list(tqdm(executor.map(_resize_image_worker, worker_tasks), total=len(worker_tasks)))

    def _process_points(self):
        colmap_dir = self._get_colmap_dir()
        if (colmap_dir / "points3D.bin").exists():
            colmap_points = colmap_utils.read_points3D_binary(colmap_dir / "points3D.bin")
        elif (colmap_dir / "points3D.txt").exists():
            colmap_points = colmap_utils.read_points3D_text(colmap_dir / "points3D.txt")
        else:
             # Warning instead of error? Code says error.
             raise ValueError(f"Could not find points3D in {colmap_dir}")

        points = np.array([p.xyz for p in colmap_points.values()], dtype=np.float32)
        # Apply transform
        points = (
            np.concatenate((points, np.ones_like(points[..., :1])), -1)
            @ self.transform.T
        )
        points *= self.final_scale_factor

        self.points = points
        self.points_rgb = np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8)
        self.points_err = np.array([p.error for p in colmap_points.values()], dtype=np.float32)

        # Scene scale calculation
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

    def _undistort_cameras(self):
        self.mapx_dict = {}
        self.mapy_dict = {}
        self.roi_undist_dict = {}
        
        for camera_id, params in self.params_dict.items():
            if len(params) == 0:
                continue
            
            K = self.Ks_dict[camera_id]
            w, h = self.imsize_dict[camera_id]
            
            # OpenCV Undistortion Setup
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(K, params, (w, h), 0)
            mapx, mapy = cv2.initUndistortRectifyMap(K, params, None, K_undist, (w, h), cv2.CV_32FC1)
            
            self.Ks_dict[camera_id] = K_undist
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist

    def _validate_and_fix_camera_centers(self):
        """Fix Center of Projection (BAD-Gaussians / Deblur-NeRF fix)."""
        if not self.Ks_dict: return
            
        # Check first camera
        first_cam_id = list(self.Ks_dict.keys())[0]
        K = self.Ks_dict[first_cam_id]
        w, h = self.imsize_dict[first_cam_id]
        
        ideal_cx, ideal_cy = w / 2.0, h / 2.0
        cx, cy = K[0, 2], K[1, 2]
        
        scale_x, scale_y = 1.0, 1.0
        
        if abs(cx - ideal_cx) / ideal_cx > 0.3:
            scale_x = cx / ideal_cx
            print(f"[WARN] Inspecting cx outlier: {cx} vs {ideal_cx}. Scale: {scale_x}")
            
        if abs(cy - ideal_cy) / ideal_cy > 0.3:
            scale_y = cy / ideal_cy
            print(f"[WARN] Inspecting cy outlier: {cy} vs {ideal_cy}. Scale: {scale_y}")

        def get_nearest_int_scale(s):
            return round(1/s) if s < 1 else 1/round(s)

        if scale_x != 1.0 or scale_y != 1.0:
            inv_scale = get_nearest_int_scale(scale_x) # Assuming uniform scaling error
            print(f"[WARN] Correcting camera intrinsics with factor {inv_scale}")
            
            for cid in self.Ks_dict:
                self.Ks_dict[cid][0, 0] *= inv_scale # fx
                self.Ks_dict[cid][0, 2] *= inv_scale # cx
                self.Ks_dict[cid][1, 1] *= inv_scale # fy
                self.Ks_dict[cid][1, 2] *= inv_scale # cy
                
                ow, oh = self.imsize_dict[cid]
                self.imsize_dict[cid] = (int(ow * inv_scale), int(oh * inv_scale))

if __name__ == "__main__":
    import argparse
    import imageio.v2 as imageio

    from datasets.colmap import Dataset


    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = ColmapParser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
