import cv2
import numpy as np
import torch
from typing import Any, Dict, List, Optional

from .colmap_dataparser import ColmapParser


class Dataset:
    """A standard dataset class for loading COLMAP data."""

    def __init__(
        self,
        parser: ColmapParser,
        split: str = "train",
        patch_size: Optional[int] = None,
        train_indices: Optional[List[int]] = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        
        # --- Indices Setup ---
        all_indices = np.arange(len(self.parser.image_names))
        if split == "train":
            # Prefer provided train_indices; fallback to parser.train_indices
            cfg_train_indices = train_indices if train_indices is not None else getattr(self.parser, "train_indices", None)
            if cfg_train_indices is not None and len(cfg_train_indices) > 0:
                try:
                    idx_arr = np.array(cfg_train_indices, dtype=int)
                    if np.any(idx_arr < 0) or np.any(idx_arr >= len(all_indices)):
                         raise ValueError(f"Training indices out of range.")
                    self.indices = idx_arr
                except Exception as e:
                    print(f"Error parse train indices: {e}. Fallback")
                    self.indices = all_indices
            else:
                # Default training set excludes test frames
                if self.parser.test_every > 1:
                    self.indices = all_indices[all_indices % self.parser.test_every != 0]
                else:
                    self.indices = all_indices
        elif split == "all":
            self.indices = all_indices
        else:
            # Test/Val split
            if self.parser.test_every > 1:
                self.indices = all_indices[all_indices % self.parser.test_every == 0]
            else:
                self.indices = np.array([], dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        
        # 1. Load Image (Use cv2 for speed)
        image_path = self.parser.image_paths[index]
        image = cv2.imread(image_path)
        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Load Depth (if requested)
        depth = None
        if self.parser.depth_paths is not None:
             # Safety check: ensure paths are available
             if index < len(self.parser.depth_paths):
                depth_path = self.parser.depth_paths[index]
                if depth_path:
                    depth = np.load(depth_path) # Assumed (H, W) or (H, W, 1) float32
                    if depth.ndim == 2:
                        depth = depth[..., None] # Ensure (H, W, 1)

        # 3. Get Camera Params
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K/new K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]

        # 4. Undistort (if params exist)
        if len(params) > 0:
            mapx = self.parser.mapx_dict[camera_id]
            mapy = self.parser.mapy_dict[camera_id]
            roi_x, roi_y, roi_w, roi_h = self.parser.roi_undist_dict[camera_id]

            # Undistort RGB
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            image = image[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            
            # Undistort Depth
            if depth is not None:
                # Use NEAREST to avoid smoothing across depth discontinuities
                depth = cv2.remap(depth, mapx, mapy, cv2.INTER_NEAREST)
                if depth.ndim == 2: depth = depth[..., None]
                depth = depth[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

        # 5. Patch Cropping
        if self.patch_size is not None:
            h, w = image.shape[:2]
            # Ensure patch size isn't larger than image
            ps = min(self.patch_size, h, w)
            
            x = np.random.randint(0, max(w - ps, 1))
            y = np.random.randint(0, max(h - ps, 1))
            
            image = image[y : y + ps, x : x + ps]
            K[0, 2] -= x
            K[1, 2] -= y
            
            if depth is not None:
                depth = depth[y : y + ps, x : x + ps]

        # 6. Prepare Output Dict
        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(), # (H, W, 3), 0-255
            "image_id": item,
            "colmap_image_id": index,
            "depth": torch.from_numpy(depth).float() if depth is not None else None,
        }

        return data


if __name__ == "__main__":
    import argparse
    import tqdm
    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = ColmapParser(data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8)
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    # Visualization check
    writer = imageio.get_writer("results/dataset_viz.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Visualizing"):
        image = data["image"].numpy().astype(np.uint8)
        
        # Draw depth heatmap if available
        if "depths" in data:
            depth = data["depths"].numpy().squeeze()
            if depth.max() > 0:
                # Normalize depth for vis
                d_valid = depth[depth > 0]
                if len(d_valid) > 0:
                    d_min, d_max = np.percentile(d_valid, [1, 99])
                    depth_vis = np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0, 1)
                    depth_vis = (depth_vis * 255).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    
                    # Stack Image and Depth side-by-side
                    if image.shape[0] == depth_vis.shape[0]:
                        image = np.hstack([image, depth_vis])
        
        writer.append_data(image)
    writer.close()
