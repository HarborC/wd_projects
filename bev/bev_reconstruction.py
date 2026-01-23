import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from depth_anything_3.api import DepthAnything3
import cv2
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
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

def get_ground_masks(img_paths):
    print("Detecting ground regions using SegFormer...")
    try:
        # Correct model name: nvidia/segformer-b0-finetuned-ade-512-512
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        # ADE20K ground-related indices
        # 3: floor, 11: sidewalk, 12: earth, 13: path, 29: field, 46: sand, 52: grass, 94: road
        ground_indices = [3, 11, 12, 13, 29, 46, 52, 94]
        
        ground_masks = []
        
        for p in img_paths:
            image = Image.open(p).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
                
            # Upsample logits to image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1], # (height, width)
                mode="bilinear",
                align_corners=False,
            )
            
            pred_seg = upsampled_logits.argmax(dim=1)[0] # (H, W)
            pred_seg = pred_seg.cpu().numpy()
            
            # Create binary mask
            mask = np.isin(pred_seg, ground_indices)
            ground_masks.append(mask)
            
        print(f"Generated ground masks for {len(img_paths)} images.")
        return ground_masks
    except Exception as e:
        print(f"Warning: Failed to load SegFormer or detect ground ({e}). Falling back to geometric alignment.")
        return None

def load_data(undistorted_dir: Path, intrinsics_path: Path):
    # Load intrinsics
    with open(intrinsics_path, 'r') as f:
        intrinsics_data = json.load(f)

    # Find undistorted images
    img_paths = sorted([p for p in undistorted_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not img_paths:
        print(f"No images found in {undistorted_dir}")
        return [], []

    # Map original paths to undistorted paths (assuming stem matches + _undistorted)
    stem_to_intrinsics = {}
    for path_str, data in intrinsics_data.items():
        p = Path(path_str)
        stem_to_intrinsics[p.stem] = data

    # Prepare inputs for DA3
    valid_img_paths = []
    valid_intrinsics = []
    
    for p in img_paths:
        original_stem = p.stem.replace('_undistorted', '')
        
        if original_stem in stem_to_intrinsics:
            data = stem_to_intrinsics[original_stem]
            if 'camera' in data and data['camera'] is not None and 'K' in data['camera']:
                K = np.array(data['camera']['K'][0]) 
                valid_img_paths.append(str(p))
                valid_intrinsics.append(K)
            else:
                print(f"Warning: No camera intrinsics for {original_stem}")
        else:
            print(f"Warning: Could not find intrinsics for {p.name} (derived stem: {original_stem})")
            
    return valid_img_paths, valid_intrinsics

def run_inference(valid_img_paths, valid_intrinsics, output_dir):
    # Initialize Depth Anything 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing DepthAnything3 (da3nested-giant-large) on {device}...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.eval()
    model = model.to(device=device)

    # Run inference
    intrinsics_np = np.stack(valid_intrinsics)
    
    print("Running inference...")
    prediction = model.inference(
        image=valid_img_paths,
        # intrinsics=intrinsics_np,
        export_dir=str(output_dir),
        export_format="npz-glb-gs_ply-gs_video", 
        # align_to_input_ext_scale=True, 
        show_cameras=True,
        infer_gs=True
    )
    print(f"Inference complete. Results saved to {output_dir}")
    return prediction

def calculate_conf_thresh(conf):
    conf_thresh = 0.0
    if conf is not None:
        # print("Calculating confidence threshold...")
        # # Default values from glb.py
        # base_conf_thresh = 1.05
        # conf_thresh_percentile = 20.0
        # ensure_thresh_percentile = 90.0
        
        # lower = np.percentile(conf, conf_thresh_percentile)
        # upper = np.percentile(conf, ensure_thresh_percentile)
        # conf_thresh = min(max(base_conf_thresh, lower), upper)
        conf_thresh = np.percentile(conf, 1.0)
        print(f"Confidence threshold: {conf_thresh:.4f}")
        
    return conf_thresh

def generate_point_cloud(depths, intrinsics, extrinsics_w2c, images_u8, conf, conf_thresh, ground_masks=None):
    all_points_world = []
    all_colors = []
    ground_points_world = [] # Points specifically identified as ground
    
    print("Back-projecting points to World Coordinates...")
    for i, (depth, img_color) in enumerate(zip(depths, images_u8)):
        K = intrinsics[i]
        w2c = extrinsics_w2c[i]
        
        # Invert w2c to get c2w
        w2c_homo = _as_homogeneous44(w2c)
        c2w = np.linalg.inv(w2c_homo)
        
        H, W = depth.shape
        
        if img_color.shape[:2] != (H, W):
            img_color = cv2.resize(img_color, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Resize ground mask if needed
        g_mask = None
        if ground_masks is not None:
            g_mask = ground_masks[i]
            if g_mask.shape != (H, W):
                g_mask = cv2.resize(g_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Backproject to 3D (Camera Coordinates)
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        # Standard OpenCV pinhole backprojection
        X_cam = (x - K[0, 2]) * depth / K[0, 0]
        Y_cam = (y - K[1, 2]) * depth / K[1, 1]
        Z_cam = depth
        
        # Filter valid points (simple depth threshold)
        mask = (depth > 0) & (depth < 200) # Increased max depth
        
        if conf is not None:
            mask = mask & (conf[i] >= conf_thresh)

        # Extract all valid points
        X_cam_valid = X_cam[mask]
        Y_cam_valid = Y_cam[mask]
        Z_cam_valid = Z_cam[mask]
        colors_valid = img_color[mask]
        
        if len(X_cam_valid) > 0:
            P_cam = np.stack([X_cam_valid, Y_cam_valid, Z_cam_valid, np.ones_like(X_cam_valid)], axis=1)
            P_world = (c2w @ P_cam.T).T
            all_points_world.append(P_world[:, :3])
            all_colors.append(colors_valid)
            
        # Extract ground points if mask is available
        if g_mask is not None:
            mask_ground = mask & g_mask
            X_cam_g = X_cam[mask_ground]
            Y_cam_g = Y_cam[mask_ground]
            Z_cam_g = Z_cam[mask_ground]
            
            if len(X_cam_g) > 0:
                P_cam_g = np.stack([X_cam_g, Y_cam_g, Z_cam_g, np.ones_like(X_cam_g)], axis=1)
                P_world_g = (c2w @ P_cam_g.T).T
                ground_points_world.append(P_world_g[:, :3])

    if not all_points_world:
        return None, None, None

    all_points_world = np.concatenate(all_points_world, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    
    if ground_points_world:
        ground_points_world = np.concatenate(ground_points_world, axis=0)
    else:
        ground_points_world = None
        
    return all_points_world, all_colors, ground_points_world

def align_gravity(all_points_world, all_colors, ground_points_world, output_dir):
    print("Estimating ground plane for gravity alignment...")
    from sklearn.linear_model import RANSACRegressor
    
    # Use ground points if available and sufficient, otherwise fallback to all points
    if ground_points_world is not None and len(ground_points_world) > 100:
        print(f"Using {len(ground_points_world)} detected ground points for alignment.")
        points_for_alignment = ground_points_world
    else:
        print("Warning: No sufficient ground points detected. Falling back to all points.")
        points_for_alignment = all_points_world
    
    # We assume the ground is roughly a plane y = ax + bz + c
    X_xz = points_for_alignment[:, [0, 2]] # Inputs: x, z
    y_y = points_for_alignment[:, 1]       # Target: y
    
    ransac = RANSACRegressor(residual_threshold=0.1, max_trials=1000, random_state=42)
    ransac.fit(X_xz, y_y)
    
    # Plane equation: y = ax + bz + c  =>  ax - y + bz + c = 0
    # Normal vector: [a, -1, b]
    a, b = ransac.estimator_.coef_
    normal = np.array([a, -1, b])
    normal = normal / np.linalg.norm(normal)
    
    print(f"Estimated ground normal: {normal}")
    
    # Align ground normal to Y-axis
    # We want the ground normal (which points "Up") to align with the Y-axis.
    # However, we need to ensure the coordinate system is consistent for DSM generation.
    # Let's align "Up" to [0, -1, 0] (Negative Y).
    # This means -Y is "Up", and +Y is "Down".
    # Then, the "Highest" points (Roofs) will have the smallest Y values (most negative).
    # The "Lowest" points (Ground) will have the largest Y values.
    target_normal = np.array([0, -1, 0])
    
    v = np.cross(normal, target_normal)
    c = np.dot(normal, target_normal)
    s = np.linalg.norm(v)
    
    if s > 1e-6:
        k = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        # Rodrigues' rotation formula
        R_align = np.eye(3) + k + k @ k * ((1 - c) / (s**2))
    else:
        R_align = np.eye(3)
        if c < 0: # Anti-parallel
            R_align = -np.eye(3)
            
    print("Applying gravity alignment rotation...")
    all_points_world = (R_align @ all_points_world.T).T
    
    # Re-save pointcloud after alignment
    ply_path = output_dir / "reconstructed_pointcloud_world_aligned.ply"
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_points_world)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(all_points_world, all_colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
    print(f"Aligned point cloud saved to {ply_path}")
    
    return all_points_world, R_align

def compute_bev_bounds(all_points_world, res=0.01, padding=5.0):
    x_vals = all_points_world[:, 0]
    z_vals = all_points_world[:, 2]
    
    # Use percentiles to ignore outliers
    x_min, x_max = np.percentile(x_vals, [1, 99])
    z_min, z_max = np.percentile(z_vals, [1, 99])
    
    x_min -= padding
    x_max += padding
    z_min -= padding
    z_max += padding
    
    print(f"BEV Bounds: X[{x_min:.2f}, {x_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")
    
    bev_w = int((x_max - x_min) / res)
    bev_h = int((z_max - z_min) / res)
    
    print(f"BEV Image Size: {bev_w} x {bev_h}")
    return x_min, z_max, bev_w, bev_h

def generate_dsm(all_points_world, x_min, z_max, res, bev_w, bev_h):
    print("Generating DSM (Digital Surface Model)...")
    
    # Estimate ground level (Since Up is -Y, Ground is at max Y)
    ground_level = np.percentile(all_points_world[:, 1], 95)
    print(f"Estimated ground level for hole filling: {ground_level:.4f}")
    
    # Initialize DSM with very high value (since we want minimum Y for top surface)
    dsm_grid = np.full((bev_h, bev_w), 1000.0, dtype=np.float32)
    
    # Project aligned points to grid
    u = ((all_points_world[:, 0] - x_min) / res).astype(np.int64)
    v = ((z_max - all_points_world[:, 2]) / res).astype(np.int64)
    
    valid_idx = (u >= 0) & (u < bev_w) & (v >= 0) & (v < bev_h)
    u = u[valid_idx]
    v = v[valid_idx]
    y = all_points_world[valid_idx, 1]
    
    # Use minimum.at to get the top surface (min Y, since Up is -Y)
    flat_idx = v * bev_w + u
    dsm_flat = dsm_grid.flatten()
    np.minimum.at(dsm_flat, flat_idx, y)
    dsm_grid = dsm_flat.reshape(bev_h, bev_w)
    
    # Raw data mask (where we actually have points)
    has_data_mask = dsm_grid < 999.0
    
    print("Computing room boundary mask to limit reconstruction area...")
    # We use Morphological Closing to fill holes inside the room but respect the boundary
    # Kernel size: 31 pixels (~30cm at 1cm/pixel) to bridge gaps
    kernel_size = 31
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Morphological Close: Dilation followed by Erosion
    room_mask = cv2.morphologyEx(has_data_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    room_mask = room_mask.astype(bool)
    
    # Fill holes INSIDE the room mask with ground_level
    holes_mask = room_mask & (~has_data_mask)
    dsm_grid[holes_mask] = ground_level
    
    # The valid_dsm_mask for projection is the room_mask (where we have geometry or filled ground)
    valid_dsm_mask = room_mask
    
    return dsm_grid, valid_dsm_mask, has_data_mask, ground_level

def generate_bev_image(dsm_grid_filled, valid_dsm_mask, has_data_mask, x_min, z_max, res, bev_w, bev_h, 
                      R_align, intrinsics, extrinsics_w2c, images_u8, depths):
    print("Inverse projecting for BEV color (Inverse Mapping)...")
    
    # Create world coordinate grid for every pixel in BEV
    grid_v, grid_u = np.meshgrid(np.arange(bev_h), np.arange(bev_w), indexing='ij')
    
    grid_x = x_min + (grid_u + 0.5) * res
    grid_z = z_max - (grid_v + 0.5) * res
    grid_y = dsm_grid_filled
    
    # Only process valid DSM pixels to save time
    valid_v, valid_u = np.where(valid_dsm_mask)
    pts_x = grid_x[valid_v, valid_u]
    pts_y = grid_y[valid_v, valid_u]
    pts_z = grid_z[valid_v, valid_u]
    
    pts_aligned = np.stack([pts_x, pts_y, pts_z], axis=1) # (N_pts, 3)
    
    # Transform back to original world frame
    print("Transforming DSM points back to original world frame...")
    pts_orig = (R_align.T @ pts_aligned.T).T
    
    # Accumulators for color
    accum_color = np.zeros((len(pts_orig), 3), dtype=np.float32)
    accum_weight = np.zeros((len(pts_orig), 1), dtype=np.float32)
    
    # Diagnostic counters
    count_in_fov = np.zeros(len(pts_orig), dtype=np.int32)
    count_visible = np.zeros(len(pts_orig), dtype=np.int32)
    
    print(f"Projecting {len(pts_orig)} points to {len(images_u8)} cameras...")
    
    for i in range(len(images_u8)):
        K = intrinsics[i]
        w2c = extrinsics_w2c[i]
        img_color = images_u8[i]
        depth_map = depths[i]
        H_img, W_img = depth_map.shape
        
        w2c_homo = _as_homogeneous44(w2c)
        
        pts_orig_homo = np.concatenate([pts_orig, np.ones((len(pts_orig), 1))], axis=1)
        pts_cam = (w2c_homo @ pts_orig_homo.T).T # (N, 4)
        
        z_cam = pts_cam[:, 2]
        valid_z = z_cam > 0.1
        
        if not np.any(valid_z):
            continue
            
        x_cam = pts_cam[:, 0]
        y_cam = pts_cam[:, 1]
        
        u_proj = (x_cam * K[0, 0] / z_cam) + K[0, 2]
        v_proj = (y_cam * K[1, 1] / z_cam) + K[1, 2]
        
        in_bounds = (u_proj >= 0) & (u_proj < W_img - 1) & \
                    (v_proj >= 0) & (v_proj < H_img - 1) & \
                    valid_z
                    
        if not np.any(in_bounds):
            continue
            
        idx_valid = np.where(in_bounds)[0]
        count_in_fov[idx_valid] += 1
        
        u_valid = u_proj[idx_valid]
        v_valid = v_proj[idx_valid]
        z_valid = z_cam[idx_valid]
        
        u_int = np.round(u_valid).astype(int)
        v_int = np.round(v_valid).astype(int)
        
        d_measured = depth_map[v_int, u_int]
        
        # Occlusion check
        is_visible = np.abs(z_valid - d_measured) < 0.2
        
        if not np.any(is_visible):
            continue
            
        idx_visible = idx_valid[is_visible]
        count_visible[idx_visible] += 1
        
        u_vis = u_int[is_visible]
        v_vis = v_int[is_visible]
        
        colors_sampled = img_color[v_vis, u_vis].astype(np.float32)
        
        accum_color[idx_visible] += colors_sampled
        accum_weight[idx_visible] += 1.0
        
    # Average
    final_colors = np.zeros_like(accum_color)
    mask_has_color = accum_weight[:, 0] > 0
    final_colors[mask_has_color] = accum_color[mask_has_color] / accum_weight[mask_has_color]
    final_colors = np.clip(final_colors, 0, 255).astype(np.uint8)
    
    # Fill BEV image
    global_bev_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    global_bev_img[valid_v, valid_u] = final_colors
    
    # Inpaint
    print("Inpainting final BEV image...")
    mask_holes = (np.sum(global_bev_img, axis=2) == 0).astype(np.uint8)
    mask_holes = mask_holes & valid_dsm_mask.astype(np.uint8)
    
    if np.sum(mask_holes) > 0:
        global_bev_img = cv2.inpaint(global_bev_img, mask_holes, 3, cv2.INPAINT_TELEA)
        
    # Generate Diagnostic Mask
    # Colors:
    # Valid Color: Green [0, 255, 0]
    # Occluded: Yellow [0, 255, 255] (In FOV but failed depth check)
    # Out of FOV: Blue [255, 0, 0] (Valid geometry but not in any camera FOV)
    # No Geometry: Gray [128, 128, 128] (Outside room mask or holes)
    # Filled Geometry (No Color): Red [0, 0, 255]
    
    final_diag = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    final_diag[:] = [128, 128, 128] # Gray: No Geometry
    
    # Map status back to grid
    grid_visible = np.zeros((bev_h, bev_w), dtype=bool)
    grid_in_fov = np.zeros((bev_h, bev_w), dtype=bool)
    
    grid_visible[valid_v, valid_u] = (count_visible > 0)
    grid_in_fov[valid_v, valid_u] = (count_in_fov > 0)
    
    # 1. Valid Color -> Green
    final_diag[grid_visible] = [0, 255, 0]
    
    # 2. Missing Color
    missing_mask = valid_dsm_mask & (~grid_visible)
    
    # Sub-cases for missing:
    # A. Filled Geometry (Synthesized) -> Red
    mask_filled_missing = missing_mask & (~has_data_mask)
    final_diag[mask_filled_missing] = [0, 0, 255] # Red
    
    # B. Raw Geometry
    mask_raw_missing = missing_mask & has_data_mask
    
    # B1. Occluded -> Yellow
    mask_occluded = mask_raw_missing & grid_in_fov
    final_diag[mask_occluded] = [0, 255, 255] # Yellow
    
    # B2. Out of FOV -> Blue
    mask_fov_missing = mask_raw_missing & (~grid_in_fov)
    final_diag[mask_fov_missing] = [255, 0, 0] # Blue
        
    return global_bev_img, final_diag

class DA3PostProcessor:
    def __init__(self):
        pass

    def run(self, undistorted_dir: Path, intrinsics_path: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load Data
        valid_img_paths, valid_intrinsics = load_data(undistorted_dir, intrinsics_path)
        if not valid_img_paths:
            return
        print(f"Found {len(valid_img_paths)} images with intrinsics.")

        # 2. Run Inference
        prediction = run_inference(valid_img_paths, valid_intrinsics, output_dir)
        
        # 3. Generate BEV
        print("Generating Merged BEV image (World Coordinate System)...")
        bev_dir = output_dir / "bev_images"
        bev_dir.mkdir(exist_ok=True)
        
        depths = prediction.depth
        intrinsics = prediction.intrinsics
        extrinsics_w2c = prediction.extrinsics
        conf = prediction.conf
        images_u8 = prediction.processed_images

        # 4. Calculate Confidence Threshold
        conf_thresh = calculate_conf_thresh(conf)
        
        # 4.5 Detect Ground Masks
        ground_masks = get_ground_masks(valid_img_paths)
        
        # 5. Generate Point Cloud
        all_points_world, all_colors, ground_points_world = generate_point_cloud(depths, intrinsics, extrinsics_w2c, images_u8, conf, conf_thresh, ground_masks)
        if all_points_world is None:
            print("No valid points found.")
            return

        # Save raw point cloud (before gravity alignment) as PLY
        try:
            ply_raw_path = output_dir / "reconstructed_pointcloud_world_raw.ply"
            with open(ply_raw_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(all_points_world)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                for p, c in zip(all_points_world, all_colors):
                    f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            print(f"Raw point cloud saved to {ply_raw_path}")
        except Exception as e:
            print(f"Warning: Failed to save raw point cloud PLY ({e})")

        # 6. Align Gravity
        all_points_world, R_align = align_gravity(all_points_world, all_colors, ground_points_world, output_dir)
        
        # 7. Compute BEV Bounds
        res = 0.01 # 1cm per pixel
        x_min, z_max, bev_w, bev_h = compute_bev_bounds(all_points_world, res=res)
        
        # 8. Generate DSM
        dsm_grid_filled, valid_dsm_mask, has_data_mask, ground_level = generate_dsm(all_points_world, x_min, z_max, res, bev_w, bev_h)
        
        # 9. Generate BEV Image (Inverse Mapping)
        global_bev_img, diag_img = generate_bev_image(dsm_grid_filled, valid_dsm_mask, has_data_mask, x_min, z_max, res, bev_w, bev_h, 
                                        R_align, intrinsics, extrinsics_w2c, images_u8, depths)
        
        # Save merged BEV
        out_path = bev_dir / "merged_bev_world_color_corrected.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(global_bev_img, cv2.COLOR_RGB2BGR))
        print(f"Merged BEV image saved to {out_path}")
        
        # Save diagnostic mask
        diag_path = bev_dir / "bev_diagnostic_mask.png"
        cv2.imwrite(str(diag_path), diag_img) # Already BGR
        print(f"Diagnostic mask saved to {diag_path}")

        # 10. Save Depth Maps at Original Resolution
        print("Processing and saving depth maps at original resolution...")
        depths_dir = output_dir / "depths_original_res"
        depths_dir.mkdir(exist_ok=True)
        
        saved_depth_paths = []
        original_shapes = []
        intrinsics_estimated_scaled = []
        
        for i, (depth, img_path) in enumerate(zip(depths, valid_img_paths)):
            # Get original size
            with Image.open(img_path) as img:
                W_orig, H_orig = img.size
                
            original_shapes.append([H_orig, W_orig])
            
            # Resize depth if needed
            H_pred, W_pred = depth.shape
            if (H_pred, W_pred) != (H_orig, W_orig):
                depth_resized = cv2.resize(depth, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            else:
                depth_resized = depth
                
            # Save as .npy
            save_path = depths_dir / f"{Path(img_path).stem}.npy"
            np.save(save_path, depth_resized)
            saved_depth_paths.append(str(save_path))
            
            # Scale estimated intrinsics (from prediction) to original resolution
            K_est = intrinsics[i]
            sx = W_orig / W_pred
            sy = H_orig / H_pred
            
            K_scaled = K_est.copy()
            K_scaled[0, 0] *= sx
            K_scaled[1, 1] *= sy
            K_scaled[0, 2] *= sx
            K_scaled[1, 2] *= sy
            intrinsics_estimated_scaled.append(K_scaled)
            
        print(f"Saved {len(saved_depth_paths)} depth maps to {depths_dir}")

        # 11. Save Metadata for Trajectory Tracking
        print("Saving metadata for trajectory tracking...")
        
        # Compute aligned extrinsics (Aligned World -> Camera)
        # P_cam = T_w2c * P_world
        # P_aligned = R_align * P_world  =>  P_world = R_align.T * P_aligned
        # P_cam = T_w2c * (R_align.T * P_aligned)
        # T_aligned = T_w2c * [R_align.T, 0; 0, 1]
        
        R_align_inv = R_align.T
        T_align_inv = np.eye(4)
        T_align_inv[:3, :3] = R_align_inv
        
        extrinsics_aligned = []
        for w2c in extrinsics_w2c:
            w2c_homo = _as_homogeneous44(w2c)
            w2c_aligned = w2c_homo @ T_align_inv
            extrinsics_aligned.append(w2c_aligned)
        extrinsics_aligned = np.stack(extrinsics_aligned)
        
        metadata_path = output_dir / "scene_metadata.npz"
        
        # Use estimated intrinsics scaled to original resolution
        intrinsics_final = np.stack(intrinsics_estimated_scaled)
        
        np.savez(metadata_path,
                intrinsics=intrinsics_final,            # (N, 3, 3) - Estimated & Scaled to Original Res
                extrinsics_aligned=extrinsics_aligned,  # (N, 4, 4) - Aligned World to Camera
                bev_params=np.array([x_min, z_max, res, ground_level], dtype=np.float32), # [x_min, z_max, res, ground_level]
                bev_shape=np.array([bev_w, bev_h], dtype=np.int32),
                img_paths=np.array(valid_img_paths),
                depth_paths=np.array(saved_depth_paths),
                img_shapes=np.array(original_shapes)
                )
        print(f"Scene metadata saved to {metadata_path}")


def main():
    # Paths
    undistorted_dir = Path('/home/disk2/jiagangchen/LJ/process_data/undistorted_first_frames/images')
    intrinsics_path = Path('/home/disk2/jiagangchen/LJ/process_data/intrinsics_first_frames_geocalib.json')
    output_dir = Path('/home/disk2/jiagangchen/LJ/process_data/da3_metric_reconstruction')
    
    processor = DA3PostProcessor()
    processor.run(undistorted_dir, intrinsics_path, output_dir)

if __name__ == "__main__":
    main()
