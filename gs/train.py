"""
Training module for 3D Gaussian Splatting.

This module implements the training loop for 3D Gaussian Splatting from
reconstruction output.
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from random import randint
from time import time
from tqdm import tqdm

from .gaussian_model import GaussianModel
from .camera import Camera
from .utils import (
    getWorld2View2, focal2fov, get_tensor_from_camera,
    save_time, BasicPointCloud
)


def save_pose(path, quat_pose, train_cams):
    """
    Save optimized camera poses to file.

    Args:
        path: Path to save poses
        quat_pose: Quaternion poses (N, 7) - quaternion_wxyz + translation_xyz
        train_cams: List of training cameras
    """
    from .utils import get_camera_from_tensor

    # Get camera IDs and convert quaternion poses to camera matrices
    camera_ids = [cam.colmap_id for cam in train_cams]
    world_to_camera = [get_camera_from_tensor(quat) for quat in quat_pose]

    # Reorder poses according to colmap IDs
    colmap_poses = []
    for i in range(len(camera_ids)):
        idx = camera_ids.index(i + 1)  # Find position of camera i+1
        pose = world_to_camera[idx]
        colmap_poses.append(pose)

    # Convert to numpy array and save
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)


def load_and_prepare_confidence(confidence_path, device='cuda', scale=(0.1, 1.0)):
    """
    Load, normalize, invert, and scale confidence values to obtain learning rate modifiers.

    Args:
        confidence_path: Path to the .npy confidence file.
        device: Device to load the tensor onto.
        scale: Desired range for the learning rate modifiers.

    Returns:
        torch.Tensor: Learning rate modifiers.
    """
    # Load and normalize
    confidence_np = np.load(confidence_path)
    confidence_tensor = torch.from_numpy(confidence_np).float().to(device)
    normalized_confidence = torch.sigmoid(confidence_tensor)

    # Invert confidence and scale to desired range
    inverted_confidence = 1.0 - normalized_confidence
    min_val, max_val = inverted_confidence.min(), inverted_confidence.max()
    if max_val - min_val > 0:
        scaled_confidence = (inverted_confidence - min_val) / (max_val - min_val)
        scaled_confidence = scaled_confidence * (scale[1] - scale[0]) + scale[0]
    else:
        scaled_confidence = torch.ones_like(inverted_confidence) * scale[0]

    return scaled_confidence


def create_cameras_from_results(results, device='cuda'):
    """
    Create camera list from reconstruction results.

    Args:
        results: Results dictionary from reconstruction
        device: Device to use

    Returns:
        List of Camera objects
    """
    forward_results = results['forward_results']
    imgs = forward_results['imgs']  # (S, H, W, 3)
    poses = forward_results['poses']  # (S, 3, 4) - w2c extrinsics
    intrs = forward_results['intrs']  # (S, 3, 3)

    cameras = []

    for i in range(len(imgs)):
        # Extract extrinsics
        extrinsic = poses[i]  # (3, 4) - w2c
        R = extrinsic[:3, :3]
        T = extrinsic[:3, 3]

        # Extract intrinsics
        intrinsic = intrs[i]  # (3, 3)
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        # Image
        img = imgs[i]
        height, width = img.shape[:2]

        # Compute FoV
        FoVx = focal2fov(fx, width)
        FoVy = focal2fov(fy, height)

        # Create camera
        cam = Camera(
            uid=i,
            colmap_id=i + 1,
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            image=img,
            image_path="",
            image_name=results['image_names'][i] if 'image_names' in results else f"{i:05d}",
            width=width,
            height=height,
            device=device
        )
        cameras.append(cam)

    return cameras


def create_point_cloud_from_results(results, device='cuda'):
    """
    Create point cloud from reconstruction results.

    Args:
        results: Results dictionary from reconstruction
        device: Device to use

    Returns:
        BasicPointCloud
    """
    forward_results = results['forward_results']
    pts3d = forward_results['pts3d']  # (S, H, W, 3)
    confs = forward_results['pts3d_conf']  # (S, H, W)

    # Flatten and filter by confidence
    S, H, W, _ = pts3d.shape
    points = pts3d.reshape(-1, 3)  # (S*H*W, 3)
    confs_flat = confs.reshape(-1)  # (S*H*W)

    # Filter by confidence threshold
    conf_threshold = np.percentile(confs_flat, 50)
    mask = confs_flat > conf_threshold

    points_filtered = points[mask]

    # Create colors from confidence (for visualization)
    confs_filtered = confs_flat[mask]
    colors = np.zeros_like(points_filtered)
    colors[:, 0] = confs_filtered  # Red channel = confidence
    colors[:, 1] = confs_filtered  # Green channel = confidence
    colors[:, 2] = confs_filtered  # Blue channel = confidence

    # If images are available, use them for colors
    imgs = forward_results['imgs']  # (S, H, W, 3)
    imgs_flat = imgs.reshape(-1, 3)
    colors_filtered = imgs_flat[mask]

    return BasicPointCloud(points=points_filtered, colors=colors_filtered, normals=np.zeros_like(points_filtered))


def getNerfppNorm(cam_info):
    """Get NeRF++ normalization parameters."""
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def create_scene_from_reconstruction(reconstruction_dir, device='cuda'):
    """
    Create scene from reconstruction output.

    Args:
        reconstruction_dir: Path to reconstruction output directory
        device: Device to use

    Returns:
        Dictionary containing scene information
    """
    reconstruction_dir = Path(reconstruction_dir)

    # Find backend-specific directory (e.g., vggtx, mast3r, etc.)
    backend_dirs = [d for d in reconstruction_dir.iterdir() if d.is_dir() and d.name != 'sparse' and d.name != 'images' and d.name != 'depths']

    if not backend_dirs:
        # Try to find results.pkl in the root directory
        results_file = reconstruction_dir / "results.pkl"
        if not results_file.exists():
            raise ValueError(f"No results.pkl found in {reconstruction_dir}")
    else:
        # Use the first backend directory
        backend_dir = backend_dirs[0]
        results_file = backend_dir / "results.pkl"
        if not results_file.exists():
            raise ValueError(f"No results.pkl found in {backend_dir}")

    # Load results
    print(f"Loading results from {results_file}")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    # Create cameras
    cameras = create_cameras_from_results(results, device=device)
    print(f"Created {len(cameras)} cameras")

    # Create point cloud
    pcd = create_point_cloud_from_results(results, device=device)
    print(f"Created point cloud with {len(pcd.points)} points")

    # Get normalization
    nerf_normalization = getNerfppNorm(cameras)

    scene_info = {
        'point_cloud': pcd,
        'train_cameras': cameras,
        'test_cameras': [],  # No test cameras for now
        'nerf_normalization': nerf_normalization,
        'ply_path': None,  # No PLY file needed
        'train_poses': None,
        'test_poses': None,
        'results': results  # Store original results
    }

    return scene_info


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, camera_pose=None):
    """
    Render the scene using differentiable Gaussian rasterization.

    Args:
        viewpoint_camera: Camera to render from
        pc: GaussianModel
        pipe: Pipeline parameters
        bg_color: Background color tensor
        scaling_modifier: Scaling modifier for Gaussians
        override_color: Override color for Gaussians
        camera_pose: Camera pose (quaternion + translation)

    Returns:
        Dictionary containing rendered image and auxiliary data
    """
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from .utils import get_camera_from_tensor, quadmultiply
    import math

    # Create zero tensor for screen-space means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # Set camera pose as identity. Then, we will transform the Gaussians around camera_pose
    w2c = torch.eye(4).cuda()
    projmatrix = (
        w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_pos = w2c.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=w2c,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=camera_pos,
        prefiltered=False,
        debug=pipe.debug if hasattr(pipe, 'debug') else False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Transform Gaussians to camera frame
    rel_w2c = get_camera_from_tensor(camera_pose)
    gaussians_xyz = pc._xyz.clone()
    gaussians_rot = pc._rotation.clone()

    xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
    xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
    gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]
    gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)

    means3D = gaussians_xyz_trans
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = pc.get_scaling
    rotations = gaussians_rot_trans

    # Use spherical harmonics for colors
    shs = pc.get_features
    colors_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def l1_loss(pred, gt):
    """L1 loss."""
    return torch.abs(pred - gt).mean()


def ssim(pred, gt):
    """SSIM loss."""
    from torch.nn.functional import avg_pool2d

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pred = pred.unsqueeze(0)
    gt = gt.unsqueeze(0)

    mu1 = avg_pool2d(pred, 3, 1, 1)
    mu2 = avg_pool2d(gt, 3, 1, 1)

    sigma1 = avg_pool2d(pred * pred, 3, 1, 1) - mu1 * mu1
    sigma2 = avg_pool2d(gt * gt, 3, 1, 1) - mu2 * mu2
    sigma12 = avg_pool2d(pred * gt, 3, 1, 1) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))

    return ssim_map.mean()


class PipelineParams:
    """Pipeline parameters."""
    def __init__(self, convert_SHs_python=False, compute_cov3D_python=False, debug=False):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug


class OptimizationParams:
    """Optimization parameters."""
    def __init__(self, iterations=30000, position_lr_init=0.00016, position_lr_final=0.0000016,
                 position_lr_delay_mult=0.01, position_lr_max_steps=30000, feature_lr=0.0025,
                 opacity_lr=0.05, scaling_lr=0.005, rotation_lr=0.001, percent_dense=0.01,
                 lambda_dssim=0.2, densify_from_iter=500, densify_until_iter=15000,
                 densify_grad_threshold=0.0002, random_background=False, pp_optimizer=False):
        self.iterations = iterations
        self.position_lr_init = position_lr_init
        self.position_lr_final = position_lr_final
        self.position_lr_delay_mult = position_lr_delay_mult
        self.position_lr_max_steps = position_lr_max_steps
        self.feature_lr = feature_lr
        self.opacity_lr = opacity_lr
        self.scaling_lr = scaling_lr
        self.rotation_lr = rotation_lr
        self.percent_dense = percent_dense
        self.lambda_dssim = lambda_dssim
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densify_grad_threshold = densify_grad_threshold
        self.random_background = random_background
        self.pp_optimizer = pp_optimizer


def train_gaussians(
    reconstruction_dir,
    output_dir,
    iterations=30000,
    position_lr_init=0.00016,
    position_lr_final=0.0000016,
    feature_lr=0.0025,
    opacity_lr=0.05,
    scaling_lr=0.005,
    rotation_lr=0.001,
    lambda_dssim=0.2,
    use_confidence=True,
    device='cuda',
    save_interval=10000
):
    """
    Train 3D Gaussian Splatting model from reconstruction output.

    Args:
        reconstruction_dir: Path to reconstruction output directory
        output_dir: Path to save training output
        iterations: Number of training iterations
        position_lr_init: Initial position learning rate
        position_lr_final: Final position learning rate
        feature_lr: Feature learning rate
        opacity_lr: Opacity learning rate
        scaling_lr: Scaling learning rate
        rotation_lr: Rotation learning rate
        lambda_dssim: Weight for SSIM loss
        use_confidence: Whether to use confidence-weighted optimization
        device: Device to use
        save_interval: Interval for saving checkpoints
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating scene from reconstruction...")
    scene_info = create_scene_from_reconstruction(reconstruction_dir, device=device)

    print("Initializing Gaussian model...")
    gaussians = GaussianModel(sh_degree=3)

    # Create Gaussians from point cloud
    gaussians.create_from_pcd(
        scene_info['point_cloud'],
        scene_info['nerf_normalization']['radius']
    )

    # Initialize camera poses
    train_cameras = scene_info['train_cameras']
    gaussians.init_RT_seq({1.0: train_cameras})

    # Setup optimizer
    opt_params = OptimizationParams(
        iterations=iterations,
        position_lr_init=position_lr_init,
        position_lr_final=position_lr_final,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scaling_lr=scaling_lr,
        rotation_lr=rotation_lr,
        lambda_dssim=lambda_dssim,
        pp_optimizer=use_confidence
    )

    # Load confidence for per-point optimization if requested
    if use_confidence:
        confidence_path = Path(reconstruction_dir) / "confidence_dsp.npy"
        if confidence_path.exists():
            print(f"Loading confidence from {confidence_path}")
            confidence_lr = load_and_prepare_confidence(str(confidence_path), device=device, scale=(0.1, 1.0))
            gaussians.training_setup_pp(opt_params, confidence_lr=confidence_lr)
            print("Using per-point Adam optimizer with confidence weighting")
        else:
            print(f"Confidence file not found at {confidence_path}, using standard optimizer")
            gaussians.training_setup(opt_params)
    else:
        gaussians.training_setup(opt_params)

    # Background color
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    # Pipeline parameters
    pipe = PipelineParams()

    # Training loop
    print(f"Starting training for {iterations} iterations...")
    start_time = time()

    viewpoint_stack = train_cameras.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    ema_loss_for_log = 0.0
    progress_bar = tqdm(total=iterations, desc="Training")

    gaussians.optimizer.zero_grad(set_to_none=True)

    for iteration in range(1, iterations + 1):
        # Pick a random camera
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        pose = gaussians.get_RT(viewpoint_cam.uid)

        # Render
        bg = torch.rand((3), device=device) if opt_params.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt_params.lambda_dssim) * Ll1 + opt_params.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        # Update progress
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)

            # Update learning rate
            gaussians.update_learning_rate(iteration)

            # Optimizer step
            if iteration < iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Save checkpoints
            if iteration % save_interval == 0 or iteration == iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                point_cloud_path = output_dir / "point_cloud" / f"iteration_{iteration}"
                point_cloud_path.mkdir(parents=True, exist_ok=True)
                gaussians.save_ply(str(point_cloud_path / "point_cloud.ply"))

                # Save poses
                pose_dir = output_dir / "pose" / f"ours_{iteration}"
                pose_dir.mkdir(parents=True, exist_ok=True)
                save_pose(str(pose_dir / "pose_optimized.npy"), gaussians.P, train_cameras)
                print(f"Saved checkpoint to {point_cloud_path}")

    progress_bar.close()

    end_time = time()
    train_time = end_time - start_time
    print(f"Training completed in {train_time:.2f} seconds")

    # Save final timing
    save_time(str(output_dir), '[2] train_joint', train_time)

    return gaussians, scene_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting from reconstruction")
    parser.add_argument("--reconstruction_dir", type=str, required=True, help="Path to reconstruction output")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save training output")
    parser.add_argument("--iterations", type=int, default=30000, help="Number of training iterations")
    parser.add_argument("--save_interval", type=int, default=10000, help="Checkpoint save interval")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    train_gaussians(
        reconstruction_dir=args.reconstruction_dir,
        output_dir=args.output_dir,
        iterations=args.iterations,
        save_interval=args.save_interval,
        device=args.device
    )
