"""
Rendering module for 3D Gaussian Splatting.

This module implements rendering of trained 3D Gaussian Splatting models.
"""

import os
import sys
import pickle
import numpy as np
import torch
import torchvision
from pathlib import Path
from tqdm import tqdm

from .gaussian_model import GaussianModel
from .camera import Camera
from .train import render, create_cameras_from_results, create_scene_from_reconstruction


def loadCameras(poses, viewpoint_stack):
    """
    Load optimized poses into camera stack.

    Args:
        poses: Optimized poses (N, 3, 4) or (N, 4, 4)
        viewpoint_stack: List of Camera objects

    Returns:
        Updated list of Camera objects
    """
    from .utils import getWorld2View2

    # Ensure poses are float32
    if poses.dtype != np.float32:
        poses = poses.astype(np.float32)

    # Handle different pose formats
    if poses.shape[1] == 3:  # (N, 3, 4) - R|T format
        for idx, cam in enumerate(viewpoint_stack):
            if idx < len(poses):
                R = poses[idx][:3, :3]
                T = poses[idx][:3, 3]
                cam.R = R
                cam.T = T
                cam.world_view_transform = torch.from_numpy(getWorld2View2(R, T)).transpose(0, 1).cuda().float()
                cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
                cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    elif poses.shape[1] == 4:  # (N, 4, 4) - full matrix
        for idx, cam in enumerate(viewpoint_stack):
            if idx < len(poses):
                pose = poses[idx]
                R = pose[:3, :3]
                T = pose[:3, 3]
                cam.R = R
                cam.T = T
                cam.world_view_transform = torch.from_numpy(getWorld2View2(R, T)).transpose(0, 1).cuda().float()
                cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
                cam.camera_center = cam.world_view_transform.inverse()[3, :3]

    return viewpoint_stack


class PipelineParams:
    """Pipeline parameters."""
    def __init__(self, convert_SHs_python=False, compute_cov3D_python=False, debug=False):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    """
    Render a set of views.

    Args:
        model_path: Path to model directory
        name: Name of the set (e.g., "train", "test")
        iteration: Iteration number to load
        views: List of Camera objects to render
        gaussians: GaussianModel
        pipeline: Pipeline parameters
        background: Background color tensor
    """
    render_path = Path(model_path) / name / f"ours_{iteration}" / "renders"
    gts_path = Path(model_path) / name / f"ours_{iteration}" / "gt"

    render_path.mkdir(parents=True, exist_ok=True)
    gts_path.mkdir(parents=True, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        # Get camera pose
        camera_pose = gaussians.get_RT(view.uid)

        # Render
        rendering = render(
            view, gaussians, pipeline, background, camera_pose=camera_pose
        )["render"]

        # Get ground truth
        gt = view.original_image[0:3, :, :]

        # Save images
        torchvision.utils.save_image(
            rendering, render_path / "{0:05d}.png".format(idx)
        )
        torchvision.utils.save_image(
            gt, gts_path / "{0:05d}.png".format(idx)
        )


def render_set_optimize(model_path, name, iteration, views, gaussians, pipeline, background, num_iter=500):
    """
    Render a set of views with pose optimization.

    Args:
        model_path: Path to model directory
        name: Name of the set (e.g., "train", "test")
        iteration: Iteration number to load
        views: List of Camera objects to render
        gaussians: GaussianModel
        pipeline: Pipeline parameters
        background: Background color tensor
        num_iter: Number of pose optimization iterations
    """
    from .utils import get_tensor_from_camera

    render_path = Path(model_path) / name / f"ours_{iteration}" / "renders"
    gts_path = Path(model_path) / name / f"ours_{iteration}" / "gt"

    render_path.mkdir(parents=True, exist_ok=True)
    gts_path.mkdir(parents=True, exist_ok=True)

    # Disable gradients for Gaussians
    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name} with pose optimization")):
        # Get initial camera pose
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))

        # Setup pose optimizer
        camera_tensor_T = camera_pose[4:7].clone().requires_grad_()
        camera_tensor_q = camera_pose[:4].clone().requires_grad_()

        pose_optimizer = torch.optim.Adam([
            {"params": [camera_tensor_T], "lr": 0.003},
            {"params": [camera_tensor_q], "lr": 0.001}
        ], betas=(0.9, 0.999), weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            pose_optimizer, T_max=num_iter, eta_min=0.0001
        )

        gt = view.original_image[0:3, :, :]
        initial_loss = None
        current_min_loss = float(1e20)
        candidate_q = camera_tensor_q.clone().detach()
        candidate_T = camera_tensor_T.clone().detach()

        # Pose optimization loop
        for iteration in range(num_iter):
            # Render with current pose
            current_pose = torch.cat([camera_tensor_q, camera_tensor_T])
            rendering = render(view, gaussians, pipeline, background, camera_pose=current_pose)["render"]

            # Mask loss
            mask = (rendering > 0.0).float()
            loss = torch.abs(rendering - gt).mean()

            loss.backward()

            with torch.no_grad():
                pose_optimizer.step()
                pose_optimizer.zero_grad(set_to_none=True)

                if iteration == 0:
                    initial_loss = loss.item()

                if loss.item() < current_min_loss:
                    current_min_loss = loss.item()
                    candidate_q = camera_tensor_q.clone().detach()
                    candidate_T = camera_tensor_T.clone().detach()

            scheduler.step()

        # Use best pose
        camera_tensor_q = candidate_q
        camera_tensor_T = candidate_T
        optimal_pose = torch.cat([camera_tensor_q, camera_tensor_T])

        # Render with optimized pose
        rendering_opt = render(view, gaussians, pipeline, background, camera_pose=optimal_pose)["render"]

        # Save images
        torchvision.utils.save_image(
            rendering_opt, render_path / view.image_name + ".png"
        )
        torchvision.utils.save_image(
            gt, gts_path / view.image_name + ".png"
        )


def render_gaussians(
    model_path,
    output_dir=None,
    iteration=-1,
    skip_train=False,
    skip_test=False,
    optimize_test_poses=True,
    optim_test_pose_iter=500,
    reconstruction_dir=None,
    device='cuda'
):
    """
    Render views from trained Gaussian model.

    Args:
        model_path: Path to trained model directory
        output_dir: Optional output directory (defaults to model_path)
        iteration: Iteration to load (-1 for latest)
        skip_train: Skip rendering training views
        skip_test: Skip rendering test views
        optimize_test_poses: Whether to optimize test poses
        optim_test_pose_iter: Number of pose optimization iterations
        reconstruction_dir: Optional path to reconstruction directory (if not provided, will try to find it)
        device: Device to use
    """
    model_path = Path(model_path)

    if output_dir is None:
        output_dir = model_path
    else:
        output_dir = Path(output_dir)

    # Find iteration to load
    if iteration == -1:
        # Find the latest iteration
        point_cloud_dir = model_path / "point_cloud"
        if point_cloud_dir.exists():
            iterations = [int(d.name.split("_")[1]) for d in point_cloud_dir.iterdir() if d.is_dir() and d.name.startswith("iteration_")]
            if iterations:
                iteration = max(iterations)
            else:
                raise ValueError(f"No checkpoints found in {point_cloud_dir}")
        else:
            raise ValueError(f"No point_cloud directory found in {model_path}")

    print(f"Loading model from iteration {iteration}")

    # Load Gaussian model
    gaussians = GaussianModel(sh_degree=3)
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

    if not ply_path.exists():
        raise ValueError(f"PLY file not found: {ply_path}")

    print(f"Loading Gaussians from {ply_path}")
    gaussians.load_ply(str(ply_path))

    # Load optimized poses if available
    pose_path = model_path / "pose" / f"ours_{iteration}" / "pose_optimized.npy"

    # Find reconstruction directory for scene info
    if reconstruction_dir is None:
        # Try to find it in parent directories
        current_path = model_path
        while current_path.parent != current_path:
            # Check if this looks like a reconstruction output directory
            has_backend_dirs = any((current_path / d).is_dir() for d in ['vggtx', 'mast3r', 'hunyuanworld', 'da3'])
            has_sparse = (current_path / 'sparse' / '0').exists()

            if has_backend_dirs or has_sparse:
                reconstruction_dir = str(current_path)
                break
            current_path = current_path.parent

    if reconstruction_dir is None:
        raise ValueError(
            f"Could not find reconstruction directory for {model_path}. "
            "Please provide the reconstruction_dir parameter."
        )

    print(f"Using reconstruction directory: {reconstruction_dir}")

    # Create scene from reconstruction
    scene_info = create_scene_from_reconstruction(str(reconstruction_dir), device=device)
    train_cameras = scene_info['train_cameras']

    # Load optimized poses
    if pose_path.exists():
        print(f"Loading optimized poses from {pose_path}")
        optimized_poses = np.load(pose_path)
        train_cameras = loadCameras(optimized_poses, train_cameras)
    else:
        print("No optimized poses found, using initial poses")

    # Initialize Gaussians with camera poses (required for rendering)
    gaussians.init_RT_seq(train_cameras)

    # Setup
    bg_color = [1, 1, 1] if True else [0, 0, 0]  # Use white background
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    pipeline = PipelineParams()

    # Render training views
    if not skip_train:
        print("Rendering training views...")
        render_set(
            output_dir,
            "train",
            iteration,
            train_cameras,
            gaussians,
            pipeline,
            background,
        )
        print(f"Training renders saved to {output_dir / 'train'}")

    # Render test views
    if not skip_test:
        print("Rendering test views...")
        if optimize_test_poses:
            render_set_optimize(
                output_dir,
                "test",
                iteration,
                train_cameras,  # Use same cameras for now
                gaussians,
                pipeline,
                background,
                num_iter=optim_test_pose_iter
            )
        else:
            render_set(
                output_dir,
                "test",
                iteration,
                train_cameras,  # Use same cameras for now
                gaussians,
                pipeline,
                background,
            )
        print(f"Test renders saved to {output_dir / 'test'}")

    print("Rendering completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render trained 3D Gaussian Splatting model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults to model_path)")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load (-1 for latest)")
    parser.add_argument("--skip_train", action="store_true", help="Skip rendering training views")
    parser.add_argument("--skip_test", action="store_true", help="Skip rendering test views")
    parser.add_argument("--no_pose_optimization", action="store_true", help="Disable test pose optimization")
    parser.add_argument("--optim_test_pose_iter", type=int, default=500, help="Number of pose optimization iterations")
    parser.add_argument("--reconstruction_dir", type=str, default=None, help="Path to reconstruction directory (optional)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    render_gaussians(
        model_path=args.model_path,
        output_dir=args.output_dir,
        iteration=args.iteration,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        optimize_test_poses=not args.no_pose_optimization,
        optim_test_pose_iter=args.optim_test_pose_iter,
        reconstruction_dir=args.reconstruction_dir,
        device=args.device
    )
