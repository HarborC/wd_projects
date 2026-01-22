#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""Training script for 3D Gaussian Splatting with pose optimization."""

import os
import sys
import uuid
from random import randint
from time import time
from argparse import ArgumentParser, Namespace
from typing import Optional, List, Tuple, Callable, Any

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.camera_utils import generate_interpolated_path
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2_torch
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.pose_utils import get_camera_from_tensor
from utils.sfm_utils import save_time

# Optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================

SH_DEGREE_UPDATE_INTERVAL = 1000
PROGRESS_UPDATE_INTERVAL = 10
EMA_LOSS_ALPHA = 0.4
EMA_LOSS_BETA = 0.6
DEFAULT_LR_SCALE_MIN = 1.0
DEFAULT_LR_SCALE_MAX = 100.0
DEFAULT_LR_SCALE = (DEFAULT_LR_SCALE_MIN, DEFAULT_LR_SCALE_MAX)
DEFAULT_CONFIDENCE_SCALE = (0.1, 1.0)


# =============================================================================
# Pose Utilities
# =============================================================================

def save_pose(path: str, quat_pose: torch.Tensor, train_cams: List[Camera], llffhold: int = 2) -> None:
    """Save camera poses in COLMAP order.

    Args:
        path: Output path for the .npy file.
        quat_pose: Quaternion poses tensor.
        train_cams: List of training cameras.
        llffhold: Hold parameter for LLFF (unused, kept for compatibility).
    """
    camera_ids = [cam.colmap_id for cam in train_cams]
    world_to_camera = [get_camera_from_tensor(quat) for quat in quat_pose]

    # Reorder poses according to colmap IDs
    colmap_poses = []
    for i in range(len(camera_ids)):
        idx = camera_ids.index(i + 1)
        pose = world_to_camera[idx]
        colmap_poses.append(pose)

    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)


def load_and_prepare_confidence(
    confidence_path: str,
    device: str = 'cuda',
    scale: Tuple[float, float] = DEFAULT_LR_SCALE
) -> torch.Tensor:
    """Load and prepare confidence values as learning rate modifiers.

    Args:
        confidence_path: Path to the .npy confidence file.
        device: Device to load the tensor onto.
        scale: Desired range for the learning rate modifiers.

    Returns:
        Learning rate modifiers tensor.
    """
    confidence_np = np.load(confidence_path)
    confidence_tensor = torch.from_numpy(confidence_np).float().to(device)
    normalized_confidence = torch.sigmoid(confidence_tensor)

    inverted_confidence = 1.0 - normalized_confidence
    min_scale, max_scale = scale
    lr_modifiers = inverted_confidence * (max_scale - min_scale) + min_scale

    return lr_modifiers


# =============================================================================
# Training Helpers
# =============================================================================

def prepare_output_and_logger(args: ModelParams) -> Optional['SummaryWriter']:
    """Create output folder and TensorBoard logger.

    Args:
        args: Model parameters containing model_path.

    Returns:
        TensorBoard writer if available, None otherwise.
    """
    if not args.model_path:
        unique_str = os.getenv('OAR_JOBID', str(uuid.uuid4()))
        args.model_path = os.path.join("./output/", unique_str[:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)

    print("Tensorboard not available: not logging progress")
    return None


def create_pose_save_dirs(scene: Scene, saving_iterations: List[int]) -> None:
    """Create directories for saving poses at specified iterations.

    Args:
        scene: Scene object containing model_path.
        saving_iterations: List of iteration numbers to save poses.
    """
    for save_iter in saving_iterations:
        os.makedirs(os.path.join(scene.model_path, f'pose/ours_{save_iter}'), exist_ok=True)


def save_initial_pose(scene: Scene, gaussians: GaussianModel, train_cams: List[Camera], saving_iterations: List[int]) -> None:
    """Save initial camera poses.

    Args:
        scene: Scene object.
        gaussians: Gaussian model with pose parameters.
        train_cams: Initial training cameras.
        saving_iterations: List of iteration numbers.
    """
    for save_iter in saving_iterations:
        path = os.path.join(scene.model_path, f'pose/ours_{save_iter}', 'pose_org.npy')
        save_pose(path, gaussians.P, train_cams)


def compute_loss(image: torch.Tensor, gt_image: torch.Tensor, lambda_dssim: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined L1 and SSIM loss.

    Args:
        image: Rendered image.
        gt_image: Ground truth image.
        lambda_dssim: Weight for SSIM loss.

    Returns:
        Tuple of (total_loss, l1_loss, ssim_value).
    """
    l1 = l1_loss(image, gt_image)

    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_value = ssim(image, gt_image)

    total_loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_value)
    return total_loss, l1, ssim_value


def update_progress_bar(progress_bar: tqdm, loss: float, ema_loss: float, iteration: int, total_iterations: int) -> float:
    """Update progress bar with exponential moving average loss.

    Args:
        progress_bar: tqdm progress bar.
        loss: Current loss value.
        ema_loss: Current exponential moving average loss.
        iteration: Current iteration number.
        total_iterations: Total number of iterations.

    Returns:
        Updated exponential moving average loss.
    """
    ema_loss = EMA_LOSS_ALPHA * loss + EMA_LOSS_BETA * ema_loss

    if iteration % PROGRESS_UPDATE_INTERVAL == 0:
        progress_bar.set_postfix({"Loss": f"{ema_loss:.7f}"})
        progress_bar.update(PROGRESS_UPDATE_INTERVAL)

    if iteration == total_iterations:
        progress_bar.close()

    return ema_loss


def run_validation(
    tb_writer: Optional['SummaryWriter'],
    iteration: int,
    scene: Scene,
    render_func: Callable,
    render_args: Tuple,
    testing_iterations: List[int]
) -> None:
    """Run validation on test and training sets.

    Args:
        tb_writer: TensorBoard writer.
        iteration: Current iteration.
        scene: Scene containing cameras and gaussians.
        render_func: Rendering function.
        render_args: Arguments to pass to render_func.
        testing_iterations: Iterations at which to run full validation.
    """
    if iteration not in testing_iterations and iteration % 5000 != 0:
        return

    torch.cuda.empty_cache()

    validation_configs = (
        {'name': 'test', 'cameras': scene.getTestCameras()},
        {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                       for idx in range(len(scene.getTrainCameras()))]}
    )

    for config in validation_configs:
        cameras = config['cameras']
        if not cameras or len(cameras) == 0:
            continue

        l1_test = 0.0
        psnr_test = 0.0

        for idx, viewpoint in enumerate(cameras):
            if config['name'] == "train":
                pose = scene.gaussians.get_RT(viewpoint.uid)
            else:
                pose = scene.gaussians.get_RT_test(viewpoint.uid)

            image = torch.clamp(
                render_func(viewpoint, scene.gaussians, *render_args, camera_pose=pose)["render"],
                0.0, 1.0
            )
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            if tb_writer and idx < 5:
                tb_writer.add_images(
                    f"{config['name']}_view_{viewpoint.image_name}/render",
                    image[None],
                    global_step=iteration
                )
                if iteration == testing_iterations[0]:
                    tb_writer.add_images(
                        f"{config['name']}_view_{viewpoint.image_name}/ground_truth",
                        gt_image[None],
                        global_step=iteration
                    )

            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()

        psnr_test /= len(cameras)
        l1_test /= len(cameras)
        print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.4f} PSNR {psnr_test:.4f}")

        if tb_writer:
            tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
            tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)

    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    torch.cuda.empty_cache()


def training_report(
    tb_writer: Optional['SummaryWriter'],
    iteration: int,
    l1_loss_val: torch.Tensor,
    total_loss: torch.Tensor,
    elapsed: float,
    testing_iterations: List[int],
    scene: Scene,
    render_func: Callable,
    render_args: Tuple
) -> None:
    """Log training metrics and run validation.

    Args:
        tb_writer: TensorBoard writer.
        iteration: Current iteration number.
        l1_loss_val: L1 loss value.
        total_loss: Total loss value.
        elapsed: Iteration elapsed time.
        testing_iterations: Iterations at which to run validation.
        scene: Scene object.
        render_func: Rendering function.
        render_args: Arguments for render_func.
    """
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', l1_loss_val.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', total_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    run_validation(tb_writer, iteration, scene, render_func, render_args, testing_iterations)


# =============================================================================
# Main Training Loop
# =============================================================================

def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations: List[int],
    saving_iterations: List[int],
    checkpoint_iterations: List[int],
    checkpoint: Optional[str],
    debug_from: int
) -> None:
    """Main training function.

    Args:
        dataset: Dataset parameters.
        opt: Optimization parameters.
        pipe: Pipeline parameters.
        testing_iterations: Iterations at which to run validation.
        saving_iterations: Iterations at which to save model.
        checkpoint_iterations: Iterations at which to save checkpoints.
        checkpoint: Path to checkpoint to resume from.
        debug_from: Iteration at which to enable debug mode.
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    # Load confidence for per-point optimizer
    sparse_folder = f"sparse_{dataset.n_views}"
    if not os.path.exists(os.path.join(dataset.source_path, sparse_folder)) and os.path.exists(os.path.join(dataset.source_path, "sparse")):
        sparse_folder = "sparse"
    
    confidence_path = os.path.join(dataset.source_path, sparse_folder, "0", "confidence_dsp.npy")
    confidence_lr = load_and_prepare_confidence(confidence_path, device='cuda', scale=(1, 100))

    scene = Scene(dataset, gaussians)

    # Setup optimizer
    if opt.pp_optimizer:
        gaussians.training_setup_pp(opt, confidence_lr)
    else:
        gaussians.training_setup(opt)

    # Restore from checkpoint if provided
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Prepare initial camera poses
    train_cams_init = scene.getTrainCameras().copy()
    create_pose_save_dirs(scene, saving_iterations)
    save_initial_pose(scene, gaussians, train_cams_init, saving_iterations)

    # Setup background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Setup timing and progress
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    start = time()

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Disable pose gradient if not optimizing pose
        if not opt.optim_pose:
            gaussians.P.requires_grad_(False)

        # Increase SH degree periodically
        if iteration % SH_DEGREE_UPDATE_INTERVAL == 0:
            gaussians.oneupSHdegree()

        # Pick a random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)
        pose = gaussians.get_RT(viewpoint_cam.uid)

        # Enable debug mode if specified
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Render
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)

        image = render_pkg["render"]
        viewspace_points = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss, l1_val, _ = compute_loss(image, gt_image, opt.lambda_dssim)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Update progress bar
            ema_loss_for_log = update_progress_bar(progress_bar, loss.item(), ema_loss_for_log, iteration, opt.iterations)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Final iteration: save time and run validation
            if iteration == opt.iterations:
                end = time()
                train_time = end - start
                save_time(scene.model_path, '[2] train_joint_TrainTime', train_time)
                training_report(
                    tb_writer, iteration, l1_val, loss,
                    iter_start.elapsed_time(iter_end),
                    testing_iterations, scene, render, (pipe, background)
                )

            # Save gaussians and poses
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                save_pose(
                    os.path.join(scene.model_path, f'pose/ours_{iteration}', 'pose_optimized.npy'),
                    gaussians.P, train_cams_init
                )

            # Save checkpoint
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    os.path.join(scene.model_path, f"chkpnt{iteration}.pth")
                )

    # Final time logging
    end = time()
    total_train_time = end - start
    save_time(scene.model_path, '[2] train_joint', total_train_time)


# =============================================================================
# Entry Point
# =============================================================================

def main() -> None:
    """Parse arguments and run training."""
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action='store_true')
    parser.add_argument('--disable_viewer', action='store_true', default=True)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)
    print(f"Optimizing {args.model_path}")

    # Initialize system state
    safe_state(args.quiet)

    # Start GUI server if enabled
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)

    # Enable anomaly detection if requested
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Run training
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
