import json
import math
import os
import time
import yaml
import yaml
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

# Import SE(3) interpolation modules
import sys
from typing_extensions import assert_never

from hybrid_sampling import generate_camera_trajectory,se3_interpolate_to_target
from scoring_model import VirtualViewQualityScorer
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
from typing_extensions import Literal
from dataclasses import dataclass, field
from pytorch_msssim import ssim as pytorch_ssim
from gsplat import export_splats
from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from nerfview.viewer import Viewer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from bad_gaussians.bad_camera_optimizer import BadCameraOptimizer, BadCameraOptimizerConfig
from datasets.blender_dataperser import BlenderParser
from datasets.colmap import Dataset
from datasets.colmap_dataparser import ColmapParser
from datasets.deblur_nerf import DeblurNerfDataset
from pose_viewer import PoseViewer
from simple_trainer import Config, Runner, create_splats_with_optimizers
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)
from utils import (
    AppearanceOptModule,
    CameraOptModuleSE3,
    CameraOptModule,
    set_random_seed,
    rgb_to_sh,
)

from pection_loss import VGG16PerceptualLoss, VGG16PerceptualLossWithMultipleLayers, VGG16DISTSLoss

from PIL import Image
from difix3d_processor import DiFix3DProcessor
    

@dataclass
class DeblurDiFix3DConfig(Config):
    
    # Data settings
    data_dir: str = "path_of_your_data"
    data_factor: int = 1
    # Optional: specify training image IDs; None uses all default training views
    train_indices: Optional[List[int]] = None
    
    # Evaluation settings
    eval_only: bool = False
    scale_factor: float = 1.0
    result_dir: str = "path_of_your_result"
    test_every: int = 0

    ########### Viewer ###############
    disable_viewer: bool = False
    port: int = 8080
    visualize_cameras: bool = True

    ########### Training ###############
    max_steps: int = 30000
    eval_steps: List[int] = field(default_factory=lambda: [1_000, 3_000, 7_000])
    save_steps: List[int] = field(default_factory=lambda: [1_000, 3_000, 7_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = True
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [1_000, 7_000, 30_000])
    
    # Use fused SSIM optimization
    fused_ssim: bool = False
    pin_memory: bool = False
    
    # Save settings
    save_only_recent_train: bool = False
    
    # Batch size for training
    batch_size: int = 1
    steps_scaler: float = 1.0
    
    ########### Gaussian Initialization ###############
    init_type: Literal["random", "sfm", "gsply"] = "gsply"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    init_opa: float = 0.1
    init_scale: float = 1.0
    global_scale: float = 1.0
    init_ply_path: Optional[str] = None
    
    ########### Spherical Harmonics ###############
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    
    ########### Loss ###############
    loss_rgb_lambda: float = 0.8  # L1 loss weight
    loss_ssim_lambda: float = 0.2  # SSIM loss weight
    ssim_lambda: float = 0.2  # for compatibility
    
    ########### Rendering ###############
    near_plane: float = 0.01
    far_plane: float = 1e10
    packed: bool = False
    sparse_grad: bool = False
    antialiased: bool = False
    
    ########### Strategy ###############
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(default_factory=DefaultStrategy)

    ########### Background ###############
    random_bkgd: bool = True

    ########### Motion Deblur (BAD-Gaussians) ###############
    camera_optimizer: BadCameraOptimizerConfig = field(
        default_factory=lambda: BadCameraOptimizerConfig(
            mode="linear",
            num_virtual_views=10,  
        )
    )

    ########### DiFix3D Integration ###############
    enable_difix3d: bool = True
    
    difix3d_model_name: str = "nvidia/difix_ref"
    
    difix3d_prompt: str = "remove degradation"

    difix3d_blend_ratio: float = 1.0

    difix3d_num_inference_steps: int = 1

    difix3d_guidance_scale: float = 0.0

    difix3d_augment_training_set: bool = True
    
    difix3d_max_augmented_samples: int = 100

    difix3d_save_comparisons: bool = True

    virtual_view_start_step: int = 100

    virtual_view_interval: int = 100

    virtual_view_poses_per_step: int = 2

    virtual_view_loss_weight: float = 0.1

    interp_quality_psnr_min: float = 4.5

    interp_quality_psnr_max: float = 14.5
    
    ########### Camera Opt ###############
    pose_opt: bool = True
    pose_opt_lr: float = 5e-3
    pose_opt_reg: float = 1e-6
    pose_opt_lr_decay: float = 1e-2
    pose_noise: float = 1e-2
    pose_gradient_accumulation_steps: int = 10

    ########### Appearance Opt ###############
    app_opt: bool = False
    app_embed_dim: int = 32
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 0.0

    ########### Bilateral Grid ###############
    use_bilateral_grid: bool = False
    bilateral_grid_shape: List[int] = field(default_factory=lambda: [16, 16, 8])
    
    ########### Novel View Eval ###############
    nvs_eval_enable_during_training: bool = True
    nvs_steps: int = 200
    nvs_steps_final: int = 1000
    nvs_pose_lr: float = 1e-3
    nvs_pose_reg: float = 0.0
    nvs_pose_lr_decay: float = 1e-2
    
    ########### Deblurring Eval ###############
    deblur_eval_enable_during_training: bool = False
    deblur_eval_enable_pose_opt: bool = False
    
    ########### Regularizations ###############
    enable_phys_scale_reg: bool = False
    max_gauss_ratio: float = 10.0
    enable_mcmc_opacity_reg: bool = False
    enable_mcmc_scale_reg: bool = True
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    
    ########### Depth Smooth Loss ###############
    enable_depth_smooth_loss: bool = True
    depth_smooth_lambda: float = 0.1
    
    ########### DiFix Enhancement Loss ###############
    enable_difix_enhancement_loss: bool = True
    difix_enhancement_loss_weight: float = 0.05
    difix_enhancement_l1_weight: float = 0.8
    difix_enhancement_perceptual_weight: float = 0.2

    # Avoid multiple initialization
    bad_gaussians_post_init_complete: bool = False

    def __post_init__(self):
        if not self.bad_gaussians_post_init_complete:
            self.bad_gaussians_post_init_complete = True
            timestr = time.strftime("%Y%m%d-%H%M%S")
            self.result_dir = Path(self.result_dir) / timestr
            if isinstance(self.strategy, DefaultStrategy):
                if self.camera_optimizer.mode == "off":
                    div_factor = 1.0
                else:
                    div_factor = self.camera_optimizer.num_virtual_views
                self.strategy.grow_grad2d = self.strategy.grow_grad2d / div_factor
                self.strategy.reset_every = 999999999


def depth_smooth_loss_4neighbor(depth_map: torch.Tensor) -> torch.Tensor:
    """Computes the smoothness loss for a depth map using L2 norm of gradients."""
    # Ensure input format is [B, H, W]
    if depth_map.dim() == 4:
        depth_map = depth_map.squeeze(-1)  # [B, H, W, 1] -> [B, H, W]
    
    if depth_map.dim() != 3:
        raise ValueError(f"Depth map must be 3D [B, H, W], got: {depth_map.shape}")
    
    # Compute gradients using torch.diff for better readability and performance
    diff_h = torch.diff(depth_map, dim=2)  # [B, H, W-1]
    diff_v = torch.diff(depth_map, dim=1)  # [B, H-1, W]
    
    # Total smoothness loss (L2)
    return diff_h.pow(2).mean() + diff_v.pow(2).mean()


class DeblurDiFix3DRunner(Runner):
    """BAD-Gaussians deblurring + DiFix3D training engine"""

    def __init__(self, local_rank: int, world_rank, world_size: int, cfg: DeblurDiFix3DConfig) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        self._init_directories()

        self._init_dataset()

        self._init_difix3d()

        self._init_gaussians()

        self._init_optimizers()

        self._init_metrics()

        self._init_viewer()

        self.cfg_to_save = cfg
        
        # Store virtual camera batches
        self.virtual_camera_batches = []
        # Store all training camera positions
        self.all_train_cameras = None
        
        # Store virtual view quality scores
        self.virtual_view_scores = []
        # Store baseline score data (training PSNR baseline)
        self.baseline_scores = {}
        
        # Initialize hybrid sampling strategy (on-demand interpolation)
        # Track hybrid sampling state
        self.hybrid_sampling_initialized = False

        self.enhanced_data = []

    def _init_directories(self):
        """Initialize all output directories."""
        cfg = self.cfg
        self.result_dir = cfg.result_dir
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)
        self.difix3d_comparison_dir = f"{cfg.result_dir}/difix3d_comparisons"
        os.makedirs(self.difix3d_comparison_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

    def _init_dataset(self):
        """Initialize dataset and related components."""
        cfg = self.cfg
        self.scene_name = Path(cfg.data_dir).name
        print(f" Scene name: {self.scene_name}")
        self.scene_scale = cfg.global_scale

        self.parser = ColmapParser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            scale_factor=cfg.scale_factor,
            downscale_rounding_mode="round",
            adjust_scene=(cfg.init_type != "gsply"),
        )
        self.parser.train_indices = cfg.train_indices
        if cfg.train_indices is not None:
            print(f"[Dataset] Using configured training indices: {cfg.train_indices}")
        
        # Debug checks
        print(f" ColmapParser config check:")
        print(f"   data_factor: {cfg.data_factor}")
        print(f"   scale_factor: {cfg.scale_factor}")
        
        self.trainset = DeblurNerfDataset(self.parser, split="train")
        self.valset = DeblurNerfDataset(self.parser, split="val")
        self.testset = DeblurNerfDataset(self.parser, split="test")
        self.trajectory_generator = generate_camera_trajectory
        self.quality_scorer = VirtualViewQualityScorer(device=self.device)
        print(f" Virtual view quality scoring model initialized")

    def _init_difix3d(self):
        """Initialize DiFix3D processor."""
        cfg = self.cfg
        if cfg.enable_difix3d:
            print(" Initializing DiFix3D processor...")
            self.difix3d_processor = DiFix3DProcessor(
                model_name=cfg.difix3d_model_name,
                device=self.device,
            )
            if self.difix3d_processor.enabled:
                print(f" DiFix3D processor initialized")
            else:
                print(f" DiFix3D processor initialization failed; disabling DiFix3D features")
                cfg.enable_difix3d = False
        else:
            self.difix3d_processor = None
            print(" DiFix3D features disabled")

    def _init_gaussians(self):
        """Initialize 3D Gaussian splats."""
        cfg = self.cfg
        feature_dim = cfg.app_embed_dim if cfg.app_opt else None

        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=self.world_rank,
            world_size=self.world_size,
            ply_path=cfg.init_ply_path,
        )
        print(len(self.splats["means"]))

        # Densification strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

    def _init_optimizers(self):
        """Initialize optimizers for camera, appearance, and bilateral grid."""
        cfg = self.cfg
        
        # Camera Optimizer
        total_cameras = len(self.trainset) + (len(self.valset) if self.valset else 0) + (len(self.testset) if self.testset else 0)
        
        if (cfg.camera_optimizer.mode == "off" and cfg.pose_opt):
            print("Switching to CameraOptModule for static pose optimization (mode=off, pose_opt=True)")
            cfg.camera_optimizer.num_virtual_views = 1
            self.camera_optimizer = CameraOptModule(total_cameras).to(self.device)
            self.camera_optimizer.zero_init()
        else:
            self.camera_optimizer = cfg.camera_optimizer.setup(
                num_cameras=total_cameras,
                device=self.device,
            )
            
        # Get params for camera optimizer
        camera_opt_module = self.camera_optimizer.module if hasattr(self.camera_optimizer, 'module') else self.camera_optimizer
        if isinstance(camera_opt_module, CameraOptModule):
            camera_params = list(camera_opt_module.parameters())
        else:
            groups = {}
            camera_opt_module.get_param_groups(groups)
            camera_params = groups["camera_opt"]
        
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_optimizers = [
                torch.optim.Adam(
                    camera_params,
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if self.world_size > 1:
            self.camera_optimizer = DDP(self.camera_optimizer)

        # Appearance Optimizer
        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), cfg.app_embed_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if self.world_size > 1:
                self.app_module = DDP(self.app_module)

        # Bilateral Grid Optimizer
        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1], 
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

    def _init_metrics(self):
        """Initialize evaluation metrics and losses."""
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        self.perceptual_loss = VGG16PerceptualLoss(feature_layer='relu2_2', device=self.device, enable_timing=False)
        self.dists_loss = VGG16DISTSLoss(device=self.device, enable_timing=False)

    def _init_viewer(self):
        """Initialize visualization server."""
        if not self.cfg.disable_viewer:
            import nerfview
            self.server = viser.ViserServer(port=self.cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def collect_train_camera_data(self):
        """
        Collect all training camera poses
        """
        if self.all_train_cameras is None:
            train_cameras = []
            for i in range(len(self.trainset)):
                camera_info = self.trainset[i]
                if 'camtoworld' in camera_info:
                    train_cameras.append(camera_info['camtoworld'])
                elif 'pose' in camera_info:
                    train_cameras.append(camera_info['pose'])
            
            if train_cameras:
                self.all_train_cameras = torch.stack(train_cameras).to(self.device)  # [N, 4, 4]
                print(f"Collected {len(train_cameras)} training camera poses")
            else:
                print("Unable to get camera poses from training dataset")

    def collect_virtual_camera_data(self, camera_poses: torch.Tensor = None, enhanced_samples: List[dict] = None, step: int = None, source: str = "unknown"):
        """
        Collect virtual camera poses (unified interface)
        
        Args:
            camera_poses: camera poses [N, 4, 4] (BAD-Gaussians)
            enhanced_samples: list of enhanced samples (DiFix3D)
            step: current step (optional)
            source: data source ("BAD-Gaussians" or "DiFix3D")
        """
        if camera_poses is not None and len(camera_poses) > 0:
            # BAD-Gaussians virtual cameras
            self.virtual_camera_batches.append(camera_poses.detach().clone())
            step_info = f"step {step}: " if step is not None else ""
            print(f"{step_info}collected {len(camera_poses)} {source} virtual cameras")
        elif enhanced_samples:
            # DiFix3D enhanced virtual cameras
            virtual_poses = []
            print(f"Debug enhanced_samples device info:")
            for i, sample in enumerate(enhanced_samples):
                pose = sample["pose"]  # [4, 4] - no batch dim
                print(f"   sample {i}: pose device={pose.device}, shape={pose.shape}, expected device={self.device}")
                if pose.device != self.device:
                    print(f"   sample {i}: moving pose from {pose.device} to {self.device}")
                    pose = pose.to(self.device)
                virtual_poses.append(pose.unsqueeze(0)) 
            
            if virtual_poses:
                print(f"Debug virtual_poses device info:")
                for i, pose in enumerate(virtual_poses):
                    print(f"   virtual_poses[{i}]: device={pose.device}, shape={pose.shape}")
                
                virtual_cameras_batch = torch.cat(virtual_poses, dim=0)  # [N, 4, 4]
                self.virtual_camera_batches.append(virtual_cameras_batch)
                print(f"Collected {len(virtual_poses)} {source} virtual cameras")
                print(f"    Current virtual camera batch count: {len(self.virtual_camera_batches)}")
                total_virtual_cameras = sum(len(batch) for batch in self.virtual_camera_batches)
                print(f"    Current total virtual cameras: {total_virtual_cameras}")

    def _save_config(self):
        """Save configuration to file."""
        if self.world_rank == 0:
            with open(f"{self.cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(self.cfg), f)

    def _create_schedulers(self):
        """Create learning rate schedulers."""
        cfg = self.cfg
        max_steps = cfg.max_steps
        
        # Means scheduler
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)),
        ]
        
        # Pose scheduler
        pose_scheduler = None
        if cfg.pose_opt:
            pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.pose_optimizers[0], gamma=cfg.pose_opt_lr_decay ** (1.0 / max_steps)
            )
            schedulers.append(pose_scheduler)

        # Bilateral grid scheduler
        if cfg.use_bilateral_grid:
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler([
                    torch.optim.lr_scheduler.LinearLR(
                        self.bil_grid_optimizers[0],
                        start_factor=0.01,
                        total_iters=1000,
                    ),
                    torch.optim.lr_scheduler.ExponentialLR(
                        self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                    ),
                ])
            )
        return schedulers, pose_scheduler

    def _init_trainloader(self):
        """Initialize data loader."""
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=self.cfg.pin_memory,
        )

    def _pre_training_tasks(self):
        """Perform tasks before training loop starts."""
        if self.cfg.visualize_cameras:
            self._init_viewer_state()

        if self.world_rank == 0:
            print("Start collecting training camera poses...")
            self.collect_train_camera_data()
            print("Saving initialization ply and rendering trajectory...")
            if self.cfg.save_ply:
                self.save_gsply(0)
            self.render_traj(step=0)

    def _compute_virtual_view_loss(self, step: int, bkgd: torch.Tensor, sh_degree_to_use: int):
        """Compute loss from virtual views (DiFix3D)."""
        cfg = self.cfg
        loss_value = 0.0
        
        # 1. Generate/Collect Virtual Views
        if step >= cfg.virtual_view_start_step and cfg.enable_difix3d and self.difix3d_processor is not None and step % cfg.virtual_view_interval == 0:
            enhanced_samples = self.difix3d_processor.process_virtual_views_batch(
                trainset=self.trainset,
                rasterize_splats_fn=self.rasterize_splats,
                cfg=cfg, step=step,
                comparison_dir=self.difix3d_comparison_dir
            )
            
            if enhanced_samples:                
                # Update sample buffer
                max_samples = getattr(cfg, 'difix3d_max_augmented_samples', 100)
                for enhanced_sample in enhanced_samples:
                    if len(self.enhanced_data) >= max_samples:
                        self.enhanced_data.pop(0)
                    self.enhanced_data.append(enhanced_sample)
                
                self.collect_virtual_camera_data(enhanced_samples=enhanced_samples, source="DiFix3D-Progressive")
                
                # Simple progress log
                if step % 1000 == 0:
                    print(f"Progressive interpolation: {len(enhanced_samples)} new samples")

            # 2. Compute Loss from Virtual Views
            if len(self.enhanced_data) > 0:
                import random
                sample = random.choice(self.enhanced_data)
                
                # Prepare virtual view data
                device = self.device
                virtual_pose = sample["pose"].unsqueeze(0).to(device)
                virtual_K = sample["K"].unsqueeze(0).to(device)
                virtual_image_id = sample["image_id"].unsqueeze(0).to(device)
                
                # Re-render
                renders_virtual, alphas_virtual, _ = self.rasterize_splats(
                    camtoworlds=virtual_pose,
                    Ks=virtual_K,
                    width=sample["width"],
                    height=sample["height"],
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=virtual_image_id,
                    render_mode="RGB+ED" if cfg.enable_depth_smooth_loss else "RGB",
                )
                
                colors_virtual = renders_virtual[..., 0:3] if renders_virtual.shape[-1] == 4 else renders_virtual
                depths_virtual = renders_virtual[..., 3:4] if renders_virtual.shape[-1] == 4 else None
                
                if cfg.random_bkgd:
                    colors_virtual = colors_virtual + bkgd * (1.0 - alphas_virtual)
                
                # DiFix distillation loss
                loss_sample = 0.0
                enhanced_image = sample["enhanced_image"].to(device)
                if enhanced_image.dim() == 3: enhanced_image = enhanced_image.unsqueeze(0)
                
                if cfg.enable_difix_enhancement_loss:
                    l1_loss = F.l1_loss(colors_virtual, enhanced_image)
                    dists_loss = self.dists_loss(colors_virtual, enhanced_image)
                    loss_sample += (l1_loss * cfg.difix_enhancement_l1_weight + dists_loss * 0.01)
                
                if cfg.enable_depth_smooth_loss and depths_virtual is not None:
                    loss_sample += depth_smooth_loss_4neighbor(depths_virtual) * cfg.depth_smooth_lambda
                    
                loss_value = cfg.virtual_view_loss_weight * loss_sample
            
        return loss_value

    def _optimization_step(self, step: int, loss: torch.Tensor, info: dict, schedulers: list, Ks: torch.Tensor):
        """Execute backward pass and optimization step."""
        loss.backward()
        
        cfg = self.cfg
        
        # Strategy post-backward
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
             self.cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                lr=schedulers[0].get_last_lr()[0],
            )

        # Sparse Gradients Handling
        if cfg.sparse_grad:
            assert cfg.packed, "Sparse gradients only work with packed mode."
            gaussian_ids = info["gaussian_ids"]
            for k in self.splats.keys():
                grad = self.splats[k].grad
                if grad is None or grad.is_sparse: continue
                self.splats[k].grad = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],
                    values=grad[gaussian_ids],
                    size=self.splats[k].size(),
                    is_coalesced=len(Ks) == 1,
                )

        # Optimizer steps
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        for optimizer in self.pose_optimizers:
            if step % cfg.pose_gradient_accumulation_steps == cfg.pose_gradient_accumulation_steps - 1:
                optimizer.step()
            if step % cfg.pose_gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
                
        for optimizer in self.app_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        for optimizer in self.bil_grid_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        for scheduler in schedulers:
            if scheduler is not None:
                scheduler.step()

    def _log_and_checkpoint(self, step: int, max_steps: int, loss_items: dict, schedulers: list):
        """Handle logging and checkpointing."""
        cfg = self.cfg
        
        # Tensorboard logging
        if self.world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            self.writer.add_scalar("train/loss", loss_items['loss'], step)
            self.writer.add_scalar("train/l1loss", loss_items['l1loss'], step)
            self.writer.add_scalar("train/ssimloss", loss_items['ssimloss'], step)
            self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
            self.writer.add_scalar("train/mem", mem, step)
            
            # Monitor camera pose optimization
            metrics_dict = {}
            camera_optimizer = self.camera_optimizer.module if hasattr(self.camera_optimizer, 'module') else self.camera_optimizer
            if hasattr(camera_optimizer, "get_metrics_dict"):
                camera_optimizer.get_metrics_dict(metrics_dict)
            for k, v in metrics_dict.items():
                self.writer.add_scalar(f"train/{k}", v, step)
                
            # Monitor pose learning rate
            if len(schedulers) > 1 and schedulers[1] is not None:
                 self.writer.add_scalar("train/poseLR", schedulers[1].get_last_lr()[0], step)

            if cfg.depth_loss:
                self.writer.add_scalar("train/depthloss", loss_items.get('depthloss', 0.0), step)
            if cfg.enable_depth_smooth_loss and loss_items.get('depth_smooth_loss', 0.0) > 0:
                self.writer.add_scalar("train/depth_smooth_loss", loss_items['depth_smooth_loss'], step)
            if cfg.use_bilateral_grid:
                self.writer.add_scalar("train/tvloss", loss_items.get('tvloss', 0.0), step)

            self.writer.flush()
        
        # Checkpointing
        if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            stats = {
                "mem": mem,
                "num_GS": len(self.splats["means"]),
            }
            print("Step: ", step, stats)
            with open(f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json", "w") as f:
                json.dump(stats, f)
            
            data = {"step": step, "splats": self.splats.state_dict()}
            if self.world_size > 1:
                data["camera_opt"] = self.camera_optimizer.module.state_dict()
            else:
                data["camera_opt"] = self.camera_optimizer.state_dict()
            
            if cfg.app_opt:
                if self.world_size > 1:
                    data["app_module"] = self.app_module.module.state_dict()
                else:
                    data["app_module"] = self.app_module.state_dict()
            
            torch.save(data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")
        
        # Save PLY
        if (step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1) and cfg.save_ply:
            self.save_gsply(step)

    def train(self):
        """
        Main training loop.
        Refactored for clarity and modularity.
        """
        cfg = self.cfg
        device = self.device

        self._save_config()
        max_steps = cfg.max_steps
        init_step = 0

        schedulers, pose_scheduler = self._create_schedulers()
        trainloader = self._init_trainloader()
        trainloader_iter = iter(trainloader)

        self._pre_training_tasks()

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        
        for step in pbar:
            # 1. Viewer Handling
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # 2. Data Loading
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
                
            camtoworlds = data["camtoworld"].to(device, non_blocking=True)
            Ks = data["K"].to(device, non_blocking=True)
            pixels = data["image"].to(device, non_blocking=True) / 255.0
            image_ids = data["image_id"].to(device, non_blocking=True)
            
            # 3. Camera Optimization
            camtoworlds, Ks = self._optimize_cameras(camtoworlds, Ks, image_ids)
            
            # 4. Rasterization
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            width, height = pixels.shape[2], pixels.shape[1]
            
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds, Ks=Ks, width=width, height=height,
                sh_degree=sh_degree_to_use, near_plane=cfg.near_plane, far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if (cfg.depth_loss or cfg.enable_depth_smooth_loss) else "RGB",
            )
            
            colors, depths = (renders[..., 0:3], renders[..., 3:4]) if renders.shape[-1] == 4 else (renders, None)

            # 5. Loss Computation
            # Background
            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)
            else:
                bkgd = torch.zeros(1, 3, device=device)

            # Bilateral Grid
            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=device) + 0.5) / height,
                    (torch.arange(width, device=device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            self.cfg.strategy.step_pre_backward(
                params=self.splats, optimizers=self.optimizers, state=self.strategy_state,
                step=step, info=info,
            )

            # Main Losses
            l1loss = F.l1_loss(colors, pixels)
            if self.cfg.fused_ssim:
                ssimloss = 1.0 - self.ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid")
            else:
                ssimloss = 1.0 - self.ssim(pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2))
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            
            # Optional Losses
            loss_items = {'loss': loss.item(), 'l1loss': l1loss.item(), 'ssimloss': ssimloss.item()}
            
            if cfg.depth_loss and data["depth"] is not None:
                depthloss = self._compute_depth_loss(depths, data["depth"].to(device))
                loss += depthloss
                loss_items['depthloss'] = depthloss.item()
                
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss
                loss_items['tvloss'] = tvloss.item()

            if cfg.enable_mcmc_opacity_reg:
                loss += cfg.opacity_reg * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
            if cfg.enable_mcmc_scale_reg:
                loss += cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
            if cfg.enable_phys_scale_reg and step % 10 == 0:
                scale_exp = torch.exp(self.splats["scales"])
                scale_reg = (torch.maximum(scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(cfg.max_gauss_ratio)) - cfg.max_gauss_ratio)
                loss += 0.1 * scale_reg.mean()

            if cfg.enable_depth_smooth_loss and step >= 25000 and depths is not None:
                depth_smooth_val = depth_smooth_loss_4neighbor(depths)
                loss += depth_smooth_val * cfg.depth_smooth_lambda
                loss_items['depth_smooth_loss'] = depth_smooth_val.item()

            # Virtual View Loss (DiFix3D)
            virtual_view_loss = self._compute_virtual_view_loss(step, bkgd, sh_degree_to_use)
            loss += virtual_view_loss
            
            # 6. Optimization Step
            self._optimization_step(step, loss, info, schedulers, Ks)
            loss_items['loss'] = loss.item() # Update with final loss
            
            # 7. Logging & Checkpoint
            if self.world_rank == 0:
                self._log_and_checkpoint(step, max_steps, loss_items, schedulers)
            
            # Progress bar
            desc = f"loss={loss.item():.3f}| sh degree={sh_degree_to_use}| "
            pbar.set_description(desc)

            # 8. Evaluation & Viz
            if step in [i - 1 for i in cfg.eval_steps]:
                self._run_evaluation(step)

            # 9. Viewer Update
            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_step / (time.time() - tic)
                self.viewer.update(step, num_train_rays_per_step)

    def _optimize_cameras(self, camtoworlds, Ks, image_ids):
        """Handle camera pose optimization logic."""
        cfg = self.cfg
        
        # Apply camera optimizer
        camera_optimizer = self.camera_optimizer.module if hasattr(self.camera_optimizer, 'module') else self.camera_optimizer
        if isinstance(camera_optimizer, CameraOptModule):
            camtoworlds = camera_optimizer(camtoworlds, image_ids)
        else:
            camtoworlds = camera_optimizer.apply_to_cameras(camtoworlds, image_ids, "uniform")[0]
            
        assert camtoworlds.shape[0] == cfg.camera_optimizer.num_virtual_views
        Ks = Ks.tile((camtoworlds.shape[0], 1, 1))
        return camtoworlds, Ks

    def _compute_depth_loss(self, depths, depths_gt):
        """Compute dense depth loss."""
        if depths_gt.ndim == 3: depths_gt = depths_gt.unsqueeze(-1)
        valid_mask = (depths_gt > 0).float()
        
        disp = torch.zeros_like(depths)
        disp_mask = (depths > 0)
        disp[disp_mask] = 1.0 / (depths[disp_mask] + 1e-6)

        disp_gt = torch.zeros_like(depths_gt)
        gt_mask = depths_gt > 0
        disp_gt[gt_mask] = 1.0 / (depths_gt[gt_mask] + 1e-6)
        
        depthloss = F.l1_loss(disp * valid_mask, disp_gt * valid_mask, reduction='sum')
        num_valid = valid_mask.sum()
        if num_valid > 0:
            depthloss = depthloss / num_valid
            return depthloss * self.cfg.depth_lambda * self.scene_scale
        return torch.tensor(0.0, device=self.device)

    def _run_evaluation(self, step):
        """Run evaluation tasks."""
        cfg = self.cfg
        if cfg.deblur_eval_enable_during_training and self.testset is not None:
            if cfg.deblur_eval_enable_pose_opt:
                self.eval_with_pose_opt(step, "deblur", self.testset)
            else:
                self.eval_deblur(step, "deblur", self.testset)
        if cfg.nvs_eval_enable_during_training and self.valset is not None:
             self.eval_with_pose_opt(step, "nvs", self.valset)
        self.render_traj(step)

        if self.world_rank == 0:
            total_virtual_cameras = sum(len(batch) for batch in self.virtual_camera_batches)
            total_cameras = (len(self.all_train_cameras) if self.all_train_cameras is not None else 0) + total_virtual_cameras
            
            if self.virtual_camera_batches:
                for i, batch in enumerate(self.virtual_camera_batches):
                    if len(batch) > 0:
                        positions = batch[:, :3, 3].cpu().numpy()

            # if self.difix3d_processor is not None:
            #     self.difix3d_processor.save_quality_scores_to_json(step=max_steps-1, result_dir=self.result_dir)

    def save_gsply(self, step: int = 0):
        if self.cfg.app_opt:
            # eval at origin to bake the appeareance into the colors
            rgb = self.app_module(
                features=self.splats["features"],
                embed_ids=None,
                dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                sh_degree=sh_degree_to_use,
            )
            rgb = rgb + self.splats["colors"]
            rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
            sh0 = rgb_to_sh(rgb)
            shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
        else:
            sh0 = self.splats["sh0"]
            shN = self.splats["shN"]

        means = self.splats["means"]
        scales = self.splats["scales"]
        quats = self.splats["quats"]
        opacities = self.splats["opacities"]
        export_splats(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
        )

    @torch.no_grad()
    def eval_deblur(self, step: int, stage: str, dataset: Dataset):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(testloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            image_ids = data["image_id"].to(device)

            # Apply learned mid-virtual-view pose optimizations
            camera_optimizer = self.camera_optimizer.module if hasattr(self.camera_optimizer, 'module') else self.camera_optimizer
            camtoworlds = camera_optimizer.apply_to_cameras(camtoworlds, image_ids, "mid")

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            if self.world_rank == 0:
                # write images
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                imageio.imwrite(f"{self.render_dir}/{step:04d}_{stage}_{i:04d}.png", (canvas * 255).astype(np.uint8))

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))
                    # write images
                    canvas = torch.cat([pixels, cc_colors], dim=2).squeeze(0).cpu().numpy()
                    imageio.imwrite(
                        f"{self.render_dir}/{step:04d}_{stage}_{i:04d}_corrected.png", (canvas * 255).astype(np.uint8)
                    )

        if self.world_rank == 0:
            ellipse_time /= len(testloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            
            best_stats = {}
            for k, v in metrics.items():
                if "psnr" in k or "ssim" in k:
                    best_stats[f"best_{k}"] = torch.stack(v).max().item()
                elif "lpips" in k:
                    best_stats[f"best_{k}"] = torch.stack(v).min().item()
            detailed_results = {}
            for k, v in metrics.items():
                detailed_results[f"{k}_per_sample"] = [float(val.item()) for val in v]
            
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            

            final_stats = {**stats, **best_stats, **detailed_results}
            
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )

           
            
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(final_stats, f, indent=2)
            
            # save stats to tensorboard
            for k, v in final_stats.items():
                if not k.endswith("_per_sample"): 
                    self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    def eval_with_pose_opt(self, step: int, stage: str, dataset: Dataset):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        # Freeze the scene
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group["params"][0].requires_grad = False

        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            height, width = pixels.shape[1:3]
            image_ids = data["image_id"].to(device)

            pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]

            eval_pose_adjust = CameraOptModuleSE3(1).to(self.device)
            eval_pose_adjust.random_init(cfg.pose_noise)
            eval_pose_optimizer = torch.optim.Adam(
                eval_pose_adjust.parameters(),
                lr=cfg.nvs_pose_lr * math.sqrt(cfg.batch_size),
                weight_decay=cfg.nvs_pose_reg,
                eps=1e-15,
            )

            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                eval_pose_optimizer, gamma=cfg.pose_opt_lr_decay ** (1.0 / cfg.max_steps)
            )

            NVS_STEPS = cfg.nvs_steps_final if step == cfg.max_steps - 1 else cfg.nvs_steps
            for j in range(NVS_STEPS):
                camtoworlds_new = eval_pose_adjust(camtoworlds, torch.tensor([0]).to(self.device))
                colors, alphas, info = self.rasterize_splats(
                    camtoworlds=camtoworlds_new,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB",
                )
                # clamping here should be fine since we are only optimizing the camera
                colors = torch.clamp(colors, 0.0, 1.0)
                colors_p = colors.permute(0, 3, 1, 2).detach()  # [1, 3, H, W]

                # loss
                l1loss = F.l1_loss(colors, pixels)
                loss = l1loss

                loss.backward()

                eval_pose_optimizer.step()
                eval_pose_optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                with torch.no_grad():
                    if j % 20 == 0:
                        psnr = self.psnr(colors_p, pixels_p)
                        ssim = self.ssim(colors_p, pixels_p)
                        lpips = self.lpips(colors_p, pixels_p)
                        print(
                            f"Stage {stage} at Step_{step:04d}:"
                            f"NVS_IMG_#{i:04d}_step_{j:04d}:"
                            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
                        )
                        if cfg.use_bilateral_grid:
                            cc_colors = color_correct(colors, pixels)
                            cc_colors_p = cc_colors.permute(0, 3, 1, 2)
                            cc_psnr = self.psnr(cc_colors_p, pixels_p)
                            cc_ssim = self.ssim(cc_colors_p, pixels_p)
                            cc_lpips = self.lpips(cc_colors_p, pixels_p)
                            print(
                                f"Corrected PSNR: {cc_psnr.item():.3f}, SSIM: {cc_ssim.item():.4f}, LPIPS: {cc_lpips.item():.3f} "
                            )
                        # # NVS Debugging
                        # stats = {
                        #     "psnr": psnr.item(),
                        #     "ssim": ssim.item(),
                        #     "lpips": lpips.item(),
                        # }
                        # for k, v in stats.items():
                        #     self.writer.add_scalar(f"nvs/{step}/{i}/{k}", v, j)
                        # self.writer.add_scalar(f"{stage}/{step}/{i}/pose_lr", scheduler.get_last_lr()[0], j)
                        # self.writer.add_scalar(f"{stage}/{step}/{i}/camera_opt_translation", eval_pose_adjust.poses_opt[:, :3].mean(), j)
                        # self.writer.add_scalar(f"{stage}/{step}/{i}/camera_opt_rotation", eval_pose_adjust.poses_opt[:, 3:].mean(), j)
                        # self.writer.flush()
            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)
            metrics["lpips"].append(lpips)
            if cfg.use_bilateral_grid:
                metrics["cc_psnr"].append(cc_psnr)
                metrics["cc_ssim"].append(cc_ssim)
                metrics["cc_lpips"].append(cc_lpips)
            
            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).detach().cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/{step:04d}_{stage}_{i:04d}_{j:04d}.png", (canvas * 255).astype(np.uint8)
            )
            if cfg.use_bilateral_grid:
                canvas = torch.cat([pixels, cc_colors], dim=2).squeeze(0).detach().cpu().numpy()
                imageio.imwrite(
                    f"{self.render_dir}/{step:04d}_{stage}_{i:04d}_{j:04d}_corrected.png",
                    (canvas * 255).astype(np.uint8),
                )
        # 计算平均值
        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        
        # 计算最佳值（对于PSNR和SSIM是最大值，对于LPIPS是最小值）
        best_stats = {}
        for k, v in metrics.items():
            if "psnr" in k or "ssim" in k:
                best_stats[f"best_{k}"] = torch.stack(v).max().item()
            elif "lpips" in k:
                best_stats[f"best_{k}"] = torch.stack(v).min().item()
        
        # 保存每个样本的详细结果
        detailed_results = {}
        for k, v in metrics.items():
            detailed_results[f"{k}_per_sample"] = [float(val.item()) for val in v]
        
        # 合并所有统计信息
        final_stats = {**stats, **best_stats, **detailed_results}
        
        # 打印最佳结果
        print(f"Best PSNR: {best_stats['best_psnr']:.3f}, Best SSIM: {best_stats['best_ssim']:.4f}, Best LPIPS: {best_stats['best_lpips']:.3f}")
        if cfg.use_bilateral_grid:
            print(f"Best Corrected PSNR: {best_stats['best_cc_psnr']:.3f}, Best Corrected SSIM: {best_stats['best_cc_ssim']:.4f}, Best Corrected LPIPS: {best_stats['best_cc_lpips']:.3f}")
        
        # save stats as json
        with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
            json.dump(final_stats, f, indent=2)

        # save stats to tensorboard
        for k, v in final_stats.items():
            if not k.endswith("_per_sample"):  
                self.writer.add_scalar(f"{stage}/{k}", v, step)
        self.writer.flush()

        # Unfreeze the scene
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group["params"][0].requires_grad = True

    @torch.no_grad()
    def eval_traj(self, step: int):
        # TODO: add gt trajectory

        camera_optimizer = self.camera_optimizer.module if hasattr(self.camera_optimizer, 'module') else self.camera_optimizer
        camtoworlds = camera_optimizer.get_cameras()

        raise NotImplementedError

    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        if not self.cfg.disable_viewer and isinstance(self.viewer, PoseViewer):
            assert self.viewer and self.trainset
            self.viewer.init_scene(train_dataset=self.trainset, train_state="training")


def main(local_rank: int, world_rank, world_size: int, cfg: DeblurDiFix3DConfig):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_size > 1:
            print("Viewer is disabled in distributed training.")

    runner = DeblurDiFix3DRunner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [torch.load(file, map_location=runner.device, weights_only=False) for file in cfg.ckpt]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k].detach().to(runner.device) for ckpt in ckpts])
        runner.camera_optimizer.load_state_dict(ckpts[0]["camera_opt"])
        step = ckpts[0]["step"]
        if runner.testset is not None:
            if cfg.deblur_eval_enable_pose_opt:
                runner.eval_with_pose_opt(step=step, stage="deblur", dataset=runner.testset)
            else:
                runner.eval_deblur(step=step, stage="deblur", dataset=runner.testset)
        if runner.valset is not None:
            runner.eval_with_pose_opt(step=step, stage="nvs", dataset=runner.valset)

        runner.render_traj(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:
    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default
    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25
    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            DeblurDiFix3DConfig(
                strategy=DefaultStrategy(
                    verbose=True,
                    grow_grad2d=3e-3,
                    absgrad=True,
                    refine_start_iter=1000,
                ),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            DeblurDiFix3DConfig(
                init_opa=0.5,
                init_scale=0.1,
                strategy=MCMCStrategy(verbose=True, cap_max=500_000),
            ),
        ),
    }

    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    cli(main, cfg, verbose=True)
