import json
import math
import os
import sys
import time
import yaml
from pathlib import Path
from typing import List, Optional, Union

from hybrid_sampling import generate_camera_trajectory,se3_interpolate_to_target
from scoring_model import VirtualViewQualityScorer
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from PIL import Image
from pipeline_difix import DifixPipeline

class DiFix3DProcessor:
    def __init__(self, model_name: str = "nvidia/difix_ref", device: str = "cuda", ref_image_dir: str = None):
        self.device = device
        self.model_name = model_name
        self.pipeline = None
        self.enabled = True
        self.ref_image_dir = ref_image_dir  # ref image directory
        
        # Progressive interpolation state
        self.is_initialized = False
        self.quality_scorer = None
        self.available_interpolation_views = []
        self.training_psnr_mean = None
        self.training_psnr_variance = None
        
        # DiFix3D comparison output directory
        self.difix3d_comparison_dir = None
        
        # Store virtual view quality scores
        self.virtual_view_scores = []
        # Store baseline scores (training view PSNR baseline)
        self.baseline_scores = {}
        
        if self.enabled:
            self._initialize_pipeline()
    
    def _initialize_pipeline(self):       
        self.pipeline = DifixPipeline.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.pipeline.to(self.device)
    
    def _ensure_tensor_format(self, image_tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:

            # Record original size
        if image_tensor.dim() == 4:  # [1, H, W, 3]
            original_height, original_width = image_tensor.shape[1:3]
            tensor = image_tensor.squeeze(0)  # [H, W, 3]
        elif image_tensor.dim() == 3:  # [H, W, 3]
            original_height, original_width = image_tensor.shape[:2]
            tensor = image_tensor
        else:
            raise ValueError(f"Unsupported tensor dims: {image_tensor.shape}")
        
        # Check channel count
        if tensor.shape[-1] != 3:
            raise ValueError(f"Unsupported channel count: {tensor.shape[-1]}, expected 3")
        
        return tensor, (original_height, original_width)
    
    def process_image(
        self, 
        image_tensor: torch.Tensor, 
        prompt: str = "remove degradation",
        num_inference_steps: int = 1,
        timesteps: List[int] = [199],
        guidance_scale: float = 0.0,
        ref_image: Optional[torch.Tensor] = None,
        save_comparison: bool = False,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        if not self.enabled or self.pipeline is None:
            return image_tensor
        
        try:
            with torch.no_grad():
                # Normalize input tensor and get original size
                input_tensor, original_size = self._ensure_tensor_format(image_tensor)
                
                
                # Ensure value range in [0,1]
                if input_tensor.max() > 1.0 or input_tensor.min() < 0.0:
                    input_tensor = torch.clamp(input_tensor, 0.0, 1.0)
                    print(f"  Value range adjusted to [0,1]")
                
                # Convert to PIL image
                image_np = (input_tensor.cpu().numpy() * 255).astype(np.uint8)
                input_image = Image.fromarray(image_np)
                print(f"   PIL image size: {input_image.size}")  # (width, height)
                
                # Handle reference image if provided
                ref_image_pil = None
                if ref_image is not None:
                    ref_tensor, _ = self._ensure_tensor_format(ref_image)
                    # Ensure value range in [0,1]
                    if ref_tensor.max() > 1.0 or ref_tensor.min() < 0.0:
                        ref_tensor = torch.clamp(ref_tensor, 0.0, 1.0)
                    ref_np = (ref_tensor.cpu().numpy() * 255).astype(np.uint8)
                    ref_image_pil = Image.fromarray(ref_np)
                    print(f"   Reference image size: {ref_image_pil.size}")

                if ref_image_pil is not None:

                    
                    # Ensure input and reference sizes match exactly
                    if input_image.size != ref_image_pil.size:
                        print(f"   Resize reference image to match input: {ref_image_pil.size} -> {input_image.size}")
                        ref_image_pil = ref_image_pil.resize(input_image.size, Image.Resampling.LANCZOS)
                    
                    # Check image size for memory concerns
                    width, height = input_image.size
                    if width * height > 1000000:  # large images may cause OOM
                        print(f"    Large image size ({width}x{height}); may cause memory issues")
                    
                    # Use single image, avoid creating batch
                    try:
                        output_image = self.pipeline(
                            prompt,
                            image=input_image,
                            ref_image=ref_image_pil,
                            num_inference_steps=num_inference_steps,
                            timesteps=timesteps,
                            guidance_scale=guidance_scale
                        ).images[0]
                    except Exception as e:
                        # If single-image fails, try batch
                        input_images = [input_image, input_image]
                        ref_images = [ref_image_pil, ref_image_pil]
                        
                        output_images = self.pipeline(
                            prompt,
                            image=input_images,
                            ref_image=ref_images,
                            num_inference_steps=num_inference_steps,
                            timesteps=timesteps,
                            guidance_scale=guidance_scale
                        ).images
                        output_image = output_images[0]
                else:
                    print(f"    No reference image; run DiFix3D directly")
                    
                    try:
                        output_image = self.pipeline(
                            prompt,
                            image=input_image,
                            num_inference_steps=num_inference_steps,
                            timesteps=timesteps,
                            guidance_scale=guidance_scale
                        ).images[0]
                    except Exception as e:
                        # If single-image fails, try batch
                        input_images = [input_image, input_image]
                        
                        output_images = self.pipeline(
                            prompt,
                            image=input_images,
                            num_inference_steps=num_inference_steps,
                            timesteps=timesteps,
                            guidance_scale=guidance_scale
                        ).images
                        output_image = output_images[0]
                
                print(f"   DiFix3D output PIL size: {output_image.size}")  # (width, height)
                
                # Convert back to tensor
                output_np = np.array(output_image).astype(np.float32) / 255.0
                output_tensor = torch.from_numpy(output_np).to(image_tensor.device)
                
                print(f"   Converted tensor shape: {output_tensor.shape}")
                
                # Restore batch dimension if original had it
                if image_tensor.dim() == 4:
                    output_tensor = output_tensor.unsqueeze(0)  # [1, H, W, 3]
                
                print(f" DiFix3D processing completed:")
                print(f"   Final output shape: {output_tensor.shape}")
                print(f"   Size change: {original_size} -> {output_tensor.shape[1:3] if output_tensor.dim() == 4 else output_tensor.shape[:2]}")
                
                # Check if size changed
                final_size = output_tensor.shape[1:3] if output_tensor.dim() == 4 else output_tensor.shape[:2]
                if final_size != original_size:
                    print(f"   ⚠️ Size changed: {original_size} -> {final_size}")
                
                # 保存处理前后对比图像
                return output_tensor
                
        except Exception as e:
            print(f" DiFix3D processing failed: {e}")
            print(f"   Input tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
            print(f"   Error details: {str(e)}")
            
            # Check for einops-related errors
            if "einops" in str(e).lower() or "rearrange" in str(e).lower():
                try:
                    # Try single image with batch dim
                    single_output = self.pipeline(
                        prompt,
                        image=input_image,
                        num_inference_steps=num_inference_steps,
                        timesteps=timesteps,
                        guidance_scale=guidance_scale
                    ).images[0]
                    
                    # 转换回张量
                    output_np = np.array(single_output).astype(np.float32) / 255.0
                    output_tensor = torch.from_numpy(output_np).to(image_tensor.device)
                    
                    if image_tensor.dim() == 4:
                        output_tensor = output_tensor.unsqueeze(0)
                    
                    print(f"   Single-image processing succeeded")
                    return output_tensor
                    
                except Exception as e2:
                    print(f"   Single-image processing also failed: {e2}")
            
            # Check for tensor size mismatch errors
            elif "size of tensor" in str(e).lower() and "must match" in str(e).lower():
                return image_tensor
            
            import traceback
            traceback.print_exc()
            # Return original input to keep training running
            return image_tensor
    
    def load_ref_image(self, train_idx: int, trainset) -> Optional[torch.Tensor]:
        try:
            # Check ref_image_dir
            if self.ref_image_dir is None:
                return None
            
            # Get COLMAP index from trainset
            try:
                train_data = trainset[train_idx]
                colmap_idx = train_data["colmap_image_id"]
                if isinstance(colmap_idx, torch.Tensor):
                    colmap_idx = colmap_idx.item()
                
                print(f"🔍 Train index {train_idx} -> COLMAP index {colmap_idx}")
                
            except Exception as e:
                return None
            
            # Build reference image path by COLMAP index
            ref_image_path = f"{self.ref_image_dir}/{colmap_idx:03d}.png"
            
            # Check file existence
            if not os.path.exists(ref_image_path):
                return None
            
            # Load image
            from PIL import Image
            import numpy as np
            
            ref_image_pil = Image.open(ref_image_path).convert('RGB')
            ref_image_np = np.array(ref_image_pil) / 255.0  # normalize to [0,1]
            ref_image_tensor = torch.from_numpy(ref_image_np).float().to(self.device)
            
            return ref_image_tensor
            
        except Exception as e:
            return None
    
    def process_virtual_views_batch(
        self, 
        trainset, 
        camera_optimizer, 
        rasterize_splats_fn,
        cfg,
        step: int,
        ref_image: Optional[torch.Tensor] = None,
        save_comparisons: bool = True,
        comparison_dir: Optional[str] = None
    ) -> List[dict]:
        # Set comparison image save directory
        if comparison_dir is not None:
            self.difix3d_comparison_dir = comparison_dir
        
        if not self.enabled or self.pipeline is None:
            print("DiFix3D disabled; cannot batch process")
            return []
        
        if not hasattr(trainset, '__len__') or len(trainset) == 0:
            print(" Training set empty; cannot batch process")
            return []
        
        # Ensure interpolation pool initialized
        if not self.is_initialized:
            self.initialize_interpolation_pool(trainset, rasterize_splats_fn, cfg)
            if not self.is_initialized:
                print(" Interpolation pool initialization failed")
                return []
        
        print(f" Step {step}: Start processing virtual view batch")
        
        enhanced_samples = []
        quality_threshold = 0  
        
        try:
            # 1. Choose interpolation strategy
            if len(self.available_interpolation_views) < 1:
                print(f"Not enough views in pool ({len(self.available_interpolation_views)} < 1); cannot interpolate")
                return []
            
            # Randomly choose two training views for forward interpolation
            train_indices = torch.randperm(len(trainset))[:2]
            train_view1 = trainset[train_indices[0]]
            train_view2 = trainset[train_indices[1]]
            
            # Backward interpolation: choose two virtual views
            print(f"  Virtual view pool: {len(self.available_interpolation_views)} available views")
            if len(self.available_interpolation_views) >= 2:
                virtual_indices = torch.randperm(len(self.available_interpolation_views))[:2]
                virtual_view1 = self.available_interpolation_views[virtual_indices[0]]
                virtual_view2 = self.available_interpolation_views[virtual_indices[1]]
                use_backward_interpolation = True
                print(f"  Backward interpolation enabled: choose virtual views {virtual_indices[0]} and {virtual_indices[1]}")
            else:
                # If not enough virtual views, use forward interpolation only
                use_backward_interpolation = False
                virtual_view1 = None
                virtual_view2 = None
            print(f"   Forward interpolation bases: training views {train_indices[0]} and {train_indices[1]}")
            if use_backward_interpolation:
                print(f"   Backward interpolation bases: virtual views {virtual_view1['source']} and {virtual_view2['source']}")
            else:
                print(f"   Backward interpolation: skipped (need at least 2 virtual views)")
            
            # 🔍 Debug: check base views
            train_pos1 = train_view1['camtoworld'][:3, 3].to(self.device)
            train_pos2 = train_view2['camtoworld'][:3, 3].to(self.device)
            if use_backward_interpolation:
                virtual_pos1 = virtual_view1['pose'][:3, 3].to(self.device)
                virtual_pos2 = virtual_view2['pose'][:3, 3].to(self.device)
                
            
            # 2. Generate forward and backward interpolations
            forward_alpha = 0.5  
            backward_alpha = 1.5  
            
            quality_scores = []
            interpolated_poses = []
            
            # Forward interpolation between training views
            print(f"   Forward interpolation between training views (α={forward_alpha})")
            train_pose1 = train_view1["camtoworld"].to(self.device)
            train_K1 = train_view1["K"].to(self.device)
            train_pose2 = train_view2["camtoworld"].to(self.device)
            train_K2 = train_view2["K"].to(self.device)
            
            interpolated_pose_forward, _ = se3_interpolate_to_target(
                train_pose1, train_K1, 
                train_pose2, train_K2, 
                t=forward_alpha
            )
            interpolated_poses.append(interpolated_pose_forward)
            
            # Backward interpolation: outside virtual views (exploration)
            interpolated_pose_backward = None
            if use_backward_interpolation:
                print(f"  Backward interpolation: explore outside virtual views (α={backward_alpha})")
                virtual_pose1 = virtual_view1["pose"].to(self.device)
                virtual_K1 = virtual_view1["K"].to(self.device)
                virtual_pose2 = virtual_view2["pose"].to(self.device)
                virtual_K2 = virtual_view2["K"].to(self.device)
                
                # Backward interpolation exploration: extend from virtual_pose1 towards virtual_pose2
                # t=1.5 means 0.5 beyond virtual_pose2
                interpolated_pose_backward, _ = se3_interpolate_to_target(
                    virtual_pose1, virtual_K1, 
                    virtual_pose2, virtual_K2, 
                    t=backward_alpha
                )
                interpolated_poses.append(interpolated_pose_backward)
                
                # 调试：检查反向插值后的相机位置
                interp_pos_backward = interpolated_pose_backward[:3, 3]
                virtual_pos1 = virtual_pose1[:3, 3]
                virtual_pos2 = virtual_pose2[:3, 3]
            else:
                print(f"   Backward interpolation: skipped (not enough virtual views)")
            
            # 3. Generate interpolated frames
            print(f"   Start generating {cfg.virtual_view_poses_per_step} interpolated frames")
            
            for i in range(cfg.virtual_view_poses_per_step):
                # 交替生成前向和后向插值
                if i == 0:
                    # Forward interpolation: between training views
                    interpolated_pose = interpolated_pose_forward
                    direction = "forward"
                    alpha = torch.rand(1).item() * 0.8 + 0.1
                elif i == 1 and use_backward_interpolation:
                    # Backward interpolation: explore outside virtual views
                    interpolated_pose = interpolated_pose_backward
                    direction = "backward"
                    alpha = backward_alpha
                elif i == 1 and not use_backward_interpolation:
                    # 如果后向插值不可用，跳过
                    continue
                else:
                    # For more than 2, randomly choose interpolation strategy
                    if torch.rand(1).item() < 0.5:
                        # Between training views
                        alpha = torch.rand(1).item() * 0.8 + 0.1
                        interpolated_pose, _ = se3_interpolate_to_target(
                            train_pose1, train_K1, 
                            train_pose2, train_K2, 
                            t=alpha
                        )
                        direction = "forward-random"
                    elif use_backward_interpolation:
                        # Explore outside virtual views (random direction)
                        if torch.rand(1).item() < 0.5:
                            # Forward exploration: t > 1.0
                            alpha = torch.rand(1).item() * 0.5 + 1.0  # [1.0, 1.5]
                            interpolated_pose, _ = se3_interpolate_to_target(
                                virtual_pose1, virtual_K1, 
                                virtual_pose2, virtual_K2, 
                                t=alpha
                            )
                            direction = "backward-random-forward"
                        else:
                            # Backward exploration: t < 0.0
                            alpha = torch.rand(1).item() * 0.5 - 0.5  # [-0.5, 0.0]
                            interpolated_pose, _ = se3_interpolate_to_target(
                                virtual_pose2, virtual_K2, 
                                virtual_pose1, virtual_K1, 
                                t=alpha
                            )
                            direction = "backward-random-backward"
                    else:
                        # If backward interpolation unavailable, use forward interpolation
                        alpha = torch.rand(1).item() * 0.8 + 0.1
                        interpolated_pose, _ = se3_interpolate_to_target(
                            train_pose1, train_K1, 
                            train_pose2, train_K2, 
                            t=alpha
                        )
                        direction = "forward-random"
                
                print(f"    Generate {direction} interpolation frame (α={alpha:.3f})")
                
                # 🔍 Debug: check interpolated camera position
                interp_pos = interpolated_pose[:3, 3]
                
                # Use training intrinsics and image ID; ensure correct device
                interp_K = train_view1["K"].unsqueeze(0).to(self.device)  # [1, 3, 3]
                interp_img_id = train_view1["image_id"].unsqueeze(0).to(self.device)
                
                # Ensure interpolated pose is on correct device
                interpolated_pose = interpolated_pose.to(self.device)  # [4, 4]
                
                # Get image size (from training view)
                
                # Check image shape
                if len(train_view1["image"].shape) == 4:  # [1, H, W, 3]
                    height, width = train_view1["image"].shape[1:3]  # [H, W]
                    print(f"     4D image shape: [1, {height}, {width}, 3]")
                elif len(train_view1["image"].shape) == 3:  # [H, W, 3]
                    height, width = train_view1["image"].shape[:2]  # [H, W]
                    print(f"     3D image shape: [{height}, {width}, 3]")
                else:
                    print(f"     Unexpected image shape: {train_view1['image'].shape}")
                    height, width = 400, 600 
                    print(f"     Use default size: height={height}, width={width}")
                
                # Dataset __getitem__ returns image as [H, W, 3]; use [:2] for size
                if len(train_view1["image"].shape) == 3:
                    height, width = train_view1["image"].shape[:2]  # [H, W]
                    print(f"    Use 3D image shape: [{height}, {width}, 3]")
                
                print(f"     Final extracted height: {height}, width: {width}")
                
                # Render interpolated view (with depth info)
                renders_interp, depths_interp, _ = rasterize_splats_fn(
                    camtoworlds=interpolated_pose.unsqueeze(0),  # [1, 4, 4]
                    Ks=interp_K,  # [1, 3, 3]
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=interp_img_id,
                    render_mode="RGB+ED" if cfg.enable_depth_smooth_loss else "RGB",
                )
                
                # Ensure render results are on correct device
                renders_interp = renders_interp.to(self.device)
                if depths_interp is not None:
                    depths_interp = depths_interp.to(self.device)
                
                
                # Find nearest training view as reference
                nearest_train_idx = self._find_nearest_training_view(interpolated_pose, trainset)
                nearest_train_data = trainset[nearest_train_idx]
                
                # Render nearest training view as reference image
                train_pose = nearest_train_data["camtoworld"].unsqueeze(0).to(self.device)
                train_K = nearest_train_data["K"].unsqueeze(0).to(self.device)
                
                # Get training view ID
                if isinstance(nearest_train_data["image_id"], int):
                    train_view_id = nearest_train_data["image_id"]
                else:
                    train_view_id = nearest_train_data["image_id"].item()
                
                # Enhance interpolated view with DiFix3D
                # Choose reference image for interpolated view
                ref_image_for_interp = None
                if cfg.difix3d_use_ref_image:
                    # Load reference image from preset directory by train index
                    ref_image_for_interp = self.load_ref_image(nearest_train_idx, trainset)
                    
                    if ref_image_for_interp is not None:
                        print(f"    Loaded reference image from directory (train_idx={nearest_train_idx})")
                        print(f"   ref_image_for_interp shape: {ref_image_for_interp.shape}")
                        print(f"   ref_image_for_interp device: {ref_image_for_interp.device}")
                    else:
                        print(f"    Cannot load reference image (train_idx={nearest_train_idx}); will not use reference")
                else:
                    print(f"    Not using reference image for DiFix3D")
                
                
                # Ensure render format correct; handle RGB+ED 4-channel output
                if renders_interp[0].dim() != 3:
                    print(f"     Render result dims incorrect; skip DiFix3D processing")
                    enhanced_interp = renders_interp[0]  
                elif renders_interp[0].shape[-1] == 4:
                    # RGB+ED: use first 3 channels (RGB) for DiFix3D
                    print(f"    RGB+ED: extract RGB channels for DiFix3D")
                    rgb_interp = renders_interp[0][:, :, :3]  # [H, W, 3]
                    enhanced_interp = self.process_image(
                        rgb_interp,  # [H, W, 3]
                        prompt=cfg.difix3d_prompt,
                        num_inference_steps=cfg.difix3d_num_inference_steps,
                        timesteps=[199],
                        guidance_scale=cfg.difix3d_guidance_scale,
                        ref_image=ref_image_for_interp,
                        save_comparison=cfg.difix3d_save_comparisons,  # save based on config
                        save_path=f"{self.difix3d_comparison_dir}/step_{step}_view_{i}_rgb_ed"
                    )
                elif renders_interp[0].shape[-1] == 3:
                    # Standard RGB mode
                    enhanced_interp = self.process_image(
                        renders_interp[0],  # [H, W, 3]
                        prompt=cfg.difix3d_prompt,
                        num_inference_steps=cfg.difix3d_num_inference_steps,
                        timesteps=[199],
                        guidance_scale=cfg.difix3d_guidance_scale,
                        ref_image=ref_image_for_interp,
                        save_comparison=cfg.difix3d_save_comparisons,  # save based on config
                        save_path=f"{self.difix3d_comparison_dir}/step_{step}_view_{i}_rgb"
                    )
                else:
                    print(f"     Incorrect render channel count; skip DiFix3D processing")
                    enhanced_interp = renders_interp[0]  # 直接使用原始渲染结果
                
                # Ensure all tensors on correct device (before scoring)
                interpolated_pose_device = interpolated_pose.to(self.device)
                interp_K_device = interp_K[0].to(self.device)
                interp_img_id_device = interp_img_id[0].to(self.device)
                
                # Compute quality score (ensure both inputs are RGB)
                try:
                    # Ensure original image used for scoring is RGB
                    if renders_interp[0].shape[-1] == 4:
                        original_rgb_for_score = renders_interp[0][:, :, :3]  # [H, W, 3]
                    else:
                        original_rgb_for_score = renders_interp[0]  # [H, W, 3]
                    
                    _, quality_score = self.quality_scorer.score_pseudo_view(
                        original_rgb_for_score, enhanced_interp
                    )
                    print(f"   Interpolated frame quality score: k={quality_score:.4f}")
                    
                    # Save virtual view quality score data
                    score_data = {
                        "step": step,
                        "view_idx": i,
                        "direction": direction,
                        "alpha": alpha,
                        "quality_score": float(quality_score),
                        "nearest_train_idx": nearest_train_idx,
                        "interpolated_pose": interpolated_pose_device.cpu().numpy().tolist(),
                        "timestamp": time.time()
                    }
                    self.virtual_view_scores.append(score_data)
                    print(f"    Saved virtual view quality score data")
                    
                except Exception as e:
                    print(f"   ⚠️ Quality scoring failed: {e}")
                    quality_score = 0.0  # 默认质量评分
                    
                    # Save data even if scoring fails
                    score_data = {
                        "step": step,
                        "view_idx": i,
                        "direction": direction,
                        "alpha": alpha,
                        "quality_score": 0.0,
                        "nearest_train_idx": nearest_train_idx,
                        "interpolated_pose": interpolated_pose_device.cpu().numpy().tolist(),
                        "timestamp": time.time(),
                        "error": str(e)
                    }
                    self.virtual_view_scores.append(score_data)
                    print(f"    Saved virtual view quality score data (scoring failed)")
                
                # Ensure tensors on correct device
                enhanced_interp_device = enhanced_interp.to(self.device)
                
                # Add to enhanced samples (for re-rendering and loss)
                sample = {
                    "enhanced_image": enhanced_interp_device.detach().clone(),  
                    "pose": interpolated_pose_device.detach().clone(),  
                    "K": interp_K_device.detach().clone(),  
                    "image_id": interp_img_id_device.detach().clone(),  
                    "width": width,  
                    "height": height,  
                    "view_idx": i,
                    "interpolated": True,
                    "alpha": alpha,
                    "nearest_train_idx": nearest_train_idx,
                    "quality_score": quality_score,  
                }
                
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        print(f"     {key}: {value.device}, {value.shape}")
                    else:
                        print(f"     {key}: {type(value)}")
                should_add_to_pool = False
                # Forward interpolation: quality check (PSNR delta)
                # Accept if PSNR delta within configured range
                if (quality_score < cfg.interp_quality_psnr_max) and (quality_score > cfg.interp_quality_psnr_min):
                    should_add_to_pool = True
                    enhanced_samples.append(sample)
                    print(f"    Forward interpolation accepted (PSNR Δ={quality_score:.4f}, range {cfg.interp_quality_psnr_min}~{cfg.interp_quality_psnr_max}); add to pool")
                else:
                    print(f"    Forward interpolation rejected (PSNR Δ={quality_score:.4f}, out of range {cfg.interp_quality_psnr_min}~{cfg.interp_quality_psnr_max}); not added")
                
                # Only accepted interpolations go to available pool
                if should_add_to_pool:
                    self.available_interpolation_views.append({
                        "pose": interpolated_pose_device,  
                        "K": interp_K_device,  
                        "image_id": interp_img_id_device,  
                        "enhanced_image": enhanced_interp,
                        "source": f"interpolated_step_{step}_view_{i}",
                        "quality_score": quality_score,  #
                        "direction": direction  
                    })
                    print(f"    Interpolation added to pool; current size: {len(self.available_interpolation_views)}")
                else:
                    print(f"    Interpolation not added; current pool size: {len(self.available_interpolation_views)}")
                
                print(f"    Interpolated frame {i+1}/{cfg.virtual_view_poses_per_step} processed (α={alpha:.3f}, reference train view={nearest_train_idx})")
                print(f"    Virtual view pool updated: now {len(self.available_interpolation_views)} views")
            
        except Exception as e:
            print(f" Virtual view batch processing failed: {e}")
            return []
        
        # Print processing results
        if enhanced_samples:
            print(f" Step {step} virtual view batch completed!")
            print(f"   Generated {len(enhanced_samples)} enhanced views")
            print(f"   Interpolation pool now has {len(self.available_interpolation_views)} views")
        else:
            print(f" Step {step} virtual view batch failed; no successful views generated")
        
        return enhanced_samples
    
    def _find_nearest_training_view(self, target_pose: torch.Tensor, trainset) -> int:
        """
        Find the nearest training view to the target pose.
        
        Args:
            target_pose: target pose [4, 4]
            trainset: training dataset
            
        Returns:
            Index of the nearest training view
        """
        min_distance = float('inf')
        nearest_idx = 0
        
        target_position = target_pose[:3, 3]  # [3]
        
        for i in range(len(trainset)):
            train_data = trainset[i]
            train_pose = train_data["camtoworld"].to(self.device)  # [4, 4]
            train_position = train_pose[:3, 3]  # [3]
            
            # Euclidean distance
            distance = torch.norm(target_position - train_position).item()
            
            if distance < min_distance:
                min_distance = distance
                nearest_idx = i
        
        return nearest_idx
    def initialize_interpolation_pool(
        self,
        trainset,
        rasterize_splats_fn,
        cfg
    ):
        """
        One-time initialization of interpolation pool and PSNR baseline.
        
        Args:
            trainset: training dataset
            rasterize_splats_fn: 3DGS rendering function
            cfg: config object
        """
        if self.is_initialized:
            print(" Interpolation pool already initialized; skipping")
            return
        
        print(" Initializing interpolation pool and PSNR baseline...")
        
        # 1. Initialize VirtualViewQualityScorer
        self.quality_scorer = VirtualViewQualityScorer()
        print("VirtualViewQualityScorer initialized")
        
        # 2. Process training views and compute fixed PSNR baseline
        print("Processing training views and computing PSNR baseline...")
        
        # Use first 3 training views as baseline
        num_training_views = min(3, len(trainset))
        all_original_views = []
        all_difix_views = []
        
        for i in range(num_training_views):
            try:
                # Get training view data
                train_data = trainset[i]
                train_pose = train_data["camtoworld"].unsqueeze(0).to(self.device)  # [1, 4, 4]
                train_K = train_data["K"].unsqueeze(0).to(self.device)  # [1, 3, 3]
                train_image = train_data["image"].unsqueeze(0).to(self.device) / 255.0  # [1, H, W, 3]
                
                # Ensure image_id is tensor
                if isinstance(train_data["image_id"], int):
                    train_img_id = torch.tensor([train_data["image_id"]], device=self.device)
                else:
                    train_img_id = train_data["image_id"].unsqueeze(0).to(self.device)
                
                height, width = train_image.shape[1:3]
                
                # Render training view
                renders_train, _, _ = rasterize_splats_fn(
                    camtoworlds=train_pose,
                    Ks=train_K,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=train_img_id,
                    render_mode="RGB",
                )
                
                # Enhance training view using DiFix3D
                print(f"    Start DiFix3D processing for training view {i+1}...")
                
                # Choose reference image: use another training view's render
                ref_image_for_training = None
                if cfg.difix3d_use_ref_image:
                    # Choose another training view's raw render as reference
                    ref_idx = (i + 1) % num_training_views
                    if ref_idx != i:  # 确保不是同一个视角
                        # Render reference view
                        ref_train_data = trainset[ref_idx]
                        ref_train_pose = ref_train_data["camtoworld"].unsqueeze(0).to(self.device)
                        ref_train_K = ref_train_data["K"].unsqueeze(0).to(self.device)
                        
                        # Ensure image_id is tensor
                        if isinstance(ref_train_data["image_id"], int):
                            ref_train_img_id = torch.tensor([ref_train_data["image_id"]], device=self.device)
                        else:
                            ref_train_img_id = ref_train_data["image_id"].unsqueeze(0).to(self.device)
                        
                        # Render reference view
                        ref_renders, _, _ = rasterize_splats_fn(
                            camtoworlds=ref_train_pose,
                            Ks=ref_train_K,
                            width=width,  # 使用当前视角的尺寸
                            height=height,
                            sh_degree=cfg.sh_degree,
                            near_plane=cfg.near_plane,
                            far_plane=cfg.far_plane,
                            image_ids=ref_train_img_id,
                            render_mode="RGB",
                        )
                        
                        ref_image_for_training = ref_renders[0].to(self.device)  # [H, W, 3] - correct device
                        print(f"    Use training view {ref_idx+1} raw render as reference image")
                    else:
                        print(f"    Cannot select a different training view; skip reference image")
                else:
                    print(f"    Not using reference image for DiFix3D")
                
                enhanced_train = self.process_image(
                    renders_train[0],  # [H, W, 3]
                    prompt=cfg.difix3d_prompt,
                    num_inference_steps=cfg.difix3d_num_inference_steps,
                    timesteps=[199],
                    guidance_scale=cfg.difix3d_guidance_scale,
                    ref_image=ref_image_for_training,  # 使用选择的参考图像
                    save_comparison=False
                )
                print(f"    DiFix3D processing completed")
                
                # Collect data for PSNR computation
                all_original_views.append(renders_train[0])
                all_difix_views.append(enhanced_train)
                
                # 🔍 Debug: check image equality
                print(f"    训练视角 {i+1} 调试信息:")
                print(f"     原始图像形状: {renders_train[0].shape}, 范围: [{renders_train[0].min():.4f}, {renders_train[0].max():.4f}]")
                print(f"     DiFix图像形状: {enhanced_train.shape}, 范围: [{enhanced_train.min():.4f}, {enhanced_train.max():.4f}]")
                
                # 计算MSE来检查图像差异
                mse = torch.mean((renders_train[0] - enhanced_train) ** 2)
                print(f"     Image MSE: {mse.item():.8f}")
                
                if mse < 1e-8:
                    print(f"      Warning: original and DiFix images nearly identical!")
                    print(f"       DiFix3D processing might not be effective")
                else:
                    print(f"      Images differ; DiFix3D processing effective")
                
                # Directly add to available interpolation pool (training views don't need scoring)
                self.available_interpolation_views.append({
                    "pose": train_pose[0],  # [4, 4]
                    "K": train_K[0],  # [3, 3]
                    "image_id": train_img_id[0],
                    "enhanced_image": enhanced_train,
                    "source": f"training_view_{i}"
                })
                
                print(f"    Training view {i+1}/{num_training_views} processed")
                
            except Exception as e:
                print(f"    Training view {i} processing failed: {e}")
                continue
        
        if len(all_original_views) == 0:
            print(" No successfully processed training views; cannot compute PSNR baseline")
            return
        
        # 3. Compute fixed PSNR baseline (not updated later)
        try:
            self.training_psnr_mean, self.training_psnr_variance = self.quality_scorer.evaluate_training_views(
                all_original_views, all_difix_views
            )
            
            # Validate PSNR values
            if np.isinf(self.training_psnr_mean) or np.isnan(self.training_psnr_mean):
                print(f" Invalid PSNR mean: {self.training_psnr_mean}")
                print(f"   This often means DiFix3D output equals original image")
                print(f"   Please verify DiFix3D works properly")
                raise ValueError("PSNR mean is inf; DiFix3D may not be effective")
            
            if np.isinf(self.training_psnr_variance) or np.isnan(self.training_psnr_variance):
                print(f"Invalid PSNR variance: {self.training_psnr_variance}")
                print(f"   This often means all PSNR values are identical (inf)")
                raise ValueError("PSNR variance is nan; images may be identical")
            
            print(f" Fixed PSNR baseline computed: mean={self.training_psnr_mean:.4f}, var={self.training_psnr_variance:.4f}")
            
            # Save baseline scores
            self.baseline_scores = {
                "training_psnr_mean": float(self.training_psnr_mean),
                "training_psnr_variance": float(self.training_psnr_variance),
                "training_views_count": len(all_original_views),
                "timestamp": time.time()
            }
            print(f" Saved baseline scores: mean={self.training_psnr_mean:.4f}, var={self.training_psnr_variance:.4f}")
            
        except Exception as e:
            print(f" PSNR baseline computation failed: {e}")
            print(f"   Possible causes:")
            print(f"   1. DiFix3D may be ineffective and returned the original image")
            print(f"   2. Image data may be problematic")
            print(f"   3. Please check if DiFix3D model is loaded correctly")
            raise e
        print(f" Available interpolation view pool initialized with {len(self.available_interpolation_views)} training views")
        
        # Mark as initialized
        self.is_initialized = True
        print(" Interpolation pool and PSNR baseline initialization done")
    
