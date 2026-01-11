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
    def __init__(self, model_name: str = "nvidia/difix_ref", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.pipeline = None
        self.enabled = True
        
        # Progressive interpolation state
        self.quality_scorer = VirtualViewQualityScorer()
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
    
    def _prepare_pil_image(self, image_tensor: torch.Tensor) -> tuple[Image.Image, tuple[int, int]]:
        """Convert tensor to PIL Image."""
        input_tensor, original_size = self._ensure_tensor_format(image_tensor)
        input_tensor = torch.clamp(input_tensor, 0.0, 1.0)
        image_np = (input_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np), original_size

    def _run_pipeline_inference(self, prompt, image, ref_image, num_inference_steps, timesteps, guidance_scale):
        """Execute the diffusers pipeline."""
        kwargs = {
            "prompt": prompt,
            "image": image,
            "num_inference_steps": num_inference_steps,
            "timesteps": timesteps,
            "guidance_scale": guidance_scale
        }
        if ref_image is not None:
            kwargs["ref_image"] = ref_image
            
        try:
             output = self.pipeline(**kwargs).images[0]
        except Exception:
             # Fallback to batch processing if single fails
             kwargs["image"] = [image] * 2
             if ref_image:
                 kwargs["ref_image"] = [ref_image] * 2
             output = self.pipeline(**kwargs).images[0]
        return output

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
    
        with torch.no_grad():
            # Prepare inputs
            input_image, original_size = self._prepare_pil_image(image_tensor)
            
            ref_image_pil = None
            if ref_image is not None:
                ref_image_pil, _ = self._prepare_pil_image(ref_image)
                if input_image.size != ref_image_pil.size:
                    print(f"   Resize ref image: {ref_image_pil.size} -> {input_image.size}")
                    ref_image_pil = ref_image_pil.resize(input_image.size, Image.Resampling.LANCZOS)
            
            # Run Pipeline
            output_image = self._run_pipeline_inference(
                prompt, input_image, ref_image_pil, 
                num_inference_steps, timesteps, guidance_scale
            )
            
            # Post-process
            output_np = np.array(output_image).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(output_np).to(image_tensor.device)
            
            if image_tensor.dim() == 4:
                output_tensor = output_tensor.unsqueeze(0)
            
            # Check size consistency
            final_size = output_tensor.shape[1:3] if output_tensor.dim() == 4 else output_tensor.shape[:2]
            if final_size != original_size:
                print(f"   ⚠️ Size changed: {original_size} -> {final_size}")
            
            return output_tensor
    
    def process_virtual_views_batch(
        self, 
        trainset, 
        rasterize_splats_fn,
        cfg,
        step: int,
        comparison_dir: Optional[str] = None
    ) -> List[dict]:
        # Set comparison image save directory
        if comparison_dir is not None:
            self.difix3d_comparison_dir = comparison_dir
        
        if not self.enabled or self.pipeline is None: return []
        if not hasattr(trainset, '__len__') or len(trainset) == 0: return []
        
        print(f" Step {step}: Start processing virtual view batch")
        
        enhanced_samples = []
        for i in range(cfg.virtual_view_poses_per_step):
            # 1. Calculate Pose
            bases = self._select_interpolation_bases(trainset)
            pose_info = self._calculate_interpolation_pose(bases)
            
            # 2. Render, Process, and Score
            sample = self._render_and_process_view(
                pose_info, i, trainset, rasterize_splats_fn, cfg, step
            )
            
            # 3. Filter and Add to Pool
            if sample:
                if self._filter_and_store_sample(sample, cfg):
                    enhanced_samples.append(sample)
        
        # Report
        if enhanced_samples:
            print(f" Step {step} virtual view batch completed!")
            print(f"   Generated {len(enhanced_samples)} enhanced views")
            print(f"   Interpolation pool now has {len(self.available_interpolation_views)} views")
        
        return enhanced_samples

    def _select_interpolation_bases(self, trainset):
        """Select base views for interpolation."""
        # Randomly choose two training views
        train_indices = torch.randperm(len(trainset))[:2]
        train_view1, train_view2 = trainset[train_indices[0]], trainset[train_indices[1]]

        return {"train": (train_view1, train_view2)}

    def _calculate_interpolation_pose(self, bases):
        """Determine interpolation parameters and compute pose."""
        train_v1, train_v2 = bases["train"]

        # Default: Forward Random Interpolation
        alpha = torch.rand(1).item() * 0.8 + 0.1 
        pose1, K1 = train_v1["camtoworld"].to(self.device), train_v1["K"].to(self.device)
        pose2, K2 = train_v2["camtoworld"].to(self.device), train_v2["K"].to(self.device)

        interpolated_pose, _ = se3_interpolate_to_target(pose1, K1, pose2, K2, t=alpha)
        
        return {
            "pose": interpolated_pose,
            "alpha": alpha,
            "ref_K": train_v1["K"], 
            "ref_img_id": train_v1["image_id"],
            "ref_image_shape": train_v1["image"].shape,
            "ref_image": train_v1["image"]
        }

    def _render_and_process_view(self, pose_info, view_idx, trainset, rasterize_fn, cfg, step):
        """Render, enhance using DiFix, and score a single virtual view."""
        # Unpack
        pose = pose_info["pose"].to(self.device)
        K = pose_info["ref_K"].unsqueeze(0).to(self.device)
        img_id = pose_info["ref_img_id"].unsqueeze(0).to(self.device)
        
        # Determine size
        shape = pose_info["ref_image_shape"]
        if len(shape) == 4: height, width = shape[1:3]
        elif len(shape) == 3: height, width = shape[:2]
        else: height, width = 400, 600

        # Render
        renders, _, _ = rasterize_fn(
            camtoworlds=pose.unsqueeze(0), Ks=K, width=width, height=height,
            sh_degree=cfg.sh_degree, near_plane=cfg.near_plane, far_plane=cfg.far_plane,
            image_ids=img_id,
            render_mode="RGB+ED" if cfg.enable_depth_smooth_loss else "RGB",
        )
        renders = renders.to(self.device)

        # Prepare for DiFix
        if renders[0].shape[-1] == 4:
            rgb_render = renders[0][:, :, :3]
        else:
            rgb_render = renders[0]
            
        # Get Reference Image
        ref_image = pose_info["ref_image"].to(self.device)

        # DiFix Enhancement
        enhanced_image = self.process_image(
            rgb_render, prompt=cfg.difix3d_prompt,
            num_inference_steps=cfg.difix3d_num_inference_steps, timesteps=[199],
            guidance_scale=cfg.difix3d_guidance_scale, ref_image=ref_image,
            save_path=f"{self.difix3d_comparison_dir}/step_{step}_view_{view_idx}"
        )

        # Quality Scoring
        try:
            _, quality_score = self.quality_scorer.score_pseudo_view(rgb_render, enhanced_image)
        except Exception:
            quality_score = 0.0

        print(f" View {view_idx}: (alpha={pose_info['alpha']:.2f}), score={quality_score:.4f}")

        # Store score data (can be logged to file if needed)
        self.virtual_view_scores.append({
            "step": step, "view_idx": view_idx, "score": float(quality_score),
            "alpha": pose_info['alpha']
        })

        return {
            "enhanced_image": enhanced_image.detach(),
            "pose": pose.detach(),
            "K": K[0].detach(),
            "image_id": img_id[0].detach(),
            "width": width, "height": height,
            "quality_score": quality_score,
            "view_idx": view_idx,
        }

    def _filter_and_store_sample(self, sample, cfg):
        """Decide whether to add sample to the pool based on quality score."""
        score = sample["quality_score"]
        if (score < cfg.interp_quality_psnr_max) and (score > cfg.interp_quality_psnr_min):
            self.available_interpolation_views.append({
                "pose": sample["pose"],
                "K": sample["K"],
                "image_id": sample["image_id"],
                "enhanced_image": sample["enhanced_image"],
                "source": f"interpolated_{sample['view_idx']}",
                "quality_score": score,
            })
            return True
        return False
