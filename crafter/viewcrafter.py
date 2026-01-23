import sys
import pycolmap

import trimesh
import torch
import numpy as np
import torchvision
import os
import copy
import cv2  
import glob
from PIL import Image
import pytorch3d
from pytorch3d.structures import Pointclouds
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.pvd_utils import *
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis
from pathlib import Path

# Add to_numpy helper
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return [to_numpy(i) for i in x]
    return np.array(x)

class SimpleScene:
    def __init__(self, c2ws, focals, pps, pts3d, depths, imgs):
        self.c2ws = c2ws
        self.focals = focals
        self.pps = pps
        self.pts3d = pts3d
        self.depths = depths
        self.imgs = imgs
        self.min_conf_thr = 0.0

    def get_im_poses(self):
        return self.c2ws
    def get_principal_points(self):
        return self.pps
    def get_focals(self):
        return self.focals
    def get_pts3d(self, clip_thred=None):
        return self.pts3d
    def get_depthmaps(self):
        return self.depths
    def get_masks(self):
        return [torch.ones_like(d, dtype=torch.bool) for d in self.depths]
    def conf_trf(self, x):
        return 0.0

class ViewCrafter:
    def __init__(self, opts, gradio = False):
        self.opts = opts
        self.device = opts.device
        
        self.setup_diffusion()
        
        if self.opts.cached_reconstruction_path and os.path.exists(self.opts.cached_reconstruction_path):
            print(f"Loading cached reconstruction from {self.opts.cached_reconstruction_path}")
            self.load_cached_reconstruction(opts.cached_reconstruction_path)
        else:
             print("Error: Cached reconstruction path required in this optimized version.")
             # Fallback or error if dust3r is gone
             raise ValueError("This version of ViewCrafter requires --cached_reconstruction_path")
        
    def _parse_colmap_folder(self, source_dir):
        if pycolmap is None:
             raise ImportError("pycolmap is required to load from reconstruction folder directly.")
             
        sparse_dir = os.path.join(source_dir, "sparse", "0")
        if not os.path.exists(sparse_dir):
            sparse_dir = os.path.join(source_dir, "sparse")
            if not os.path.exists(os.path.join(sparse_dir, "cameras.bin")) and not os.path.exists(os.path.join(sparse_dir, "cameras.txt")):
                 raise ValueError(f"Sparse reconstruction not found at {sparse_dir}")
            
        print(f"Loading direct reconstruction from: {sparse_dir}")
        reconstruction = pycolmap.Reconstruction(sparse_dir)
        
        img_paths = []
        depth_paths = []
        extrinsics = [] 
        intrinsics = []
        
        images = sorted(reconstruction.images.values(), key=lambda x: x.name)
        depths_dir = os.path.join(source_dir, "depths")
        has_depths = os.path.exists(depths_dir)
        
        if not has_depths:
            raise ValueError("depths directory is required in source directory")

        for img in images:
            image_name = img.name
            candidates_img = [
                os.path.join(source_dir, "images", image_name),
                os.path.join(source_dir, image_name)
            ]
            img_path = None
            for cand in candidates_img:
                if os.path.exists(cand):
                    img_path = cand
                    break
            
            if img_path is None:
                continue

            # Extrinsics
            w2c = np.eye(4)
            rigid = img.cam_from_world()
            w2c[:3, :3] = rigid.rotation.matrix()
            w2c[:3, 3] = rigid.translation
            
            # Intrinsics
            cam = reconstruction.cameras[img.camera_id]
            fx = cam.focal_length_x
            fy = cam.focal_length_y
            cx = cam.principal_point_x
            cy = cam.principal_point_y
            
            K = np.eye(3)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            
            # Depth
            candidates_d = [
                os.path.join(depths_dir, image_name + ".npy"),
                os.path.join(depths_dir, os.path.splitext(image_name)[0] + ".npy")
            ]
            found_depth = None
            for cand in candidates_d:
                if os.path.exists(cand):
                    found_depth = cand
                    break
            
            if not found_depth:
                # If depth missing, skip this image
                continue

            intrinsics.append(K)
            extrinsics.append(w2c)
            img_paths.append(img_path)
            depth_paths.append(found_depth)
        
        print(f"Parsed {len(img_paths)} images directly from COLMAP.")
        return {
            'intrinsics': np.array(intrinsics),
            'extrinsics': np.array(extrinsics),
            'img_paths': np.array(img_paths),
            'depth_paths': np.array(depth_paths)
        }

    def load_cached_reconstruction(self, metadata_path):
        if os.path.isdir(metadata_path):
             data = self._parse_colmap_folder(metadata_path)
        else:
             data = np.load(metadata_path)
        
        # 1. Extrinsics (w2c -> c2w)
        w2cs = torch.tensor(data['extrinsics'], dtype=torch.float32).to(self.device)
        c2ws = torch.inverse(w2cs)
        
        # 2. Intrinsics
        Ks = torch.tensor(data['intrinsics'], dtype=torch.float32).to(self.device)
        focals = torch.stack([Ks[:, 0, 0], Ks[:, 1, 1]], dim=-1) # (N, 2)
        pps = torch.stack([Ks[:, 0, 2], Ks[:, 1, 2]], dim=-1) # (N, 2)
        
        img_paths = data['img_paths']
        try:
            depth_paths = data['depth_paths']
        except KeyError:
             print("Warning: depth_paths not found in metadata. Assuming standard location.")
             raise
             
        imgs_np = []
        pts3d_list = []
        depths_list = []
        self.images = [] # ViewCrafter expected format
        self.img_ori = [] # Normalized tensors [-1, 1]
        
        for i, img_p in enumerate(img_paths):
            pil_img = Image.open(str(img_p)).convert("RGB")
            
            # Downsample logic
            max_dim = 1024
            W_orig, H_orig = pil_img.size
            if max(W_orig, H_orig) > max_dim:
                scale = max_dim / max(W_orig, H_orig)
                new_W = int(W_orig * scale)
                new_H = int(H_orig * scale)
                pil_img = pil_img.resize((new_W, new_H), Image.BILINEAR)
                focals[i] *= scale
                pps[i] *= scale
                
            img_np = np.array(pil_img)
            H, W = img_np.shape[:2]
            imgs_np.append(img_np.astype(np.float32) / 255.0)
            
            img_tensor = transforms.ToTensor()(pil_img).to(self.device).unsqueeze(0) 
            self.images.append({
                'img': img_tensor,
                'true_shape': np.array([[H, W]]),
                'idx': i,
                'img_ori': img_tensor * 2.0 - 1.0 
            })
            self.img_ori.append(img_tensor * 2.0 - 1.0)
            
            d_path = str(depth_paths[i])
            depth = np.load(d_path)
            
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            
            depth_tensor = torch.tensor(depth, dtype=torch.float32).to(self.device)
            depths_list.append(depth_tensor)
            
            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            x = x.to(self.device)
            y = y.to(self.device)
            
            z = depth_tensor
            fx, fy = focals[i, 0], focals[i, 1]
            cx, cy = pps[i, 0], pps[i, 1]
            
            x_cam = (x - cx) * z / fx
            y_cam = (y - cy) * z / fy
            
            xyz_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=-1).reshape(-1, 4).T 
            xyz_world = (c2ws[i] @ xyz_cam).T[:, :3] 
            
            pts3d_list.append(xyz_world.reshape(H, W, 3))
            
        self.scene = SimpleScene(c2ws, focals, pps, pts3d_list, depths_list, imgs_np)
        print(f"Successfully loaded {len(img_paths)} frames from cache.")

    def render_pcd(self,pts3d,imgs,masks,views,renderer,device,nbv=False):
        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)

        if masks == None:
            pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
            col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
        else:
            # masks = to_numpy(masks)
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
        
        # Subsample points
        num_points = pts.shape[0]
        max_points = 500000
        if num_points > max_points:
            print(f"Subsampling point cloud from {num_points} to {max_points} points...")
            indices = torch.randperm(num_points)[:max_points]
            pts = pts[indices]
            col = col[indices]

        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)

        if nbv:
            color_mask = torch.ones(col.shape).to(device)
            point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
            view_masks = renderer(point_cloud_mask)
        else: 
            view_masks = None

        return images, view_masks
    
    def run_render(self, pcd, imgs,masks, H, W, camera_traj,num_views,nbv=False):
        chunk_size = 1 
        all_images = []
        all_viewmasks = []
        
        for i in range(0, num_views, chunk_size):
            idx_end = min(i + chunk_size, num_views)
            indices = list(range(i, idx_end))
            current_cameras = camera_traj[indices]
            current_views = len(current_cameras)
            
            render_setup = setup_renderer(current_cameras, image_size=(H,W))
            renderer = render_setup['renderer']
            
            imgs_chunk, mask_chunk = self.render_pcd(pcd, imgs, masks, current_views, renderer, self.device, nbv=nbv)
            
            all_images.append(imgs_chunk)
            if mask_chunk is not None:
                all_viewmasks.append(mask_chunk)
                
        render_results = torch.cat(all_images, dim=0)
        viewmask = torch.cat(all_viewmasks, dim=0) if all_viewmasks else None
        
        return render_results, viewmask

    
    def run_diffusion(self, renderings):
        prompts = [self.opts.prompt]
        videos = (renderings * 2. - 1.).permute(3,0,1,2).unsqueeze(0).to(self.device)
        condition_index = [0]
        with torch.no_grad(), torch.amp.autocast('cuda'):
            batch_samples = image_guided_synthesis(self.diffusion, prompts, videos, self.noise_shape, self.opts.n_samples, self.opts.ddim_steps, self.opts.ddim_eta, \
                               self.opts.unconditional_guidance_scale, self.opts.cfg_img, self.opts.frame_stride, self.opts.text_input, self.opts.multiple_cond_cfg, self.opts.timestep_spacing, self.opts.guidance_rescale, condition_index)

        return torch.clamp(batch_samples[0][0].permute(1,2,3,0), -1., 1.) 

    
    def nvs_sparse_view_interp(self):
        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)]
        depth = [i.detach() for i in self.scene.get_depthmaps()]

        # We force cached path usage
        masks = None
        mask_pc = False

        imgs = np.array(self.scene.imgs)

        camera_traj,num_views = generate_traj_interp(c2ws, H, W, focals, principal_points, self.opts.video_length, self.device)
        render_results, viewmask = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
        
        for i in range(len(self.img_ori)):
            ref_img = self.img_ori[i] 
            ref_img = F.interpolate(ref_img, size=(576, 1024), mode='bilinear', align_corners=False)
            ref_img = ref_img.squeeze(0).permute(1, 2, 0) 
            ref_img = (ref_img + 1.0) / 2.0 
            
            # Bound checking just in case
            idx = i*(self.opts.video_length - 1)
            if idx < len(render_results):
                render_results[idx] = ref_img

        save_video(render_results, os.path.join(self.opts.save_dir, f'render.mp4'))
        save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path=os.path.join(self.opts.save_dir, f'pcd.ply') , mask_pc=mask_pc, reduce_pc=False)

        diffusion_results = []
        print(f'Generating {len(self.img_ori)-1} clips\n')
        for i in range(len(self.img_ori)-1 ):
            print(f'Generating clip {i} ...\n')
            diffusion_results.append(self.run_diffusion(render_results[i*(self.opts.video_length - 1):self.opts.video_length+i*(self.opts.video_length - 1)]))
        print(f'Finish!\n')
        diffusion_results = torch.cat(diffusion_results)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion.mp4'))
        return diffusion_results

    def setup_diffusion(self):
        seed_everything(self.opts.seed)
        config = OmegaConf.load(self.opts.config)
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model

        h, w = self.opts.height // 8, self.opts.width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = self.opts.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]
