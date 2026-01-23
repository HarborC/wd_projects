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

import os
import json
from os import makedirs
from time import time, perf_counter
from argparse import ArgumentParser

import torch
import torchvision
from tqdm import tqdm
import imageio
import numpy as np
from pathlib import Path
import glob
import shutil

from scene import Scene
from scene.dataset_readers import loadCameras
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from utils.pose_utils import get_tensor_from_camera
from utils.loss_utils import l1_loss, ssim, l1_loss_mask, ssim_loss_mask
from utils.sfm_utils import save_time
from utils.camera_utils import generate_interpolated_path, generate_interpolated_intrinsics
from utils.camera_utils import visualizer
from arguments import ModelParams, PipelineParams, get_combined_args

def save_interpolate_pose(model_path, iter, n_views):

    org_pose = np.load(model_path / f"pose/ours_{iter}/pose_optimized.npy")
    visualizer(org_pose, ["green" for _ in org_pose], model_path / f"pose/ours_{iter}/poses_optimized.png")

    # Convert w2c to c2w for interpolation (generate_interpolated_path expects c2w)
    # org_pose is (N, 4, 4) w2c format
    org_pose_c2w = np.linalg.inv(org_pose)[:, :3, :]

    n_interp = int(10 * 30 / (n_views - 1))  # 10second, fps=30
    # Use spline_degree=3 for smooth path, smoothness=0 to force path through keyframes
    all_inter_pose = generate_interpolated_path(
        poses=org_pose_c2w, 
        n_interp=n_interp, 
        spline_degree=3, 
        smoothness=0.0
    )

    # Add the last pose manually because generate_interpolated_path uses endpoint=False internally
    all_inter_pose = np.concatenate([all_inter_pose, org_pose_c2w[-1][:3, :].reshape(1, 3, 4)], axis=0)

    # Convert back to w2c for saving (keep 4x4 format)
    inter_pose_c2w_4x4 = np.concatenate([all_inter_pose, np.array([[0, 0, 0, 1]] * all_inter_pose.shape[0]).reshape(-1, 1, 4)], axis=1)
    inter_pose_w2c = np.linalg.inv(inter_pose_c2w_4x4)

    visualizer(inter_pose_w2c, ["blue" for _ in inter_pose_w2c], model_path / f"pose/ours_{iter}/poses_interpolated.png")
    np.save(model_path / f"pose/ours_{iter}/pose_interpolated.npy", inter_pose_w2c)


def images_to_video(image_folder, output_video_path, fps=30):
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    images = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            images.append(image)

    imageio.mimwrite(output_video_path, images, fps=fps)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        rendering = render(
            view, gaussians, pipeline, background, camera_pose=camera_pose
        )["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        if name != "interp":
            torchvision.utils.save_image(   
                gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
            )

def render_set_optimize(model_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        num_iter = args.optim_test_pose_iter
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))

        camera_tensor_T = camera_pose[-3:].requires_grad_()
        camera_tensor_q = camera_pose[:4].requires_grad_()
        pose_optimizer = torch.optim.Adam([
            {"params": [camera_tensor_T], "lr": 0.003},
            {"params": [camera_tensor_q], "lr": 0.001}
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-4
        )

        # Add a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pose_optimizer, T_max=num_iter, eta_min=0.0001)
        with tqdm(total=num_iter, desc=f"Tracking Time Step: {idx+1}", leave=True) as progress_bar:
            candidate_q = camera_tensor_q.clone().detach()
            candidate_T = camera_tensor_T.clone().detach()
            current_min_loss = float(1e20)
            gt = view.original_image[0:3, :, :]
            initial_loss = None

            for iteration in range(num_iter):
                rendering = render(view, gaussians, pipeline, background, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))["render"]
                black_hole_threshold = 0.0
                mask = (rendering > black_hole_threshold).float()
                loss = l1_loss_mask(rendering, gt, mask)
                loss.backward()
                with torch.no_grad():
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)

                    if iteration == 0:
                        initial_loss = loss.item()  # Capture initial loss

                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_q = camera_tensor_q.clone().detach()
                        candidate_T = camera_tensor_T.clone().detach()

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item(), initial_loss=initial_loss)
                scheduler.step()

            camera_tensor_q = candidate_q
            camera_tensor_T = candidate_T

        optimal_pose = torch.cat([camera_tensor_q, camera_tensor_T])
        # print("optimal_pose-camera_pose: ", optimal_pose-camera_pose)
        rendering_opt = render(view, gaussians, pipeline, background, camera_pose=optimal_pose)["render"]
            
        torchvision.utils.save_image(
            rendering_opt, os.path.join(render_path, view.image_name + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, view.image_name + ".png")
        )

    if args.test_fps:
        print(">>> Calculate FPS: ")
        fps_list = []
        for _ in range(1000):
            start = perf_counter()
            _ = render(view, gaussians, pipeline, background, camera_pose=optimal_pose)
            end = perf_counter()
            fps_list.append(end - start)        
        fps_list.sort()
        fps_list = fps_list[100:900]
        fps = 1 / (sum(fps_list) / len(fps_list))
        print(">>> FPS = ", fps)
        with open(f"{model_path}/total_fps.json", 'a') as fp:
            json.dump(f'{fps}', fp, indent=True)
            fp.write('\n')


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # if not skip_train:
    if not skip_train and not args.infer_video and not dataset.eval:        
        optimized_pose = np.load(Path(args.model_path) / 'pose' / f'ours_{iteration}' / 'pose_optimized.npy')
        viewpoint_stack = loadCameras(optimized_pose, scene.getTrainCameras())
        render_set(
            dataset.model_path,
            "train",
            scene.loaded_iter,
            viewpoint_stack,
            gaussians,
            pipeline,
            background,
        )

    else:
        start_time = time()
        if not skip_test:
            render_set_optimize(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                args
            )
        end_time = time()
        save_time(dataset.model_path, '[4] render', end_time - start_time)

    if args.infer_video and not dataset.eval:
        save_interpolate_pose(Path(args.model_path), iteration, args.n_views)
        interp_pose = np.load(Path(args.model_path) / 'pose' / f'ours_{iteration}' / 'pose_interpolated.npy')
        
        # Interpolate intrinsics (FOV) to match interpolated poses
        train_cam_list = scene.getTrainCameras()
        fovx_list = [c.FoVx for c in train_cam_list]
        fovy_list = [c.FoVy for c in train_cam_list]
        n_interp = int(10 * 30 / (args.n_views - 1))
        
        interp_fovx = generate_interpolated_intrinsics(fovx_list, n_interp)
        interp_fovy = generate_interpolated_intrinsics(fovy_list, n_interp)
        
        viewpoint_stack = loadCameras(interp_pose, scene.getTrainCameras(), interp_fovs=(interp_fovx, interp_fovy))
        render_set(
            dataset.model_path,
            "interp",
            scene.loaded_iter,
            viewpoint_stack,
            gaussians,
            pipeline,
            background,
        )
        image_folder = os.path.join(dataset.model_path, f'interp/ours_{iteration}/renders')
        output_video_file = os.path.join(dataset.model_path, f'interp/ours_{iteration}/interp_{args.n_views}_view.mp4')
        images_to_video(image_folder, output_video_file, fps=30)
        
        # 自动提取测试样本匹配对
        print(f"\n[Info] 开始生成测试修复样本对...")
        
        # 1. 创建保存目录
        fix_pairs_dir = os.path.join(dataset.model_path, f'fix_pairs_{iteration}')
        if os.path.exists(fix_pairs_dir):
            shutil.rmtree(fix_pairs_dir)
        os.makedirs(fix_pairs_dir)
        
        # 2. 准备原始训练相机路径映射及几何信息
        train_cams = scene.getTrainCameras()
        source_images_dir = os.path.join(dataset.source_path, "images")
        
        # 预计算训练相机的中心和方向 (World Space)
        train_cam_infos_cache = []
        for t_cam in train_cams:
            # R, T are W2C
            # Camera Center C = -R^T * T
            R_wc = t_cam.R.T
            C = -np.dot(R_wc, t_cam.T)
            # View Direction (assuming -Z is looking direction in generic OpenCV/Colmap, but let's verify)
            # In GS Utils: Z is forward? 
            # Usually column 2 of R_wc is the Z-axis vector of the camera in world space.
            # R_wc = [Right, Down, Forward] (roughly)
            # view_dir = R_wc[:, 2] 
            view_dir = R_wc[:, 2] 
            
            train_cam_infos_cache.append({
                'cam': t_cam,
                'center': C,
                'view_dir': view_dir,
                'uid': t_cam.uid
            })

        # 缓存: uid -> file_path
        train_cam_paths = {}
        for cam in train_cams:
            # cam.image_name 通常是不带后缀的文件名
            # 在 source/images 中查找匹配的文件 (忽略大小写或后缀差异)
            pattern = os.path.join(source_images_dir, f"{cam.image_name}.*")
            candidates = glob.glob(pattern)
            if candidates:
                train_cam_paths[cam.uid] = candidates[0] # 取第一个匹配项
            else:
                 # 尝试直接假设 png 或 jpg
                if os.path.exists(os.path.join(source_images_dir, f"{cam.image_name}.png")):
                     train_cam_paths[cam.uid] = os.path.join(source_images_dir, f"{cam.image_name}.png")
                elif os.path.exists(os.path.join(source_images_dir, f"{cam.image_name}.jpg")):
                     train_cam_paths[cam.uid] = os.path.join(source_images_dir, f"{cam.image_name}.jpg")

        # 3. 遍历每一张渲染图，寻找最近邻的参考图
        # viewpoint_stack 是渲染用的插值相机列表
        
        # [Modify] 限制生成的样本对数量，只取约 10 帧进行修复测试
        TARGET_FIX_COUNT = 10
        total_frames = len(viewpoint_stack)
        
        # 1. 计算每一帧到最近训练相机的距离
        frame_min_dists = []
        for idx in range(total_frames):
            render_cam = viewpoint_stack[idx]
            r_R_wc = render_cam.R.T
            r_center = -np.dot(r_R_wc, render_cam.T)
            
            min_train_dist = 1e9
            for t_info in train_cam_infos_cache:
                # Euclidean distance
                dist = np.linalg.norm(r_center - t_info['center'])
                if dist < min_train_dist:
                    min_train_dist = dist
            frame_min_dists.append(min_train_dist)
            
        # [Modify] 3. 分层最大距离采样 (Stratified Max-Distance Sampling)
        # 将时间轴划分为N个区间，在每个区间内选择"距离训练集最远"的那一帧。
        # 优势: 1. 绝对的均匀覆盖 (解决了扎堆问题，如49,50,51)
        #       2. 选出局部最难帧 (解决了均匀采样可能选到简单帧的问题)
        
        indices_to_process = []
        
        # 动态计算分段大小
        segment_size = total_frames / TARGET_FIX_COUNT
        print(f"[Info] 采用分层最大距离采样: 将 {total_frames} 帧分为 {TARGET_FIX_COUNT} 段，每段约 {segment_size:.1f} 帧。")
        
        for i in range(TARGET_FIX_COUNT):
            # 定义当前分段的起始和结束索引 (时间轴上的连续片段)
            start_idx = int(i * segment_size)
            end_idx = int((i + 1) * segment_size)
            
            # 最后一个分段确保覆盖到结尾
            if i == TARGET_FIX_COUNT - 1:
                end_idx = total_frames
                
            # 获取该段内的所有帧的最短距离
            segment_dists = frame_min_dists[start_idx:end_idx]
            
            if len(segment_dists) == 0:
                continue
                
            # 找到该段内距离最大的帧的局部索引
            local_max_idx = np.argmax(segment_dists)
            global_idx = start_idx + local_max_idx
            
            # 这里我们不直接过滤 < 0.05 的帧，因为如果整段都不满足，我们依然要选一个最好的
            # 否则就会导致采样的帧数不足 20 帧，影响最终视频的连贯性修复
            indices_to_process.append(global_idx)

        # 重新排序回时间轴顺序
        indices_to_process = sorted([int(x) for x in indices_to_process])
        print(f"[Info] 分层采样完成: 已选 {len(indices_to_process)} 帧。")
        
        # Filter掉太近的 (Double check, although peaks should handle it)
        # 如果所有峰值都很小(甚至<0.01)，说明轨迹本身就贴着训练集，那还是得修
        
        for idx in tqdm(indices_to_process, desc=f"Matching pairs ({len(indices_to_process)} frames)"):
            render_cam = viewpoint_stack[idx]
            
            # 渲染图路径 (Input)
            render_img_name = f"{idx:05d}.png"
            render_img_path = os.path.join(image_folder, render_img_name)
            
            if not os.path.exists(render_img_path):
                continue
                
            # 计算当前渲染相机的几何信息
            r_R_wc = render_cam.R.T
            r_center = -np.dot(r_R_wc, render_cam.T)
            r_view_dir = r_R_wc[:, 2]
            
            best_cam = None
            min_dist = 1e9
            
            # 策略: 
            # 1. 过滤掉角度差异过大 (>60度, cos_sim < 0.5) 的
            # 2. 在剩下的里面找距离最近的 Top-K
            # [Modify] 回退到 1-对-1 模式 (User feedback: 1-to-many is problematic)
            TOP_K = 1
            
            # Step 1: Filter by angle
            candidates = []
            for t_info in train_cam_infos_cache:
                # Dot product of unit vectors
                cos_sim = np.dot(r_view_dir, t_info['view_dir'])
                if cos_sim > 0.5: # > 60 degree View Cone
                    candidates.append(t_info)
            
            # Filtered candidates
            valid_candidates = candidates if candidates else train_cam_infos_cache
            
            # Step 2: Sort by distance
            # x['center'] is world position
            valid_candidates.sort(key=lambda x: np.linalg.norm(r_center - x['center']))
            
            # Select Top K
            selected_cams = [x['cam'] for x in valid_candidates[:TOP_K]]
            
            # 创建可以保存多张参考图的子目录结构
            if selected_cams:
                # 创建子目录
                sub_dir = os.path.join(fix_pairs_dir, f"{idx:05d}")
                os.makedirs(sub_dir, exist_ok=True)
                
                # 创建软链接: Input (渲染图)
                target_input = os.path.join(sub_dir, "input.png")
                if os.path.exists(target_input):
                    os.remove(target_input)
                os.symlink(os.path.abspath(render_img_path), target_input)
                
                # 创建软链接: Reference (Single Best)
                # 只保存一个 ref.jpg，方便下游处理
                cam = selected_cams[0]
                if cam.uid in train_cam_paths:
                    src_ref = train_cam_paths[cam.uid]
                    ext = os.path.splitext(src_ref)[1]
                    # 统一命名为 ref.ext
                    target_ref = os.path.join(sub_dir, f"ref{ext}") 
                    if os.path.exists(target_ref):
                        os.remove(target_ref)
                    os.symlink(os.path.abspath(src_ref), target_ref)
        
        print(f"[Info] 样本对已生成至: {fix_pairs_dir} (Best-1 Reference)")



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")    
    parser.add_argument("--optim_test_pose_iter", default=500, type=int)
    parser.add_argument("--infer_video", action="store_true")
    parser.add_argument("--test_fps", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    render_sets(model.extract(args), args.iterations, pipeline.extract(args), args.skip_train, args.skip_test, args)
