"""Gen3R inference module."""
import os
import sys
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from einops import rearrange
from torchvision.transforms.functional import resize
from accelerate.utils import set_seed

# Import from migrated modules
from gen3r.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from gen3r.utils.data_utils import center_crop, compute_rays, preprocess_poses
from gen3r.utils.common_utils import colorize_depth_map, downsample_and_save_pointcloud, save_videos_grid

# Lazy import of Gen3R pipeline to avoid dependency issues during testing
_Gen3RPipeline = None

def _get_gen3r_pipeline():
    """Lazy import of Gen3RPipeline from test_code."""
    global _Gen3RPipeline
    if _Gen3RPipeline is not None:
        return _Gen3RPipeline
    
    import importlib.util
    TEST_CODE_GEN3R_PIPELINE = "/data/cjg/projects/wd_projects/test_code/Gen3R/gen3r/pipeline/pipeline_gen3r.py"
    
    spec = importlib.util.spec_from_file_location("gen3r_pipeline", TEST_CODE_GEN3R_PIPELINE)
    gen3r_pipeline_module = importlib.util.module_from_spec(spec)
    sys.modules['gen3r_pipeline_temp'] = gen3r_pipeline_module
    spec.loader.exec_module(gen3r_pipeline_module)
    _Gen3RPipeline = gen3r_pipeline_module.Gen3RPipeline
    return _Gen3RPipeline


def get_poses(cam_type='', num_frames=49, scene_scale=1.0):
    """Generate camera poses for built-in trajectories."""
    import math

    F = int(num_frames)
    if F <= 0:
        raise ValueError("num_frames must be > 0")

    poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(F, 1, 1)

    def rot_y(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32)

    def rot_x(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float32)

    if cam_type == 'zoom_in' or cam_type == 'zoom_out':
        end = 1.0 * float(scene_scale)
        if cam_type == 'zoom_out':
            end = -end
        zs = torch.linspace(0.0, end, steps=F, dtype=torch.float32)
        for i in range(F):
            poses[i, 2, 3] = zs[i]

    elif cam_type == 'arc_right' or cam_type == 'arc_left':
        end_angle = math.radians(15)
        if cam_type == 'arc_left':
            end_angle = -end_angle
        angles = torch.linspace(0.0, end_angle, steps=F, dtype=torch.float32)
        for i in range(F):
            R = rot_y(float(angles[i]))
            poses[i, :3, :3] = R

    elif cam_type == 'translate_up' or cam_type == 'translate_down':
        end_angle = math.radians(15.0)
        if cam_type == 'translate_down':
            end_angle = -end_angle
        angles = torch.linspace(0.0, end_angle, steps=F, dtype=torch.float32)
        for i in range(F):
            R = rot_x(float(angles[i]))
            poses[i, :3, :3] = R

    elif cam_type == 'free':
        pass
    else:
        raise NotImplementedError(f"Camera type {cam_type} not implemented.")

    return poses


def save_results(output_dir, results, prompts):
    """Save Gen3R results."""
    import json
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    for key, value in results.items():
        if key == "rgbs" and value is not None:
            rgb = rearrange(value, "b f h w c -> b c f h w").float().cpu()
            filename = os.path.join(output_dir, f"rgb.mp4")
            save_videos_grid(rgb, filename, rescale=False)

        elif key == "pcds":
            downsample_and_save_pointcloud(
                os.path.join(output_dir, f"pcds.ply"),
                value[0],
                results['rgbs'][0],
                voxel_size=0.005,
                filter_outliers=True,
                depth_percentile=0,
                remove_far_points=False,
            )

        elif key == 'cameras':
            filename = os.path.join(output_dir, f"cameras.json")
            extrinsics, Ks = value
            extrinsics = torch.cat(
                [extrinsic,
                 torch.tensor([0, 0, 0, 1], device=extrinsics.device).view(1, 1, 1, 4).repeat(
                     extrinsics.shape[0], extrinsics.shape[1], 1, 1)], dim=2)

            cameras = {
                'extrinsics': extrinsics[0].float().cpu().numpy().tolist(),
                'intrinsics': Ks[0].float().cpu().numpy().tolist(),
            }
            with open(filename, "w") as f:
                json.dump(cameras, f, indent=4)

    # Save prompts
    filename = os.path.join(output_dir, f"prompts.txt")
    with open(filename, "w") as f:
        f.write(prompts[0])


def run_gen3r_inference(frame_path, prompts, output_dir, task='allview',
                        cameras='free', pretrained_model_name_or_path='./checkpoints',
                        device='cuda'):
    """
    Run Gen3R inference.

    Args:
        frame_path: Path to input images or video
        prompts: Text prompt
        output_dir: Output directory
        task: Task type ('1view', '2view', 'allview')
        cameras: Camera trajectory or JSON file path
        pretrained_model_name_or_path: Path to model checkpoints
        device: Device to run on

    Returns:
        dict with paths to output files
    """
    import imageio
    import numpy as np
    from datetime import datetime

    BUILTIN_CAMERA_TRAJECTORIES = ["zoom_in", "zoom_out", "arc_left", "arc_right",
                                    "translate_up", "translate_down", "free"]

    # Create output directory
    output_dir = os.path.join(output_dir, f'{task}', f'{datetime.now().strftime("%Y-%m-%d-%H-%M-00")}')
    os.makedirs(output_dir, exist_ok=True)

    set_seed(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load pipeline (lazy import)
    Gen3RPipeline = _get_gen3r_pipeline()
    pipeline = Gen3RPipeline.from_pretrained(pretrained_model_name_or_path)
    pipeline.to(device).to(torch.bfloat16)

    # Load control images
    control_images = []
    if not task == 'allview' or (task == 'allview' and os.path.isdir(frame_path)):
        if task == 'allview':
            frame_path = sorted([os.path.join(frame_path, f) for f in os.listdir(frame_path)
                                if f.endswith('.png') or f.endswith('.jpg')])
        for fp in frame_path if isinstance(frame_path, list) else [frame_path]:
            frame = torch.from_numpy(imageio.v2.imread(fp))[..., :3]
            control_images.append(frame)
        control_images = torch.stack(control_images).to(device, torch.bfloat16)
    else:
        control_images = torch.from_numpy(imageio.v3.imread(frame_path))

    if task == '1view':
        control_images = control_images[:1, ...]
    elif task == '2view':
        control_images = control_images[[0, -1], ...]
    else:
        control_images = control_images[:49, ...]

    control_images = control_images.permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
    frame_height, frame_width = control_images.shape[3], control_images.shape[4]
    scale = 560 / min(frame_height, frame_width)
    new_height, new_width = round(frame_height * scale), round(frame_width * scale)
    control_images = resize(control_images[0], [new_height, new_width])
    control_images = center_crop(control_images, (560, 560))[None, ...].to(device, torch.bfloat16)

    # Handle cameras
    if cameras in BUILTIN_CAMERA_TRAJECTORIES:
        print("Using built-in camera trajectory:", cameras)
        with torch.no_grad():
            aggregated_token_list, ps_idx = pipeline.vggt.aggregator(control_images)
            aggregated_token_list = [aggregated_token_list[i]
                                     for i in pipeline.vggt.depth_head.intermediate_layer_idx]
            pose_enc = pipeline.vggt.camera_head(aggregated_token_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, control_images.shape[-2:])
            depth_maps, _ = pipeline.vggt.depth_head(aggregated_token_list, control_images, ps_idx)

        Ks = intrinsic[:, :1].repeat(1, 49, 1, 1)
        scene_scale = 0.8 * torch.median(depth_maps).item()
        c2ws = get_poses(cam_type=cameras, num_frames=49, scene_scale=scene_scale).to(device)
    else:
        import json
        with open(cameras, "r") as f:
            cameras_data = json.load(f)
        extrinsics = torch.from_numpy(np.array(cameras_data['extrinsics']))[:49].to(device)
        Ks = torch.from_numpy(np.array(cameras_data['intrinsics']))[None, :49].to(device)
        c2ws = torch.linalg.inv(extrinsics)

    c2ws = preprocess_poses(c2ws)[None, ...]

    # Compute plucker embeddings
    plucker_embeddings_list = []
    for i in range(len(c2ws)):
        rays_o, rays_d = compute_rays(c2ws[i], Ks[i], h=560, w=560, device=device)
        o_cross_d = torch.cross(rays_o, rays_d, dim=1)
        plucker_embeddings = torch.cat([o_cross_d, rays_d], dim=1)
        plucker_embeddings_list.append(plucker_embeddings)
    plucker_embeddings = torch.stack(plucker_embeddings_list, dim=0)
    if cameras == 'free' or task == 'allview':
        plucker_embeddings = torch.zeros_like(plucker_embeddings)

    # Load prompts
    if os.path.isfile(prompts):
        with open(prompts, "r", encoding="utf-8") as file:
            prompts = file.readlines()[0]
    else:
        prompts = prompts

    # Run inference
    sample = pipeline(
        prompt=prompts,
        control_cameras=plucker_embeddings,
        control_images=control_images,
        num_frames=49,
        negative_prompt="bad detailed",
        height=560,
        width=560,
        guidance_scale=5,
        return_dict=True,
        min_max_depth_mask=True,
    )

    artifacts = {
        'rgbs': sample.rgbs,
        'depth_maps': sample.depth_maps,
        "pcds": sample.pcds,
        'point_masks': sample.point_masks,
        'cameras': sample.cameras,
    }

    save_results(output_dir, artifacts, [prompts])

    return {
        "rgb_video": os.path.join(output_dir, "rgb.mp4"),
        "pointcloud": os.path.join(output_dir, "pcds.ply"),
        "cameras": os.path.join(output_dir, "cameras.json"),
        "output_dir": output_dir,
    }
