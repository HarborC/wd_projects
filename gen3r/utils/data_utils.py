from typing import Tuple
from einops import rearrange

import torch


def center_crop(frames: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Center crops a tensor of frames.
    """
    height, width = frames.shape[2:]
    crop_height, crop_width = size
    top = max(0, (height - crop_height) // 2)
    down = min(height, (height + crop_height) // 2)
    left = max(0, (width - crop_width) // 2)
    right = min(width, (width + crop_width) // 2)
    return frames[:, :, top:down, left:right]


@torch.no_grad()
def compute_rays(
    c2w: torch.Tensor,
    K: torch.Tensor,
    h: int,
    w: int,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        c2w (torch.tensor): [v, 4, 4]
        K (torch.tensor): [v, 3, 3]
        h (int): height of the image
        w (int): width of the image
        device (str): device to compute on
    Returns:
        ray_o (torch.tensor): [v, 3, h, w]
        ray_d (torch.tensor): [v, 3, h, w]
    """

    if c2w.ndim == 4:
        c2w = c2w.squeeze(0)
    if K.ndim == 4:
        K = K.squeeze(0)

    v = c2w.shape[0]

    fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]  # [v]
    fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1).to(device)  # [v, 4]

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    x, y = x.to(device), y.to(device)
    x = x[None, :, :].expand(v, -1, -1).reshape(v, -1)
    y = y[None, :, :].expand(v, -1, -1).reshape(v, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / (fxfycxcy[:, 0:1] + 1e-8)
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / (fxfycxcy[:, 1:2] + 1e-8)
    z = torch.ones_like(x).to(device)
    ray_d = torch.stack([x, y, z], dim=2).to(c2w.dtype)  # [v, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [v, h*w, 3]
    ray_d = ray_d / (torch.norm(ray_d, dim=2, keepdim=True) + 1e-8)  # normalize in camera space
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [v, h*w, 3]

    ray_o = rearrange(ray_o, "v (h w) c -> v c h w", v=v, h=h, w=w, c=3)
    ray_d = rearrange(ray_d, "v (h w) c -> v c h w", v=v, h=h, w=w, c=3)

    return ray_o, ray_d


def preprocess_poses(
    abs_c2ws: torch.Tensor,  # [v, 4, 4]
) -> torch.Tensor:
    """
    Preprocess the poses to:
    1. translate and rotate the scene so that the first frame is the identity
    """
    abs_w2cs = torch.linalg.inv(abs_c2ws)
    cam_to_origin = 0
    target_cam_c2w = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).to(device=abs_w2cs.device, dtype=abs_w2cs.dtype)
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    rel_c2ws = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    rel_c2ws = torch.stack(rel_c2ws, dim=0)

    return rel_c2ws
