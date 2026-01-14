"""
Utility functions for Gaussian Splatting.
"""

import numpy as np
import torch


def save_time(path, key, value):
    """Save timing information to JSON file."""
    import json
    from pathlib import Path

    path = Path(path)
    time_file = path / 'time.json'

    # Load existing data if file exists
    if time_file.exists():
        with open(time_file, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    data[key] = value

    with open(time_file, 'w') as f:
        json.dump(data, f, indent=2)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1, max_steps=10000
):
    """
    Copied from In3D: https://github.com/apple/ARKitScalableMoI/blob/main/in3d/utils/general_utils.py
    """

    def helper(step):
        if step < 0:
            # return warmup_lr_init
            return lr_init
        if lr_delay_steps > 0:
            # a warmup
            if step < lr_delay_steps:
                # lr_delay_steps is the number of delay steps
                # step is the current step
                # so the warmup lr is linearly increasing from lr_init * lr_delay_mult to lr_init
                # but in the first lr_delay_steps, the lr is lr_init * lr_delay_mult
                # wait, the original code is:
                # delay_rate = lr_delay_mult + (1 - lr_delay_mult) * step / lr_delay_steps
                # return lr_init * delay_rate
                # so the warmup lr is linearly increasing from lr_init * lr_delay_mult to lr_init
                # but lr_delay_mult is usually 0.01, so the warmup lr is increasing from 0.01 * lr_init to lr_init
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * step / lr_delay_steps
                return lr_init * delay_rate
        if step < 0:
            return lr_init
        t = np.clip(step / max_steps, 0, 1)
        log_lrs = np.log(lr_init) * (1 - t) + np.log(lr_final) * t
        return np.exp(log_lrs)

    return helper


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


class BasicPointCloud:
    """Basic point cloud class."""

    def __init__(self, points=None, colors=None, normals=None):
        self.points = points if points is not None else np.empty((0, 3))
        self.colors = colors if colors is not None else np.empty((0, 3))
        self.normals = normals if normals is not None else np.empty((0, 3))


def focal2fov(focal, pixels):
    """Convert focal length to field of view."""
    return 2 * np.arctan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    """Convert field of view to focal length."""
    return pixels / (2 * np.tan(fov / 2))


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """Get world to view matrix."""
    Rt = np.eye(4)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    Rt[:3, 3] = -(R.transpose() @ (scale * cam_center + translate))
    return Rt


def getWorld2View2_torch(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    """Get world to view matrix (PyTorch version)."""
    Rt = torch.eye(4).cuda()
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    Rt[:3, 3] = -(R.T @ (scale * cam_center + translate))
    return Rt


def rotation2quad(R):
    """Convert rotation matrix to quaternion."""
    from scipy.spatial.transform import Rotation as R_scipy
    return R_scipy.from_matrix(R.cpu().numpy()).as_quat()


def quadmultiply(q1, q2):
    """Multiply two quaternions."""
    q1_x, q1_y, q1_z, q1_w = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    q2_x, q2_y, q2_z, q2_w = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = q1_w * q2_w - q1_x * q2_x - q1_y * q2_y - q1_z * q2_z
    x = q1_w * q2_x + q1_x * q2_w + q1_y * q2_z - q1_z * q2_y
    y = q1_w * q2_y - q1_x * q2_z + q1_y * q2_w + q1_z * q2_x
    z = q1_w * q2_z + q1_x * q2_y - q1_y * q2_x + q1_z * q2_w

    return torch.stack([x, y, z, w], dim=-1)


def get_tensor_from_camera(camera):
    """
    Convert camera matrix to tensor representation (quaternion + translation).
    camera: 4x4 matrix (torch.Tensor or numpy array)
    Returns: 7D tensor (quaternion_wxyz + translation_xyz)
    """
    from scipy.spatial.transform import Rotation as R_scipy

    # Determine device
    if isinstance(camera, torch.Tensor):
        device = camera.device
        R = camera[:3, :3].cpu().numpy()
        t = camera[:3, 3].cpu().numpy()
    else:
        device = 'cuda'
        R = camera[:3, :3]
        t = camera[:3, 3]

    # Convert rotation matrix to quaternion (w, x, y, z format)
    quat = R_scipy.from_matrix(R).as_quat()
    quat_tensor = torch.from_numpy(quat).float().to(device)

    # Scipy returns (x, y, z, w), we want (w, x, y, z)
    quat_tensor = torch.cat([quat_tensor[3:4], quat_tensor[:3]])

    # Convert translation to tensor
    t_tensor = torch.from_numpy(t).float().to(device)

    return torch.cat([quat_tensor, t_tensor])


def get_camera_from_tensor(camera_tensor):
    """
    Convert tensor representation to camera matrix.
    camera_tensor: 7D tensor (quaternion_wxyz + translation_xyz)
    Returns: 4x4 camera matrix
    """
    from scipy.spatial.transform import Rotation as R_scipy

    quat = camera_tensor[:4]
    t = camera_tensor[4:7]

    # Convert quaternion (w, x, y, z) to (x, y, z, w) for scipy
    # Use detach() when converting to numpy
    quat_np = torch.cat([quat[1:4], quat[0:1]]).detach().cpu().numpy()
    R = R_scipy.from_quat(quat_np).as_matrix()

    # Build 4x4 matrix
    device = camera_tensor.device
    camera = np.eye(4)
    camera[:3, :3] = R
    camera[:3, 3] = t.detach().cpu().numpy()

    return torch.from_numpy(camera).float().to(device)


def RGB2SH(rgb):
    """Convert RGB to spherical harmonics."""
    return (rgb - 0.5) / 0.28209479177387814


def build_scaling_rotation(s, r):
    """Build scaling and rotation matrix."""
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def build_rotation(r):
    """Build rotation matrix from quaternion."""
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def strip_symmetric(sym):
    """Strip symmetric matrix to upper triangular."""
    return torch.cat((sym[:, :1], sym[:, 1:, 1:]), dim=1)
