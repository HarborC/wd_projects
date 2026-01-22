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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import scipy
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform

def generate_interpolated_path(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth path between input keyframe camera poses using Piecewise Linear (Translation) 
    and Slerp (Rotation) interpolation.

  Args:
    poses: (n, 3, 4) array of input pose keyframes (camera-to-world format).
    n_interp: number of interpolated frames per segment (between two keyframes).
              Wait, checking legacy code: "returned path will have n_interp * (n - 1) total poses".
              The logic in render.py calculates n_interp based on total time divided by segments.
              So yes, n_interp is frames per segment.
    spline_degree: (Ignored)
    smoothness: (Ignored)
    rot_weight: (Ignored)

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """
    poses = np.array(poses)
    N = poses.shape[0]
    
    # Extract Translation and Rotation
    trans = poses[:, :3, 3] # (N, 3)
    rots = poses[:, :3, :3] # (N, 3, 3)
    
    # Convert to quaternions for interpolation
    quats = Rotation.from_matrix(rots).as_quat() # (N, 4) (x, y, z, w) format in scipy
    
    all_interp_poses = []
    
    def slerp(q0, q1, t):
        """Simple Spherical Linear Interpolation"""
        dot = np.sum(q0 * q1)
        
        # Ensure shortest path
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        
        DOT_THRESHOLD = 0.9995
        if dot > DOT_THRESHOLD:
            # Linear interpolation for very close orientations
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        
        theta_0 = np.arccos(min(dot, 1.0)) # Clamp to avoid numerical errors
        sin_theta_0 = np.sin(theta_0)
        
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q0 + s1 * q1

    # Loop through each segment between keyframes
    for i in range(N - 1):
        t0 = trans[i]
        t1 = trans[i+1]
        
        q0 = quats[i]
        q1 = quats[i+1]
        
        # Generate interpolated frames for this segment
        # n_interp frames. Usually we want to include start but exclude end to avoid duplicates,
        # or logic in render.py expects n_interp * (n-1).
        # render.py logic:
        # n_interp = int(TotalFrames / (N-1))
        # all_inter_pose = ... returns (n_interp * (N-1)) frames
        # It then appends the LAST frame manually.
        # So for each segment we need exactly n_interp frames.
        
        for j in range(n_interp):
            alpha = j / n_interp 
            # If j goes from 0 to n_interp-1, alpha goes from 0 to <1.  
            # This matches "endpoint=False" logic typically.
            
            # Linear Interpolation for Translation
            t_curr = (1.0 - alpha) * t0 + alpha * t1
            
            # Slerp for Rotation
            q_curr = slerp(q0, q1, alpha)
            
            # Reconstruct Rotation Matrix
            R_curr = Rotation.from_quat(q_curr).as_matrix()
            
            # Construct Pose Matrix
            pose_new = np.zeros((3, 4))
            pose_new[:3, :3] = R_curr
            pose_new[:3, 3] = t_curr
            
            all_interp_poses.append(pose_new)
            
    return np.array(all_interp_poses) 

def generate_interpolated_intrinsics(values, n_interp):
    """Interpolate scalar values (like FOV or focal length).
    Args:
        values: list or array of scalar values from keyframes.
        n_interp: number of interpolated frames per segment.
    Returns:
        Array of interpolated values.
    """
    values = np.array(values)
    N = len(values)
    all_interp = []
    
    for i in range(N - 1):
        v0 = values[i]
        v1 = values[i+1]
        
        for j in range(n_interp):
            alpha = j / n_interp
            v_curr = (1.0 - alpha) * v0 + alpha * v1
            all_interp.append(v_curr)
            
    return np.array(all_interp)


def viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def visualizer(camera_poses, colors, save_path="/mnt/data/1.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for pose, color in zip(camera_poses, colors):
        rotation = pose[:3, :3]
        translation = pose[:3, 3]  # Corrected to use 3D translation component
        camera_positions = np.einsum(
            "...ij,...j->...i", np.linalg.inv(rotation), -translation
        )

        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Poses")

    plt.savefig(save_path)
    plt.close()

    return save_path