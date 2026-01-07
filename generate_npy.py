import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as scipyR
from scipy.spatial.transform import Slerp


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion [x, y, z, w] (scipy convention)."""
    # scipy returns [x, y, z, w], scalar last
    R_np = R.detach().cpu().numpy()
    if R_np.ndim == 2:
        q = scipyR.from_matrix(R_np).as_quat()
    else:
        q = np.array([scipyR.from_matrix(r).as_quat() for r in R_np])
    return torch.from_numpy(q).to(device=R.device, dtype=R.dtype)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [x, y, z, w] to rotation matrix."""
    q_np = q.detach().cpu().numpy()
    if q_np.ndim == 1:
        R = scipyR.from_quat(q_np).as_matrix()
    else:
        R = np.array([scipyR.from_quat(qi).as_matrix() for qi in q_np])
    return torch.from_numpy(R).to(device=q.device, dtype=q.dtype)


def slerp_quaternions(q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between quaternions using scipy."""
    q1_np = q1.detach().cpu().numpy()
    q2_np = q2.detach().cpu().numpy()

    key_rots = scipyR.from_quat([q1_np, q2_np])
    slerp = Slerp([0, 1], key_rots)
    q_interp = slerp([t]).as_quat()[0]

    return torch.from_numpy(q_interp).to(device=q1.device, dtype=q1.dtype)


def interpolate_trajectory(
    w2cs_key: torch.Tensor,
    K_key: torch.Tensor,
    interp_per_pair: int = 20,
    device: torch.device = torch.device('cpu')
):
    """Interpolate camera trajectory in c2w (camera-to-world) space."""
    n_key = w2cs_key.shape[0]

    if n_key == 1:
        return w2cs_key, K_key

    # Convert w2c to c2w for interpolation
    c2ws_key = torch.linalg.inv(w2cs_key)

    w2cs_list, Ks_list = [], []

    for i in range(n_key - 1):
        # Add current key frame (w2c format)
        w2cs_list.append(w2cs_key[i:i + 1])
        Ks_list.append(K_key[i:i + 1])

        # Get c2w poses for interpolation
        c2w_0, c2w_1 = c2ws_key[i], c2ws_key[i + 1]
        R0, t0 = c2w_0[:3, :3], c2w_0[:3, 3]
        R1, t1 = c2w_1[:3, :3], c2w_1[:3, 3]

        # Convert to quaternions and interpolate
        q0 = rotation_matrix_to_quaternion(R0)
        q1 = rotation_matrix_to_quaternion(R1)

        for j in range(1, interp_per_pair + 1):
            alpha = j / (interp_per_pair + 1)

            # Interpolate in c2w space
            t_interp = (1 - alpha) * t0 + alpha * t1
            q_interp = slerp_quaternions(q0, q1, alpha)
            R_interp = quaternion_to_rotation_matrix(q_interp)

            # Build interpolated c2w matrix
            c2w_interp = torch.eye(4, device=device, dtype=c2ws_key.dtype)
            c2w_interp[:3, :3] = R_interp
            c2w_interp[:3, 3] = t_interp

            # Convert back to w2c for storage
            w2c_interp = torch.linalg.inv(c2w_interp)

            # Interpolate intrinsics
            K_interp = K_key[i]

            w2cs_list.append(w2c_interp.unsqueeze(0))
            Ks_list.append(K_interp.unsqueeze(0))

    # Add last key frame
    w2cs_list.append(w2cs_key[-1:])
    Ks_list.append(K_key[-1:])

    w2cs_all = torch.cat(w2cs_list, dim=0)
    Ks_all = torch.cat(Ks_list, dim=0)

    return w2cs_all, Ks_all


def load_data(output_dir: str, device: torch.device):
    """
    Load reconstruction data from HunyuanWorld-Mirror output directory.

    Expected structure:
    - output_dir/images/ : key frame images (original filenames)
    - output_dir/depths/ : depth maps (.npy files, matching image resolutions)
    - output_dir/sparse/0/ : COLMAP reconstruction
    """
    import pycolmap

    output_path = Path(output_dir)

    # Find images
    images_dir = output_path / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = sorted(glob.glob(str(images_dir / "*.png")) +
                         glob.glob(str(images_dir / "*.jpg")),
                         key=lambda x: Path(x).stem)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    S = len(image_paths)
    print(f"Found {S} images in {images_dir}")

    # Load COLMAP data
    sparse_dir = output_path / "sparse" / "0"
    if not sparse_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse directory not found: {sparse_dir}")

    reconstruction = pycolmap.Reconstruction(str(sparse_dir))

    w2cs_key, Ks_key = [], []

    for img_id in sorted(reconstruction.images.keys()):
        img = reconstruction.images[img_id]
        cam = reconstruction.cameras[img.camera_id]

        # Get w2c pose from pycolmap
        rigid3d = img.cam_from_world()
        w2c = np.eye(4)
        w2c[:3, :3] = rigid3d.rotation.matrix()
        w2c[:3, 3] = rigid3d.translation
        w2cs_key.append(w2c)

        # Intrinsics from cam.params
        model_name = str(cam.model)
        if "SIMPLE_PINHOLE" in model_name or cam.model == 0:
            f, cx, cy = cam.params
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
        elif "PINHOLE" in model_name or cam.model == 1:
            fx, fy, cx, cy = cam.params
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported camera model: {cam.model}")
        Ks_key.append(K)

    w2cs_key = np.stack(w2cs_key, axis=0)
    Ks_key = np.stack(Ks_key, axis=0)

    # Load images, depths, masks
    images_key, depth_key, mask_key = [], [], []
    depths_dir = output_path / "depths"

    for img_path in image_paths:
        stem = Path(img_path).stem
        depth_path = depths_dir / f"{stem}.npy"

        if not depth_path.exists():
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        # Load image
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]

        # Load depth
        depth = np.load(depth_path)
        if depth.shape[:2] != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        # Convert image to [-1, 1]
        img = img.astype(np.float32) / 255.0 * 2 - 1

        images_key.append(img.transpose(2, 0, 1))
        depth_key.append(depth[np.newaxis, ...])
        mask_key.append((depth > 0).astype(np.float32)[np.newaxis, ...])

    images_key = np.stack(images_key, axis=0)
    depth_key = np.stack(depth_key, axis=0)
    mask_key = np.stack(mask_key, axis=0)

    # Convert to torch
    return {
        "images_key": torch.from_numpy(images_key).to(device),
        "depth_key": torch.from_numpy(depth_key).to(device),
        "mask_key": torch.from_numpy(mask_key).to(device),
        "K_key": torch.from_numpy(Ks_key).to(device),
        "w2c_key": torch.from_numpy(w2cs_key).to(device),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate integrated NPY file from HunyuanWorld-Mirror reconstruction outputs"
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to HunyuanWorld-Mirror output directory")
    parser.add_argument("--output_path", type=str, default="output.npz",
                        help="Output NPY file path")
    parser.add_argument("--interp_per_pair", type=int, default=20,
                        help="Number of interpolated frames between key frames")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading from: {args.input_path}")
    data = load_data(args.input_path, device)

    images_key = data["images_key"]
    depth_key = data["depth_key"]
    mask_key = data["mask_key"]
    K_key = data["K_key"]
    w2c_key = data["w2c_key"]

    N = images_key.shape[0]
    print(f"Loaded {N} key frames")
    print(f"  Images: {images_key.shape} {images_key.dtype}")
    print(f"  Depth: {depth_key.shape} {depth_key.dtype}")
    print(f"  Mask: {mask_key.shape} {mask_key.dtype}")
    print(f"  K: {K_key.shape} {K_key.dtype}")
    print(f"  w2c: {w2c_key.shape} {w2c_key.dtype}")

    print(f"\nInterpolating trajectory ({args.interp_per_pair} frames per pair)...")
    w2cs_all, Ks_all = interpolate_trajectory(
        w2c_key, K_key, interp_per_pair=args.interp_per_pair, device=device
    )
    print(f"  Total frames: {w2cs_all.shape[0]}")

    output_dict = {
        "images_key_frames": images_key.cpu().numpy(),
        "depth_key_frames": depth_key.cpu().numpy(),
        "mask_key_frames": mask_key.cpu().numpy(),
        "K_key_frames": K_key.cpu().numpy(),
        "w2cs_key_frames": w2c_key.cpu().numpy().astype(np.float32),
        "w2cs_all": w2cs_all.cpu().numpy().astype(np.float32),
        "Ks_all": Ks_all.cpu().numpy(),
    }

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_path, **output_dict)

    print(f"\nSaved to: {args.output_path}")
    for k, v in output_dict.items():
        print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()
