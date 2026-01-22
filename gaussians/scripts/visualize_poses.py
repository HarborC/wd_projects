import time
import argparse
import numpy as np
from pathlib import Path
import viser
from scipy.spatial.transform import Rotation
try:
    from plyfile import PlyData
except ImportError:
    print("Warning: plyfile not found. Point cloud visualization will be disabled.")
    PlyData = None

def main():
    parser = argparse.ArgumentParser(description="Visualize camera poses using Viser")
    parser.add_argument("--model_path", "-m", required=True, type=Path, help="Path to the model directory")
    parser.add_argument("--iteration", "-i", default=1000, type=int, help="Iteration number to visualize")
    parser.add_argument("--port", "-p", default=8080, type=int, help="Viser server port")
    parser.add_argument("--downsample", "-d", default=5, type=int, help="Point cloud downsample factor (default: 5)")
    args = parser.parse_args()

    # 构建文件路径
    pose_dir = args.model_path / "pose" / f"ours_{args.iteration}"
    opt_pose_path = pose_dir / "pose_optimized.npy"
    interp_pose_path = pose_dir / "pose_interpolated.npy"
    
    # 尝试查找点云文件
    ply_path = args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    
    if not pose_dir.exists():
        print(f"Error: Pose directory {pose_dir} does not exist.")
        return

    # 启动 Viser Server
    server = viser.ViserServer(port=args.port)
    print(f"Viser server started at http://localhost:{args.port}")

    # ========================================================
    # 0. 可视化 3DGS 点云 (Point Cloud)
    # ========================================================
    if PlyData is not None and ply_path.exists():
        print(f"Loading 3DGS point cloud from {ply_path} ...")
        try:
            plydata = PlyData.read(ply_path)
            
            # 读取位置
            xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])),  axis=1)
            
            # 读取颜色 (SH DC component)
            f_dc = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
                             np.asarray(plydata.elements[0]["f_dc_1"]),
                             np.asarray(plydata.elements[0]["f_dc_2"])),  axis=1)
            
            # SH -> RGB 转换 (C0 * f_dc + 0.5)
            SH_C0 = 0.28209479177387814
            rgb = f_dc * SH_C0 + 0.5
            rgb = np.clip(rgb, 0, 1) # 限制在 0-1 之间
            
            # 下采样以提高性能
            if args.downsample > 1:
                xyz = xyz[::args.downsample]
                rgb = rgb[::args.downsample]
                print(f"Downsampled point cloud by {args.downsample}x. Points: {len(xyz)}")
            
            # 添加点云到场景
            server.scene.add_point_cloud(
                "/3dgs_point_cloud",
                points=xyz,
                colors=rgb,
                point_size=0.03, # 可以根据场景尺度调整
            )
            print("Added 3DGS point cloud.")
            
        except Exception as e:
            print(f"Failed to load point cloud: {e}")
    elif not ply_path.exists():
        print(f"Warning: Point cloud file not found at {ply_path}")

    # 辅助函数：将旋转矩阵转换为 wxyz 四元数
    def matrix_to_wxyz(R):
        quat = Rotation.from_matrix(R).as_quat() # 返回 (x, y, z, w)
        return np.array([quat[3], quat[0], quat[1], quat[2]])

    # 1. 可视化优化后的关键帧 Pose (Green)
    if opt_pose_path.exists():
        print(f"Loading optimized poses from {opt_pose_path}")
        # loaded pose is w2c (world-to-camera)
        w2c_opt = np.load(opt_pose_path) 
        
        # convert to c2w (camera-to-world) for visualization
        c2w_opt = np.linalg.inv(w2c_opt)
        
        for i, pose in enumerate(c2w_opt):
            R = pose[:3, :3]
            t = pose[:3, 3]
            
            server.scene.add_camera_frustum(
                f"/optimized/cam_{i}",
                fov=0.8, 
                scale=0.15,
                aspect=1.5,
                wxyz=matrix_to_wxyz(R),
                position=t,
                color=(0, 255, 0) # Green
            )
            
            # 也可以加个文字标签
            server.scene.add_label(
                f"/optimized/cam_{i}/label",
                text=f"K{i}",
                position=t
            )
            
        print(f"Added {len(c2w_opt)} optimized poses.")
    else:
        print(f"Warning: {opt_pose_path} not found.")

    # 2. 可视化插值后的轨迹 Pose (Blue)
    if interp_pose_path.exists():
        print(f"Loading interpolated poses from {interp_pose_path}")
        w2c_interp = np.load(interp_pose_path)
        c2w_interp = np.linalg.inv(w2c_interp)

        points = []
        for i, pose in enumerate(c2w_interp):
            R = pose[:3, :3]
            t = pose[:3, 3]
            points.append(t)
            
            # 为了不让画面太乱，每隔几帧画一个 frustum
            if i % 5 == 0:
                server.scene.add_camera_frustum(
                    f"/interpolated/cam_{i}",
                    fov=0.8,
                    scale=0.05, # 小一点
                    aspect=1.5,
                    wxyz=matrix_to_wxyz(R),
                    position=t,
                    color=(50, 50, 255) # Blue
                )

        # 3. 画出连续的轨迹线
        if len(points) > 0:
            server.scene.add_spline_catmull_rom(
                "/trajectory",
                positions=np.array(points),
                color=(200, 200, 0), # Yellow line
                line_width=3.0,
                tension=0.5
            )
            print(f"Added trajectory with {len(points)} points.")

    else:
        print(f"Warning: {interp_pose_path} not found.")

    # 保持运行
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    main()
