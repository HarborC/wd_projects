import numpy as np
import cv2
from pathlib import Path
import sys

def main():
    metadata_path = Path("/data/cjg/projects/wd_projects/test_bev_output/scene_metadata.npz")
    bev_image_path = Path("/data/cjg/projects/wd_projects/test_bev_output/bev_images/merged_bev_world_color_corrected.png")
    
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found.")
        return

    print(f"Loading metadata from {metadata_path}...")
    data = np.load(metadata_path)
    
    # 1. Check Keys
    print("\n--- Check 1: Keys ---")
    keys = list(data.keys())
    print(f"Available keys: {keys}")
    required_keys = ['intrinsics', 'extrinsics', 'bev_params', 'depth_paths']
    for k in required_keys:
        if k not in keys:
            print(f"❌ Missing key: {k}")
        else:
            print(f"✅ Found key: {k}")

    # 2. Check Shapes
    print("\n--- Check 2: Shapes ---")
    extrinsics = data['extrinsics']
    intrinsics = data['intrinsics']
    bev_params = data['bev_params']
    
    print(f"Extrinsics shape: {extrinsics.shape} (Expected: N x 4 x 4)")
    print(f"Intrinsics shape: {intrinsics.shape} (Expected: N x 3 x 3)")
    print(f"BEV Params shape: {bev_params.shape} (Expected: 6,)")

    # 3. Visual Verification (Project Trajectory onto BEV)
    print("\n--- Check 3: Visual Verification (Trajectory Projection) ---")
    
    x_min = bev_params[0]
    z_max = bev_params[1]
    res = bev_params[2]
    ground_level = bev_params[3]
    bev_w = int(bev_params[4])
    bev_h = int(bev_params[5])
    
    print(f"BEV Params: x_min={x_min:.2f}, z_max={z_max:.2f}, res={res:.4f}")
    print(f"BEV Size: {bev_w} x {bev_h}")
    
    if bev_image_path.exists():
        bev_img = cv2.imread(str(bev_image_path))
        print(f"Loaded BEV image: {bev_img.shape}")
    else:
        print("Warning: BEV image not found, creating blank canvas.")
        bev_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

    # Calculate Camera Centers in Aligned World Frame
    traj_points = []
    
    for i in range(len(extrinsics)):
        w2c = extrinsics[i]
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        
        # Camera center in world frame: C = -R^T * t
        C = -R.T @ t
        
        # Project to BEV (Using the formulas we discussed)
        # u = (x - x_min) / res
        # v = (z_max - z) / res
        u = (C[0] - x_min) / res
        v = (z_max - C[2]) / res
        
        pt = (int(u), int(v))
        traj_points.append(pt)
        
        # Draw camera point (Red)
        cv2.circle(bev_img, pt, 5, (0, 0, 255), -1) 
        
        # Mark Start (Green) and End (Blue)
        if i == 0:
             cv2.circle(bev_img, pt, 8, (0, 255, 0), -1)
        if i == len(extrinsics) - 1:
             cv2.circle(bev_img, pt, 8, (255, 0, 0), -1)

    # Draw connection lines
    if len(traj_points) > 1:
        cv2.polylines(bev_img, [np.array(traj_points)], False, (0, 255, 255), 2)

    output_viz = "test_bev_output/verification_traj_viz.png"
    cv2.imwrite(output_viz, bev_img)
    print(f"\n✅ Verification image saved to: {output_viz}")
    print("Please check this image. You should see the camera trajectory (Red dots/Yellow line) overlaying the map.")
    print("Green dot = Start, Blue dot = End.")

if __name__ == "__main__":
    main()
