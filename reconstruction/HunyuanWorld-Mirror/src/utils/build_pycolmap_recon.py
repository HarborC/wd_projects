import numpy as np
import pycolmap


def _get_pycolmap_version():
    """Get pycolmap version."""
    try:
        import importlib.metadata
        return importlib.metadata.version('pycolmap')
    except:
        return "unknown"


def build_pycolmap_reconstruction2(
    points,
    pixel_coords,
    point_colors,
    poses,
    intrinsics,
    image_size,
    image_names=None,
    shared_camera_model=False,
    camera_model="SIMPLE_PINHOLE",
):
    """
    Convert numpy arrays to pycolmap reconstruction format (for pycolmap >= 3.13.0).
    """
    num_frames = len(poses)
    num_points = len(points)
    
    reconstruction = pycolmap.Reconstruction()
    
    # Add 3D points to scene
    point3d_ids = []
    for pt_idx in range(num_points):
        # add_point3D returns the point3D_id in newer versions
        point3d_id = reconstruction.add_point3D(points[pt_idx], pycolmap.Track(), point_colors[pt_idx])
        point3d_ids.append(point3d_id)
        
    current_camera = None
    current_rig = None
    
    for frame_idx in range(num_frames):
        # Setup camera and rig
        if current_camera is None or not shared_camera_model:
            camera = pycolmap.Camera()
            camera.camera_id = frame_idx + 1
            
            if camera_model == "PINHOLE":
                camera.model = pycolmap.CameraModelId.PINHOLE
                params = np.array([
                    intrinsics[frame_idx][0, 0], intrinsics[frame_idx][1, 1], 
                    intrinsics[frame_idx][0, 2], intrinsics[frame_idx][1, 2]
                ])
            elif camera_model == "SIMPLE_PINHOLE":
                camera.model = pycolmap.CameraModelId.SIMPLE_PINHOLE
                focal_avg = (intrinsics[frame_idx][0, 0] + intrinsics[frame_idx][1, 1]) / 2
                params = np.array([focal_avg, intrinsics[frame_idx][0, 2], intrinsics[frame_idx][1, 2]])
            else:
                raise ValueError(f"Unsupported camera model: {camera_model}")
                
            camera.width = int(image_size[0])
            camera.height = int(image_size[1])
            camera.params = params
            
            reconstruction.add_camera(camera)
            current_camera = camera
            
            # Setup Rig
            rig = pycolmap.Rig()
            rig.rig_id = camera.camera_id
            rig.add_ref_sensor(camera.sensor_id)
            reconstruction.add_rig(rig)
            current_rig = rig
        
        # Setup Pose
        extrinsic = poses[frame_idx]
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(extrinsic[:3, :3]), extrinsic[:3, 3])
        
        # Setup Image
        image = pycolmap.Image()
        image.image_id = frame_idx + 1
        image.camera_id = current_camera.camera_id
        if image_names and frame_idx < len(image_names):
            image.name = image_names[frame_idx]
        else:
            image.name = f"frame_{frame_idx + 1}"
            
        # Setup Frame
        frame = pycolmap.Frame()
        frame.frame_id = image.image_id
        frame.rig_id = current_rig.rig_id
        frame.add_data_id(image.data_id)
        frame.rig_from_world = cam_from_world
        reconstruction.add_frame(frame)
            
        # Process points
        frame_mask = pixel_coords[:, 2].astype(np.int32) == frame_idx
        valid_indices = np.nonzero(frame_mask)[0]
        
        point2d_list = []
        for idx, batch_idx in enumerate(valid_indices):
            point3d_id = point3d_ids[batch_idx]
            xy_coords = pixel_coords[batch_idx][:2]
            
            point2d_list.append(pycolmap.Point2D(xy_coords, point3d_id))
            
            # Update track
            reconstruction.point3D(point3d_id).track.add_element(image.image_id, idx)
            
        image.points2D = pycolmap.Point2DList(point2d_list)
        image.frame_id = frame.frame_id
        reconstruction.add_image(image)
        
    return reconstruction

def _create_camera_params(frame_idx, cam_matrices, model_type, distortion_coeffs=None):
    """Build camera parameter array for different model types."""
    if model_type == "PINHOLE":
        return np.array([
            cam_matrices[frame_idx][0, 0], cam_matrices[frame_idx][1, 1], 
            cam_matrices[frame_idx][0, 2], cam_matrices[frame_idx][1, 2]
        ])
    elif model_type == "SIMPLE_PINHOLE":
        focal_avg = (cam_matrices[frame_idx][0, 0] + cam_matrices[frame_idx][1, 1]) / 2
        return np.array([focal_avg, cam_matrices[frame_idx][0, 2], cam_matrices[frame_idx][1, 2]])
    elif model_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL model not supported")
    else:
        raise ValueError(f"Unsupported camera model: {model_type}")


def _setup_camera_object(frame_idx, cam_matrices, img_dims, model_type, use_shared):
    """Create and configure camera object."""
    if use_shared and frame_idx > 0:
        return None
    
    params = _create_camera_params(frame_idx, cam_matrices, model_type)
    return pycolmap.Camera(
        model=model_type, 
        width=img_dims[0], 
        height=img_dims[1], 
        params=params, 
        camera_id=frame_idx + 1
    )


def _process_frame_points(scene_points, point_coords, frame_idx):
    """Extract and process 2D points belonging to specific frame."""
    frame_mask = point_coords[:, 2].astype(np.int32) == frame_idx
    valid_indices = np.nonzero(frame_mask)[0]
    
    point2d_list = []
    for idx, batch_idx in enumerate(valid_indices):
        point3d_id = batch_idx + 1
        xy_coords = point_coords[batch_idx][:2]
        point2d_list.append(pycolmap.Point2D(xy_coords, point3d_id))
        
        # Update track information
        track = scene_points.points3D[point3d_id].track
        track.add_element(frame_idx + 1, idx)
    
    return point2d_list


def build_pycolmap_reconstruction(
    points,
    pixel_coords,
    point_colors,
    poses,
    intrinsics,
    image_size,
    shared_camera_model=False,
    camera_model="SIMPLE_PINHOLE",
):
    """
    Convert numpy arrays to pycolmap reconstruction format.
    
    Creates 3D scene structure without track optimization.
    Suitable for initialization of neural rendering methods.
    """
    num_frames = len(poses)
    num_points = len(points)
    
    scene = pycolmap.Reconstruction()
    
    # Add 3D points to scene
    for pt_idx in range(num_points):
        scene.add_point3D(points[pt_idx], pycolmap.Track(), point_colors[pt_idx])
    
    current_camera = None
    
    # Process each frame
    for frame_idx in range(num_frames):
        # Setup camera if needed
        if current_camera is None or not shared_camera_model:
            current_camera = _setup_camera_object(
                frame_idx, intrinsics, image_size, camera_model, shared_camera_model
            )
            scene.add_camera(current_camera)
        
        # Create pose transformation
        rotation_matrix = poses[frame_idx][:3, :3]
        translation_vec = poses[frame_idx][:3, 3]
        world_to_cam = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation_matrix), translation_vec)
        
        # Create image object
        frame_image = pycolmap.Image(
            image_id=frame_idx + 1,
            name=f"frame_{frame_idx + 1}",
            camera_id=current_camera.camera_id,
            cam_from_world=world_to_cam
        )
        
        # Process 2D points for this frame
        frame_points = _process_frame_points(scene, pixel_coords, frame_idx)
        
        # Set image points and registration status
        try:
            frame_image.points2D = pycolmap.ListPoint2D(frame_points)
            frame_image.registered = True
        except:
            print(f"Warning: Frame {frame_idx + 1} has no valid points")
            frame_image.registered = False
        
        scene.add_image(frame_image)
    
    return scene