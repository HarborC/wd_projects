"""SEVA Visualizer

Visualize SEVA format outputs (transforms.json with camera-to-world poses).

**Features:**

* SEVA transforms.json file parsing
* Camera frustum visualization with :meth:`viser.SceneApi.add_camera_frustum`
* Train/test frame distinction with different colors
* Interactive camera controls

Usage:
    python seva_vis.py --seva_path test_reconstruction_output2_seva
"""

import json
import random
import time
from pathlib import Path
from typing import List

import imageio.v3 as iio
import numpy as np
import tyro
from tqdm.auto import tqdm

import viser
import viser.transforms as vtf


def main(
    seva_path: Path = Path(__file__).parent / "test_reconstruction_output2_seva",
    downsample_factor: int = 2,
    reorient_scene: bool = True,
) -> None:
    """Visualize SEVA format outputs.

    Args:
        seva_path: Path to the SEVA format directory (contains transforms.json and images/).
        downsample_factor: Downsample factor for the images.
        reorient_scene: Whether to reorient the scene based on average camera direction.
    """
    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Load SEVA info
    transforms_path = seva_path / "transforms.json"
    images_path = seva_path / "images"

    with open(transforms_path, 'r') as f:
        transforms_data = json.load(f)

    # Load train/test split if available
    split_path = None
    for split_file in seva_path.glob("train_test_split_*.json"):
        split_path = split_file
        break

    if split_path:
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        train_ids = set(split_data.get("train_ids", []))
        test_ids = set(split_data.get("test_ids", []))
    else:
        train_ids = set()
        test_ids = set()

    frames_data = transforms_data["frames"]
    intrinsics = {
        "fl_x": transforms_data.get("fl_x", 1.0),
        "fl_y": transforms_data.get("fl_y", 1.0),
        "cx": transforms_data.get("cx", 0.5),
        "cy": transforms_data.get("cy", 0.5),
        "w": transforms_data.get("w", 1.0),
        "h": transforms_data.get("h", 1.0),
    }

    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    # Let's rotate the scene so the average camera direction is pointing up.
    if reorient_scene:
        # Extract rotations from c2w matrices
        # transforms.json stores OpenGL format (already has Y/Z flipped)
        c2w_matrices = []
        for frame in frames_data:
            transform = np.array(frame["transform_matrix"])
            c2w_matrices.append(transform)
        c2w_matrices = np.array(c2w_matrices)

        # Get the "up" direction from cameras (camera's -y in world space)
        average_up = (-c2w_matrices[:, :3, 1]).mean(axis=0)
        average_up /= np.linalg.norm(average_up)
        server.scene.set_up_direction((average_up[0], average_up[1], average_up[2]))

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = vtf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )

    gui_frames = server.gui.add_slider(
        "Max frames",
        min=1,
        max=len(frames_data),
        step=1,
        initial_value=min(len(frames_data), 50),
    )

    gui_frustum_scale = server.gui.add_slider(
        "Frustum scale", min=0.05, max=0.5, step=0.01, initial_value=0.15
    )

    # Add checkboxes for train/test visibility
    gui_show_train = server.gui.add_checkbox("Show train frames", initial_value=True)
    gui_show_test = server.gui.add_checkbox("Show test frames", initial_value=True)

    frames: List[viser.FrameHandle] = []

    def visualize_frames() -> None:
        """Send all SEVA elements to viser for visualization."""

        # Remove existing image frames.
        for frame in frames:
            frame.remove()
        frames.clear()

        # Filter frames based on checkboxes
        frame_indices = []
        for i, frame in enumerate(frames_data):
            is_train = i in train_ids
            is_test = i in test_ids

            if (not train_ids and not test_ids) or (is_train and gui_show_train.value) or (is_test and gui_show_test.value):
                frame_indices.append(i)

        # Shuffle and limit
        random.shuffle(frame_indices)
        frame_indices = sorted(frame_indices[: gui_frames.value])

        for idx in tqdm(frame_indices):
            frame_data = frames_data[idx]
            transform_matrix = np.array(frame_data["transform_matrix"])

            # transforms.json stores OpenGL format c2w (already has Y/Z flipped)
            # No additional conversion needed for visualization
            R = transform_matrix[:3, :3]
            t = transform_matrix[:3, 3]
            transform_matrix[:, [1, 2]] *= -1

            # Create viser transform from c2w
            frame = server.scene.add_frame(
                f"/seva/frame_{idx:06d}",
                wxyz=vtf.SO3.from_matrix(R).wxyz,
                position=t,
                axes_length=0.1,
                axes_radius=0.005,
            )
            frames.append(frame)

            # Load image
            image_filename = images_path / Path(frame_data["file_path"]).name
            if not image_filename.exists():
                # Try with the full path from file_path
                image_filename = seva_path / frame_data["file_path"]

            if image_filename.exists():
                image = iio.imread(image_filename)
                image = image[::downsample_factor, ::downsample_factor]

                # Calculate FOV from intrinsics
                # fov = 2 * arctan2(h / 2, fy)
                fy = intrinsics["fl_y"]
                H = intrinsics["h"]
                W = intrinsics["w"]
                fov = 2 * np.arctan2(H / 2, fy)

                # Color code: train = greenish, test = reddish
                is_train = idx in train_ids
                is_test = idx in test_ids

                if train_ids or test_ids:
                    color = (0.2, 0.8, 0.2, 0.8) if is_train else (0.8, 0.2, 0.2, 0.8)
                else:
                    color = (0.5, 0.5, 0.5, 0.8)

                frustum = server.scene.add_camera_frustum(
                    f"/seva/frame_{idx:06d}/frustum",
                    fov=fov,
                    aspect=W / H,
                    scale=gui_frustum_scale.value,
                    image=image,
                )

                @frustum.on_click
                def _(_, frame=frame) -> None:
                    for client in server.get_clients().values():
                        client.camera.wxyz = frame.wxyz
                        client.camera.position = frame.position

    need_update = True

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_frustum_scale.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_show_train.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_show_test.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    # Initial visualization
    visualize_frames()

    while True:
        if need_update:
            need_update = False
            visualize_frames()

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)
