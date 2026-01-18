"""Test common_utils module."""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gen3r.utils.common_utils import colorize_depth_map

def test_colorize_depth_map():
    """Test depth map colorization."""
    # Create dummy depth map
    depth = torch.randn(1, 100, 100, 1)

    # Call function
    colored = colorize_depth_map(depth)

    # Verify output shape
    assert colored.shape[-1] == 3, f"Expected RGB output, got shape {colored.shape}"
    assert colored.shape[0:3] == depth.shape[0:3], "Spatial dimensions should match"
