"""Unit tests for gen3r subpackage structure.

This module tests that the new gen3r subpackage structure (gen3r.models,
gen3r.models.vggt, gen3r.utils) can be imported correctly and have proper
documentation.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_gen3r_models_import():
    """Test that gen3r.models can be imported."""
    from gen3r import models
    assert models.__doc__ == "Gen3R model-related modules."

def test_gen3r_models_vggt_import():
    """Test that gen3r.models.vggt can be imported."""
    from gen3r.models import vggt
    assert vggt.__doc__ == "VGGT model components."

def test_gen3r_models_vggt_utils_import():
    """Test that gen3r.models.vggt.utils can be imported."""
    from gen3r.models.vggt import utils
    assert utils.__doc__ == "VGGT utility functions."

def test_gen3r_utils_import():
    """Test that gen3r.utils can be imported."""
    from gen3r import utils
    assert utils.__doc__ == "Gen3R utility functions."
