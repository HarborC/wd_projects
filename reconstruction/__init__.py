"""
Reconstruction package.

This package provides a unified interface for 3D reconstruction
using multiple backends (DA3, MASt3R, HunyuanWorld-Mirror).
"""

from reconstruction.base_reconstructor import BaseReconstructor
from reconstruction.da3_reconstructor import DA3Reconstructor
from reconstruction.mast3r_reconstructor import MASt3RReconstructor
from reconstruction.hunyuanworld_reconstructor import HunyuanWorldReconstructor
from reconstruction.factory import ReconstructionFactory

__all__ = [
    'BaseReconstructor',
    'DA3Reconstructor',
    'MASt3RReconstructor',
    'HunyuanWorldReconstructor',
    'ReconstructionFactory',
]

__version__ = '0.1.0'
