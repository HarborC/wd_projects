"""
Undistortion package.

This package provides a unified interface for camera calibration and image undistortion
using multiple backends (GeoCalib, AnyCalib).
"""

from undistortion.base_undistorter import BaseUndistorter
from undistortion.geocalib_undistorter import GeoCalibUndistorter
from undistortion.anycalib_undistorter import AnyCalibUndistorter
from undistortion.factory import UndistorterFactory

__all__ = [
    'BaseUndistorter',
    'GeoCalibUndistorter',
    'AnyCalibUndistorter',
    'UndistorterFactory',
]

__version__ = '0.1.0'
