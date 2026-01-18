"""
Pipeline package for multi-view reconstruction.

This package provides a modular, extensible pipeline for processing
multi-view images through undistortion and reconstruction stages.
"""
from pipeline.base import BaseStage, StageOutput
from pipeline.pipeline import Pipeline
from pipeline.stages import UndistortStage, ReconstructionStage

__all__ = ["Pipeline", "BaseStage", "StageOutput", "UndistortStage", "ReconstructionStage"]
