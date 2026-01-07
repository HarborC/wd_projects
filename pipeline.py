import argparse
import logging
import sys
import os
from pathlib import Path

from undistortion import GeoCalibUndistorter
from reconstruction import Reconstructor
from da3_reconstruction import DA3PostProcessor
import numpy as np
import cv2
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(input_dir: str, output_dir: str, intrinsics_path: str, reconstruction_output_dir: str, method: str = "da3"):
    """
    Run the processing pipeline.
    
    Args:
        input_dir (str): Path to input images.
        output_dir (str): Path to save output images (undistorted).
        intrinsics_path (str): Path to save intrinsics JSON.
        reconstruction_output_dir (str): Path to save reconstruction results.
        method (str): Reconstruction method ('da3' or 'mast3r').
    """
    # Step 1: Undistortion
    logger.info("Starting Step 1: Image Undistortion")
    try:
        undistorter = GeoCalibUndistorter()
        undistorter.process_directory(input_dir, output_dir, intrinsics_path)
        logger.info("Step 1 completed successfully.")
    except Exception as e:
        logger.error(f"Step 1 failed: {e}")
        return

    # Step 2: 3D Reconstruction
    logger.info(f"Starting Step 2: 3D Reconstruction with {method}")
    try:
        reconstructor = Reconstructor(method=method)
        # Use the output of step 1 (undistorted images) as input for step 2
        reconstructor.process_directory(output_dir, reconstruction_output_dir)
        
        logger.info("Step 2 completed successfully.")
    except Exception as e:
        logger.error(f"Step 2 failed: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Pipeline")
    parser.add_argument("--input_dir", type=str, default="/home/disk2/jiagangchen/LJ/process_data/first_frames", help="Input directory containing images")
    parser.add_argument("--output_dir", type=str, default="/home/disk2/jiagangchen/wd_projects/test_data/undistorted_first_frames", help="Output directory for undistorted images")
    parser.add_argument("--intrinsics_path", type=str, default="/home/disk2/jiagangchen/wd_projects/test_data/intrinsics_first_frames_geocalib.json", help="Path to save intrinsics JSON")
    parser.add_argument("--reconstruction_output_dir", type=str, default="/home/disk2/jiagangchen/wd_projects/test_data/reconstruction_output", help="Output directory for reconstruction results")
    parser.add_argument("--method", type=str, default="mast3r", choices=["da3", "mast3r"], help="Reconstruction method: 'da3', 'mast3r'")
    
    args = parser.parse_args()
    
    run_pipeline(args.input_dir, args.output_dir, args.intrinsics_path, args.reconstruction_output_dir, args.method)
