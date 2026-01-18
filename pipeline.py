#!/usr/bin/env python3
"""
Multi-view Reconstruction Pipeline - CLI Entry Point

A modular pipeline for multi-view image processing including undistortion
and 3D reconstruction using various state-of-the-art methods.

Usage:
    # Using command line arguments (main branch - default)
    python pipeline.py --input-dir ./images --output-dir ./output --reconstruct-method hunyuanworld

    # Using configuration file
    python pipeline.py --input-dir ./images --output-dir ./output --config configs/hunyuanworld.json

    # Skip undistortion (input already undistorted)
    python pipeline.py --input-dir ./undistorted --output-dir ./output --skip-undistort

    # Using gen3r branch (requires config file)
    python pipeline.py --branch gen3r --input-dir ./images --output-dir ./output --config configs/gen3r.json
"""
import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Pipeline after logging is configured
from pipeline import Pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-view Reconstruction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/output
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output base directory"
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        choices=["main", "gen3r"],
        help="Pipeline branch to use: 'main' for standard reconstruction with output rendering, 'gen3r' for Gen3R-specific processing"
    )

    # Undistortion configuration
    parser.add_argument(
        "--undistort-method",
        type=str,
        default="geocalib",
        choices=["geocalib", "anycalib"],
        help="Undistortion method"
    )
    parser.add_argument(
        "--skip-undistort",
        action="store_true",
        help="Skip undistortion step (input images are already undistorted)"
    )

    # Reconstruction configuration
    parser.add_argument(
        "--reconstruct-method",
        type=str,
        default="da3",
        choices=["da3", "mast3r", "hunyuanworld", "vggtx"],
        help="Reconstruction method"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (cuda/cpu). Auto-detect if not specified."
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file (overrides command-line args)"
    )

    return parser.parse_args()


def build_config_from_args(args) -> dict:
    """Build configuration dictionary from command-line arguments."""
    config = {
        "branch": args.branch
    }

    if not args.skip_undistort:
        config["undistort"] = {
            "method": args.undistort_method,
            "output_dir": "undistorted",
            "intrinsics_path": "undistorted/intrinsics.json"
        }

    config["reconstruction"] = {
        "method": args.reconstruct_method,
        "device": args.device,
        "output_dir": "reconstruction"
    }

    # Add output stage for main branch
    if args.branch == "main":
        config["output"] = {
            "method": "render",
            "output_dir": "output",
            "interpolation": "default"
        }

    return config


def main():
    """Main entry point."""
    args = parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1

    # Validate gen3r branch requires config file
    if args.branch == "gen3r" and not args.config:
        logger.error("Gen3R branch requires a configuration file. Please provide --config argument.")
        return 1

    # Build configuration
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        pipeline = Pipeline.from_config_file(args.config, branch=args.branch)
    else:
        config = build_config_from_args(args)
        pipeline = Pipeline(config, branch=args.branch)

    # Print configuration
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline Configuration")
    logger.info("=" * 60)
    logger.info(f"Input directory:  {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Branch:           {args.branch}")
    logger.info(f"Configured stages: {pipeline.list_stages()}")
    logger.info("=" * 60)
    logger.info("")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run pipeline
    try:
        outputs = pipeline.run(str(input_dir), args.output_dir)

        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Pipeline Summary")
        logger.info("=" * 60)
        for stage_name, output in outputs.items():
            logger.info(f"{stage_name}: {output.output_dir}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.exception("Exception details:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
