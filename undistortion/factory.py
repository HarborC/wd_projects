"""
Undistorter factory module.

This module provides a factory interface for creating undistorter instances,
making it easy to switch between different backends without changing client code.
"""
import logging
from typing import Optional, Dict, Any

from undistortion.base_undistorter import BaseUndistorter
from undistortion.geocalib_undistorter import GeoCalibUndistorter
from undistortion.anycalib_undistorter import AnyCalibUndistorter

logger = logging.getLogger(__name__)


class UndistorterFactory:
    """
    Factory class for creating undistorter instances.

    This factory provides a unified interface for creating different undistorter
    implementations, making it easy to switch between backends.
    """

    # Registry of available undistorters
    _UNDISTORTERS = {
        'geocalib': GeoCalibUndistorter,
        'anycalib': AnyCalibUndistorter,
    }

    @classmethod
    def create(
        cls,
        backend: str,
        device: Optional[str] = None,
        **kwargs
    ) -> BaseUndistorter:
        """
        Create an undistorter instance.

        Args:
            backend (str): Backend to use ('geocalib' or 'anycalib').
            device (str, optional): Device to run the model on ('cuda' or 'cpu').
                                    Defaults to auto-detect.
            **kwargs: Additional arguments passed to the specific undistorter:
                - For GeoCalib:
                    - weights (str, default="distorted")
                - For AnyCalib:
                    - model_id (str, default="anycalib_gen")
                    - camera_model (str, default="simple_radial")
                    - undistort_scale (float, default=-1.0 for adaptive)
                    - center_principal_point (bool, default=False)

        Returns:
            BaseUndistorter: An instance of the requested undistorter.

        Raises:
            ValueError: If the backend is not supported.

        Example:
            >>> # Create GeoCalib undistorter
            >>> undistorter = UndistorterFactory.create('geocalib', weights='distorted')
            >>>
            >>> # Create AnyCalib undistorter
            >>> undistorter = UndistorterFactory.create(
            ...     'anycalib',
            ...     model_id='anycalib_gen',
            ...     camera_model='simple_radial',
            ...     undistort_scale=-1.0
            ... )
        """
        backend_lower = backend.lower()

        if backend_lower not in cls._UNDISTORTERS:
            available = ', '.join(cls._UNDISTORTERS.keys())
            raise ValueError(
                f"Unknown backend: '{backend}'. "
                f"Available backends: {available}"
            )

        undistorter_class = cls._UNDISTORTERS[backend_lower]

        logger.info(f"Creating {backend_lower} undistorter with kwargs: {kwargs}")

        try:
            instance = undistorter_class(device=device, **kwargs)
            return instance
        except Exception as e:
            logger.error(f"Failed to create {backend_lower} undistorter: {e}")
            raise

    @classmethod
    def create_geocalib(
        cls,
        device: Optional[str] = None,
        weights: str = "distorted"
    ) -> GeoCalibUndistorter:
        """
        Create a GeoCalib undistorter (convenience method).

        Args:
            device (str, optional): Device to run the model on.
            weights (str): Weights to load for GeoCalib.

        Returns:
            GeoCalibUndistorter: GeoCalib undistorter instance.
        """
        return cls.create('geocalib', device=device, weights=weights)

    @classmethod
    def create_anycalib(
        cls,
        device: Optional[str] = None,
        model_id: str = "anycalib_gen",
        camera_model: str = "kb",
        undistort_scale: float = -1.0,
        center_principal_point: bool = True
    ) -> AnyCalibUndistorter:
        """
        Create an AnyCalib undistorter (convenience method).

        Args:
            device (str, optional): Device to run the model on.
            model_id (str): AnyCalib model ID.
            camera_model (str): Camera model to use.
            undistort_scale (float): Scale factor for undistortion.
            center_principal_point (bool): Force principal point to center.

        Returns:
            AnyCalibUndistorter: AnyCalib undistorter instance.
        """
        return cls.create(
            'anycalib',
            device=device,
            model_id=model_id,
            camera_model=camera_model,
            undistort_scale=undistort_scale,
            center_principal_point=center_principal_point
        )

    @classmethod
    def register_backend(cls, name: str, undistorter_class: type):
        """
        Register a new undistorter backend.

        This allows extending the factory with custom undistorter implementations.

        Args:
            name (str): Name of the backend.
            undistorter_class (type): Undistorter class (must inherit from BaseUndistorter).

        Raises:
            TypeError: If the class doesn't inherit from BaseUndistorter.

        Example:
            >>> class CustomUndistorter(BaseUndistorter):
            ...     pass
            >>> UndistorterFactory.register_backend('custom', CustomUndistorter)
        """
        if not issubclass(undistorter_class, BaseUndistorter):
            raise TypeError(
                f"{undistorter_class.__name__} must inherit from BaseUndistorter"
            )

        cls._UNDISTORTERS[name.lower()] = undistorter_class
        logger.info(f"Registered backend '{name}' -> {undistorter_class.__name__}")

    @classmethod
    def list_backends(cls) -> list:
        """
        List all available backends.

        Returns:
            list: List of available backend names.
        """
        return list(cls._UNDISTORTERS.keys())

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseUndistorter:
        """
        Create an undistorter from a configuration dictionary.

        Args:
            config (Dict): Configuration dictionary with keys:
                - backend (str): Backend name
                - device (str, optional): Device
                - Additional backend-specific parameters

        Returns:
            BaseUndistorter: Undistorter instance.

        Example:
            >>> config = {
            ...     'backend': 'anycalib',
            ...     'device': 'cuda',
            ...     'model_id': 'anycalib_gen',
            ...     'camera_model': 'simple_radial'
            ... }
            >>> undistorter = UndistorterFactory.from_config(config)
        """
        backend = config.get('backend')
        if not backend:
            raise ValueError("Config must contain 'backend' key")

        # Extract device
        device = config.get('device')

        # Extract backend-specific params (exclude 'backend' and 'device')
        kwargs = {k: v for k, v in config.items() if k not in ['backend', 'device']}

        return cls.create(backend, device=device, **kwargs)


def main():
    """
    Test function for AnyCalib undistortion.

    This function demonstrates how to use the UndistorterFactory to create
    an AnyCalib undistorter and process images.
    """
    import argparse
    from pathlib import Path

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Test AnyCalib undistortion using the factory pattern',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing input images to undistort'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save undistorted images'
    )
    parser.add_argument(
        '--intrinsics',
        type=str,
        default=None,
        help='Path to save intrinsics JSON file (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda or cpu). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--model-id',
        type=str,
        default='anycalib_gen',
        help='AnyCalib model ID to use'
    )
    parser.add_argument(
        '--camera-model',
        type=str,
        default='kb',
        help='Camera model for calibration'
    )
    parser.add_argument(
        '--undistort-scale',
        type=float,
        default=-1.0,
        help='Scale factor for undistortion. < 0 for adaptive, > 0 for fixed scale'
    )
    parser.add_argument(
        '--center-principal-point',
        action='store_true',
        help='Force principal point to image center'
    )

    args = parser.parse_args()

    # Validate paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Testing AnyCalib Undistortion via Factory Pattern")
    logger.info("=" * 80)
    logger.info(f"Input directory:  {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device:           {args.device or 'auto-detect'}")
    logger.info(f"Model ID:         {args.model_id}")
    logger.info(f"Camera model:     {args.camera_model}")
    logger.info(f"Undistort scale:  {args.undistort_scale} {'(adaptive)' if args.undistort_scale < 0 else '(fixed)'}")
    logger.info(f"Center PP:        {args.center_principal_point}")
    logger.info("=" * 80)

    try:
        # Method 1: Using create_anycalib convenience method
        logger.info("\n[Method 1] Creating undistorter using create_anycalib()...")
        undistorter = UndistorterFactory.create_anycalib(
            device=args.device,
            model_id=args.model_id,
            camera_model=args.camera_model,
            undistort_scale=args.undistort_scale,
            center_principal_point=args.center_principal_point
        )

        # Process images
        logger.info("\nProcessing images...")
        results = undistorter.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            intrinsics_output_path=args.intrinsics
        )

        # Summary
        successful = sum(1 for v in results.values() if 'error' not in v)
        failed = sum(1 for v in results.values() if 'error' in v)

        logger.info("\n" + "=" * 80)
        logger.info("Processing Complete!")
        logger.info(f"Total images:     {len(results)}")
        logger.info(f"Successful:       {successful}")
        logger.info(f"Failed:           {failed}")

        if args.intrinsics:
            logger.info(f"Intrinsics saved: {args.intrinsics}")

        logger.info("=" * 80)

        # Show first result as example
        if results:
            first_key = list(results.keys())[0]
            first_result = results[first_key]
            if 'error' not in first_result:
                logger.info(f"\nExample result for {Path(first_key).name}:")
                logger.info(f"  Output: {first_result.get('output_path')}")
                if 'camera' in first_result:
                    cam_info = first_result['camera']
                    if 'predicted' in cam_info:
                        logger.info(f"  Predicted model: {cam_info['predicted']['model']}")
                    if 'undistorted_pinhole' in cam_info:
                        logger.info(f"  Undistorted model: {cam_info['undistorted_pinhole']['model']}")

        return 0

    except Exception as e:
        logger.error(f"\nError during processing: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

"""
python -m undistortion.factory \
    --input-dir /home/disk2/jiagangchen/LJ/process_data/first_frames \
    --output-dir ./test_undistort_output \
    --intrinsics ./test_undistort_output/intrinsics.json \
    --device cuda
"""