"""
Reconstruction factory module.

This module provides a factory interface for creating reconstructor instances,
making it easy to switch between different backends without changing client code.
"""
import logging
from typing import Optional, Dict, Any

from reconstruction.base_reconstructor import BaseReconstructor
from reconstruction.da3_reconstructor import DA3Reconstructor
from reconstruction.mast3r_reconstructor import MASt3RReconstructor
from reconstruction.hunyuanworld_reconstructor import HunyuanWorldReconstructor

logger = logging.getLogger(__name__)


class ReconstructionFactory:
    """
    Factory class for creating reconstructor instances.

    This factory provides a unified interface for creating different reconstructor
    implementations, making it easy to switch between backends.
    """

    # Registry of available reconstructors
    _RECONSTRUCTORS = {
        'da3': DA3Reconstructor,
        'mast3r': MASt3RReconstructor, # mast3r depth you wenti
        'hunyuanworld': HunyuanWorldReconstructor,
    }

    @classmethod
    def create(
        cls,
        backend: str,
        device: Optional[str] = None,
        **kwargs
    ) -> BaseReconstructor:
        """
        Create a reconstructor instance.

        Args:
            backend (str): Backend to use ('da3', 'mast3r', or 'hunyuanworld').
            device (str, optional): Device to run the model on ('cuda' or 'cpu').
                                    Defaults to auto-detect.
            **kwargs: Additional arguments passed to the specific reconstructor:
                - For DA3:
                    - model_name (str, default="depth-anything/DA3NESTED-GIANT-LARGE")
                - For MASt3R:
                    - ckpt_path (str, path to MASt3R checkpoint)
                - For HunyuanWorld-Mirror:
                    - model_path (str, default="tencent/HunyuanWorld-Mirror")

        Returns:
            BaseReconstructor: An instance of the requested reconstructor.

        Raises:
            ValueError: If the backend is not supported.

        Example:
            >>> # Create DA3 reconstructor
            >>> reconstructor = ReconstructionFactory.create('da3', model_name='depth-anything/DA3NESTED-GIANT-LARGE')
            >>>
            >>> # Create MASt3R reconstructor
            >>> reconstructor = ReconstructionFactory.create('mast3r')
            >>>
            >>> # Create HunyuanWorld-Mirror reconstructor
            >>> reconstructor = ReconstructionFactory.create('hunyuanworld')
        """
        backend_lower = backend.lower()

        if backend_lower not in cls._RECONSTRUCTORS:
            available = ', '.join(cls._RECONSTRUCTORS.keys())
            raise ValueError(
                f"Unknown backend: '{backend}'. "
                f"Available backends: {available}"
            )

        reconstructor_class = cls._RECONSTRUCTORS[backend_lower]

        logger.info(f"Creating {backend_lower} reconstructor with kwargs: {kwargs}")

        try:
            instance = reconstructor_class(device=device, **kwargs)
            return instance
        except Exception as e:
            logger.error(f"Failed to create {backend_lower} reconstructor: {e}")
            raise

    @classmethod
    def create_da3(
        cls,
        device: Optional[str] = None,
        model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    ) -> DA3Reconstructor:
        """
        Create a DA3 reconstructor (convenience method).

        Args:
            device (str, optional): Device to run the model on.
            model_name (str): DA3 model name.

        Returns:
            DA3Reconstructor: DA3 reconstructor instance.
        """
        return cls.create('da3', device=device, model_name=model_name)

    @classmethod
    def create_mast3r(
        cls,
        device: Optional[str] = None,
        ckpt_path: str = '/home/disk2/jiagangchen/LJ/sparse_view/InstantSplat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    ) -> MASt3RReconstructor:
        """
        Create a MASt3R reconstructor (convenience method).

        Args:
            device (str, optional): Device to run the model on.
            ckpt_path (str): Path to MASt3R checkpoint.

        Returns:
            MASt3RReconstructor: MASt3R reconstructor instance.
        """
        return cls.create('mast3r', device=device, ckpt_path=ckpt_path)

    @classmethod
    def create_hunyuanworld(
        cls,
        device: Optional[str] = None,
        model_path: str = "tencent/HunyuanWorld-Mirror"
    ) -> HunyuanWorldReconstructor:
        """
        Create a HunyuanWorld-Mirror reconstructor (convenience method).

        Args:
            device (str, optional): Device to run the model on.
            model_path (str): Path or HuggingFace ID for HunyuanWorld-Mirror model.

        Returns:
            HunyuanWorldReconstructor: HunyuanWorld-Mirror reconstructor instance.
        """
        return cls.create('hunyuanworld', device=device, model_path=model_path)

    @classmethod
    def register_backend(cls, name: str, reconstructor_class: type):
        """
        Register a new reconstructor backend.

        This allows extending the factory with custom reconstructor implementations.

        Args:
            name (str): Name of the backend.
            reconstructor_class (type): Reconstructor class (must inherit from BaseReconstructor).

        Raises:
            TypeError: If the class doesn't inherit from BaseReconstructor.

        Example:
            >>> class CustomReconstructor(BaseReconstructor):
            ...     pass
            >>> ReconstructionFactory.register_backend('custom', CustomReconstructor)
        """
        if not issubclass(reconstructor_class, BaseReconstructor):
            raise TypeError(
                f"{reconstructor_class.__name__} must inherit from BaseReconstructor"
            )

        cls._RECONSTRUCTORS[name.lower()] = reconstructor_class
        logger.info(f"Registered backend '{name}' -> {reconstructor_class.__name__}")

    @classmethod
    def list_backends(cls) -> list:
        """
        List all available backends.

        Returns:
            list: List of available backend names.
        """
        return list(cls._RECONSTRUCTORS.keys())

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseReconstructor:
        """
        Create a reconstructor from a configuration dictionary.

        Args:
            config (Dict): Configuration dictionary with keys:
                - backend (str): Backend name
                - device (str, optional): Device
                - Additional backend-specific parameters

        Returns:
            BaseReconstructor: Reconstructor instance.

        Example:
            >>> config = {
            ...     'backend': 'da3',
            ...     'device': 'cuda',
            ...     'model_name': 'depth-anything/DA3NESTED-GIANT-LARGE'
            ... }
            >>> reconstructor = ReconstructionFactory.from_config(config)
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
    Test function for reconstruction.

    This function demonstrates how to use the ReconstructionFactory to create
    a reconstructor and process images.
    """
    import argparse
    from pathlib import Path

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Test 3D reconstruction using the factory pattern',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing input images for reconstruction'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save reconstruction results'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='da3',
        choices=['da3', 'mast3r', 'hunyuanworld'],
        help='Backend to use for reconstruction'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda or cpu). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='depth-anything/DA3NESTED-GIANT-LARGE',
        help='Model name (for DA3 backend)'
    )
    parser.add_argument(
        '--ckpt-path',
        type=str,
        default='/home/disk2/jiagangchen/LJ/sparse_view/InstantSplat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
        help='Checkpoint path (for MASt3R backend)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='tencent/HunyuanWorld-Mirror',
        help='Model path (for HunyuanWorld backend)'
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
    logger.info("Testing 3D Reconstruction via Factory Pattern")
    logger.info("=" * 80)
    logger.info(f"Input directory:  {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Backend:          {args.backend}")
    logger.info(f"Device:           {args.device or 'auto-detect'}")
    logger.info("=" * 80)

    try:
        # Create reconstructor
        logger.info(f"\nCreating {args.backend} reconstructor...")

        if args.backend == 'da3':
            reconstructor = ReconstructionFactory.create_da3(
                device=args.device,
                model_name=args.model_name
            )
        elif args.backend == 'mast3r':
            reconstructor = ReconstructionFactory.create_mast3r(
                device=args.device,
                ckpt_path=args.ckpt_path
            )
        elif args.backend == 'hunyuanworld':
            reconstructor = ReconstructionFactory.create_hunyuanworld(
                device=args.device,
                model_path=args.model_path
            )
        else:
            raise ValueError(f"Unknown backend: {args.backend}")

        # Process images
        logger.info("\nProcessing images...")
        reconstructor.process_directory(
            input_dir=input_dir,
            output_dir=output_dir
        )

        logger.info("\n" + "=" * 80)
        logger.info("Reconstruction Complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\nError during processing: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

"""
python -m reconstruction.factory \
    --input-dir ./data/data3 \
    --output-dir ./test_reconstruction_output2 \
    --backend hunyuanworld \
    --device cuda

python -m reconstruction.factory \
    --input-dir ./test_undistort_output \
    --output-dir ./test_reconstruction_output \
    --backend hunyuanworld \
    --device cuda
"""
