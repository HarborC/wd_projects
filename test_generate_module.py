#!/usr/bin/env python3
"""Test Generate (SEVA) module functionality."""

import sys
from pathlib import Path

sys.path.insert(0, '.')

from generate.seva_generator import SevaGenerator

def test_seva_generator():
    """Test SEVA generator initialization and configuration."""

    # 配置
    config = {
        'model_path': 'stabilityai/stable-virtual-camera',
        'guidance_scale': 2.0,
        'num_inference_steps': 10  # 测试用少步数
    }

    # 初始化
    generator = SevaGenerator(config)

    print(f"Model path: {generator.model_path}")
    print(f"Guidance scale: {generator.guidance_scale}")
    print(f"Inference steps: {generator.num_inference_steps}")

    print("✅ SevaGenerator initialized successfully")

if __name__ == "__main__":
    test_seva_generator()
