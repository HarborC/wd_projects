#!/usr/bin/env python3
"""Test Render module functionality."""

import sys
from pathlib import Path

sys.path.insert(0, '.')

from render.instant_splat_wrapper import InstantSplatWrapper

def test_render_wrapper():
    """Test InstantSplat wrapper initialization and configuration."""

    # 配置
    config = {
        'iterations': 100,  # 测试用小迭代次数
        'resolution': [512, 512],  # 降低分辨率用于测试
        'sh_degree': 0,
        'densify_until_iter': 0
    }

    # 初始化
    wrapper = InstantSplatWrapper(config)

    print(f"Iterations: {wrapper.iterations}")
    print(f"Resolution: {wrapper.resolution}")

    print("✅ InstantSplat wrapper initialized successfully")

if __name__ == "__main__":
    test_render_wrapper()
