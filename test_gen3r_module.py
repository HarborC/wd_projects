#!/usr/bin/env python3
"""Test Gen3R module functionality."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, '.')

from gen3r.gen3r_pipeline import Gen3RPipelineWrapper

def test_gen3r_wrapper():
    """Test Gen3R wrapper initialization and configuration."""

    # 配置
    config = {
        'model_path': 'test_code/Gen3R/checkpoints',
        'task': 'allview',  # or '1view', '2view'
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 初始化
    wrapper = Gen3RPipelineWrapper(config)

    print(f"Model path: {wrapper.model_path}")
    print(f"Task: {wrapper.task}")
    print(f"Device: {wrapper.device}")

    print("✅ Gen3R wrapper initialized successfully")

if __name__ == "__main__":
    test_gen3r_wrapper()
