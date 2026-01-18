#!/usr/bin/env python3
"""Test Pipeline creation and configuration."""

import sys
sys.path.insert(0, '.')

from pipeline.pipeline import Pipeline

def test_pipeline_creation():
    """Test pipeline creation for both branches."""

    print("=" * 60)
    print("Testing Pipeline Creation")
    print("=" * 60)

    # Test 1: Main branch (minimal config)
    print("\n1. Main branch (without undistort):")
    main_config = {
        'branch': 'main',
        'reconstruction': {
            'method': 'hunyuanworld',
            'device': 'cuda',
            'output_dir': 'reconstruction'
        },
        'output': {
            'method': 'render',
            'output_dir': 'output'
        }
    }

    try:
        pipeline = Pipeline(main_config, branch='main')
        print(f"   Stages: {list(pipeline.stages.keys())}")
        print("   ✅ Main branch pipeline created")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Gen3R branch
    print("\n2. Gen3R branch:")
    gen3r_config = {
        'branch': 'gen3r',
        'gen3r': {
            'task': 'allview',
            'prompts': 'a beautiful scene'
        }
    }

    try:
        pipeline = Pipeline(gen3r_config, branch='gen3r')
        print(f"   Stages: {list(pipeline.stages.keys())}")
        print("   ✅ Gen3R branch pipeline created")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("Pipeline Creation Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline_creation()
