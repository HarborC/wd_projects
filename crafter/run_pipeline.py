import os
import argparse
import sys
from pathlib import Path
import numpy as np
import pycolmap

# Make sure we can import from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from viewcrafter import ViewCrafter
from configs.infer_config import get_parser

def run_pipeline(source_path, model_path, gpu):
    print(f"Running ViewCrafter Pipeline")
    print(f"Source: {source_path}")
    
    # Support directory input by looking for scene_metadata.npz
    if os.path.isdir(source_path):
        metadata_file = os.path.join(source_path, "scene_metadata.npz")
        
        # If metadata exists, prioritize it
        if os.path.exists(metadata_file):
            print(f"Found metadata file: {metadata_file}")
            source_path = metadata_file
        else:
            # If not, assume it's a raw reconstruction folder that ViewCrafter can parse
            if os.path.exists(os.path.join(source_path, "sparse")) and os.path.exists(os.path.join(source_path, "depths")):
                print(f"Using raw reconstruction directory: {source_path}")
            else:
                print(f"Warning: {source_path} is a directory but missing typical reconstruction subfolders.")

    print(f"Output: {model_path}")
    print(f"GPU: {gpu}")
    
    # Fix paths to be absolute based on crafter root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check/Download checkpoint logic moved here to use crafter_root
    ckpt_dir = os.path.join(project_root, 'checkpoints')
    ckpt_file = os.path.join(ckpt_dir, 'model_sparse.ckpt')
    if not os.path.exists(ckpt_file):
        print(f"Downloading checkpoint to {ckpt_file}...")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.system(f'wget https://hf-mirror.com/Drexubery/ViewCrafter_25_sparse/resolve/main/model_sparse.ckpt -O {ckpt_file}')

    parser = get_parser()
    
    # Map arguments to ViewCrafter expected opts
    # We pass a dummy image_dir because the config parser probably requires it or defaults it.
    # We force sparse_view_interp mode.
    
    cmd_args = [
        "--cached_reconstruction_path", source_path,
        "--out_dir", model_path,
        "--exp_name", "", # Empty so we don't create extra subfolder if possible, or we handle save_dir manually
        "--mode", "sparse_view_interp",
        "--image_dir", "dummy", 
        "--device", f"cuda:{gpu}"
    ]
    
    opts = parser.parse_args(cmd_args)
    
    # Fix paths to be absolute based on crafter root
    crafter_root = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(opts.config):
        opts.config = os.path.join(crafter_root, opts.config)
        
    opts.ckpt_path = ckpt_file

    # Auto-fix checkpoint name if model.ckpt doesn't exist but model_sparse.ckpt does
    if not os.path.exists(opts.ckpt_path) and os.path.exists(os.path.join(os.path.dirname(opts.ckpt_path), 'model_sparse.ckpt')):
        print("Redirecting checkpoint to model_sparse.ckpt")
        opts.ckpt_path = os.path.join(os.path.dirname(opts.ckpt_path), 'model_sparse.ckpt')
    
    print(f"Final Checkpoint Path: {opts.ckpt_path}")
        
    # Override save_dir (inference.py logic was: save_dir = join(out_dir, exp_name))
    # We want save_dir = model_path
    opts.save_dir = model_path
    os.makedirs(opts.save_dir, exist_ok=True)
    
    pvd = ViewCrafter(opts)
    pvd.nvs_sparse_view_interp()
    print("Pipeline Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True, help="Path to scene_metadata.npz or reconstruction folder")
    parser.add_argument("-m", "--model_path", required=True, help="Output directory")
    parser.add_argument("-gpu", "--gpu", default="0", help="GPU ID")
    
    args = parser.parse_args()
    
    run_pipeline(args.source_path, args.model_path, args.gpu)

"""
python -m crafter.run_pipeline \
    --source_path test_reconstruction_output \
    --model_path test_crafter_output \
    --gpu 0
"""