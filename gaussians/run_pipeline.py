import os
import sys
from argparse import ArgumentParser, Namespace

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, PipelineParams, OptimizationParams
from train import training
from render import render_sets

def run_pipeline(source_path, model_path, gpu=0, n_views=None, iterations=1000, skip_train=False):
    """
    Run the InstantSplat training and rendering pipeline.
    
    Args:
        source_path (str): Path to source data (images, sparse folder).
        model_path (str): Path to output model.
        gpu (int): GPU ID to use.
        n_views (int): Number of views (optional, auto-detected if None).
        iterations (int): Number of training iterations.
        skip_train (bool): If True, skip the training step.
    """
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    source_path = os.path.abspath(source_path)
    model_path = os.path.abspath(model_path)
    
    if n_views is None:
        # Auto-detect logic
        try:
             # Try to count images in 'images' folder
             img_dir = os.path.join(source_path, "images")
             if os.path.exists(img_dir):
                 n_views = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
             else:
                 # Fallback if structure is different
                 n_views = 0 
        except Exception as e:
            print(f"Warning: Could not auto-detect n_views: {e}")
            n_views = 0
            
    print(f"=======================================================")
    print(f"Running InstantSplat Pipeline")
    print(f"Source: {source_path}")
    print(f"Output: {model_path}")
    print(f"GPU: {gpu} | Views: {n_views} | Iters: {iterations}")
    print(f"Skip Train: {skip_train}")
    print(f"=======================================================")

    # Ensure output directory exists
    os.makedirs(model_path, exist_ok=True)

    # =========================================================================
    # 1. TRAINING
    # =========================================================================
    if not skip_train:
        print("\n[Step 1/2] Starting Training...")
        
        # Reset/Create Parser for Training
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        
        # Add standard training arguments that are usually in train.py
        parser.add_argument('--debug_from', type=int, default=-1)
        parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--start_checkpoint", type=str, default = None)
        
        # Define arguments to simulate CLI call
        train_args_list = [
            "-s", source_path,
            "-m", model_path,
            "-r", "1",
            "--n_views", str(n_views),
            "--iterations", str(iterations),
            "--pp_optimizer",
            "--optim_pose"
        ]
        
        # Parse arguments
        args = parser.parse_args(train_args_list)
        
        # Ensure we save the model at the final iteration
        if iterations not in args.save_iterations:
            args.save_iterations.append(iterations)
        
        # Run Training
        training(
            lp.extract(args), 
            op.extract(args), 
            pp.extract(args), 
            args.test_iterations, 
            args.save_iterations, 
            args.checkpoint_iterations, 
            args.start_checkpoint, 
            args.debug_from
        )
        print("Training Completed.")
    else:
        print("\n[Step 1/2] Training Skipped.")
    
    # =========================================================================
    # 2. RENDERING
    # =========================================================================
    
    # =========================================================================
    # 2. RENDERING
    # =========================================================================
    print("\n[Step 2/2] Starting Rendering...")
    
    # Reset/Create Parser for Rendering (render.py has slightly different args)
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    # op = OptimizationParams(parser) # render doesn't use OptParams explicitly but pipeline params exist
    pp = PipelineParams(parser)
    
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--optim_test_pose_iter", default=500, type=int)
    parser.add_argument("--infer_video", action="store_true")
    parser.add_argument("--test_fps", action="store_true")
    
    render_args_list = [
        "-s", source_path,
        "-m", model_path,
        "-r", "1",
        "--n_views", str(n_views),
        "--iterations", str(iterations),
        "--infer_video"
    ]
    
    args = parser.parse_args(render_args_list)
    
    # Run Rendering
    render_sets(
        lp.extract(args), 
        args.iterations, 
        pp.extract(args), 
        args.skip_train, 
        args.skip_test, 
        args
    )
    print("Rendering Completed.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True, help="Source path containing 'images' and 'sparse' folder")
    parser.add_argument("-m", "--model_path", required=True, help="Model output path")
    parser.add_argument("-g", "--gpu", "-gpu", default=0, type=int, help="GPU ID to use")
    parser.add_argument("-n", "--n_views", default=None, type=int, help="Number of views")
    parser.add_argument("-i", "--iterations", default=1000, type=int, help="Number of iterations")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and run only rendering")
    
    cli_args = parser.parse_args()
    
    run_pipeline(
        cli_args.source_path,
        cli_args.model_path,
        cli_args.gpu,
        cli_args.n_views,
        cli_args.iterations,
        cli_args.skip_train
    )

"""
python -m gaussians.run_pipeline -s test_reconstruction_output -m test_gaussians_output -gpu 0
"""