#!/bin/bash

# Standalone script to run simple_trainer_deblur.py
# Dedicated to running simple_trainer_deblur.py across all scenes

# Base paths
DATA_DIR="test_reconstruction_output"
RESULTS_DIR="$DATA_DIR/render_results"

# Script path
SCRIPT_TRAINER="render/simple_deblur_difix.py"

# Set GPU device
export CUDA_VISIBLE_DEVICES=2

# Create results directory
mkdir -p "$RESULTS_DIR"

# Colored output functions
print_header() {
    echo -e "\n\033[1;34m========================================\033[0m"
    echo -e "\033[1;34m$1\033[0m"
    echo -e "\033[1;34m========================================\033[0m\n"
}

print_info() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check if trainer script exists
    if [ ! -f "$SCRIPT_TRAINER" ]; then
        print_error "Script not found: $SCRIPT_TRAINER"
        exit 1
    fi
    
    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        print_error "Data directory not found: $DATA_DIR"
        exit 1
    fi
    
    # Check GPU availability
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not available, cannot check GPU status"
    else
        print_info "GPU status:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | head -1
    fi
    
    print_info "Dependency check complete"
}

# Script entry
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --check        Only check dependencies, do not run experiments"
    echo ""
    echo "Notes:"
    echo "  This script runs simple_trainer_deblur.py across all scenes (using GPU)"
    echo "  Total runs: $(( ${#BAD_NERF_SCENES[@]} + ${#REAL_BLUR_SCENES[@]} + ${#DL3DV_SCENES[@]} ))"
    exit 0
fi

if [ "$1" = "--check" ]; then
    check_dependencies
    exit 0
fi

# Run main program
check_dependencies

# Create result directory
mkdir -p "$RESULTS_DIR"

# Run training
TRAJ_TYPE="off"
    #--virtual-view-st--virtual-view-start-step 2000 \
    #   --virtual-view-interval 250 art-step 1500 \
    #  --virtual-view-interval 200 \
    #--max-steps 7000 \
python "$SCRIPT_TRAINER" default \
    --data-dir "$DATA_DIR" \
    --result-dir "$RESULTS_DIR" \
    --disable-viewer \
    --max-steps 7000 \
    --camera-optimizer.mode $TRAJ_TYPE \
    --init-ply-path "$DATA_DIR/gaussians.ply" \
    2>&1 | tee "${RESULTS_DIR}/training.log"
