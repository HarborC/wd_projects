#!/bin/bash

# Configuration
DATA_ROOT="/data/cjg/projects/wd_projects/data"
PROJECT_ROOT="/data/cjg/projects/results"
GPU_ID=0

# Step Toggles (Set to 1 to run, 0 to skip)
RUN_UNDISTORT=1
RUN_RECON=1
RUN_GAUSSIAN=1
RUN_CRAFTER=1
RUN_BEV=1

# Output directories
DIR_UNDISTORTED="${PROJECT_ROOT}/data_undistorted"
DIR_RECON="${PROJECT_ROOT}/reconstruction_output"
DIR_GAUSSIAN="${PROJECT_ROOT}/gaussians_output"
DIR_CRAFTER="${PROJECT_ROOT}/crafter_output"
DIR_BEV="${PROJECT_ROOT}/bev_output"

mkdir -p "${DIR_UNDISTORTED}"
mkdir -p "${DIR_RECON}"
mkdir -p "${DIR_GAUSSIAN}"
mkdir -p "${DIR_CRAFTER}"
mkdir -p "${DIR_BEV}"

# Activate environment if needed
# source activate wd

# Iterate over 'weidian-*' datasets
for dataset_path in ${DATA_ROOT}/weidian-*; do
    [ -d "$dataset_path" ] || continue
    
    dataset_name=$(basename "$dataset_path")
    echo "=========================================================="
    echo "Processing Dataset: ${dataset_name}"
    echo "=========================================================="

    # 1. Undistortion
    if [ "$RUN_UNDISTORT" -eq 1 ]; then
        echo ">>> [1/5] Running Undistortion..."
        python -m undistortion.factory \
            --input-dir "${dataset_path}" \
            --output-dir "${DIR_UNDISTORTED}/${dataset_name}" \
            --backend anycalib \
            --device cuda
    fi

    # 2. Reconstruction
    if [ "$RUN_RECON" -eq 1 ]; then
        echo ">>> [2/5] Running Reconstruction..."
        python -m reconstruction.factory \
            --input-dir "${DIR_UNDISTORTED}/${dataset_name}" \
            --output-dir "${DIR_RECON}/${dataset_name}" \
            --backend da3 \
            --device cuda
    fi

    # 3. BEV Generation
    if [ "$RUN_BEV" -eq 1 ]; then
        echo ">>> [3/5] Running BEV Generation..."
        python -m bev.bev_reconstruction \
            --reconstruction_dir "${DIR_RECON}/${dataset_name}" \
            --output_dir "${DIR_BEV}/${dataset_name}"
    fi

    # 4. Gaussian Splatting
    if [ "$RUN_GAUSSIAN" -eq 1 ]; then
        echo ">>> [4/5] Running Gaussian Splatting..."
        python -m gaussians.run_pipeline \
            -s "${DIR_RECON}/${dataset_name}" \
            -m "${DIR_GAUSSIAN}/${dataset_name}" \
            -gpu 0
    fi

    # 4. ViewCrafter
    if [ "$RUN_CRAFTER" -eq 1 ]; then
        echo ">>> [5/5] Running ViewCrafter Generation..."
        # ViewCrafter output path behaves like checkpoints usually, 
        # ensure directories exist
        mkdir -p "${DIR_CRAFTER}/${dataset_name}"
        
        python -m crafter.run_pipeline \
            --source_path "${DIR_RECON}/${dataset_name}" \
            --model_path "${DIR_CRAFTER}/${dataset_name}" \
            --gpu $GPU_ID
    fi

    
    
    echo "Done processing ${dataset_name}."
    echo ""
done

echo "All datasets processed."
