#!/bin/bash

# Default values
GPU_ID=0
ITERATIONS=1000
N_VIEWS=6

# Help function
help() {
    echo "Usage: $0 -s <SOURCE_PATH> -m <MODEL_PATH> [-g <GPU_ID>] [-n <N_VIEWS>] [-i <ITERATIONS>]"
    echo ""
    echo "Arguments:"
    echo "  -s  Source path containing 'images' and 'sparse' (or 'sparse_N') folder."
    echo "  -m  Model output path."
    echo "  -g  GPU ID to use (default: 0)."
    echo "  -n  Number of views (default: 6)."
    echo "  -i  Number of iterations (default: 1000)."
    exit 1
}

# Parse args
while getopts "s:m:g:n:i:h" opt; do
    case $opt in
        s) SOURCE_PATH="$OPTARG" ;;
        m) MODEL_PATH="$OPTARG" ;;
        g) GPU_ID="$OPTARG" ;;
        n) N_VIEWS="$OPTARG" ;;
        i) ITERATIONS="$OPTARG" ;;
        h) help ;;
        *) help ;;
    esac
done

if [ -z "$SOURCE_PATH" ] || [ -z "$MODEL_PATH" ]; then
    echo "Error: Source and Model paths are required."
    help
fi

# Determine views if not set (try to count images)
if [ -z "$N_VIEWS" ]; then
     N_VIEWS=$(ls -1 ${SOURCE_PATH}/images | wc -l)
     echo "Auto-detected N_VIEWS: $N_VIEWS"
fi

echo "======================================================="
echo "Running InstantSplat Training (Scheme 2)"
echo "Source: $SOURCE_PATH"
echo "Output: $MODEL_PATH"
echo "GPU: $GPU_ID | Views: $N_VIEWS | Iters: $ITERATIONS"
echo "======================================================="

mkdir -p $MODEL_PATH

# 1. Train
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    -r 1 \
    --n_views ${N_VIEWS} \
    --iterations ${ITERATIONS} \
    --pp_optimizer \
    --optim_pose \
    > ${MODEL_PATH}/train.log 2>&1

if [ $? -ne 0 ]; then
    echo "Training failed! Check ${MODEL_PATH}/train.log"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed. Log in ${MODEL_PATH}/train.log"

# 2. Render
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    -r 1 \
    --n_views ${N_VIEWS} \
    --iterations ${ITERATIONS} \
    --infer_video \
    > ${MODEL_PATH}/render.log 2>&1

if [ $? -ne 0 ]; then
    echo "Rendering failed! Check ${MODEL_PATH}/render.log"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done. Video saved in ${MODEL_PATH}/interp/ours_${ITERATIONS}/"
