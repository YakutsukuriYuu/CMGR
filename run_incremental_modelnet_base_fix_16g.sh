#!/bin/bash
# Run the ModelNet40 incremental stage from the latest base-only output.
# Usage: bash run_incremental_modelnet_base_fix_16g.sh [gpu_id]

set -e
cd "$(dirname "$0")"

PYTHON="/home/qiansu/main/clip/.venv/bin/python"
CONFIG="configs/modelnet_full_16g.yaml"
DATA_ROOT="../data/ModelNet40_Align"
RECON_CKPT="../deps/ReCon/pretrained/recon.pth"
DEPTH_CKPT="../deps/CLIP2Point/pretrained/vit32/best_eval.pth"

BASE_DIR="outputs_modelnet_base_fix/base"
INC_DIR="outputs_modelnet_base_fix/incremental"
GPU=${1:-0}
LOGFILE="modelnet_incremental_base_fix.log"

exec > >(tee "$LOGFILE") 2>&1

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "CMGR ModelNet40 Incremental Training"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: $GPU"
echo "Config: $CONFIG"
echo "Base dir: $BASE_DIR"
echo "Output dir: $INC_DIR"
echo "============================================"

if [ ! -f "$BASE_DIR/netB_best.pth" ] && [ ! -f "$BASE_DIR/netB_final.pth" ]; then
    echo "Base checkpoint not found in $BASE_DIR"
    echo "Expected netB_best.pth or netB_final.pth."
    exit 1
fi

if [ ! -f "$BASE_DIR/exemplars.pth" ]; then
    echo "Exemplars not found: $BASE_DIR/exemplars.pth"
    echo "Run base training first, or point BASE_DIR to the finished base output."
    exit 1
fi

"$PYTHON" train_incremental.py \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --base_dir "$BASE_DIR" \
    --output_dir "$INC_DIR" \
    --recon_ckpt "$RECON_CKPT" \
    --depth_ckpt "$DEPTH_CKPT" \
    --gpu 0 \
    --num_base_classes 20 \
    --classes_per_task 5

echo ""
echo "============================================"
echo "Results:"
echo "============================================"
if [ -f "$INC_DIR/results.yaml" ]; then
    cat "$INC_DIR/results.yaml"
else
    echo "Incremental training finished without results.yaml."
fi
