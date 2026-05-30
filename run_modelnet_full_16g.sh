#!/bin/bash
# CMGR ModelNet40 full run for a 16GB GPU.
# Usage: bash run_modelnet_full_16g.sh [gpu_id]

set -e

cd "$(dirname "$0")"

PYTHON="/home/qiansu/main/clip/.venv/bin/python"
CONFIG="configs/modelnet_full_16g.yaml"
DATA_ROOT="../data/ModelNet40_Align"
RECON_CKPT="../deps/ReCon/pretrained/recon.pth"
DEPTH_CKPT="../deps/CLIP2Point/pretrained/vit32/best_eval.pth"
OUT_ROOT="outputs_modelnet_full_16g"
BASE_DIR="$OUT_ROOT/base"
INC_DIR="$OUT_ROOT/incremental"
GPU=${1:-0}

LOGFILE="modelnet_full_16g.log"
exec > >(tee "$LOGFILE") 2>&1

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "CMGR ModelNet40 Full Experiment (16GB GPU)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log:        $LOGFILE"
echo "Config:     $CONFIG"
echo "Data root:  $DATA_ROOT"
echo "GPU:        $GPU"
echo "Output:     $OUT_ROOT"
echo "Batch/view: batch_size=8, grad_accum_steps=2, effective_batch=16, num_views=10"
echo "============================================"

mkdir -p "$OUT_ROOT"

echo ""
echo ">>> Step 1/2: Base training"
echo ""
"$PYTHON" train_base.py \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --output_dir "$BASE_DIR" \
    --recon_ckpt "$RECON_CKPT" \
    --depth_ckpt "$DEPTH_CKPT" \
    --gpu 0 \
    --num_base_classes 20

echo ""
echo ">>> Step 2/2: Incremental training"
echo ""
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
echo "Experiment complete"
echo "============================================"
echo "Results: $INC_DIR/results.yaml"
if [ -f "$INC_DIR/results.yaml" ]; then
    cat "$INC_DIR/results.yaml"
else
    echo "(incremental results not found; check $LOGFILE)"
fi
