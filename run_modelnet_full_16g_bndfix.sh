#!/bin/bash
# Run a full ModelNet40 base + incremental experiment with the BND routing fix.
# Logs and outputs are timestamped so previous runs are not overwritten.
# Usage: bash run_modelnet_full_16g_bndfix.sh [gpu_id]

set -e
cd "$(dirname "$0")"

PYTHON="/home/qiansu/main/clip/.venv/bin/python"
CONFIG="configs/modelnet_full_16g.yaml"
DATA_ROOT="../data/ModelNet40_Align"
RECON_CKPT="../deps/ReCon/pretrained/recon.pth"
DEPTH_CKPT="../deps/CLIP2Point/pretrained/vit32/best_eval.pth"

GPU=${1:-0}
RUN_ID=$(date '+%Y%m%d_%H%M%S')
OUT_DIR="outputs_modelnet_full_16g_bndfix_${RUN_ID}"
BASE_DIR="${OUT_DIR}/base"
INC_DIR="${OUT_DIR}/incremental"
LOGFILE="modelnet_full_16g_bndfix_${RUN_ID}.log"

exec > >(tee "$LOGFILE") 2>&1

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "CMGR ModelNet40 Full Experiment"
echo "Run ID: $RUN_ID"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: $GPU"
echo "Config: $CONFIG"
echo "Output dir: $OUT_DIR"
echo "Log file: $LOGFILE"
echo "============================================"

echo ""
echo "============================================"
echo "Stage 1/2: Base training"
echo "============================================"
"$PYTHON" train_base.py \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --output_dir "$BASE_DIR" \
    --recon_ckpt "$RECON_CKPT" \
    --depth_ckpt "$DEPTH_CKPT" \
    --gpu 0 \
    --num_base_classes 20

echo ""
echo "============================================"
echo "Stage 2/2: Incremental training"
echo "============================================"
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
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output dir: $OUT_DIR"
echo "Log file: $LOGFILE"
echo "Results:"
echo "============================================"
if [ -f "$INC_DIR/results.yaml" ]; then
    cat "$INC_DIR/results.yaml"
else
    echo "Incremental results not found: $INC_DIR/results.yaml"
fi
