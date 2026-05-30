#!/bin/bash
# Run only the ModelNet40 base stage for quick base-accuracy iteration.
# Usage: bash run_base_modelnet_16g.sh [gpu_id]

set -e

cd "$(dirname "$0")"

PYTHON="/home/qiansu/main/clip/.venv/bin/python"
CONFIG="configs/modelnet_full_16g.yaml"
DATA_ROOT="../data/ModelNet40_Align"
RECON_CKPT="../deps/ReCon/pretrained/recon.pth"
DEPTH_CKPT="../deps/CLIP2Point/pretrained/vit32/best_eval.pth"
OUT_DIR="outputs_modelnet_base_fix/base"
GPU=${1:-0}

LOGFILE="modelnet_base_fix.log"
exec > >(tee "$LOGFILE") 2>&1

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "CMGR ModelNet40 Base-Only Experiment"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log:        $LOGFILE"
echo "Config:     $CONFIG"
echo "Data root:  $DATA_ROOT"
echo "GPU:        $GPU"
echo "Output:     $OUT_DIR"
echo "============================================"

"$PYTHON" train_base.py \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUT_DIR" \
    --recon_ckpt "$RECON_CKPT" \
    --depth_ckpt "$DEPTH_CKPT" \
    --gpu 0 \
    --num_base_classes 20

echo ""
echo "============================================"
echo "Base experiment complete"
echo "============================================"
echo "Result: $OUT_DIR/best_acc.yaml"
if [ -f "$OUT_DIR/best_acc.yaml" ]; then
    cat "$OUT_DIR/best_acc.yaml"
fi

