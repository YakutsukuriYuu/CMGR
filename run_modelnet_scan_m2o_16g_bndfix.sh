#!/bin/bash
# Run the CMGR cross-domain ModelNet -> ScanObjectNN (M2O) experiment.
# Usage: bash run_modelnet_scan_m2o_16g_bndfix.sh [gpu_id] [scan_root]
# By default this reuses the existing 26-class ModelNet base checkpoint and
# only reruns M2O incremental training. Set SKIP_BASE_TRAIN=0 to retrain base.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/qiansu/main/clip/.venv/bin/python}"
CONFIG="configs/modelnet_scan_m2o_16g.yaml"
MODELNET_ROOT="${MODELNET_ROOT:-../data/ModelNet40_Align}"
SCAN_ROOT="${2:-${SCAN_ROOT:-../data/ScanObjectNN/main_split}}"
RECON_CKPT="${RECON_CKPT:-../deps/ReCon/pretrained/recon.pth}"
DEPTH_CKPT="${DEPTH_CKPT:-../deps/CLIP2Point/pretrained/vit32/best_eval.pth}"
SKIP_BASE_TRAIN="${SKIP_BASE_TRAIN:-1}"
REUSE_BASE_DIR="${REUSE_BASE_DIR:-outputs_modelnet_scan_m2o_16g_bndfix_20260609_153400/base}"

GPU="${1:-0}"
RUN_ID=$(date '+%Y%m%d_%H%M%S')
OUT_DIR="outputs_modelnet_scan_m2o_16g_bndfix_${RUN_ID}"
INC_DIR="${OUT_DIR}/incremental"
LOGFILE="modelnet_scan_m2o_16g_bndfix_${RUN_ID}.log"

if [ "$SKIP_BASE_TRAIN" = "1" ]; then
    BASE_DIR="${BASE_DIR:-$REUSE_BASE_DIR}"
else
    BASE_DIR="${BASE_DIR:-${OUT_DIR}/base}"
fi

exec > >(tee "$LOGFILE") 2>&1

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT_DIR"
cp -f "$CONFIG" "$OUT_DIR/config.yaml"

echo "============================================"
echo "CMGR ModelNet -> ScanObjectNN M2O Experiment"
echo "Run ID: $RUN_ID"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: $GPU"
echo "Config: $CONFIG"
echo "ModelNet root: $MODELNET_ROOT"
echo "ScanObjectNN root: $SCAN_ROOT"
echo "Output dir: $OUT_DIR"
echo "Base dir: $BASE_DIR"
echo "Skip base training: $SKIP_BASE_TRAIN"
echo "Log file: $LOGFILE"
echo "Protocol: 26 base + 3 incremental tasks (4, 4, 3)"
echo "============================================"

echo ""
if [ "$SKIP_BASE_TRAIN" = "1" ]; then
    echo "============================================"
    echo "Stage 1/2: Skipping base training"
    echo "============================================"
    if [ ! -f "$BASE_DIR/netB_best.pth" ] && [ ! -f "$BASE_DIR/netB_final.pth" ]; then
        echo "Missing base checkpoint in: $BASE_DIR"
        echo "Set REUSE_BASE_DIR=/path/to/base or SKIP_BASE_TRAIN=0 to retrain base."
        exit 1
    fi
    if [ ! -f "$BASE_DIR/exemplars.pth" ]; then
        echo "Missing base exemplars: $BASE_DIR/exemplars.pth"
        echo "Set REUSE_BASE_DIR=/path/to/base or SKIP_BASE_TRAIN=0 to retrain base."
        exit 1
    fi
else
    echo "============================================"
    echo "Stage 1/2: Base training on ModelNet"
    echo "============================================"
    "$PYTHON" train_base.py \
        --config "$CONFIG" \
        --data_root "$MODELNET_ROOT" \
        --output_dir "$BASE_DIR" \
        --recon_ckpt "$RECON_CKPT" \
        --depth_ckpt "$DEPTH_CKPT" \
        --gpu 0 \
        --num_base_classes 26
fi

echo ""
echo "============================================"
echo "Stage 2/2: Incremental training on ScanObjectNN"
echo "============================================"
"$PYTHON" train_incremental.py \
    --config "$CONFIG" \
    --data_root "$MODELNET_ROOT" \
    --train_data_root "$MODELNET_ROOT" \
    --test_data_root "$SCAN_ROOT" \
    --train_dataset modelnet \
    --test_dataset scanobjectnn \
    --scan_variant OBJ_BG \
    --base_dir "$BASE_DIR" \
    --output_dir "$INC_DIR" \
    --recon_ckpt "$RECON_CKPT" \
    --depth_ckpt "$DEPTH_CKPT" \
    --gpu 0 \
    --num_base_classes 26 \
    --task_splits 4,4,3

echo ""
echo "============================================"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output dir: $OUT_DIR"
echo "Log file: $LOGFILE"
echo "Results: $INC_DIR/results.yaml"
echo "============================================"
if [ -f "$INC_DIR/results.yaml" ]; then
    cat "$INC_DIR/results.yaml"
else
    echo "Incremental results not found: $INC_DIR/results.yaml"
fi
