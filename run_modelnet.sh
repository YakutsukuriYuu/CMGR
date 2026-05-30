#!/bin/bash
# CMGR ModelNet40 完整实验脚本
# Base training (20 classes) + Incremental training (4 tasks x 5 classes)
# 用法: bash run_modelnet.sh [gpu_id]  (默认 GPU 1)
# 输出重定向到 modelnet.log（同时打印到终端）

set -e

cd "$(dirname "$0")"

LOGFILE="modelnet.log"
exec > >(tee "$LOGFILE") 2>&1

PYTHON="/home/qiansu/main/clip/.venv/bin/python"
DATA_ROOT="../data/ModelNet40_Align"
RECON_CKPT="../deps/ReCon/pretrained/recon.pth"
DEPTH_CKPT="../deps/CLIP2Point/pretrained/vit32/best_eval.pth"
GPU=${1:-1}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "CMGR ModelNet40 Experiment"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log:     $LOGFILE"
echo "============================================"
echo "Config:    batch_size=8, grad_accum_steps=2, effective_batch=16, num_views=10"
echo "           base_epochs=100 max with early stopping, base_lr=5e-4, beta=0.1"
echo "           inc_epochs=50 max with early stopping, inc_lr=1e-3 -> 1e-4"
echo "           classification: geo_sim only (training), +clip_sim (inference)"
echo "Data root:    $DATA_ROOT"
echo "GPU:          $GPU"
echo "ReCon ckpt:   $RECON_CKPT"
echo "Depth ckpt:   $DEPTH_CKPT"
echo "============================================"

# Step 1: Base training
echo ""
echo ">>> Step 1/2: Base Training (20 base classes, max 100 epochs with early stopping)"
echo ""
$PYTHON train_base.py \
    --data_root "$DATA_ROOT" \
    --output_dir outputs_v3/base \
    --recon_ckpt "$RECON_CKPT" \
    --depth_ckpt "$DEPTH_CKPT" \
    --gpu 0 \
    --num_base_classes 20

# Step 2: Incremental training
echo ""
echo ">>> Step 2/2: Incremental Training (4 tasks x 5 novel classes, 30 epochs each)"
echo ""
$PYTHON train_incremental.py \
    --data_root "$DATA_ROOT" \
    --base_dir outputs_v3/base \
    --output_dir outputs_v3/incremental \
    --recon_ckpt "$RECON_CKPT" \
    --depth_ckpt "$DEPTH_CKPT" \
    --gpu 0 \
    --num_base_classes 20 \
    --classes_per_task 5

echo ""
echo "============================================"
echo "Experiment Complete!"
echo "============================================"
echo "Results: outputs_v3/incremental/results.yaml"
if [ -f outputs_v3/incremental/results.yaml ]; then
    cat outputs_v3/incremental/results.yaml
else
    echo "(incremental may have failed, check outputs_v3/base/)"
    if [ -f outputs_v3/base/best_acc.yaml ]; then
        cat outputs_v3/base/best_acc.yaml
    fi
fi
