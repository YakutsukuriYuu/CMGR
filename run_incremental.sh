#!/bin/bash
# 只跑增量训练（需要先有 outputs_v3/base 目录）

set -e
cd "$(dirname "$0")"

LOGFILE="incremental.log"
exec > >(tee "$LOGFILE") 2>&1

echo "============================================"
echo "CMGR Incremental Training"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=${1:-1} /home/qiansu/main/clip/.venv/bin/python train_incremental.py \
    --data_root ../data/ModelNet40_Align \
    --base_dir outputs_v3/base \
    --output_dir outputs_v3/incremental_kd \
    --recon_ckpt ../deps/ReCon/pretrained/recon.pth \
    --depth_ckpt ../deps/CLIP2Point/pretrained/vit32/best_eval.pth \
    --gpu 0 \
    --num_base_classes 20 \
    --classes_per_task 5

echo ""
echo "============================================"
echo "Results:"
echo "============================================"
if [ -f outputs_v3/incremental_kd/results.yaml ]; then
    cat outputs_v3/incremental_kd/results.yaml
else
    echo "(incremental failed)"
fi
