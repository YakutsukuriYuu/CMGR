#!/bin/bash
# Run ModelNet40 M2M ablation variants for SAGR/TAM/BND.
#
# Default mode reuses the existing 20-class ModelNet base checkpoint and only
# reruns incremental training for each variant. For a full pipeline ablation,
# set SKIP_BASE_TRAIN=0 to retrain base for every variant.
#
# Usage:
#   bash run_modelnet_ablation_m2m_16g.sh [gpu_id]
#   VARIANT_LIST="baseline full" bash run_modelnet_ablation_m2m_16g.sh 0
#   SKIP_BASE_TRAIN=0 bash run_modelnet_ablation_m2m_16g.sh 0

set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/qiansu/main/clip/.venv/bin/python}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/modelnet_ablation_m2m_16g.yaml}"
DATA_ROOT="${DATA_ROOT:-../data/ModelNet40_Align}"
RECON_CKPT="${RECON_CKPT:-../deps/ReCon/pretrained/recon.pth}"
DEPTH_CKPT="${DEPTH_CKPT:-../deps/CLIP2Point/pretrained/vit32/best_eval.pth}"
REUSE_BASE_DIR="${REUSE_BASE_DIR:-outputs_modelnet_full_16g_bndfix_20260530_172453/base}"
SKIP_BASE_TRAIN="${SKIP_BASE_TRAIN:-1}"

GPU=${1:-0}
if [ $# -ge 2 ]; then
    VARIANT_LIST="$2"
else
    VARIANT_LIST="${VARIANT_LIST:-baseline v1_sagr v2_tam v3_bnd v4_sagr_tam v5_sagr_bnd v6_tam_bnd full}"
fi

RUN_ID=$(date '+%Y%m%d_%H%M%S')
OUT_ROOT="outputs_modelnet_ablation_m2m_16g_${RUN_ID}"
CONFIG_DIR="${OUT_ROOT}/configs"
LOGFILE="modelnet_ablation_m2m_16g_${RUN_ID}.log"

mkdir -p "$CONFIG_DIR"

exec > >(tee "$LOGFILE") 2>&1

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

variant_flags() {
    case "$1" in
        baseline) echo "0 0 0" ;;
        v1_sagr) echo "1 0 0" ;;
        v2_tam) echo "0 1 0" ;;
        v3_bnd) echo "0 0 1" ;;
        v4_sagr_tam) echo "1 1 0" ;;
        v5_sagr_bnd) echo "1 0 1" ;;
        v6_tam_bnd) echo "0 1 1" ;;
        full) echo "1 1 1" ;;
        *)
            echo "Unknown variant: $1" >&2
            exit 1
            ;;
    esac
}

write_variant_config() {
    local variant="$1"
    local use_sagr="$2"
    local use_tam="$3"
    local use_bnd="$4"
    local output_config="$5"

    "$PYTHON" - "$CONFIG_TEMPLATE" "$output_config" "$variant" \
        "$use_sagr" "$use_tam" "$use_bnd" <<'PY'
import sys
import yaml

template_path, output_path, variant, use_sagr, use_tam, use_bnd = sys.argv[1:]
with open(template_path, 'r') as f:
    config = yaml.safe_load(f)

config['ablation_variant'] = variant
config['use_sagr'] = use_sagr == '1'
config['use_tam'] = use_tam == '1'
config['use_bnd'] = use_bnd == '1'

with open(output_path, 'w') as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY
}

echo "============================================"
echo "CMGR ModelNet40 M2M Ablation"
echo "Run ID: $RUN_ID"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: $GPU"
echo "Template config: $CONFIG_TEMPLATE"
echo "Data root: $DATA_ROOT"
echo "Output root: $OUT_ROOT"
echo "Log file: $LOGFILE"
echo "Skip base training: $SKIP_BASE_TRAIN"
echo "Reuse base dir: $REUSE_BASE_DIR"
echo "Variants: $VARIANT_LIST"
echo "============================================"

if [ "$SKIP_BASE_TRAIN" = "1" ]; then
    if [ ! -f "$REUSE_BASE_DIR/netB_best.pth" ] && [ ! -f "$REUSE_BASE_DIR/netB_final.pth" ]; then
        echo "Missing reusable base checkpoint in $REUSE_BASE_DIR" >&2
        exit 1
    fi
    if [ ! -f "$REUSE_BASE_DIR/exemplars.pth" ]; then
        echo "Missing reusable exemplars in $REUSE_BASE_DIR" >&2
        exit 1
    fi
fi

for VARIANT in $VARIANT_LIST; do
    read -r USE_SAGR USE_TAM USE_BND <<< "$(variant_flags "$VARIANT")"
    VARIANT_DIR="${OUT_ROOT}/${VARIANT}"
    VARIANT_CONFIG="${CONFIG_DIR}/${VARIANT}.yaml"
    INC_DIR="${VARIANT_DIR}/incremental"

    if [ "$SKIP_BASE_TRAIN" = "1" ]; then
        BASE_DIR="$REUSE_BASE_DIR"
    else
        BASE_DIR="${VARIANT_DIR}/base"
    fi

    mkdir -p "$VARIANT_DIR" "$INC_DIR"
    write_variant_config "$VARIANT" "$USE_SAGR" "$USE_TAM" "$USE_BND" "$VARIANT_CONFIG"
    cp "$VARIANT_CONFIG" "${VARIANT_DIR}/config.yaml"

    echo ""
    echo "============================================"
    echo "Variant: $VARIANT"
    echo "SAGR=$USE_SAGR TAM=$USE_TAM BND=$USE_BND"
    echo "Config: $VARIANT_CONFIG"
    echo "Base dir: $BASE_DIR"
    echo "Incremental dir: $INC_DIR"
    echo "============================================"

    if [ "$SKIP_BASE_TRAIN" = "1" ]; then
        echo "Stage 1/2: Base training skipped"
    else
        echo "Stage 1/2: Base training"
        "$PYTHON" train_base.py \
            --config "$VARIANT_CONFIG" \
            --data_root "$DATA_ROOT" \
            --output_dir "$BASE_DIR" \
            --recon_ckpt "$RECON_CKPT" \
            --depth_ckpt "$DEPTH_CKPT" \
            --gpu 0 \
            --num_base_classes 20
    fi

    echo "Stage 2/2: Incremental training"
    "$PYTHON" train_incremental.py \
        --config "$VARIANT_CONFIG" \
        --data_root "$DATA_ROOT" \
        --base_dir "$BASE_DIR" \
        --output_dir "$INC_DIR" \
        --recon_ckpt "$RECON_CKPT" \
        --depth_ckpt "$DEPTH_CKPT" \
        --gpu 0 \
        --num_base_classes 20 \
        --classes_per_task 5

    echo "Variant $VARIANT results:"
    if [ -f "$INC_DIR/results.yaml" ]; then
        cat "$INC_DIR/results.yaml"
    else
        echo "Incremental results not found: $INC_DIR/results.yaml"
    fi
done

echo ""
echo "============================================"
echo "Ablation summary"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output root: $OUT_ROOT"
echo "Log file: $LOGFILE"
echo "============================================"

"$PYTHON" - "$OUT_ROOT" $VARIANT_LIST <<'PY'
import os
import sys
import yaml

out_root = sys.argv[1]
variants = sys.argv[2:]
print("| Variant | SAGR | TAM | BND | Base | Task1 | Task2 | Task3 | Task4 | Final | AA | Delta A |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for variant in variants:
    result_path = os.path.join(out_root, variant, 'incremental', 'results.yaml')
    if not os.path.exists(result_path):
        print(f"| {variant} | - | - | - | missing | missing | missing | missing | missing | missing | missing | missing |")
        continue
    with open(result_path, 'r') as f:
        result = yaml.safe_load(f)
    protocol = result.get('protocol', {})
    acc = result.get('accuracies', [])
    acc_pct = [f"{value * 100:.2f}" for value in acc]
    while len(acc_pct) < 5:
        acc_pct.append('-')
    final = result.get('final_accuracy')
    aa = result.get('AA')
    delta = result.get('delta_A')
    print(
        f"| {variant} | {int(bool(protocol.get('use_sagr')))} | "
        f"{int(bool(protocol.get('use_tam')))} | "
        f"{int(bool(protocol.get('use_bnd')))} | "
        f"{acc_pct[0]} | {acc_pct[1]} | {acc_pct[2]} | {acc_pct[3]} | {acc_pct[4]} | "
        f"{final * 100:.2f} | {aa * 100:.2f} | {delta * 100:.2f} |"
    )
PY
