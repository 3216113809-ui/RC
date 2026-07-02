#!/bin/bash
# ================================================================
# RCIL 消融实验 —— 统一定义运行脚本
#
# 用法:
#   bash run_all.sh voc 15-1 0,1 A      # 只运行 A 组 (RC模块对照)
#   bash run_all.sh voc 15-1 0,1 all    # 运行全部
#   bash run_all.sh voc 15-1 0,1 B,C    # 运行 B 和 C 组
#
# 对照分组:
#   A = RC模块:     rcil_full vs no_rc
#   B = PCD蒸馏:    rcil_full vs no_pcd
#   C = 冻结分支:   rcil_full vs no_frozen
#   D = DropPath:   η=0.0 vs η=0.5 vs η=1.0
#   E = PCD模式:    skd_only vs ckd_only vs rcil_full
#   F = 下界:       finetune vs rcil_full
# ================================================================

set -e

DATASET=${1:-"voc"}
SETTING=${2:-"15-1"}
GPU=${3:-"0,1"}
GROUPS=${4:-"all"}

# -------- 分组 → 实验名映射 --------
declare -A GROUP_MAP
GROUP_MAP["A"]="rcil_full no_rc"
GROUP_MAP["B"]="rcil_full no_pcd"
GROUP_MAP["C"]="rcil_full no_frozen"
GROUP_MAP["D"]="droppath_0.0 droppath_0.5 droppath_1.0"
GROUP_MAP["E"]="skd_only ckd_only rcil_full"
GROUP_MAP["F"]="finetune rcil_full"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SAVE_DIR="${PROJECT_ROOT}/ablation_results/${DATASET}_${SETTING}"
NGPU=$(echo "${GPU}" | tr ',' '\n' | wc -l)
PORT=28000

# -------- 解析要运行的组 --------
if [ "${GROUPS}" = "all" ]; then
    RUN_GROUPS="A B C D E F"
else
    RUN_GROUPS=$(echo "${GROUPS}" | tr ',' ' ')
fi

echo "=============================================="
echo "  RCIL Ablation Study"
echo "  Dataset: ${DATASET}  Setting: ${SETTING}"
echo "  GPU: ${GPU}  Groups: ${RUN_GROUPS}"
echo "  Results: ${SAVE_DIR}"
echo "=============================================="

mkdir -p "${SAVE_DIR}"

# -------- 去重实验列表 (rcil_full 可能被多组引用) --------
declare -A SEEN
EXPERIMENTS=()
for g in ${RUN_GROUPS}; do
    for exp in ${GROUP_MAP[${g}]}; do
        if [ -z "${SEEN[${exp}]}" ]; then
            SEEN[${exp}]=1
            EXPERIMENTS+=("${exp}")
        fi
    done
done

echo ""
echo "  Experiments (${#EXPERIMENTS[@]}): ${EXPERIMENTS[*]}"
echo ""

# -------- 运行 --------
for exp in "${EXPERIMENTS[@]}"; do
    echo ">>> [${exp}] starting..."

    cd "${PROJECT_ROOT}"

    CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch \
        --master_port ${PORT} \
        --nproc_per_node ${NGPU} \
        run.py \
        --data_root ./data \
        --dataset ${DATASET} \
        --setting ${SETTING} \
        --ablation_exp ${exp} \
        --ablation_save_dir "${SAVE_DIR}"

    PORT=$((PORT + 1))
    echo "<<< [${exp}] done."
    echo ""
done

echo "=============================================="
echo "  All done! Results: ${SAVE_DIR}"
echo "=============================================="
