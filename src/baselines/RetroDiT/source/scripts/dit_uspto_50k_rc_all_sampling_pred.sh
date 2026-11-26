#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: PROJECT_ROOT=/path/to/repo CONDA_PATH=/path/to/miniconda3 NODE_COUNT=... NODE_RANK=... PROC_PER_NODE=... MASTER_ADDR=... CKPT_PATH=/path/to.ckpt [MASTER_PORT=6000] [SAMPLING_STEPS=50] bash scripts/dit_uspto_50k_rc_all_sampling_pred.sh [hydra_overrides...]"
}

required_vars=(PROJECT_ROOT CONDA_PATH NODE_COUNT NODE_RANK PROC_PER_NODE MASTER_ADDR CKPT_PATH)
for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required env: ${var_name}"
    usage
    exit 1
  fi
done

MASTER_PORT="${MASTER_PORT:-6000}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
. "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate retrodit
cd "${PROJECT_ROOT}"


sampling_steps="${SAMPLING_STEPS}"

task_name=pred_rc_uspto50k_step${sampling_steps}
args=(
    experiment=retro
    data=uspto_50k
    data.perm_types=data/uspto_50k/test_json/pred_rc_uspto50k.json
    +model/sde=dfm
    model.sde.test_sampling_steps=${sampling_steps}
    +model/net=graph_dit/8m
    ckpt_path=${CKPT_PATH}
    task_name=${task_name}
    paths.log_dir=\${paths.root_dir}/logs/sampling/uspto_50k_sota_pred
    trainer.num_nodes=$NODE_COUNT
    ~logger
)
torchrun \
    --nnodes "${NODE_COUNT}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${PROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    -m src.eval "${args[@]}" "$@"





