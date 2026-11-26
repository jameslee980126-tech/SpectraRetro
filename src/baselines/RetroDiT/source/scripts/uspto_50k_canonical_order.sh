#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: PROJECT_ROOT=/path/to/repo CONDA_PATH=/path/to/miniconda3 NODE_COUNT=... NODE_RANK=... PROC_PER_NODE=... MASTER_ADDR=... [MASTER_PORT=6000] bash scripts/uspto_50k_canonical_order.sh [hydra_overrides...]"
}

required_vars=(PROJECT_ROOT CONDA_PATH NODE_COUNT NODE_RANK PROC_PER_NODE MASTER_ADDR)
for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required env: ${var_name}"
    usage
    exit 1
  fi
done

MASTER_PORT="${MASTER_PORT:-6000}"
. "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate retrodit
cd "${PROJECT_ROOT}"


model_sizes=("65m" "8m" "2m" "280k")

for model_size in "${model_sizes[@]}"; do
    task_name=dit${model_size}_uspto50k_cano
    args=(
        experiment=retro
        data=uspto_50k
        data.perm_prob_sampler=null
        data.perm_types=null
        +model/sde=dfm
        +model/net=graph_dit/${model_size}
        task_name=${task_name}
        paths.log_dir=\${paths.root_dir}/logs/grid_model_size/dit
        trainer.num_nodes=$NODE_COUNT
    )

    torchrun \
        --nnodes "${NODE_COUNT}" \
        --node_rank "${NODE_RANK}" \
        --nproc_per_node "${PROC_PER_NODE}" \
        --master_addr "${MASTER_ADDR}" \
        --master_port "${MASTER_PORT}" \
        -m src.train "${args[@]}" "$@"

done

