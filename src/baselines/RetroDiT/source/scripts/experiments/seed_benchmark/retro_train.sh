#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed is required}"
project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
pytorch_python="${PYTORCH_PYTHON:-python}"
retro_data_root="${RETRO_DATA_ROOT:-${project_root}/data}"
run_dir="${SEED_BENCHMARK_ROOT:-${project_root}/outputs/seed_benchmark}/retro/seed_${seed}/train"

cd "${project_root}"
export PROJECT_ROOT="${project_root}"
run_started_at="$(date +%s)"
set +e
"${pytorch_python}" -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node 8 \
  -m src.train \
  experiment=retro \
  data=uspto_50k \
  +model/sde=dfm \
  +model/net=graph_dit/8m \
  data.n_dummy=10 \
  "seed=${seed}" \
  "task_name=retrodit_uspto50k_seed${seed}" \
  "paths.data_dir=${retro_data_root}" \
  "hydra.run.dir=${run_dir}" \
  data.forward_syn_model=null \
  callbacks.model_checkpoint_top3.save_top_k=1 \
  test=false \
  '~logger'
torchrun_status=$?
set -e

completion_marker="${run_dir}/TRAINING_COMPLETE"
if [[ "${torchrun_status}" -ne 0 && -f "${completion_marker}" ]]; then
  marker_time="$(stat -c %Y "${completion_marker}")"
  if [[ "${marker_time}" -ge "${run_started_at}" ]]; then
    echo "torchrun returned ${torchrun_status} after successful training teardown; accepting completion marker" >&2
    exit 0
  fi
fi
exit "${torchrun_status}"
