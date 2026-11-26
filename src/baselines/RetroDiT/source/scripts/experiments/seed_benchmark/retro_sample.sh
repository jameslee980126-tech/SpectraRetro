#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed is required}"
project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
pytorch_python="${PYTORCH_PYTHON:-python}"
retro_data_root="${RETRO_DATA_ROOT:-${project_root}/data}"
benchmark_root="${SEED_BENCHMARK_ROOT:-${project_root}/outputs/seed_benchmark}"
train_dir="${benchmark_root}/retro/seed_${seed}/train"
sample_dir="${benchmark_root}/retro/seed_${seed}/sample"
rc_dir="${benchmark_root}/rc/seed_${seed}/predictions/test"

shopt -s nullglob
checkpoints=("${train_dir}"/checkpoints/best_top3acc_model_*.ckpt)
rc_predictions=("${rc_dir}"/*strict_root_center_top7_root_top7_temp1p25_agg_split_edge.json)
if [[ "${#checkpoints[@]}" -ne 1 ]]; then
  echo "expected exactly one top-3 checkpoint under ${train_dir}, found ${#checkpoints[@]}" >&2
  exit 1
fi
if [[ "${#rc_predictions[@]}" -ne 1 ]]; then
  echo "expected exactly one RC test prediction under ${rc_dir}, found ${#rc_predictions[@]}" >&2
  exit 1
fi

cd "${project_root}"
export PROJECT_ROOT="${project_root}"
run_started_at="$(date +%s)"
set +e
"${pytorch_python}" -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node 8 \
  -m src.eval \
  experiment=retro \
  data=uspto_50k \
  +model/sde=dfm \
  +model/net=graph_dit/8m \
  data.n_dummy=10 \
  "seed=${seed}" \
  "task_name=retrodit_uspto50k_seed${seed}_pred_rc" \
  "paths.data_dir=${retro_data_root}" \
  "hydra.run.dir=${sample_dir}" \
  "ckpt_path=${checkpoints[0]}" \
  "data.external_perm_paths.test=${rc_predictions[0]}" \
  data.forward_syn_model=null \
  '~logger'
torchrun_status=$?
set -e

completion_marker="${sample_dir}/EVALUATION_COMPLETE"
if [[ "${torchrun_status}" -ne 0 && -f "${completion_marker}" ]]; then
  marker_time="$(stat -c %Y "${completion_marker}")"
  if [[ "${marker_time}" -ge "${run_started_at}" ]]; then
    echo "torchrun returned ${torchrun_status} after successful evaluation teardown; accepting completion marker" >&2
    exit 0
  fi
fi
exit "${torchrun_status}"
