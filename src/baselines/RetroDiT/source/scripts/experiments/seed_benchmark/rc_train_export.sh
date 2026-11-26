#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed is required}"
project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
torchdrug_python="${TORCHDRUG_PYTHON:-python}"
data_root="${TORCHDRUG_DATA_ROOT:-${project_root}/rc_prediction/data/processed}"
output_dir="${SEED_BENCHMARK_ROOT:-${project_root}/outputs/seed_benchmark}/rc/seed_${seed}"

cd "${project_root}/rc_prediction"
export PYTHONPATH="${project_root}/rc_prediction/src${PYTHONPATH:+:${PYTHONPATH}}"

"${torchdrug_python}" -m torchdrug_pred.cli train_eval_export \
  --dataset 50k \
  --train-path "${data_root}/train.pt" \
  --valid-path "${data_root}/valid.pt" \
  --test-path "${data_root}/test.pt" \
  --output-dir "${output_dir}" \
  --hidden-dims 256 256 256 256 256 256 \
  --learning-rate 0.001 \
  --batch-size 64 \
  --epochs 20 \
  --seed "${seed}" \
  --gpu 0 \
  --objective single_positive \
  --center-top-k 7 \
  --strict-root-top-k 7 \
  --export-policies strict_root \
  --temperature 1.25 \
  --root-aggregation split_edge \
  --prediction-prefix "retrodit_rc_seed${seed}" \
  --export-splits test \
  --selection-center-top-k 5 \
  --selection-root-limit 5 \
  --selection-temperature 1.0
