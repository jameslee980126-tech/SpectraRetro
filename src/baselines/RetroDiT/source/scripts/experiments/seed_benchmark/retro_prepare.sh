#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
pytorch_python="${PYTORCH_PYTHON:-python}"
retro_data_root="${RETRO_DATA_ROOT:-${project_root}/data}"

cd "${project_root}"
export PROJECT_ROOT="${project_root}"
"${pytorch_python}" -m src.setup_dataset \
  experiment=retro \
  data=uspto_50k \
  +model/sde=dfm \
  +model/net=graph_dit/8m \
  data.n_dummy=10 \
  "paths.data_dir=${retro_data_root}" \
  data.forward_syn_model=null \
  '~logger'
