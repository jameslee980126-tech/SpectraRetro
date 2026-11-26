#!/usr/bin/env bash
set -euo pipefail

env_name="${1:-retrodit}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found in PATH"
  exit 1
fi

conda create -n "${env_name}" python==3.11.4 -y
eval "$(conda shell.bash hook)"
conda activate "${env_name}"

pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 torchtext==0.18.0
pip install lightning==2.5.2
pip install hydra-core==1.3.2
pip install omegaconf==2.3.0
pip install rich==13.9.4

pip install tensorboard wandb torchmetrics numpy pandas matplotlib seaborn scikit-learn openpyxl

pip install tqdm opencv-python notebook jupyter ipython ipykernel imageio easydict

pip install rootutils

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.1+cu124.html
pip install torch_geometric rdkit entmax

pip install transformers==4.55.0 diffusers==0.34.0

pip install pathos

if [[ "${INSTALL_SYSTEM_PACKAGES:-0}" == "1" ]]; then
  sudo apt-get install libxrender1 libxext6 libsm6 libxrandr2 libfontconfig1 libice6 -y
  sudo apt-get install ttf-mscorefonts-installer
  sudo apt-get install fonts-liberation
fi

pip install flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl --no-build-isolation

