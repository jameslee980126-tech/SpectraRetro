# Installation

## Read the saved results

Use Git LFS to [download the complete repository](../../README.md#download-the-complete-repository), including data and weights. The saved-result workflows in the [reproduction guide](../guides/reproduction.md) use Python 3.10 or later and the standard library.

## Run the model on two reactions

Use a 64-bit CPython 3.10 environment on Windows or Linux x86-64. The following commands install the CPU builds of PyTorch and its compiled graph extensions. Run them in the repository root. Python 3.10 must be the active `python` when creating the environment.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` in Bash, then install the dependencies.

```bash
python -m pip install --upgrade pip
python -m pip install "torch==2.8.0+cpu" "torchvision==0.23.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=:all: --find-links https://data.pyg.org/whl/torch-2.8.0+cpu.html "torch-scatter==2.1.2+pt28cpu" "torch-sparse==0.6.18+pt28cpu" "torch-cluster==1.6.3+pt28cpu"
python -m pip install -r requirements-model.txt -r requirements-decode.txt "numpy<2.1" decorator jinja2
python -m pip check
```

TorchDrug's molecular modules are included under `src/spectraretro/reaction_center/vendor/stage1/torchdrug`. Model execution from the checkout uses this version. The decoder uses Open Babel's XYZ reader and SMILES writer. Check their availability before launching the two-reaction run.

```bash
python -c "import torch, torch_scatter, torch_sparse, torch_cluster; from openbabel import openbabel as ob; assert ob.OBConversion().SetInAndOutFormats('xyz', 'smi'); print(torch.__version__)"
python src/spectraretro/unknown-class/run.py --smoke
```

Use `python src/spectraretro/known-class/run.py --smoke` to supply the reaction class. Each run writes a new directory under `outputs/`. Open `evaluation/ranked_candidates.jsonl` inside it to read the final candidates for each of the two reactions.

## Full benchmark and training

For full generation or training, use a compatible NVIDIA driver and the CUDA builds of PyTorch and the graph extensions. For example, in a new Python 3.10 environment, the CUDA 12.8 counterparts of the two CPU install commands above are:

```bash
python -m pip install "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install --only-binary=:all: --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html "torch-scatter==2.1.2+pt28cu128" "torch-sparse==0.6.18+pt28cu128" "torch-cluster==1.6.3+pt28cu128"
```

Install the remaining dependencies as above, then check `python -c "import torch; assert torch.cuda.is_available()"`. Run the full pretrained workflow with `python src/spectraretro/unknown-class/run.py` or `python src/spectraretro/known-class/run.py`. [PyTorch binary versions](https://pytorch.org/get-started/previous-versions/) · [PyG compiled extension wheels](https://data.pyg.org/whl/) · [Training and evaluation guide](../guides/reproduction.md).

The launcher defaults to the active interpreter for all four steps. Use `--center-python`, `--model-python`, `--decode-python` and `--score-python` when those steps have separate environments. The reference coordinate decoder used Windows Open Babel 3.1.1.22. Operating system and Open Babel build can affect bond-order and aromaticity perception. Saved reference strings are supplied for comparison.

## Inspect CHRetroAgent routes offline

The viewer requires RDKit. Run it in the model environment above or in the environment supplied by `environment-cpu.yml`. The following command exports the five-step lapatinib route from Figure 5.

```bash
python CHRetroAgent/preview.py --report results/cases/agent/case_l/result.json --export outputs/lapatinib.html
```

The [CHRetroAgent guide](../../CHRetroAgent/README.md) links the saved routes and assessments for lapatinib, nemiralisib and tafamidis. Baseline implementations have their own environment specifications under `src/baselines/`. Quantum calculation settings, software versions and input files accompany the [SI §S7 experiments](../../experiments/supporting_information/s7_mechanistic_analyses/README.md).
