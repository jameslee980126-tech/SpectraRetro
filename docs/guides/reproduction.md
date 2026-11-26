# Reproducing the SpectraRetro benchmark

Run these commands from the repository root after downloading the data and weights with Git LFS.

## Check the data and recount the results

```bash
python scripts/verify_release.py
python scripts/verify_paper_materials.py
python scripts/reproduce_results.py --check
```

The first two commands check the distributed files and their links to the paper. The third recounts the available reaction-level outcomes and seed summaries and writes tables to `outputs/reproduced_results/`.

## Predict precursor sets

Install the [model dependencies](../getting_started/installation.md). For a two-reaction run, add `--smoke` to either command below. For the full 9,250-reaction benchmark, use CUDA and run the command for the desired setting.

```bash
python src/spectraretro/unknown-class/run.py
python src/spectraretro/known-class/run.py
```

Known-class includes the reaction class as a model input. Unknown-class withholds it. Each run writes ranked precursors to `evaluation/ranked_candidates.jsonl` and Top-k accuracy to `RESULTS.md` in its output directory. Use `--check-only` to validate the setup. [Evaluation protocol](evaluation.md) · [Model inputs and outputs](../../src/spectraretro/README.md)

## Train and evaluate

```bash
python src/spectraretro/unknown-class/run.py train
python src/spectraretro/known-class/run.py train
```

Training selects reaction-center and completion weights using the validation split, then runs candidate generation and scoring. The mode-specific configurations specify the model settings, checkpoints and fragment sizes. To select separate Python environments for prediction, generation, decoding and scoring, use `--center-python`, `--model-python`, `--decode-python` and `--score-python`. [Unknown-class configuration](../../experiments/section_4_results/4_1_benchmark/unknown-class/configs/full.json) · [Known-class configuration](../../experiments/section_4_results/4_1_benchmark/known-class/configs/full.json)
