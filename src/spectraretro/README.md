# SpectraRetro single-step model

SpectraRetro uses full-spectrum atom-pair learning to predict reaction centers on the product graph and chemical-prior-guided conditional 3D diffusion to reconstruct reactants from synthons. It assembles the completed fragments into precursor sets and ranks them by a model score that incorporates candidate frequency. [Prediction pipeline](pipeline/runner.py).

## Reproduce the §4.1 benchmark

Fetch the Git LFS data and weights and install the model dependencies. From the repository root, run the command for your class setting. Both use the same 9,250 test reactions. [Installation](../../docs/getting_started/installation.md) · [Known-class source](known-class/) · [Unknown-class source](unknown-class/).

```bash
# Reaction class supplied
python src/spectraretro/known-class/run.py

# Reaction class withheld
python src/spectraretro/unknown-class/run.py
```

Append `--smoke` for a two-reaction example. In the run directory under `outputs/`, `evaluation/ranked_candidates.jsonl` contains each reaction’s ranked precursor sets and `RESULTS.md` reports Top-k accuracy. [Prepared inputs](../../experiments/spectraretro_inputs/README.md) · [Benchmark inputs and scoring](../../docs/guides/evaluation.md) · [Recorded benchmark results](../../experiments/section_4_results/4_1_benchmark/README.md).
