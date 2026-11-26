# S3. Size prediction and conditional diffusion

[Table S4](tables/table_s4.tex)

SpectraRetro conditions on a 3D synthon to generate the missing part of a precursor. In the full benchmark it samples 300 times per fragment with 100 reverse diffusion intervals. Table S4 specifies the size-model and diffusion settings. [Benchmark inputs and evaluation](../../../docs/guides/evaluation.md)

| Paper item or input | What it contains | Files |
|---|---|---|
| Table S4 | The two class modes have separate model and evaluation configurations. | [known class](configs/known_full.json), [unknown class](configs/unknown_full.json), [Table S4](tables/table_s4.tex) |
| Model selection | Epoch metrics and selection files identify which completion parameters entered each evaluation. Size-model curves are supplied separately. | [known-class selection](results/training/known/completion/selection.json), [unknown-class selection](results/training/unknown/completion/selection.json), [known size curve](results/training/known/size/curves.jsonl), [unknown size curve](results/training/unknown/size/curves.jsonl), [parameter identities](models/README.md) |
| Evaluation inputs | The shared geometry data and mode-specific evaluation directories contain the inputs used for sampling and ranking. | [SpectraRetro experiment inputs](../../spectraretro_inputs/README.md), [known-class evaluation inputs](../../section_4_results/4_1_benchmark/known-class/data/evaluation), [unknown-class evaluation inputs](../../section_4_results/4_1_benchmark/unknown-class/data/evaluation) |

[Complete file list, sources and SHA-256 digests](artifacts.json).

The [known-class](../../../src/spectraretro/known-class/) and [unknown-class](../../../src/spectraretro/unknown-class/) model directories each include the conditioning and ranking code for that setting. [Shared diffusion and decoding](../../../src/spectraretro/completion/) · [§S3.6 ranking parameters and feature definitions](../../../src/spectraretro/fusion_reranker/manifest.json).

## Reproduce

Inspect the training configuration for either class setting.

```bash
python src/spectraretro/known-class/run.py train --check-only
python src/spectraretro/unknown-class/run.py train --check-only
```
