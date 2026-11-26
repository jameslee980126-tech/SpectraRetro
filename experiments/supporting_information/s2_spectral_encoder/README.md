# S2. Full-spectrum reaction-center encoder

[Table S3](tables/table_s3.tex)

The encoder adapts the bivariate spectral filter of [FSpecGNN](https://proceedings.mlr.press/v306/wang26m.html) for reaction-center scoring (SI §S2). Table S3 defines the full-data spectral ablation. Use the model configurations and learning curves to inspect each variant. The small-data local propagation comparison reports three seeds separately.

| Paper item or input | What it contains | Files |
|---|---|---|
| Full-data comparison | The encoder and its no-spectral control use the settings recorded for the full ablation. | [training settings](../../../results/reaction_center/full_spectral_ablation/configuration.json), [full-spectrum curve](../../../results/reaction_center/full_spectral_ablation/training_curves/full_fspec.jsonl), [no-spectral curve](../../../results/reaction_center/full_spectral_ablation/training_curves/no_spectral.jsonl), [Table S3](tables/table_s3.tex) |
| Small-data comparison | The local-pair control has a separate configuration and a curve for each seed. | [local-pair settings](../../../results/reaction_center/small_local_propagation/configuration.json), [curves for seeds 0–2](../../../results/reaction_center/small_local_propagation/training_curves), [parameter roles](models/README.md) |

[Complete file list, sources and SHA-256 digests](artifacts.json).

[Known-class reaction-center predictor](../../../src/spectraretro/known-class/reaction_center/), [unknown-class reaction-center predictor](../../../src/spectraretro/unknown-class/reaction_center/), [shared graph encoder](../../../src/spectraretro/reaction_center/) and [local-pair and control implementations](reference_code).

## Reproduce

Recount reaction-center ranks from the saved predictions and check the seed summaries.

```bash
python results/reaction_center/reproduce.py --check
```
