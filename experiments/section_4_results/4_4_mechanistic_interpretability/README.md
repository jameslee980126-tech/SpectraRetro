# Chemical, spatial and electronic context of model predictions

[Figure 4](figures/figure_4_mechanistic_context.png)

Figure 4 relates chemical-environment associations to controlled spatial and electronic perturbations. Panel a compares chemical descriptors and reaction-center scores. In panel b, 91 of 100 qualifying orientations agreed with the predicted steric response across 38 fragments and 22 scaffold clusters. Panel c shows a shift toward C2 in five of six thiazole N-oxidation pairs.

| Paper item or input | What it contains | Files |
|---|---|---|
| Chemical context | Association tables report chemical preferences and correlations between descriptors. | [chemical preferences](results/chemical_preferences.csv), [descriptor correlations](results/descriptor_correlations.csv), [factor definitions](data/factor_catalog.csv) |
| Spatial perturbation | Fragment-level records identify the scaffolds used to summarize model responses to changes in geometry. | [38 fragments](data/joined_38_fragments.csv), [orientation measurements](data/steric_orientations.csv), [leave-one-cluster-out summaries](results/steric_leave_one_cluster_out.csv) |
| Electronic perturbation | Use the six figure pairs, DFT contrasts and alternative-site controls to identify which response was measured on each structure. | [six figure pairs](data/main_figure_six_pairs.csv), [paired electronic contrasts](data/dft_paired_electronic_comparison.csv), [site controls](data/dft_alternative_N_controls.csv), [Figure 4](figures/figure_4_mechanistic_context.png) |

[Complete file list, sources and SHA-256 digests](artifacts.json).

[Saved-observation analysis](../../../scripts/reproduce_mechanistic_tables.py) and [quantum calculation code](../../../results/source_data/mechanistic_interpretability/supplement/quantum/calculation_code).

## Reproduce

Check the saved electronic contrasts, pair identities and spatial statistics.

```bash
python scripts/reproduce_mechanistic_tables.py
```
