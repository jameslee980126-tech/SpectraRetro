# CHRetro reaction families, similarity and molecular properties

[Figure 1](figures/figure_1_retrosynthetic_overview.png) · [Figure 2](figures/figure_2_chretro_dataset_overview.png)

CHRetro contains 92,791 reactions. Figure 2a uses a stratified sample of 12,000 reactions to draw its similarity landscape. Panels b–d summarize product properties across all reaction records. Panel e counts distinct products with fully specified elements.

| Paper item or input | What it contains | Files |
|---|---|---|
| Full and small partitions | The frozen reaction CSVs contain the reaction SMILES and family labels. | [full partitions](../../../datasets/full) and [small partitions](../../../datasets/small) |
| Figure 1 | Overview of C–H activation retrosynthesis, the single-step model and multistep route planning. | [task diagram](figures/figure_1_retrosynthetic_overview.png) |
| Figure 2a | The sampled reactions map to 11,565 fingerprint nodes and 11,564 tree edges. The coordinates locate those nodes in the figure. | [selected reactions](../../../results/supporting_information/source_data_figures/figure2_selected_reactions.csv), [nodes](../../../results/supporting_information/source_data_figures/figure2_nodes.csv), [edges](../../../results/supporting_information/source_data_figures/figure2_edges.csv), [coordinates](../../../results/supporting_information/source_data_figures/figure2_node_coordinates.csv) |
| Figure 2b–e | The reaction-weighted and unique-product tables use different denominators. Of 73,522 unique products, 73,521 have fully specified elements. | [reaction-weighted descriptors](../../../datasets/supplementary_data/figure3_reaction_weighted_descriptors.csv), [unique-product descriptors](../../../datasets/supplementary_data/figure3_unique_product_descriptors.csv), [summary statistics](../../../results/supporting_information/source_data_figures/figure3_property_summary.csv) |

<details>
<summary>Additional source records</summary>

- [`results/supporting_information/source_data_figures/tableS1_split_family_counts.csv`](../../../results/supporting_information/source_data_figures/tableS1_split_family_counts.csv)

</details>

[Complete file list, sources and SHA-256 digests](artifacts.json).

[Structural descriptors](../../../datasets/supplementary_data/dataset_structural_mining.py) and [subset export](../../../scripts/data/export_frozen_subset.py).

## Reproduce

Recount the landscape records and check the reported descriptor quantiles from the supplied observations.

```bash
python scripts/reproduce_supporting_analyses.py
```

Redraw the descriptor panels of Figure 2b–e from the reaction-weighted descriptor records and the fully specified unique products. The script checks the percentiles against the published summary and the number of products included in the hexagonal bins. With NumPy, pandas, SciPy and Matplotlib installed, run:

```bash
python scripts/plot_paper_figure.py --figure 2
```

The plot is written to `outputs/figures/figure_2.png`.
