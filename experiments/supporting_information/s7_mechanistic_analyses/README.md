# S7. Chemical associations, spatial robustness and electronic response

[Table S22](tables/table_s22.tex) · [Table S23](tables/table_s23.tex) · [Table S24](tables/table_s24.tex) · [Table S25](tables/table_s25.tex) · [Table S26](tables/table_s26.tex) · [Table S27](tables/table_s27.tex) · [Figure S4](figures/figure_s4_spatial_robustness.png) · [Figure S5](figures/figure_s5_electronic_validation.png) · [Figure S6a: parent structures](figures/figure_s6a_structure_catalog.png) · [S6b: thiazole N-oxides](figures/figure_s6b_structure_catalog.png) · [S6c: alternative-site N-oxides](figures/figure_s6c_structure_catalog.png)

The chemical analysis measures context associations, the spatial analysis examines 38 fragments in 22 scaffold clusters, and the electronic calculations compare 17 fixed molecular geometries in 34 electronic states.

| Paper item or input | What it contains | Files |
|---|---|---|
| Tables S22–S23 | Descriptor definitions and product support precede the reaction-family association summaries. | [factor catalogue](data/chemical/factor_catalog.csv), [association measurements](results/chemical/chemical_preferences.csv), [family summary](results/chemical/Table_S23_family_summary.csv) |
| Figure S4 and Table S24 | Use fragment-level records and scaffold IDs to interpret sensitivity to diffusion time and coordinate centering. | [fragment details](data/spatial/spatial_fragment_details.csv), [cluster dictionary](data/spatial/spatial_cluster_dictionary.csv), [summary statistics](results/spatial/spatial_noise_statistics.csv) |
| Figures S5–S6 and Tables S25–S26 | The structure inventory maps geometries to the primary pairs and alternative-site controls. | [structure inventory and neutral/cation energies](data/electronic/Table_S25_structure_inventory.csv), [geometries](data/quantum/geometries), [primary method comparison](data/electronic/Table_S26_primary_method_comparison.csv), [electronic figure data](data/electronic/FigS2_source_data.csv), [alternative-site controls](data/electronic/dft_alternative_N_controls.csv) |
| Table S27 | These diagnostics cover SCF convergence and population integration. The electronic-state index links the 34 state results. | [calculation diagnostics](results/electronic/Table_S27_calculation_quality_summary.csv), [state summary](results/quantum/dft_34_state_summary.csv), [electronic-state index](../../../results/source_data/mechanistic_interpretability/supplement/quantum/electronic_states.json) |

[Additional electronic-structure calculations](data/electronic/source_exports).

[Complete file list, sources and SHA-256 digests](artifacts.json).

[Numerical replay](../../../scripts/reproduce_mechanistic_tables.py), [calculation routines](../../../results/source_data/mechanistic_interpretability/supplement/quantum/calculation_code) and [DFT protocol](configs/quantum/original_dft_protocol.json).

## Reproduce

Check saved atomic-population contrasts, pair identities and spatial statistics.

```bash
python scripts/reproduce_mechanistic_tables.py
```
