# Complete mechanistic supporting data

This directory accompanies main Figure 4, SI Section S7, Figures S4–S6 and Tables S22--S27.

| Directory | Content | Displays |
|---|---|---|
| chemical/ | 19-descriptor candidate catalogue, 16 retained definitions, 114-slot archived Mantel/concordance tables, 120 descriptor correlations, and record/product inputs | Figure 4a; Tables S22--S23 |
| spatial/ | Noise/centering summaries, all 38 fragment responses, all 22 cluster-omission calculations and original reference files | Figure 4b; Figure S4; Table S24 |
| electronic/ | Full-precision molecular inventories, six primary-pair contrasts, five alternative-site controls, population-scheme contrasts and convergence checks | Figure 4c; Figure S5; Tables S25--S27 |
| structures/ | Mapped molecular identities, formulas and oxidation positions | Figure S6; Table S25 |
| quantum/ | 17 unchanged Cartesian inputs, 34 completed electronic-state records, atom populations, SCF logs, integration diagnostics and archived calculation code/settings | Tables S25--S27; Section S7.4 |

`data_index.csv` and `data_index.json` provide one entry per data/provenance file, with its purpose, linked displays, byte count and SHA-256 hash. `manifest.json` also covers these indices and this README; the manifest itself is excluded to avoid a circular digest.

The printed SI contains selected summaries; complete per-case values remain in these data files. Legacy electronic filenames beginning Table_S28 through Table_S32 retain their original archive names, and the data index maps them to the current Tables S25--S27 and Figure S5.

The chemical table retains all 65 tested Mantel associations and 49 untested slots. All 86 finite concordance estimates are preserved, including 21 below-threshold additional estimates; two one-product cases have no recorded bootstrap interval. The 120 descriptor correlations use pairwise-finite values: 78 pairs have 2,055 products and 42 have 2,054 products. Record-level contrasts, factor-specific product cohorts, exact probabilities and permutation seeds are retained. Exact numeric values are retained rather than rounded table strings.

Spatial statistics use the archived independent replay and the original hierarchy of 100 orientations, 38 fragments and 22 retained-scaffold graph clusters. These are controlled one-step responses for fixed reference-derived completion groups; linked precursor-recovery outcomes are separately identified in the fragment table.

Electronic calculations use fixed GFN2-xTB-optimized geometries. The three DFT population partitions belong to the same PBE0/def2-TZVP electronic states. Relative changes are oxide minus parent or target-site oxide minus alternative-site oxide, as indicated. Model values use bond-direction-averaged logits; original readouts are retained under distinct column names.

The [electronic-state index](quantum/electronic_states.json) identifies the calculation results for 34 charge states across 17 molecular geometries.
