# Scientific command-line tools

| Script | Purpose |
|---|---|
| `verify_release.py` | Verify every distributed file against the SHA-256 manifest |
| `verify_paper_materials.py` | Verify dataset counts, paper mapping and linked source-data identities |
| `reproduce_results.py --check` | Recount full and small precursor-recovery results, including similarity strata and available baseline ranks |
| `reproduce_supporting_analyses.py` | Recount prior-ablation, paired geometry, landscape and descriptor summaries |
| `reproduce_mechanistic_tables.py` | Replay saved spatial and electronic table calculations and identities |
| `plot_paper_figure.py --figure 2` or `--figure S1` | Redraw the dataset descriptor panels or the reaction-center ablation figure |
| `differ.py` | Prepare and execute the configured full-data DiffER workflow |
| `unpack_cache.py` | Verify and extract a separately obtained preprocessing ZIP with its matching member inventory |
| `data/export_frozen_subset.py` | Recreate the selected subset using the included parent-row indices |
| `data/similarity.py` | Molecular similarity and stratum definitions |

Run these scripts from the repository root. [Experiments by paper section](../experiments/README.md)
