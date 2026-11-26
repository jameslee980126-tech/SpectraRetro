# Dataset construction files

`dataset_structural_mining.py` provides reaction parsing, map-free reaction-core
keys, and the structural mining helpers. `dataset_cleaning.py` imports this
packaged module by its released filename. Its input paths identify the source
mining table, generated mappings, and original arylation files used by the
historical curation stage. These upstream inputs are required to run that stage.

`dataset_small_selection_plan.json` specifies the frozen grouped-subset quotas
and constraints. `dataset_small_train_index.csv`,
`dataset_small_valid_index.csv`, and `dataset_small_test_index.csv` identify the
selected source rows and product groups. Their checksums match the existing
`dataset_small_manifest.json`. `dataset_small_selected_summary.json` gives the
selected counts and structural summaries. The original subset builder also
requires the parent partitions and per-reaction structural metadata referenced
in its manifest.

The three figure provenance JSON files record the graph, layout, and product
population configurations used by the supplementary methods description.
