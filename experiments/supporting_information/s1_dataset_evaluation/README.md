# S1. CHRetro partitions and evaluation inputs

[Table S1](tables/table_s1.tex) · [Table S2](tables/table_s2.tex)

The full train, validation and test partitions contain 74,146, 9,395 and 9,250 reactions. Their small counterparts contain 18,370, 2,400 and 2,400. Source-row indices identify the exact parent reaction for each subset row. The SI describes the upstream corpus and selection criteria.

| Paper item or input | What it contains | Files |
|---|---|---|
| Table S1 | Family counts for the full dataset. The CSVs contain the reaction records used for each split. | [training](../../../datasets/full/train.csv), [validation](../../../datasets/full/valid.csv), [test](../../../datasets/full/test.csv), [family mapping](configs/dataset_class_mapping.json), [split summary](results/dataset_split_summary.json) |
| Table S2 | The small set is drawn from the full partitions. Parent indices use the original within-split row numbering. | [small reaction files](../../../datasets/small), [training indices](splits/train_parent_indices.csv), [validation indices](splits/valid_parent_indices.csv), [test indices](splits/test_parent_indices.csv), [small-set selection rules](configs/dataset_small_selection_plan.json) |
| Completion inputs | SpectraRetro uses synthon geometries and reaction-to-fragment mappings for the full and small benchmarks. | [SpectraRetro experiment inputs](../../spectraretro_inputs/README.md) |

[Complete file list, sources and SHA-256 digests](artifacts.json).

[Reaction cleaning](../../../datasets/supplementary_data/dataset_cleaning.py), [center annotation](../../../datasets/supplementary_data/dataset_center_annotation.py) and [subset export](../../../scripts/data/export_frozen_subset.py).
