# CHRetro datasets

| Data | Train | Validation | Test | Files |
|---|---:|---:|---:|---|
| Full CHRetro | 74,146 | 9,395 | 9,250 | [full](full) |
| Balanced small subset | 18,370 | 2,400 | 2,400 | [small](small) |

All six reaction CSVs use the columns `class`, `rxn_smiles`, and `original_class`. The reaction string is atom-mapped and follows the reactants>agents>products convention. The three `small/*_index.csv` files identify the selected rows in the corresponding full partitions. Keep row order when matching predictions to references.

SpectraRetro joins these reactions to prepared synthon geometries by reaction and fragment ID. Its [experiment inputs](../experiments/spectraretro_inputs/README.md) include the geometries, atom masks and elemental priors for 3D sampling. [Table 1 comparison methods and preprocessing](../docs/guides/baselines.md)

See the [dataset card](DATASET_CARD.md), [construction source records](supplementary_data/dataset_construction_README.md), and [family mapping](supplementary_data/dataset_class_mapping.json) for definitions and source requirements. The repository license does not replace the terms of the original reaction sources.

```bash
python scripts/data/export_frozen_subset.py --help
```
