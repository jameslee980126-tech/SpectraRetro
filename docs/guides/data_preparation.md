# Data preparation

The included fixed reaction CSVs and geometry caches define
the reference benchmark inputs. The data-preparation source supports inspecting
and constructing related reaction datasets.

| Operation | Module under `src/spectraretro/data_preparation/` | Additional package |
|---|---|---|
| Inspect or extract ORD records | `inspect_ord_*`, `extract_ord_ch_activation.py` | `ord-schema` |
| Mine mapped C–H activation patterns | `mine_mapped_ch_activation.py` | RDKit |
| Read mapped parquet collections | `mine_crd_mapped_parquet.py` | PyArrow and RDKit |
| Add atom maps to reactions | `map_unmapped_ch_activation_reactions.py` | RXNMapper |
| Clean and filter reactions | `clean_ch_activation_dataset.py`, `filter_single_center_small_classes.py` | RDKit |
| Combine and inspect reaction classes | `combine_local_ch_activation_mining.py`, `analyze_ch_arylation_subclasses.py` | RDKit |

These optional packages are needed only for the corresponding source formats.
`mine_mapped_ch_activation.py` reads a mapped reaction CSV via `--input`.
`mine_crd_mapped_parquet.py` reads a Parquet directory via `--input-dir`.
Both write to the directory passed with `--output-dir`. The cleaning script reads
the mined reaction table, RXNMapper results and original arylation partitions
from paths relative to its own directory. Its `INPUT`, `GENERATED_MAPS` and
`ORIGINAL_FILES` definitions specify the expected layout.

`scripts/data/export_frozen_subset.py` exports the balanced subset using its
fixed parent-row indices. `scripts/data/similarity.py` implements molecular and
reaction similarity calculations. The [dataset card](../../datasets/DATASET_CARD.md)
defines sample counts, partition overlap, and class-field use.
