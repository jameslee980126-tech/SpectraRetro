# CHRetro dataset card

CHRetro comprises six classes of C–H functionalization reactions. Each CSV row
is a complete reaction with `class`, mapped `rxn_smiles`, and `original_class`.
Reaction SMILES have the form `reactants>agents>products`. The middle field
is populated for 84,033 of the 92,791 full-set reactions and empty for the
remaining 8,758. A reaction can yield several completion fragments.

Section S1 of the Supporting Information identifies five sources for the reaction records.

- [USPTO-FULL patent reactions](https://doi.org/10.6084/m9.figshare.5104873.v1)
- [MappedCRD atom-mapped reactions](https://huggingface.co/datasets/lukaskim/MappedCRD)
- [The computational C–H functionalization benchmark released with RXNGraphormer](https://doi.org/10.1038/s42256-025-01098-4)
- [Selected arylation studies in the Open Reaction Database](https://doi.org/10.1021/jacs.1c09820)
- [The archived C–H arylation collection of Li et al.](https://doi.org/10.1021/acs.jcim.5c01735)

| Partition | Training reactions | Validation reactions | Test reactions |
|---|---:|---:|---:|
| Full | 74,146 | 9,395 | 9,250 |
| Balanced small | 18,370 | 2,400 | 2,400 |
| Full completion fragments | 126,561 | 16,059 | 15,830 |

The full split is fixed. With unmapped, stereochemistry-insensitive canonical
product keys, train–validation, train–test, and validation–test share 95, 123,
and 12 product keys respectively. These involve 132 validation, 187 test, and
14 test records respectively. It is therefore a reaction-level split with
product overlap.

The balanced small set has disjoint product keys under its construction rule,
and balances structural and local-similarity groups. Validation and test each
contain 800 records in each of the low, medium, and high groups. Its parent
data participated in method selection; use it as a structured exploratory
diagnostic. Small-set `source_row` identifies the parent record and need not
be contiguous. `*_index.csv` retains that mapping.

## Files

| Location | Content |
|---|---|
| `full/`, `small/` | Fixed reaction CSV partitions and small-set indices |
| `split_summary.json` | Counts, class distributions, and file hashes |
| [Prepared SpectraRetro inputs](../experiments/spectraretro_inputs/README.md) | Geometries, elemental priors, reaction-to-fragment maps and references for the model experiments |
| `experiments/section_4_results/4_1_benchmark/{mode}-class/data/evaluation/` | Mode-specific size draws, batch order, and decoded references |

In the evaluation path, `{mode}` is `known` or `unknown`. Data files are
included alongside the reaction CSVs.
Unknown-class models use class labels for reporting only; known-class models
use their declared class condition. The evaluation denominator counts complete
source reactions, including zero-completion and unsuccessful reactions.

## Reuse

File identities are listed in `MANIFEST_SHA256.csv` and verified with
`python scripts/verify_release.py`. Reaction data retain their source-specific
reuse conditions, separate from software licensing. A single reference reaction
does not enumerate all chemically valid synthesis routes. Mining and mapping
algorithms are available under `src/spectraretro/data_preparation/`;
the fixed CSVs and geometry caches define the reference benchmark inputs.
