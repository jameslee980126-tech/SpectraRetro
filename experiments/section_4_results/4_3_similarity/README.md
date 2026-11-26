# Complete-precursor recovery by local similarity

[Table 2](tables/table_2.tex)

The small test partition contains 800 reactions in each of three similarity strata. For SpectraRetro, the saved seed-level hits give the mean and sample standard deviation for each class mode and stratum. The table also contains baseline rows, including a single value per metric for RetroSim.

| Paper item or input | What it contains | Files |
|---|---|---|
| Stratum assignment | The low and high cutoffs were calibrated on 9,381 eligible validation structures. The protocol index identifies the inputs and weights for the small-set runs. | [thresholds](configs/similarity_thresholds.json), [protocol index](provenance/small_completion.json), [frozen test reactions](../../../datasets/small/test.csv) |
| Table 2 | SpectraRetro integer hit counts by seed, class setting and similarity group. Each group contains 800 reactions. | [counts and three-seed statistics](results/small_stratum_seed_hits.csv), [printed table](tables/table_2.tex) |

[Complete file list, sources and SHA-256 digests](artifacts.json). [Model files and selection details](models/README.md).

[Similarity calculation](../../../scripts/data/similarity.py) and [Top-k recount](../../../scripts/reproduce_results.py).

## Reproduce

Recount SpectraRetro hits by joining the six small-set runs to the [frozen stratum assignments](../../../datasets/small/test_index.csv). The script writes `outputs/reproduced_results/small_stratum_seed_hits.csv`.

```bash
python scripts/reproduce_results.py --check
```
