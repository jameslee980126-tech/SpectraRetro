# SpectraRetro experiment inputs

Reaction and fragment IDs link these inputs to the [CHRetro partitions](../../datasets/README.md). Both reaction-class settings use the same full-test synthon geometries.

| Directory | Contents and use |
|---|---|
| [old/newdata/](old/newdata/) | Synthon geometry tensors for training and validation. |
| [data/](data/) | Metadata identifying the associated training and validation fragments. |
| [cycle/](cycle/) | Elemental-composition probabilities used for training and validation. |
| [final_test/](final_test/) | Geometry tensors, elemental priors, reaction-to-fragment map and reference coordinates for the full test set. |
| [small/](small/) | Prepared small-test geometries, center predictions, size draws, elemental priors and evaluation references. |
| [chemical_prior/](chemical_prior/) | Fixed inputs, reference outputs and evaluation scripts for the [chemical-prior ablation](../section_4_results/4_2_ablation/README.md). |

For class-specific weights, center predictions and size draws, see the [known-class](../section_4_results/4_1_benchmark/known-class/) and [unknown-class](../section_4_results/4_1_benchmark/unknown-class/) benchmark directories.
