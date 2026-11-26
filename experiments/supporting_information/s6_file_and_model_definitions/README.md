# S6. Supplementary files and model definitions

[Table S18](tables/table_s18.tex) · [Table S19](tables/table_s19.tex) · [Table S20](tables/table_s20.tex) · [Table S21](tables/table_s21.tex)

Tables S18–S21 describe the supplementary files, reaction-center features, module dimensions and composition descriptors. The source indices map these definitions to their numerical records and implementations.

| Paper item or input | What it contains | Files |
|---|---|---|
| Table S18 | The indices enumerate the main supplementary files and the mechanistic source data. | [supplementary file index](file_indices/supplementary_file_index.csv), [mechanistic file index](file_indices/mechanistic_file_index.csv) |
| Tables S19–S20 | The feature dictionary specifies concatenation order, and the dimension table specifies the full-spectrum modules. | [feature dictionary](tables/table_s19.tex), [module dimensions](tables/table_s20.tex), [center implementation](../../../src/spectraretro/reaction_center) |
| Table S21 | Scalar descriptors are appended to the composition fingerprint. | [descriptors](tables/table_s21.tex), [composition model](../../../src/spectraretro/completion/composition_model.py) |
| Model files | The registry lists the parameter path, scientific role, selection evidence and digest for each registered file. | [model registry](models/README.md) |

[Complete file list, sources and SHA-256 digests](artifacts.json).
