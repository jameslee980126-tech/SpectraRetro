# Experiments by paper section

| Experiment | Paper location | Scientific resources |
|---|---|---|
| [CHRetro construction and chemical diversity](section_2_dataset/2_dataset/) | Introduction and dataset curation · Figures 1–2 | Reaction partitions, similarity landscape and physicochemical descriptors. |
| [CHRetroAgent and SpectraRetro architecture](section_3_methods/3_1_to_3_4_architecture/) | Methods · Figure 1 | CHRetroAgent tools and route planning (b, f), reaction-center scoring and synthon completion (c–e). |
| [Known-class and unknown-class benchmarks](section_4_results/4_1_benchmark/) | Retrosynthetic prediction performance · Table 1 | Full and small test sets, selected checkpoints and baseline results. |
| [Complete-precursor recovery by local similarity](section_4_results/4_3_similarity/) | Retrosynthetic prediction performance · Table 2 | Similarity thresholds and recovery in three strata of the balanced subset. |
| [Incorrect precursor predictions](section_4_results/4_5_error_analysis/) | Incorrect predictions · Figures 3a,b and S3a | Reference and predicted structures for three errors, with reaction-level records for the full test set. |
| [Full-spectrum and chemical-prior ablations](section_4_results/4_2_ablation/) | Ablation study · Table 3 and Figures S1–S2 | Reaction-center comparisons, chemical-prior controls and heavy-atom graph recovery. |
| [Six single-step predictions](section_4_results/4_6_single_step_cases/) | Representative C–H activation reactions · Figures 3c,d and S3b–e | Selected centers and precursor candidates for six reactions. |
| [Mechanistic analysis](section_4_results/4_4_mechanistic_interpretability/) | Mechanistic interpretability · Figure 4 | Chemical context, spatial interventions and paired electronic calculations. |
| [Three two-step predictions](section_4_results/4_7_multistep_cases/) | Multistep retrosynthetic planning · Figures 3e,f and S3f | Target and intermediate predictions with the reported forward syntheses. |
| [CHRetroAgent pharmaceutical routes](section_4_results/4_8_chretroagent/) | Multistep retrosynthetic planning · Figure 5 | Lapatinib, nemiralisib and tafamidis routes, candidate pools and recorded assessments. |
| [S1: Dataset construction and evaluation](supporting_information/s1_dataset_evaluation/) | SI §S1 · Table S1, Table S2 | Full and small partition compositions, small-set selection settings and parent-row indices. |
| [S2: Full-spectrum atom-pair learning](supporting_information/s2_spectral_encoder/) | SI §S2 · Table S3 | Spectral encoder reference implementation, ablation training curves and training settings. |
| [S3: Chemical-prior-guided 3D completion](supporting_information/s3_conditional_diffusion/) | SI §S3 · Table S4 | Completion and size model training records, validation-based model selection and diffusion settings. |
| [S4. Diagnostics and additional analyses](supporting_information/s4_additional_results/) | Figures S1–S2 and Tables S5–S10, S12–S13 | Encoder checks, reaction-center and precursor recovery, chemical and electronic observations. |
| [S5. CHRetroAgent configuration and case studies](supporting_information/s5_agent_cases/) | Figure S3 and Tables S14–S17 | Pharmaceutical routes, structural errors, single-step predictions and two-step examples. |
| [S6: Supplementary files and model definitions](supporting_information/s6_file_and_model_definitions/) | SI §S6 · Table S18, Table S19, Table S20, Table S21 | Supplementary file inventories, feature dictionary, module dimensions and composition descriptors. |
| [S7. Mechanistic analyses](supporting_information/s7_mechanistic_analyses/) | Figures S4–S6 and Tables S22–S27 | Chemical, spatial and electronic records, atom populations and molecular geometries. |
| [S8. CHRetroAgent interface and model details](supporting_information/s8_agent_interface_routes/) | Figures S7–S8 | Route viewer, full-spectrum center scoring and chemical-prior-guided 3D completion. |

[Figure and table index](../docs/paper_map.md) · [Result files](../results/README.md) · [Experiment manifest](manifest.json)

To verify the linked data and paper materials, run `python scripts/verify_paper_materials.py` from the repository root. The [command index](../scripts/README.md) lists the scripts for recounting saved results.
