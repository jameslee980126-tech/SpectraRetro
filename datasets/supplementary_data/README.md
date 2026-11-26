# Supplementary source records for Sections S1–S6

`source_index.csv` identifies the selected original records and their SHA-256 checksums. The records cover dataset construction and split selection, figure-population definitions, paired and seedwise evaluation summaries, selected training checkpoints, and the pharmaceutical and example-case inputs and outputs. Original numerical values and molecular strings are preserved. The two Figure 2b–e descriptor CSVs retain all 92,791 reaction records and 73,522 normalized product records (73,521 with fully specified elements), respectively.

The [mechanistic source data](../../results/source_data/mechanistic_interpretability/supplement/) support Figure 4 and SI Section S7.

## Figure and table numbering

For the three incorrect predictions in **Figure 3a,b and Figure S3a**, see the [case structures](figure8_case_structures.csv) and their [source records](figure8_case_structures_provenance.json). [Table S16](../../experiments/supporting_information/s5_agent_cases/tables/table_s16.tex) lists the structural difference for each case.

## Benchmarks and planning records

`main_table1_small_spectraretro.csv` gives the small-set entries printed in the main article. The seedwise hit counts and sample standard deviations in `small_stratum_seed_hits.csv` document the separate three-run analysis in the SI.

The planner can inspect a state or expand an intermediate, incorporating tool feedback in subsequent decisions. The two implementation excerpts, `feedback_controller.py` and `feedback_planner.py`, document those operations. The pharmaceutical case files give the recorded route structures, single-step ranks and whole-route scores used in the article. Fields describing candidate banks and generation settings identify archived records. The assessment times refer to whole-route scoring.

## Scientific source dependencies

- `dataset_structural_mining.py` uses RDKit and a caller-supplied atom-mapped source CSV. `dataset_cleaning.py` additionally needs the original mined reaction tables, generated atom-mapping results and arylation source partitions identified in the code.
- `dataset_small_builder.py` records the grouped selection procedure and expects the original parent partitions, per-reaction structural metadata and historical subset indices. The selected row indices, source hashes, selection policy and selected-population summary are included here. See `dataset_construction_README.md` for the input layout.
- `dataset_center_annotation.py` requires NumPy, pandas, RDKit, PyTorch, torch-scatter and TorchDrug; it is an implementation record intended for the original data-processing workflow.
- The feedback planner imports `agentic_retrosynthesis.planner` and requires a candidate provider and evidence tools. The feedback controller takes a configured LLM provider. These files can be inspected as source excerpts.
- `llm_client.py` records the request defaults and response handling. Configure the endpoint, model and any required authentication locally.
- `route_assessment.py` records the assessment prompt and validation logic. Its original package imports `agentic_retrosynthesis.tool_policy` and `route_diagram`; those modules and a configured client are required for execution. The included case and route tables preserve the reported observations without making new service requests.

The [Figure S2 summary](../../results/supporting_information/source_data_figures/reported_figure7_summary.csv) gives the mean heavy-atom bond-graph match rate with and without the chemical prior and the counts of reactions with increased or decreased match rates. For complete-precursor recovery in Table 3, see the [chemical-prior comparison](../../experiments/section_4_results/4_2_ablation/README.md).
