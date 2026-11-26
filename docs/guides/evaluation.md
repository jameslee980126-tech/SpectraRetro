# Evaluation definitions

## Full CHRetro benchmark

Top-k accuracy is the fraction of test reactions whose reference precursor set occurs among the first k ranked candidates. Each class setting uses the same 9,250 test reactions and 15,830 fragments. Reactions with no successful completion remain in the denominator.

1. **Prediction inputs.** Stage 1 scores atoms and bonds on the product graph and samples possible centers from those scores. For Stage 2, the benchmark uses precomputed 3D synthons. Their fragmentation follows the bond changes in the mapped test reactions, and their retained atoms take their coordinates from the corresponding precursor conformers. [Reaction annotation](../../datasets/supplementary_data/dataset_center_annotation.py) · [Test fragment table](../../experiments/spectraretro_inputs/final_test/prepared/similarity_test_table.csv)
2. **Generation and scoring.** Stage 2 generates 300 draws per fragment over 100 reverse intervals. For scoring, the evaluator compares Stage 1’s sampled indices with the test label (`gold_candidate`). For a reaction with *h* correct predictions among the 300 draws, fragment slots 1 through *h* enter assembly. Other slots and failed decodes are marked `MISSING`. [Sampling and evaluation](../../src/spectraretro/completion/api.py)
3. **Ranking and matching.** The evaluator joins companion fragments with the same draw index. SpectraRetro scores distinct assembled strings using features that include occurrence frequency. A hit requires exact equality to the complete reference string obtained with the same XYZ/Open Babel decoder. The main result is stored under `stage1_gated`. The `stage2_ungated_diagnostic` result uses all draws. [Candidate ranking](../../src/spectraretro/fusion_reranker/runtime.py)

## Scientific scope

The full benchmark uses one seed per mode. The balanced set has three seeds per mode on the same 2,400 test reactions. Both datasets participated in method development. The full split has some shared products across partitions. [Partitions and overlap counts](../../datasets/DATASET_CARD.md)

The reported reaction-center Top-1 accuracy uses the saved evaluator ranking (89.36%). [Reaction-center prediction records and scoring rules](../../results/reaction_center/README.md)

## Cross-method results

The baseline guide lists the evaluation unit and ranking procedure for each protocol. In particular, original-slot baseline scores compare candidates to the source reaction's reactants and preserve invalid and duplicate positions. SpectraRetro uses the decoded reference and ranks distinct candidates. [Baseline protocols](baselines.md)

## Route cases

For each pharmaceutical case, the generation inputs specify the intermediates and their sites for extension. The case-specific models were trained with exposure to these examples. CHRetroAgent connects the stored candidates into routes that reach the case’s terminal inventory. Reported search times cover route assembly alone. The saved LLM assessments record the provider alias `deepseek-flash`. [Candidate generation and route search](../../experiments/section_4_results/4_8_chretroagent/README.md)
