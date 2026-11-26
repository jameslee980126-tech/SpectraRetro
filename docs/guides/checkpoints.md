# Model checkpoints

The [weight registry](../../checkpoints/WEIGHTS.csv) lists the component, reaction-class setting, selected checkpoint and SHA-256 digest. For the full benchmark, each launcher selects its reaction-center, fragment-size and 3D completion checkpoints. Both settings use the same elemental composition prior and have their own candidate-ranking parameters. [Known-class configuration](../../experiments/section_4_results/4_1_benchmark/known-class/configs/full.json) · [Unknown-class configuration](../../experiments/section_4_results/4_1_benchmark/unknown-class/configs/full.json) · [Ranking parameters](../../src/spectraretro/fusion_reranker/assets)

| Experiment | Weights and inputs |
|---|---|
| Balanced CHRetro subset | [Six training runs and their inputs](../../configs/experiments/small_completion.json) |
| Chemical-prior ablation | [Control and compatibility weights](../../checkpoints/ablations/chemical_prior) and [paired evaluation records](../../results/reproduction/chemical_prior) |
| Single-step and two-step cases | [Case weights](../../checkpoints/cases/single_multistep) and [case records](../../results/cases) |
| Pharmaceutical routes | [Case weights](../../checkpoints/cases/agent) and [generation inputs and candidates](../../results/cases/agent_generation) |
| Baselines | [Baseline checkpoint index](../../configs/baseline_checkpoints.json) and [scoring protocols](baselines.md) |

The original TransGraphEdit evaluator loads a file named `epoch_78.pt`. For full-data evaluation, place a copy of the selected checkpoint under that name in the evaluator's experiment directory. Use epoch 56 for known-class or epoch 20 for unknown-class. [Validation and prediction records](../../results/baselines/checkpoint_selection/TransGraphEdit)

Run `python scripts/verify_weights.py` from the repository root to check checkpoint hashes and validation records.
