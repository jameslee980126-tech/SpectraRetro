# Architecture of CHRetroAgent and SpectraRetro

[Figure 1](../../section_2_dataset/2_dataset/figures/figure_1_retrosynthetic_overview.png) · [Figure S8](../../supporting_information/s8_agent_interface_routes/figures/figure_s8_spectraretro_method_details.png)

Figure 1b and 1f show the tools CHRetroAgent uses for route search and assessment. Figure 1c–e follows a single-step prediction through disconnection, reaction-center scoring and conditional 3D completion. Figure S8 gives the model inputs and architecture.

| Paper item or input | What it contains | Files |
|---|---|---|
| Figure 1b,f | Tools, route planning and whole-route assessment. | [CHRetroAgent and viewer](../../../CHRetroAgent/README.md), [planner](../../../src/agentic_retrosynthesis) |
| Figure 1c–e and Figure S8 | Product encoding, reaction-center scores and conditional 3D completion. | [Single-step model](../../../src/spectraretro/README.md), [model details](../../supporting_information/s8_agent_interface_routes/figures/figure_s8_spectraretro_method_details.png) |
| Model inputs | These configurations specify whether the model receives a reaction-family label. Both describe the full benchmark. | [class known](configs/known_full.json), [class unknown](configs/unknown_full.json) |

[Complete file list, sources and SHA-256 digests](artifacts.json).

The [known-class](../../../src/spectraretro/known-class/) and [unknown-class](../../../src/spectraretro/unknown-class/) directories contain the two settings of SpectraRetro. Both predict reaction centers, complete the test synthons in 3D and rank assembled precursor sets for each reaction. [Model source guide](../../../src/spectraretro/README.md).

## Reproduce

Inspect prediction inputs and execution order for either class setting.

```bash
python src/spectraretro/known-class/run.py --check-only
python src/spectraretro/unknown-class/run.py --check-only
```
