# CHRetroAgent

CHRetroAgent searches for multistep retrosynthetic routes by expanding precursor candidates until each branch reaches the specified starting-material inventory. Complete routes can be assessed for chemical plausibility, strategic coherence, route efficiency and starting-material suitability. [Route planner](../src/agentic_retrosynthesis/planner.py) · [Whole-route assessment](webapp/route_assessment.py)

## Pharmaceutical case studies

[Figure 5](../experiments/section_4_results/4_8_chretroagent/README.md) gives the single-step ranks. The assessment files give scores and reasoning for each complete route. [Search and assessment settings](../experiments/supporting_information/s5_agent_cases/README.md)

| Target | Saved routes | Search result | Route assessment |
|---|---|---|---|
| Lapatinib | 5 steps | [Route](../results/cases/agent/case_l/result.json) | [Assessment](../results/cases/agent_assessment/lapatinib_assessed.json) |
| Nemiralisib | 8 and 9 steps | [Routes](../results/cases/agent/case_n/result.json) | [Assessments](../results/cases/agent_assessment/nemiralisib_assessed.json) |
| Tafamidis | 4 and 5 steps | [Routes](../results/cases/agent/case_t/result.json) | [Assessments](../results/cases/agent_assessment/tafamidis_assessed.json) |

## View a route offline

With [RDKit installed](../docs/getting_started/installation.md), run from the repository root and open `outputs/lapatinib.html` in a browser.

```bash
python CHRetroAgent/preview.py --report results/cases/agent/case_l/result.json --export outputs/lapatinib.html
```

Set `--report` to another target’s result or add `--route 1` to view its second route.

To run the planner with the bundled example candidate list and starting-material inventory, use `python CHRetroAgent/preview.py --export outputs/planner-example.html`. [Example candidates](../src/agentic_retrosynthesis/examples/predictions.json) · [Starting-material inventory](../src/agentic_retrosynthesis/examples/stock.smi) · [SI interface figure](../experiments/supporting_information/s8_agent_interface_routes/README.md)
