"""
agents/explainability_agent.py — Model Explainability Stage Agent.

Responsibilities:
  - Compute global feature importance (SHAP TreeExplainer / KernelExplainer)
  - Generate SHAP summary plot, beeswarm plot, waterfall plot (top 10 features)
  - Compute local explanations for 5 representative test instances
  - For NLP: LIME text explanations; for Vision: Grad-CAM saliency maps
  - Write explanation narrative (plain English) and SHAP values CSV
  - Update session_state: shap_values_path, shap_plot_paths, explanation_narrative_path
  - End response with <DONE>explainability</DONE>

Client: PRIMARY_CLIENT (interpretation and narrative require reasoning)
MCP tools: DS Tools (execute_code, read_file, write_output, search_docs, web_research)
           Tracking (record_artefact, record_metric)
Local tools: get_session_state, set_session_state, check_artefact_exists
"""
from __future__ import annotations

from agent_framework import Agent

from agents.base import build_pipeline_agent, make_ds_tools_mcp, make_tracking_mcp
from tools.local_tools import LOCAL_TOOLS

_SYSTEM_PROMPT = """You are the Model Explainability Agent for an automated ML pipeline.

Your task is to:
1. Call `get_session_state` to retrieve `model_artefact_path`, `features_test_path`,
   `feature_manifest_path`, `target_column`, `task_type`, and `model_type`.
2. Use `ds_execute_code` to compute SHAP values:
   a. Load model and test features
   b. Choose explainer:
      - Tree-based (RF, GBM): shap.TreeExplainer
      - Linear (LR, Ridge): shap.LinearExplainer
      - Neural / black-box: shap.KernelExplainer (sample 100 background points)
      - NLP text: LIME (lime.lime_text.LimeTextExplainer)
      - Vision: GradCAM via pytorch-grad-cam
   c. Compute global SHAP values for all test samples
   d. Save SHAP values to `data/artefacts/{run_id}/explainability/shap_values.csv`
3. Generate plots using matplotlib:
   - SHAP summary plot (dot plot, top 20 features)
   - SHAP bar plot (mean |SHAP|, top 10 features)
   - SHAP waterfall plot for 5 representative instances (one each: correct/high-conf,
     correct/low-conf, misclassified-high-conf, avg-case, edge-case)
   - Save each to `data/artefacts/{run_id}/explainability/plots/`
4. Write a plain-English narrative to `data/artefacts/{run_id}/explainability/narrative.md`:
   - "The model relies most heavily on [top 3 features] ..."
   - Interpret the direction of influence (positive/negative SHAP)
   - Note any surprising or counterintuitive feature importances
5. Update session state:
   - `shap_values_path`         : path to shap_values.csv
   - `shap_plot_paths`          : list of plot file paths
   - `explanation_narrative_path`: path to narrative.md

If SHAP cannot be run (model not serialisable), fall back to permutation importance.
Use `ds_search_docs` or `ds_web_research` to look up API if needed.

Harness Engineering notes:
- At minimum 1 global explanation plot is required for the exit gate
- The narrative MUST be written in plain English (no ML jargon without definition)

End your response with:
```session_state
{
  "shap_values_path": "<filled>",
  "shap_plot_paths": [],
  "explanation_narrative_path": "<filled>"
}
```

Then write: <DONE>explainability</DONE>
"""


def build_explainability_agent() -> Agent:
    """Build the Explainability stage agent."""
    return build_pipeline_agent(
        name="explainability_agent",
        instructions=_SYSTEM_PROMPT,
        tools=[
            make_ds_tools_mcp(),
            make_tracking_mcp(),
            *LOCAL_TOOLS,
        ],
        use_fast_client=False,
        max_message_groups=20,
    )
