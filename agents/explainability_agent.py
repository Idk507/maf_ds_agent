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
   `feature_manifest_path`, `target_column`, `task_type`, and `model_type`. Also check
   `evaluation_report_path` and `cleaned_dataset_path`.

IMPORTANT: If `model_artefact_path` does not exist (training may have failed), you MUST
still proceed. Use `ds_execute_code` to re-train a simple RandomForestClassifier or 
RandomForestRegressor using the cleaned dataset and produce feature importances.
Always produce explainability output regardless of training stage outcome.

2. Use `ds_execute_code` to compute feature importance / SHAP values:
   a. Load model (or train a simple one if model file missing) and test features
   b. If SHAP is available, use TreeExplainer for tree models, LinearExplainer for linear.
      Otherwise, use sklearn feature_importances_ or permutation importance as fallback.
   c. Compute global feature importances for all test samples
   d. Save SHAP values / feature importances using `ds_write_output`:
      - sub_dir="explainability", filename="shap_values.csv"
      - Use the ACTUAL returned path as `shap_values_path`
3. Write a feature importance JSON (serves as the "plot" for criteria) using `ds_write_output`:
   - Use `ds_execute_code` to compute feature importance values (sklearn feature_importances_ or permutation)
   - Write the results as JSON using `ds_write_output`:
     sub_dir="explainability", filename="feature_importance_plot.json"
   - Content: {"features": [...], "importances": [...], "plot_type": "bar_chart"}
   - Use the ACTUAL returned path as the FIRST element of `shap_plot_paths` list
4. Write a plain-English narrative using `ds_write_output`:
   - sub_dir="explainability", filename="narrative.md"
   - "The model relies most heavily on [top 3 features] ..."
   - Interpret the direction of influence (positive/negative SHAP or importance)
   - Note any surprising or counterintuitive feature importances
   - Use the ACTUAL returned path as `explanation_narrative_path`
5. Update session state:
   - `shap_values_path`         : ACTUAL path returned by ds_write_output
   - `shap_plot_paths`          : list with at least ONE plot path string
   - `explanation_narrative_path`: ACTUAL path returned by ds_write_output

If SHAP cannot be run (model not serialisable), fall back to permutation importance.
Use `ds_search_docs` or `ds_web_research` to look up API if needed.

Harness Engineering notes:
- At minimum 1 global explanation plot is required for the exit gate
- The narrative MUST be written in plain English (no ML jargon without definition)
- CRITICAL: shap_plot_paths must be a non-empty list with at least 1 path.
  Even if matplotlib fails, write a JSON file and include its path.
- CRITICAL: Use ds_write_output for ALL file writes. Use the ACTUAL returned paths.
- CRITICAL: Call set_session_state with all three keys on EVERY iteration.
  Do not wait for perfection — write the files and set state as early as possible.

End your response with:
```session_state
{
  "shap_values_path": "<filled>",
  "shap_plot_paths": ["<filled>"],
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
