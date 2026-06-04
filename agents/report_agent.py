"""
agents/report_agent.py — Final Report Generation Stage Agent.

Responsibilities:
  - Aggregate results from all pipeline stages
  - Generate a comprehensive Markdown report
  - Convert Markdown to a self-contained HTML file (all charts embedded as base64)
  - Include executive summary, method details, metrics tables, SHAP plots, recommendations
  - Update session_state: report_md_path, report_html_path
  - End response with <DONE>report</DONE>

Client: PRIMARY_CLIENT (report writing requires reasoning and synthesis)
MCP tools: DS Tools (read_file, execute_code, write_output, get_sample)
           Tracking (record_artefact, query_run)
Local tools: get_session_state, set_session_state, check_artefact_exists
"""
from __future__ import annotations

from agent_framework import Agent

from agents.base import build_pipeline_agent, make_ds_tools_mcp, make_tracking_mcp
from tools.local_tools import LOCAL_TOOLS

_SYSTEM_PROMPT = """You are the Report Generation Agent for an automated ML pipeline.

Your task is to:
1. Call `get_session_state` to retrieve all available pipeline stage results.
2. Use `ds_read_file` to load key artefacts:
   - EDA report (eda_report_path)
   - Evaluation report (evaluation_report_path)
   - Explanation narrative (explanation_narrative_path)
   - Transformation log (transformation_log_path)
   - Best params (best_params_path)
3. Use `ds_execute_code` to generate the final Markdown report at
   `data/artefacts/{run_id}/report/report.md` with these sections:
   ```
   # ML Pipeline Report — {task_description}
   Generated: {timestamp}
   
   ## Executive Summary
   ## Dataset Overview
   ## Data Quality & Cleaning
   ## Feature Engineering
   ## Model Selection & Training
   ## Hyperparameter Tuning
   ## Evaluation Results
   ## Fairness & Bias Analysis
   ## Model Explainability
   ## Deployment Recommendation
   ## Appendix: Transformation Log
   ```
4. Convert the Markdown to self-contained HTML:
   - Embed all chart images as base64 data URIs
   - Include Bootstrap CSS via CDN for styling
   - Include a collapsible Table of Contents
   - Save to `data/artefacts/{run_id}/report/report.html`
5. Update session state:
   - `report_md_path`  : path to report.md
   - `report_html_path`: path to report.html

Report writing guidelines:
- Use plain English in the executive summary
- Include exact metric values with 4 decimal places
- Highlight the deployment_recommendation with a coloured badge
- If any stages produced warnings/errors, note them in a "Caveats" section

Harness Engineering notes:
- The HTML report must be self-contained (no external file dependencies)
- Both .md and .html files MUST exist before <DONE>

End your response with:
```session_state
{
  "report_md_path": "<filled>",
  "report_html_path": "<filled>"
}
```

Then write: <DONE>report</DONE>
"""


def build_report_agent() -> Agent:
    """Build the Report Generation stage agent."""
    return build_pipeline_agent(
        name="report_agent",
        instructions=_SYSTEM_PROMPT,
        tools=[
            make_ds_tools_mcp(),
            make_tracking_mcp(),
            *LOCAL_TOOLS,
        ],
        use_fast_client=False,
        max_message_groups=20,
    )
