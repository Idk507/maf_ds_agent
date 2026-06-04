"""
agents/eda_agent.py — Exploratory Data Analysis Stage Agent.

Responsibilities:
  - Compute descriptive statistics (mean, std, quartiles, missing rates, cardinality)
  - Generate correlation matrix and detect multicollinearity
  - Produce distribution charts (histograms, box plots, correlation heatmap)
  - Write EDA report (Markdown) and narrative summary
  - Update session_state: eda_report_path, eda_narrative_path, chart_paths, data_quality_flags
  - End response with <DONE>eda</DONE>

Client: FAST_CLIENT (statistical computation)
MCP tools: DS Tools (read_file, execute_code, get_sample, write_output, log_metrics)
           Tracking (record_artefact, record_metric, record_checkpoint)
Local tools: get_session_state, set_session_state, check_artefact_exists
"""
from __future__ import annotations

from agent_framework import Agent

from agents.base import build_pipeline_agent, make_ds_tools_mcp, make_tracking_mcp
from tools.local_tools import LOCAL_TOOLS

_SYSTEM_PROMPT = """You are the Exploratory Data Analysis (EDA) Agent for an automated ML pipeline.

Your task is to:
1. Call `get_session_state` to retrieve the current pipeline state (get schema and input_artefact_path).
2. Use `ds_execute_code` to run Python analysis:
   - Load data with pandas
   - Compute: shape, dtypes, missing counts, unique counts, basic stats (describe())
   - Compute correlation matrix for numeric columns
   - Detect: high-cardinality columns (>50% unique), constant columns, duplicate rows
3. Generate charts using matplotlib/seaborn via `ds_execute_code`:
   - Histogram for each numeric column
   - Box plot for outlier visualisation
   - Correlation heatmap
   - Save each chart to `data/artefacts/{run_id}/eda/charts/`
4. Write an EDA Markdown report to `data/artefacts/{run_id}/eda/eda_report.md` using `ds_write_output`.
5. Write a plain-English narrative to `data/artefacts/{run_id}/eda/eda_narrative.txt`.
6. Use `tracking_log_metrics` to record key stats (missing_rate, duplicate_rate, n_features, n_rows).
7. Update session state:
   - `eda_report_path`      : path to EDA markdown report
   - `eda_narrative_path`   : path to plain-English narrative
   - `chart_paths`          : list of chart file paths
   - `data_quality_flags`   : list of quality warnings (e.g. ["high_missing: col_A", "constant: col_B"])

Harness Engineering notes:
- All file outputs MUST exist on disk before you emit <DONE>
- Record every chart as an artefact via tracking_record_artefact
- If data has no numeric columns, note this in the narrative and skip numeric charts

End your response with:
```session_state
{
  "eda_report_path": "<filled>",
  "eda_narrative_path": "<filled>",
  "chart_paths": [],
  "data_quality_flags": []
}
```

Then write: <DONE>eda</DONE>
"""


def build_eda_agent() -> Agent:
    """Build the EDA stage agent."""
    return build_pipeline_agent(
        name="eda_agent",
        instructions=_SYSTEM_PROMPT,
        tools=[
            make_ds_tools_mcp(),
            make_tracking_mcp(),
            *LOCAL_TOOLS,
        ],
        use_fast_client=True,
        max_message_groups=20,
    )
