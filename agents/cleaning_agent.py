"""
agents/cleaning_agent.py — Data Cleaning Stage Agent.

Responsibilities:
  - Handle missing values (impute or drop based on EDA flags)
  - Remove duplicate rows
  - Fix data type mismatches
  - Clip or winsorise outliers (IQR method)
  - Pseudonymise / mask PII columns if pii_detected=True
  - Write cleaned dataset to data/artefacts/{run_id}/cleaned/
  - Record transformation log (JSON) with every transformation applied
  - Update session_state: cleaned_dataset_path, transformation_log_path, cleaning_summary, content_hash
  - End response with <DONE>cleaning</DONE>

Client: PRIMARY_CLIENT (complex decisions on cleaning strategy)
MCP tools: DS Tools (read_file, execute_code, write_output, log_metrics, search_docs)
           Tracking (record_artefact, record_metric, record_lineage)
Local tools: get_session_state, set_session_state, check_artefact_exists
"""
from __future__ import annotations

from agent_framework import Agent

from agents.base import build_pipeline_agent, make_ds_tools_mcp, make_tracking_mcp
from tools.local_tools import LOCAL_TOOLS

_SYSTEM_PROMPT = """You are the Data Cleaning Agent for an automated ML pipeline.

Your task is to:
1. Call `get_session_state` to retrieve `input_artefact_path`, `schema`, `pii_detected`,
   and `data_quality_flags` from the EDA stage.
2. Use `ds_execute_code` to implement the cleaning pipeline:
   a. Load the raw dataset
   b. Remove exact duplicate rows
   c. Handle missing values:
      - Numeric: median imputation (if missing < 30%), else drop column
      - Categorical: mode imputation (if missing < 50%), else drop column
   d. Fix dtype issues (e.g. numeric stored as string)
   e. Clip outliers: winsorise at 1.5×IQR for numeric columns
   f. If `pii_detected` is True: hash/mask PII columns using SHA-256 prefix
   g. Compute content_hash (SHA-256 of the cleaned file bytes)
3. Write the cleaned dataset to `data/artefacts/{run_id}/cleaned/dataset_cleaned.parquet`
   using `ds_write_output`.
4. Write a JSON transformation log to `data/artefacts/{run_id}/cleaned/transformation_log.json`
   documenting every transformation with: {"column", "action", "reason", "rows_affected"}.
5. Use `tracking_record_lineage` to link raw → cleaned artefact.
6. Update session state:
   - `cleaned_dataset_path`   : path to cleaned parquet file
   - `transformation_log_path`: path to transformation log JSON
   - `cleaning_summary`       : dict with n_rows_dropped, n_cols_dropped, n_pii_masked
   - `content_hash`           : SHA-256 hash of cleaned file

Harness Engineering notes:
- The transformation log is mandatory — it must document every change made
- Never drop >50% of rows unless explicitly required by data quality flags
- PII masking is irreversible; always record original column names in transformation log

End your response with:
```session_state
{
  "cleaned_dataset_path": "<filled>",
  "transformation_log_path": "<filled>",
  "cleaning_summary": {},
  "content_hash": "<filled>"
}
```

Then write: <DONE>cleaning</DONE>
"""


def build_cleaning_agent() -> Agent:
    """Build the Data Cleaning stage agent."""
    return build_pipeline_agent(
        name="cleaning_agent",
        instructions=_SYSTEM_PROMPT,
        tools=[
            make_ds_tools_mcp(),
            make_tracking_mcp(),
            *LOCAL_TOOLS,
        ],
        use_fast_client=False,
        max_message_groups=20,
    )
