"""
agents/ingestion_agent.py — Data Ingestion Stage Agent.

Responsibilities:
  - Detect file type (Magika + filetype + heuristics)
  - Read raw schema and sample data
  - Detect PII (names, emails, SSNs, phone numbers)
  - Write raw dataset to data/artefacts/{run_id}/raw/ and record lineage
  - Update session_state with: file_type_result, schema, input_artefact_path, pii_detected
  - End response with <DONE>ingestion</DONE>

Client: FAST_CLIENT (data reading is fast, no complex reasoning needed)
MCP tools: DS Tools (read_file, get_sample, web_research, search_docs, write_output)
           Tracking (record_artefact, record_lineage, record_metric)
Local tools: get_session_state, set_session_state, check_artefact_exists
"""
from __future__ import annotations

from agent_framework import Agent

from agents.base import build_pipeline_agent, make_ds_tools_mcp, make_tracking_mcp
from tools.local_tools import LOCAL_TOOLS

_SYSTEM_PROMPT = """You are the Data Ingestion Agent for an automated ML pipeline.

Your task is to:
1. Call `get_session_state` to retrieve the current session state. The `file_path` key contains the original input file path.
2. Use `check_artefact_exists` to verify the original file exists.
3. Use `ds_read_file` to inspect the file: get column names, dtypes, shape, separator, and encoding.
4. Identify any Personally Identifiable Information (PII) in the data:
   - Names, email addresses, phone numbers, SSNs, passport numbers, credit card numbers
   - Set `pii_detected` = true if any PII found, false otherwise
5. Use `ds_execute_code` to copy the full file to the outputs directory. Example code:
   ```python
   import shutil, os
   run_id = "<run_id from session_state>"
   src = "<file_path from session_state>"
   dst = f"outputs/{run_id}/raw/dataset.csv"
   os.makedirs(os.path.dirname(dst), exist_ok=True)
   shutil.copy2(src, dst)
   print(dst)
   ```
   The printed output is the artefact path.
6. Use `tracking_record_artefact` to record the raw artefact (use the dst path from step 5).
7. Update session state with these EXACT keys using `set_session_state` (call once per key):
   - `input_artefact_path`  : the dst path from ds_execute_code (e.g. "outputs/{run_id}/raw/dataset.csv")
   - `pii_detected`         : true or false (boolean)
   - `schema`               : JSON string of columns, dtypes, shape (from ds_read_file result)
   - `file_type_result`     : string describing the detected file type (csv/tabular/etc.)
8. End your response with the exact completion block below.

CRITICAL — input_artefact_path:
- MUST be the path where you copied the file in step 5 (e.g. "outputs/{run_id}/raw/dataset.csv")
- MUST be a file that actually EXISTS on disk
- Do NOT use the original file_path — use the COPIED path
- Do NOT use placeholder text like "<filled>"

After completing all steps, end your response with:
```session_state
{
  "input_artefact_path": "outputs/{run_id}/raw/dataset.csv",
  "pii_detected": false,
  "schema": "{}",
  "file_type_result": "tabular/csv"
}
```
(Fill in the actual run_id and real values above — do NOT use placeholders.)

Then write: <DONE>ingestion</DONE>
"""


def build_ingestion_agent() -> Agent:
    """Build the Data Ingestion stage agent."""
    return build_pipeline_agent(
        name="ingestion_agent",
        instructions=_SYSTEM_PROMPT,
        tools=[
            make_ds_tools_mcp(),
            make_tracking_mcp(),
            *LOCAL_TOOLS,
        ],
        use_fast_client=True,
        max_message_groups=20,
    )
