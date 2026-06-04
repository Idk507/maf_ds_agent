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
1. Use `ds_read_file` to inspect the provided file path and detect its type.
2. Use `ds_get_sample` to extract a representative sample (up to 5 rows / 2000 chars).
3. Identify any Personally Identifiable Information (PII) in the data:
   - Names, email addresses, phone numbers, SSNs, passport numbers, credit card numbers
   - Set `pii_detected` = true if any PII found, false otherwise
4. Use `ds_write_output` to copy/register the file as the raw artefact at:
   `data/artefacts/{run_id}/raw/dataset{extension}`
5. Use `tracking_record_artefact` to record the raw artefact.
6. Update session state with these keys using `set_session_state`:
   - `input_artefact_path`  : path to the raw artefact file
   - `pii_detected`         : true/false
   - `schema`               : JSON of columns, dtypes, shape (from read_file result)
   - `file_type_result`     : detected file type info (category, mime_type, confidence)
7. End your response with the completion block below.

Harness Engineering notes:
- Every file write MUST be recorded via tracking_record_artefact
- If the file path does not exist, raise an error immediately (do NOT fabricate data)
- PII detection is mandatory — err on the side of caution (false positive is OK)

After completing all steps, end your response with:
```session_state
{
  "input_artefact_path": "<filled>",
  "pii_detected": false,
  "schema": {},
  "file_type_result": {}
}
```

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
