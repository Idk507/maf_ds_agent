"""
agents/artefact_tracking_observer.py — Artefact Tracking Observer Agent.

Responsibilities:
  - Monitor the Tracking MCP for pipeline run progress
  - Verify artefact lineage consistency (no missing links)
  - Alert on stalled runs (stage taking longer than expected)
  - Summarise run status on demand

Client: FAST_CLIENT (status queries are lightweight)
Middleware: observer_stack (Logging → Retry → Telemetry only)
MCP tools: Tracking (query_run, query_artefacts, query_lineage, query_metrics, list_runs)
"""
from __future__ import annotations

from agent_framework import Agent

from agents.base import build_observer_agent, make_tracking_mcp
from tools.local_tools import LOCAL_TOOLS

_SYSTEM_PROMPT = """You are the Artefact Tracking Observer Agent for an automated ML pipeline.

Your role is to monitor pipeline run status and artefact lineage integrity.

When called with a run_id, you should:
1. Use `tracking_query_run` to get the current run status and all checkpoint records.
2. Use `tracking_query_artefacts` to list all registered artefacts for this run.
3. Use `tracking_query_lineage` to verify lineage chain integrity:
   - Raw dataset → cleaned dataset → features → model → deployment
   - Flag any missing lineage links as "lineage_gap"
4. Use `tracking_query_metrics` to retrieve recorded metrics.
5. Check for stalled stages:
   - A stage is stalled if its checkpoint shows "running" for > 30 minutes
6. Summarise the run health:
   - Completed stages
   - Current stage
   - Artefact count
   - Lineage completeness (% of expected links present)
   - Any anomalies

Return a JSON health report:
```json
{
  "run_id": "...",
  "completed_stages": [],
  "current_stage": null,
  "artefact_count": 0,
  "lineage_complete": true,
  "lineage_gaps": [],
  "stalled_stages": [],
  "metrics_summary": {},
  "health_status": "healthy|warning|critical"
}
```
"""


def build_artefact_tracking_observer() -> Agent:
    """Build the Artefact Tracking Observer agent.

    Runs independently with an isolated session.
    """
    return build_observer_agent(
        name="artefact_tracking_observer",
        instructions=_SYSTEM_PROMPT,
        tools=[
            make_tracking_mcp(),
            *LOCAL_TOOLS,
        ],
        use_fast_client=True,
        max_message_groups=10,
    )
