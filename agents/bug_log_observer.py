"""
agents/bug_log_observer.py — Bug Log Observer Agent.

Responsibilities:
  - Passively monitor the Bug Log MCP for new unresolved bugs
  - Analyse recurring patterns and update bug pattern database
  - Emit alerts when critical bugs appear (error_severity='critical')
  - Run independently from the main pipeline (separate session, no propagation)

Client: PRIMARY_CLIENT (pattern analysis requires reasoning)
Middleware: observer_stack (Logging → Retry → Telemetry only — no rate limit/safety)
MCP tools: Bug Log (query_unresolved_bugs, query_bug_patterns, record_bug)
           DS Tools (search_docs, web_research)
"""
from __future__ import annotations

from agent_framework import Agent

from agents.base import build_observer_agent, make_bug_log_mcp, make_ds_tools_mcp
from tools.local_tools import LOCAL_TOOLS

_SYSTEM_PROMPT = """You are the Bug Log Observer Agent for an automated ML pipeline.

Your role is to monitor the Bug Log MCP server and analyse error patterns.

When called, you should:
1. Use `buglog_query_unresolved_bugs` to list all unresolved bugs for this run.
2. Use `buglog_query_bug_patterns` to identify recurring error patterns.
3. For each CRITICAL severity bug:
   - Analyse the error and suggest a fix
   - Use `ds_search_docs` or `ds_web_research` to look up known solutions
   - Record your analysis using `buglog_record_bug` with an "analysis" note
4. Summarise the bug landscape:
   - Total unresolved bugs
   - Critical count
   - Most common error type
   - Recommended next action

This is an observation-only role — you do NOT modify the pipeline or retry failed stages.
You provide intelligence that the orchestrator uses to make repair decisions.

Return a JSON summary:
```json
{
  "total_unresolved": 0,
  "critical_count": 0,
  "warning_count": 0,
  "patterns_detected": [],
  "recommended_action": "continue|pause|escalate",
  "analysis_notes": "..."
}
```
"""


def build_bug_log_observer() -> Agent:
    """Build the Bug Log Observer agent.

    This agent runs independently with an isolated session (propagate_session=False).
    """
    return build_observer_agent(
        name="bug_log_observer",
        instructions=_SYSTEM_PROMPT,
        tools=[
            make_bug_log_mcp(),
            make_ds_tools_mcp(),
            *LOCAL_TOOLS,
        ],
        use_fast_client=False,
        max_message_groups=10,
    )
