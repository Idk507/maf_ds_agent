"""
agents/base.py — Shared MCP tool instances and agent builder factories.

All MCP tool instances are created here once and reused across agent builders.
This avoids duplicate HTTP connections and ensures consistent URL config.

Pattern:
  - `make_tracking_mcp()` → MCPStreamableHTTPTool for Tracking MCP (port 8100)
  - `make_ds_tools_mcp()` → MCPStreamableHTTPTool for DS Tools MCP (port 8101)
  - `make_bug_log_mcp()` → MCPStreamableHTTPTool for Bug Log MCP (port 8102)
  - `build_pipeline_agent(...)` → Agent with pipeline_stack middleware
  - `build_observer_agent(...)` → Agent with observer_stack middleware
  - `build_debug_agent_base(...)` → Agent with debug_stack middleware (no safety)
"""
from __future__ import annotations

from typing import Any, Sequence

from agent_framework import Agent, InMemoryHistoryProvider, MCPStreamableHTTPTool, SlidingWindowStrategy

from agents.clients import FAST_CLIENT, PRIMARY_CLIENT
from agents.middleware import debug_stack, observer_stack, pipeline_stack
from config.settings import get_settings

settings = get_settings()

# ── Framework-injected kwargs that MCP servers must not receive ────────
# agent.run(function_invocation_kwargs={...}) injects these into EVERY
# tool call, including MCP tools. FastMCP rejects unknown kwargs via
# Pydantic validation (isError=True). Strip them before forwarding.
_FRAMEWORK_KWARGS: frozenset[str] = frozenset({"session_state"})


class _FilteredMCPTool(MCPStreamableHTTPTool):
    """MCPStreamableHTTPTool that strips agent-framework runtime kwargs.

    agent.run(function_invocation_kwargs={"run_id": ..., "stage_name": ...,
    "session_state": ...}) injects those kwargs into every tool call.
    Local tools handle them via FunctionInvocationContext; MCP tools reject
    them with a Pydantic ValidationError → isError=True → consecutive errors.
    This subclass intercepts call_tool and removes those keys before the MCP
    protocol call, so the MCP server only sees its declared parameters.
    """

    async def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        if getattr(self, "session", None) is None:
            await self.connect()
        # Only strip session_state — it's a large Python dict that no MCP tool
        # accepts and would cause Pydantic validation errors on every call.
        # run_id and stage_name ARE declared parameters on several MCP tools
        # (write_output, execute_code, log_metrics, verify_stage, etc.) so they
        # must be passed through. When those tools don't need them, optional
        # dummy params (default="") are declared in the server to absorb them.
        clean = {k: v for k, v in kwargs.items() if k not in _FRAMEWORK_KWARGS}
        return await super().call_tool(tool_name, **clean)


# ── MCP tool factories ────────────────────────────────────────────────
# Returns a new _FilteredMCPTool instance each time (one per agent).
# Instances are lightweight; actual HTTP sessions are established on first use.


def make_tracking_mcp() -> _FilteredMCPTool:
    """Create a Tracking MCP tool (port 8100) — audit trail, checkpoints."""
    return _FilteredMCPTool(
        name="tracking_mcp",
        url=settings.tracking_mcp_url,
        tool_name_prefix="tracking",
    )


def make_ds_tools_mcp() -> _FilteredMCPTool:
    """Create a DS Tools MCP tool (port 8101) — data science operations."""
    return _FilteredMCPTool(
        name="ds_tools_mcp",
        url=settings.ds_tools_mcp_url,
        tool_name_prefix="ds",
    )


def make_bug_log_mcp() -> _FilteredMCPTool:
    """Create a Bug Log MCP tool (port 8102) — error and repair recording."""
    return _FilteredMCPTool(
        name="bug_log_mcp",
        url=settings.bug_log_mcp_url,
        tool_name_prefix="buglog",
    )


# ── Agent builder factories ───────────────────────────────────────────


def build_pipeline_agent(
    name: str,
    instructions: str,
    tools: Sequence[Any],
    *,
    use_fast_client: bool = False,
    max_message_groups: int = 20,
) -> Agent:
    """Build a pipeline stage agent with full middleware stack.

    Args:
        name:              Agent identifier (used in logs and spans).
        instructions:      System-level instructions for this agent.
        tools:             List of tools (MCP tools + local tools).
        use_fast_client:   If True, use FAST_CLIENT; else PRIMARY_CLIENT.
        max_message_groups: Number of message groups to keep in history.

    Returns:
        An Agent ready to be called via run_ralph_loop().
    """
    client = FAST_CLIENT if use_fast_client else PRIMARY_CLIENT
    history = InMemoryHistoryProvider()
    compaction = SlidingWindowStrategy(keep_last_groups=max_message_groups)

    return Agent(
        client=client,
        instructions=instructions,
        name=name,
        tools=list(tools),
        context_providers=[history],
        middleware=pipeline_stack(),
        compaction_strategy=compaction,
    )


def build_observer_agent(
    name: str,
    instructions: str,
    tools: Sequence[Any],
    *,
    use_fast_client: bool = False,
    max_message_groups: int = 10,
) -> Agent:
    """Build an observer agent with observer middleware stack (no safety, no rate limit).

    Observer agents run independently from the pipeline and monitor events.
    """
    client = FAST_CLIENT if use_fast_client else PRIMARY_CLIENT
    history = InMemoryHistoryProvider()
    compaction = SlidingWindowStrategy(keep_last_groups=max_message_groups)

    return Agent(
        client=client,
        instructions=instructions,
        name=name,
        tools=list(tools),
        context_providers=[history],
        middleware=observer_stack(),
        compaction_strategy=compaction,
    )


def build_debug_agent_base(
    name: str,
    instructions: str,
    tools: Sequence[Any],
    *,
    max_message_groups: int = 20,
) -> Agent:
    """Build a debug agent with no safety middleware.

    The Debug Agent must be able to inspect and reason about unsafe/broken code,
    so SafetyMiddleware is intentionally excluded from its stack.
    """
    history = InMemoryHistoryProvider()
    compaction = SlidingWindowStrategy(keep_last_groups=max_message_groups)

    return Agent(
        client=PRIMARY_CLIENT,
        instructions=instructions,
        name=name,
        tools=list(tools),
        context_providers=[history],
        middleware=debug_stack(),
        compaction_strategy=compaction,
    )
