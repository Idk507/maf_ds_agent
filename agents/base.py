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


# ── MCP tool factories ────────────────────────────────────────────────
# Returns a new MCPStreamableHTTPTool instance each time (one per agent).
# Instances are lightweight; actual HTTP sessions are established on first use.


def make_tracking_mcp() -> MCPStreamableHTTPTool:
    """Create a Tracking MCP tool (port 8100) — audit trail, checkpoints."""
    return MCPStreamableHTTPTool(
        name="tracking_mcp",
        url=settings.tracking_mcp_url,
        tool_name_prefix="tracking",
    )


def make_ds_tools_mcp() -> MCPStreamableHTTPTool:
    """Create a DS Tools MCP tool (port 8101) — data science operations."""
    return MCPStreamableHTTPTool(
        name="ds_tools_mcp",
        url=settings.ds_tools_mcp_url,
        tool_name_prefix="ds",
    )


def make_bug_log_mcp() -> MCPStreamableHTTPTool:
    """Create a Bug Log MCP tool (port 8102) — error and repair recording."""
    return MCPStreamableHTTPTool(
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
