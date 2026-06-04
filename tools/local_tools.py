"""
tools/local_tools.py — Local tools registered on every pipeline agent.

These tools give agents access to the shared session_state dict and file system
checks. They are injected via function_invocation_kwargs at agent.run() time.

Tool functions that accept `FunctionInvocationContext` get the context injected
automatically by the agent framework — the LLM does NOT see or set those parameters.

Tools provided:
  get_session_state        — Return the current session state as JSON string
  set_session_state        — Write a key/value pair into session state
  check_artefact_exists    — Check if a file/directory path exists on disk
  format_verification_result — Format Ralph Loop failure list for agent display
"""
from __future__ import annotations

import json
import os
from typing import Any

from agent_framework import tool, FunctionInvocationContext


@tool
def get_session_state(ctx: FunctionInvocationContext) -> str:
    """Return the current pipeline session state as a JSON string.

    Call this to inspect what previous pipeline stages have recorded
    (e.g. file paths, schema info, metrics).
    """
    session_state: dict[str, Any] = ctx.kwargs.get("session_state", {})
    return json.dumps(session_state, indent=2, default=str)


@tool
def set_session_state(key: str, value_json: str, ctx: FunctionInvocationContext) -> str:
    """Write a key/value pair into the shared pipeline session state.

    Args:
        key:        The session state key to set (e.g. "cleaned_dataset_path").
        value_json: The value to store, serialized as a JSON string.
                    Use json.dumps() for complex objects, or a plain quoted string.

    Returns:
        Confirmation string with the key that was set.
    """
    session_state: dict[str, Any] = ctx.kwargs.get("session_state", {})
    try:
        value = json.loads(value_json)
    except json.JSONDecodeError:
        value = value_json  # treat as plain string if not valid JSON
    session_state[key] = value
    return f"session_state['{key}'] = {json.dumps(value, default=str)}"


@tool
def check_artefact_exists(path: str) -> str:
    """Check whether a file or directory exists on disk.

    Args:
        path: Absolute or workspace-relative path to check.

    Returns:
        JSON string with 'exists' (bool), 'is_file', 'is_dir', 'size_bytes'.
    """
    exists = os.path.exists(path)
    result: dict[str, Any] = {
        "path": path,
        "exists": exists,
        "is_file": os.path.isfile(path) if exists else False,
        "is_dir": os.path.isdir(path) if exists else False,
        "size_bytes": os.path.getsize(path) if (exists and os.path.isfile(path)) else None,
    }
    return json.dumps(result)


@tool
def format_verification_result(failures_json: str) -> str:
    """Format a list of Ralph Loop verification failures into a readable report.

    Args:
        failures_json: JSON array of failure dicts, each with 'assertion',
                       'expected', 'actual' keys (from verify_stage tool).

    Returns:
        Human-readable string listing each failing assertion.
    """
    try:
        failures: list[dict[str, str]] = json.loads(failures_json)
    except json.JSONDecodeError:
        return f"Could not parse failures JSON: {failures_json}"

    if not failures:
        return "All verification criteria passed. No failures."

    lines = [f"Verification found {len(failures)} failure(s):"]
    for i, f in enumerate(failures, 1):
        lines.append(
            f"  [{i}] Assertion : {f.get('assertion', '?')}\n"
            f"       Expected  : {f.get('expected', '?')}\n"
            f"       Actual    : {f.get('actual', '?')}"
        )
    return "\n".join(lines)


# ── Convenience list for use in agent builders ────────────────────────

LOCAL_TOOLS = [
    get_session_state,
    set_session_state,
    check_artefact_exists,
    format_verification_result,
]
