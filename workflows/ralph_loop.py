"""
workflows/ralph_loop.py — The core self-correcting execution loop.

Ralph Loop (Retry And Loop for Pipeline Health):
  1. Run agent for the given stage with the current prompt
  2. Check for <DONE>stage_name</DONE> tag in agent response
  3. If not DONE: increment iteration, re-run (up to max_iterations)
  4. If DONE: call verify_stage (deterministic criteria check)
  5. If criteria pass: record checkpoint via Tracking MCP → return success
  6. If criteria fail: inject structured failure feedback → retry
  7. If max_iterations reached: escalate (return failure with full audit)

Design notes:
  - Inner loop is stateless (agent re-run from fresh prompt each iteration)
  - Outer loop carries mutable session_state across iterations
  - Failure feedback is specific: assertion names, expected vs actual, missing files
  - The loop is the continuous verification backbone (Harness Engineering principle)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_DONE_TAG_RE = re.compile(r"<DONE>(\w+)</DONE>", re.IGNORECASE)

_FAILURE_FEEDBACK_TEMPLATE = """\
=== RALPH LOOP ITERATION {iteration} — VERIFICATION FAILED ===

The stage '{stage_name}' completed with <DONE>{stage_name}</DONE> but the
deterministic exit-gate found {failure_count} failing assertion(s):

{failure_details}

Fix each failure listed above, then end your response with <DONE>{stage_name}</DONE>.
Do NOT repeat work that already passed. Focus only on the failed assertions.
"""


def _format_failure_details(failures: list[dict[str, str]]) -> str:
    lines = []
    for i, f in enumerate(failures, 1):
        lines.append(
            f"  [{i}] assertion : {f.get('assertion', '?')}\n"
            f"       expected  : {f.get('expected', '?')}\n"
            f"       actual    : {f.get('actual', '?')}"
        )
    return "\n\n".join(lines)


def _extract_done_stage(text: str) -> str | None:
    """Return the stage name from the first <DONE>stage_name</DONE> tag, or None."""
    m = _DONE_TAG_RE.search(text)
    return m.group(1).lower() if m else None


async def run_ralph_loop(
    agent: Any,
    stage_name: str,
    prompt: str,
    session_state: dict[str, Any],
    run_id: str,
    tracking_mcp_client: Any | None = None,
    ds_tools_mcp_client: Any | None = None,
    max_iterations: int = 8,
) -> tuple[dict[str, Any], int, bool]:
    """
    Run the Ralph Loop for a single pipeline stage.

    Args:
        agent:               The ChatAgent instance for this stage.
        stage_name:          Pipeline stage identifier (e.g. "eda", "training").
        prompt:              Initial user prompt for this stage.
        session_state:       Mutable dict of inter-stage state (updated in-place).
        run_id:              Unique pipeline run identifier.
        tracking_mcp_client: Optional pre-connected MCP client for checkpoint recording.
        ds_tools_mcp_client: Optional pre-connected MCP client for verify_stage calls.
        max_iterations:      Maximum retry attempts before escalation.

    Returns:
        (session_state, iterations_taken, passed)
        - session_state: updated with any values written by the agent
        - iterations_taken: number of loop iterations consumed
        - passed: True if all criteria passed before max_iterations
    """
    current_prompt = prompt
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        logger.info(
            "ralph_loop.iteration",
            extra={"stage": stage_name, "run_id": run_id, "iteration": iteration},
        )

        # ── Run agent ─────────────────────────────────────────────────
        try:
            response = await agent.run(
                current_prompt,
                function_invocation_kwargs={
                    "run_id": run_id,
                    "stage_name": stage_name,
                    "session_state": session_state,
                },
            )
        except Exception as exc:
            logger.error(
                "ralph_loop.agent_error",
                extra={"stage": stage_name, "run_id": run_id, "iteration": iteration, "error": str(exc)},
            )
            current_prompt = (
                f"The previous attempt raised an exception:\n{exc}\n\n"
                f"Fix the issue and complete stage '{stage_name}'. "
                f"End with <DONE>{stage_name}</DONE>."
            )
            continue

        response_text: str = response.text if response is not None else ""

        # ── Check for DONE tag ────────────────────────────────────────
        done_stage = _extract_done_stage(response_text)
        if done_stage != stage_name.lower():
            logger.warning(
                "ralph_loop.no_done_tag",
                extra={"stage": stage_name, "run_id": run_id, "iteration": iteration},
            )
            current_prompt = (
                f"Your response did not include the completion tag.\n"
                f"Complete stage '{stage_name}' and end your response with: "
                f"<DONE>{stage_name}</DONE>"
            )
            continue

        # ── Parse updated session_state from response (if any) ────────
        _merge_session_state_from_response(response_text, session_state)

        # ── verify_stage via ds_tools MCP or inline ───────────────────
        failures = await _verify_stage(stage_name, run_id, session_state, ds_tools_mcp_client)

        if not failures:
            # ── All criteria passed — record checkpoint ────────────────
            logger.info(
                "ralph_loop.passed",
                extra={"stage": stage_name, "run_id": run_id, "iteration": iteration},
            )
            if tracking_mcp_client:
                try:
                    await tracking_mcp_client.call_tool(
                        "record_checkpoint",
                        run_id=run_id,
                        stage_name=stage_name,
                        iteration=iteration,
                        status="passed",
                        notes=f"Ralph Loop: all criteria satisfied in {iteration} iteration(s)",
                    )
                except Exception as track_exc:
                    logger.warning("ralph_loop.tracking_error: %s", track_exc)
            session_state["ralph_loop_iteration"] = iteration
            return session_state, iteration, True

        # ── Criteria failed — inject failure feedback ─────────────────
        logger.warning(
            "ralph_loop.failed",
            extra={
                "stage": stage_name,
                "run_id": run_id,
                "iteration": iteration,
                "failures": len(failures),
            },
        )

        if tracking_mcp_client:
            try:
                await tracking_mcp_client.call_tool(
                    "record_checkpoint",
                    run_id=run_id,
                    stage_name=stage_name,
                    iteration=iteration,
                    status="failed",
                    notes=f"{len(failures)} assertion(s) failed",
                )
            except Exception:
                pass

        failure_details = _format_failure_details(failures)
        current_prompt = _FAILURE_FEEDBACK_TEMPLATE.format(
            iteration=iteration,
            stage_name=stage_name,
            failure_count=len(failures),
            failure_details=failure_details,
        )

    # ── Max iterations reached ────────────────────────────────────────
    logger.error(
        "ralph_loop.max_iterations",
        extra={"stage": stage_name, "run_id": run_id, "max": max_iterations},
    )

    escalation_key = f"escalation_report_{stage_name}"
    session_state[escalation_key] = {
        "stage": stage_name,
        "run_id": run_id,
        "iterations_exhausted": max_iterations,
        "last_failures": await _verify_stage(stage_name, run_id, session_state, ds_tools_mcp_client),
    }
    session_state["ralph_loop_iteration"] = iteration
    return session_state, iteration, False


# ── Helpers ───────────────────────────────────────────────────────────


def _merge_session_state_from_response(response_text: str, session_state: dict) -> None:
    """
    Look for a JSON block tagged ```session_state ... ``` in the agent's response.
    Merge any keys found into session_state (agent convention for updating shared state).
    """
    pattern = re.compile(r"```session_state\s*(.*?)```", re.DOTALL)
    for match in pattern.finditer(response_text):
        try:
            update = json.loads(match.group(1).strip())
            if isinstance(update, dict):
                session_state.update(update)
        except json.JSONDecodeError:
            pass


async def _verify_stage(
    stage_name: str,
    run_id: str,
    session_state: dict,
    ds_tools_mcp_client: Any | None,
) -> list[dict[str, str]]:
    """
    Run deterministic criteria check.
    Prefers calling the MCP tool (remote), falls back to direct import.
    """
    if ds_tools_mcp_client:
        try:
            result = await ds_tools_mcp_client.call_tool(
                "verify_stage",
                run_id=run_id,
                stage_name=stage_name,
                session_state_json=json.dumps(session_state),
            )
            result_text = result if isinstance(result, str) else str(result)
            parsed = json.loads(result_text)
            return parsed.get("failures", [])
        except Exception as exc:
            logger.warning("ralph_loop._verify_stage MCP error: %s; falling back to local", exc)

    # Local fallback
    from workflows.criteria import get_criteria

    criteria = get_criteria(stage_name)
    return criteria.verify(run_id, session_state)
