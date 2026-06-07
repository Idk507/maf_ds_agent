"""
agents/orchestrator.py — Pipeline Orchestrator.

The Orchestrator drives the full ML pipeline end-to-end:
  1. Detects file type (Magika / filetype / heuristics)
  2. Reads file schema and sample
  3. Selects pipeline variant (tabular / document_text / image / existing_model)
  4. Records pipeline start via Tracking MCP
  5. Executes each stage in order using run_ralph_loop()
  6. On stage failure: calls Debug Agent for repair, then retries
  7. Records pipeline end with full audit trail
  8. Returns final session_state

Design notes:
  - The orchestrator itself is NOT an LLM agent — it is a Python driver class
  - Each stage IS an LLM agent wrapped in run_ralph_loop (the self-correcting loop)
  - The orchestrator uses an inner LLM Agent to generate stage-specific prompts
  - Debug Agent is called with propagate_session=False (isolated repair context)
  - Bug Log Observer and Artefact Tracking Observer run at key checkpoints
  - PIPELINE_HUMAN_IN_THE_LOOP env var triggers a gate before each stage advance
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

from agent_framework import Agent, AgentSession, InMemoryHistoryProvider, SlidingWindowStrategy

from agents.base import make_bug_log_mcp, make_ds_tools_mcp, make_tracking_mcp
from agents.clients import PRIMARY_CLIENT
from agents.middleware import pipeline_stack
from agents.artefact_tracking_observer import build_artefact_tracking_observer
from agents.bug_log_observer import build_bug_log_observer
from agents.cleaning_agent import build_cleaning_agent
from agents.debug_agent import build_debug_agent
from agents.deployment_agent import build_deployment_agent
from agents.eda_agent import build_eda_agent
from agents.evaluation_agent import build_evaluation_agent
from agents.explainability_agent import build_explainability_agent
from agents.feature_agent import build_feature_agent
from agents.ingestion_agent import build_ingestion_agent
from agents.report_agent import build_report_agent
from agents.training_agent import build_training_agent
from agents.tuning_agent import build_tuning_agent
from config.settings import get_settings
from tools.file_type_detector import detect_file_type
from tools.readers.dispatcher import read_file_dispatch
from workflows.pipeline_graph import get_pipeline_stages, select_variant
from workflows.ralph_loop import run_ralph_loop

logger = logging.getLogger(__name__)

settings = get_settings()

# ── Orchestrator prompt-generator system prompt ───────────────────────

_PROMPT_GEN_INSTRUCTIONS = """You are the Pipeline Prompt Generator for an automated ML pipeline.

Given the pipeline stage name, task description, and current session state, generate a
concise, specific prompt for that stage's LLM agent. The prompt should:
1. State the exact goal for this stage
2. Reference relevant session_state values (file paths, column names, task type)
3. Remind the agent of the DONE tag format required
4. Be 150-300 words maximum

Respond with ONLY the stage prompt text — no preamble, no markdown fences.
"""


class PipelineOrchestrator:
    """
    End-to-end ML pipeline driver.

    Usage:
        orchestrator = PipelineOrchestrator()
        result = await orchestrator.run_pipeline(
            task_description="Predict customer churn",
            file_path="/data/customers.csv"
        )
    """

    def __init__(self) -> None:
        self._settings = get_settings()

        # ── MCP tool instances (shared across agents) ─────────────────
        self._tracking_mcp = make_tracking_mcp()
        self._ds_tools_mcp = make_ds_tools_mcp()
        self._bug_log_mcp = make_bug_log_mcp()

        # ── Stage agents ──────────────────────────────────────────────
        self._stage_agents: dict[str, Agent] = {
            "ingestion": build_ingestion_agent(),
            "eda": build_eda_agent(),
            "cleaning": build_cleaning_agent(),
            "feature_engineering": build_feature_agent(),
            "training": build_training_agent(),
            "tuning": build_tuning_agent(),
            "evaluation": build_evaluation_agent(),
            "explainability": build_explainability_agent(),
            "report": build_report_agent(),
            "deployment": build_deployment_agent(),
        }

        # ── Support agents ────────────────────────────────────────────
        self._debug_agent = build_debug_agent()
        self._bug_log_observer = build_bug_log_observer()
        self._tracking_observer = build_artefact_tracking_observer()

        # ── Prompt generator LLM agent ────────────────────────────────
        self._prompt_gen = Agent(
            client=PRIMARY_CLIENT,
            instructions=_PROMPT_GEN_INSTRUCTIONS,
            name="prompt_generator",
            context_providers=[InMemoryHistoryProvider()],
            middleware=pipeline_stack(),
            compaction_strategy=SlidingWindowStrategy(keep_last_groups=10),
        )

        # Human-in-the-loop gate flag
        self._human_gate: bool = os.environ.get("PIPELINE_HUMAN_IN_THE_LOOP", "false").lower() == "true"
        self._max_iterations: int = int(os.environ.get("PIPELINE_MAX_RALPH_ITERATIONS", "8"))

    # ── Public API ────────────────────────────────────────────────────

    async def run_pipeline(
        self,
        task_description: str,
        file_path: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full ML pipeline for the given task and input file.

        Args:
            task_description: Natural language description of the ML task.
            file_path:        Absolute path to the input dataset/model file.
            run_id:           Optional explicit run identifier; auto-generated if None.

        Returns:
            Final session_state dict with all stage results.
        """
        run_id = run_id or str(uuid.uuid4())
        logger.info("orchestrator.start run_id=%s task=%s file=%s", run_id, task_description, file_path)

        # ── Step 1: Detect file type ──────────────────────────────────
        detection = detect_file_type(file_path)
        logger.info("orchestrator.file_type category=%s mime=%s conf=%.2f",
                    detection.category, detection.mime_type, detection.confidence)

        # ── Step 2: Read schema ───────────────────────────────────────
        file_info = read_file_dispatch(file_path, detection, sample_rows=5)

        # ── Step 3: Select pipeline variant ──────────────────────────
        variant = select_variant(detection.category)
        stages = get_pipeline_stages(variant)
        logger.info("orchestrator.variant variant=%s stages=%s", variant, stages)

        # ── Step 4: Build initial session state ───────────────────────
        session_state: dict[str, Any] = {
            "run_id": run_id,
            "task_description": task_description,
            "pipeline_variant": variant,
            "file_type_result": {
                "category": detection.category,
                "mime_type": detection.mime_type,
                "confidence": detection.confidence,
                "source": detection.source,
            },
            "schema": file_info.get("schema", {}),
            "input_artefact_path": file_path,
            "human_gate_decisions": [],
        }

        # ── Step 5: Record pipeline start ─────────────────────────────
        try:
            await self._tracking_mcp.call_tool(
                "record_start",
                run_id=run_id,
                task_description=task_description,
                pipeline_variant=variant,
                input_file=file_path,
                file_type=detection.category,
            )
        except Exception as exc:
            logger.warning("orchestrator.tracking_start_error: %s", exc)

        # ── Step 6: Execute stages ────────────────────────────────────
        shared_session = AgentSession()

        for stage_name in stages:
            session_state, success = await self._run_stage(
                stage_name=stage_name,
                session_state=session_state,
                run_id=run_id,
                shared_session=shared_session,
            )
            if not success:
                logger.error("orchestrator.stage_failed stage=%s run_id=%s", stage_name, run_id)
                # Escalation: call bug log observer then continue (don't abort whole pipeline)
                await self._run_observer_check(run_id, session_state)

            # Human-in-the-loop gate (if enabled)
            if self._human_gate and stage_name != stages[-1]:
                decision = await self._human_gate_check(stage_name, session_state)
                session_state["human_gate_decisions"].append({
                    "after_stage": stage_name,
                    "decision": decision,
                })
                if decision == "abort":
                    logger.info("orchestrator.human_aborted stage=%s run_id=%s", stage_name, run_id)
                    break

        # ── Step 7: Record pipeline end ───────────────────────────────
        try:
            await self._tracking_mcp.call_tool(
                "record_end",
                run_id=run_id,
                status="completed",
                final_session_state_json=json.dumps(session_state, default=str),
            )
        except Exception as exc:
            logger.warning("orchestrator.tracking_end_error: %s", exc)

        logger.info("orchestrator.complete run_id=%s", run_id)
        return session_state

    # ── Private helpers ───────────────────────────────────────────────

    async def _run_stage(
        self,
        stage_name: str,
        session_state: dict[str, Any],
        run_id: str,
        shared_session: AgentSession,
    ) -> tuple[dict[str, Any], bool]:
        """Run a single pipeline stage inside run_ralph_loop.

        If the stage fails all iterations, the Debug Agent is called for repair.
        Returns (updated_session_state, success).
        """
        agent = self._stage_agents.get(stage_name)
        if agent is None:
            logger.error("orchestrator.unknown_stage stage=%s", stage_name)
            return session_state, False

        # Generate stage-specific prompt via LLM prompt generator
        prompt = await self._generate_stage_prompt(stage_name, session_state)

        logger.info("orchestrator.stage_start stage=%s run_id=%s", stage_name, run_id)

        session_state, iterations, success = await run_ralph_loop(
            agent=agent,
            stage_name=stage_name,
            prompt=prompt,
            session_state=session_state,
            run_id=run_id,
            tracking_mcp_client=None,  # direct call_tool on MCPTool without active session fails
            ds_tools_mcp_client=None,  # use local criteria fallback (always available, no session needed)
            max_iterations=self._max_iterations,
        )

        if not success:
            # Call Debug Agent to repair the failure
            logger.warning("orchestrator.calling_debug_agent stage=%s", stage_name)
            session_state = await self._call_debug_agent(
                stage_name=stage_name,
                session_state=session_state,
                run_id=run_id,
            )

        return session_state, success

    async def _generate_stage_prompt(
        self, stage_name: str, session_state: dict[str, Any]
    ) -> str:
        """Use the LLM prompt generator to create a tailored stage prompt."""
        run_id = session_state.get("run_id", "")
        context = {
            "stage_name": stage_name,
            "run_id": run_id,
            "task_description": session_state.get("task_description", ""),
            "pipeline_variant": session_state.get("pipeline_variant", ""),
            "relevant_state_keys": {
                k: v for k, v in session_state.items()
                if k in (
                    "input_artefact_path", "cleaned_dataset_path", "features_train_path",
                    "model_artefact_path", "target_column", "task_type", "schema",
                )
            },
        }
        request = (
            f"Generate a prompt for the '{stage_name}' pipeline stage.\n"
            f"IMPORTANT: The run_id for this pipeline run is '{run_id}'. "
            f"Use this exact value wherever run_id is needed in tool calls or file paths.\n"
            f"Context: {json.dumps(context, default=str)}"
        )
        try:
            response = await self._prompt_gen.run(
                request,
                function_invocation_kwargs={"run_id": run_id, "session_state": session_state},
            )
            prompt_text = response.text.strip()
            if prompt_text:
                # Prepend run_id reminder so LLM agents never use placeholders
                return (
                    f"[Pipeline run_id: {run_id}]\n\n"
                    + prompt_text
                )
        except Exception as exc:
            logger.warning("orchestrator.prompt_gen_error: %s", exc)

        # Fallback: generic stage prompt
        return (
            f"[Pipeline run_id: {run_id}]\n\n"
            f"Execute the '{stage_name}' stage of the ML pipeline.\n"
            f"Task: {session_state.get('task_description', '')}\n"
            f"Input file: {session_state.get('input_artefact_path', '')}\n"
            f"Complete all required steps for this stage and end with: <DONE>{stage_name}</DONE>"
        )

    async def _call_debug_agent(
        self,
        stage_name: str,
        session_state: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any]:
        """Call the Debug Agent after a Ralph Loop exhaustion."""
        escalation_key = f"escalation_report_{stage_name}"
        escalation = session_state.get(escalation_key, {})

        debug_prompt = (
            f"Pipeline stage '{stage_name}' has failed all Ralph Loop iterations.\n\n"
            f"Error report: {json.dumps(escalation, default=str)}\n\n"
            f"Task description: {session_state.get('task_description', '')}\n"
            f"Session state summary: {json.dumps({k: str(v)[:200] for k, v in session_state.items()}, default=str)}\n\n"
            f"Apply the 5-attempt repair protocol and return the repaired code/approach."
        )

        try:
            debug_response = await self._debug_agent.run(
                debug_prompt,
                function_invocation_kwargs={
                    "run_id": run_id,
                    "stage_name": stage_name,
                    "session_state": session_state,
                },
            )
            # Record debug attempts count
            session_state["debug_attempts"] = session_state.get("debug_attempts", 0) + 1

            # Record bug in Bug Log MCP
            await self._bug_log_mcp.call_tool(
                "record_bug",
                run_id=run_id,
                stage_name=stage_name,
                error_message=str(escalation.get("last_failures", "")),
                error_severity="high",
                debug_agent_response=debug_response.text[:2000],
            )
        except Exception as exc:
            logger.error("orchestrator.debug_agent_error: %s", exc)

        return session_state

    async def _run_observer_check(
        self, run_id: str, session_state: dict[str, Any]
    ) -> None:
        """Run Bug Log and Artefact Tracking observers as background health checks."""
        try:
            await self._bug_log_observer.run(
                f"Check bug status for run_id={run_id}",
                function_invocation_kwargs={"run_id": run_id, "session_state": session_state},
            )
        except Exception as exc:
            logger.warning("orchestrator.bug_observer_error: %s", exc)

        try:
            await self._tracking_observer.run(
                f"Check artefact lineage for run_id={run_id}",
                function_invocation_kwargs={"run_id": run_id, "session_state": session_state},
            )
        except Exception as exc:
            logger.warning("orchestrator.tracking_observer_error: %s", exc)

    async def _human_gate_check(self, after_stage: str, session_state: dict) -> str:
        """Human-in-the-loop gate. Returns 'continue' or 'abort'."""
        print(f"\n[HUMAN GATE] Stage '{after_stage}' completed.")
        print(f"  Deployment recommendation: {session_state.get('deployment_recommendation', 'N/A')}")
        print(f"  Baseline metrics: {json.dumps(session_state.get('baseline_metrics', {}), indent=2)}")
        decision = input("Continue pipeline? [Y/n]: ").strip().lower()
        return "abort" if decision in ("n", "no", "abort") else "continue"


def build_orchestrator() -> PipelineOrchestrator:
    """Factory function — construct a fully initialised PipelineOrchestrator."""
    return PipelineOrchestrator()
