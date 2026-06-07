"""
Integration tests for the three MCP servers.

These tests require the servers to be running on their configured ports. They
exercise the same MCPStreamableHTTPTool path used by the orchestrator, rather
than raw JSON-RPC calls that do not establish a streamable-HTTP session.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import uuid
from pathlib import Path

import httpx
import pytest

from agents.base import _FilteredMCPTool
from workflows.ralph_loop import _mcp_result_text


TRACKING_URL = os.environ.get("TRACKING_MCP_URL", "http://localhost:8100")
DS_TOOLS_URL = os.environ.get("DS_TOOLS_MCP_URL", "http://localhost:8101")
BUG_LOG_URL = os.environ.get("BUG_LOG_MCP_URL", "http://localhost:8102")


def _health_url(base_url: str) -> str:
    return base_url.split("/mcp/mcp", 1)[0].rstrip("/") + "/health"


def _mcp_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/mcp/mcp") else f"{base}/mcp/mcp"


def _servers_reachable() -> bool:
    for url in [TRACKING_URL, DS_TOOLS_URL, BUG_LOG_URL]:
        try:
            r = httpx.get(_health_url(url), timeout=3.0)
            if r.status_code != 200:
                return False
        except Exception:
            return False
    return True


pytestmark = pytest.mark.skipif(
    not _servers_reachable(),
    reason="MCP servers not reachable; start them with: python scripts/start_servers.py",
)


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_tool(base_url: str, prefix: str, tool_name: str, **kwargs) -> str:
    tool = _FilteredMCPTool(
        name=f"{prefix}_integration",
        url=_mcp_url(base_url),
        tool_name_prefix=prefix,
    )
    try:
        result = await tool.call_tool(tool_name, **kwargs)
        return _mcp_result_text(result)
    finally:
        await _maybe_await(tool.close())


@pytest.fixture(scope="module")
def run_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


class TestHealthEndpoints:
    def test_tracking_health(self):
        r = httpx.get(_health_url(TRACKING_URL), timeout=5)
        assert r.status_code == 200
        assert r.json()["server"] == "tracking"

    def test_ds_tools_health(self):
        r = httpx.get(_health_url(DS_TOOLS_URL), timeout=5)
        assert r.status_code == 200
        assert r.json()["server"] == "ds_tools"

    def test_bug_log_health(self):
        r = httpx.get(_health_url(BUG_LOG_URL), timeout=5)
        assert r.status_code == 200
        assert r.json()["server"] == "bug_log"


class TestTrackingRoundTrip:
    def test_full_tracking_lifecycle(self, run_id):
        rid = f"{run_id}-tracking"

        start = asyncio.run(
            _call_tool(
                TRACKING_URL,
                "tracking",
                "record_start",
                run_id=rid,
                task_description="Integration test tracking",
                pipeline_variant="tabular",
            )
        )
        assert f"Run {rid} started" in start

        checkpoint = asyncio.run(
            _call_tool(
                TRACKING_URL,
                "tracking",
                "record_checkpoint",
                run_id=rid,
                stage_name="ingestion",
                iteration=1,
                status="passed",
                notes="integration",
            )
        )
        assert "Checkpoint recorded" in checkpoint

        end = asyncio.run(
            _call_tool(
                TRACKING_URL,
                "tracking",
                "record_end",
                run_id=rid,
                status="success",
                error="",
            )
        )
        assert f"Run {rid} ended" in end

        queried = asyncio.run(
            _call_tool(TRACKING_URL, "tracking", "query_run", run_id=rid)
        )
        assert rid in queried


class TestBugLogRoundTrip:
    def test_bug_lifecycle(self, run_id):
        rid = f"{run_id}-buglog"
        bug_id = f"{rid}-bug"

        recorded = asyncio.run(
            _call_tool(
                BUG_LOG_URL,
                "buglog",
                "record_bug",
                bug_id=bug_id,
                run_id=rid,
                stage_name="integration_test",
                error_type="IntegrationTestError",
                error_message="Test bug recorded by integration test",
                traceback="No traceback",
                library_name="",
            )
        )
        assert f"Bug {bug_id} recorded" in recorded

        unresolved = asyncio.run(
            _call_tool(BUG_LOG_URL, "buglog", "query_unresolved_bugs", run_id=rid)
        )
        assert bug_id in unresolved


class TestDSTools:
    def test_verify_stage(self, run_id, tmp_path):
        artefact = tmp_path / "input.csv"
        artefact.write_text("x,y\n1,2\n", encoding="utf-8")
        session_state = {
            "file_type_result": {"category": "tabular"},
            "schema": {"columns": ["x", "y"]},
            "pipeline_variant": "tabular",
            "input_artefact_path": str(artefact),
        }

        result = asyncio.run(
            _call_tool(
                DS_TOOLS_URL,
                "ds",
                "verify_stage",
                run_id=run_id,
                stage_name="ingestion",
                session_state_json=json.dumps(session_state),
            )
        )
        parsed = json.loads(result)
        assert parsed["passed"] is True

    def test_log_metrics(self, run_id):
        result = asyncio.run(
            _call_tool(
                DS_TOOLS_URL,
                "ds",
                "log_metrics",
                run_id=run_id,
                stage_name="integration_test",
                metrics_json=json.dumps({"accuracy": 0.95, "f1_score": 0.92}),
            )
        )
        assert Path(result).exists()


class TestCrossServerCoordination:
    def test_cross_server_run(self, run_id):
        rid = f"{run_id}-e2e"

        asyncio.run(
            _call_tool(
                TRACKING_URL,
                "tracking",
                "record_start",
                run_id=rid,
                task_description="E2E integration test",
                pipeline_variant="tabular",
            )
        )
        metrics_path = asyncio.run(
            _call_tool(
                DS_TOOLS_URL,
                "ds",
                "log_metrics",
                run_id=rid,
                stage_name="e2e",
                metrics_json=json.dumps({"accuracy": 0.88}),
            )
        )
        assert Path(metrics_path).exists()

        bug_id = f"{rid}-observation"
        asyncio.run(
            _call_tool(
                BUG_LOG_URL,
                "buglog",
                "record_bug",
                bug_id=bug_id,
                run_id=rid,
                stage_name="e2e",
                error_type="ObservationNote",
                error_message="Synthetic E2E observation",
                traceback="",
                library_name="",
            )
        )

        end = asyncio.run(
            _call_tool(
                TRACKING_URL,
                "tracking",
                "record_end",
                run_id=rid,
                status="success",
                error="",
            )
        )
        assert f"Run {rid} ended" in end
