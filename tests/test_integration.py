"""
tests/test_integration.py — Integration tests for MCP servers.

These tests require the three MCP servers to be running on their standard ports:
  - Tracking MCP:  http://localhost:8100
  - DS Tools MCP:  http://localhost:8101
  - Bug Log MCP:   http://localhost:8102

Run them after starting servers with:
  python scripts/start_servers.py &
  python -m pytest tests/test_integration.py -v

In CI the servers are started by the workflow before this suite runs.

Each test also has a pytest mark so they can be skipped when servers are offline:
  pytest -m "not integration"
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import httpx
import pytest
import pytest_asyncio  # noqa: F401  — registers asyncio backend

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TRACKING_URL = os.environ.get("TRACKING_MCP_URL", "http://localhost:8100")
DS_TOOLS_URL = os.environ.get("DS_TOOLS_MCP_URL", "http://localhost:8101")
BUG_LOG_URL = os.environ.get("BUG_LOG_MCP_URL", "http://localhost:8102")


def _servers_reachable() -> bool:
    """Return True only when all three MCP servers are healthy."""
    for url in [TRACKING_URL, DS_TOOLS_URL, BUG_LOG_URL]:
        try:
            r = httpx.get(f"{url}/health", timeout=3.0)
            if r.status_code != 200:
                return False
        except Exception:
            return False
    return True


# Skip the whole module when servers are not running (e.g. local dev without servers up).
pytestmark = pytest.mark.skipif(
    not _servers_reachable(),
    reason="MCP servers not reachable — start them first with: python scripts/start_servers.py",
)


@pytest.fixture(scope="module")
def run_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# 1. Health-check all three servers
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    def test_tracking_health(self):
        r = httpx.get(f"{TRACKING_URL}/health", timeout=5)
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "healthy"
        assert body.get("server") == "tracking"

    def test_ds_tools_health(self):
        r = httpx.get(f"{DS_TOOLS_URL}/health", timeout=5)
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "healthy"
        assert body.get("server") == "ds_tools"

    def test_bug_log_health(self):
        r = httpx.get(f"{BUG_LOG_URL}/health", timeout=5)
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "healthy"
        assert body.get("server") == "bug_log"


# ---------------------------------------------------------------------------
# 2. MCP protocol: tools/list on each server
# ---------------------------------------------------------------------------


class TestMCPToolsList:
    """Verify that each server advertises its tools via the MCP protocol."""

    def _list_tools(self, base_url: str) -> list[str]:
        """Send an MCP tools/list request and return tool names."""
        request_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }
        # MCP streamable-http endpoint is mounted at /mcp/mcp
        r = httpx.post(
            f"{base_url}/mcp/mcp",
            json=request_body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            timeout=10,
        )
        # Response may be SSE or JSON depending on transport
        if r.headers.get("content-type", "").startswith("text/event-stream"):
            # Parse SSE events
            tool_names = []
            for line in r.text.splitlines():
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                        if "result" in data and "tools" in data["result"]:
                            tool_names = [t["name"] for t in data["result"]["tools"]]
                    except json.JSONDecodeError:
                        pass
            return tool_names
        else:
            body = r.json()
            return [t["name"] for t in body.get("result", {}).get("tools", [])]

    def test_tracking_tools_list(self):
        names = self._list_tools(TRACKING_URL)
        assert len(names) >= 5, f"Expected ≥5 tracking tools, got {names}"
        # Verify at least core tracking tools are present
        for expected in ("record_start", "record_end", "record_checkpoint"):
            assert any(expected in n for n in names), f"Missing tool containing '{expected}' in {names}"

    def test_ds_tools_tools_list(self):
        names = self._list_tools(DS_TOOLS_URL)
        assert len(names) >= 5, f"Expected ≥5 ds_tools tools, got {names}"
        for expected in ("read_file", "execute_code", "web_research"):
            assert any(expected in n for n in names), f"Missing tool '{expected}' in {names}"

    def test_bug_log_tools_list(self):
        names = self._list_tools(BUG_LOG_URL)
        assert len(names) >= 4, f"Expected ≥4 bug_log tools, got {names}"
        for expected in ("record_bug", "list_bugs"):
            assert any(expected in n for n in names), f"Missing tool '{expected}' in {names}"


# ---------------------------------------------------------------------------
# 3. Tracking MCP: record_start → record_checkpoint → record_end round-trip
# ---------------------------------------------------------------------------


class TestTrackingRoundTrip:
    """Call the tracking server via httpx to exercise its SQLite layer."""

    def _call_tool(self, tool_name: str, args: dict) -> dict:
        """Generic MCP tool call using streamable-http transport."""
        body = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": args},
        }
        r = httpx.post(
            f"{TRACKING_URL}/mcp/mcp",
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            timeout=10,
        )
        if r.headers.get("content-type", "").startswith("text/event-stream"):
            for line in r.text.splitlines():
                if line.startswith("data:"):
                    try:
                        return json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
            return {}
        return r.json()

    def test_full_tracking_lifecycle(self, run_id):
        """start → checkpoint → end → query"""
        rid = f"{run_id}-tracking"

        # Start
        resp = self._call_tool("tracking_record_start", {
            "run_id": rid,
            "task": "Integration test tracking",
            "file_path": "/tmp/test.csv",
            "pipeline_variant": "tabular",
        })
        assert "error" not in resp, f"record_start failed: {resp}"

        # Checkpoint
        resp = self._call_tool("tracking_record_checkpoint", {
            "run_id": rid,
            "stage": "ingestion",
            "status": "completed",
            "details": '{"rows": 100}',
        })
        assert "error" not in resp, f"record_checkpoint failed: {resp}"

        # End
        resp = self._call_tool("tracking_record_end", {
            "run_id": rid,
            "status": "success",
            "summary": "Integration test completed",
        })
        assert "error" not in resp, f"record_end failed: {resp}"

        # Query the run
        resp = self._call_tool("tracking_query_run", {"run_id": rid})
        assert "error" not in resp, f"query_run failed: {resp}"


# ---------------------------------------------------------------------------
# 4. Bug Log MCP: record_bug → list_bugs round-trip
# ---------------------------------------------------------------------------


class TestBugLogRoundTrip:
    def _call_tool(self, tool_name: str, args: dict) -> dict:
        body = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": args},
        }
        r = httpx.post(
            f"{BUG_LOG_URL}/mcp/mcp",
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            timeout=10,
        )
        if r.headers.get("content-type", "").startswith("text/event-stream"):
            for line in r.text.splitlines():
                if line.startswith("data:"):
                    try:
                        return json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
            return {}
        return r.json()

    def test_bug_lifecycle(self, run_id):
        rid = f"{run_id}-buglog"

        # Record a bug
        resp = self._call_tool("buglog_record_bug", {
            "run_id": rid,
            "stage": "integration_test",
            "error_type": "IntegrationTestError",
            "error_message": "Test bug recorded by integration test",
            "traceback": "No traceback — synthetic",
            "severity": "low",
        })
        assert "error" not in resp, f"record_bug failed: {resp}"

        # List bugs for this run
        resp = self._call_tool("buglog_list_bugs", {"run_id": rid})
        assert "error" not in resp, f"list_bugs failed: {resp}"


# ---------------------------------------------------------------------------
# 5. DS Tools MCP: verify_stage (lightweight, no heavy computation)
# ---------------------------------------------------------------------------


class TestDSToolsVerify:
    def _call_tool(self, tool_name: str, args: dict) -> dict:
        body = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": args},
        }
        r = httpx.post(
            f"{DS_TOOLS_URL}/mcp/mcp",
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            timeout=10,
        )
        if r.headers.get("content-type", "").startswith("text/event-stream"):
            for line in r.text.splitlines():
                if line.startswith("data:"):
                    try:
                        return json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
            return {}
        return r.json()

    def test_verify_stage(self, run_id, tmp_path):
        """verify_stage with a dummy artefact — checks stage gate logic."""
        # Create a fake artefact so the verification can find it
        artefact = tmp_path / "test_model.pkl"
        artefact.write_bytes(b"dummy model content")

        resp = self._call_tool("ds_verify_stage", {
            "run_id": run_id,
            "stage": "integration_test",
            "criteria": json.dumps({
                "artefact_exists": True,
                "artefact_path": str(artefact),
            }),
        })
        # Just confirm no server error — criteria may pass or fail
        assert "error" not in resp, f"verify_stage returned error: {resp}"

    def test_log_metrics(self, run_id):
        """Log synthetic metrics — exercises metric recording path."""
        metrics = {
            "accuracy": 0.95,
            "f1_score": 0.92,
            "training_time_s": 12.5,
        }
        resp = self._call_tool("ds_log_metrics", {
            "run_id": run_id,
            "stage": "integration_test",
            "metrics": json.dumps(metrics),
        })
        assert "error" not in resp, f"log_metrics failed: {resp}"


# ---------------------------------------------------------------------------
# 6. End-to-end: all three servers cooperate on one run_id
# ---------------------------------------------------------------------------


class TestCrossServerCoordination:
    """
    Simulate a minimal pipeline execution:
      Tracking records the run → DS Tools logs metrics → Bug Log notes a synthetic bug.
    Verifies that all three servers share a consistent run_id lifecycle.
    """

    def test_cross_server_run(self, run_id, tmp_path):
        e2e_run = f"{run_id}-e2e"

        # 1. Tracking: start
        r = httpx.post(
            f"{TRACKING_URL}/mcp/mcp",
            json={
                "jsonrpc": "2.0", "id": 10, "method": "tools/call",
                "params": {"name": "tracking_record_start", "arguments": {
                    "run_id": e2e_run, "task": "E2E integration test",
                    "file_path": str(tmp_path / "data.csv"),
                    "pipeline_variant": "tabular",
                }},
            },
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            timeout=10,
        )
        assert r.status_code == 200, f"Tracking start failed: {r.status_code}"

        # 2. DS Tools: log metrics
        metrics_resp = httpx.post(
            f"{DS_TOOLS_URL}/mcp/mcp",
            json={
                "jsonrpc": "2.0", "id": 11, "method": "tools/call",
                "params": {"name": "ds_log_metrics", "arguments": {
                    "run_id": e2e_run, "stage": "e2e",
                    "metrics": json.dumps({"accuracy": 0.88}),
                }},
            },
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            timeout=10,
        )
        assert metrics_resp.status_code == 200, f"DS Tools metrics failed: {metrics_resp.status_code}"

        # 3. Bug Log: record synthetic observation
        bug_resp = httpx.post(
            f"{BUG_LOG_URL}/mcp/mcp",
            json={
                "jsonrpc": "2.0", "id": 12, "method": "tools/call",
                "params": {"name": "buglog_record_bug", "arguments": {
                    "run_id": e2e_run, "stage": "e2e",
                    "error_type": "ObservationNote", "severity": "low",
                    "error_message": "Synthetic E2E observation — not a real bug",
                    "traceback": "",
                }},
            },
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            timeout=10,
        )
        assert bug_resp.status_code == 200, f"Bug Log failed: {bug_resp.status_code}"

        # 4. Tracking: end run
        end_resp = httpx.post(
            f"{TRACKING_URL}/mcp/mcp",
            json={
                "jsonrpc": "2.0", "id": 13, "method": "tools/call",
                "params": {"name": "tracking_record_end", "arguments": {
                    "run_id": e2e_run, "status": "success",
                    "summary": "E2E integration test passed",
                }},
            },
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            timeout=10,
        )
        assert end_resp.status_code == 200, f"Tracking end failed: {end_resp.status_code}"
