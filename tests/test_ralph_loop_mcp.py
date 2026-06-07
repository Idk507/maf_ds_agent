from __future__ import annotations


class _Response:
    text = "<DONE>ingestion</DONE>"


class _Agent:
    async def run(self, prompt, function_invocation_kwargs=None):
        return _Response()


class _DSToolsMCP:
    def __init__(self) -> None:
        self.calls = []

    async def call_tool(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        return '{"passed": true, "failures": []}'


class _TextContent:
    text = '{"passed": true, "failures": []}'


class _WrappedDSToolsMCP(_DSToolsMCP):
    async def call_tool(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        return {"content": [_TextContent()]}


class _TrackingMCP:
    def __init__(self) -> None:
        self.calls = []

    async def call_tool(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        return "ok"


async def test_ralph_loop_uses_mcp_verification_and_tracking(tmp_path):
    from workflows.ralph_loop import run_ralph_loop

    data_file = tmp_path / "data.csv"
    data_file.write_text("x,y\n1,2\n", encoding="utf-8")
    session_state = {
        "file_type_result": {"category": "tabular"},
        "schema": {"columns": ["x", "y"]},
        "pipeline_variant": "tabular",
        "input_artefact_path": str(data_file),
    }
    ds_tools = _DSToolsMCP()
    tracking = _TrackingMCP()

    updated_state, iterations, success = await run_ralph_loop(
        agent=_Agent(),
        stage_name="ingestion",
        prompt="run ingestion",
        session_state=session_state,
        run_id="test-run",
        tracking_mcp_client=tracking,
        ds_tools_mcp_client=ds_tools,
        max_iterations=1,
    )

    assert success is True
    assert iterations == 1
    assert updated_state["ralph_loop_iteration"] == 1
    assert ds_tools.calls[0][0] == "verify_stage"
    assert ds_tools.calls[0][1]["stage_name"] == "ingestion"
    assert tracking.calls[0][0] == "record_checkpoint"
    assert tracking.calls[0][1]["status"] == "passed"


async def test_ralph_loop_parses_wrapped_mcp_verification_response(tmp_path):
    from workflows.ralph_loop import run_ralph_loop

    data_file = tmp_path / "data.csv"
    data_file.write_text("x,y\n1,2\n", encoding="utf-8")
    session_state = {
        "file_type_result": {"category": "tabular"},
        "schema": {"columns": ["x", "y"]},
        "pipeline_variant": "tabular",
        "input_artefact_path": str(data_file),
    }
    ds_tools = _WrappedDSToolsMCP()

    _, _, success = await run_ralph_loop(
        agent=_Agent(),
        stage_name="ingestion",
        prompt="run ingestion",
        session_state=session_state,
        run_id="test-run",
        ds_tools_mcp_client=ds_tools,
        max_iterations=1,
    )

    assert success is True
    assert ds_tools.calls[0][0] == "verify_stage"
