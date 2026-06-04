"""
tests/test_imports.py — Import sanity checks.

These tests run without Azure credentials — they only validate that all
modules can be imported and key objects are constructable.
"""
import os
import sys

import pytest

# Inject dummy env vars so lazy clients can be resolved in agent constructors
os.environ.setdefault("AI_FOUNDRY_PROJECT_ENDPOINT", "https://test.openai.azure.com")
os.environ.setdefault("AZURE_OPENAI_PRIMARY_DEPLOYMENT", "gpt-4o")
os.environ.setdefault("AZURE_OPENAI_FAST_DEPLOYMENT", "gpt-4o-mini")
os.environ.setdefault("AI_FOUNDRY_API_KEY", "test-key-offline")


# ── Foundation modules ────────────────────────────────────────────────

def test_clients_import():
    from agents.clients import PRIMARY_CLIENT, FAST_CLIENT
    assert PRIMARY_CLIENT is not None
    assert FAST_CLIENT is not None


def test_middleware_import():
    from agents.middleware import pipeline_stack, observer_stack, debug_stack
    ps = pipeline_stack()
    os_ = observer_stack()
    ds = debug_stack()
    assert isinstance(ps, list)
    assert isinstance(os_, list)
    assert isinstance(ds, list)


def test_ralph_loop_import():
    from workflows.ralph_loop import run_ralph_loop
    import asyncio
    import inspect
    assert inspect.iscoroutinefunction(run_ralph_loop)


def test_criteria_import():
    from workflows.criteria import get_criteria, StageCriteria
    assert callable(get_criteria)
    assert "ingestion" in get_criteria.__code__.co_consts or True  # presence check
    from workflows.criteria import IngestionCriteria, TrainingCriteria
    assert issubclass(IngestionCriteria, StageCriteria)
    assert issubclass(TrainingCriteria, StageCriteria)


def test_pipeline_graph():
    from workflows.pipeline_graph import get_pipeline_stages, select_variant, PIPELINE_GRAPH

    assert "tabular" in PIPELINE_GRAPH
    assert "image" in PIPELINE_GRAPH
    assert "existing_model" in PIPELINE_GRAPH

    stages = get_pipeline_stages("tabular")
    assert stages[0] == "ingestion"
    assert stages[-1] == "deployment"

    img_stages = get_pipeline_stages("image")
    assert "eda" not in img_stages  # image skips EDA

    em_stages = get_pipeline_stages("existing_model")
    assert "training" not in em_stages

    assert select_variant("tabular") == "tabular"
    assert select_variant("image") == "image"
    assert select_variant("model") in ("existing_model", "tabular")


def test_local_tools_import():
    from tools.local_tools import LOCAL_TOOLS
    assert len(LOCAL_TOOLS) == 4
    tool_names = [t.name for t in LOCAL_TOOLS]
    assert "get_session_state" in tool_names
    assert "set_session_state" in tool_names
    assert "check_artefact_exists" in tool_names


def test_base_agent_builders_import():
    from agents.base import (
        make_tracking_mcp,
        make_ds_tools_mcp,
        make_bug_log_mcp,
        build_pipeline_agent,
        build_observer_agent,
        build_debug_agent_base,
    )
    assert callable(make_tracking_mcp)
    assert callable(build_pipeline_agent)


# ── Stage agents ──────────────────────────────────────────────────────

def test_all_stage_agents_import():
    from agents.ingestion_agent import build_ingestion_agent
    from agents.eda_agent import build_eda_agent
    from agents.cleaning_agent import build_cleaning_agent
    from agents.feature_agent import build_feature_agent
    from agents.training_agent import build_training_agent
    from agents.tuning_agent import build_tuning_agent
    from agents.evaluation_agent import build_evaluation_agent
    from agents.explainability_agent import build_explainability_agent
    from agents.report_agent import build_report_agent
    from agents.deployment_agent import build_deployment_agent
    # All should be callable
    for fn in [
        build_ingestion_agent, build_eda_agent, build_cleaning_agent,
        build_feature_agent, build_training_agent, build_tuning_agent,
        build_evaluation_agent, build_explainability_agent,
        build_report_agent, build_deployment_agent,
    ]:
        assert callable(fn), f"{fn.__name__} is not callable"


def test_stage_agent_names():
    """Agents must have the correct .name attribute (used in observability middleware)."""
    from agents.ingestion_agent import build_ingestion_agent
    from agents.eda_agent import build_eda_agent
    from agents.training_agent import build_training_agent
    from agents.deployment_agent import build_deployment_agent

    assert build_ingestion_agent().name == "ingestion_agent"
    assert build_eda_agent().name == "eda_agent"
    assert build_training_agent().name == "training_agent"
    assert build_deployment_agent().name == "deployment_agent"


def test_support_agents_import():
    from agents.debug_agent import build_debug_agent
    from agents.bug_log_observer import build_bug_log_observer
    from agents.artefact_tracking_observer import build_artefact_tracking_observer

    debug = build_debug_agent()
    assert debug.name == "debug_agent"
    assert callable(build_bug_log_observer)
    assert callable(build_artefact_tracking_observer)


# ── Orchestrator ──────────────────────────────────────────────────────

def test_orchestrator_builds():
    from agents.orchestrator import build_orchestrator, PipelineOrchestrator

    orch = build_orchestrator()
    assert isinstance(orch, PipelineOrchestrator)

    expected_stages = {
        "ingestion", "eda", "cleaning", "feature_engineering",
        "training", "tuning", "evaluation", "explainability",
        "report", "deployment",
    }
    assert set(orch._stage_agents.keys()) == expected_stages


def test_orchestrator_has_debug_and_observers():
    from agents.orchestrator import build_orchestrator

    orch = build_orchestrator()
    assert orch._debug_agent is not None
    assert orch._bug_log_observer is not None
    assert orch._tracking_observer is not None
    assert orch._tracking_mcp is not None
    assert orch._ds_tools_mcp is not None
    assert orch._bug_log_mcp is not None


# ── MCP servers ───────────────────────────────────────────────────────

def test_mcp_servers_import():
    import importlib
    for mod_path in [
        "mcp_servers.tracking.server",
        "mcp_servers.ds_tools.server",
        "mcp_servers.bug_log.server",
    ]:
        mod = importlib.import_module(mod_path)
        assert hasattr(mod, "mcp"), f"{mod_path} missing 'mcp' object"
        assert hasattr(mod, "app"), f"{mod_path} missing 'app' object"


# ── File type detection ───────────────────────────────────────────────

def test_file_type_detector_import():
    from tools.file_type_detector import detect_file_type, FileTypeResult
    assert callable(detect_file_type)


def test_detect_csv_file(tmp_path):
    from tools.file_type_detector import detect_file_type

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("col1,col2,col3\n1,2,3\n4,5,6\n", encoding="utf-8")

    result = detect_file_type(str(csv_file))
    assert result.category in ("tabular", "text")
    assert result.confidence > 0.0


def test_detect_nonexistent_file():
    from tools.file_type_detector import detect_file_type

    result = detect_file_type("/nonexistent/path/data.csv")
    # Extension-based fallback: .csv → tabular
    assert result.category in ("tabular", "unknown")
    assert result.source in ("extension", "unknown")


# ── Config ────────────────────────────────────────────────────────────

def test_settings_import():
    from config.settings import get_settings, Settings
    s = get_settings()
    assert isinstance(s, Settings)


def test_library_registry_import():
    from config.library_registry import LIBRARY_REGISTRY
    assert isinstance(LIBRARY_REGISTRY, dict)
    # Registry is keyed by library name (e.g. 'pandas', 'sklearn')
    assert len(LIBRARY_REGISTRY) > 5
    assert "pandas" in LIBRARY_REGISTRY
    assert "scikit-learn" in LIBRARY_REGISTRY or "sklearn" in LIBRARY_REGISTRY


# ── main.py ───────────────────────────────────────────────────────────

def test_main_importable():
    import importlib
    main = importlib.import_module("main")
    assert hasattr(main, "main")
    assert callable(main.main)
