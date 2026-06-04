"""
config/settings.py — Centralised application settings.

All environment variable reads happen here. Never read os.environ elsewhere.
"""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Azure OpenAI ──────────────────────────────────────────────────
    ai_foundry_project_endpoint: str = ""
    ai_foundry_api_version: str = "2024-12-01-preview"
    ai_foundry_api_key: str | None = None

    azure_openai_primary_deployment: str = "gpt-4o"
    azure_openai_fast_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"

    # ── MCP Ports ─────────────────────────────────────────────────────
    tracking_mcp_port: int = 8100
    ds_tools_mcp_port: int = 8101
    bug_log_mcp_port: int = 8102

    tracking_mcp_url: str = "http://localhost:8100/mcp/mcp"
    ds_tools_mcp_url: str = "http://localhost:8101/mcp/mcp"
    bug_log_mcp_url: str = "http://localhost:8102/mcp/mcp"

    # ── Pipeline Limits ───────────────────────────────────────────────
    pipeline_token_budget: int = 500_000
    pipeline_max_ralph_iterations: int = 8
    pipeline_human_in_the_loop: bool = True

    # ── Data Paths ────────────────────────────────────────────────────
    memory_db_path: str = "data/memory.db"
    semantic_memory_path: str = "data/semantic_memory.json"
    procedural_memory_path: str = "data/procedural_memory.json"
    tracking_db_path: str = "data/tracking.db"
    buglog_db_path: str = "data/buglog.db"
    output_base_dir: str = "outputs"

    # ── OpenTelemetry (optional) ──────────────────────────────────────
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "ds-agent"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return Settings()
