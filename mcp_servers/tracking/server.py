"""
mcp_servers/tracking/server.py — Tracking MCP Server (port 8100).

Stateful (stateless_http=False). Maintains SQLite WAL for run audit trail.
11 tools: record_start, record_end, record_checkpoint, record_artefact,
          record_metric, record_lineage, query_run, query_artefacts,
          query_metrics, query_lineage, list_runs.

MCP URL: http://localhost:8100/mcp/mcp
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI
from fastmcp import FastMCP
from pydantic import Field

load_dotenv()

DB_PATH = os.environ.get("TRACKING_DB_PATH", "data/tracking.db")
PORT = int(os.environ.get("TRACKING_MCP_PORT", "8100"))

# ── Database helpers ────────────────────────────────────────────────


def _get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id       TEXT PRIMARY KEY,
            task         TEXT,
            variant      TEXT,
            status       TEXT DEFAULT 'running',
            started_at   TEXT,
            ended_at     TEXT,
            error        TEXT
        );

        CREATE TABLE IF NOT EXISTS checkpoints (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id       TEXT REFERENCES pipeline_runs(run_id),
            stage_name   TEXT,
            iteration    INTEGER,
            status       TEXT,
            notes        TEXT,
            created_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS artefacts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id       TEXT REFERENCES pipeline_runs(run_id),
            stage_name   TEXT,
            name         TEXT,
            path         TEXT,
            size_bytes   INTEGER,
            created_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS metrics (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id       TEXT REFERENCES pipeline_runs(run_id),
            stage_name   TEXT,
            metric_name  TEXT,
            metric_value REAL,
            created_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS lineage (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id       TEXT REFERENCES pipeline_runs(run_id),
            from_stage   TEXT,
            to_stage     TEXT,
            artefact_name TEXT,
            created_at   TEXT
        );
    """)
    conn.commit()
    conn.close()


_init_db()

# ── FastMCP Server ──────────────────────────────────────────────────

mcp = FastMCP("TrackingMCP")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@mcp.tool()
async def record_start(
    run_id: Annotated[str, Field(description="Unique pipeline run identifier.")],
    task_description: Annotated[str, Field(description="Human-readable description of the task.")],
    pipeline_variant: Annotated[str, Field(description="Pipeline variant: tabular, document_text, image, existing_model.")],
) -> str:
    """Record the start of a pipeline run."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO pipeline_runs (run_id, task, variant, status, started_at) VALUES (?,?,?,?,?)",
        (run_id, task_description, pipeline_variant, "running", _now()),
    )
    conn.commit()
    conn.close()
    return f"Run {run_id} started."


@mcp.tool()
async def record_end(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    status: Annotated[str, Field(description="Final status: success or failed.")],
    error: Annotated[str, Field(description="Error message if status is failed, otherwise empty string.")] = "",
) -> str:
    """Record the end of a pipeline run."""
    conn = _get_conn()
    conn.execute(
        "UPDATE pipeline_runs SET status=?, ended_at=?, error=? WHERE run_id=?",
        (status, _now(), error or None, run_id),
    )
    conn.commit()
    conn.close()
    return f"Run {run_id} ended with status={status}."


@mcp.tool()
async def record_checkpoint(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    stage_name: Annotated[str, Field(description="Name of the pipeline stage.")],
    iteration: Annotated[int, Field(description="Ralph Loop iteration number (0 = first attempt).")],
    status: Annotated[str, Field(description="Checkpoint status: passed or failed.")],
    notes: Annotated[str, Field(description="Optional notes or verification details.")] = "",
) -> str:
    """Record a stage verification checkpoint."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO checkpoints (run_id, stage_name, iteration, status, notes, created_at) VALUES (?,?,?,?,?,?)",
        (run_id, stage_name, iteration, status, notes, _now()),
    )
    conn.commit()
    conn.close()
    return f"Checkpoint recorded: run={run_id} stage={stage_name} iter={iteration} status={status}."


@mcp.tool()
async def record_artefact(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    stage_name: Annotated[str, Field(description="Stage that produced the artefact.")],
    name: Annotated[str, Field(description="Logical artefact name, e.g. cleaned_dataset.parquet.")],
    path: Annotated[str, Field(description="File system path to the artefact.")],
    size_bytes: Annotated[int, Field(description="File size in bytes.")] = 0,
) -> str:
    """Record a produced artefact."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO artefacts (run_id, stage_name, name, path, size_bytes, created_at) VALUES (?,?,?,?,?,?)",
        (run_id, stage_name, name, path, size_bytes, _now()),
    )
    conn.commit()
    conn.close()
    return f"Artefact '{name}' recorded for run {run_id}."


@mcp.tool()
async def record_metric(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    stage_name: Annotated[str, Field(description="Stage that produced the metric.")],
    metric_name: Annotated[str, Field(description="Metric name, e.g. roc_auc.")],
    metric_value: Annotated[float, Field(description="Numeric metric value.")],
) -> str:
    """Record a performance metric for a stage."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO metrics (run_id, stage_name, metric_name, metric_value, created_at) VALUES (?,?,?,?,?)",
        (run_id, stage_name, metric_name, metric_value, _now()),
    )
    conn.commit()
    conn.close()
    return f"Metric {metric_name}={metric_value} recorded for run {run_id}."


@mcp.tool()
async def record_lineage(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    from_stage: Annotated[str, Field(description="Source stage name.")],
    to_stage: Annotated[str, Field(description="Consuming stage name.")],
    artefact_name: Annotated[str, Field(description="Artefact passed between stages.")],
) -> str:
    """Record data lineage between two stages."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO lineage (run_id, from_stage, to_stage, artefact_name, created_at) VALUES (?,?,?,?,?)",
        (run_id, from_stage, to_stage, artefact_name, _now()),
    )
    conn.commit()
    conn.close()
    return f"Lineage recorded: {from_stage} → {to_stage} via '{artefact_name}'."


@mcp.tool()
async def query_run(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
) -> str:
    """Return full details of a specific run."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM pipeline_runs WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    if row is None:
        return f"Run {run_id} not found."
    return dict(row).__repr__()


@mcp.tool()
async def query_artefacts(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
) -> str:
    """Return all artefacts for a run."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM artefacts WHERE run_id=?", (run_id,)).fetchall()
    conn.close()
    return str([dict(r) for r in rows])


@mcp.tool()
async def query_metrics(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    stage_name: Annotated[str, Field(description="Filter by stage name, or empty string for all stages.")] = "",
) -> str:
    """Return metrics for a run, optionally filtered by stage."""
    conn = _get_conn()
    if stage_name:
        rows = conn.execute(
            "SELECT * FROM metrics WHERE run_id=? AND stage_name=?", (run_id, stage_name)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM metrics WHERE run_id=?", (run_id,)).fetchall()
    conn.close()
    return str([dict(r) for r in rows])


@mcp.tool()
async def query_lineage(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
) -> str:
    """Return full lineage graph for a run."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM lineage WHERE run_id=?", (run_id,)).fetchall()
    conn.close()
    return str([dict(r) for r in rows])


@mcp.tool()
async def list_runs(
    limit: Annotated[int, Field(description="Maximum number of runs to return.")] = 20,
) -> str:
    """List the most recent pipeline runs."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT run_id, status, started_at, ended_at, variant FROM pipeline_runs ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return str([dict(r) for r in rows])


# ── FastAPI Application ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp._lifespan_manager():
        yield


app = FastAPI(title="Tracking MCP Server", lifespan=lifespan)
mcp_app = mcp.http_app(transport="streamable-http", stateless_http=False)
app.mount("/mcp", mcp_app)


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "server": "tracking", "port": PORT}


@app.head("/health")
async def health_head():
    from fastapi.responses import Response
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
