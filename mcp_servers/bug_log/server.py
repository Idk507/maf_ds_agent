"""
mcp_servers/bug_log/server.py — Bug Log MCP Server (port 8102).

Stateful (stateless_http=False). SQLite WAL.
8 tools: record_bug, record_repair_attempt, mark_bug_resolved,
         record_ralph_loop_iteration, record_tool_call,
         query_bug_history, query_bug_patterns, query_unresolved_bugs.

MCP URL: http://localhost:8102/mcp/mcp
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI
from fastmcp import FastMCP
from pydantic import Field

load_dotenv()

DB_PATH = os.environ.get("BUGLOG_DB_PATH", "data/buglog.db")
PORT = int(os.environ.get("BUG_LOG_MCP_PORT", "8102"))

# Rate-limit tracking for DuckDuckGo (in-memory, resets on restart)
_ddg_call_log: list[float] = []
DDG_HOURLY_LIMIT = 30

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
        CREATE TABLE IF NOT EXISTS bugs (
            bug_id          TEXT PRIMARY KEY,
            run_id          TEXT,
            stage_name      TEXT,
            error_type      TEXT,
            error_message   TEXT,
            traceback       TEXT,
            library_name    TEXT,
            status          TEXT DEFAULT 'open',
            created_at      TEXT,
            resolved_at     TEXT,
            resolution_note TEXT
        );

        CREATE TABLE IF NOT EXISTS repair_attempts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            bug_id          TEXT REFERENCES bugs(bug_id),
            attempt_number  INTEGER,
            strategy        TEXT,
            patched_code    TEXT,
            success         INTEGER DEFAULT 0,
            error_after     TEXT,
            created_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS ralph_loop_iterations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT,
            stage_name      TEXT,
            iteration       INTEGER,
            verification_failures TEXT,
            feedback_injected TEXT,
            created_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS tool_calls (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT,
            tool_name       TEXT,
            args_json       TEXT,
            result_summary  TEXT,
            created_at      TEXT
        );
    """)
    conn.commit()
    conn.close()


_init_db()

# ── FastMCP Server ──────────────────────────────────────────────────

mcp = FastMCP("BugLogMCP")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@mcp.tool()
async def record_bug(
    bug_id: Annotated[str, Field(description="Unique identifier for this bug (uuid or stage+timestamp).")],
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    stage_name: Annotated[str, Field(description="Stage where the error occurred.")],
    error_type: Annotated[str, Field(description="Python exception class name.")],
    error_message: Annotated[str, Field(description="Exception message (first 500 chars).")],
    traceback: Annotated[str, Field(description="Full traceback as a string.")],
    library_name: Annotated[str, Field(description="Inferred library name from traceback, or empty string.")] = "",
) -> str:
    """Record a new bug encountered during pipeline execution."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO bugs
           (bug_id,run_id,stage_name,error_type,error_message,traceback,library_name,status,created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (bug_id, run_id, stage_name, error_type, error_message[:500], traceback, library_name, "open", _now()),
    )
    conn.commit()
    conn.close()
    return f"Bug {bug_id} recorded."


@mcp.tool()
async def record_repair_attempt(
    bug_id: Annotated[str, Field(description="Bug identifier being repaired.")],
    attempt_number: Annotated[int, Field(description="Repair attempt number (1-5).")],
    strategy: Annotated[str, Field(description="Strategy used: pattern_match, browser_use, duckduckgo, first_principles, escalate.")],
    patched_code: Annotated[str, Field(description="The patched code snippet, or empty string if no patch.")],
    success: Annotated[bool, Field(description="True if the patch resolved the error.")],
    error_after: Annotated[str, Field(description="New error after patch, or empty string if success.")] = "",
    run_id: Annotated[str, Field(description="Pipeline run identifier (ignored by this tool).")] = "",
    stage_name: Annotated[str, Field(description="Pipeline stage (ignored by this tool).")] = "",
) -> str:
    """Record a Debug Agent repair attempt for a bug."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO repair_attempts
           (bug_id, attempt_number, strategy, patched_code, success, error_after, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (bug_id, attempt_number, strategy, patched_code, int(success), error_after, _now()),
    )
    conn.commit()
    conn.close()
    return f"Repair attempt {attempt_number} recorded for bug {bug_id}."


@mcp.tool()
async def mark_bug_resolved(
    bug_id: Annotated[str, Field(description="Bug identifier.")],
    resolution_note: Annotated[str, Field(description="Short description of how the bug was resolved.")],
    run_id: Annotated[str, Field(description="Pipeline run identifier (ignored by this tool).")] = "",
    stage_name: Annotated[str, Field(description="Pipeline stage (ignored by this tool).")] = "",
) -> str:
    """Mark a bug as resolved."""
    conn = _get_conn()
    conn.execute(
        "UPDATE bugs SET status='resolved', resolved_at=?, resolution_note=? WHERE bug_id=?",
        (_now(), resolution_note, bug_id),
    )
    conn.commit()
    conn.close()
    return f"Bug {bug_id} marked resolved."


@mcp.tool()
async def record_ralph_loop_iteration(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    stage_name: Annotated[str, Field(description="Stage being retried.")],
    iteration: Annotated[int, Field(description="Iteration number (1-indexed).")],
    verification_failures: Annotated[str, Field(description="JSON-encoded list of failure dicts.")],
    feedback_injected: Annotated[str, Field(description="The feedback string injected into the next attempt.")],
) -> str:
    """Record a Ralph Loop iteration for audit and analysis."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO ralph_loop_iterations
           (run_id, stage_name, iteration, verification_failures, feedback_injected, created_at)
           VALUES (?,?,?,?,?,?)""",
        (run_id, stage_name, iteration, verification_failures, feedback_injected, _now()),
    )
    conn.commit()
    conn.close()
    return f"Ralph Loop iteration {iteration} for {stage_name} recorded."


@mcp.tool()
async def record_tool_call(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    tool_name: Annotated[str, Field(description="MCP tool name called.")],
    args_summary: Annotated[str, Field(description="JSON-encoded summary of arguments (no secrets).")],
    result_summary: Annotated[str, Field(description="One-line summary of the result.")] = "",
    stage_name: Annotated[str, Field(description="Pipeline stage (ignored by this tool).")] = "",
) -> str:
    """Record a tool call for rate-limit tracking and audit."""
    import time

    if tool_name == "web_research":
        now_ts = time.time()
        # Trim calls older than 1 hour
        cutoff = now_ts - 3600
        _ddg_call_log[:] = [t for t in _ddg_call_log if t > cutoff]
        if len(_ddg_call_log) >= DDG_HOURLY_LIMIT:
            return "rate_limit_exceeded"
        _ddg_call_log.append(now_ts)

    conn = _get_conn()
    conn.execute(
        "INSERT INTO tool_calls (run_id, tool_name, args_json, result_summary, created_at) VALUES (?,?,?,?,?)",
        (run_id, tool_name, args_summary, result_summary, _now()),
    )
    conn.commit()
    conn.close()
    return "tool_call_recorded"


@mcp.tool()
async def query_bug_history(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    stage_name: Annotated[str, Field(description="Filter by stage name, or empty string for all.")] = "",
) -> str:
    """Return full bug history for a run, with all repair attempts."""
    conn = _get_conn()
    if stage_name:
        bugs = conn.execute(
            "SELECT * FROM bugs WHERE run_id=? AND stage_name=?", (run_id, stage_name)
        ).fetchall()
    else:
        bugs = conn.execute("SELECT * FROM bugs WHERE run_id=?", (run_id,)).fetchall()

    result = []
    for bug in bugs:
        bug_dict = dict(bug)
        attempts = conn.execute(
            "SELECT * FROM repair_attempts WHERE bug_id=? ORDER BY attempt_number",
            (bug_dict["bug_id"],),
        ).fetchall()
        bug_dict["repair_attempts"] = [dict(a) for a in attempts]
        result.append(bug_dict)
    conn.close()
    return json.dumps(result, indent=2)


@mcp.tool()
async def query_bug_patterns(
    error_type: Annotated[str, Field(description="Exception class to look up patterns for.")],
    limit: Annotated[int, Field(description="Maximum number of past bugs to return.")] = 10,
    run_id: Annotated[str, Field(description="Pipeline run identifier (ignored by this tool).")] = "",
    stage_name: Annotated[str, Field(description="Pipeline stage (ignored by this tool).")] = "",
) -> str:
    """Return past bugs of the same error type with their resolutions (for pattern matching)."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT bug_id, stage_name, error_message, resolution_note
           FROM bugs WHERE error_type=? AND status='resolved'
           ORDER BY resolved_at DESC LIMIT ?""",
        (error_type, limit),
    ).fetchall()
    conn.close()
    return json.dumps([dict(r) for r in rows], indent=2)


@mcp.tool()
async def query_unresolved_bugs(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    stage_name: Annotated[str, Field(description="Pipeline stage (ignored by this tool).")] = "",
) -> str:
    """Return all unresolved bugs for a run."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM bugs WHERE run_id=? AND status='open' ORDER BY created_at DESC",
        (run_id,),
    ).fetchall()
    conn.close()
    return json.dumps([dict(r) for r in rows], indent=2)


# ── FastAPI Application ─────────────────────────────────────────────
# mcp_app must be created before FastAPI so its lifespan can be used.

mcp_app = mcp.http_app(transport="streamable-http", stateless_http=False)
app = FastAPI(title="Bug Log MCP Server", lifespan=mcp_app.lifespan)
app.mount("/mcp", mcp_app)


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "server": "bug_log", "port": PORT}


@app.head("/health")
async def health_head():
    from fastapi.responses import Response
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
