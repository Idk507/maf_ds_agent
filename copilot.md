# copilot.md — DS Agent Implementation Guide

> **Source of Truth:** This file is the Copilot / agent implementation guide derived from the full specification in `README.md`.
> **Framework:** Microsoft Agent Framework 1.0 (Python) — `pip install agent-framework`
> **Credentials:** All Azure OpenAI values live in `.env`. Never hardcode them.
> **Rule:** If anything here conflicts with `README.md`, fix this file — `README.md` governs.

---

## Table of Contents

1. [Environment and Credentials](#1-environment-and-credentials)
2. [Planning Phase — What to Build and in What Order](#2-planning-phase--what-to-build-and-in-what-order)
3. [Microsoft Agent Framework — Key Concepts](#3-microsoft-agent-framework--key-concepts)
4. [Directory Structure](#4-directory-structure)
5. [Phase 1 — Environment Bootstrap and MCP Servers](#5-phase-1--environment-bootstrap-and-mcp-servers)
6. [Phase 2 — Core Agents and Read File Pipeline](#6-phase-2--core-agents-and-read-file-pipeline)
7. [Phase 3 — EDA, Cleaning, and Feature Engineering](#7-phase-3--eda-cleaning-and-feature-engineering)
8. [Phase 4 — Training, Tuning, and Evaluation](#8-phase-4--training-tuning-and-evaluation)
9. [Phase 5 — Explainability, Reporting, and Deployment](#9-phase-5--explainability-reporting-and-deployment)
10. [Phase 6 — Memory System](#10-phase-6--memory-system)
11. [Phase 7 — Hardening and Production Gate](#11-phase-7--hardening-and-production-gate)
12. [Agent Construction Patterns](#12-agent-construction-patterns)
13. [Ralph Loop Implementation Checklist](#13-ralph-loop-implementation-checklist)
14. [MCP Server Verification Checklist](#14-mcp-server-verification-checklist)
15. [Inter-Agent Contract Quick Reference](#15-inter-agent-contract-quick-reference)
16. [Known Pitfalls and Remediation](#16-known-pitfalls-and-remediation)
17. [Testing Strategy](#17-testing-strategy)
18. [CI/CD Reference](#18-cicd-reference)

---

## 1. Environment and Credentials

### 1.1 Current `.env` Values

```dotenv
# Azure AI Foundry / Azure OpenAI
AI_FOUNDRY_PROJECT_ENDPOINT="https://idkopenai.services.ai.azure.com"
AI_FOUNDRY_DEPLOYMENT_NAME="gpt-4o"
embedding_model="text-embedding-ada-002"
AI_FOUNDRY_API_VERSION="2024-12-01-preview"
AI_FOUNDRY_API_KEY=<loaded from .env — never commit>
```

> **Security:** The API key must never appear in source code, commit history, or logs.
> Rotate immediately if it has ever been committed. Use `git filter-repo` to purge history.

### 1.2 Additional `.env` Keys to Add

Append the following to `.env` (values to match your Azure portal settings):

```dotenv
# MCP Server ports
TRACKING_MCP_PORT=8100
DS_TOOLS_MCP_PORT=8101
BUG_LOG_MCP_PORT=8102

# Full MCP endpoint URLs (FastAPI mount /mcp + FastMCP suffix /mcp)
TRACKING_MCP_URL=http://localhost:8100/mcp/mcp
DS_TOOLS_MCP_URL=http://localhost:8101/mcp/mcp
BUG_LOG_MCP_URL=http://localhost:8102/mcp/mcp

# Deployment names (match Azure OpenAI portal)
AZURE_OPENAI_PRIMARY_DEPLOYMENT=gpt-4o
AZURE_OPENAI_FAST_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Pipeline limits
PIPELINE_TOKEN_BUDGET=500000
PIPELINE_MAX_RALPH_ITERATIONS=8
PIPELINE_HUMAN_IN_THE_LOOP=true

# Memory / data paths
MEMORY_DB_PATH=data/memory.db
SEMANTIC_MEMORY_PATH=data/semantic_memory.json
PROCEDURAL_MEMORY_PATH=data/procedural_memory.json
```

### 1.3 Loading `.env` in Code

MAF does **not** auto-load `.env`. Call `load_dotenv()` at the very top of `main.py` before any import that reads env vars.

```python
# main.py — first two lines
from dotenv import load_dotenv
load_dotenv()
```

### 1.4 Azure OpenAI Client — Two Instances Only

```python
# agents/clients.py
import os
from agent_framework.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential

def _make_client(deployment_env_key: str) -> AzureOpenAIChatCompletionClient:
    """Create one AzureOpenAI client. Called twice at startup only."""
    return AzureOpenAIChatCompletionClient(
        azure_endpoint=os.environ["AI_FOUNDRY_PROJECT_ENDPOINT"],
        azure_deployment=os.environ[deployment_env_key],
        api_version=os.environ["AI_FOUNDRY_API_VERSION"],
        api_key=os.environ.get("AI_FOUNDRY_API_KEY"),  # None in prod → DefaultAzureCredential
        credential=DefaultAzureCredential() if not os.environ.get("AI_FOUNDRY_API_KEY") else None,
    )

PRIMARY_CLIENT = _make_client("AZURE_OPENAI_PRIMARY_DEPLOYMENT")
FAST_CLIENT    = _make_client("AZURE_OPENAI_FAST_DEPLOYMENT")
```

---

## 2. Planning Phase — What to Build and in What Order

### 2.1 The Seven Phases

The project is implemented across seven sequential phases. Each phase has a clear exit gate — do not proceed to the next phase until the gate passes.

| Phase | Name | Key Deliverables | Exit Gate |
|---|---|---|---|
| 1 | Environment + MCP Servers | `requirements.txt`, three FastMCP servers running, health checks green | All three `curl /health` return `{"status":"healthy"}` |
| 2 | read\_file + Ingestion + Observers | `read_file` tool (three-layer detection), Ingestion Agent, Artefact Tracking Agent, Bug Log Agent | Unit test on 10 file types; artefact record in Tracking DB; bug record in Bug Log DB |
| 3 | EDA + Cleaning + Feature Eng | EDA Agent, Cleaning Agent, Feature Eng Agent, all with Ralph Loop | Full mini-pipeline on `iris.csv` passes all stage verifications |
| 4 | Training + Tuning + Evaluation | Training Agent, Tuning Agent, Evaluation Agent | Tuned metric ≥ baseline metric on iris; fairness report present |
| 5 | Explainability + Report + Deployment | Explainability Agent, Report Agent, Deployment Agent | ≥2 SHAP plots; HTML report self-contained; 10/10 smoke test |
| 6 | Memory System | Memory Agent, three-tier memory, episodic/semantic/procedural | Second run on iris uses memory; retrieval test passes |
| 7 | Hardening + Production | Safety patterns, token budget, full integration test, CI | Full pipeline on 3 datasets passes all gates |

### 2.2 Decision Points Before Writing Any Code

**Before Phase 1:** Confirm Azure OpenAI quota for `gpt-4o` and `text-embedding-ada-002` in the portal. Confirm API key in `.env` is active with `curl`.

**Before Phase 3:** Confirm Ralph Loop exit criteria are documented in `workflows/criteria.py` for all three stages.

**Before Phase 5:** Confirm FastAPI deployment endpoint (local `uvicorn`, or Azure Container Apps) is decided and documented.

**Before Phase 6:** Confirm `data/memory.db` schema migrations strategy (use `sqlite3` directly, no ORM).

**Before Phase 7:** Confirm CI runner (GitHub Actions) and that `AZURE_OPENAI_API_KEY` is stored as a GitHub Secret.

### 2.3 Agent Inventory

Fifteen agents total. Build them in this order:

```
Priority 1 (Phase 1-2):
  1. Orchestrator Agent        — primary client, workflow director
  2. Ingestion Agent           — fast client, file intake
  3. Artefact Tracking Agent   — fast client, observer
  4. Bug Log Agent             — fast client, observer

Priority 2 (Phase 3):
  5. EDA Agent                 — fast/primary split
  6. Cleaning Agent            — primary client
  7. Feature Engineering Agent — primary client

Priority 3 (Phase 4):
  8. Model Selection + Training Agent — primary client
  9. HP Tuning Agent                  — fast client (Optuna loop)
 10. Evaluation Agent                 — primary client

Priority 4 (Phase 5):
 11. Explainability Agent — primary client
 12. Report Agent         — primary client
 13. Deployment Agent     — fast client

Priority 5 (Phase 6):
 14. Memory Agent  — fast client
 15. Debug Agent   — primary client (no safety middleware)
```

### 2.4 MCP Server Inventory

Three servers, three crash domains:

| Server | Port | `stateless_http` | Tools |
|---|---|---|---|
| Tracking MCP Server | 8100 | `False` | `record_start/end/checkpoint/artefact/metric/lineage`, `query_*` |
| DS Tools MCP Server | 8101 | `True` | `read_file`, `execute_code`, `get_sample`, `write_output`, `search_docs`, `web_research`, `embed_text`, `semantic_search`, `log_metrics`, `verify_stage` |
| Bug Log MCP Server | 8102 | `False` | `record_bug`, `record_repair_attempt`, `mark_bug_resolved`, `record_ralph_loop_iteration`, `query_bug_history`, `query_bug_patterns`, `query_unresolved_bugs` |

---

## 3. Microsoft Agent Framework — Key Concepts

### 3.1 What MAF Is

Microsoft Agent Framework (MAF) is the production successor to both **Semantic Kernel** and **AutoGen**, built by the same teams. It combines AutoGen's simple agent abstractions with Semantic Kernel's enterprise features (session state, type safety, middleware, telemetry) and adds graph-based workflows for multi-agent orchestration.

Key capabilities relevant to this project:

- **Agents** — wrap an LLM client, instructions, tools, middleware, and session context.
- **Workflows** — graph-based execution across multiple agents with explicit control flow.
- **Middleware** — intercepts every LLM call (ChatMiddleware) or every tool call (FunctionMiddleware).
- **Session threads** — serialisable multi-turn conversation state. `propagate_session=True` shares state between parent and child agents.
- **MCP integration** — `MCPStreamableHTTPTool` connects to any MCP Streamable HTTP server.
- **Observability** — OpenTelemetry built in; spans propagate via W3C `traceparent` headers into MCP calls automatically.
- **Provider flexibility** — Azure OpenAI, OpenAI Responses, OpenAI Chat, Foundry, Anthropic, Ollama, A2A agents.

### 3.2 Agent Construction (MAF 1.0 Pattern)

```python
from agent_framework import ChatAgent
from agent_framework.openai import AzureOpenAIChatCompletionClient
from agent_framework import InMemoryHistoryProvider
from agents.clients import PRIMARY_CLIENT
from agents.middleware import LoggingMiddleware, RetryMiddleware, SafetyMiddleware, TelemetryMiddleware

def build_cleaning_agent() -> ChatAgent:
    return ChatAgent(
        chat_client=PRIMARY_CLIENT,
        name="CleaningAgent",
        instructions=CLEANING_AGENT_SYSTEM_PROMPT,
        tools=[
            # MCP tools via MCPStreamableHTTPTool — registered below
        ],
        middleware=[
            LoggingMiddleware(),
            SafetyMiddleware(),
            RetryMiddleware(max_retries=3),
            TelemetryMiddleware(),
        ],
        context_providers=[InMemoryHistoryProvider(max_messages=20)],
    )
```

### 3.3 Tool Definition (MAF 1.0 Pattern)

```python
from typing import Annotated
from pydantic import Field
from agent_framework import tool, FunctionInvocationContext

@tool
async def write_output(
    content: Annotated[str, Field(description="The content to write to the file.")],
    filename: Annotated[str, Field(description="Filename inside the run output directory.")],
    ctx: FunctionInvocationContext,
) -> str:
    """Write content to a file in the run's output directory."""
    run_id = ctx.function_invocation_kwargs["run_id"]
    output_dir = Path(f"outputs/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)
```

**Rules:**
- Every parameter has a type annotation.
- Every parameter has a `Field(description=...)`.
- Return type is always `str`.
- `FunctionInvocationContext` is never in the model's JSON schema.
- No `**kwargs`.

### 3.4 Sub-Agent as Tool (MAF 1.0 Pattern)

```python
# Orchestrator setup
from agent_framework import ChatAgent

cleaning_agent = build_cleaning_agent()

orchestrator = ChatAgent(
    chat_client=PRIMARY_CLIENT,
    name="Orchestrator",
    instructions=ORCHESTRATOR_SYSTEM_PROMPT,
    tools=[
        # pipeline sub-agents — propagate_session shares session state
        cleaning_agent.as_tool(
            name="run_cleaning_agent",
            description="Run the Cleaning Agent to impute nulls, handle outliers, and produce cleaned_dataset.parquet.",
            propagate_session=True,
        ),
        # debug agent — does NOT propagate session (read-only)
        debug_agent.as_tool(
            name="run_debug_agent",
            description="Run the Debug Agent to repair failing code and return corrected code.",
            propagate_session=False,
        ),
    ],
)
```

### 3.5 Session Threading (MAF 1.0)

```python
# Non-streaming — get complete response
result = await orchestrator.run(
    "Begin the data science pipeline.",
    run_id=run_id,        # passed as function_invocation_kwarg
    stage_name="orchestrator",
)

# Streaming — yield tokens as they arrive
async for update in orchestrator.run_stream("...", run_id=run_id):
    if update.text:
        print(update.text, end="", flush=True)
```

### 3.6 MCP Client Connection (MAF 1.0)

```python
from agent_framework import MCPStreamableHTTPTool
import os

ds_tools = MCPStreamableHTTPTool(url=os.environ["DS_TOOLS_MCP_URL"])
tracking  = MCPStreamableHTTPTool(url=os.environ["TRACKING_MCP_URL"])
bug_log   = MCPStreamableHTTPTool(url=os.environ["BUG_LOG_MCP_URL"])

# Pass to agent as a tool source — MAF discovers individual tools from the MCP server
agent = ChatAgent(
    ...,
    tools=[ds_tools, tracking],
)
```

> **URL Note:** FastAPI mounts FastMCP at `/mcp`; FastMCP adds its own `/mcp` suffix.
> Full path is always `/mcp/mcp`. Configured via `TRACKING_MCP_URL` etc. in `.env`.

### 3.7 Middleware Pattern (MAF 1.0)

```python
from agent_framework import AgentMiddleware, ChatContext
from typing import Callable, Awaitable

class LoggingMiddleware(AgentMiddleware):
    async def process(
        self,
        context: ChatContext,
        next: Callable[[ChatContext], Awaitable[None]],
    ) -> None:
        import time, json
        start = time.monotonic()
        print(json.dumps({"event": "agent_start", "agent": context.agent_name}))
        await next(context)
        elapsed = round((time.monotonic() - start) * 1000)
        print(json.dumps({"event": "agent_end", "duration_ms": elapsed}))
```

> **Important:** `ChatMiddleware` runs on **every LLM call** within a single `run()`, not once per `run()`.
> All middleware must be re-entrant and side-effect-safe.

---

## 4. Directory Structure

```
maf_ds_agent/
├── .env                          # Credentials — gitignored
├── .env.example                  # Safe template — committed
├── .gitignore
├── pyproject.toml                # All deps pinned with ==
├── README.md                     # Specification (source of truth)
├── copilot.md                    # This file — implementation guide
│
├── main.py                       # Entry point — load_dotenv(), start servers, run pipeline
│
├── config/
│   ├── settings.py               # Pydantic-Settings model loading .env
│   ├── safety_patterns.py        # Blocked tool-input regex patterns
│   ├── error_catalogue.py        # Canonical error ID map
│   └── library_registry.py       # Documentation URLs per library
│
├── agents/
│   ├── clients.py                # PRIMARY_CLIENT, FAST_CLIENT (two instances only)
│   ├── middleware.py             # LoggingMiddleware, SafetyMiddleware, RetryMiddleware,
│   │                             #   TelemetryMiddleware, RateLimitMiddleware
│   ├── base.py                   # MCP tool instances (one per server per agent)
│   ├── orchestrator.py
│   ├── ingestion.py
│   ├── eda.py
│   ├── cleaning.py
│   ├── feature_engineering.py
│   ├── training.py
│   ├── tuning.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── report.py
│   ├── deployment.py
│   ├── debug.py
│   ├── artefact_tracker.py
│   ├── bug_logger.py
│   └── memory.py
│
├── mcp_servers/
│   ├── tracking/
│   │   ├── server.py             # FastMCP + FastAPI app (port 8100, stateless=False)
│   │   ├── tools.py              # All @mcp.tool() definitions
│   │   └── db.py                 # SQLite WAL setup and query helpers
│   ├── ds_tools/
│   │   ├── server.py             # FastMCP + FastAPI app (port 8101, stateless=True)
│   │   ├── tools.py              # read_file, execute_code, web_research, etc.
│   │   └── file_type_detector.py # Magika singleton + three-layer detection
│   └── bug_log/
│       ├── server.py             # FastMCP + FastAPI app (port 8102, stateless=False)
│       ├── tools.py
│       └── db.py
│
├── tools/
│   └── local_tools.py            # get_session_state, set_session_state,
│                                  #   check_artefact_exists, format_verification_result
│
├── workflows/
│   ├── criteria.py               # DONE_CRITERIA dict for every stage
│   ├── ralph_loop.py             # RalphLoop class wrapping every stage invocation
│   └── graph.py                  # WorkflowBuilder — edge definitions
│
├── templates/
│   ├── eda_template.py
│   ├── cleaning_template.py
│   ├── feature_template.py
│   ├── training_templates/       # One per task_type x algorithm
│   ├── tuning_template.py
│   ├── evaluation_template.py
│   ├── shap_template.py
│   └── report_template.html
│
├── data/
│   ├── memory.db                 # SQLite — episodic_events table
│   ├── semantic_memory.json      # Artefact records for semantic search
│   ├── semantic_embeddings.npy   # Embedding vectors (NumPy)
│   └── procedural_memory.json    # error_strategies, template_performance, etc.
│
├── outputs/                      # Created at runtime — gitignored
│   └── {run_id}/
│       ├── ingestion/
│       ├── eda/
│       ├── cleaning/
│       ├── features/
│       ├── training/
│       ├── tuning/
│       ├── evaluation/
│       ├── explainability/
│       ├── report/
│       ├── deployment/
│       ├── debug/
│       ├── metrics/
│       └── escalations/
│
└── tests/
    ├── unit/
    │   ├── test_file_detection.py
    │   ├── test_ralph_loop.py
    │   ├── test_mcp_servers.py
    │   └── test_memory.py
    ├── integration/
    │   ├── test_phase1.py         # MCP health checks
    │   ├── test_phase2.py         # Ingestion on 10 file types
    │   ├── test_phase3.py         # EDA + Cleaning + Features on iris
    │   ├── test_phase4.py         # Training + Tuning + Evaluation
    │   └── test_full_pipeline.py  # End-to-end on iris, titanic, boston
    └── fixtures/
        └── sample_datasets/
```

---

## 5. Phase 1 — Environment Bootstrap and MCP Servers

### 5.1 Installation

Install in this exact order to avoid conflicts:

```bash
pip install agent-framework
pip install mcp fastmcp
pip install browser-use duckduckgo-search
pip install magika filetype chardet
pip install pandas pyarrow openpyxl xlrd
pip install pymupdf python-docx pillow
pip install scikit-learn xgboost lightgbm catboost
pip install torch torchvision transformers datasets sentence-transformers
pip install shap lime captum fairlearn
pip install optuna
pip install fastapi uvicorn
pip install opentelemetry-sdk opentelemetry-exporter-otlp
pip install azure-identity
pip install python-dotenv pydantic-settings
pip install numpy scipy statsmodels
pip install matplotlib seaborn plotly kaleido
pip install onnx onnxruntime safetensors
pip install pytest pytest-asyncio ruff mypy
```

Pin all to `==` in `pyproject.toml` after confirming compatibility. Run `pip freeze > requirements.lock` after a clean install.

### 5.2 FastMCP Server Pattern (apply to all three servers)

```python
# mcp_servers/tracking/server.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastmcp import FastMCP
from .tools import register_tools

mcp = FastMCP(name="TrackingMCPServer")
register_tools(mcp)  # all @mcp.tool() definitions in tools.py

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager:     # ← REQUIRED: initialises StreamableHTTPSessionManager
        yield

parent_app = FastAPI(lifespan=lifespan)
parent_app.mount("/mcp", mcp.streamable_http_app(stateless_http=False))

@parent_app.get("/health")
async def health():
    return {"status": "healthy", "server": "tracking"}

# Run with: uvicorn mcp_servers.tracking.server:parent_app --port 8100
```

> **Critical:** `async with mcp.session_manager:` inside the lifespan is **mandatory**.
> Omitting it causes `BrokenResourceError` on every MCP connection. This is the most common Phase 1 mistake.

### 5.3 MCP Server Verification (run after starting each server)

```bash
# Check 1: HEAD must return 200
curl -X HEAD http://localhost:8100/mcp/mcp

# Check 2: GET must return 405 with Allow: POST header (not 501)
curl -I http://localhost:8100/mcp/mcp

# Check 3: MCP initialise handshake must return Mcp-Session-Id header
curl -X POST http://localhost:8100/mcp/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}},"id":1}'
```

All three checks must pass before proceeding. Document failures with exact error output.

### 5.4 SQLite Setup for Tracking and Bug Log Servers

```python
# mcp_servers/tracking/db.py
import sqlite3
import os

DB_PATH = os.environ.get("MEMORY_DB_PATH", "data/memory.db")

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent readers + one writer
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS run_log (
            log_entry_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            stage_name TEXT NOT NULL,
            session_id TEXT,
            agent_name TEXT,
            status TEXT,
            started_at TEXT,
            ended_at TEXT,
            duration_ms INTEGER
        );
        CREATE TABLE IF NOT EXISTS artefacts (
            artefact_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            stage_name TEXT NOT NULL,
            artefact_path TEXT NOT NULL,
            artefact_type TEXT NOT NULL,
            content_hash TEXT,
            metadata_json TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS metrics (
            metric_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            stage_name TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            metadata_json TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS lineage (
            parent_artefact_id TEXT NOT NULL,
            child_artefact_id TEXT NOT NULL,
            relationship TEXT NOT NULL,
            PRIMARY KEY (parent_artefact_id, child_artefact_id)
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
            run_id TEXT NOT NULL,
            stage_name TEXT NOT NULL,
            created_at TEXT,
            PRIMARY KEY (run_id, stage_name)
        );
    """)
    conn.commit()
```

### 5.5 Phase 1 Exit Gate

All of the following must be true before starting Phase 2:

- [ ] `pip install agent-framework` succeeds; `from agent_framework import ChatAgent` imports without error.
- [ ] All three MCP servers start without errors: `uvicorn mcp_servers.tracking.server:parent_app --port 8100`, same for 8101 and 8102.
- [ ] All three servers pass the three-step curl verification (§5.3).
- [ ] `GET http://localhost:8100/health`, `8101/health`, `8102/health` return `{"status":"healthy"}`.
- [ ] `python -c "from agents.clients import PRIMARY_CLIENT, FAST_CLIENT; print('OK')"` succeeds.

---

## 6. Phase 2 — Core Agents and Read File Pipeline

### 6.1 `read_file` Tool — Three-Layer Detection

```python
# mcp_servers/ds_tools/file_type_detector.py
from magika import Magika
import filetype, chardet, csv, ast, json

_MAGIKA_INSTANCE = None

def get_magika() -> Magika:
    global _MAGIKA_INSTANCE
    if _MAGIKA_INSTANCE is None:
        _MAGIKA_INSTANCE = Magika()
    return _MAGIKA_INSTANCE

def detect_file_type(file_path: str) -> dict:
    """Three-layer detection: Magika → filetype → content heuristics."""
    path = Path(file_path)

    # Layer 1: Magika
    result = get_magika().identify_path(path)
    if result.output.score >= 0.85:
        return _build_result(result.output.ct_label, result.output.mime_type, file_path)

    # Layer 2: magic bytes
    ft = filetype.guess(file_path)
    if ft:
        return _build_result(ft.extension, ft.mime, file_path)

    # Layer 3: content heuristics (text disambiguation)
    return _text_heuristic(file_path, result)
```

Routing table (per §14.3 of README):

| Detected type | `pipeline_variant` |
|---|---|
| `csv`, `parquet`, `xlsx`, `json`, `jsonl`, `sqlite` | `tabular` |
| `pdf`, `docx`, `txt`, `md` | `document_text` |
| `png`, `jpeg`, `webp`, `gif`, `bmp` | `image` |
| `pickle`, `onnx`, `safetensors` | `existing_model` (skip to Evaluation) |
| `zip`, `tar` | human gate |
| unknown | `supported=False` → human gate |

### 6.2 Ingestion Agent System Prompt (abbreviated)

```
You are the Ingestion Agent. Your sole responsibility is to call read_file,
validate the result against the task description, detect PII, and record
the raw artefact.

Authorised tools: read_file, record_start, record_end, record_artefact, write_output.

Processing protocol:
1. Call record_start.
2. Call read_file with input_artefact_path from session state.
3. If supported=False: return routing mismatch error.
4. If requires_ocr=True or requires_transcription=True: return flag for human gate.
5. Validate routing (file type must be consistent with task description).
6. Run PII heuristic scan on schema column names and sample values.
7. If PII detected: set session state pii_detected=True and log artefact quality warning.
8. Write file_type_result, schema, input_artefact_path to session state.
9. Call record_artefact with artefact_type=raw_dataset.
10. Call record_end. Output <DONE>ingest</DONE> and final JSON summary.

Boundaries: Do not clean, transform, or analyse data.
```

### 6.3 Artefact Tracking Agent — Polling Loop

```python
# agents/artefact_tracker.py
import asyncio, json
from pathlib import Path
from agents.base import TRACKING_TOOL, DS_TOOLS_TOOL

class ArtefactTrackingAgent:
    def __init__(self, agent: ChatAgent, run_id: str):
        self.agent = agent
        self.run_id = run_id
        self._seen: set[str] = set()

    async def run(self):
        while True:
            history = await TRACKING_TOOL.call("query_artefact_history", run_id=self.run_id)
            for record in json.loads(history):
                path = record["artefact_path"]
                if path not in self._seen:
                    self._seen.add(path)
                    await self._process_new_artefact(record)
            await asyncio.sleep(10)

    async def _process_new_artefact(self, record: dict):
        # Steps 1-6 from §12.3: hash, type, metadata, lineage, memory update, write record
        ...
```

### 6.4 Phase 2 Exit Gate

- [ ] `read_file` correctly detects and parses all file types: CSV, Parquet, JSON, JSONL, XLSX, PDF, PNG, DOCX, pickle, ONNX.
- [ ] Ingestion Agent runs on `tests/fixtures/sample_datasets/iris.csv` and writes all session state keys.
- [ ] Artefact Tracking Agent records the artefact in the Tracking DB within 15 seconds.
- [ ] Bug Log Agent records a synthetic bug injected via a test, with the enrichment data.
- [ ] `python -m pytest tests/unit/test_file_detection.py -v` passes.

---

## 7. Phase 3 — EDA, Cleaning, and Feature Engineering

### 7.1 Ralph Loop Pattern

Every stage is wrapped by the `RalphLoop` class:

```python
# workflows/ralph_loop.py
import asyncio
from workflows.criteria import DONE_CRITERIA

class RalphLoop:
    def __init__(self, stage_name: str, agent_tool_fn, bug_log_tool, max_iterations: int = 8):
        self.stage = stage_name
        self.run_agent = agent_tool_fn
        self.bug_log = bug_log_tool
        self.max_iterations = max_iterations
        self.criteria = DONE_CRITERIA[stage_name]

    async def run(self, session, run_id: str) -> dict:
        for iteration in range(1, self.max_iterations + 1):
            result = await self.run_agent(session=session, run_id=run_id, iteration=iteration)

            # Check completion promise tag
            if f"<DONE>{self.stage}</DONE>" not in (result.text or ""):
                await self.bug_log.call("record_ralph_loop_iteration",
                    run_id=run_id, stage_name=self.stage,
                    iteration_number=iteration,
                    verification_failures=["missing_completion_promise"],
                    action_taken="inject_reminder")
                continue

            # Run deterministic verification
            verify = await DS_TOOLS_TOOL.call("verify_stage", run_id=run_id, stage_name=self.stage)
            if verify["passed"]:
                return {"status": "success", "iterations": iteration}

            # Inject failure context and retry
            failures = verify["failures"]
            await self.bug_log.call("record_ralph_loop_iteration",
                run_id=run_id, stage_name=self.stage,
                iteration_number=iteration,
                verification_failures=failures,
                action_taken="retry_with_feedback")
            session.inject_feedback(failures)

        # MAX_ITERATIONS reached — escalate
        return {"status": "escalate", "iterations": self.max_iterations}
```

### 7.2 Stage Criteria (`workflows/criteria.py`)

```python
DONE_CRITERIA = {
    "eda": {
        "required_files": ["stats.json", "eda_narrative.md"],
        "required_chart_count": 3,
        "required_keys_in_stats": ["describe", "null_counts", "correlation_matrix", "data_quality_flags"],
        "required_session_keys": ["eda_report_path", "chart_paths"],
    },
    "clean": {
        "required_files": ["cleaned_dataset.parquet", "transformation_log.json"],
        "assertions": [
            lambda session, outdir: (
                _null_count_zero_in_imputed_columns(outdir / "cleaned_dataset.parquet", session),
                "Null values remain in imputed columns"
            ),
        ],
        "required_session_keys": ["cleaned_dataset_path", "transformation_log_path"],
    },
    "feature_eng": {
        "required_files": ["features_train.parquet", "features_test.parquet", "feature_manifest.json"],
        "assertions": [
            lambda session, outdir: (
                _manifest_covers_all_columns(outdir / "feature_manifest.json", session),
                "feature_manifest.json does not cover 100% of columns"
            ),
        ],
        "required_session_keys": ["features_train_path", "features_test_path", "task_type"],
    },
    # ... training, tuning, evaluation, etc.
}
```

### 7.3 EDA Template Skeleton

```python
# templates/eda_template.py  (generated code, executed via execute_code MCP tool)
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

df = pd.read_parquet("{{input_path}}")
out = Path("{{output_dir}}")
out.mkdir(parents=True, exist_ok=True)

stats = {
    "describe": df.describe(include="all").to_dict(),
    "null_counts": df.isnull().sum().to_dict(),
    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    "correlation_matrix": df.select_dtypes("number").corr().to_dict(),
    "data_quality_flags": [],
}

# Flag high null rate columns
for col, nc in stats["null_counts"].items():
    if nc / len(df) > 0.2:
        stats["data_quality_flags"].append({"column": col, "issue": "high_null_rate", "rate": nc/len(df)})

(out / "stats.json").write_text(json.dumps(stats, default=str))

# Charts
fig, ax = plt.subplots(); df.select_dtypes("number").hist(ax=ax); plt.savefig(out/"hist.png"); plt.close()
# ... correlation heatmap, missing value bar chart
```

### 7.4 Phase 3 Exit Gate

- [ ] Full mini-pipeline (EDA → Cleaning → Features) completes on `iris.csv` without human intervention in dev mode.
- [ ] All three stage verifications pass (Ralph Loop exits `status=success` on the first or second iteration for iris).
- [ ] `transformation_log.json` contains a `justification` field for every transform applied.
- [ ] `feature_manifest.json` covers 100% of columns; zero nulls in features.

---

## 8. Phase 4 — Training, Tuning, and Evaluation

### 8.1 Task Type Routing

`task_type` is set during Feature Engineering. It determines the training template:

| `task_type` | Templates Available |
|---|---|
| `tabular_binary_classification` | LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost |
| `tabular_multiclass_classification` | same + SoftMax variants |
| `tabular_regression` | Ridge, RF, XGBoost, LightGBM |
| `tabular_clustering` | KMeans, DBSCAN, GaussianMixture |
| `nlp_classification` | DistilBERT fine-tune, TF-IDF + LogReg |
| `nlp_generation` | GPT-style fine-tune via HuggingFace |
| `cv_classification` | ResNet fine-tune, EfficientNet |
| `cv_detection` | YOLO, Faster-RCNN |
| `cv_generation` | Stable Diffusion LoRA |
| `time_series_forecasting` | ARIMA, Prophet, LSTM, Transformer |
| `anomaly_detection` | IsolationForest, AutoEncoder |
| `recommendation` | SVD, LightFM, Two-Tower |

### 8.2 Training Agent — Algorithm Comparison Protocol

```
1. Select top 3 candidate algorithms for the task_type using the Model Selection sub-prompt.
2. Query procedural memory template_performance for this task_type.
3. For each candidate:
   a. Generate training code from the matching template.
   b. Call execute_code.
   c. Record baseline_metrics.
4. Select the algorithm with the best primary metric as the winner.
5. Save the winning model as model_artefact.pkl (sklearn) or model_artefact.pt (PyTorch).
6. Export to ONNX via the appropriate converter.
7. Call log_metrics with algorithm_comparison list.
8. Write model_artefact_path, model_type, baseline_metrics to session state.
```

### 8.3 Tuning Agent — Optuna Integration

```python
# Generated tuning code (executed via execute_code MCP tool)
import optuna, json
from pathlib import Path

# Load hyperparameter priors from procedural memory
with open("{{priors_path}}") as f:
    priors = json.load(f).get("{{model_type}}", {})

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators",
            priors.get("n_estimators_min", 50),
            priors.get("n_estimators_max", 500)),
        "max_depth": trial.suggest_int("max_depth",
            priors.get("max_depth_min", 3),
            priors.get("max_depth_max", 10)),
        # ... other params
    }
    # train + cross-val
    return cv_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=300)
(Path("{{output_dir}}") / "best_params.json").write_text(json.dumps(study.best_params))
```

**Completion criteria:** `tuned_metric >= baseline_metric` (verified deterministically by Ralph Loop).

### 8.4 Evaluation Agent — Fairness Checks

The Evaluation Agent generates an `evaluation_report.json` that **must** contain:

```json
{
  "primary_metric": {"name": "roc_auc", "value": 0.94},
  "secondary_metrics": {},
  "confusion_matrix": [],
  "roc_curve": {},
  "fairness_metrics": {
    "demographic_parity_difference": null,
    "equalized_odds_difference": null,
    "note": "No protected column detected in dataset"
  },
  "drift_report": {"features_drifted": []},
  "deployment_recommendation": {
    "recommend": "proceed",
    "blocking_issues": [],
    "advisory_issues": []
  }
}
```

`fairness_metrics` section is **mandatory** even if it only contains a note. Its absence triggers Ralph Loop retry.

### 8.5 Phase 4 Exit Gate

- [ ] Training Agent produces `model_artefact_path` in session state and the file exists on disk.
- [ ] Tuning Agent achieves tuned metric ≥ baseline metric (verified by `verify_stage("tuning")`).
- [ ] `evaluation_report.json` has `fairness_metrics` section, all primary metric keys, and a `deployment_recommendation`.
- [ ] All three Ralph Loops exit `status=success`.

---

## 9. Phase 5 — Explainability, Reporting, and Deployment

### 9.1 Explainability Agent — SHAP Requirements

The Explainability Agent **must** produce:
- At minimum **2** SHAP plots (summary beeswarm + bar chart for feature importance).
- `explanation_narrative.md` — non-empty prose explaining the three most influential features.
- `shap_values.npy` — raw SHAP values array.

```python
# Generated SHAP code
import shap, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Load model + test data
explainer = shap.TreeExplainer(model)  # or KernelExplainer for non-tree
shap_values = explainer.shap_values(X_test)
np.save(out / "shap_values.npy", shap_values)

plt.figure(); shap.summary_plot(shap_values, X_test, show=False); plt.savefig(out/"shap_beeswarm.png"); plt.close()
plt.figure(); shap.summary_plot(shap_values, X_test, plot_type="bar", show=False); plt.savefig(out/"shap_bar.png"); plt.close()
```

### 9.2 Report Agent — HTML Self-Containment

The `final_report.html` must be fully self-contained (no external CDN links). All charts are embedded as base64 data URIs. The Ralph Loop verification checks for these required section headings: `Executive Summary`, `Data Profile`, `Model Performance`, `Feature Importance`, `Fairness Assessment`, `Recommendations`.

### 9.3 Deployment Agent — Smoke Test

The Deployment Agent wraps the ONNX model in a FastAPI endpoint and runs 10 smoke test predictions against it. All 10 must succeed.

```python
# Generated deployment code
from fastapi import FastAPI
import onnxruntime as ort, numpy as np

app = FastAPI()
session = ort.InferenceSession("{{onnx_model_path}}")

@app.post("/predict")
def predict(data: dict):
    input_name = session.get_inputs()[0].name
    arr = np.array(data["features"], dtype=np.float32).reshape(1, -1)
    result = session.run(None, {input_name: arr})
    return {"prediction": result[0].tolist()}
```

### 9.4 Phase 5 Exit Gate

- [ ] ≥ 2 SHAP plots exist in the explainability output directory.
- [ ] `explanation_narrative.md` is non-empty.
- [ ] `final_report.html` passes the section-heading check and is self-contained (no external URLs).
- [ ] Deployment smoke test: 10/10 predictions succeed.

---

## 10. Phase 6 — Memory System

### 10.1 Memory Agent — Startup Registration

```python
# agents/memory.py
import asyncio, json
from pathlib import Path

class MemoryAgent:
    async def run(self, run_id: str):
        while True:
            await self._sync_episodic(run_id)
            await self._sync_semantic(run_id)
            await asyncio.sleep(15)

    async def _sync_episodic(self, run_id: str):
        """Read new stage completion events and write episode records to memory.db."""
        ...

    async def _sync_semantic(self, run_id: str):
        """Embed new artefact descriptions and update semantic_embeddings.npy."""
        ...
```

### 10.2 Semantic Search — NumPy Cosine Similarity

```python
# mcp_servers/ds_tools/tools.py — semantic_search tool
import numpy as np, json
from pathlib import Path

def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix @ query_vec) / (norms.flatten() * np.linalg.norm(query_vec) + 1e-9)

@mcp.tool()
async def semantic_search(query: str, top_k: int = 5) -> str:
    records = json.loads(Path(SEMANTIC_MEMORY_PATH).read_text())
    embeddings = np.load(SEMANTIC_EMBEDDINGS_PATH)
    query_vec = await embed_text_impl(query)
    scores = cosine_similarity(np.array(query_vec), embeddings)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return json.dumps([{**records[i], "score": float(scores[i])} for i in top_idx])
```

### 10.3 Procedural Memory Update

After each resolved bug, the Bug Log Agent updates `data/procedural_memory.json`:

```python
def update_error_strategy(error_class: str, library: str, strategy: str, success: bool):
    pm = json.loads(Path(PROCEDURAL_MEMORY_PATH).read_text())
    key = f"{error_class}:{library}"
    strategies = pm.setdefault("error_strategies", {}).setdefault(key, [])
    entry = next((s for s in strategies if s["name"] == strategy), None)
    if not entry:
        entry = {"name": strategy, "success": 0, "failure": 0}
        strategies.append(entry)
    entry["success" if success else "failure"] += 1
    # Sort by success rate descending
    strategies.sort(key=lambda s: s["success"] / max(s["success"] + s["failure"], 1), reverse=True)
    Path(PROCEDURAL_MEMORY_PATH).write_text(json.dumps(pm, indent=2))
```

### 10.4 Phase 6 Exit Gate

- [ ] Second run on `iris.csv` retrieves at least one relevant episode from episodic memory.
- [ ] `semantic_search("classification model trained on iris")` returns the iris model artefact with score > 0.7.
- [ ] `data/procedural_memory.json` contains at least one `error_strategies` entry after an intentionally injected error.

---

## 11. Phase 7 — Hardening and Production Gate

### 11.1 Safety Patterns

```python
# config/safety_patterns.py
import re

BLOCKED_PATTERNS = [
    re.compile(r"os\.system\s*\("),
    re.compile(r"subprocess\.call\s*\("),
    re.compile(r"eval\s*\("),
    re.compile(r"exec\s*\("),
    re.compile(r"__import__\s*\("),
    re.compile(r"open\s*\([^)]*['\"]w['\"]"),   # write to arbitrary paths
    re.compile(r"shutil\.rmtree"),
    re.compile(r"DROP\s+TABLE", re.IGNORECASE),
    re.compile(r"DELETE\s+FROM", re.IGNORECASE),
]

def check_tool_input(tool_name: str, input_json: str) -> tuple[bool, str]:
    """Returns (safe, reason). Called by SafetyMiddleware before every tool call."""
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(input_json):
            return False, f"Blocked pattern matched: {pattern.pattern}"
    return True, ""
```

### 11.2 Token Budget Wrapper

```python
# agents/clients.py
import threading

class TokenBudgetWrapper:
    def __init__(self, client, budget: int):
        self._client = client
        self._budget = budget
        self._totals: dict[str, int] = {}
        self._lock = threading.Lock()

    async def complete(self, *args, run_id: str, **kwargs):
        result = await self._client.complete(*args, **kwargs)
        tokens = result.usage.total_tokens if result.usage else 0
        with self._lock:
            self._totals[run_id] = self._totals.get(run_id, 0) + tokens
            if self._totals[run_id] > self._budget:
                raise TokenBudgetExceededError(
                    f"Run {run_id} exceeded token budget of {self._budget:,}"
                )
        return result
```

### 11.3 Integration Test Datasets

Run the full pipeline on all three before declaring Phase 7 complete:

| Dataset | Task | Expected primary metric |
|---|---|---|
| `iris.csv` | multiclass classification | accuracy > 0.95 |
| `titanic.csv` | binary classification | roc_auc > 0.80 |
| `boston_housing.csv` | regression | r2 > 0.75 |

### 11.4 Phase 7 Exit Gate

- [ ] All safety pattern tests pass (100 injected malicious inputs blocked, 0 false positives on normal inputs).
- [ ] Token budget triggers correctly when injected with a synthetic overrun.
- [ ] Full pipeline on all three integration datasets passes all stage verifications.
- [ ] `python -m pytest tests/integration/test_full_pipeline.py -v` green.

---

## 12. Agent Construction Patterns

### 12.1 Correct Import Map (MAF 1.0)

```python
from agent_framework import ChatAgent
from agent_framework import tool, AgentMiddleware
from agent_framework import AgentRunContext, FunctionInvocationContext
from agent_framework import InMemoryHistoryProvider
from agent_framework.openai import AzureOpenAIChatCompletionClient
from agent_framework import MCPStreamableHTTPTool
from fastmcp import FastMCP
from fastapi import FastAPI
from azure.identity import DefaultAzureCredential
```

> These are the **exact** import paths for MAF 1.0. Do not guess alternatives.

### 12.2 Agent Middleware Ordering (all pipeline agents)

```
LoggingMiddleware → RateLimitMiddleware → SafetyMiddleware → RetryMiddleware → TelemetryMiddleware
```

Debug Agent (no safety middleware — needs to reason about dangerous code patterns):
```
LoggingMiddleware → RetryMiddleware → TelemetryMiddleware
```

Observer agents (no rate limiting, no safety):
```
LoggingMiddleware → RetryMiddleware → TelemetryMiddleware
```

### 12.3 Final Message Contract (all sub-agents)

Every sub-agent run **must** end with this structure in its final message:

```json
{
  "session_state_keys_written": ["cleaned_dataset_path", "transformation_log_path"],
  "artefacts_produced": [
    "outputs/{run_id}/cleaning/cleaned_dataset.parquet",
    "outputs/{run_id}/cleaning/transformation_log.json"
  ],
  "status": "success",
  "completion_promise": "<DONE>clean</DONE>",
  "summary": "Applied mean imputation to 3 columns; IQR capping to 2 outlier columns."
}
```

The Orchestrator reads `artefacts_produced`; the Ralph Loop reads `completion_promise`.

---

## 13. Ralph Loop Implementation Checklist

Use this checklist for every stage. All boxes must be checked before moving to the next stage.

- [ ] `DONE_CRITERIA[stage_name]` defined in `workflows/criteria.py`.
- [ ] Stage agent's system prompt includes `<DONE>stage_name</DONE>` completion promise in §6 (Completion Signal).
- [ ] `verify_stage` MCP tool implementation runs `DONE_CRITERIA[stage_name]` assertions deterministically (no LLM).
- [ ] Ralph Loop injects failure details (exact file path missing, exact assertion failure message) into the next iteration's context.
- [ ] Each retry uses a fresh context window (new session thread); disk state and bug log carry state between retries.
- [ ] `MAX_ITERATIONS` check escalates to human gate with bug escalation report.
- [ ] Escalation report written to `outputs/{run_id}/escalations/{stage_name}.md`.

---

## 14. MCP Server Verification Checklist

Run this checklist each time an MCP server is started or updated:

```bash
SERVER=localhost:8100   # change to 8101 or 8102

# 1. Health endpoint
curl -s http://$SERVER/health | python -m json.tool

# 2. HEAD → 200
curl -o /dev/null -w "%{http_code}" -X HEAD http://$SERVER/mcp/mcp

# 3. GET → 405 with Allow: POST
curl -I http://$SERVER/mcp/mcp 2>&1 | grep -E "HTTP/|Allow:"

# 4. MCP initialize → Mcp-Session-Id in response headers
curl -v -X POST http://$SERVER/mcp/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}},"id":1}' \
  2>&1 | grep -i "mcp-session"

# 5. List tools (use session ID from step 4)
curl -X POST http://$SERVER/mcp/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: <session-id-from-step-4>" \
  -d '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":2}'
```

**All five checks must pass.** A `501 Not Implemented` on step 3 means `async with mcp.session_manager:` is missing from the lifespan (§5.2).

---

## 15. Inter-Agent Contract Quick Reference

### 15.1 Session State Keys Written per Stage

| Stage | Keys Written |
|---|---|
| Orchestrator | `run_id`, `task_description`, `pipeline_variant` |
| Ingestion | `file_type_result`, `schema`, `input_artefact_path`, `pii_detected` |
| EDA | `eda_report_path`, `eda_narrative_path`, `chart_paths`, `data_quality_flags` |
| Cleaning | `cleaned_dataset_path`, `transformation_log_path`, `cleaning_summary`, `content_hash` |
| Feature Eng | `features_train_path`, `features_test_path`, `feature_manifest_path`, `target_column`, `task_type` |
| Training | `model_artefact_path`, `model_type`, `onnx_model_path`, `baseline_metrics`, `algorithm_comparison` |
| Tuning | `model_artefact_path` (updated), `best_params_path`, `tuning_metrics` |
| Evaluation | `evaluation_report_path`, `deployment_recommendation`, `fairness_metrics`, `drift_report` |
| Explainability | `shap_values_path`, `shap_plot_paths`, `explanation_narrative_path` |
| Report | `report_md_path`, `report_html_path` |
| Deployment | `endpoint_url`, `smoke_test_results` |

### 15.2 Required Files per Stage (for `verify_stage`)

| Stage | Required Files |
|---|---|
| `ingest` | `{run_id}/ingestion/schema.json` |
| `eda` | `stats.json`, `eda_narrative.md`, ≥3 chart PNGs |
| `clean` | `cleaned_dataset.parquet`, `transformation_log.json` |
| `feature_eng` | `features_train.parquet`, `features_test.parquet`, `feature_manifest.json` |
| `training` | `model_artefact.*`, `baseline_metrics.json`, `algorithm_comparison.json` |
| `tuning` | updated `model_artefact.*`, `best_params.json` |
| `evaluation` | `evaluation_report.json` (with `fairness_metrics` key) |
| `explainability` | `shap_values.npy`, ≥2 SHAP PNGs, `explanation_narrative.md` |
| `report` | `final_report.md`, `final_report.html` (self-contained) |
| `deployment` | `smoke_test_results.json` (10/10 successes) |

---

## 16. Known Pitfalls and Remediation

| Pitfall | Symptom | Fix |
|---|---|---|
| `BrokenResourceError` on MCP connect | All MCP calls fail | Add `async with mcp.session_manager:` inside `lifespan` (§5.2) |
| `405 on MCPStreamableHTTPTool.connect()` | MCP client fails on init | Upgrade `agent-framework` (GitHub issue #5317); check Framework version |
| `Magika()` called per request | High latency, memory growth | Use `get_magika()` singleton in `file_type_detector.py` |
| `load_dotenv()` not first line | `None` env vars at client construction | Move `load_dotenv()` to top of `main.py` before all imports |
| Sub-agent modifies orchestrator session | Unexpected session state corruption | Set `propagate_session=False` on Debug Agent; only pipeline agents use `True` |
| `ChatMiddleware` side effects | Middleware runs multiple times per `run()` | Make all middleware stateless and re-entrant |
| Ralph Loop never exits | Max iterations reached on every stage | Check that `verify_stage` is deterministic and not calling an LLM |
| Pickle untrusted content | Security: arbitrary code execution on load | Read only first 64 bytes; return `requires_verification=True` — never `pickle.load()` |
| API key committed to git | Security incident | `git filter-repo`; rotate key immediately in Azure portal |
| `DROP TABLE` in generated code | DB corruption | SafetyMiddleware blocked-pattern check on all tool inputs (§11.1) |

---

## 17. Testing Strategy

### 17.1 Unit Tests

```bash
python -m pytest tests/unit/ -v
```

| Test file | What it tests |
|---|---|
| `test_file_detection.py` | All 10 file types detected correctly; routing table correct |
| `test_ralph_loop.py` | Loop exits on success; loop retries on failure; escalates at max_iterations |
| `test_mcp_servers.py` | All three servers pass 5-step verification; tools callable |
| `test_memory.py` | Episodic write + read round-trip; cosine similarity returns correct top-k |

### 17.2 Integration Tests

```bash
python -m pytest tests/integration/ -v --timeout=300
```

Run after each phase is complete. Use `iris.csv` for Phases 1-6; full dataset suite for Phase 7.

### 17.3 Safety Tests

```bash
python -m pytest tests/unit/test_safety.py -v
```

Inject 100 malicious inputs (blocked-pattern strings) and verify `SafetyMiddleware` blocks all of them. Inject 50 normal tool calls and verify 0 false positives.

---

## 18. CI/CD Reference

### 18.1 GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      AI_FOUNDRY_PROJECT_ENDPOINT: ${{ secrets.AI_FOUNDRY_PROJECT_ENDPOINT }}
      AI_FOUNDRY_API_KEY: ${{ secrets.AI_FOUNDRY_API_KEY }}
      AZURE_OPENAI_PRIMARY_DEPLOYMENT: gpt-4o
      AZURE_OPENAI_FAST_DEPLOYMENT: gpt-4o
      AZURE_OPENAI_EMBEDDING_DEPLOYMENT: text-embedding-ada-002
      AI_FOUNDRY_API_VERSION: "2024-12-01-preview"

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.lock
      - run: python -m pytest tests/unit/ -v
      - run: python -m pytest tests/integration/test_phase1.py -v
      - run: ruff check .
      - run: mypy agents/ mcp_servers/ workflows/ --ignore-missing-imports
```

Store `AI_FOUNDRY_API_KEY` as a GitHub Secret. Never put it in the workflow YAML.

### 18.2 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: detect-private-key   # blocks accidental key commits
      - id: check-json
      - id: check-yaml
      - id: end-of-file-fixer
```

---

## Quick Command Reference

```bash
# Start all three MCP servers (separate terminals)
uvicorn mcp_servers.tracking.server:parent_app --port 8100 --reload
uvicorn mcp_servers.ds_tools.server:parent_app --port 8101 --reload
uvicorn mcp_servers.bug_log.server:parent_app --port 8102 --reload

# Run the pipeline
python main.py --file path/to/data.csv --task "Predict customer churn"

# Resume a failed run
python main.py --resume-run-id <uuid>

# Run tests
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v --timeout=300

# Lint + type check
ruff check . && mypy agents/ mcp_servers/ workflows/

# Check API key is working
curl https://idkopenai.services.ai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-12-01-preview \
  -H "api-key: $AI_FOUNDRY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"ping"}],"max_tokens":5}'
```

---

*Generated from `README.md` specification. Last reviewed: 2026-06-04.*
*For all authoritative decisions, refer to `README.md` section numbers.*
