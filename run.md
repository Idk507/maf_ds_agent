# MAF DS Agent — Complete Run & Test Guide

> **Stack:** Microsoft Agent Framework 1.0 (Python) · Azure OpenAI (GPT-4o) · MCP Streamable HTTP · FastMCP 3.x
>
> **Purpose:** This guide documents every step needed to install, configure, run, and test the **MAF DS Agent** — an autonomous ML pipeline that takes a raw data file and produces a trained model, evaluation report, and a deployed FastAPI inference endpoint.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Environment Configuration](#4-environment-configuration)
5. [Starting the MCP Servers](#5-starting-the-mcp-servers)
6. [Running the Pipeline](#6-running-the-pipeline)
7. [Running Tests](#7-running-tests)
8. [End-to-End (E2E) Test Runner](#8-end-to-end-e2e-test-runner)
9. [Docker Deployment](#9-docker-deployment)
10. [Outputs & Artefacts](#10-outputs--artefacts)
11. [Troubleshooting](#11-troubleshooting)
12. [Code Architecture Reference](#12-code-architecture-reference)

---

## 1. Architecture Overview

```
data file (CSV/Excel/image/model)
      │
      ▼
  main.py (CLI entry point)
      │
      ▼
PipelineOrchestrator
  ├── File type detection  (Magika → filetype → extension)
  ├── Pipeline variant selection (tabular / document_text / image / existing_model)
  ├── Prompt Generator LLM agent (per-stage prompt generation)
  └── Stage loop (10 stages max)
        │
        ▼
  run_ralph_loop()  ← self-correcting "verify before done" loop
        │
        ├── Agent.run(prompt)        ← LLM stage agent
        ├── Check for <DONE>tag</DONE>
        ├── verify_stage criteria    ← deterministic checks
        │     ├── PASS → checkpoint → advance
        │     └── FAIL → inject failure feedback → retry (max 8 iterations)
        │
        └── Debug Agent (5-attempt repair on exhaustion)

MCP Servers (run as separate processes):
  ├── Tracking MCP  :8100  — audit trail, checkpoints, artefact lineage (SQLite WAL)
  ├── DS Tools MCP  :8101  — execute_code, read_file, web_research, verify_stage, ...
  └── Bug Log MCP   :8102  — record_bug, list_bugs, export_report, ... (SQLite WAL)
```

### Pipeline Variants

| Input File Type | Variant | Stages Run |
|---|---|---|
| CSV, Excel, Parquet, JSON | `tabular` | ingestion → eda → cleaning → feature_engineering → training → tuning → evaluation → explainability → report → deployment (10 stages) |
| PDF, DOCX, TXT, HTML | `document_text` | same 10 stages |
| PNG, JPG, BMP | `image` | ingestion → feature_engineering → training → tuning → evaluation → explainability → report → deployment (8 stages) |
| ONNX, SafeTensors, .pkl, .joblib | `existing_model` | ingestion → evaluation → explainability → report → deployment (5 stages) |

### Ralph Loop (Self-Correcting Control)

Every stage is wrapped in the Ralph Loop:
```
while iteration < max_iterations (default 8):
    response = agent.run(prompt)
    if "<DONE>stage_name</DONE>" in response:
        check deterministic criteria
        if all pass → advance to next stage
        else → inject failure details → retry
    else → remind agent to include DONE tag → retry
if exhausted → call Debug Agent (5-attempt repair)
```

---

## 2. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.12 | Required (type hints, asyncio improvements) |
| pip | Latest | Package manager |
| Azure OpenAI | GPT-4o deployment | Required for LLM agents |
| Git | Any | To clone the repository |
| Docker (optional) | Latest | For MCP server containerisation |

> **Azure OpenAI Required:** The pipeline uses Azure AI Foundry / Azure OpenAI GPT-4o for all LLM reasoning stages. You must have an active Azure subscription with a deployed GPT-4o endpoint.

---

## 3. Installation

### 3.1 Clone and navigate

```bash
git clone https://github.com/your-org/maf-ds-agent.git
cd maf_ds_agent
```

### 3.2 Create a virtual environment (recommended)

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3.3 Install dependencies

```bash
# Editable install (development mode — recommended)
pip install -e .
```

> **Note:** The project has 60+ packages including PyTorch (`torch==2.5.1`), SHAP, Optuna, XGBoost, CatBoost, FastAPI, and FastMCP. Installation may take 5–10 minutes on first run.

### 3.4 Verify installation

```bash
python -c "import agent_framework; import fastmcp; import pandas; print('All core imports OK')"
```

---

## 4. Environment Configuration

### 4.1 Copy the example env file

```bash
cp .env.example .env
```

### 4.2 Edit `.env` with your values

The `.env` file in this project is already partially configured. Below is the full reference:

```ini
# ── Azure AI Foundry / Azure OpenAI (REQUIRED) ──────────────────────
AI_FOUNDRY_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com
AI_FOUNDRY_API_VERSION=2024-12-01-preview
AI_FOUNDRY_API_KEY=your-api-key-here

# Deployment names (must match what you created in Azure Portal)
AZURE_OPENAI_PRIMARY_DEPLOYMENT=gpt-4o     # Used by: cleaning, evaluation, explainability, report, debug
AZURE_OPENAI_FAST_DEPLOYMENT=gpt-4o        # Used by: ingestion, eda, tuning, deployment
AI_FOUNDRY_DEPLOYMENT_NAME=gpt-4o          # Legacy fallback name

# ── MCP Server Ports (optional — shown defaults) ─────────────────────
TRACKING_MCP_PORT=8100
DS_TOOLS_MCP_PORT=8101
BUG_LOG_MCP_PORT=8102

TRACKING_MCP_URL=http://localhost:8100/mcp/mcp
DS_TOOLS_MCP_URL=http://localhost:8101/mcp/mcp
BUG_LOG_MCP_URL=http://localhost:8102/mcp/mcp

# ── Pipeline Behaviour ───────────────────────────────────────────────
PIPELINE_HUMAN_IN_THE_LOOP=false     # Set to true to pause at each stage gate
PIPELINE_MAX_RALPH_ITERATIONS=8      # Max retry attempts per stage
PIPELINE_TOKEN_BUDGET=500000         # Total token budget across all stages

# ── Data Paths ───────────────────────────────────────────────────────
TRACKING_DB_PATH=data/tracking.db
BUGLOG_DB_PATH=data/buglog.db
MEMORY_DB_PATH=data/memory.db
OUTPUT_BASE_DIR=outputs

# ── OpenTelemetry (optional) ─────────────────────────────────────────
# OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
# OTEL_SERVICE_NAME=ds-agent
```

> **Important:** The `.env` file must NOT be committed to version control. It contains your API keys.

### 4.3 Verify credentials

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
endpoint = os.environ.get('AI_FOUNDRY_PROJECT_ENDPOINT', 'NOT SET')
key = os.environ.get('AI_FOUNDRY_API_KEY', 'NOT SET')
print(f'Endpoint: {endpoint}')
print(f'Key set: {bool(key and key != \"your-api-key-here\")}')
"
```

---

## 5. Starting the MCP Servers

The MAF DS Agent requires **three FastMCP servers** running as background processes before the pipeline can execute.

### 5.1 Start all three servers (recommended)

Open a **dedicated terminal window** and run:

```bash
python scripts/start_servers.py
```

**What this does:**
- Launches `mcp_servers.tracking.server` on port **8100** (SQLite-backed audit trail)
- Launches `mcp_servers.ds_tools.server` on port **8101** (stateless data science tools)
- Launches `mcp_servers.bug_log.server` on port **8102** (SQLite-backed error tracking)
- Performs exponential back-off health checks on all three (30 second timeout)
- Exits non-zero (pipeline gate) if any server fails to start
- Forwards Ctrl+C to all child processes for clean shutdown

**Expected output:**
```
Starting MCP servers: tracking, ds_tools, bug_log
  → tracking on port 8100
  → ds_tools on port 8101
  → bug_log on port 8102
  ✓ tracking healthy on port 8100
  ✓ ds_tools healthy on port 8101
  ✓ bug_log healthy on port 8102

All MCP servers are healthy. Press Ctrl+C to stop.
```

### 5.2 Start individual servers

```bash
python scripts/start_servers.py tracking           # Port 8100 only
python scripts/start_servers.py ds_tools bug_log   # Two servers
```

### 5.3 Health-check only (without starting)

```bash
python scripts/start_servers.py --check
```

### 5.4 Manual server URLs

Once running, the servers are accessible at:

| Server | Health URL | MCP URL |
|---|---|---|
| Tracking MCP | `http://localhost:8100/health` | `http://localhost:8100/mcp/mcp` |
| DS Tools MCP | `http://localhost:8101/health` | `http://localhost:8101/mcp/mcp` |
| Bug Log MCP  | `http://localhost:8102/health` | `http://localhost:8102/mcp/mcp` |

Verify with curl (PowerShell):
```powershell
Invoke-WebRequest -Uri http://localhost:8100/health | Select-Object -ExpandProperty Content
# → {"status":"healthy","server":"tracking","port":8100}
```

Or with Python:
```bash
python -c "import httpx; print(httpx.get('http://localhost:8100/health').json())"
```

---

## 6. Running the Pipeline

### 6.1 Basic usage

```bash
python main.py --file data/titanic.csv --task "Predict passenger survival"
```

### 6.2 Full argument reference

```
python main.py
  --file   /path/to/your/input.csv     REQUIRED: input data or model file
  --task   "Natural language ML task"  REQUIRED: what you want to predict/do
  --run-id my-experiment-001           OPTIONAL: explicit run ID (auto-UUID if omitted)
  --output ./results/output.json       OPTIONAL: write JSON result to file (default: stdout)
```

### 6.3 Example runs with included datasets

The `data/` directory contains three real datasets ready to use:

#### Titanic — Binary Classification
```bash
python main.py \
  --file data/titanic.csv \
  --task "Build a binary classifier to predict passenger survival on the Titanic dataset. The target column is 'Survived' (1=survived, 0=did not survive). Use features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked. Handle missing Age values with median imputation." \
  --output outputs/titanic_result.json
```

#### Wine Quality — Regression
```bash
python main.py \
  --file data/winequality_red.csv \
  --task "Build a regression model to predict red wine quality scores from physicochemical properties. The target column is 'quality'. The dataset uses semicolon as separator. Train a Gradient Boosting Regressor." \
  --output outputs/wine_result.json
```

#### Student Scores — Simple Regression
```bash
python main.py \
  --file data/student_scores.csv \
  --task "Build a simple linear regression model to predict student exam scores based on study hours. Target column is 'Scores', feature is 'Hours'." \
  --output outputs/student_result.json
```

### 6.4 Human-in-the-loop mode

```bash
# Windows PowerShell
$env:PIPELINE_HUMAN_IN_THE_LOOP="true"; python main.py --file data/titanic.csv --task "Predict survival"

# macOS / Linux
PIPELINE_HUMAN_IN_THE_LOOP=true python main.py --file data/titanic.csv --task "Predict survival"
```

After each stage completes, you will be prompted:
```
[HUMAN GATE] Stage 'eda' completed.
  Deployment recommendation: N/A
  Baseline metrics: {}
Continue pipeline? [Y/n]:
```

Press Enter (or Y) to advance, type `n` to abort the pipeline.

### 6.5 Pipeline output

A successful run produces:

```
=== PIPELINE COMPLETE ===
{
  "run_id": "abc-123",
  "pipeline_variant": "tabular",
  "model_type": "RandomForestClassifier",
  "deployment_recommendation": "deploy",
  "report_html_path": "outputs/abc-123/report.html",
  "endpoint_url": "http://localhost:8500/predict",
  "baseline_metrics": {
    "accuracy": 0.83,
    "roc_auc": 0.87,
    "f1": 0.81
  }
}

=== PIPELINE SUMMARY ===
  Run ID              : abc-123
  Pipeline Variant    : tabular
  Model Type          : RandomForestClassifier
  Deployment Status   : deploy
  Report              : outputs/abc-123/report.html
  Endpoint URL        : http://localhost:8500/predict
  Metrics:
    accuracy: 0.83
    roc_auc: 0.87
```

---

## 7. Running Tests

### 7.1 Unit Tests (no network, no Azure credentials required)

The unit tests only check that all modules can be imported and key objects are constructable. They inject dummy environment variables automatically.

```bash
python -m pytest tests/test_imports.py -v
```

**Expected result: 19/19 passing**

What each test validates:

| Test | What It Checks |
|---|---|
| `test_clients_import` | `PRIMARY_CLIENT` and `FAST_CLIENT` are non-None lazy proxies |
| `test_middleware_import` | `pipeline_stack()`, `observer_stack()`, `debug_stack()` return lists |
| `test_ralph_loop_import` | `run_ralph_loop` is an async coroutine function |
| `test_criteria_import` | `get_criteria`, `StageCriteria`, `IngestionCriteria`, `TrainingCriteria` import correctly |
| `test_pipeline_graph` | `PIPELINE_GRAPH` has all 4 variants; `tabular` starts with `ingestion`, ends with `deployment`; `image` skips `eda` |
| `test_local_tools_import` | `LOCAL_TOOLS` has exactly 4 tools including `get_session_state`, `set_session_state`, `check_artefact_exists` |
| `test_base_agent_builders_import` | All 6 MCP/agent factory functions are callable |
| `test_all_stage_agents_import` | All 10 stage builder functions are callable |
| `test_stage_agent_names` | ingestion, eda, training, deployment agents have correct `.name` attribute |
| `test_support_agents_import` | `debug_agent`, `bug_log_observer`, `artefact_tracking_observer` import and construct |
| `test_orchestrator_builds` | `build_orchestrator()` returns a `PipelineOrchestrator` with all 10 stage agents |
| `test_orchestrator_has_debug_and_observers` | Orchestrator has debug agent, observers, and all 3 MCP tools |
| `test_mcp_servers_import` | All 3 server modules import and expose `mcp` and `app` objects |
| `test_file_type_detector_import` | `detect_file_type` and `FileTypeResult` import correctly |
| `test_detect_csv_file` | A real CSV file is detected as `tabular` or `text` with confidence > 0 |
| `test_detect_nonexistent_file` | Non-existent `.csv` path falls back to extension-based detection |
| `test_settings_import` | `get_settings()` returns a valid `Settings` instance |
| `test_library_registry_import` | `LIBRARY_REGISTRY` has > 5 entries including `pandas` and `scikit-learn` |
| `test_main_importable` | `main.py` imports and exposes a callable `main` function |

```bash
# Run with coverage report
python -m pytest tests/test_imports.py \
  --cov=agents --cov=workflows --cov=tools --cov=config \
  --cov-report=term-missing \
  -v
```

### 7.2 Integration Tests (MCP servers must be running)

Integration tests require all three MCP servers to be healthy. They auto-skip gracefully if servers are offline.

```bash
# Terminal 1 — start servers (keep running)
python scripts/start_servers.py

# Terminal 2 — run integration tests
python -m pytest tests/test_integration.py -v
```

**Test classes and what they cover:**

| Class | Tests |
|---|---|
| `TestHealthEndpoints` | All 3 `/health` endpoints return `{"status":"healthy"}` |
| `TestMCPToolsList` | Each server exposes >= expected number of tools via MCP `tools/list` |
| `TestTrackingRoundTrip` | Full `record_start → record_checkpoint → record_end → query_run` lifecycle |
| `TestBugLogRoundTrip` | `record_bug → list_bugs` round-trip |
| `TestDSToolsVerify` | `verify_stage` and `log_metrics` tools respond without error |
| `TestCrossServerCoordination` | All 3 servers share a single `run_id` across a simulated pipeline run |

To skip integration tests explicitly:
```bash
python -m pytest tests/ -m "not integration" -v
```

### 7.3 Run all tests together

```bash
# Unit tests always run; integration tests skip if servers offline
python -m pytest tests/ -v
```

### 7.4 Lint check

```bash
python -m ruff check .
python -m ruff format --check .
```

---

## 8. End-to-End (E2E) Test Runner

The E2E test runner in `scripts/run_e2e_test.py` runs the **complete full pipeline** against real datasets with real Azure OpenAI calls.

> **WARNING: This uses real Azure OpenAI tokens and takes 15–60 minutes per dataset.**

### 8.1 Prerequisites

- MCP servers running: `python scripts/start_servers.py`
- `.env` configured with valid Azure OpenAI credentials
- Dataset files present in `data/`

### 8.2 Run against Titanic (default)

```bash
python scripts/run_e2e_test.py
```

### 8.3 Run against specific datasets

```bash
# Single dataset
python scripts/run_e2e_test.py --datasets titanic

# Multiple datasets
python scripts/run_e2e_test.py --datasets titanic wine student

# With verbose JSON output
python scripts/run_e2e_test.py --datasets titanic --verbose
```

**Available datasets:**

| Name | File | Task |
|---|---|---|
| `titanic` | `data/titanic.csv` | Binary classification — passenger survival |
| `wine` | `data/winequality_red.csv` | Regression — wine quality score prediction |
| `student` | `data/student_scores.csv` | Simple linear regression — score from study hours |

### 8.4 E2E output

Results are saved per-dataset:
```
outputs/
  titanic/
    titanic-20260607_120000/
      titanic-20260607_120000_result.json   <- full session state JSON
  e2e_summary_20260607_120000.json          <- cross-dataset summary
```

Summary table printed at the end:
```
======================================================================
  SUMMARY
======================================================================
  DATASET         STATUS       DURATION  STAGES OK    STAGES FAIL
  ------------------------------------------------------------
  titanic         success        743.2s  10           0
  wine            success        812.5s  10           0
  student         success        421.0s  10           0
======================================================================
```

---

## 9. Docker Deployment

Docker containers are provided for the three MCP servers (the pipeline itself runs on the host).

### 9.1 Start MCP servers via Docker

```bash
# Build and start all three servers
docker compose up --build -d

# Check status
docker compose ps
docker compose logs -f

# Stop all servers
docker compose down
```

### 9.2 Override ports

```bash
TRACKING_MCP_PORT=9100 DS_TOOLS_MCP_PORT=9101 BUG_LOG_MCP_PORT=9102 docker compose up -d
```

Update your `.env` accordingly:
```ini
TRACKING_MCP_PORT=9100
TRACKING_MCP_URL=http://localhost:9100/mcp/mcp
```

### 9.3 Individual Dockerfiles

| File | Purpose |
|---|---|
| `Dockerfile.tracking` | Tracking MCP server (port 8100) |
| `Dockerfile.ds_tools` | DS Tools MCP server (port 8101) |
| `Dockerfile.bug_log` | Bug Log MCP server (port 8102) |

---

## 10. Outputs & Artefacts

All pipeline artefacts are saved under `data/artefacts/{run_id}/` and results under `outputs/`.

### Typical output structure after a tabular pipeline run

```
data/artefacts/{run_id}/
  ingestion_sample.json        # File schema and 5-row sample
  eda_narrative.md             # EDA narrative summary
  eda_charts/                  # Distribution plots, correlation heatmap
  cleaned_dataset.parquet      # Cleaned dataset
  transformation_log.json      # Cleaning transformations applied
  features_train.parquet       # Train split with engineered features
  features_test.parquet        # Test split with engineered features
  feature_manifest.json        # Feature names, types, importances
  model.pkl                    # Trained model (joblib)
  model.onnx                   # ONNX export (if supported by framework)
  best_params.json             # Optuna best hyperparameters
  evaluation_report.md         # Metrics, confusion matrix, fairness check
  shap_summary.png             # SHAP feature importance plot
  shap_values.npy              # Raw SHAP values array
  explanation_narrative.md     # Plain-English SHAP explanation
  report.md                    # Full Markdown pipeline report
  report.html                  # HTML report with embedded charts

data/
  tracking.db                  # SQLite: pipeline run audit trail
  buglog.db                    # SQLite: error and repair log

logs/
  pipeline.log                 # Full pipeline execution log
```

### Querying run history

```bash
# Using SQLite CLI
sqlite3 data/tracking.db "SELECT run_id, status, started_at FROM pipeline_runs ORDER BY started_at DESC LIMIT 10;"

# Query bug log
sqlite3 data/buglog.db "SELECT stage, error_type, error_message FROM bugs ORDER BY created_at DESC LIMIT 5;"
```

---

## 11. Troubleshooting

### MCP servers not starting

```powershell
# Check if ports are already in use (Windows)
netstat -ano | Select-String "8100|8101|8102"

# Kill a process on a port (Windows)
Stop-Process -Id <PID>

# Run health check
python scripts/start_servers.py --check
```

```bash
# macOS/Linux
lsof -i :8100 -i :8101 -i :8102
```

### Azure OpenAI authentication errors

```
openai.AuthenticationError: ...
```

1. Check your `.env` has valid `AI_FOUNDRY_API_KEY` and `AI_FOUNDRY_PROJECT_ENDPOINT`
2. Ensure the endpoint format matches your Azure resource type:
   - Azure OpenAI: `https://your-resource.openai.azure.com`
   - Azure AI Foundry: `https://your-project.services.ai.azure.com`
3. Verify the deployment name matches exactly what you set in Azure Portal

### Import errors for `agent_framework`

```
ModuleNotFoundError: No module named 'agent_framework'
```

Install the Microsoft Agent Framework package:
```bash
pip install agent-framework-openai==1.0.0
# OR reinstall the whole project
pip install -e .
```

### Pipeline stages failing repeatedly

If a stage hits max_iterations (8) and triggers the Debug Agent:

1. Check `logs/pipeline.log` for detailed failure information
2. Query the bug log:
   ```bash
   sqlite3 data/buglog.db "SELECT stage, error_type, error_message FROM bugs ORDER BY created_at DESC LIMIT 5;"
   ```
3. Increase `PIPELINE_MAX_RALPH_ITERATIONS` in `.env`:
   ```ini
   PIPELINE_MAX_RALPH_ITERATIONS=12
   ```

### `PIPELINE_HUMAN_IN_THE_LOOP=true` but no gate appears

Check that `PIPELINE_HUMAN_IN_THE_LOOP` in `.env` is literally `true` (lowercase, not `True` or `1`):
```ini
PIPELINE_HUMAN_IN_THE_LOOP=true
```

### Unsafe code blocked by SafetyMiddleware

```
ValueError: SafetyMiddleware: blocked unsafe pattern 'os.system' in generated code.
```

The Safety middleware blocks known dangerous patterns in LLM-generated code (`os.system`, `eval`, `exec`, `subprocess.Popen(shell=True)`, `shutil.rmtree`, SQL DROP/DELETE). These are intentional safety guards. If a legitimate use case is blocked, adjust `_UNSAFE_PATTERNS` in `agents/middleware.py`.

### SQLite database locked

```
sqlite3.OperationalError: database is locked
```

The MCP servers use WAL (Write-Ahead Logging) mode for concurrent reads. If you see this error after an unclean shutdown:
```bash
del data\tracking.db-wal data\tracking.db-shm   # Windows
rm data/tracking.db-wal data/tracking.db-shm     # macOS/Linux
```

---

## 12. Code Architecture Reference

### Key files at a glance

| File | Purpose |
|---|---|
| `main.py` | CLI entry point — argparse → `asyncio.run(_run(...))` |
| `agents/orchestrator.py` | Python driver that orchestrates all 10 stages |
| `agents/clients.py` | Lazy Azure OpenAI client proxies (`PRIMARY_CLIENT`, `FAST_CLIENT`) |
| `agents/middleware.py` | 5 middleware: Logging, RateLimit, Safety, Retry, Telemetry |
| `agents/base.py` | MCP tool factories + agent builder factories |
| `workflows/ralph_loop.py` | Ralph Loop — self-correcting stage execution |
| `workflows/criteria.py` | Per-stage deterministic exit-gate criteria classes |
| `workflows/pipeline_graph.py` | 4 pipeline variant stage graphs |
| `tools/file_type_detector.py` | 3-layer detection: Magika → filetype → extension |
| `tools/local_tools.py` | 4 `@tool` functions: get/set session state, check artefact |
| `config/settings.py` | Pydantic `BaseSettings` — all env vars centralised |
| `mcp_servers/tracking/server.py` | Tracking MCP — 11 tools, SQLite WAL |
| `mcp_servers/ds_tools/server.py` | DS Tools MCP — 10 stateless tools |
| `mcp_servers/bug_log/server.py` | Bug Log MCP — 8 tools, SQLite WAL |
| `scripts/start_servers.py` | Start + health-gate all 3 MCP servers |
| `scripts/run_e2e_test.py` | Real-world E2E test runner (3 datasets) |
| `tests/test_imports.py` | 19 unit tests — no network required |
| `tests/test_integration.py` | MCP server integration tests |

### Stage Agents and LLM clients

| Stage | Agent File | LLM Client | Done Tag |
|---|---|---|---|
| Ingestion | `agents/ingestion_agent.py` | FAST | `<DONE>ingestion</DONE>` |
| EDA | `agents/eda_agent.py` | FAST | `<DONE>eda</DONE>` |
| Cleaning | `agents/cleaning_agent.py` | PRIMARY | `<DONE>cleaning</DONE>` |
| Feature Engineering | `agents/feature_agent.py` | PRIMARY | `<DONE>feature_engineering</DONE>` |
| Training | `agents/training_agent.py` | PRIMARY | `<DONE>training</DONE>` |
| Tuning | `agents/tuning_agent.py` | FAST | `<DONE>tuning</DONE>` |
| Evaluation | `agents/evaluation_agent.py` | PRIMARY | `<DONE>evaluation</DONE>` |
| Explainability | `agents/explainability_agent.py` | PRIMARY | `<DONE>explainability</DONE>` |
| Report | `agents/report_agent.py` | PRIMARY | `<DONE>report</DONE>` |
| Deployment | `agents/deployment_agent.py` | FAST | `<DONE>deployment</DONE>` |
| Debug (support) | `agents/debug_agent.py` | PRIMARY | — |
| Bug Observer | `agents/bug_log_observer.py` | PRIMARY | — |
| Artefact Observer | `agents/artefact_tracking_observer.py` | FAST | — |

### MCP Tools Reference

#### Tracking MCP (port 8100) — 11 tools

| Tool | Description |
|---|---|
| `tracking_record_start` | Start a pipeline run audit entry |
| `tracking_record_end` | Mark run as complete/failed |
| `tracking_record_checkpoint` | Record a Ralph Loop iteration checkpoint |
| `tracking_record_artefact` | Record a produced file artefact |
| `tracking_record_metric` | Record a stage performance metric |
| `tracking_record_lineage` | Record data lineage between stages |
| `tracking_query_run` | Get full run details |
| `tracking_query_artefacts` | List all artefacts for a run |
| `tracking_query_metrics` | Get metrics for a run (optionally by stage) |
| `tracking_query_lineage` | Get full lineage graph for a run |
| `tracking_list_runs` | List recent pipeline runs |

#### DS Tools MCP (port 8101) — 10 tools

| Tool | Description |
|---|---|
| `ds_read_file` | Read file content (CSV, Parquet, JSON, text) |
| `ds_execute_code` | Execute Python code in a sandboxed subprocess |
| `ds_get_sample` | Return N rows of a dataset as JSON |
| `ds_write_output` | Write content to the artefacts directory |
| `ds_search_docs` | Search documentation (library registry + DuckDuckGo) |
| `ds_web_research` | General web research via DuckDuckGo |
| `ds_embed_text` | Embed text via Azure OpenAI embedding model |
| `ds_semantic_search` | Search embedded memory for relevant context |
| `ds_log_metrics` | Log stage metrics to the tracking system |
| `ds_verify_stage` | Run deterministic stage criteria checks |

#### Bug Log MCP (port 8102) — 8 tools

| Tool | Description |
|---|---|
| `buglog_record_bug` | Record an error event |
| `buglog_update_bug` | Update an existing bug entry |
| `buglog_list_bugs` | List bugs for a run (optionally by stage) |
| `buglog_get_bug` | Get full details of a specific bug |
| `buglog_search_bugs` | Full-text search across bug messages |
| `buglog_get_stats` | Aggregate bug statistics for a run |
| `buglog_export_report` | Export bugs as Markdown report |
| `buglog_clear_run` | Clear all bugs for a specific run |

### Middleware Stack

| Stack | Applied To | Layers |
|---|---|---|
| `pipeline_stack()` | All 10 stage agents + Prompt Generator | Logging → RateLimit → Safety → Retry → Telemetry |
| `observer_stack()` | Bug Log Observer, Artefact Tracking Observer | Logging → Retry → Telemetry |
| `debug_stack()` | Debug Agent | Logging → Retry → Telemetry (no Safety — must inspect bad code) |

---

## Quick Reference Cheatsheet

```bash
# ── Setup ────────────────────────────────────────────────────────────
pip install -e .                            # Install all dependencies
cp .env.example .env                        # Configure credentials

# ── Start MCP servers ─────────────────────────────────────────────────
python scripts/start_servers.py             # All three servers (keep this terminal open)
python scripts/start_servers.py --check     # Health check only

# ── Run pipeline ──────────────────────────────────────────────────────
python main.py --file data/titanic.csv --task "Predict survival"
python main.py --file data.csv --task "..." --run-id exp-001 --output out.json

# ── Tests ────────────────────────────────────────────────────────────
python -m pytest tests/test_imports.py -v              # Unit tests (19 tests, no network)
python -m pytest tests/test_integration.py -v          # Integration (servers must run)
python -m pytest tests/ -v                             # All tests
python -m pytest tests/ --cov=agents --cov=workflows   # With coverage

# ── E2E test ─────────────────────────────────────────────────────────
python scripts/run_e2e_test.py                          # Titanic (default)
python scripts/run_e2e_test.py --datasets titanic wine student  # All 3 datasets

# ── Docker ───────────────────────────────────────────────────────────
docker compose up --build -d               # Start MCP servers in Docker
docker compose ps                          # Check status
docker compose down                        # Stop all

# ── Lint ─────────────────────────────────────────────────────────────
python -m ruff check .
python -m ruff format --check .
```
