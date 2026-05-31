# DS_AGENT.md — Data Science Agent System

> **Stack:** Microsoft Agent Framework 1.0 (Python) · Azure OpenAI (exclusive) · MCP Streamable HTTP · FastMCP · Browser-Use · DuckDuckGo
> **Loop Pattern:** Ralph Loop (self-correcting, verify-before-done, fresh-context-on-retry)
> **Memory:** Episodic + Long-term Semantic + Procedural (three-tier)
> **Observability:** Artefact Tracking Agent + Bug Log Agent (dedicated observers)
> **Philosophy:** Trap the system in a loop that never declares done until tests pass and outputs are verified. Never trust "I finished" — trust passing criteria only.

---

## How to Use This Document

This is the complete implementation specification for the DS Agent project. Read every section before writing code. Every section governs a specific part of the system. Nothing is optional. If a section conflicts with code you have already written, fix the code — this document is the source of truth.

Section numbers are stable. Reference them in commit messages, PR descriptions, and decision log entries. "Per §8.3 — RetryMiddleware" is a complete explanation in context.

---

## Table of Contents

1. [Architecture and Mental Model](#1-architecture-and-mental-model)
2. [Ralph Loop — The Core Execution Pattern](#2-ralph-loop--the-core-execution-pattern)
3. [Package Installation and Import Map](#3-package-installation-and-import-map)
4. [Azure OpenAI Client Construction](#4-azure-openai-client-construction)
5. [MCP Streamable HTTP — Server Design](#5-mcp-streamable-http--server-design)
6. [MCP Streamable HTTP — Client via MCPStreamableHTTPTool](#6-mcp-streamable-http--client-via-mcpstreamablehttptool)
7. [Session Management and State Schema](#7-session-management-and-state-schema)
8. [Three-Tier Memory System](#8-three-tier-memory-system)
9. [AgentMiddleware — Pipeline and Ordering](#9-agentmiddleware--pipeline-and-ordering)
10. [Tool Definition — @tool and FunctionInvocationContext](#10-tool-definition--tool-and-functioninvocationcontext)
11. [Sub-Agent Pattern — agent.as_tool()](#11-sub-agent-pattern--agentastool)
12. [Artefact Tracking Agent](#12-artefact-tracking-agent)
13. [Bug Log Agent](#13-bug-log-agent)
14. [read_file Tool — File Type Detection Pipeline](#14-read_file-tool--file-type-detection-pipeline)
15. [Agent Catalogue and System Prompt Contracts](#15-agent-catalogue-and-system-prompt-contracts)
16. [Workflow Graph and Orchestration](#16-workflow-graph-and-orchestration)
17. [Ralph Loop Implementation — Self-Correcting Execution](#17-ralph-loop-implementation--self-correcting-execution)
18. [Browser-Use Integration](#18-browser-use-integration)
19. [DuckDuckGo Deep Research](#19-duckduckgo-deep-research)
20. [Deep Learning and Generative AI Coverage](#20-deep-learning-and-generative-ai-coverage)
21. [Inter-Agent Contract Tables](#21-inter-agent-contract-tables)
22. [Phase 1 — Environment and MCP Servers](#22-phase-1--environment-and-mcp-servers)
23. [Phase 2 — read_file, Ingestion, Tracking and Bug Log Agents](#23-phase-2--read_file-ingestion-tracking-and-bug-log-agents)
24. [Phase 3 — EDA, Cleaning, Feature Engineering with Ralph Loop](#24-phase-3--eda-cleaning-feature-engineering-with-ralph-loop)
25. [Phase 4 — Training, Tuning, Evaluation](#25-phase-4--training-tuning-evaluation)
26. [Phase 5 — Explainability, Reporting, Deployment](#26-phase-5--explainability-reporting-deployment)
27. [Phase 6 — Memory System and Retrospection](#27-phase-6--memory-system-and-retrospection)
28. [Phase 7 — Hardening and Production Gate](#28-phase-7--hardening-and-production-gate)
29. [Project Directory Structure](#29-project-directory-structure)
30. [Configuration Reference](#30-configuration-reference)
31. [Safety Patterns and Responsible AI](#31-safety-patterns-and-responsible-ai)
32. [Testing Strategy](#32-testing-strategy)
33. [CI/CD and Operations](#33-cicd-and-operations)
34. [Error Catalogue and Auto-Remediation Map](#34-error-catalogue-and-auto-remediation-map)
35. [Known Limitations and Roadmap](#35-known-limitations-and-roadmap)
36. [Glossary](#36-glossary)
37. [Reference Documents](#37-reference-documents)

---

## 1. Architecture and Mental Model

### 1.1 The Core Idea

Every traditional pipeline fails silently or partially. An agent reports "done" while outputs are wrong, metrics are missing, and artefacts are corrupt. The DS Agent system refuses this pattern. It applies the Ralph Loop at every stage: every agent runs, produces outputs, and is immediately verified against hard acceptance criteria. If criteria are not met, the loop restarts the stage with a fresh context window, the error recorded in memory, and the fix strategy updated. The loop does not exit until the criteria pass.

Alongside the pipeline, two dedicated observer agents run continuously: the Artefact Tracking Agent (records every file produced, every metric logged, every model saved) and the Bug Log Agent (records every error, every repair attempt, every loop iteration, and the final resolution). These are not middleware — they are full agents with their own sessions, their own memory, and their own MCP connections.

### 1.2 Three-Layer Stack

**Layer 1 — Intelligence (Agents)**
Fifteen named agents. Each wraps an Azure OpenAI completion, a typed system prompt, MCP tool references, context providers (including memory), and middleware. Agents reason and delegate. They never import data science libraries. All computation is delegated through MCP tools.

**Layer 2 — Protocol (MCP Streamable HTTP)**
Two FastMCP servers: the Tracking MCP Server (port 8100, the coordination backbone and audit log) and the DS Tools MCP Server (port 8101, the action surface for all data science operations). A dedicated Bug Log MCP Server (port 8102) handles all error and repair records and is never co-located with the Tracking Server (separate crash domain). All three use MCP protocol version 2025-06-18 with Streamable HTTP transport.

**Layer 3 — Execution (Subprocess with Ralph Loop)**
Generated Python code runs in a managed subprocess. The Ralph Loop wraps every execution: run → verify against acceptance criteria → if pass exit loop → if fail capture error, update memory, fresh-context retry. The loop never trusts the agent's self-report of completion. It trusts only the output of the verification command.

### 1.3 Canonical Data Flow

```
User: --file path/to/file --task "description"
  ↓
[Ralph Loop Wrapper begins]
  ↓
Orchestrator Agent creates run session
  ↓
read_file → file type detection → typed parse result → routing decision
  ↓
[Artefact Tracking Agent observing asynchronously]
[Bug Log Agent observing asynchronously]
  ↓
Ingestion Agent    → validated schema + artefact record
  ↓ [Ralph Loop verifies: schema exists, artefact recorded]
EDA Agent          → stats.json + charts + narrative
  ↓ [Ralph Loop verifies: all required output files exist and are non-empty]
Cleaning Agent     → cleaned_dataset.parquet + transformation_log.json
  ↓ [Ralph Loop verifies: null count = 0 in imputed columns, row count delta documented]
Feature Eng Agent  → features_train.parquet + feature_manifest.json
  ↓ [Ralph Loop verifies: manifest covers 100% of columns, no nulls in features]
Training Agent     → model artefact + baseline_metrics
  ↓ [Ralph Loop verifies: model file exists, metrics.json has primary metric key]
Tuning Agent       → tuned model + best_params.json
  ↓ [Ralph Loop verifies: tuned metric ≥ baseline metric]
Evaluation Agent   → evaluation_report.json + deployment_recommendation
  ↓ [Ralph Loop verifies: fairness_metrics section present, all metric keys present]
Explainability     → shap_values + narrative.md
  ↓ [Ralph Loop verifies: ≥2 SHAP plots, narrative.md non-empty]
Report Agent       → final_report.html + final_report.md
  ↓ [Ralph Loop verifies: HTML is self-contained, all sections present]
Deployment Agent   → endpoint URL + smoke_test_results
  ↓ [Ralph Loop verifies: 10/10 smoke test predictions succeed]
[Ralph Loop exits — completion criteria met]
  ↓
Memory Agent writes episode record to long-term episodic store
```

Every arrow is a checkpoint. Every verification failure triggers a Ralph Loop retry.

### 1.4 The Two Observer Agents

The Artefact Tracking Agent and the Bug Log Agent are always running. They are not in the linear pipeline — they observe it.

The Artefact Tracking Agent subscribes to artefact events from the Tracking MCP Server. Every time any agent produces a file, the Tracking Agent records: what was produced, by which agent, in which stage, at what time, with what content hash, and which artefacts it depends on (lineage).

The Bug Log Agent subscribes to error events from the Bug Log MCP Server. Every time any execution fails: the error class, message, traceback, the code that failed, the stage, the attempt number, the repair strategy applied, and whether the repair succeeded. When a Ralph Loop retries, the Bug Log Agent appends the new attempt to the same bug record, creating a complete failure-to-resolution narrative. When the loop succeeds, the Bug Log Agent marks the bug as resolved and records the fix.

---

## 2. Ralph Loop — The Core Execution Pattern

### 2.1 What Ralph Loop Is

The Ralph Loop is an automation pattern where an AI agent is trapped inside a loop that never ends until it actually passes — the agent detects failures, makes fixes, and keeps going. Named after Ralph Wiggum from The Simpsons: not particularly smart, but never gives up.

The Ralph Wiggum Loop is an iterative pattern: run an AI agent in an iterative loop that repeatedly attempts a task, executes or checks the attempt against a concrete criterion, and feeds the resulting feedback back into the next attempt.

In this system, the Ralph Loop is implemented at the stage level — every pipeline stage is wrapped in a Ralph Loop. The loop has four components:

**1. The Agent Run** — the stage agent executes with its current context and produces outputs.

**2. The Verification Command** — a deterministic Python function (not an LLM call) that checks the stage's acceptance criteria. The verification is objective: does `cleaned_dataset.parquet` exist? Does `null_count == 0` for imputed columns? Does the primary metric key exist in `metrics.json`? Pass or fail — no LLM judgement.

**3. The Feedback Injection** — if verification fails, the error output of the verification command (the exact assertion failure message, the exact missing file path, the exact wrong metric value) is injected into the next iteration's context. The agent sees precisely what went wrong, not a vague "please try again."

**4. The Fresh Context** — each retry runs with a fresh context window. No compaction, no degradation. State survives between iterations through files on disk, the bug log, and session state — not through conversation history.

### 2.2 Completion Criteria — Never Trust Self-Report

The core idea: even when AI says "done," automatically restart it and make it verify itself. Traditional AI coding tools stop the moment the agent decides the task is done, without verifying whether that decision is actually correct.

Every stage has a `DONE_CRITERIA` dict defined in `workflows/criteria.py`. The dict specifies:

`required_files` — list of file paths that must exist and be non-empty after the stage completes.

`required_session_keys` — list of session state keys that must be present and non-null.

`assertions` — list of Python callable assertions that take the session state and the output directory and return `(passed: bool, message: str)`. These check domain-specific correctness: no nulls in imputed columns, tuned metric exceeds baseline, SHAP plot count ≥ 2, HTML has all required section headings.

`completion_promise` — a string tag the agent must output verbatim in its final message to signal it believes it is done. The loop checks for this tag, then runs the verification criteria. If criteria fail despite the promise, the loop continues. The completion promise uses exact string matching. The loop checks for `<DONE>stage_name</DONE>` in the agent's output. If found, run verification. If verification passes, exit. If verification fails, feed back the failure and continue.

### 2.3 Loop Architecture

The Ralph Loop is implemented in `workflows/ralph_loop.py`. It wraps every stage agent invocation.

The loop operates as follows:

On each iteration:
- Read the stage's DONE_CRITERIA from `workflows/criteria.py`.
- Read the current bug context from the Bug Log Agent (what has failed so far in this stage, what repairs were attempted).
- Construct the stage prompt: base system prompt + current session state summary + bug context from previous iterations (if any) + explicit verification criteria the agent must satisfy.
- Run the stage agent. Capture its output and the completion promise tag.
- If the completion promise tag is present: run the verification function.
  - If all criteria pass: write checkpoint to Tracking MCP Server, advance to next stage, exit loop.
  - If any criterion fails: log the failure to the Bug Log Agent, increment iteration counter, inject the failure details into the next iteration's prompt, loop.
- If the completion promise tag is absent after the agent's turn: log a `missing_completion_promise` event, inject a reminder, loop.
- If the iteration counter reaches `MAX_ITERATIONS` (configurable, default 8): trigger the human-in-the-loop gate with the full bug log for this stage.

### 2.4 Channel.io Two-Loop Pattern

Channel.io's two-loop separation — Stateless and Stateful — leaves a practical lesson: don't force a single pattern when requirements differ.

Apply this to the DS Agent:

**Stateless Loop** — used for individual code execution blocks within a stage (the `execute_code` tool). Each attempt runs fresh code with fresh subprocess. No state between attempts except the error message. Applies to: EDA stat computation, cleaning transforms, feature engineering, training, SHAP computation. This is the inner loop.

**Stateful Loop** — used for entire stage re-runs. The stage agent is re-run with accumulated context from all previous iterations of this stage: what was tried, what failed, what files already exist, what has been partially completed. The Memory Agent's episodic store is queried for relevant past failures. This is the outer loop.

---

## 3. Package Installation and Import Map

### 3.1 Installation Order

Install in this exact order to avoid dependency conflicts.

```
pip install agent-framework-openai
pip install mcp
pip install fastmcp
pip install browser-use
pip install duckduckgo-search
pip install magika
pip install filetype
pip install chardet
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

All packages must be pinned to exact versions in `pyproject.toml` using `==`. No floating ranges. Reproduce builds exactly.

### 3.2 Critical Import Map

Use exactly these import paths. The Framework renamed several modules at 1.0.

```
from agent_framework import ChatAgent
from agent_framework import tool, agent_middleware, AgentMiddleware
from agent_framework import AgentRunContext, FunctionInvocationContext
from agent_framework import InMemoryHistoryProvider
from agent_framework.openai import AzureOpenAIChatCompletionClient
from agent_framework import MCPStreamableHTTPTool
from fastmcp import FastMCP
from fastapi import FastAPI
from azure.identity import DefaultAzureCredential
```

The `AzureOpenAIChatCompletionClient` detects Azure vs consumer OpenAI by checking for `AZURE_OPENAI_ENDPOINT` in the environment. When that variable is set, it routes to Azure endpoints automatically. Never explicitly choose between them in code — control it via the environment variable.

### 3.3 Package Notes

`agent-framework-openai` installs `agent-framework-core` as a dependency. Do not install `agent-framework-core` separately — version conflicts result.

`fastmcp` is for building MCP servers. `MCPStreamableHTTPTool` from `agent_framework` is for connecting to them as a client. Never mix: `fastmcp` is server-side, `MCPStreamableHTTPTool` is client-side.

`magika` downloads a model on first import. Initialise it once at application startup as a module-level singleton in `tools/file_type_detector.py`. Never reinitialise per call.

---

## 4. Azure OpenAI Client Construction

### 4.1 Client Strategy

Create exactly two `AzureOpenAIChatCompletionClient` instances for the entire application. Reuse them across all agents.

**Primary client** — for: Orchestrator, Debug Agent, Cleaning Agent, Model Selection Agent, Evaluation Agent, Explainability Agent, Report Agent, Bug Log Agent. These agents need extended reasoning, produce complex outputs, or handle error analysis.

**Fast client** — for: Ingestion Agent, EDA Agent (stat prompts), Feature Engineering Agent, HP Tuning Agent, Memory Agent (query synthesis), Artefact Tracking Agent. These agents make short, frequent calls.

Both clients are constructed once at application startup in `agents/clients.py` and imported by all agent modules. Never construct a client inside an agent module — that violates the single-instance rule.

### 4.2 Deployment Names vs Model IDs

Azure OpenAI uses deployment names in API calls, not model IDs. Your deployment names are what you set in the Azure portal (e.g., `gpt4o-prod`, not `gpt-4o`). Store deployment names in configuration variables `AZURE_OPENAI_PRIMARY_DEPLOYMENT` and `AZURE_OPENAI_FAST_DEPLOYMENT`. Never hardcode them.

A third deployment is required: `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` for the embedding model. Used exclusively by the `embed_text` tool and the Memory Agent.

### 4.3 Authentication

**Development:** set `AZURE_OPENAI_API_KEY` in `.env`. The client uses it automatically.

**Production:** remove `AZURE_OPENAI_API_KEY` from the environment entirely. Pass `credential=DefaultAzureCredential()` to the `AzureOpenAIChatCompletionClient` constructor. Assign the `Cognitive Services OpenAI User` role on the Azure OpenAI resource to the identity running the application.

Never commit `AZURE_OPENAI_API_KEY`. If it has ever been committed, rotate it immediately before any other action.

### 4.4 Token Budget Tracking

Each client wraps the raw `AzureOpenAIChatCompletionClient` in a `TokenBudgetWrapper` defined in `agents/clients.py`. This wrapper intercepts every completion response, reads `usage.total_tokens`, and accumulates it in a thread-safe counter keyed by `run_id`. When the cumulative total for a `run_id` exceeds `PIPELINE_TOKEN_BUDGET` (configurable, default 500,000), the wrapper raises `TokenBudgetExceededError`. The Orchestrator catches this and triggers the human gate.

---

## 5. MCP Streamable HTTP — Server Design

### 5.1 Three Servers — Crash Domain Isolation

Three separate MCP server processes: Tracking (port 8100), DS Tools (port 8101), Bug Log (port 8102). Three separate crash domains. A bug in a DS tool crashing port 8101 leaves the Tracking and Bug Log servers fully operational. The Orchestrator can still write checkpoints, the Ralph Loop can still record errors, and the pipeline can be debugged and resumed. Co-locating any two of these would eliminate this safety property.

### 5.2 FastMCP Setup Pattern (all three servers)

Every MCP server uses the same structural pattern:

Create a `FastMCP` instance. Define all tools using the `@mcp.tool()` decorator. Call `mcp.streamable_http_app()` to get a Starlette ASGI app. Mount it into a parent FastAPI app at `/mcp`. Pass the FastMCP app's lifespan to the parent FastAPI app's lifespan. Add a `/health` REST endpoint to the parent FastAPI app (not to the MCP subpath).

The FastMCP app's `StreamableHTTPSessionManager` is initialised inside the lifespan context. If this step is skipped, all MCP connections will fail with `BrokenResourceError`. This is the most common setup mistake — verify it in the Phase 1 exit gate.

The full MCP endpoint path after mounting is `/mcp/mcp` (FastAPI mount at `/mcp` + FastMCP's own `/mcp` suffix). The `MCPStreamableHTTPTool` client must point to this exact path. Document this in `config/settings.py` and `.env.example`.

For `stateless_http`: Tracking Server uses `stateless_http=False` (stateful — clients maintain persistent sessions with the audit database). DS Tools Server uses `stateless_http=True` (stateless — each tool call is independent). Bug Log Server uses `stateless_http=False` (stateful — bug records accumulate within a run's session).

### 5.3 MCP Protocol Compliance Verification

Before considering any MCP server complete, verify all three checks:

Check 1 (HEAD method): `curl -X HEAD http://localhost:{port}/mcp/mcp` must return `200 OK`. This is how the Framework's `MCPStreamableHTTPTool` discovers the server during initialisation.

Check 2 (Method Not Allowed): `curl -I http://localhost:{port}/mcp/mcp` must return `405 Method Not Allowed` with `Allow: POST` in the response headers. A `501 Not Implemented` response means the session manager was not initialised in the lifespan context.

Check 3 (MCP initialisation): `curl -X POST http://localhost:{port}/mcp/mcp -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}},"id":1}'` must return a JSON response containing an `Mcp-Session-Id` header. The session ID must be echoed on all subsequent requests.

### 5.4 Tracking MCP Server — Complete Tool List

**Persistence:** SQLite with WAL mode. WAL enables concurrent readers while a write is in progress — critical because multiple stage agents read checkpoint status simultaneously while the Artefact Tracking Agent writes new artefact records.

Tools:

`record_start(run_id, stage_name, session_id, agent_name, input_artefact_paths, started_at)` → `log_entry_id`

`record_end(log_entry_id, run_id, stage_name, status, output_artefact_paths, duration_ms, ended_at)` → nothing

`record_checkpoint(run_id, stage_name)` → nothing. Queried by Orchestrator on resume to skip completed stages.

`record_artefact(run_id, stage_name, artefact_path, artefact_type, content_hash, metadata_json)` → `artefact_id`. Called by the Artefact Tracking Agent after every file production.

`record_metric(run_id, stage_name, metric_name, metric_value, metadata_json)` → `metric_id`. Called after every `log_metrics` tool call.

`record_lineage(parent_artefact_id, child_artefact_id, relationship)` → nothing. `relationship` is one of `derived_from`, `trained_on`, `evaluated_with`, `explained_using`.

`query_run_status(run_id)` → list of stages with their statuses. Used by Orchestrator on pipeline start/resume.

`query_artefact_history(run_id)` → all artefacts produced in a run with their paths and types.

`query_best_artefact(artefact_type, task_type, metric_name, ascending)` → artefact path of the historically best artefact by that metric. Used by Memory Agent.

`query_lineage(artefact_path)` → upstream artefacts (what produced this) and downstream artefacts (what this produced). Full provenance chain.

`health_check()` → `{"status": "healthy", "tool_count": N, "db_path": path, "record_count": N}`

### 5.5 DS Tools MCP Server — Complete Tool List

`read_file(file_path)` → `FileReadResult` JSON. Full file type detection and type-specific parsing. See §14.

`execute_code(code, data_paths, output_dir, timeout_seconds)` → `{stdout, stderr, exit_code, output_files, duration_ms}`. Runs code in a subprocess. The Ralph Loop wraps every call to this tool.

`get_sample(file_path, n_rows)` → JSON array of first N rows. For CSV and Parquet only.

`write_output(content, filename, run_id)` → full file path. Writes to `outputs/{run_id}/{filename}`.

`search_docs(query, library_name)` → extracted documentation text (max 4,000 characters). Invokes Browser-Use.

`web_research(query, max_results)` → list of result objects `{rank, title, url, snippet}`. Invokes DuckDuckGo.

`embed_text(text)` → embedding vector(s). Calls Azure OpenAI embedding deployment. Cached by SHA-256 of input.

`semantic_search(query, top_k)` → top-k matching artefact records with similarity scores. Queries the Memory Agent's in-process index.

`log_metrics(run_id, stage_name, metrics_dict)` → file path. Appends to `outputs/{run_id}/metrics/{stage_name}.json`.

`verify_stage(run_id, stage_name)` → `{passed: bool, failures: list[str]}`. Runs the DONE_CRITERIA assertions for a stage. This is what the Ralph Loop's verification step calls.

### 5.6 Bug Log MCP Server — Complete Tool List

`record_bug(run_id, stage_name, iteration_number, error_class, error_message, traceback, failed_code, repair_strategy)` → `bug_id`

`record_repair_attempt(bug_id, attempt_number, strategy_name, repaired_code, attempt_result)` → `repair_id`. `attempt_result` is `success` or `failure`.

`mark_bug_resolved(bug_id, final_fix_description, resolution_type)` → nothing. `resolution_type` is one of `pattern_match`, `browser_use_docs`, `duckduckgo_search`, `llm_reasoning`, `rewrite`, `human_fix`.

`record_ralph_loop_iteration(run_id, stage_name, iteration_number, verification_failures, action_taken)` → `iteration_id`

`query_bug_history(run_id)` → all bugs for a run with their repair histories. Full narrative from first failure to resolution.

`query_bug_patterns(error_class, library_name)` → historically successful repair strategies for this error class and library. Used by the Debug Agent to prioritise repair strategies based on what has worked before.

`query_unresolved_bugs(run_id)` → bugs that were not resolved (human gate was triggered). Used by Memory Agent to update procedural memory.

`health_check()` → `{"status": "healthy", "bug_count": N, "unresolved_count": N}`

---

## 6. MCP Streamable HTTP — Client via MCPStreamableHTTPTool

### 6.1 Construction and Lifecycle

Create one `MCPStreamableHTTPTool` instance per MCP server per agent at agent construction time in `agents/base.py`. Reuse across all sessions. Never create per call or per run.

The tool manages the MCP session lifecycle internally. It initialises the session (receiving `Mcp-Session-Id`) on first use and reuses it for all subsequent calls within the same application lifecycle.

### 6.2 Correct URL for the Tool

The URL must include the full path including the FastMCP suffix. If FastAPI mounts the FastMCP app at `/mcp`, and FastMCP adds its own `/mcp`, the full path is `/mcp/mcp`. Default URLs:

Tracking: `http://localhost:8100/mcp/mcp`
DS Tools: `http://localhost:8101/mcp/mcp`
Bug Log: `http://localhost:8102/mcp/mcp`

Document these in `config/settings.py` as `TRACKING_MCP_URL`, `DS_TOOLS_MCP_URL`, `BUG_LOG_MCP_URL`.

### 6.3 Known Issue — MCPStreamableHTTPTool Legacy GET

GitHub issue #5317 in the agent-framework repo documents a known issue where `MCPStreamableHTTPTool` uses a legacy GET SSE handshake that strict MCP 2025-06-18 servers reject with `405`. Verify your installed Framework version has the fix before testing MCP connections. If your server returns `405` on `MCPStreamableHTTPTool.connect()`, check the Framework version and upgrade if needed.

### 6.4 Distributed Tracing

When OpenTelemetry is installed, the Framework automatically injects W3C `traceparent` headers into MCP requests via `params._meta`. This propagates trace context from the agent's span into the MCP server's processing. No additional code is required. Configure an OTLP exporter in `agents/clients.py` at startup and all agent → MCP tool call traces are stitched together automatically.

---

## 7. Session Management and State Schema

### 7.1 Session Hierarchy per Run

```
Pipeline Run (run_id: uuid)
├── Orchestrator Session
│   ├── Ingestion Agent Session       (propagate_session=True)
│   ├── EDA Agent Session             (propagate_session=True)
│   ├── Cleaning Agent Session        (propagate_session=True)
│   ├── Feature Engineering Session   (propagate_session=True)
│   ├── Training Agent Session        (propagate_session=True)
│   ├── Tuning Agent Session          (propagate_session=True)
│   ├── Evaluation Agent Session      (propagate_session=True)
│   ├── Explainability Session        (propagate_session=True)
│   ├── Report Agent Session          (propagate_session=True)
│   └── Deployment Agent Session      (propagate_session=True)
├── Artefact Tracking Agent Session   (independent, async observer)
├── Bug Log Agent Session             (independent, async observer)
└── Memory Agent Session              (independent, async observer)
```

All sub-agent sessions use `propagate_session=True` so session state written by a sub-agent is visible to the Orchestrator after the tool call returns. The three observer agents have independent sessions — they do not share the pipeline session.

### 7.2 Session State Schema

Session state is the mechanism for passing structured data between tool calls within a session and between parent and child agents. The schema below is authoritative. Do not add undocumented keys to session state without updating this section and §21.

**Identity keys** — set by Orchestrator, read-only for all sub-agents:
`run_id` (string UUID), `task_description` (string), `pipeline_variant` (string: tabular/document_text/image/existing_model).

**Ingestion outputs:**
`file_type_result` (dict: FileReadResult), `schema` (dict: column→dtype), `input_artefact_path` (string).

**EDA outputs:**
`eda_report_path` (string), `eda_narrative_path` (string), `chart_paths` (list[str]), `data_quality_flags` (list[dict]).

**Cleaning outputs:**
`cleaned_dataset_path` (string), `transformation_log_path` (string), `cleaning_summary` (dict), `content_hash` (string).

**Feature Engineering outputs:**
`features_train_path` (string), `features_test_path` (string), `feature_manifest_path` (string), `target_column` (string), `task_type` (string: scenario label from §20.1).

**Training outputs:**
`model_artefact_path` (string), `model_type` (string: sklearn/xgboost/lightgbm/pytorch/huggingface), `onnx_model_path` (string), `baseline_metrics` (dict), `algorithm_comparison` (list[dict]).

**Tuning outputs:**
`model_artefact_path` (string, updated), `best_params_path` (string), `tuning_metrics` (dict).

**Evaluation outputs:**
`evaluation_report_path` (string), `deployment_recommendation` (dict: {recommend, blocking_issues, advisory_issues}), `fairness_metrics` (dict), `drift_report` (dict).

**Explainability outputs:**
`shap_values_path` (string), `shap_plot_paths` (list[str]), `explanation_narrative_path` (string).

**Report outputs:**
`report_md_path` (string), `report_html_path` (string).

**Deployment outputs:**
`endpoint_url` (string), `smoke_test_results` (dict).

**Loop control keys** — managed by the Ralph Loop:
`debug_attempts` (dict: stage_name→int), `ralph_loop_iteration` (dict: stage_name→int), `human_gate_decisions` (dict: gate_name→{decision, override_params}).

### 7.3 Session Resumability

After every successful stage, the Orchestrator writes a checkpoint to the Tracking MCP Server. On resume (`--resume-run-id` provided), the Orchestrator calls `query_run_status` and `query_artefact_history` to reconstruct session state from recorded artefact paths. It then skips all stages with successful checkpoints and starts execution from the first pending stage.

Every stage must be idempotent. Running a stage twice with the same inputs produces the same outputs at the same deterministic paths (keyed by `run_id`). The Ralph Loop does not cause non-idempotency — each loop iteration either overwrites intermediate outputs or picks up from where a previous iteration left off via the session state and artefact history.

### 7.4 Human-in-the-Loop Gates

Gates pause the pipeline at five points:

After file type detection: if `supported=False` or unexpected type, pause and ask user to confirm or abort.

After EDA: optional review (auto-approve in dev mode). User reviews the EDA report and confirms the cleaning approach.

After Cleaning: pause for outlier handling approval. The cleaning agent proposes strategies; the user approves or overrides.

After Evaluation: if `deployment_recommendation.recommend != "proceed"`, pause. User reviews fairness and metrics and makes the deployment decision.

After Ralph Loop exhaustion: if `ralph_loop_iteration[stage] >= MAX_ITERATIONS`, pause with the full bug log and ask the user to provide a fix or abort.

Gate mechanism: the Orchestrator calls `write_output` with the gate prompt and then polls `POST /approve/{run_id}/{gate_name}` on the Tracking MCP Server. The workflow resumes when the user calls that endpoint.

In dev mode (`PIPELINE_HUMAN_IN_THE_LOOP=false`): all gates auto-approve after a 1-second delay.

---

## 8. Three-Tier Memory System

### 8.1 Memory Architecture

The DS Agent implements a three-tier memory system directly inspired by cognitive architecture research (CoALA, MemGPT/Letta) and agent memory best practices for 2026. Memory mirrors human cognition: episodic (what happened), semantic (what I know), and procedural (how to do it).

Episodic memory captures retrievable logs of past interactions, decisions, and outcomes — the agent's record of what happened. Semantic memory stores persistent, structured knowledge extracted and indexed from prior experience — distilled facts, preferences, and learned generalisations. Procedural memory stores fix strategies: what repair approaches succeeded for specific error patterns, and which templates performed best for which task types.

### 8.2 Tier 1 — Working Memory (In-Session)

Working memory is the current session state (§7.2) plus the rolling chat history provided by `InMemoryHistoryProvider`. It is scoped to one pipeline run and one agent session. It does not persist across application restarts.

`InMemoryHistoryProvider` configuration:
- Orchestrator: `max_messages=40` (needs longer history to track pipeline status).
- Sub-agents: `max_messages=20` (stage-specific context only).
- Observer agents (Artefact Tracking, Bug Log, Memory): `max_messages=10` (event-driven, not conversational).

Working memory is intentionally limited. Large amounts of data are never stored in session state — only file paths. When an agent needs to inspect data, it calls `get_sample` or reads a file via `execute_code`.

### 8.3 Tier 2 — Episodic Memory (Run-Level Long-Term)

Episodic memory records what happened in each pipeline run. Episodic memory persists across sessions and allows an agent to recall prior conversations or workflow steps on demand.

**Storage:** SQLite table `episodic_events` in `data/memory.db`. Each row is one episode record.

**Episode record schema:** `episode_id`, `run_id`, `stage_name`, `timestamp`, `event_type` (stage_complete, bug_encountered, bug_resolved, human_gate_triggered, human_gate_approved), `summary` (a 2-3 sentence LLM-generated summary of what happened), `outcome` (success/failure/partial), `key_artefacts` (JSON list of artefact paths), `key_metrics` (JSON dict of metric values), `errors_encountered` (JSON list of error classes), `fixes_applied` (JSON list of fix strategy names), `duration_seconds`.

**Writing episodic records:** The Memory Agent writes one episode record per stage completion (or failure) by subscribing to the Tracking MCP Server and Bug Log MCP Server event streams. After each stage, it reads the artefact history and bug log for that stage, calls the fast Azure OpenAI model to generate a 2-3 sentence summary, and writes the episode record to `data/memory.db`.

**Reading episodic records:** Agents query episodic memory via the `semantic_search` tool on the DS Tools MCP Server. The tool embeds the query, computes cosine similarity against episode summaries, and returns the most relevant episodes. The Debug Agent uses this to find past successful repairs for similar errors. The Orchestrator uses this to check if similar tasks have been run before and what approaches worked.

### 8.4 Tier 3 — Long-Term Semantic Memory (Cross-Run Knowledge Base)

Semantic memory stores distilled, structured knowledge extracted across many runs. Unlike episodic memory which records events, semantic memory captures facts, preferences, and learned generalisations — backed by embedding similarity.

**Storage:** `data/semantic_memory.json` (artefact records) + `data/semantic_embeddings.npy` (embedding vectors). Loaded on startup, saved on every update.

**What is stored:** every artefact produced by any run (path, type, run_id, stage, key metrics, task type, dataset characteristics, timestamp). These are the same records the Artefact Tracking Agent produces, but indexed for semantic retrieval.

**Index structure:** an in-process NumPy array of embedding vectors, one per artefact record. Cosine similarity search at query time. For indexes under 50,000 records, pure NumPy is fast enough (< 100ms). Document in §35 that migration to a vector database is the recommended upgrade when the index exceeds 50,000 records.

**Reading semantic memory:** via the `semantic_search` MCP tool. Query → embed → cosine similarity → top-k records → LLM synthesis.

### 8.5 Tier 4 — Procedural Memory (Fix Strategy Knowledge Base)

Procedural memory stores how to do things: which repair strategies succeeded for which error patterns, which templates worked best for which task types, and which hyperparameter ranges produced good results for which dataset sizes.

**Storage:** `data/procedural_memory.json`. A structured dict with these top-level keys:

`error_strategies` — dict keyed by `error_class:library_name`. Each entry is a list of repair strategies sorted by historical success rate. Updated by the Bug Log Agent after each resolved bug.

`template_performance` — dict keyed by `task_type`. Each entry is a list of template names with their historical primary metric values. Updated by the Memory Agent after each successful training stage.

`hyperparameter_priors` — dict keyed by `model_type`. Each entry is a distribution over hyperparameter values that historically performed well. Updated by the Memory Agent after each successful tuning stage.

`domain_features` — dict keyed by `domain_keyword` (inferred from column names and task description). Each entry is a list of domain-specific feature engineering approaches that historically improved model performance. Updated by the Memory Agent after each successful feature engineering stage.

**Reading procedural memory:** The Debug Agent reads `error_strategies` when constructing its repair attempt. The Feature Engineering Agent reads `domain_features` before calling DuckDuckGo. The Tuning Agent reads `hyperparameter_priors` to initialise its Optuna search space.

### 8.6 Memory Agent Responsibilities

The Memory Agent is a single agent responsible for all three tiers. It runs as an async background task, not in the linear pipeline. It:

Subscribes to Tracking MCP Server and Bug Log MCP Server event streams (polling every 15 seconds).

After each stage event: reads the stage's artefact history and bug log, generates the episode summary, writes the episodic record, updates the semantic index with new artefacts, updates the procedural memory with new evidence (successful fixes, successful template-metric pairs, successful hyperparameter values).

After each full pipeline completion: writes a consolidated "run summary" episodic record that captures the full pipeline's outcome, the best model artefact, the primary metric achieved, and the total duration.

Answers retrospective queries via `semantic_search`. Queries the episodic store and the semantic index. Synthesises a grounded answer citing specific artefact paths and episode records.

---

## 9. AgentMiddleware — Pipeline and Ordering

### 9.1 Middleware Architecture in Framework 1.0

The middleware pipeline wraps every call to an agent's `run()` method. In Framework 1.0, `ChatMiddleware` runs per LLM call (including each iteration of the tool-calling loop), not once per invocation. If you implement `ChatMiddleware`, it must be safe for repeated execution within a single `run()` call.

Register middleware on individual agent instances, not on the client. Different agents need different middleware configurations.

### 9.2 Required Middleware per Agent

All pipeline agents (Orchestrator through Deployment): `LoggingMiddleware` → `RateLimitMiddleware` → `SafetyMiddleware` → `RetryMiddleware` → `TelemetryMiddleware`.

Observer agents (Artefact Tracking, Bug Log, Memory): `LoggingMiddleware` → `RetryMiddleware` → `TelemetryMiddleware`. No rate limiting (they make low-frequency calls). No safety middleware (they do not accept user input).

Debug Agent: `LoggingMiddleware` → `RetryMiddleware` → `TelemetryMiddleware`. No safety middleware (it needs to read and reason about unsafe code patterns without triggering safety blocks).

### 9.3 LoggingMiddleware

Subclass `AgentMiddleware`. Implement `process(context, next)`.

Before `await next(context)`: write a JSON entry to stdout with fields `timestamp`, `run_id` (from session state), `agent_name`, `event=agent_start`, `stage_name` (from session state).

After `await next(context)`: write another entry with `event=agent_end`, `duration_ms`, `input_token_count` (from `context.result.usage`), `output_token_count`, `total_tokens`, `status` (success or error).

Write JSONL (one JSON object per line) to stdout. No disk I/O in the middleware. In production, stdout is collected by the process supervisor and forwarded to the log backend.

The `run_id` is read from `context.function_invocation_kwargs.get("run_id")` — it is passed as a function invocation kwarg, not as a visible tool parameter. This is the correct pattern for runtime values that tools need but the model should not see.

### 9.4 SafetyMiddleware

Subclass `AgentMiddleware`. Before `await next(context)`: iterate over pending tool calls in the context. For each, serialise its input dict to JSON and run the blocked-pattern check from `config/safety_patterns.py`. If any pattern matches: do not call `await next(context)`, set `context.result` to an error result, call `record_bug` on the Bug Log MCP Server with `error_class=SafetyViolation`. Safety violations never retry.

The check runs on tool call inputs specifically, not on the full message. This is more precise than checking the full message string and avoids false positives from agents that legitimately discuss dangerous patterns (such as the Debug Agent reasoning about errors).

### 9.5 RetryMiddleware

Subclass `AgentMiddleware`. In `process`, run a loop up to `max_retries` (default 3). Call `await next(context)`. After each call, check `context.result.is_error`. If the error code is in `retryable_codes` (429, 502, 503, 504), sleep with exponential backoff (`min(base * 2**attempt + jitter, max_delay)`) and continue. If the error is non-retryable or retries are exhausted, break.

Never implement retry inside tool functions. The middleware handles it uniformly for all agents.

### 9.6 TelemetryMiddleware

Subclass `AgentMiddleware`. Use `opentelemetry.trace.get_tracer(__name__)`. In `process`, create a span named `{agent_name}.run` before `await next(context)`. Add attributes: `run_id`, `agent_name`, `model_deployment`, `stage_name`. After `await next(context)`, add final attributes: `input_tokens`, `output_tokens`, `duration_ms`, `status`. End the span. The Framework propagates this span context into all MCP tool calls via W3C `traceparent` headers automatically.

### 9.7 RateLimitMiddleware

Subclass `AgentMiddleware`. Implement a per-agent token bucket with capacity `calls_per_minute` (configurable per agent) and a refill rate of `calls_per_minute / 60` tokens per second. In `process`, attempt to take a token. If empty, sleep until one is available. Never fail on rate limit — always wait. Record the wait event in the Tracking MCP Server via `record_metric`. Continue with `await next(context)` after acquiring the token.

---

## 10. Tool Definition — @tool and FunctionInvocationContext

### 10.1 Tool Definition Pattern

Use the `@tool` decorator from `agent_framework` to define tools. The decorator creates a Pydantic model from the function's type annotations for input validation and JSON schema generation.

All parameters must have explicit type annotations. All parameters must have docstrings — the decorator uses the docstring for the tool description shown to the model. Tools must return a string. Use `FunctionInvocationContext` for runtime values that should not be visible in the model's tool schema.

Do not use `**kwargs` in tool definitions. This pattern is deprecated at Framework 1.0. Use `FunctionInvocationContext` instead.

### 10.2 FunctionInvocationContext

Add a parameter annotated as `FunctionInvocationContext` (named `ctx` by convention) to any tool function that needs runtime values. The Framework injects it automatically — it is not part of the tool's JSON schema.

Access via `ctx.session_state` (read/write session state), `ctx.function_invocation_kwargs` (runtime values passed to `agent.run()`). The `run_id` and `stage_name` are passed as function invocation kwargs so all tools can access them without including them in the model-visible schema.

### 10.3 Local vs MCP Tools

**Local tools** (in `tools/local_tools.py`, registered directly with agents): `get_session_state`, `set_session_state`, `check_artefact_exists`, `format_verification_result`. These are short utility functions that do not need the MCP server layer.

**DS Tools MCP tools** (in `mcp_servers/ds_tools/tools.py`, registered via `MCPStreamableHTTPTool`): all data science operations. Every operation that produces files, runs code, searches the web, or calls Azure OpenAI for embeddings goes through the MCP server so it is uniformly logged and retryable.

**Tracking MCP tools** (in `mcp_servers/tracking/tools.py`): all audit operations. Every `record_*` and `query_*` call goes through the Tracking MCP Server.

**Bug Log MCP tools** (in `mcp_servers/bug_log/tools.py`): all error and repair recording.

---

## 11. Sub-Agent Pattern — agent.as_tool()

### 11.1 Sub-Agent Invocation

Convert every pipeline stage agent to a tool using `agent.as_tool(name="run_{stage}_agent", description="...", propagate_session=True)`. Pass the resulting tool in the Orchestrator's `tools` list. The Orchestrator calls sub-agents like any other tool — in its conversation, as tool calls.

`propagate_session=True` is mandatory for all pipeline sub-agents. This causes the child agent's run to share the parent Orchestrator's session context. Session state written by a sub-agent is immediately visible to the Orchestrator after the tool call returns.

`propagate_session=False` (the default) is used only for the Debug Agent. The Debug Agent should not modify the pipeline session — it receives input, produces repaired code, and returns. It should not accidentally write to session state keys that the pipeline depends on.

### 11.2 Sub-Agent Final Message Contract

Every sub-agent must end its run with a structured final message containing:

A JSON summary object with these keys: `session_state_keys_written` (list of keys written this invocation), `artefacts_produced` (list of file paths), `status` (success or error), `completion_promise` (the stage-specific `<DONE>stage_name</DONE>` tag), `summary` (one sentence describing what was accomplished).

The Orchestrator reads this summary to know what the sub-agent produced. The Ralph Loop reads the `completion_promise` tag to trigger verification. Never rely on implicit outputs — the final message is the explicit contract.

### 11.3 Tool Naming Convention

`run_ingestion_agent`, `run_eda_agent`, `run_cleaning_agent`, `run_feature_engineering_agent`, `run_training_agent`, `run_tuning_agent`, `run_evaluation_agent`, `run_explainability_agent`, `run_report_agent`, `run_deployment_agent`, `run_debug_agent`.

Consistent naming makes the Orchestrator's conversation logs readable at a glance and makes the tool list comprehensible in the system prompt.

---

## 12. Artefact Tracking Agent

### 12.1 Role and Purpose

The Artefact Tracking Agent is a dedicated observer agent whose single responsibility is to maintain a complete, accurate record of every file produced by the pipeline. It is the system's provenance backbone. When a downstream agent needs to know what the Training Agent produced, when it was produced, what data it was trained on, and what model it contains, the Artefact Tracking Agent has the answer.

It is not a middleware hook and not a tool — it is a full agent with its own session, its own MCP connections, its own memory, and its own processing loop.

### 12.2 Event Subscription

The Artefact Tracking Agent polls the Tracking MCP Server's `query_artefact_history` tool every 10 seconds, passing the current `run_id`. It compares the result against its local in-memory set of already-processed artefacts. For any new artefact, it runs the full tracking protocol.

### 12.3 Tracking Protocol for Each New Artefact

Step 1 — Compute content hash. Open the file and compute its SHA-256 hash. Store in the artefact record.

Step 2 — Detect artefact type. For files the agent does not recognise by extension or path pattern, call `read_file` on the DS Tools MCP Server to confirm the type.

Step 3 — Extract metadata. For model files: load metadata without loading weights (ONNX metadata, safetensors metadata, sklearn `__class__.__name__` and `get_params()`). For dataset files: read row count, column count, null counts. For report files: extract section headings. For metrics files: read all key-value pairs.

Step 4 — Resolve lineage. Based on the stage name and the current session state, determine which upstream artefacts produced this file. Call `record_lineage` on the Tracking MCP Server to record the relationship.

Step 5 — Update Memory Agent. Call `semantic_search` with a query about the artefact type and task type to check if a similar artefact already exists in the index. If it does, compare metrics and record whether this is an improvement. Then call `embed_text` with the artefact's text representation and add it to the semantic memory index.

Step 6 — Write tracking record. Call `record_artefact` on the Tracking MCP Server with all collected metadata.

### 12.4 Artefact Tracking Agent — System Prompt Contract

**Identity:** You are the Artefact Tracking Agent. Your sole responsibility is to observe every file produced by the pipeline and maintain a complete, accurate, queryable record of what exists, what produced it, and what it contains.

**Authorised tools:** `read_file`, `embed_text`, `semantic_search`, `record_artefact`, `record_lineage`, `record_metric`, `query_artefact_history`.

**Processing protocol:**
1. Poll `query_artefact_history` every 10 seconds for the current `run_id`.
2. For each new artefact path, run the tracking protocol (steps 1–6 from §12.3).
3. Log all tracking actions to stdout in JSONL format.
4. Never modify any artefact file. Never modify session state. Read-only access to all pipeline outputs.

**Boundaries:** Do not clean or transform data. Do not call `execute_code`. Do not call `web_research` or `search_docs`. Do not modify any pipeline session state. Your role is observation and recording only.

### 12.5 Bug Detection by the Artefact Tracking Agent

The Artefact Tracking Agent performs passive bug detection: it checks every new artefact file against its expected schema. If a `cleaned_dataset.parquet` file has nulls in columns that should have been imputed, it logs an artefact quality warning to the Bug Log MCP Server via `record_bug` with `error_class=ArtefactQualityViolation`. The Ralph Loop's verification command would have already caught this — the Artefact Tracking Agent provides a second, independent detection path.

---

## 13. Bug Log Agent

### 13.1 Role and Purpose

The Bug Log Agent is a dedicated observer whose single responsibility is to maintain a complete, structured record of every error, every repair attempt, and every resolution. It transforms the chaotic event stream of failures and fixes into a searchable, queryable knowledge base that feeds the Debug Agent's repair strategies and the Memory Agent's procedural memory.

It runs as an async background task. It has its own session, its own MCP connections (to the Bug Log MCP Server), and its own memory access.

### 13.2 Event Subscription

The Bug Log Agent subscribes to the Bug Log MCP Server. When any agent calls `record_bug`, the Bug Log Agent receives the event and begins processing.

### 13.3 Bug Processing Protocol

When a new bug is received:

Step 1 — Classify the error. Map the `error_class` and `error_message` against `config/error_catalogue.py` to identify the canonical error pattern. Assign a `canonical_error_id` that groups related errors (e.g., all `CUDA out of memory` errors across different runs share the same canonical ID).

Step 2 — Query historical solutions. Call `query_bug_patterns` on the Bug Log MCP Server to retrieve historically successful repair strategies for this `error_class` and `library_name`. Read the `error_strategies` section of procedural memory. This informs the Debug Agent's Attempt 1 priority ordering.

Step 3 — Enrich the bug record. Add: the `canonical_error_id`, the top 3 recommended repair strategies (sorted by historical success rate), a link to the relevant documentation URL from `config/library_registry.py`, and any related past episodes from episodic memory.

Step 4 — Notify the Debug Agent. The Bug Log Agent does not fix bugs — it provides enriched context to the Debug Agent. It writes the enriched bug record to a file at `outputs/{run_id}/debug/{bug_id}.json` that the Debug Agent reads at the start of each repair attempt.

Step 5 — Monitor repair attempts. As each `record_repair_attempt` event arrives, update the bug record. When `mark_bug_resolved` arrives, update procedural memory: increment the success count for the winning repair strategy in `data/procedural_memory.json`.

Step 6 — Track Ralph Loop iterations. For each `record_ralph_loop_iteration` event, append to the bug's iteration history. When `MAX_ITERATIONS` is reached, flag the bug as requiring human escalation and write a structured escalation report to `outputs/{run_id}/escalations/{stage_name}.md`.

### 13.4 Bug Log Agent — System Prompt Contract

**Identity:** You are the Bug Log Agent. Your sole responsibility is to maintain a complete, accurate, queryable record of every error, repair attempt, and resolution in the pipeline.

**Authorised tools:** `query_bug_patterns`, `record_repair_attempt`, `mark_bug_resolved`, `record_ralph_loop_iteration`, `semantic_search`, `write_output`, `embed_text`.

**Processing protocol:**
1. Poll the Bug Log MCP Server for new bug events every 5 seconds.
2. For each new bug, run the bug processing protocol (§13.3).
3. When a bug is resolved, update procedural memory in `data/procedural_memory.json`.
4. When `MAX_ITERATIONS` is reached, write the escalation report and return it to the Orchestrator via session state key `escalation_report_{stage_name}`.

**Boundaries:** Do not fix bugs yourself. Do not call `execute_code`. Do not modify session state beyond the `escalation_report_{stage_name}` key. Your role is recording, enriching, and pattern learning — not execution.

### 13.5 Bug Log Schema

Every bug record in `outputs/{run_id}/debug/{bug_id}.json` contains:

`bug_id`, `run_id`, `stage_name`, `iteration_number`, `timestamp`, `error_class`, `error_message`, `traceback`, `failed_code`, `canonical_error_id`, `recommended_strategies` (list of strategy names sorted by historical success rate), `documentation_url`, `related_episodes` (list of episode summaries from episodic memory), `repair_attempts` (list of {attempt_number, strategy_name, repaired_code, result}), `resolved` (boolean), `resolution_type`, `final_fix_description`.

---

## 14. read_file Tool — File Type Detection Pipeline

### 14.1 Design Principle

Every pipeline starts with a file. The system detects the file's true type from its content, not its extension. A misidentified file produces corrupt data silently. Three detection layers run in sequence, stopping when confidence is sufficient.

Magika is initialised once as a module-level singleton in `tools/file_type_detector.py`:

```
_MAGIKA_INSTANCE = None

def get_magika():
    global _MAGIKA_INSTANCE
    if _MAGIKA_INSTANCE is None:
        from magika import Magika
        _MAGIKA_INSTANCE = Magika()
    return _MAGIKA_INSTANCE
```

Never call `Magika()` inside a request handler or tool function. Always call `get_magika()`.

### 14.2 Three-Layer Detection

**Layer 1 — Magika (primary)**
Call `get_magika().identify_path(file_path)`. Returns `content_type_label`, `mime_type`, `confidence` (0–1), `group`. Confidence below 0.85 → advance to Layer 2.

**Layer 2 — filetype (magic bytes)**
Call `filetype.guess(file_path)`. Returns a type object with `mime` and `extension`, or `None`. Use this to confirm Magika's result when confidence is low. If Layer 2 contradicts Magika, defer to Magika and record the disagreement as a warning in the result.

**Layer 3 — Content Heuristics (for ambiguous text)**
When both layers indicate `text/plain`: count delimiters in first 10 lines (CSV/TSV), check first character for JSON/JSONL, check for XML/HTML tags, try `ast.literal_eval` on the first line.

### 14.3 File Type Routing Table

**Tabular — Structured**

`csv` — detect encoding with `chardet.detect()` on first 10,000 bytes. Detect delimiter with `csv.Sniffer().sniff()`. Return: `schema`, `row_count` (fast line count), `sample_rows` (5 rows), `encoding`, `delimiter`, `has_header`.

`parquet` — read schema with `pyarrow.parquet.read_schema()` (zero data load). Return: `schema`, `row_count`, `sample_rows`, `compression`.

`xlsx` / `xls` — enumerate sheets with `openpyxl`. Return: `sheet_names`, `sheet_info` (per-sheet: rows, cols, headers). Do not load all data.

`json` — detect shape (list of dicts = tabular, dict = record). Return: `json_shape`, `schema`, `row_count`.

`jsonl` — read first 10 lines. Return: `schema`, `total_lines`, `sample_rows`.

`sqlite` — enumerate tables. Return: `tables` (list of table names with column info and row counts).

**Documents — Unstructured**

`pdf` — open with `pymupdf`. Extract text from first 5 pages. If total < 50 chars/page, set `requires_ocr=True`. Return: `page_count`, `text_preview` (3,000 chars), `requires_ocr`.

`docx` — open with `python-docx`. Return: `paragraph_count`, `table_count`, `text_preview`, `embedded_image_count`.

`txt` / `md` — return: `line_count`, `char_count`, `encoding`, `text_preview`.

**Images**

`png`, `jpeg`, `webp`, `gif`, `bmp` — open with Pillow. Return: `width`, `height`, `mode`, `file_size_bytes`. No pixel data embedding.

**Audio and Video**

`mp3`, `wav`, `mp4`, `avi` — read metadata with `mutagen`. Return: `duration_seconds`, `bitrate`, `codec`, `requires_transcription=True`.

**Archives**

`zip` — list contents. Return: `member_count`, `member_names`, `total_uncompressed_bytes`. Do not auto-extract.

`tar`, `gz`, `bz2` — same pattern using `tarfile`.

**ML Artefacts**

`pickle` (magic bytes `\x80\x04`) — read first 64 bytes only (do not fully unpickle untrusted content). Return: `pickle_protocol`, `requires_verification=True`.

`onnx` — use `onnx.load()`. Return: `graph_inputs`, `graph_outputs`, `opset_version`.

`safetensors` — use `safetensors.safe_open()`. Return: `tensor_names`, `tensor_shapes`, `metadata`.

**Unknown / Unsupported**

Return: `detected_type=unknown`, `mime_type`, `hex_preview` (first 64 bytes as hex), `supported=False`.

### 14.4 Downstream Routing Logic

`supported=False` → human gate.
`requires_ocr=True` → human gate with OCR suggestion.
`requires_transcription=True` → check task description. If transcription task, route to Whisper DL template. Otherwise, human gate.
Detected type is `pickle`, `onnx`, `safetensors` → `pipeline_variant=existing_model`. Skip to Evaluation.
Detected type is `pdf`, `docx`, `txt`, `md` → `pipeline_variant=document_text`. Use embedding-based features.
Detected type is image format → `pipeline_variant=image`.
Detected type is `csv`, `parquet`, `xlsx`, `json`, `jsonl`, `sqlite` → `pipeline_variant=tabular`.
Detected type is `zip` or `tar` → human gate with member list.

---

## 15. Agent Catalogue and System Prompt Contracts

### 15.1 System Prompt Structure (mandatory for all agents)

Every system prompt must follow this exact structure. An agent with a poorly structured prompt produces inconsistent results, skips steps, and produces incomplete outputs.

**Section 1 — Identity** (1 sentence): who this agent is and its single responsibility.

**Section 2 — Authorised tools**: bullet list — tool name + one sentence on when to use it. Never list tools the agent cannot access.

**Section 3 — Input contract**: session state keys this agent reads at the start of its run.

**Section 4 — Processing protocol**: numbered steps in strict execution order. Each step names a specific tool call. No vague instructions.

**Section 5 — Output contract**: session state keys this agent writes + files it produces + the completion promise tag it must output.

**Section 6 — Completion signal**: the exact `<DONE>stage_name</DONE>` tag and the final JSON summary structure.

**Section 7 — Escalation**: what to do if any step fails (call `record_bug` on Bug Log MCP Server, then let the Ralph Loop handle retry).

**Section 8 — Boundaries**: explicit list of what this agent must not do.

### 15.2 Orchestrator Agent

**Identity:** You are the Orchestrator. You direct the data science pipeline by routing tasks to sub-agents, monitoring progress, managing human gates, and resuming after failures. You never execute data science logic yourself.

**Model:** primary deployment.

**Tools:** all `run_{stage}_agent` tools, `query_run_status`, `record_checkpoint`, `record_start`, `record_end`, `write_output`, `verify_stage`.

**Processing protocol:**
1. Generate a `run_id` UUID. Write to session state. Call `record_start` for the orchestrator stage.
2. If `--resume-run-id` provided: call `query_run_status` to identify completed stages. Reconstruct session state from `query_artefact_history`. Skip completed stages.
3. Call `run_ingestion_agent`. Verify output with `verify_stage("ingest")`. If fail: route to Ralph Loop retry.
4. For each subsequent stage in the workflow graph: call the corresponding `run_{stage}_agent` tool. After each, call `verify_stage(stage_name)`. If verification passes: call `record_checkpoint`. If fails: log to Bug Log MCP Server, increment Ralph Loop iteration counter, re-invoke the stage with failure context injected.
5. At each human gate: call `write_output` with the gate prompt. Poll the approval endpoint every 5 seconds until a decision arrives.
6. When `ralph_loop_iteration[stage] >= MAX_ITERATIONS`: trigger the human gate for that stage with the escalation report.
7. On full pipeline completion: call `record_end` with success status. Return the final summary to the user.

**Completion signal:** `<DONE>orchestrator</DONE>` emitted only when all stages are verified and the deployment smoke test passes.

**Boundaries:** Do not call `execute_code`. Do not generate training code. Do not compute statistics. Do not make deployment decisions (that is the Evaluation Agent's role).

### 15.3 Ingestion Agent

**Identity:** You are the Ingestion Agent. Your sole responsibility is to call `read_file`, validate the result against the task description, detect PII, and record the raw artefact.

**Model:** fast deployment.

**Processing protocol:**
1. Call `record_start`.
2. Call `read_file` with `input_artefact_path` from session state.
3. If `supported=False`: return routing mismatch error immediately.
4. If `requires_ocr=True` or `requires_transcription=True`: return with the flag set for the human gate.
5. Validate routing: check that the file type is consistent with the task description. Image file + regression task = routing mismatch.
6. Run PII heuristic scan on the schema: check column names against PII-indicative list. Check sample values for email, phone, SSN patterns.
7. If PII detected: log a PII warning artefact to Tracking MCP Server. Set session state key `pii_detected=True`. This triggers the human gate in the Orchestrator.
8. Write `file_type_result`, `schema`, and `input_artefact_path` to session state.
9. Call `record_artefact` with artefact type `raw_dataset`.
10. Call `record_end`. Output the final summary JSON with `<DONE>ingest</DONE>`.

**Boundaries:** Do not clean, transform, or analyse data. Do not make routing decisions — report the type and let the Orchestrator route.

### 15.4 EDA Agent

**Identity:** You are the EDA Agent. Your sole responsibility is to produce a complete statistical profile of the dataset, generate visualisation charts, and write a natural-language narrative.

**Model:** fast deployment for stat computation prompts; primary deployment for narrative generation.

**Processing protocol:**
1. Call `record_start`. Call `get_sample` to inspect 5 rows.
2. Read `schema` and `input_artefact_path` from session state.
3. Generate the EDA code using `templates/eda_template.py` with column names and dtypes substituted.
4. Call `execute_code` with the generated code. The Ralph Loop wraps this — if the code fails, the Bug Log Agent enriches the error and the Debug Agent produces a repair.
5. Read `stats.json` from the output files. Verify it is non-empty and contains all required keys.
6. Call the primary model (a direct Azure OpenAI completion, not a tool call) with the stats to generate the 3-paragraph narrative. Write to `eda_narrative.md` via `write_output`.
7. Write all session state keys. Call `record_artefact` for each output file. Call `log_metrics` with summary statistics.
8. Call `record_end`. Output final summary with `<DONE>eda</DONE>`.

**Completion criteria (verified by Ralph Loop):** `stats.json` exists and is non-empty. `eda_narrative.md` exists and is non-empty. At least 3 chart PNG files exist. All required keys present in `stats.json`: `describe`, `null_counts`, `correlation_matrix`, `data_quality_flags`.

**Boundaries:** Do not clean or transform data. Do not train models. Do not make cleaning recommendations — only flag issues in `data_quality_flags`.

### 15.5 Cleaning Agent

**Identity:** You are the Cleaning Agent. Your sole responsibility is to detect data quality issues and apply corrective transformations, documenting every decision with justification.

**Model:** primary deployment.

**Processing protocol:**
1. Call `record_start`. Read `eda_report_path` and `data_quality_flags` from session state.
2. Apply the chain-of-thought protocol in strict order:
   a. Enumerate all null-value columns. Classify missingness (MCAR/MAR/MNAR) from stats.
   b. For each column, select and justify imputation strategy.
   c. Enumerate outlier-affected columns. Propose handling strategy.
   d. Present outlier strategy as human gate prompt. Wait for approval.
   e. Generate cleaning code encoding all approved decisions.
3. Call `execute_code` with the cleaning code.
4. Read `cleaning_summary.json` from output. Verify null counts in imputed columns are zero.
5. Write session state keys. Call `record_artefact`. Call `record_end`. Output `<DONE>clean</DONE>`.

**Completion criteria:** `cleaned_dataset.parquet` exists, is non-empty, and has zero nulls in all columns that were imputed. `transformation_log.json` exists and documents every transformation with a justification field.

**Boundaries:** Do not drop columns without logging justification. Do not impute the target column. Do not run model training or EDA.

### 15.6 Feature Engineering Agent

**Identity:** You are the Feature Engineering Agent. Your sole responsibility is to research domain-specific features, create and apply feature transforms, assess importance, and produce the feature manifest.

**Model:** primary deployment.

**Processing protocol:**
1. Call `record_start`. Read procedural memory `domain_features` for the inferred domain.
2. Call `web_research` with a domain-specific query (enriched by procedural memory).
3. Call `get_sample` on the cleaned dataset.
4. Generate feature engineering code including: all applicable transforms, importance assessment via RandomForest, and train/test split (fit transforms on train only — no leakage).
5. Call `execute_code`. Debug loop on failure.
6. Read `feature_manifest.json`. Verify all columns are documented. Detect `task_type` from target column characteristics.
7. Write session state keys. Call `record_artefact`. Call `log_metrics`. Call `record_end`. Output `<DONE>feature_eng</DONE>`.

**Completion criteria:** `features_train.parquet` and `features_test.parquet` exist with zero nulls. `feature_manifest.json` exists, is valid JSON, and covers 100% of the columns in the cleaned dataset. `task_type` is set to a valid scenario label.

**Boundaries:** Do not train models beyond the quick RandomForest. Do not modify the target column. Do not re-clean the dataset.

### 15.7 Model Selection and Training Agent

**Identity:** You are the Model Selection and Training Agent. Your sole responsibility is to select algorithms, generate training code, execute training, and identify the best baseline model.

**Model:** primary deployment.

**Processing protocol:**
1. Call `record_start`. Read `task_type`, `feature_manifest_path`, `features_train_path`, `target_column` from session state.
2. Read procedural memory `template_performance` for the `task_type`. This informs which templates have historically performed well.
3. If DL scenario: call `web_research` for architecture recommendations.
4. Run the `before_model_train` validator on the generated code before calling `execute_code`.
5. Run minimum 3 algorithms (classical ML) or 1 DL template. Call `execute_code` for each. The Ralph Loop wraps each execution.
6. Select the best model by primary metric. Update `model_artefact_path`. Write `baseline_metrics`, `algorithm_comparison`.
7. Call `record_artefact` for each model file. Call `log_metrics`. Call `record_end`. Output `<DONE>train</DONE>`.

**Completion criteria:** `model_artefact_path` in session state points to an existing file. `baseline_metrics` contains the primary metric key for the task type. `model.onnx` exists. `metrics/training.json` is non-empty.

**Boundaries:** Do not tune hyperparameters. Do not evaluate on the test set. Do not deploy. Maximum 5 algorithm trials.

### 15.8 HP Tuning Agent

**Identity:** You are the HP Tuning Agent. Your sole responsibility is to define a model-appropriate search space, run Optuna optimisation, and produce the tuned model.

**Model:** fast deployment.

**Processing protocol:**
1. Call `record_start`. Read `model_type`, `model_artefact_path`, `features_train_path`, `baseline_metrics`.
2. Read procedural memory `hyperparameter_priors` for `model_type`. Use these as the initial search space centre.
3. Generate Optuna study code with `n_trials=50` and `timeout=1800` (30 minutes).
4. Call `execute_code`. Ralph Loop wraps it.
5. Read `best_params.json`. Verify the tuned CV metric is ≥ baseline metric.
6. Update `model_artefact_path` to tuned model. Write `best_params_path`, `tuning_metrics`.
7. Call `record_artefact`. Call `log_metrics`. Call `record_end`. Output `<DONE>tune</DONE>`.

**Completion criteria:** `model_artefact_path` updated to tuned model file. `best_params.json` exists. `tuning_metrics[primary_metric] >= baseline_metrics[primary_metric]`.

**Boundaries:** Do not change the algorithm. Do not run more than 50 Optuna trials. Do not evaluate on the test set.

### 15.9 Evaluation Agent

**Identity:** You are the Evaluation Agent. Your sole responsibility is to compute all metrics, run fairness analysis, assess drift, and produce a deployment recommendation.

**Model:** primary deployment.

**Processing protocol:**
1. Call `record_start`. Read model and data paths from session state.
2. Generate evaluation code. Metrics auto-selected by `task_type` (see §20.4). Include Fairlearn analysis. Include KS drift test.
3. Call `execute_code`. Ralph Loop wraps it.
4. Read `evaluation_report.json`, `fairness_metrics.json`, `drift_report.json`.
5. Compute `deployment_recommendation`: `recommend` based on metric thresholds and fairness violation severity. `blocking_issues` list. `advisory_issues` list.
6. If `recommend != "proceed"`: trigger human gate.
7. Write all session state keys. Call `record_artefact`. Call `log_metrics`. Call `record_end`. Output `<DONE>evaluate</DONE>`.

**Completion criteria:** `evaluation_report.json` exists with all task-appropriate metric keys. `fairness_metrics.json` exists (mandatory for classification tasks). `deployment_recommendation` dict has all three keys: `recommend`, `blocking_issues`, `advisory_issues`.

**Boundaries:** Do not modify the model. Do not apply transforms to the test set not present in the training pipeline. Never deploy without computing fairness metrics for classification tasks.

### 15.10 Explainability Agent

**Identity:** You are the Explainability Agent. Your sole responsibility is to compute SHAP values, produce explanation plots, and generate a plain-English narrative.

**Model:** primary deployment.

**Processing protocol:**
1. Call `record_start`. Read model, data, and manifest paths from session state.
2. Select SHAP explainer by `model_type`: `TreeExplainer` (sklearn trees, XGBoost, LightGBM), `LinearExplainer` (linear models), `DeepExplainer` (PyTorch), `KernelExplainer` (fallback).
3. Generate SHAP code. For DL models, also generate Captum Integrated Gradients code.
4. Call `execute_code`. If SHAP API error: call `search_docs` with `library_name="shap"` before the Debug Agent attempts repair. Ralph Loop wraps it.
5. Read SHAP values and plots. Call primary model to generate narrative. Write to `narrative.md`.
6. Write session state keys. Call `record_artefact`. Call `record_end`. Output `<DONE>explain</DONE>`.

**Completion criteria:** `shap_values.npy` exists. At least 2 PNG plot files exist (beeswarm + bar minimum). `narrative.md` exists and is at least 100 characters.

**Boundaries:** Use test set only for SHAP (no training set leakage). Produce at most 20 plots per run.

### 15.11 Report Agent

**Identity:** You are the Report Agent. Your sole responsibility is to assemble all artefacts into a Markdown and self-contained HTML final report.

**Model:** primary deployment.

**Processing protocol:**
1. Call `record_start`. Read all input artefact paths from session state.
2. Read the content of each artefact file (stats, transformation log, feature manifest, evaluation report, fairness metrics, narrative).
3. Fill the report template. Embed all charts as base64 inline in the HTML version.
4. Call `write_output` for `final_report.md`. Call `write_output` for `final_report.html`.
5. Verify the HTML file: check that all required section headings are present (executive summary, dataset overview, EDA, data quality, features, model comparison, best model, fairness, explainability, deployment recommendation, appendix). Check that no external URLs are referenced in the HTML.
6. Write session state keys. Call `record_artefact`. Call `record_end`. Output `<DONE>report</DONE>`.

**Completion criteria:** `final_report.html` exists, is non-empty, passes the offline rendering check (all section headings present, no external URLs), and all charts are embedded as base64. `final_report.md` exists.

**Boundaries:** Do not compute any statistics. Do not call `execute_code`. Do not modify any input artefact.

### 15.12 Deployment Agent

**Identity:** You are the Deployment Agent. Your sole responsibility is to package the model as a FastAPI endpoint, verify it with a smoke test, and record the endpoint URL.

**Model:** fast deployment.

**Processing protocol:**
1. Call `record_start`. Read `deployment_recommendation`, model paths, `feature_manifest_path` from session state.
2. Apply `before_deploy` check: if `deployment_recommendation.recommend == "block"`, do not proceed. Log the blocking issues and return an error.
3. Generate FastAPI wrapper code (using `templates/deploy_template.py`). The endpoint includes `/predict`, `/health`, `/model_card`.
4. Write the FastAPI app file via `write_output`.
5. Generate server startup code. Call `execute_code` to start uvicorn in the background. Wait 3 seconds for server initialisation.
6. Call `execute_code` with smoke test code. Send 10 POST requests. Verify response schema. Compute P99 latency. If P99 > 500ms: log advisory warning to Bug Log MCP Server.
7. Write `endpoint_url`, `smoke_test_results` to session state. Call `record_artefact`. Call `record_end`. Output `<DONE>deploy</DONE>`.

**Completion criteria:** `endpoint_url` is set in session state. `smoke_test_results.passed_count == 10`. The endpoint returns HTTP 200 on `/health`.

**Boundaries:** Do not retrain the model. Do not proceed if `before_deploy` check fails. Do not skip the smoke test.

### 15.13 Debug Agent

**Identity:** You are the Debug Agent. You receive failing Python code, an error message, a traceback, an attempt number, and a bug record from the Bug Log Agent. You return repaired Python code and nothing else — no explanations, no markdown fences, no preamble.

**Model:** primary deployment.

**Processing protocol (strictly by attempt number):**

Attempt 1 — Pattern match from bug record. The enriched bug record from the Bug Log Agent contains the recommended repair strategies sorted by historical success rate. Apply the top-ranked strategy. Read `config/error_catalogue.py` for the fix transformation. Return repaired code.

Attempt 2 — Browser-Use documentation. Call `search_docs` with the error message and `library_name` from the traceback. Pass extracted documentation + original code + error to the model. Return repaired code.

Attempt 3 — DuckDuckGo search. Call `web_research` with query: `"{error_class}: {error_message[:100]} {library_name} python fix site:stackoverflow.com OR site:github.com"`. Pass results + original code + error to the model. Return repaired code.

Attempt 4 — First-principles reasoning. Present original code, error, traceback to the model with explicit instruction: "Do not reference the failing code's structure. Reason about what the code is trying to accomplish. Implement it correctly from scratch." Return repaired code.

Attempt 5 — Full rewrite. Present original task description only (no failing code). Implement the task using the simplest, most reliable approach. May use a different library or algorithm. Return repaired code.

**Output constraint:** the output is raw Python source code and nothing else. It is written directly to a file and executed without modification.

**Boundaries:** Do not modify session state (except `debug_attempts`). Do not call `execute_code` or any tracking tool. Do not change the algorithm unless on Attempt 5.

---

## 16. Workflow Graph and Orchestration

### 16.1 Static Graph Definition

The workflow graph is defined statically in `workflows/pipeline_graph.py` as a Python dict keyed by `pipeline_variant`. Each value is an ordered list of stage names. Define it once — do not generate dynamically.

**`tabular` variant:** `read_file_ingest → eda → clean → feature_eng → train → tune → evaluate → explain → report → deploy`

**`document_text` variant:** `read_file_ingest → eda → clean → feature_eng_text → train → tune → evaluate → explain → report → deploy`

**`image` variant:** `read_file_ingest → feature_eng_image → train → tune → evaluate → explain → report → deploy`

**`existing_model` variant:** `read_file_ingest → evaluate → explain → report → deploy`

Human gates are inserted between: `read_file_ingest` and `eda` (file type confirmation), `eda` and `clean` (EDA review), `clean` and `feature_eng` (outlier handling), `evaluate` and `explain` (fairness review), `report` and `deploy` (deploy approval). All gates are bypassed in dev mode.

### 16.2 Stage Execution with Ralph Loop

For each stage in the graph, the Orchestrator:

1. Checks if the stage has a successful checkpoint (skip if yes — resume mode).
2. Reads the Ralph Loop iteration counter for this stage from session state.
3. Reads the current bug context for this stage from the Bug Log Agent (enriched bug records, if any previous iterations failed).
4. Constructs the stage prompt: base system prompt + current session state summary + bug context + verification criteria.
5. Calls the sub-agent tool with `propagate_session=True`.
6. Checks for the completion promise tag in the sub-agent's output.
7. If tag present: calls `verify_stage(stage_name)` on the DS Tools MCP Server.
   - If all criteria pass: records checkpoint, writes episode to Memory Agent, advances to next stage.
   - If any criterion fails: logs the failure to Bug Log MCP Server, increments Ralph Loop counter, constructs failure feedback, goes to step 4.
8. If iteration counter ≥ `MAX_ITERATIONS`: triggers human gate.

### 16.3 Verification Command Design

Each stage's DONE_CRITERIA in `workflows/criteria.py` defines three elements:

`required_files` — paths that must exist (relative to `outputs/{run_id}/`). Checked with `os.path.exists` and `os.path.getsize > 0`.

`required_session_keys` — session state keys that must be present and non-null.

`assertions` — Python callables that take `(session_state, output_dir)` and return `(bool, str)`. These are domain checks: null counts, metric ranges, row count preservation, file format validity.

The verification runs in under 1 second (no LLM calls). Verification is the arbiter of done — not the agent's self-report.

---

## 17. Ralph Loop Implementation — Self-Correcting Execution

### 17.1 The Core Loop

The Ralph Loop in `workflows/ralph_loop.py` implements what Ralphify describes: each iteration re-reads the criteria from disk, runs commands, captures output, replaces placeholders with output, and pipes the assembled context to the agent. Failed commands still capture output — that is what makes the loop self-healing.

The loop for a single stage:

```
iteration = 0
while iteration < MAX_ITERATIONS:
    bug_context = read_bug_context(run_id, stage_name)
    stage_prompt = build_prompt(base_prompt, session_state, bug_context, criteria)
    output = run_stage_agent(stage_name, stage_prompt)
    
    if COMPLETION_TAG in output:
        verification = verify_stage(run_id, stage_name)
        if verification.passed:
            record_checkpoint(run_id, stage_name)
            update_episodic_memory(run_id, stage_name, "success")
            return SUCCESS
        else:
            log_ralph_iteration(run_id, stage_name, iteration, verification.failures)
            inject_failure_feedback(stage_name, verification.failures)
    else:
        log_ralph_iteration(run_id, stage_name, iteration, ["missing_completion_tag"])
    
    iteration += 1

trigger_human_gate(run_id, stage_name)
```

### 17.2 Fresh Context on Retry

After each run, the loop checks whether there are still open tasks. The key insight: each retry runs with a completely fresh context window. No compaction, no degradation. State survives between iterations through the filesystem and bug log — not through conversation history.

This is implemented by creating a new agent session for each Ralph Loop iteration. The `InMemoryHistoryProvider` is not reused across iterations. Instead, the bug context from the Bug Log Agent and the verification failure messages are injected as fresh system context into the new session. The agent sees exactly what failed, but without the cognitive overhead of a long, degraded conversation history.

### 17.3 Stateless Inner Loop vs Stateful Outer Loop

**Stateless inner loop** (for `execute_code` calls within a stage): each `execute_code` call uses a fresh subprocess. The code itself accumulates fixes across attempts, but the subprocess is always fresh. This is the Debug Agent's repair loop — up to 5 attempts, each with a new subprocess.

**Stateful outer loop** (for entire stage re-runs): the stage agent is re-run with the accumulated context from all previous iterations. Files already written in previous iterations persist. The stage does not re-run work that already succeeded — it picks up from where the last iteration left off using session state and artefact existence checks.

### 17.4 Failure Feedback Construction

When verification fails, the feedback injected into the next iteration must be specific and actionable. Vague feedback ("please fix the errors") is as unhelpful as no feedback.

Feedback template: `"The previous iteration of {stage_name} failed verification with the following specific issues:\n\n{for each failure: '- {assertion_name}: {failure_message}. Expected: {expected}. Actual: {actual}.'}\n\nThe following files exist from the previous attempt: {list of existing output files}. The following files are still missing: {list of missing files}. Address each issue specifically. Do not re-run work that already succeeded."`

This precision is what makes the Ralph Loop effective for data science pipelines. A failing assertion like "null_count_in_age_column == 0 failed: actual null count was 12" is far more actionable than "cleaning failed."

### 17.5 Escalation Report

When `MAX_ITERATIONS` is reached, the Bug Log Agent writes `outputs/{run_id}/escalations/{stage_name}.md`. This report contains: the stage name, total iterations attempted, each iteration's verification failures, each Debug Agent repair attempt and its outcome, the Browser-Use documentation extracted (if invoked), the DuckDuckGo search results (if invoked), the LLM's first-principles reasoning (from Attempt 4), and a structured recommendation for the human reviewer.

The Orchestrator presents this report to the user via the human gate mechanism.

---

## 18. Browser-Use Integration

### 18.1 When Invoked

Browser-Use is called inside the `search_docs` DS Tools MCP tool in exactly two situations:

The Debug Agent requests documentation for a specific library error (Attempt 2 of the repair loop). The `library_name` is inferred from the first non-project file path in the traceback.

The Explainability Agent encounters an API compatibility error with SHAP, LIME, or Captum. Before the Debug Agent is invoked, the Explainability Agent calls `search_docs` directly as part of its processing protocol.

Do not use Browser-Use for general web search. DuckDuckGo handles that.

### 18.2 Azure OpenAI as the Browser-Use LLM

Configure Browser-Use to use `ChatAzureOpenAI` (from `langchain_openai`) with the fast Azure OpenAI deployment. Pass the same `AZURE_OPENAI_ENDPOINT`. Browser-Use's internal navigation reasoning consumes from the fast deployment quota. Account for 500–2,000 tokens per navigation step in token budget planning.

### 18.3 Library Documentation Registry

`config/library_registry.py` maps library names to their documentation root URLs. Required entries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `torch`, `torchvision`, `transformers`, `datasets`, `sentence-transformers`, `shap`, `lime`, `captum`, `optuna`, `pyarrow`, `openpyxl`, `pymupdf`, `pillow`, `fastmcp`, `agent-framework`, `magika`, `filetype`.

When the inferred library is not in the registry, the `search_docs` tool falls through to DuckDuckGo immediately (does not attempt Browser-Use with an unknown URL).

### 18.4 Configuration

```
max_steps: 50
headless: True
model: ChatAzureOpenAI (fast deployment)
task: "Find information about: {query}. Focus on: API usage examples, correct function signatures, and fix examples for the error: {error_message}. Extract code examples relevant to the error."
```

### 18.5 Output Handling

Truncate `final_result` to 4,000 characters. If `final_result` is empty or off-topic, log a `browser_use_miss` event to the Bug Log MCP Server and advance to Attempt 3 (DuckDuckGo). Never retry Browser-Use on the same query within the same debug session.

---

## 19. DuckDuckGo Deep Research

### 19.1 Three Invocation Contexts

**Debug Agent Attempt 3** — query format: `f"{error_class}: {error_message[:100]} {library_name} python fix site:stackoverflow.com OR site:github.com"`. Returns top 10 results.

**Feature Engineering Agent** — query format: `f"{domain_keywords} feature engineering best practices {year}"`. Domain keywords inferred from column names and task description. Enriched by procedural memory `domain_features` before calling.

**Model Selection Agent** — query format: `f"best {task_type} algorithm {data_characteristics} benchmark 2025 2026"`. Data characteristics: small/medium/large, class imbalance, text/tabular/image.

### 19.2 Query Rules

Include year (2025 or 2026) in research queries. Omit year from debug queries. Include `site:` filters for debug queries. Omit `site:` filters for research queries.

### 19.3 Rate Limiting

Track via the Bug Log MCP Server's `record_tool_call`. Global limit: 30 queries/hour. If exceeded, return `rate_limit_exceeded` and the calling agent advances to the next strategy without waiting.

### 19.4 Result Processing

Return top 10 as structured list: `{rank, title, url, snippet (300 chars)}`. Pass directly to the LLM — do not pre-summarise.

---

## 20. Deep Learning and Generative AI Coverage

### 20.1 Scenario Detection Table

The Feature Engineering Agent sets `task_type` based on data characteristics and `task_description`. Explicit user statement overrides inference.

| Scenario | Detection Signal | Template |
|---|---|---|
| Binary Classification | target has 2 unique values | `templates/sklearn_train.py` |
| Multi-class Classification | target has 3–100 unique values | `templates/sklearn_train.py` |
| Regression | numeric continuous target | `templates/sklearn_train.py` |
| Imbalanced Classification | class ratio > 10:1 | `templates/imbalanced.py` |
| Time Series Forecasting | datetime index + numeric target | `templates/dl/time_series.py` |
| Text Classification | primary feature is text column | `templates/dl/bert_classify.py` |
| NER | text + BIO-tagged labels | `templates/dl/bert_ner.py` |
| Text Generation | language modelling task | `templates/dl/llm_finetune.py` |
| Image Classification | image file paths as features | `templates/dl/cnn_classify.py` |
| Object Detection | image paths + bounding boxes | `templates/dl/yolo_detect.py` |
| Tabular DL | tabular + explicit DL request | `templates/dl/tabnet.py` |
| Clustering | no target column | `templates/sklearn_cluster.py` |
| Anomaly Detection | highly imbalanced, outlier task | `templates/anomaly.py` |
| RAG System | document corpus + Q&A task | `templates/dl/rag.py` |
| Audio Transcription | audio files detected | `templates/dl/whisper.py` |
| Multimodal | image + text features | `templates/dl/multimodal.py` |
| LLM Evaluation | existing LLM outputs | `templates/dl/llm_eval.py` |

### 20.2 DL Training Loop Checklist

Every DL template must satisfy all items before being submitted via `execute_code`. The `before_model_train` validator in `tools/validators.py` checks for the presence of these function names using `ast.parse()`:

`DataLoader`, configurable `batch_size`, augmentation appropriate to data modality, normalisation matching pre-trained model expectations, class weighting or `WeightedRandomSampler` for imbalanced data, task-appropriate loss function (not default MSE for classification), `AdamW` with `weight_decay`, LR scheduler (one of `CosineAnnealingLR`, `OneCycleLR`, `get_linear_schedule_with_warmup`), `torch.autocast` for mixed precision, `torch.nn.utils.clip_grad_norm_` with `max_norm=1.0`, early stopping with configurable patience and min_delta, `torch.save(model.state_dict(), ...)` on validation improvement, `torch.manual_seed` and `np.random.seed` and `random.seed` all set to the same value, training loss and validation loss logged to `metrics.json` per epoch, `torch.cuda.empty_cache()` at start of each epoch.

### 20.3 DL Auto-Remediation (Attempt 1 Pattern Match)

| Error Pattern | Remediation |
|---|---|
| `CUDA out of memory` | Halve `batch_size`. If still OOM, enable `gradient_checkpointing_enable()`. If still OOM, set `device='cpu'` |
| `loss is nan` | Check `torch.isnan(batch).any()`. If clean, reduce LR by factor of 10 |
| `training accuracy equals class prior` | Verify labels dtype is `long` for cross-entropy. Verify loss function |
| `DataLoader worker crashed` | Set `num_workers=0` |
| `NCCL error` | Set `nproc_per_node=1` |
| HuggingFace `KeyError` on model load | Call `search_docs` with `library_name="transformers"`. Fall back to base variant |
| Loss does not decrease for 5 epochs | Apply LR warmup over 10% of total steps |
| Overfitting after epoch 5 | Increase dropout by 0.1. Add `weight_decay=0.01` |

### 20.4 Metric Table by Task Type

| Task Type | Primary Metric | Additional Required Metrics |
|---|---|---|
| Binary Classification | ROC-AUC | Accuracy, Precision, Recall, F1, PR-AUC, MCC, Confusion Matrix |
| Multi-class | Macro F1 | Accuracy, per-class F1, Cohen's Kappa, OvR ROC-AUC |
| Regression | RMSE | MAE, R², MAPE, Huber Loss |
| Imbalanced | PR-AUC | ROC-AUC, Weighted F1, MCC |
| Time Series | SMAPE | MASE, Coverage |
| Text Classification | Accuracy | F1, ROC-AUC |
| NER | Entity F1 | Per-class F1 |
| Clustering | Silhouette Score | Davies-Bouldin |
| Anomaly Detection | PR-AUC | ROC-AUC |

---

## 21. Inter-Agent Contract Tables

### 21.1 Complete Session State Flow

| Key | Written by | Read by |
|---|---|---|
| `run_id` | Orchestrator | Every agent, every tool call |
| `task_description` | Orchestrator | Feature Eng, Model Selection, Report |
| `pipeline_variant` | Orchestrator | Orchestrator routing |
| `file_type_result` | Ingestion | Orchestrator |
| `schema` | Ingestion | EDA, Cleaning, Feature Eng |
| `input_artefact_path` | Ingestion | EDA |
| `pii_detected` | Ingestion | Orchestrator (human gate trigger) |
| `eda_report_path` | EDA | Cleaning, Report |
| `eda_narrative_path` | EDA | Report |
| `chart_paths` | EDA | Report |
| `data_quality_flags` | EDA | Cleaning |
| `cleaned_dataset_path` | Cleaning | Feature Eng, Training, Evaluation |
| `transformation_log_path` | Cleaning | Report |
| `cleaning_summary` | Cleaning | Report |
| `content_hash` | Cleaning | Memory Agent (deduplication) |
| `features_train_path` | Feature Eng | Training |
| `features_test_path` | Feature Eng | Evaluation |
| `feature_manifest_path` | Feature Eng | Training, Evaluation, Explainability, Report, Deployment |
| `target_column` | Feature Eng | Training, Evaluation, Explainability |
| `task_type` | Feature Eng | Training (template selection), Evaluation (metrics selection) |
| `model_artefact_path` | Training (set), Tuning (updated) | Evaluation, Explainability, Deployment |
| `model_type` | Training | Evaluation, Explainability, Deployment |
| `onnx_model_path` | Training | Deployment |
| `baseline_metrics` | Training | Tuning, Report |
| `algorithm_comparison` | Training | Report |
| `best_params_path` | Tuning | Report |
| `tuning_metrics` | Tuning | Report |
| `evaluation_report_path` | Evaluation | Explainability, Report, Deployment |
| `deployment_recommendation` | Evaluation | `before_deploy` logic, Report |
| `fairness_metrics` | Evaluation | Report |
| `drift_report` | Evaluation | Report, Deployment |
| `shap_values_path` | Explainability | Report |
| `shap_plot_paths` | Explainability | Report |
| `explanation_narrative_path` | Explainability | Report |
| `report_md_path` | Report | Deployment, Memory Agent |
| `report_html_path` | Report | Delivered to user |
| `endpoint_url` | Deployment | User, Memory Agent |
| `smoke_test_results` | Deployment | Report appendix |
| `debug_attempts` | Ralph Loop | Orchestrator |
| `ralph_loop_iteration` | Ralph Loop | Orchestrator |
| `human_gate_decisions` | User (via REST) | Orchestrator |
| `escalation_report_{stage}` | Bug Log Agent | Orchestrator (human gate) |

### 21.2 Artefact Files Produced per Stage

| Stage | Output Files | Output Directory |
|---|---|---|
| Ingestion | `raw/{filename}` | `outputs/{run_id}/raw/` |
| EDA | `stats.json`, `eda_narrative.md`, `charts/*.png` | `outputs/{run_id}/eda/` |
| Cleaning | `cleaned_dataset.parquet`, `transformation_log.json`, `cleaning_summary.json` | `outputs/{run_id}/clean/` |
| Feature Eng | `features_train.parquet`, `features_test.parquet`, `feature_manifest.json`, `importance_chart.png` | `outputs/{run_id}/features/` |
| Training | `model.joblib` or `model_state_dict.pt`, `model.onnx`, `metrics/training.json`, `training_summary.md` | `outputs/{run_id}/training/` |
| Tuning | `tuned_model.*`, `best_params.json`, `tuning_report.json` | `outputs/{run_id}/tuning/` |
| Evaluation | `evaluation_report.json`, `fairness_metrics.json`, `drift_report.json`, `confusion_matrix.png` | `outputs/{run_id}/eval/` |
| Explainability | `shap_values.npy`, `shap_beeswarm.png`, `shap_bar.png`, `shap_waterfall.png`, `narrative.md` | `outputs/{run_id}/explain/` |
| Report | `final_report.md`, `final_report.html` | `outputs/{run_id}/report/` |
| Deployment | `app.py`, `requirements.txt`, `endpoint.json` | `outputs/{run_id}/deploy/` |
| Debug | `{bug_id}.json` | `outputs/{run_id}/debug/` |
| Escalations | `{stage_name}.md` | `outputs/{run_id}/escalations/` |

### 21.3 Human Gate Summary

| Gate | Trigger | Blocking | User Decision |
|---|---|---|---|
| File type confirmation | `supported=False` or unexpected type | Yes | Confirm file type or abort |
| PII detection | `pii_detected=True` | Yes | Approve processing or abort |
| EDA review | After EDA (optional) | No (auto-approve in dev) | Confirm cleaning approach |
| Outlier handling | Before cleaning transforms | No (auto-approve in dev) | Approve/override strategy |
| Fairness review | `recommend != "proceed"` | Yes (for "block") | Override or abort deployment |
| Deploy approval | Before Deployment Agent | No (auto-approve in dev) | Approve or defer |
| Ralph Loop exhaustion | `ralph_loop_iteration >= MAX_ITERATIONS` | Yes | Provide fix or abort |

---

## 22. Phase 1 — Environment and MCP Servers

### Goal

All three MCP servers are running, MCP-compliant, and verified. A hello-world agent call logs to the Tracking MCP Server, fires all middleware, and creates an OpenTelemetry trace. Magika correctly identifies five different file types. The Bug Log Agent records and enriches a synthetic bug. All unit tests pass.

### Steps

**Step 1.1 — Repository and environment**

Initialise Git repository. Add `.gitignore` covering `.env`, `__pycache__`, `.venv`, `*.pyc`, `outputs/`, `data/memory_index.json`, `data/memory_embeddings.npy`, `data/semantic_memory.json`, `data/procedural_memory.json`, `data/memory.db`, `data/tracking.db`, `data/buglog.db`. Create `pyproject.toml` with all dependencies pinned with `==`. Create `README.md` with running instructions.

Create Python 3.12 virtual environment. Install packages in the order in §3.1.

**Step 1.2 — Azure OpenAI verification**

Create `.env` from `.env.example`. Fill in all five Azure OpenAI variables. Verify access with a one-line completion test. If this fails, resolve before proceeding to any other step.

**Step 1.3 — Tracking MCP Server**

Implement `mcp_servers/tracking/server.py`. Define all 11 tools from §5.4. Create SQLite schema with WAL mode. Mount into FastAPI at `/mcp`. Add `/health` endpoint. Pass FastMCP lifespan to FastAPI app. Start server. Run all three compliance checks from §5.3. All must pass.

**Step 1.4 — DS Tools MCP Server (stubs)**

Implement `mcp_servers/ds_tools/server.py`. Define all 10 tools from §5.5 as stubs. Use `stateless_http=True`. Start on port 8101. Run compliance checks.

**Step 1.5 — Bug Log MCP Server**

Implement `mcp_servers/bug_log/server.py`. Define all 8 tools from §5.6. Create SQLite schema. Mount into FastAPI at `/mcp`. Start on port 8102. Run compliance checks.

**Step 1.6 — Middleware and hooks**

Implement all 5 middleware classes. Write unit tests for each (from §32.3). All must pass. Implement all hook functions in `hooks/hooks.py`. Write unit tests. All must pass.

**Step 1.7 — Hello-world integration**

Create a test agent with the full middleware chain, connected to all three MCP servers. Send `"What is 2 + 2?"`. Verify: response contains `4`, Tracking MCP Server has start/end records, Bug Log MCP Server is healthy, an OpenTelemetry span appears in console output.

**Step 1.8 — Magika initialisation test**

Implement `tools/file_type_detector.py` with the module-level singleton. Run 10 detection unit tests from §32.2. All must pass. Verify the singleton is initialised only once across multiple calls (add a counter to `get_magika()` for the test).

### Phase 1 Exit Gate

All three MCP servers pass compliance checks. Hello-world integration test passes. Magika singleton is verified. All middleware unit tests pass. All hook unit tests pass. CI passes lint and type checks.

---

## 23. Phase 2 — read_file, Ingestion, Tracking and Bug Log Agents

### Goal

The `read_file` tool correctly detects and reads all file type categories. The Ingestion Agent runs on five file types. The Artefact Tracking Agent records the ingestion artefact. The Bug Log Agent records and enriches a synthetic bug. All contract validations pass.

### Steps

**Step 2.1 — Reader implementations**

Implement one reader per file type in `tools/readers/`. Implementation order: CSV, Parquet, JSON, Excel, SQLite, PDF, DOCX, image, archive, ML artefacts. Write unit tests with real small sample files. Edge cases: empty files, encoding issues, malformed files.

**Step 2.2 — read_file tool (real implementation)**

Replace the stub in DS Tools MCP Server. Run integration tests for all 14 file type categories.

**Step 2.3 — Ingestion Agent**

Implement `agents/ingestion.py` with the system prompt from §15.3. Test with CSV, Parquet, PDF, Excel, and JSON. For each: verify correct `schema` and `file_type_result` in session state and artefact in Tracking MCP Server. Test routing mismatch.

**Step 2.4 — Artefact Tracking Agent**

Implement `memory/artefact_tracking_agent.py`. Run a pipeline that produces 3 artefacts. Verify all 3 are indexed by the Tracking Agent within 15 seconds. Verify lineage is correctly recorded.

**Step 2.5 — Bug Log Agent**

Implement `memory/bug_log_agent.py`. Inject a synthetic bug by calling `record_bug` directly on the Bug Log MCP Server. Verify the Bug Log Agent enriches it with recommended strategies from procedural memory and writes `outputs/{run_id}/debug/{bug_id}.json`.

**Step 2.6 — Contract validators**

Implement `tools/validators.py`. Write tests: missing `schema` key → `ContractViolationError`. Missing `input_artefact_path` → `ContractViolationError`. All keys present → passes.

### Phase 2 Exit Gate

`read_file` correctly detects all 14 file type categories. Ingestion Agent passes for all 5 file types. Artefact Tracking Agent indexes artefacts within 15 seconds. Bug Log Agent enriches synthetic bugs with repair strategy recommendations. Contract validators catch missing keys.

---

## 24. Phase 3 — EDA, Cleaning, Feature Engineering with Ralph Loop

### Goal

The first three analytical agents run end-to-end with the Ralph Loop verified: at least one injected failure per stage is detected by verification and repaired by the Debug Agent before the loop advances. All agents produce correct artefacts.

### Steps

**Step 3.1 — execute_code tool (real implementation)**

Replace the stub. Subprocess execution with configurable timeout. Implement `before_code_execute` and `after_code_execute` hooks inside the tool. Test: success case with output file, exception case, timeout case.

**Step 3.2 — EDA template and agent with Ralph Loop**

Write `templates/eda_template.py`. Implement `agents/eda.py`. Implement DONE_CRITERIA for EDA in `workflows/criteria.py`.

Ralph Loop injection test: after EDA code runs, manually corrupt `stats.json` (delete the `null_counts` key). Verify: verification fails, Ralph Loop increments iteration counter, Bug Log Agent records the failure, the next iteration's EDA code regenerates the correct `stats.json`, verification passes, loop exits.

**Step 3.3 — Cleaning Agent with Ralph Loop**

Write `templates/cleaning_template.py`. Implement `agents/cleaning.py`. Implement DONE_CRITERIA for cleaning.

Ralph Loop injection test: make the cleaning code produce a `cleaned_dataset.parquet` with nulls still present in an imputed column. Verify: verification fails (`null_count > 0` assertion), loop retries, Debug Agent diagnoses and fixes the imputation code, next iteration produces null-free output.

**Step 3.4 — Feature Engineering Agent with Ralph Loop**

Write `templates/feature_template.py`. Implement `agents/feature_engineering.py`. Implement DONE_CRITERIA for feature engineering.

Verify: DuckDuckGo research fires with domain-relevant query. Procedural memory `domain_features` is read before the research call. Feature manifest covers 100% of columns (verification assertion). `task_type` is set.

**Step 3.5 — Sub-pipeline integration with Ralph Loop**

Wire Ingestion → EDA → Cleaning → Feature Engineering as a sub-pipeline. Run on Titanic. Verify all checkpoints are in Tracking MCP Server. Run with `--resume-run-id` after a simulated crash at Cleaning. Verify it resumes from Cleaning, not from Ingestion.

### Phase 3 Exit Gate

Full Ingestion → Feature Engineering pipeline on Titanic produces all required artefacts. Ralph Loop correctly detects and repairs injected failures in EDA and Cleaning stages. DuckDuckGo research fires for feature engineering. Pipeline resumes from last checkpoint after simulated crash. Bug Log Agent records all failures and their resolutions.

---

## 25. Phase 4 — Training, Tuning, Evaluation

### Goal

Classical ML and at least one DL scenario run end-to-end with Ralph Loop verification. All metrics logged. Evaluation Agent produces complete report with fairness analysis. Procedural memory is updated with template performance data.

### Steps

**Step 4.1 — Training templates**

Write `templates/sklearn_train.py`. Write `templates/dl/bert_classify.py` satisfying all 15 checklist items from §20.2. Implement `before_model_train` validator. Verify both templates pass validation.

**Step 4.2 — Training Agent with Ralph Loop**

Implement `agents/model_selection.py`. Implement DONE_CRITERIA for training.

Run on Titanic: verify ≥ 3 algorithms tried, `model.onnx` produced, `metrics/training.json` non-empty, `model_artefact_path` in session state.

DL test with IMDB Sentiment: verify BERT template selected, training checklist passes validator, model artefact produced.

OOM injection test: set batch size to 1024 for BERT on IMDB. Verify Attempt 1 pattern match fires, batch size is halved, training succeeds.

After successful training: verify procedural memory `template_performance[task_type]` is updated with the achieved metric.

**Step 4.3 — Tuning Agent with Ralph Loop**

Implement `agents/hp_tuning.py`. Implement DONE_CRITERIA for tuning.

Verify: procedural memory `hyperparameter_priors` is read before search space definition. Optuna runs 50 trials (or hits 30-minute timeout). `best_params.json` produced. Tuned metric ≥ baseline (verification assertion).

**Step 4.4 — Evaluation Agent with fairness**

Implement `agents/evaluation.py`. Implement DONE_CRITERIA for evaluation.

Run on Titanic with `Sex` as protected attribute. Verify all required metric keys present. Verify `fairness_metrics.json` exists. Verify human gate fires when recommendation is not `"proceed"` (inject a fairness violation by skipping the imputation of a protected attribute column).

### Phase 4 Exit Gate

Classical ML pipeline achieves ROC-AUC > 0.83 on Titanic. BERT on IMDB runs without manual intervention. OOM auto-remediation fires and succeeds. Fairness analysis runs. Human gate fires on fairness violation. Procedural memory is updated after each successful stage.

---

## 26. Phase 5 — Explainability, Reporting, Deployment

### Goal

SHAP computation succeeds with Browser-Use error recovery verified. HTML report is self-contained and passes offline rendering check. Deployment smoke test passes 10/10. `before_deploy` hook correctly blocks deployment on bad recommendation.

### Steps

**Step 5.1 — Explainability Agent with Ralph Loop**

Implement `agents/explainability.py`. Implement DONE_CRITERIA for explainability.

Browser-Use trigger test: use a deprecated SHAP API call (valid in SHAP 0.40, not current). Verify: `execute_code` fails, Attempt 2 fires Browser-Use, SHAP documentation is crawled, correct API is found, repaired code runs successfully. Bug Log Agent records the failure and resolution.

**Step 5.2 — Report Agent**

Implement `agents/report.py` and report template. Implement DONE_CRITERIA for report (HTML offline check + section heading presence check).

Run on full Titanic pipeline. Open `final_report.html` offline. Verify all sections present. Verify no external URLs. Verify all charts embedded.

**Step 5.3 — Deployment Agent**

Implement `agents/deployment.py`. Implement DONE_CRITERIA for deployment (smoke test 10/10, endpoint health check).

Test `before_deploy` hook: set `deployment_recommendation.recommend = "block"`. Verify agent returns error without starting server.

Run normal deployment on Titanic model. Verify smoke test passes 10/10.

### Phase 5 Exit Gate

Browser-Use repairs injected SHAP API error on Attempt 2. HTML report passes offline check. Deployment smoke test 10/10. `before_deploy` blocks correctly. All artefacts tracked by Artefact Tracking Agent.

---

## 27. Phase 6 — Memory System and Retrospection

### Goal

All three memory tiers are populated after a complete Titanic pipeline run. Three retrospective queries return correct, grounded answers. Procedural memory is updated with bug resolution data. Memory Agent's episodic store contains entries for all 10 stages.

### Steps

**Step 6.1 — embed_text tool (real implementation)**

Implement in DS Tools MCP Server. SHA-256 caching verified. Test: two calls with identical content → one Azure OpenAI API call.

**Step 6.2 — Memory Agent — all three tiers**

Implement `memory/agent.py` with all three tier handlers. Implement `memory/indexer.py` for semantic index. Implement `memory/episodic.py` for the SQLite episodic store. Implement `memory/procedural.py` for the JSON procedural memory.

**Step 6.3 — Retrospective query tests**

After a complete Titanic run, verify:

Query 1: `"What was the highest ROC-AUC achieved?"` → correct value, cites evaluation report artefact path.

Query 2: `"What repair strategies successfully fixed CUDA OOM errors?"` → cites procedural memory entry with success rate.

Query 3: `"Which model artefact from the last run should I use for deployment?"` → returns tuned model path with metric justification.

**Step 6.4 — Procedural memory update verification**

After a successful bug resolution: verify `data/procedural_memory.json` has the winning strategy's success count incremented for the relevant `error_class:library_name` key. After a successful training run: verify `template_performance[task_type]` has a new entry.

### Phase 6 Exit Gate

All three memory tiers populated after Titanic run. All three retrospective queries return correct answers. Procedural memory is updated after bug resolutions. Episodic store has entries for all 10 stages. `embed_text` caching verified.

---

## 28. Phase 7 — Hardening and Production Gate

### Goal

25 safety injection tests blocked. Distributed traces complete in observability backend. All 5 benchmark datasets meet metric and runtime targets without human intervention. All documentation complete.

### Steps

**Step 7.1 — Safety penetration tests**

Write and run 25 injection tests. All must be blocked. Categories: shell injection via `execute_code` input, SQL injection, path traversal, environment variable access, `__import__` bypass, subprocess spawn attempts. Document in `docs/security_review.md`.

**Step 7.2 — Distributed trace verification**

Run a full Titanic pipeline. In the OpenTelemetry console exporter output (or Application Insights if configured), find the trace by `run_id`. Verify: spans for all 10 pipeline stages, all MCP tool calls as child spans, all Ralph Loop iterations as child spans of their respective stage spans, no orphaned spans.

**Step 7.3 — Benchmark dataset runs**

All with `PIPELINE_HUMAN_IN_THE_LOOP=false`:

Titanic (binary): ROC-AUC > 0.83, runtime < 20 min.

California Housing (regression): R² > 0.78, runtime < 20 min.

Credit Card Fraud (imbalanced): PR-AUC > 0.65, runtime < 25 min.

IMDB Sentiment (BERT): Accuracy > 0.91, runtime < 90 min.

Air Passengers (LSTM): SMAPE < 12.0, runtime < 45 min.

For each: verify no human intervention needed, all artefacts produced, Memory Agent indexes all artefacts, Ralph Loop fires and resolves at least one injected failure.

**Step 7.4 — Documentation**

Verify `docs/architecture.md` (Mermaid diagram), `docs/error_catalogue.md`, `docs/library_registry.md`, `docs/decision_log.md`, `docs/security_review.md` all exist and are current. Verify `README.md` covers: prerequisites, installation, running a pipeline, resuming after crash, querying memory, adding a new agent.

### Phase 7 Exit Gate

All 25 safety tests blocked. Distributed trace is complete. All 5 benchmarks meet targets. All Ralph Loop injections resolved automatically. All documentation current. CI passes all tests.

---

## 29. Project Directory Structure

```
ds-agent/
│
├── main.py                            # CLI: --file, --task, --resume-run-id
├── pyproject.toml                     # all deps pinned with ==
├── .env.example                       # all env var keys with comments
│
├── agents/
│   ├── clients.py                     # two AzureOpenAIChatCompletionClient singletons + TokenBudgetWrapper
│   ├── base.py                        # shared agent construction: client, middleware, MCPStreamableHTTPTools, providers
│   ├── orchestrator.py
│   ├── ingestion.py
│   ├── eda.py
│   ├── cleaning.py
│   ├── feature_engineering.py
│   ├── model_selection.py
│   ├── hp_tuning.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── report.py
│   ├── deployment.py
│   └── debug.py
│
├── middleware/
│   ├── chain.py                       # assembles middleware per agent type
│   ├── logging_mw.py
│   ├── safety_mw.py
│   ├── retry_mw.py
│   ├── telemetry_mw.py
│   └── rate_limit_mw.py
│
├── hooks/
│   └── hooks.py                       # before_code_execute, after_code_execute,
│                                      # before_model_train, before_deploy, after_deploy,
│                                      # on_code_error, on_max_debug_attempts
│
├── mcp_servers/
│   ├── tracking/
│   │   ├── server.py                  # FastMCP + FastAPI, port 8100, stateless_http=False
│   │   ├── tools.py                   # all 11 tracking tools
│   │   └── db.py                      # SQLite WAL schema
│   ├── ds_tools/
│   │   ├── server.py                  # FastMCP + FastAPI, port 8101, stateless_http=True
│   │   └── tools.py                   # all 10 DS tools
│   └── bug_log/
│       ├── server.py                  # FastMCP + FastAPI, port 8102, stateless_http=False
│       ├── tools.py                   # all 8 bug log tools
│       └── db.py                      # SQLite schema
│
├── tools/
│   ├── file_type_detector.py          # Magika singleton + 3-layer detection
│   ├── validators.py                  # before_model_train checklist, contract validators
│   ├── local_tools.py                 # @tool: get/set session state, check artefact exists
│   └── readers/
│       ├── csv_reader.py
│       ├── parquet_reader.py
│       ├── json_reader.py
│       ├── excel_reader.py
│       ├── sqlite_reader.py
│       ├── pdf_reader.py
│       ├── docx_reader.py
│       ├── image_reader.py
│       ├── archive_reader.py
│       └── model_reader.py            # pickle, ONNX, safetensors
│
├── templates/
│   ├── eda_template.py
│   ├── cleaning_template.py
│   ├── feature_template.py
│   ├── sklearn_train.py
│   ├── imbalanced.py
│   ├── sklearn_cluster.py
│   ├── anomaly.py
│   ├── eval_template.py
│   ├── deploy_template.py
│   └── dl/
│       ├── bert_classify.py
│       ├── bert_ner.py
│       ├── llm_finetune.py
│       ├── cnn_classify.py
│       ├── yolo_detect.py
│       ├── time_series.py
│       ├── tabnet.py
│       ├── rag.py
│       ├── whisper.py
│       ├── multimodal.py
│       └── llm_eval.py
│
├── workflows/
│   ├── pipeline_graph.py              # static graph per pipeline_variant
│   ├── routing.py                     # read_file result → pipeline_variant
│   ├── criteria.py                    # DONE_CRITERIA per stage (required_files, assertions)
│   ├── ralph_loop.py                  # core Ralph Loop implementation
│   └── checkpointing.py              # checkpoint read/write
│
├── memory/
│   ├── agent.py                       # Memory Agent: three-tier coordinator
│   ├── episodic.py                    # SQLite episodic store (data/memory.db)
│   ├── semantic.py                    # JSON + NumPy semantic index
│   ├── procedural.py                  # JSON procedural memory (fix strategies, etc.)
│   ├── artefact_tracking_agent.py     # Artefact Tracking Agent
│   └── bug_log_agent.py               # Bug Log Agent
│
├── config/
│   ├── settings.py                    # Pydantic BaseSettings
│   ├── safety_patterns.py             # versioned blocked pattern list
│   ├── library_registry.py            # library → docs URL
│   ├── error_catalogue.py             # error patterns → fix strategies
│   └── scenario_routing.py            # task_type → template file
│
├── data/                              # all persistent data (git-ignored)
│   ├── tracking.db
│   ├── buglog.db
│   ├── memory.db
│   ├── semantic_memory.json
│   ├── semantic_embeddings.npy
│   └── procedural_memory.json
│
├── outputs/                           # all pipeline run outputs (git-ignored)
│   └── {run_id}/
│       ├── raw/, eda/, clean/, features/
│       ├── training/, tuning/, eval/
│       ├── explain/, report/, deploy/
│       ├── metrics/, debug/, escalations/
│
├── tests/
│   ├── fixtures/                      # small CSV, PDF, ONNX, etc.
│   ├── unit/
│   │   ├── test_file_type_detector.py
│   │   ├── test_readers.py
│   │   ├── test_middleware.py
│   │   ├── test_hooks.py
│   │   ├── test_validators.py
│   │   ├── test_ralph_loop.py
│   │   └── test_mcp_compliance.py
│   ├── integration/
│   │   ├── test_tracking_mcp.py
│   │   ├── test_ds_tools_mcp.py
│   │   ├── test_bug_log_mcp.py
│   │   ├── test_ingestion_agent.py
│   │   ├── test_ralph_loop_recovery.py
│   │   ├── test_browser_use_trigger.py
│   │   ├── test_artefact_tracking_agent.py
│   │   ├── test_bug_log_agent.py
│   │   └── test_memory_agent.py
│   └── e2e/
│       ├── test_titanic.py
│       ├── test_housing.py
│       ├── test_fraud.py
│       ├── test_imdb.py
│       └── test_air_passengers.py
│
├── docs/
│   ├── architecture.md
│   ├── error_catalogue.md
│   ├── library_registry.md
│   ├── decision_log.md
│   └── security_review.md
│
└── .github/
    └── workflows/
        ├── ci.yml
        └── release.yml
```

---

## 30. Configuration Reference

All configuration is read from environment variables by `config/settings.py` using Pydantic `BaseSettings`. Keys must appear in `.env.example` with comments.

### Azure OpenAI (mandatory)

`AZURE_OPENAI_ENDPOINT` — resource endpoint. Format: `https://<resource>.openai.azure.com/`. Presence triggers Azure path.

`AZURE_OPENAI_PRIMARY_DEPLOYMENT` — deployment name for primary model.

`AZURE_OPENAI_FAST_DEPLOYMENT` — deployment name for fast model.

`AZURE_OPENAI_EMBEDDING_DEPLOYMENT` — deployment name for embedding model.

`AZURE_OPENAI_API_VERSION` — API version string (latest stable).

`AZURE_OPENAI_API_KEY` — development only. Remove in production.

### MCP Servers

`TRACKING_MCP_URL` — default `http://localhost:8100/mcp/mcp`.

`DS_TOOLS_MCP_URL` — default `http://localhost:8101/mcp/mcp`.

`BUG_LOG_MCP_URL` — default `http://localhost:8102/mcp/mcp`.

`TRACKING_DB_PATH` — default `./data/tracking.db`.

`BUG_LOG_DB_PATH` — default `./data/buglog.db`.

### Agent Configuration

`AGENT_PRIMARY_MAX_TOKENS` — integer, default 4096.

`AGENT_FAST_MAX_TOKENS` — integer, default 2048.

`AGENT_TEMPERATURE` — float, default 0.1.

`ORCHESTRATOR_HISTORY_MAX_MESSAGES` — integer, default 40.

`AGENT_HISTORY_MAX_MESSAGES` — integer, default 20.

### Ralph Loop Configuration

`RALPH_LOOP_MAX_ITERATIONS` — integer, default 8. Maximum loop iterations per stage before human gate.

`EXECUTE_CODE_TIMEOUT_SECONDS` — integer, default 120.

`MAX_DEBUG_ATTEMPTS` — integer, default 5. Debug Agent repair attempts within one Ralph Loop iteration.

### Pipeline Configuration

`PIPELINE_HUMAN_IN_THE_LOOP` — boolean, default `true`. Set `false` for automated testing.

`PIPELINE_TOKEN_BUDGET` — integer, default 500000.

`OUTPUT_BASE_DIR` — string, default `./outputs`.

`DATA_DIR` — string, default `./data`.

### Memory Configuration

`EPISODIC_DB_PATH` — default `./data/memory.db`.

`SEMANTIC_INDEX_PATH` — default `./data/semantic_memory.json`.

`SEMANTIC_EMBEDDINGS_PATH` — default `./data/semantic_embeddings.npy`.

`PROCEDURAL_MEMORY_PATH` — default `./data/procedural_memory.json`.

`MEMORY_AGENT_POLL_INTERVAL_SECONDS` — integer, default 15.

`ARTEFACT_TRACKING_POLL_INTERVAL_SECONDS` — integer, default 10.

`BUG_LOG_POLL_INTERVAL_SECONDS` — integer, default 5.

### Research Tools

`BROWSER_USE_MAX_STEPS` — integer, default 50.

`BROWSER_USE_HEADLESS` — boolean, default `true`.

`DUCKDUCKGO_MAX_RESULTS` — integer, default 10.

`DUCKDUCKGO_RATE_LIMIT_PER_HOUR` — integer, default 30.

### Deployment

`DEPLOYMENT_SERVER_PORT` — integer, default 8200.

`DEPLOYMENT_SMOKE_TEST_COUNT` — integer, default 10.

`DEPLOYMENT_LATENCY_P99_MS_ADVISORY` — integer, default 500.

---

## 31. Safety Patterns and Responsible AI

### 31.1 Blocked Pattern List (config/safety_patterns.py)

The `SafetyMiddleware` checks tool call inputs against this versioned list. Initial mandatory entries:

File system manipulation outside outputs: `rm -rf`, `rmdir`, `shutil.rmtree`, `os.remove` (when path doesn't start with output directory), `os.unlink`.

Environment variable access: `os.environ`, `os.getenv`, `os.putenv`, `dotenv.load_dotenv`.

Subprocess spawning outside `execute_code` tool: `subprocess.Popen`, `subprocess.call`, `subprocess.check_output`, `os.system`, `os.popen`.

Network access from generated code: `requests.get`, `requests.post`, `urllib.request.urlopen`, `httpx.get`, `aiohttp.ClientSession` (in code submitted to `execute_code` — the tool itself uses network access; generated code should not).

Dangerous module imports in generated code: `import socket`, `import subprocess`, `from subprocess`, `import ctypes`.

Review and extend before each production release. Document reviews in `docs/security_review.md`.

### 31.2 Responsible AI Obligations

Fairlearn analysis is mandatory for every classification model. No classification model is deployed without `fairness_metrics.json`. This is enforced by the Evaluation Agent's processing protocol and the `before_deploy` check.

`before_deploy` blocks deployment when `deployment_recommendation.recommend == "block"`. Code-enforced. The Deployment Agent's system prompt explicitly states it must not proceed.

Every deployed endpoint exposes `/model_card`. The model card is extracted from `final_report.md` by the Deployment Agent. It must include: training data description, known limitations, intended use cases, fairness analysis summary, full performance metrics.

SHAP explanations are produced for every deployed model. The `before_deploy` check also verifies that `shap_values_path` is set in session state.

### 31.3 PII Detection

After `read_file` returns a tabular result, the Ingestion Agent scans the schema for PII-indicative column names: `name`, `email`, `phone`, `ssn`, `dob`, `address`, `ip_address`, `user_id`, and domain equivalents. It also checks a sample of values for email format, phone format, and US SSN format. PII detection triggers a human gate — never proceed without explicit user approval.

---

## 32. Testing Strategy

### 32.1 Three-Tier Philosophy

Unit tests — single function, all dependencies mocked, run in milliseconds. No real network calls. Any unit test taking more than 1 second is wrong — fix it.

Integration tests — two or more components, real MCP servers, recorded Azure OpenAI responses, real file I/O.

End-to-end tests — full pipeline, real Azure OpenAI calls. Run only on PRs to `main` and on release.

### 32.2 FileTypeDetector Unit Tests

Test 1: CSV with `.csv` → `detected_type="csv"`, `confidence > 0.9`.
Test 2: Parquet with `.parquet` → `detected_type="parquet"`.
Test 3: CSV renamed as `.json` → `detected_type="csv"` (extension mismatch test).
Test 4: PDF → `detected_type="pdf"`.
Test 5: PyTorch pickle → `detected_type="pickle"`.
Test 6: ZIP archive → `detected_type="zip"`, `member_count > 0`.
Test 7: Empty file → `supported=False`, no exception.
Test 8: Corrupted magic bytes → graceful fallback, no crash.
Test 9: Scanned PDF → `requires_ocr=True`.
Test 10: Text file containing JSON → `detected_type="json"`, not `"txt"`.

### 32.3 Middleware Unit Tests

**LoggingMiddleware**: after one call, exactly one JSONL entry in captured stdout with all required fields populated.

**SafetyMiddleware**: 10 blocked patterns — each raises `SafetyViolationError`. 5 clean inputs sharing blocked keywords but not matching — each passes unchanged.

**RetryMiddleware**: HTTP 429 twice then 200 → exactly 2 retries, success returned, backoff delay measured. HTTP 400 → zero retries.

**TelemetryMiddleware**: after one call, one span created and finished with required attributes. Use in-memory span exporter.

**RateLimitMiddleware**: 2-token bucket, 1 token/sec refill, three calls → third call delayed 0.8–1.5 seconds.

### 32.4 Ralph Loop Unit Tests

Test 1: successful stage on first iteration → loop exits with SUCCESS, checkpoint written.

Test 2: verification fails once then passes → loop runs exactly 2 iterations, Bug Log Agent receives 1 bug record.

Test 3: verification fails `MAX_ITERATIONS` times → loop exits with ESCALATION, human gate triggered, escalation report written.

Test 4: completion promise tag missing → Bug Log Agent receives `missing_completion_tag` event, loop continues.

Test 5: fresh context on retry → new session created for each iteration (verify `InMemoryHistoryProvider` instance is new).

### 32.5 End-to-End Test Structure

Every e2e test must follow this structure:

1. Start all three MCP servers. Verify all `/health` endpoints.
2. Copy fixture file to temp directory.
3. Run `python main.py --file {tmp} --task "{task}" --pipeline_human_in_the_loop=false`. Capture `run_id`.
4. After completion, call `query_run_status`. Assert all stages `status="success"`.
5. For each expected artefact type, verify existence via `query_artefact_history`.
6. Read `final_report.html`. Assert all section headings present.
7. POST to `/predict` 3 times. Assert all HTTP 200 with correct schema.
8. Query Memory Agent: `"show me the metrics from the last run"`. Assert correct `run_id` cited.
9. Query Bug Log MCP Server: assert all bugs for this run are marked resolved.
10. Teardown: stop deployed endpoint subprocess, delete temp directory.

### 32.6 Minimum Acceptable Metrics

| Dataset | Task | Primary Metric | Minimum |
|---|---|---|---|
| Titanic | Binary Classification | ROC-AUC | 0.83 |
| California Housing | Regression | R² | 0.78 |
| Credit Card Fraud | Imbalanced Classification | PR-AUC | 0.65 |
| IMDB Sentiment | Text Classification | Accuracy | 0.91 |
| Air Passengers | Time Series | SMAPE | < 12.0 |

---

## 33. CI/CD and Operations

### 33.1 CI Workflow (every push, target < 8 minutes)

Step 1 — Lint (60s): `ruff check .` and `ruff format --check .`. `mypy --strict`. Fail fast.

Step 2 — Unit tests (3 min): `pytest tests/unit/ -x --timeout=30`. No real network calls.

Step 3 — MCP compliance (2 min): start all three MCP servers. Run `test_mcp_compliance.py`. All three servers, three checks each.

### 33.2 PR to Main (target < 30 minutes)

Steps 1–3 plus:

Step 4 — Integration tests (10 min): recorded Azure OpenAI responses. Real MCP servers. Real file I/O.

Step 5 — Ralph Loop recovery tests (5 min): inject 5 known error patterns, verify auto-repair.

Step 6 — Security scan (3 min): `bandit -r agents/ tools/ mcp_servers/ -ll`. `safety check --json`. HIGH/CRITICAL fails build.

### 33.3 Starting the System

Start in this order. Verify each `/health` before starting the next:

```
uvicorn mcp_servers.tracking.server:app --port 8100
uvicorn mcp_servers.ds_tools.server:app --port 8101
uvicorn mcp_servers.bug_log.server:app --port 8102
python main.py --file tests/fixtures/titanic.csv \
  --task "Predict passenger survival. Binary classification." \
  --pipeline_human_in_the_loop=false
```

### 33.4 Resuming After a Crash

`python main.py --file {same_file} --task "{same_task}" --resume-run-id {run_id_from_stdout}`

### 33.5 Querying Memory

`python -c "from memory.agent import query_memory; print(query_memory('your question here'))"`

Returns: `{"answer": str, "sources": list[dict], "confidence": float}`.

---

## 34. Error Catalogue and Auto-Remediation Map

Maintained in `config/error_catalogue.py`. Loaded by the Bug Log Agent and Debug Agent at startup. Every entry has `error_class`, `pattern`, `library`, `fix_strategy`, `historical_success_rate` (updated by the Bug Log Agent after each resolution).

| Error Class | Pattern | Library | Fix Strategy |
|---|---|---|---|
| `ModuleNotFoundError` | `No module named 'X'` | any | Add correct `import X` at top of code |
| `KeyError` | column name | pandas | Look up correct column name in `feature_manifest.json` |
| `ValueError` | `could not convert string to float` | pandas/numpy | Add `.astype(float, errors='coerce')` |
| `ValueError` | `Input contains NaN` | sklearn | Add `SimpleImputer(strategy='median')` before failing step |
| `TypeError` | `unexpected keyword argument` | any | Remove the unknown kwarg; API has changed |
| `AttributeError` | `has no attribute` | any | Method renamed; trigger Browser-Use Attempt 2 |
| `RuntimeError` | `CUDA out of memory` | torch | Halve `batch_size`; if still OOM enable gradient checkpointing; if still OOM set `device='cpu'` |
| `RuntimeError` | `Expected all tensors on same device` | torch | Add `.to(device)` to all tensors and model |
| loss is NaN | `torch.isnan(loss)` | torch | Check inputs for NaN; reduce LR by 10x |
| `FileNotFoundError` | any | any | Verify path from session state; substitute correct path |
| `MemoryError` | any | pandas | Switch to `pd.read_csv(chunksize=10000)` |
| `TimeoutExpired` | any | subprocess | Add early stopping with shorter patience; sample dataset |
| `ImportError` | `cannot import name` | any | Symbol moved; trigger Browser-Use Attempt 2 |
| DataLoader hang | `num_workers` | torch | Set `num_workers=0` |
| SHAP TypeError | API mismatch | shap | Trigger Browser-Use on Attempt 2 for SHAP API compatibility |
| `NCCL error` | distributed training | torch | Set `nproc_per_node=1` |

---

## 35. Known Limitations and Roadmap

### 35.1 Known Limitations

**Subprocess execution, not Docker isolation** — generated code runs in a subprocess with safety pattern filtering. The safety patterns mitigate most risks, but Docker isolation would be stronger. Docker is excluded to reduce operational complexity in v1.

**Single concurrent pipeline** — one run at a time. SQLite and in-memory Memory Agent index are not designed for concurrent writes. Migration to PostgreSQL + thread-safe index is the scaling path.

**Memory Agent index resets on restart** — the in-process semantic and procedural memory is loaded from disk on startup. There is a brief window after writing an artefact and before flushing the index where a crash could cause an index entry to be missing. Mitigate: on startup, re-index any artefacts in the Tracking MCP Server not already in the semantic index.

**No streaming progress to external clients** — JSONL stdout only. No WebSocket/SSE endpoint for real-time web UI.

**Semantic memory scales to ~50,000 artefacts** — pure NumPy cosine similarity is O(n). Beyond 50,000 records, query latency exceeds 500ms. Migrate to Chroma or Qdrant at that scale.

**Ralph Loop MAX_ITERATIONS is a blunt instrument** — 8 iterations per stage is the default. Some errors (particularly in DL training with novel architectures) may need more. Make this per-stage configurable in `workflows/criteria.py`.

### 35.2 Prioritised Roadmap

1. **Docker isolation for subprocess execution** — wrap `execute_code` in Docker. Stronger safety guarantee than pattern matching alone. Estimated effort: 1 week.

2. **Concurrent pipeline support** — PostgreSQL Tracking Server, thread-safe Memory Agent, run-scoped file namespacing. Estimated effort: 2 weeks.

3. **Real-time web UI** — FastAPI WebSocket endpoint on Tracking Server, React frontend for pipeline progress, Ralph Loop status, human gate approval. Estimated effort: 3 weeks.

4. **Vector database for semantic memory** — migrate from NumPy to Chroma when index exceeds 50,000 records. Chroma runs in-process, no additional service. Estimated effort: 2 days.

5. **A2A protocol** — implement A2A so agents are discoverable by LangGraph, CrewAI, and Google ADK. Estimated effort: 1 week after A2A spec stabilises.

6. **Drift monitoring** — scheduled daily drift check against training baseline. Estimated effort: 1 week.

---

## 36. Glossary

**Artefact** — any file produced by a pipeline stage. Always accessed by path. Never stored in session state as content.

**Artefact Tracking Agent** — dedicated observer agent that records every artefact produced, its lineage, its content hash, and its metadata. One of three observer agents.

**Bug Log Agent** — dedicated observer agent that records every error, enriches it with historical repair strategies, tracks repair attempts, and updates procedural memory on resolution. One of three observer agents.

**Completion promise** — the exact tag an agent must output verbatim in its final message (e.g., `<DONE>eda</DONE>`) to signal it believes the stage is complete. The Ralph Loop checks for this tag before running verification.

**DONE_CRITERIA** — the per-stage verification specification in `workflows/criteria.py`. Contains `required_files`, `required_session_keys`, and `assertions`. The Ralph Loop runs verification against these after every completion promise tag is detected.

**Episodic memory** — Tier 2 memory. Records what happened in each pipeline run as structured episode records. Persists in SQLite (`data/memory.db`). Queried by agents for past execution context.

**Fresh context** — the Ralph Loop pattern of creating a new agent session for each retry iteration, with bug context and verification failures injected as system context rather than accumulated in conversation history.

**Human gate** — a workflow pause that requires explicit user approval before the pipeline continues. Triggered by: unsupported file types, PII detection, fairness violations, Ralph Loop exhaustion.

**MCPStreamableHTTPTool** — the Framework class managing client-side MCP sessions over Streamable HTTP. Created once per MCP server per agent at construction time.

**Memory Agent** — the three-tier memory coordinator. Manages working memory, episodic memory, semantic memory, and procedural memory. Runs as an async observer. One of three observer agents.

**Procedural memory** — Tier 3 memory. Stores how to do things: successful error repair strategies sorted by historical success rate, template performance by task type, hyperparameter priors by model type, domain feature ideas by keyword. Persists in `data/procedural_memory.json`.

**Ralph Loop** — the self-correcting execution wrapper that runs every pipeline stage in a loop until verification criteria pass or MAX_ITERATIONS is reached. Named after Ralph Wiggum from The Simpsons. The core execution pattern of this system.

**Ralph Loop iteration** — one pass through the loop: agent run → completion tag check → verification → feedback injection (if failed) → repeat.

**run_id** — a UUID generated by the Orchestrator at the start of every pipeline run. The primary correlation key for all Tracking MCP Server records, Bug Log MCP Server records, artefact files, and OpenTelemetry traces.

**Semantic memory** — Tier 2 long-term memory. Stores all artefact records indexed by embedding vectors. Persists in `data/semantic_memory.json` + `data/semantic_embeddings.npy`. Queried by cosine similarity.

**Session state** — the key-value store persisted within an agent session. Used to pass artefact paths and metadata between tool calls and between parent/child agents when `propagate_session=True`.

**Tracking MCP Server** — the MCP server (port 8100) recording all pipeline events, artefacts, metrics, checkpoints. The coordination backbone. Backed by SQLite WAL.

**Working memory** — Tier 1 memory. The current session state plus the rolling chat history from `InMemoryHistoryProvider`. Scoped to one agent session. Does not persist across application restarts.

---

## 37. Reference Documents

Read the relevant reference before implementing each section. When this document conflicts with an official source, the official source takes precedence — update this document and record the change in `docs/decision_log.md`.

**Microsoft Agent Framework**

`learn.microsoft.com/en-us/agent-framework/overview/` — overview and quickstart.

`learn.microsoft.com/en-us/agent-framework/support/upgrade/python-2026-significant-changes` — all 1.0 breaking changes. Critical reading before writing any import. Specifically: tool context parameter changed from `**kwargs` to `FunctionInvocationContext`; `ChatMiddleware` now runs per LLM call not once per invocation; `as_tool(propagate_session=True)` is the correct sub-agent pattern.

`learn.microsoft.com/en-us/agent-framework/agents/agent-pipeline` — full middleware and context provider architecture.

`learn.microsoft.com/en-us/agent-framework/agents/conversations/context-providers` — context providers and dynamic middleware injection.

`learn.microsoft.com/en-us/python/api/agent-framework-core/agent_framework.agentmiddleware` — `AgentMiddleware` API with code example.

`learn.microsoft.com/en-us/python/api/agent-framework-core/agent_framework.mcpstreamablehttptool` — `MCPStreamableHTTPTool` API.

`devblogs.microsoft.com/agent-framework/chat-history-storage-patterns-in-microsoft-agent-framework/` — `InMemoryHistoryProvider` usage patterns.

`github.com/microsoft/agent-framework/issues/5317` — known `MCPStreamableHTTPTool` legacy GET issue. Verify your version has the fix.

**MCP Protocol and FastMCP**

`modelcontextprotocol.io/specification` — MCP protocol 2025-06-18.

`gofastmcp.com/deployment/http` — FastMCP HTTP deployment. Read the lifespan and CORS sections.

`github.com/modelcontextprotocol/python-sdk/issues/713` — multiple FastMCP server lifespan management in one FastAPI app.

`github.com/modelcontextprotocol/python-sdk/issues/880` — `StreamableHTTPSessionManager` horizontal scaling limitation (context for single-node design decision).

`github.com/modelcontextprotocol/python-sdk/issues/1180` — stateless_http and session management in Kubernetes.

**Ralph Loop**

`ice-ice-bear.github.io/posts/2026-03-06-ralph-loop-ai-automation/` — canonical analysis of the Ralph Loop pattern and its economic viability.

`beuke.org/ralph-wiggum-loop/` — formal definition of the Ralph Wiggum Loop pattern.

`ralphify.co/docs/how-it-works/` — each iteration: re-read criteria, run commands, replace placeholders, pipe to agent, loop. The lifecycle model this system implements.

`awesomeclaude.ai/ralph-wiggum` — completion promise pattern and `--max-iterations` as primary safety net.

**Memory**

`atlan.com/know/episodic-memory-ai-agents/` — episodic memory architecture, three-tier model, enterprise patterns.

`47billion.com/blog/ai-agent-memory-types-implementation-best-practices/` — episodic/semantic/procedural memory types and 2026 best practices.

`www.sitepoint.com/ai-agent-memory-guide/` — working memory, episodic, semantic — complete 2026 guide.

**Browser-Use**

`docs.browser-use.com/llms-full.txt` — full Browser-Use SDK reference. Read this, not individual pages.

`docs.browser-use.com/customize/supported-models` — confirms `ChatAzureOpenAI` support.

**File Type Detection**

`github.com/google/magika` — Magika documentation, 200+ content types, ~99% accuracy.

**Explainability and Fairness**

`shap.readthedocs.io` — SHAP documentation. The explainer selection guide is critical.

`fairlearn.org/main/user_guide/index.html` — Fairlearn user guide. Read `MetricFrame` section.

---

*This is the complete implementation specification for the DS Agent project. Total sections: 37. Build phases: 7. All agents: 15 (10 pipeline agents + 3 observer agents + Orchestrator + Debug Agent). No agent, tool, template, or middleware class should be implemented without first reading the governing section. All architectural decisions are recorded in `docs/decision_log.md`. The Ralph Loop is non-negotiable — every stage is wrapped. The three-tier memory system is non-negotiable — episodic, semantic, and procedural stores must all be populated and queryable before Phase 7 is signed off.*

---

## 38. Stage Verification — Complete DONE_CRITERIA Specifications

### 38.1 Design Rules for Criteria

Every DONE_CRITERIA entry in `workflows/criteria.py` has three components:

`required_files` — paths relative to `outputs/{run_id}/`. The Ralph Loop checks `os.path.exists(path)` and `os.path.getsize(path) > 0`. Both must be true.

`required_session_keys` — session state keys that must be present and not None/empty. The Ralph Loop reads these from the shared session state object.

`assertions` — Python callables with signature `(session_state: dict, output_dir: str) -> tuple[bool, str]`. Return `(True, "")` on pass. Return `(False, "descriptive failure message including expected vs actual values")` on fail. Assertions must run in under 1 second — no LLM calls, no network calls.

Assertions are the most important component. They encode domain correctness. Write them to be specific: instead of "model file exists" (which is covered by `required_files`), write "tuned metric is at least as good as baseline metric" or "null count in imputed columns is exactly 0".

### 38.2 Ingestion Stage Criteria

**required_files:**
- `raw/{original_filename}` (the raw input file copied to the outputs directory)

**required_session_keys:**
- `file_type_result` — must be a non-empty dict
- `schema` — must be a dict with at least one key
- `input_artefact_path` — must be a non-empty string

**assertions:**
- `schema_has_columns`: `len(session_state["schema"]) > 0`. Failure: `"Schema is empty — file has no detectable columns."`
- `file_type_supported`: `session_state["file_type_result"].get("supported", False) is True`. Failure: `"File type is not supported. Detected type: {session_state['file_type_result'].get('detected_type', 'unknown')}."`
- `no_routing_mismatch`: `session_state.get("routing_mismatch") is not True`. Failure: `"Routing mismatch detected. File type {file_type} is incompatible with task type {task_description}."`

### 38.3 EDA Stage Criteria

**required_files:**
- `eda/stats.json`
- `eda/eda_narrative.md`
- `eda/charts/correlation_heatmap.png`

**required_session_keys:**
- `eda_report_path`, `eda_narrative_path`, `chart_paths`, `data_quality_flags`

**assertions:**
- `stats_json_has_required_keys`: open `stats.json`, parse JSON, check keys `describe`, `null_counts`, `null_percentages`, `correlation_matrix`, `data_quality_flags` all present. Failure: `"stats.json missing keys: {missing_keys}."`
- `narrative_is_substantive`: `os.path.getsize(narrative_path) >= 500`. Failure: `"EDA narrative is too short ({size} bytes). Minimum 500 bytes expected."`
- `at_least_three_charts`: `len(session_state["chart_paths"]) >= 3`. Failure: `"Only {n} charts produced. Minimum 3 required."`
- `data_quality_flags_is_list`: `isinstance(session_state["data_quality_flags"], list)`. Failure: `"data_quality_flags must be a list. Got {type}."`

### 38.4 Cleaning Stage Criteria

**required_files:**
- `clean/cleaned_dataset.parquet`
- `clean/transformation_log.json`
- `clean/cleaning_summary.json`

**required_session_keys:**
- `cleaned_dataset_path`, `transformation_log_path`, `cleaning_summary`, `content_hash`

**assertions:**
- `no_nulls_in_imputed_columns`: read `cleaned_dataset.parquet` with `pyarrow`. For every column marked as `imputed` in `transformation_log.json`, check that null count equals 0. Failure: `"Column {col} still has {n} nulls after imputation."`
- `row_count_documented`: open `cleaning_summary.json`, check that `rows_before` and `rows_after` are both present and `rows_after > 0`. Failure: `"cleaning_summary.json missing rows_before or rows_after, or rows_after is 0."`
- `transformation_log_has_justifications`: open `transformation_log.json`, parse as list. For every entry, check that the `justification` key is present and non-empty string. Failure: `"transformation_log entry for column {col} missing justification field."`
- `no_columns_silently_dropped`: open `cleaning_summary.json`. For every column in `columns_dropped`, verify a matching entry exists in `transformation_log.json` with `transformation_type == 'drop_column'`. Failure: `"Column {col} was dropped without a transformation log entry."`

### 38.5 Feature Engineering Stage Criteria

**required_files:**
- `features/features_train.parquet`
- `features/features_test.parquet`
- `features/feature_manifest.json`
- `features/importance_chart.png`

**required_session_keys:**
- `features_train_path`, `features_test_path`, `feature_manifest_path`, `target_column`, `task_type`

**assertions:**
- `manifest_covers_all_columns`: open `feature_manifest.json`. Extract all `source_column` values (flatten lists). Compare against the schema columns from session state. Every schema column must appear. Failure: `"feature_manifest.json missing coverage for columns: {missing}."`
- `no_nulls_in_features`: read `features_train.parquet` with `pyarrow`. `null_count() == 0` for all columns. Failure: `"features_train.parquet has {n} nulls in column {col}."`
- `train_test_no_overlap`: read both parquets. Compare index or a stable identifier column. Verify zero overlap. Failure: `"features_train and features_test share {n} rows — data leakage."`
- `task_type_is_valid`: check `session_state["task_type"]` against the valid scenario list from `config/scenario_routing.py`. Failure: `"task_type '{task_type}' is not a valid scenario label."`
- `target_column_in_features`: verify `session_state["target_column"]` is a column in `features_train.parquet`. Failure: `"Target column '{col}' not found in features_train.parquet."`

### 38.6 Training Stage Criteria

**required_files:**
- `training/model.joblib` OR `training/model_state_dict.pt` (at least one must exist)
- `training/model.onnx`
- `training/metrics/training.json`
- `training/training_summary.md`

**required_session_keys:**
- `model_artefact_path`, `model_type`, `onnx_model_path`, `baseline_metrics`, `algorithm_comparison`

**assertions:**
- `model_file_exists`: `os.path.exists(session_state["model_artefact_path"])`. Failure: `"model_artefact_path {path} does not exist."`
- `onnx_file_exists`: `os.path.exists(session_state["onnx_model_path"])`. Failure: `"ONNX model not found at {path}."`
- `metrics_has_primary_metric`: open `training/metrics/training.json`. Get the primary metric key for `task_type` from `config/scenario_routing.py`. Verify it is present and numeric. Failure: `"Primary metric key '{key}' missing from training metrics."`
- `algorithm_comparison_has_entries`: `len(session_state["algorithm_comparison"]) >= 1`. Failure: `"algorithm_comparison is empty."`
- `training_summary_substantive`: `os.path.getsize("training/training_summary.md") >= 100`. Failure: `"training_summary.md too short."`

### 38.7 Tuning Stage Criteria

**required_files:**
- The tuned model file (either `.joblib` or `.pt` depending on model_type, at a path under `tuning/`)
- `tuning/best_params.json`
- `tuning/tuning_report.json`

**required_session_keys:**
- `model_artefact_path` (updated to tuned model), `best_params_path`, `tuning_metrics`

**assertions:**
- `tuned_model_path_updated`: verify `session_state["model_artefact_path"]` now points to a file under `tuning/`, not `training/`. Failure: `"model_artefact_path was not updated to the tuned model. Still pointing to training artefact."`
- `tuning_improves_baseline`: get primary metric key from task type. Compare `tuning_metrics[key]` with `baseline_metrics[key]`. For metrics where higher is better (AUC, R², F1, Accuracy), verify `tuning >= baseline`. For metrics where lower is better (RMSE, SMAPE), verify `tuning <= baseline`. Failure: `"Tuned metric {tuned_val} is worse than baseline {baseline_val} for metric {key}."`
- `best_params_is_valid_json`: open `best_params.json`, parse JSON, verify it is a non-empty dict. Failure: `"best_params.json is empty or invalid JSON."`

### 38.8 Evaluation Stage Criteria

**required_files:**
- `eval/evaluation_report.json`
- `eval/fairness_metrics.json` (mandatory for classification tasks)
- `eval/drift_report.json`

**required_session_keys:**
- `evaluation_report_path`, `deployment_recommendation`, `fairness_metrics`, `drift_report`

**assertions:**
- `evaluation_report_has_all_metrics`: open `evaluation_report.json`. Get required metric keys for task type. Verify all are present and numeric. Failure: `"evaluation_report.json missing metrics: {missing}."`
- `deployment_recommendation_has_all_keys`: verify `deployment_recommendation` dict has keys `recommend`, `blocking_issues`, `advisory_issues`. Failure: `"deployment_recommendation missing key: {missing_key}."`
- `recommend_is_valid_value`: verify `deployment_recommendation["recommend"]` is one of `proceed`, `conditional`, `block`. Failure: `"deployment_recommendation.recommend has invalid value: '{value}'."`
- `fairness_metrics_present_for_classification`: if `task_type` is any classification variant, verify `eval/fairness_metrics.json` exists and is non-empty. Failure: `"Fairness analysis is mandatory for classification tasks. fairness_metrics.json is missing or empty."`

### 38.9 Explainability Stage Criteria

**required_files:**
- `explain/shap_values.npy`
- `explain/shap_beeswarm.png`
- `explain/shap_bar.png`
- `explain/narrative.md`

**required_session_keys:**
- `shap_values_path`, `shap_plot_paths`, `explanation_narrative_path`

**assertions:**
- `shap_values_non_empty`: load `shap_values.npy` with `numpy.load`. Verify `arr.size > 0`. Failure: `"shap_values.npy is empty."`
- `at_least_two_plots`: `len(session_state["shap_plot_paths"]) >= 2`. Failure: `"Only {n} SHAP plots produced. Minimum 2 required (beeswarm + bar)."`
- `narrative_substantive`: `os.path.getsize(narrative_path) >= 100`. Failure: `"SHAP narrative is too short ({size} bytes)."`

### 38.10 Report Stage Criteria

**required_files:**
- `report/final_report.md`
- `report/final_report.html`

**required_session_keys:**
- `report_md_path`, `report_html_path`

**assertions:**
- `html_has_all_sections`: open `final_report.html`, read as text. Check for the presence of these substrings (section heading markers): `"Executive Summary"`, `"Dataset Overview"`, `"Exploratory Data Analysis"`, `"Data Quality"`, `"Feature Engineering"`, `"Model Comparison"`, `"Best Model"`, `"Fairness Analysis"`, `"Explainability"`, `"Deployment Recommendation"`, `"Appendix"`. Failure: `"final_report.html missing sections: {missing}."`
- `html_no_external_urls`: open `final_report.html`, check that no `http://` or `https://` URL appears in `src=` or `href=` attributes (base64 embedded content starts with `data:`). Failure: `"final_report.html contains external URL reference: {url}. Report must be self-contained."`
- `html_has_base64_charts`: verify at least 3 occurrences of `src="data:image/` in the HTML file. Failure: `"HTML report has only {n} inline base64 images. Expected ≥ 3."`
- `markdown_non_empty`: `os.path.getsize(report_md_path) >= 1000`. Failure: `"final_report.md too short ({size} bytes)."`

### 38.11 Deployment Stage Criteria

**required_files:**
- `deploy/app.py`
- `deploy/endpoint.json`

**required_session_keys:**
- `endpoint_url`, `smoke_test_results`

**assertions:**
- `endpoint_url_set`: `len(session_state.get("endpoint_url", "")) > 0`. Failure: `"endpoint_url is empty — deployment did not complete."`
- `smoke_test_all_pass`: `session_state["smoke_test_results"].get("passed_count", 0) == session_state["smoke_test_results"].get("total_count", 10)`. Failure: `"Smoke test: {passed}/{total} predictions succeeded. All 10 must pass."`
- `endpoint_health_check`: make a GET request to `session_state["endpoint_url"] + "/health"`. Verify HTTP 200. Failure: `"Deployed endpoint /health returned {status_code}, expected 200."`
- `endpoint_json_has_url`: open `deploy/endpoint.json`, verify `url` key is present. Failure: `"deploy/endpoint.json missing 'url' key."`

---

## 39. Template Authoring Guide

### 39.1 What a Template Is

A template is a Python source file in `templates/`. It is a parameterised code skeleton that an agent fills in and submits via `execute_code`. Templates are never executed by the agent process directly. They run in a subprocess with a configurable timeout.

Templates are not Python f-strings generated at runtime. They are files that live on disk, are version-controlled, and are updated independently of the agents. An agent reads the template file (or has it in its context via the system prompt), substitutes the placeholders, and submits the result via `execute_code`.

### 39.2 Template Header Format

Every template must begin with a comment block that documents the template's contract. The comment block is parsed by the `before_model_train` validator and by the Template Authoring CI check. Without this header, the CI check fails.

The header format:

```
# TEMPLATE: {template_name}
# PURPOSE: {one sentence describing what this template does}
# PARAMETERS:
#   {PARAM_NAME}: {type} — {description}
#   {PARAM_NAME}: {type} — {description}
# OUTPUT_FILES:
#   {relative_path}: {description of the file}
#   {relative_path}: {description}
# EXPECTED_RUNTIME_SECONDS: {integer}
# REQUIRED_LIBRARIES: {comma-separated list}
```

The `PARAMETERS` section lists every placeholder using `{PLACEHOLDER_NAME}` syntax. The template body uses these exact placeholder names. The agent substitutes them by string replacement before submission.

The `OUTPUT_FILES` section lists every file the template writes to `{OUTPUT_DIR}`. The `after_code_execute` hook uses this list to collect output files. If a listed output file is missing after execution, the hook records a `missing_output_file` bug event.

### 39.3 Template Rules

**Self-contained:** no imports from the `ds-agent` project. The subprocess has no knowledge of the project's code structure.

**Deterministic output paths:** all output files are written to `{OUTPUT_DIR}`. The placeholder `{OUTPUT_DIR}` is always substituted by the `execute_code` tool at submission time. Never hardcode output paths.

**Progress reporting:** print progress statements to stdout in the format `[STAGE] message`. Example: `[DATA] Loaded 10842 rows, 28 columns`. The subprocess captures stdout — agents and the Ralph Loop read it for diagnostic context on failure.

**Seed at the top:** every template that uses randomness sets seeds in the very first executable line, before any imports that trigger random initialisation. The seed is passed as `{RANDOM_SEED}` placeholder (default value from config, always provided by the agent).

**Structured metrics output:** every template that produces metrics writes them to `{OUTPUT_DIR}/metrics.json` as a JSON dict. This is the only place metrics are written — never to stdout. The `log_metrics` MCP tool reads this file after execution.

**Graceful partial failure:** if a template produces multiple outputs (multiple charts, multiple model trials), it catches exceptions per output, writes failed outputs to `{OUTPUT_DIR}/errors.json`, and continues. A template that crashes on one bad column fails the entire stage.

**Training loop checklist compliance:** every DL template must pass `tools/validators.py::validate_training_checklist(code)` before it is ever submitted. The validator parses the code with `ast.parse()` and checks for the 15 required function names listed in §20.2.

### 39.4 EDA Template Specifications

**Parameters:** `{DATASET_PATH}`, `{OUTPUT_DIR}`, `{TARGET_COLUMN}` (optional, empty string if not identified), `{RANDOM_SEED}`, `{COLUMN_NAMES}` (comma-separated list), `{NUMERIC_COLUMNS}` (comma-separated), `{CATEGORICAL_COLUMNS}` (comma-separated).

**Required output files:**
- `{OUTPUT_DIR}/stats.json` — all statistics as defined in §38.3's assertions
- `{OUTPUT_DIR}/eda_narrative_data.json` — raw data for the narrative (passed to the Azure OpenAI narrative call separately)
- `{OUTPUT_DIR}/charts/histogram_{col}.png` — one per numeric column
- `{OUTPUT_DIR}/charts/boxplot_{col}.png` — one per numeric column
- `{OUTPUT_DIR}/charts/correlation_heatmap.png`
- `{OUTPUT_DIR}/charts/class_balance.png` — only if `{TARGET_COLUMN}` is non-empty
- `{OUTPUT_DIR}/charts/missing_values_heatmap.png`
- `{OUTPUT_DIR}/errors.json` — any per-column failures

**Key library usage:** pandas for stats, matplotlib/seaborn for charts, `plt.savefig(path, bbox_inches='tight', dpi=150)` for all chart saves. Never call `plt.show()` — subprocess has no display.

### 39.5 Cleaning Template Specifications

**Parameters:** `{DATASET_PATH}`, `{OUTPUT_DIR}`, `{RANDOM_SEED}`, `{IMPUTATION_PLAN}` (JSON string: list of `{column, strategy, params}`), `{OUTLIER_PLAN}` (JSON string: list of `{column, method, action, threshold}`), `{TARGET_COLUMN}`.

**Required output files:**
- `{OUTPUT_DIR}/cleaned_dataset.parquet`
- `{OUTPUT_DIR}/transformation_log.json`
- `{OUTPUT_DIR}/cleaning_summary.json`
- `{OUTPUT_DIR}/errors.json`

**Imputation strategies the template must support:** `median`, `mode`, `mean`, `constant` (with value), `knn` (with `n_neighbors`), `mice` (using `IterativeImputer` from sklearn). The `{IMPUTATION_PLAN}` JSON specifies which strategy per column — the agent fills this in from its chain-of-thought reasoning.

**Outlier handling the template must support:** `iqr_cap` (cap at Q1 − k×IQR and Q3 + k×IQR), `zscore_remove` (remove rows where |z| > threshold), `isolation_forest` (flag with Isolation Forest), `leave` (document and leave unchanged). The `{OUTLIER_PLAN}` JSON specifies which method per column.

**Leakage prevention:** fit all imputers on the training portion of the data only. The cleaning template splits the data into train/validate portions using `{RANDOM_SEED}` before fitting. This is the only place in the pipeline where train/validate splitting is done for cleaning purposes — the feature engineering template receives the full cleaned dataset and does its own train/test split.

### 39.6 Feature Engineering Template Specifications

**Parameters:** `{DATASET_PATH}`, `{OUTPUT_DIR}`, `{RANDOM_SEED}`, `{TARGET_COLUMN}`, `{NUMERIC_COLUMNS}`, `{CATEGORICAL_COLUMNS}`, `{TEXT_COLUMNS}` (empty list if none), `{DATETIME_COLUMNS}` (empty list if none), `{DOMAIN_FEATURES}` (JSON: list of feature ideas from DuckDuckGo research and procedural memory), `{TEST_SIZE}` (float, default 0.2).

**Required output files:**
- `{OUTPUT_DIR}/features_train.parquet`
- `{OUTPUT_DIR}/features_test.parquet`
- `{OUTPUT_DIR}/feature_manifest.json`
- `{OUTPUT_DIR}/importance_chart.png`
- `{OUTPUT_DIR}/errors.json`

**Train/test split:** perform the split before fitting any transformer. Use `stratify=y` for classification tasks. Write the split indices to `{OUTPUT_DIR}/split_indices.json` so the split can be reproduced exactly by the Evaluation Agent.

**Feature manifest schema:** a JSON array where each entry is: `{feature_name, source_column (str or list), transform_type, transform_params (dict), importance_score (float or null), retained (bool)}`. Every column in the original dataset must have at least one corresponding entry in the manifest. Every new engineered feature must also have an entry.

**Leakage prevention:** all encoders (OneHotEncoder, TargetEncoder) and scalers are fitted on `X_train` only. Applied to both `X_train` and `X_test` using `transform()` (not `fit_transform()`) for the test set. The template must include a comment on the line where each encoder is fitted: `# FITTED ON TRAIN ONLY — applying to test uses transform()`.

### 39.7 Training Template Specifications

**Parameters (all templates):** `{FEATURES_TRAIN_PATH}`, `{FEATURES_TEST_PATH}`, `{TARGET_COLUMN}`, `{TASK_TYPE}`, `{OUTPUT_DIR}`, `{RANDOM_SEED}`, `{MODEL_TYPE}`, `{HYPERPARAMS}` (JSON dict).

**Required output files (all templates):**
- `{OUTPUT_DIR}/model.joblib` (sklearn) or `{OUTPUT_DIR}/model_state_dict.pt` (PyTorch)
- `{OUTPUT_DIR}/model.onnx`
- `{OUTPUT_DIR}/metrics/training.json`
- `{OUTPUT_DIR}/training_summary.md`

**DL-only required files:**
- `{OUTPUT_DIR}/loss_curve.json` — `[{epoch, train_loss, val_loss, lr}]` per epoch
- `{OUTPUT_DIR}/model_config.json` — the architecture configuration (input size, hidden dims, num_classes, etc.)

**ONNX export requirement:** every template must attempt ONNX export. If ONNX export fails, write the failure to `{OUTPUT_DIR}/errors.json` and continue — the primary model format is still valid. Do not abort training if ONNX export fails.

**Training metrics.json schema:** `{task_type, algorithm, model_type, train_metric, val_metric, cv_scores (list), cv_mean, cv_std, training_time_seconds, model_size_bytes, random_seed, hyperparams (dict)}`. For DL additionally: `epochs_trained, best_epoch, final_train_loss, final_val_loss`.

---

## 40. Agent System Prompt Writing Guide

### 40.1 The Twenty-Word Test

The first sentence of every system prompt must state the agent's single responsibility in 20 words or fewer. Run this test: can you describe what the agent does in one sentence without using the word "and" more than once? If you use "and" twice or more, the agent has multiple responsibilities — split it.

Passing: `"You are the Cleaning Agent. You apply corrective transformations to a dataset and document every decision."`

Failing: `"You are the Cleaning Agent. You analyse the EDA report, decide on cleaning strategies, implement them, verify the results, and produce documentation."` — this passes the word test but packs too much implicit complexity. The verification step should be part of the Ralph Loop criteria, not part of the agent's self-described responsibility.

### 40.2 Processing Protocol Design Rules

The processing protocol (Section 4 of every system prompt) must be specific enough that two different LLMs reading the same prompt produce the same sequence of tool calls on the same input. Vague steps produce inconsistent behaviour.

Bad: `"Analyse the dataset and produce statistics."`

Good: `"Step 3: Generate the EDA code by substituting {column_names} and {dtypes} into the EDA template. The template code is provided in the system context under the heading 'EDA TEMPLATE'. Call execute_code with the substituted code and with data_paths set to {{'df': session_state['input_artefact_path']}}. If exit_code is non-zero, call record_bug on the Bug Log MCP Server with the error details."`

Every step must:
- Name a specific tool call or a specific decision.
- Reference a specific session state key or file path.
- State what happens on failure.

### 40.3 Tool List Rules

Only list tools the agent can actually call. Do not list tools "for reference" that the agent cannot use. The Framework uses the tool list to populate the model's available tools — if you list a tool the agent's `MCPStreamableHTTPTool` instances do not expose, the model will try to call it and fail.

The tool list format: `- tool_name: Call this when {specific condition}.`

The `when` clause must be specific. Agents must not call tools outside their list. Agents that call unlisted tools are hallucinating tool names — their system prompt is insufficiently constraining.

### 40.4 Output Contract Precision

The output contract (Section 5) must name every file the agent produces. Do not use vague descriptions like "produces the model artefact" — name the exact relative path: `"Produces outputs/{run_id}/training/model.joblib and outputs/{run_id}/training/model.onnx."`

The output contract must also name every session state key the agent writes, and the type of value written. "Writes model metrics to session state" is vague. "Writes session_state['baseline_metrics'] as a dict keyed by metric name with float values" is precise.

### 40.5 Completion Signal Format

Every agent must output this exact structure at the end of its run, after all work is complete:

```
<DONE>{stage_name}</DONE>
{"session_state_keys_written": [...], "artefacts_produced": [...], "status": "success", "summary": "one sentence"}
```

The `<DONE>{stage_name}</DONE>` tag is the completion promise. The JSON summary is for the Orchestrator to verify the sub-agent understood its own output. If the agent cannot produce this structured final message, its context window is exhausted or it is confused — the Ralph Loop treats this as a failed iteration.

### 40.6 Escalation Protocol

The escalation protocol (Section 7) has one invariant: the agent never tries to fix code errors itself. It records the error and stops. The Ralph Loop and Debug Agent handle repair.

The correct escalation: `"If execute_code returns exit_code != 0: call record_bug on the Bug Log MCP Server with the error_class, error_message, traceback, and failed_code. Then output your final message with status='error' and the bug_id. Do not attempt to repair the code."`

### 40.7 Boundary Constraints

Boundaries (Section 8) must state explicit prohibitions. The word "never" must appear. Vague boundaries like "do not do unnecessary work" are not enforced.

Good boundaries: `"Never call deploy_endpoint."`, `"Never modify files in outputs/{run_id}/raw/."`, `"Never call execute_code more than 5 times in a single session."`, `"Never output code in your messages — all code goes through execute_code."`.

---

## 41. Ralph Loop RALPH.md Format Integration

### 41.1 Per-Stage RALPH.md Files

For each pipeline stage, create a corresponding `workflows/ralph/{stage_name}.ralph.md` file. This file is re-read on every Ralph Loop iteration for that stage. It follows the Ralphify format:

```
---
commands:
  - name: verify
    run: python workflows/verify_stage.py {run_id} {stage_name}
---

# Stage: {stage_name}

## Task
{base task description for this stage}

## Session Context
run_id: {run_id}
stage_name: {stage_name}
previous_failures: {ralph_loop_failures_from_bug_log}

## Verification Results (previous iteration)
{commands.verify}

## Instructions
If "PASSED" appears above: output <DONE>{stage_name}</DONE> and your summary.
If failures are listed above: address each failure specifically. Do not re-run work that already succeeded.
The following session state is available: {session_state_summary}
```

The `{commands.verify}` placeholder is replaced with the output of `python workflows/verify_stage.py {run_id} {stage_name}` before the prompt is piped to the agent. This script runs the DONE_CRITERIA assertions and returns either `"PASSED"` or a list of specific failures.

### 41.2 verify_stage.py

`workflows/verify_stage.py` is a CLI script that takes `run_id` and `stage_name`, runs all criteria from `workflows/criteria.py` for that stage, and prints either `"ALL CRITERIA PASSED"` or a structured failure report listing each failed assertion with its expected and actual values.

This script is the ground truth for the Ralph Loop. It is not an LLM call. It is a deterministic Python script. Its output is injected into the next iteration's prompt. Its exit code signals to the loop whether to continue (non-zero) or exit (zero).

### 41.3 Iteration Context Accumulation

On each Ralph Loop iteration, the `ralph_loop.py` implementation builds the stage prompt by:

1. Reading the base `workflows/ralph/{stage_name}.ralph.md` file.
2. Substituting `{run_id}` and `{stage_name}`.
3. Running `verify_stage.py` and substituting `{commands.verify}`.
4. Querying the Bug Log MCP Server for all bug events in this stage and this run. Substituting `{ralph_loop_failures_from_bug_log}` with a concise summary of what has been tried and what failed.
5. Generating a session state summary (all current non-null keys with their values, paths only). Substituting `{session_state_summary}`.

This assembled prompt is the complete context for the new agent session. The agent session is fresh — no prior conversation history. Everything the agent needs to know is in this assembled prompt.

---

## 42. Operational Runbook

### 42.1 System Startup

Start all processes in this order. Verify each health check before starting the next.

```
# Terminal 1
uvicorn mcp_servers.tracking.server:app --port 8100 --log-level info

# Verify
curl http://localhost:8100/health
# Expected: {"status": "healthy", "tool_count": 11, ...}

# Terminal 2
uvicorn mcp_servers.ds_tools.server:app --port 8101 --log-level info

# Verify
curl http://localhost:8101/health
# Expected: {"status": "healthy", "tool_count": 10, ...}

# Terminal 3
uvicorn mcp_servers.bug_log.server:app --port 8102 --log-level info

# Verify
curl http://localhost:8102/health
# Expected: {"status": "healthy", "bug_count": 0, "unresolved_count": 0}
```

Do not start the main application until all three health checks return healthy. The Orchestrator calls all three `/health` endpoints at startup and exits with a non-zero code if any are unhealthy.

### 42.2 Running a Pipeline

```
python main.py \
  --file path/to/your_data.csv \
  --task "Predict customer churn. Binary classification. Optimise for F1." \
  --pipeline_human_in_the_loop=false
```

The `run_id` is printed to stdout in the first log line: `{"event": "pipeline_start", "run_id": "abc-123", ...}`. Record it.

Monitor progress by tailing stdout. Each stage start and end produces a JSONL log line. Each Ralph Loop iteration produces a log line. Each human gate trigger produces a log line with the gate name and the prompt.

### 42.3 Resuming After a Crash

```
python main.py \
  --file path/to/your_data.csv \
  --task "Predict customer churn. Binary classification. Optimise for F1." \
  --resume-run-id abc-123
```

The Orchestrator queries `query_run_status` for `run_id=abc-123`, identifies the last successful checkpoint, reconstructs session state from artefact records, and starts from the first pending stage.

If the resume fails (database corruption or missing artefact files), start fresh. Do not attempt multiple resume retries — investigate and fix the root cause first.

### 42.4 Approving Human Gates

When a human gate is triggered, the Orchestrator writes a gate prompt to `outputs/{run_id}/gates/{gate_name}.md` and waits. To approve:

```
curl -X POST http://localhost:8100/approve/{run_id}/{gate_name} \
  -H "Content-Type: application/json" \
  -d '{"decision": "proceed", "override_params": {}}'
```

For the outlier handling gate with override:
```
curl -X POST http://localhost:8100/approve/{run_id}/outlier_handling \
  -H "Content-Type: application/json" \
  -d '{"decision": "proceed", "override_params": {"Fare": "iqr_cap"}}'
```

Valid decisions: `proceed`, `abort`. For gates with override parameters (outlier handling, fairness review), include the override in `override_params`.

### 42.5 Querying Memory

Episodic query (what happened in past runs):
```
python -c "
from memory.episodic import query_episodic
results = query_episodic('best model for binary classification this month')
print(results)
"
```

Semantic query (best artefacts by metric):
```
python -c "
from memory.semantic import query_semantic
results = query_semantic('high ROC-AUC binary classification model')
print(results)
"
```

Procedural query (what repair strategies worked):
```
python -c "
from memory.procedural import get_top_strategies
strategies = get_top_strategies('RuntimeError', 'torch')
print(strategies)
"
```

Full Memory Agent query (synthesised answer):
```
python -c "
from memory.agent import query_memory
answer = query_memory('What was the best F1 score achieved on a fraud detection task?')
print(answer)
"
```

### 42.6 Inspecting the Bug Log

List all bugs for a run:
```
sqlite3 data/buglog.db "
SELECT bug_id, stage_name, iteration_number, error_class, resolved, resolution_type
FROM bugs
WHERE run_id = 'your-run-id'
ORDER BY timestamp;
"
```

Get the full repair history for a specific bug:
```
sqlite3 data/buglog.db "
SELECT b.bug_id, b.error_class, b.error_message,
       r.attempt_number, r.strategy_name, r.attempt_result
FROM bugs b
LEFT JOIN repair_attempts r ON b.bug_id = r.bug_id
WHERE b.bug_id = 'your-bug-id'
ORDER BY r.attempt_number;
"
```

View unresolved bugs (escalated to human gate):
```
sqlite3 data/buglog.db "
SELECT bug_id, stage_name, error_class, error_message
FROM bugs
WHERE run_id = 'your-run-id' AND resolved = 0;
"
```

### 42.7 Inspecting the Tracking Database

List all artefacts for a run:
```
sqlite3 data/tracking.db "
SELECT artefact_type, artefact_path, stage_name, timestamp
FROM artefacts
WHERE run_id = 'your-run-id'
ORDER BY timestamp;
"
```

Get the lineage of a specific artefact:
```
sqlite3 data/tracking.db "
SELECT p.artefact_path AS parent, l.relationship, c.artefact_path AS child
FROM lineage l
JOIN artefacts p ON l.parent_artefact_id = p.artefact_id
JOIN artefacts c ON l.child_artefact_id = c.artefact_id
WHERE c.artefact_path = 'outputs/your-run-id/training/model.joblib';
"
```

Get the best model ever recorded for a task type:
```
# Via MCP tool
curl -X POST http://localhost:8100/mcp/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: your-session-id" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"query_best_artefact","arguments":{"artefact_type":"model","task_type":"binary_classification","metric_name":"roc_auc","ascending":false}},"id":1}'
```

### 42.8 Debugging a Ralph Loop Exhaustion

When a stage reaches `MAX_ITERATIONS` and triggers the human gate:

Step 1: Read the escalation report at `outputs/{run_id}/escalations/{stage_name}.md`. This contains all iterations, all verification failures, all Debug Agent repair attempts, and the Browser-Use/DuckDuckGo results.

Step 2: Query the bug log for all bugs in this stage. Identify the canonical error ID and whether it has been seen before.

Step 3: If the error is a new pattern not in the error catalogue: fix the template or the tool implementation. Add the new pattern to `config/error_catalogue.py`. Restart the Ralph Loop for that stage.

Step 4: If the error is a library version incompatibility: update the relevant package in the installed environment. Restart the Ralph Loop.

Step 5: If the template has a logical error (data leakage, wrong metric computation): fix `templates/{template_name}.py`. Restart the Ralph Loop.

Step 6: After fixing the root cause, approve the human gate with `decision=proceed` and optional override params.

Step 7: After the run completes, update `docs/error_catalogue.md` with the new pattern and fix. Update `config/error_catalogue.py` with the new entry and an initial `historical_success_rate`.

---

## 43. Decision Log — Pre-Populated Entries

Maintain `docs/decision_log.md` with these pre-populated entries. Every new decision made during implementation must be added in the same format: **Decision**, **Alternatives Considered**, **Rationale**, **Date**, **Consequences**.

### Decision 1 — Ralph Loop as the Core Execution Pattern

**Decision:** Wrap every pipeline stage in a Ralph Loop that runs until DONE_CRITERIA pass, with fresh context on each retry.

**Alternatives considered:** standard single-pass pipeline with manual error handling; retry only on specific exception types; no automatic retry.

**Rationale:** The Ralph Loop pattern emerged in 2025 as a proven approach for getting correct outputs from LLM agents. The key insight — even when AI says "done," automatically restart it and make it verify itself — directly addresses the failure mode of agents that report completion without verifying outputs. Fresh context on retry prevents the compounding confusion of a long, degraded conversation history. The verification step (DONE_CRITERIA) is deterministic Python, not an LLM call — it cannot be manipulated by the agent.

**Consequences:** Higher Azure OpenAI token usage per stage (multiple LLM calls per stage rather than one). Longer wall-clock time per stage when failures occur. Better output correctness and lower manual debugging time in exchange.

### Decision 2 — Three MCP Servers with Separate Crash Domains

**Decision:** Use three separate FastMCP server processes: Tracking (port 8100), DS Tools (port 8101), Bug Log (port 8102).

**Alternatives considered:** one combined MCP server; two servers (tracking + tools); co-locate bug log with tracking.

**Rationale:** Separate crash domains mean a DS tool bug cannot corrupt the audit trail. The Tracking Server must be the most reliable component — it is the pipeline's memory. The Bug Log Server must be independently accessible even when the DS Tools Server is being debugged. Co-locating any two would eliminate safety properties that are hard to restore after an incident.

**Consequences:** Three processes to manage. Three health checks to pass at startup. Additional operational complexity in exchange for stronger fault isolation.

### Decision 3 — Three-Tier Memory (Episodic + Semantic + Procedural)

**Decision:** Implement three distinct memory tiers: working memory (session state + InMemoryHistoryProvider), episodic memory (SQLite, per-run event log), and long-term semantic+procedural memory (JSON + NumPy, cross-run knowledge base).

**Alternatives considered:** single-tier vector store only; episodic only; no persistent memory; use an external memory service (Mem0, Zep).

**Rationale:** Episodic memory captures what happened so agents can recall past failures. Semantic memory captures what exists so agents can find the best past artefact. Procedural memory captures how to succeed so the Debug Agent and Feature Engineering Agent improve with every run. External services (Mem0, Zep) add network dependencies and authentication overhead. The JSON + NumPy approach is sufficient for the scale of this system and requires no additional services.

**Consequences:** Memory Agent must be kept running alongside the pipeline. Memory files (`.json`, `.npy`, `.db`) must be included in backup procedures. At > 50,000 artefact records, semantic search latency will exceed 500ms — migration to an in-process vector database is required at that scale.

### Decision 4 — Azure OpenAI Exclusively

**Decision:** Azure OpenAI is the only model provider. No other provider is configured.

**Alternatives considered:** multi-provider (Azure primary + OpenAI fallback); Ollama for local development; Anthropic Claude for specific agents.

**Rationale:** One provider means one authentication surface, one rate limit management layer, one billing account, one SLA. Multi-provider adds complexity at every layer of the stack. The productivity cost of managing multiple providers outweighs the resilience benefit for a team running one pipeline at a time.

**Consequences:** Dependency on Azure OpenAI availability. Regional capacity constraints during peak hours. Mitigated by retry middleware and the token budget guardrail.

### Decision 5 — Subprocess Execution (Not Docker)

**Decision:** Generated code runs in a subprocess with safety pattern filtering. No Docker isolation in v1.

**Alternatives considered:** Docker container per execution; Azure Container Instance; Python `exec()` sandbox; Pyodide WebAssembly sandbox.

**Rationale:** Docker reduces cold-start time by 2–15 seconds per execution, which adds 20–150 seconds per stage (10–20 executions per stage). For a pipeline targeting a 20-minute wall-clock time, this overhead is significant. Safety patterns cover the main injection risks. Docker isolation is listed as the first item on the roadmap for v2.

**Consequences:** Generated code can potentially access the host filesystem outside the output directory if safety patterns are bypassed. Document this in the security review. Mitigate with principle of least privilege on the user running the application.

### Decision 6 — In-Process Memory Index (JSON + NumPy)

**Decision:** Use `data/semantic_memory.json` + `data/semantic_embeddings.npy` as the semantic memory store.

**Alternatives considered:** Chroma (in-process vector DB), Qdrant (external vector DB), FAISS, pgvector.

**Rationale:** For the scale of this system (hundreds to low thousands of artefacts), pure NumPy cosine similarity is fast enough (< 100ms). No additional service to deploy or manage. Easy to inspect and debug (the JSON file is human-readable). Chroma is the recommended migration target when the index exceeds 50,000 records.

### Decision 7 — Magika as the Primary File Type Detector

**Decision:** Use Google's Magika for file type detection, with `filetype` (magic bytes) and content heuristics as fallback layers.

**Alternatives considered:** extension-only detection; `python-magic` (libmagic); `filetype` library only.

**Rationale:** Extension-only detection fails on misnamed files (common in real-world data). `python-magic` requires a C library (`libmagic`) that adds OS-level dependencies and fails on Windows without manual DLL installation. `filetype` is pure Python but is inaccurate for ambiguous text formats. Magika is AI-powered, trained on 100M+ files, achieves ~99% accuracy, and runs in ~5ms. The ~1MB model download is a one-time cost at first import.

---

## 44. New Engineer Onboarding Guide

### 44.1 Prerequisites (install before day one)

Python 3.12 via `pyenv` (not system Python, not Conda). Docker Desktop or Docker Engine (needed for future Docker isolation work, and for running containerised services in testing). Azure CLI with `az login`. Git with SSH key. VS Code with Python and Pylance extensions. SQLite browser tool for inspecting tracking and bug log databases.

### 44.2 Day One Setup (follow in order, verify each step)

**Step 1** — Clone repository. Read `README.md` completely. Read Sections 1 and 2 of this document before touching any code.

**Step 2** — Create virtual environment: `python3.12 -m venv .venv && source .venv/bin/activate`.

**Step 3** — Install: `pip install -e ".[dev]"`.

**Step 4** — Copy `.env.example` to `.env`. Fill in all five Azure OpenAI variables using values from the team's shared secrets manager.

**Step 5** — Verify Azure OpenAI access: `python -c "from agents.clients import primary_client; print('OK')"`. If this fails, resolve before proceeding.

**Step 6** — Start all three MCP servers (three terminals). Verify all three `/health` endpoints return healthy.

**Step 7** — Run unit tests: `pytest tests/unit/ -x -v --timeout=30`. All must pass. If any fail, stop and investigate before proceeding.

**Step 8** — Run the hello-world integration test: `pytest tests/integration/test_tracking_mcp.py::test_hello_world -v`.

**Step 9** — Run the Titanic e2e test (takes 15–20 minutes): `pytest tests/e2e/test_titanic.py -v --timeout=1800`.

**Step 10** — Inspect the outputs: `ls -la outputs/$(cat /tmp/last_run_id)/`. Open the HTML report in a browser. Browse the tracking database with your SQLite tool.

### 44.3 Codebase Reading Order

Read source files in this order. Each file references concepts from files you have already read:

1. `config/settings.py` — all configuration options.
2. `config/safety_patterns.py` — what is blocked.
3. `config/error_catalogue.py` — known errors and fix strategies.
4. `middleware/chain.py` — how middleware is assembled per agent type.
5. `middleware/logging_mw.py` — the simplest middleware; learn the AgentMiddleware interface.
6. `hooks/hooks.py` — all lifecycle hook functions.
7. `tools/file_type_detector.py` — the Magika singleton and three-layer detection.
8. `mcp_servers/tracking/tools.py` — every tracking operation.
9. `mcp_servers/ds_tools/tools.py` — every DS tool.
10. `mcp_servers/bug_log/tools.py` — every bug log operation.
11. `workflows/criteria.py` — DONE_CRITERIA for every stage.
12. `workflows/ralph_loop.py` — the core execution loop.
13. `workflows/pipeline_graph.py` — the static workflow graph.
14. `agents/base.py` — shared agent construction.
15. `agents/orchestrator.py` — the top-level coordinator.
16. The agent file for your assigned work.
17. The template file(s) your agent uses.

### 44.4 Making Your First Change

**Step 1** — Read Section 21's contract table for the agent you are modifying. Understand what it receives and what it must produce.

**Step 2** — Read the agent's system prompt in its `agents/*.py` file.

**Step 3** — Read the DONE_CRITERIA for the agent's stage in `workflows/criteria.py`.

**Step 4** — Write a failing unit test demonstrating the intended change.

**Step 5** — Make the change.

**Step 6** — Verify the unit test passes.

**Step 7** — Run the full unit suite: `pytest tests/unit/ -x -v`. All must pass.

**Step 8** — If you changed a template, run the `before_model_train` validator against it: `python -c "from tools.validators import validate_training_checklist; validate_training_checklist(open('templates/dl/bert_classify.py').read())"`.

**Step 9** — If you changed a contract (agent input or output): update Section 21's contract tables, update both the producing and receiving agent's system prompts, add an entry to `docs/decision_log.md`.

**Step 10** — Commit format: `<type>(<scope>): <description>`. Types: `feat`, `fix`, `docs`, `test`, `refactor`. Scope: module name. Example: `fix(cleaning_agent): handle MNAR imputation for text columns`.

### 44.5 Common Mistakes

**Storing data in session state** — session state is for paths and small scalar metadata only. Never store DataFrames, numpy arrays, or model weights. These go in files, accessed by path.

**Creating LLM clients inside agent modules** — client instances live in `agents/clients.py` only. Import `primary_client` and `fast_client` from there.

**Adding retry logic inside tool functions** — RetryMiddleware handles retry uniformly. Tool-level retry creates double-retry with exponentially multiplied wait times.

**Writing output files to paths not under `OUTPUT_DIR`** — templates must only write to `{OUTPUT_DIR}`. Any other path may be outside the process's permissions or invisible to the `after_code_execute` hook.

**Calling `plt.show()` in templates** — the subprocess has no display. Use `plt.savefig(path)` and `plt.close()` after every chart. Forgetting `plt.close()` causes matplotlib to accumulate figures in memory across multiple chart generations.

**Using `from subprocess import *` in generated code** — this is a blocked pattern. It will trigger SafetyMiddleware and the code will be rejected. Use explicit imports if subprocess access is genuinely needed, and add a justification to the safety review.

**Skipping the `before_model_train` validator** — the validator enforces the 15-item DL training checklist. A DL template that skips seeds is non-reproducible. A template that skips gradient clipping may diverge. A template that skips early stopping may overfit. These failures are hard to debug in later stages.

**Updating session state keys without updating Section 21** — the contract tables are the source of truth for inter-agent communication. Undocumented session state keys are invisible to downstream agents, invisible to the contract validator, and invisible to the Orchestrator's resume logic. Always update §21 alongside any session state change.

---

## 45. Performance and Token Budget Guide

### 45.1 Token Usage Profile per Pipeline Run

Understanding where tokens are consumed helps you stay within the `PIPELINE_TOKEN_BUDGET` and identify optimisation opportunities.

Typical token distribution for a Titanic-class tabular classification run:

Orchestrator coordination calls: ~15,000 tokens total (many short calls).

EDA Agent: ~5,000 tokens for stat prompts + ~3,000 for narrative generation = ~8,000.

Cleaning Agent: ~8,000 tokens (chain-of-thought is the most token-intensive per stage).

Feature Engineering Agent: ~6,000 tokens including DuckDuckGo research summarisation.

Training Agent: ~10,000 tokens (multiple algorithm trials, each with a prompt).

Tuning Agent: ~4,000 tokens (short, structured prompts).

Evaluation Agent: ~6,000 tokens.

Explainability Agent: ~5,000 tokens + narrative = ~7,000.

Report Agent: ~12,000 tokens (long input context from all previous artefacts).

Deployment Agent: ~4,000 tokens.

Memory Agent + Artefact Tracking Agent + Bug Log Agent: ~5,000 tokens combined.

**Baseline total (no errors):** ~85,000–100,000 tokens.

**With Ralph Loop iterations:** each retry adds approximately the stage's baseline token count. Two retries on the EDA stage add ~16,000 tokens. Three retries on the Training stage add ~30,000 tokens.

The default budget of 500,000 tokens provides ~5× headroom over the baseline. For DL scenarios (BERT fine-tuning, complex model selection), the baseline is higher — budget 200,000–300,000 tokens for the DL stages.

### 45.2 Prompt Size Reduction Strategies

The EDA Agent's stat prompt is the easiest win. Instead of passing the full `describe()` output (which can be hundreds of lines for wide datasets), pass only: column names, dtypes, top-5 null columns, top-5 skewed columns, class balance summary. This reduces the EDA stat prompt from ~3,000 tokens to ~800 tokens with no loss of output quality.

The Report Agent's context is the most token-intensive because it reads the content of multiple artefact files. Reduce it by: passing artefact summaries (key metrics only) rather than full file contents; reading charts as base64 only at render time (not in the prompt); using the fast deployment model for report sections that are template-fill rather than reasoning-intensive.

The Cleaning Agent's chain-of-thought prompt benefits from a structured format. Instead of free-form reasoning about each column, provide a structured template with one row per column and three fields to fill in (missingness pattern, recommended strategy, justification). This reduces the prompt by ~40% for wide datasets.

### 45.3 Ralph Loop Token Optimisation

The most expensive scenario is a stage that fails verification multiple times. Each retry includes the accumulated bug context from the Bug Log Agent, which grows with each iteration. Cap the bug context at 2,000 tokens per iteration: pass the most recent 2 failures in full detail, summarise older failures in one sentence each.

The Debug Agent's Browser-Use call (Attempt 2) produces up to 4,000 tokens of documentation content. This is the most expensive single prompt in the system (~6,000 tokens total with context). Budget for it in the `PIPELINE_TOKEN_BUDGET` — it fires only once per error pattern per run, and it is money well spent compared to manual debugging.

### 45.4 Speed Optimisation

The most impactful speed optimisation is running multiple algorithm trials in parallel within the Training Agent. The Training Agent issues three `execute_code` calls (one per algorithm). Issue all three concurrently using `asyncio.gather()`. This reduces the Training stage wall-clock time by ~60% for classical ML tasks.

The second most impactful is the `embed_text` cache. Without caching, the Memory Agent calls Azure OpenAI for an embedding every time a new artefact is produced. With the SHA-256 content hash cache in the DS Tools MCP Server, repeat artefacts (same content across resumptions) never trigger a second embedding call.

The third most impactful is Magika singleton initialisation. Without the singleton pattern, each `read_file` call takes ~200ms for Magika model load. With the singleton, subsequent calls take ~5ms. For a pipeline that calls `read_file` dozens of times, this saves several seconds.

---

## 46. Final Checklist — Before Claiming Production-Ready

Work through this checklist before declaring the system production-ready. Every item must be checked, not estimated. Unchecked items are known risks that must be documented in `docs/decision_log.md` before deployment.

### 46.1 Architecture

- All three MCP servers pass the three compliance checks from §5.3.
- `MCPStreamableHTTPTool` instances are created at application startup and reused (verify no per-call construction with a memory profiler).
- Magika is initialised as a module-level singleton (verify with a call counter added to `get_magika()`).
- Azure OpenAI clients are two instances total, not per-agent (verify by checking `agents/clients.py` is imported, not instantiated in agent modules).
- All sub-agents use `propagate_session=True` in `as_tool()` calls (verify with a grep: `grep -r "as_tool" agents/ | grep -v "propagate_session=True"`).

### 46.2 Ralph Loop

- Every stage has a DONE_CRITERIA entry in `workflows/criteria.py`.
- Every DONE_CRITERIA entry has at least one domain-specific assertion (not just file existence checks).
- Every stage has a `workflows/ralph/{stage_name}.ralph.md` file.
- `workflows/verify_stage.py` exits with code 0 on a passing stage and non-zero on a failing stage.
- Ralph Loop unit tests pass: success case, single-failure-then-pass, MAX_ITERATIONS exhaustion.
- Completion promise tags are present in every agent's system prompt under Section 6.

### 46.3 Memory System

- Episodic memory has entries after a complete pipeline run (query `data/memory.db`).
- Semantic memory index has entries (check `data/semantic_memory.json` is non-empty after a run).
- Procedural memory is updated after a successful bug resolution (check `data/procedural_memory.json` has incremented success counts).
- All three retrospective queries from §27.3 return correct answers.
- `embed_text` caching is verified: two identical calls produce one Azure OpenAI API call.

### 46.4 Observer Agents

- Artefact Tracking Agent indexes new artefacts within 15 seconds (verify with a timing test).
- Bug Log Agent enriches synthetic bugs with recommended strategies within 10 seconds.
- Memory Agent writes episode records after each stage completion.
- All three observer agents are running as async background tasks (not blocking the main pipeline).

### 46.5 Safety

- 25 injection tests all blocked by SafetyMiddleware and `before_code_execute`.
- `before_deploy` hook correctly blocks deployment with `recommend=="block"`.
- `before_model_train` validator correctly rejects DL templates missing any of the 15 checklist items.
- PII detection triggers the human gate on a dataset with known PII columns.
- No API key appears in any log file or artefact file (scan with `grep -r "AZURE_OPENAI_API_KEY" outputs/`).

### 46.6 Responsible AI

- Fairlearn analysis runs for every classification task (verify by checking `eval/fairness_metrics.json` exists after a classification run).
- Deployment is blocked when fairness violation is detected (inject a demographic parity violation and verify the `before_deploy` hook fires).
- All deployed endpoints expose `/model_card` (verify with `curl http://localhost:8200/model_card`).
- SHAP explanations are present before any deployment proceeds (verify by clearing `explain/` and confirming deployment fails).

### 46.7 Operations

- All five benchmark datasets complete with `PIPELINE_HUMAN_IN_THE_LOOP=false` without manual intervention.
- All five benchmark datasets meet minimum metric thresholds from §32.6.
- Pipeline resumes correctly from every stage checkpoint (test by simulating crashes at stages 3, 5, and 8).
- Distributed trace in the observability backend shows all spans for a complete pipeline run with no orphaned spans.
- All documentation files in `docs/` are current and match the implementation.

---

*End of specification. Total sections: 46. This document governs every implementation decision in the DS Agent project. The Ralph Loop wraps every stage — this is non-negotiable. The three-tier memory system (episodic, semantic, procedural) is non-negotiable. The two observer agents (Artefact Tracking Agent, Bug Log Agent) are non-negotiable. Azure OpenAI is the exclusive model provider. All architectural decisions are in §43. All stage verification criteria are in §38. All template rules are in §39. All prompt writing rules are in §40.*
