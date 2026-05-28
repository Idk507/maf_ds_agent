# Data Science Agent — Complete Implementation Specification

> **Stack:** Microsoft Agent Framework 1.0 · Python 3.12 · Azure OpenAI (exclusive)
> **Protocols:** MCP Streamable HTTP · Streamable SSE Sessions · A2A
> **Scope:** Autonomous end-to-end Data Science and Generative AI workloads
> **Author perspective:** Senior AI Architect and Engineer
> **Last updated:** May 2026

---

## Architect's Preface

This document is a complete, opinionated, production-grade implementation specification. It is not a tutorial. It is the single source of truth your engineering team uses to build, test, harden, and ship a fully autonomous Data Science Agent system.

Every decision here is deliberate. Azure OpenAI is the exclusive model provider — not because other providers do not work with the Microsoft Agent Framework, but because a single provider means a single authentication surface, a single rate-limit management layer, a single billing account, and a single SLA. Complexity removed early is complexity that never becomes a production incident.

The `read_file` tool is the system's front door. Every pipeline begins with a file. That file might be a CSV, a Parquet, a PDF report, a PNG chart, a JSON config, an Excel workbook, an SQLite database, a Python pickle of a pre-trained model, or a ZIP archive containing all of the above. The system must never assume what a file is based on its extension alone. It must detect the true type, route accordingly, and proceed. This is not a nice-to-have. It is a correctness requirement.

Read this document end-to-end before writing a single line of code. Architectural decisions compound — an error in Phase 1 metastasises into ten errors in Phase 5.

---

## Table of Contents

1. [Architecture Mental Model](#1-architecture-mental-model)
2. [Technology Decisions and Rationale](#2-technology-decisions-and-rationale)
3. [Microsoft Agent Framework 1.0 — What You Must Know](#3-microsoft-agent-framework-10--what-you-must-know)
4. [Azure OpenAI Integration Layer](#4-azure-openai-integration-layer)
5. [MCP Design — Streamable HTTP Sessions](#5-mcp-design--streamable-http-sessions)
6. [The read\_file Tool — File Type Detection Pipeline](#6-the-read_file-tool--file-type-detection-pipeline)
7. [Agent Catalogue and System Prompts](#7-agent-catalogue-and-system-prompts)
8. [Workflow Graph and Pipeline Orchestration](#8-workflow-graph-and-pipeline-orchestration)
9. [Middleware Pipeline — Design and Ordering](#9-middleware-pipeline--design-and-ordering)
10. [Hooks Registry — Lifecycle Events](#10-hooks-registry--lifecycle-events)
11. [Code Sandbox — Execution, Auto-Fix, and Debug Loop](#11-code-sandbox--execution-auto-fix-and-debug-loop)
12. [Browser-Use Integration — Docs and Error Crawling](#12-browser-use-integration--docs-and-error-crawling)
13. [DuckDuckGo Deep Research Integration](#13-duckduckgo-deep-research-integration)
14. [Deep Learning and Generative AI Coverage Matrix](#14-deep-learning-and-generative-ai-coverage-matrix)
15. [Session Management with Streamable HTTP](#15-session-management-with-streamable-http)
16. [In-Process Memory and Context Providers](#16-in-process-memory-and-context-providers)
17. [Phase 1 — Foundation, Environment, MCP Servers](#17-phase-1--foundation-environment-mcp-servers)
18. [Phase 2 — read\_file Tool and Data Ingestion](#18-phase-2--read_file-tool-and-data-ingestion)
19. [Phase 3 — EDA, Cleaning, Feature Engineering](#19-phase-3--eda-cleaning-feature-engineering)
20. [Phase 4 — Model Training, Tuning, and Evaluation](#20-phase-4--model-training-tuning-and-evaluation)
21. [Phase 5 — Explainability, Reporting, Deployment](#21-phase-5--explainability-reporting-deployment)
22. [Phase 6 — Memory Agent and Retrospection](#22-phase-6--memory-agent-and-retrospection)
23. [Phase 7 — Hardening, Observability, Production Gate](#23-phase-7--hardening-observability-production-gate)
24. [Directory and File Map](#24-directory-and-file-map)
25. [Configuration Reference — Azure OpenAI Only](#25-configuration-reference--azure-openai-only)
26. [Error Catalogue and Auto-Remediation Map](#26-error-catalogue-and-auto-remediation-map)
27. [Security and Responsible AI Obligations](#27-security-and-responsible-ai-obligations)
28. [Verification Gates per Phase](#28-verification-gates-per-phase)
29. [Reference Documents](#29-reference-documents)

---

## 1. Architecture Mental Model

### 1.1 The System in One Sentence

A directed graph of specialised agents, each owning exactly one pipeline stage, coordinated by an Orchestrator that reads from a shared Tracking MCP Server, writes checkpoints after every stage, and reroutes failures to a Debug Agent before retrying — all operating through Azure OpenAI as the sole LLM backend.

### 1.2 Three-Layer Stack

Think of the system as three layers, each with a distinct concern:

**Layer 1 — Intelligence Layer**
The Microsoft Agent Framework agents. Each agent wraps an Azure OpenAI completion call, a system prompt defining its role, a set of MCP tool references, and a session state object. Agents do not import data science libraries. They reason and delegate.

**Layer 2 — Protocol Layer**
Two MCP servers communicate via Streamable HTTP. The Tracking MCP Server is the audit log and coordination backbone. The DS Tools MCP Server is the action surface. The Framework's `MCPStreamableHTTPTool` class bridges agents to both servers. Every action produces an MCP tool call, which is logged, retried, and versioned.

**Layer 3 — Execution Layer**
A Docker sandbox with no outbound network. All generated code runs here. All file I/O goes through a defined `/workspace` path. All outputs are collected as artifacts. The `read_file` tool is the entry point into this layer from the outside world.

### 1.3 Data Flow (one canonical path)

```
User input (file path + task description)
  ↓
Orchestrator Agent creates a Workflow Session
  ↓
read_file tool → FileTypeDetector → specific reader → parsed data object
  ↓
Ingestion Agent → validated raw snapshot
  ↓
EDA Agent → statistics + charts + narrative
  ↓
Cleaning Agent → versioned cleaned dataset
  ↓
Feature Engineering Agent → feature manifest
  ↓
Model Selection + Training Agent → best baseline model
  ↓
HP Tuning Agent → tuned model
  ↓
Evaluation Agent → metrics + fairness report
  ↓
Explainability Agent → SHAP + LIME + narrative
  ↓
Report Agent → HTML/Markdown report
  ↓
Deployment Agent → live endpoint + monitoring config
  ↓
Memory Agent indexes all artifacts for retrospective queries
```

Every arrow is a checkpoint in the Tracking MCP Server. Every failure on any arrow routes to the Debug Agent.

### 1.4 What the Orchestrator Does Not Do

The Orchestrator does not execute data science logic. It does not parse files. It does not train models. It does not compute statistics. It routes, monitors, checkpoints, and reroutes. Any agent that accumulates responsibilities beyond its defined stage is incorrectly designed. Refactor it.

---

## 2. Technology Decisions and Rationale

### 2.1 Azure OpenAI Exclusively

The Microsoft Agent Framework ships a first-party `agent-framework-openai` connector that supports both OpenAI and Azure OpenAI endpoints. Use the Azure OpenAI path exclusively. This means:

- Authentication uses Azure Active Directory managed identity — no API keys stored in environment variables in production
- Deployments are versioned in your Azure OpenAI resource (e.g., `gpt-4o`, `gpt-4o-mini`) and referenced by deployment name, not by model ID
- Rate limits are quota-managed per deployment in the Azure portal
- All completions flow through your Azure tenant, never through OpenAI's consumer endpoints

The correct package is `agent-framework-openai`, and it is a released package at 1.0 (no `--pre` flag required).

### 2.2 Microsoft Agent Framework 1.0 — Python

Released April 3, 2026. Stable APIs with a long-term support commitment. The framework unifies Semantic Kernel (foundation layer: kernel, plugin model, service connectors) with AutoGen (orchestration patterns: sequential, concurrent, handoff, group chat, Magentic-One) into one SDK.

Key architecture facts relevant to this system:

- `agent-framework-core` is intentionally minimal. You install provider packages separately.
- For Azure OpenAI: install `agent-framework-openai`. This is sufficient — you do not need `agent-framework-foundry`.
- For MCP tool support: `mcp` must be installed. For WebSocket MCP support: `mcp[ws]`.
- Tool handlers receive a `ToolInvocation` dataclass, not a raw dict. Return `ToolResult` with snake_case fields.
- The `agent_framework.openai` client automatically distinguishes Azure OpenAI from OpenAI by checking for `AZURE_OPENAI_ENDPOINT` in the environment.

### 2.3 MCP with Streamable HTTP (Spec 2025-06-18)

Older SSE-based MCP is deprecated for remote servers. Use Streamable HTTP everywhere. Key implementation requirements:

- Mount the MCP server at `/mcp` subpath using `mcp.streamable_http_app()` from the Python MCP SDK, then mount it into a FastAPI app. The full endpoint path becomes `/mcp/mcp` by default — document this clearly.
- The Framework's `MCPStreamableHTTPTool` class is the client-side connector. It handles session management, the `Mcp-Session-Id` header, and the sampling callback for LLM-driven tools.
- HTTP status codes are semantic signals to the MCP client. A `501` on GET means the server is broken. A `405 Method Not Allowed` with `Allow: POST` means it is a POST-only server by design. Always return `405` correctly.
- As of April 2026, there is a known issue (GitHub issue #5317 in the agent-framework repo) where the `MCPStreamableHTTPTool` uses a legacy GET SSE handshake that some strict servers reject with `405`. The fix is to use the current Streamable HTTP transport handshake — single POST endpoint with proper session management. Verify your Framework version has this fix before deployment.

### 2.4 File Type Detection — Magika

Use Google's Magika library for file type detection. It is AI-powered, trained on 100M+ files across 200+ content types, achieves ~99% accuracy, runs in ~5ms on a single CPU, and weighs ~1MB. It is significantly more accurate than extension-based detection and more reliable than magic-byte heuristics for the full range of data science file types encountered in practice.

Magika is used exclusively in the `read_file` tool (see Section 6). Nowhere else.

### 2.5 In-Process Persistence (No Azure Blob, No Azure ML, No Cosmos DB)

This specification uses Azure OpenAI as the sole Azure service. All other persistence is:

- Local filesystem inside the sandbox for temporary execution artifacts
- Structured JSON files written to a defined `outputs/` directory that the sandbox exposes
- An in-process SQLite database for the Tracking MCP Server (file-backed, durable across restarts)
- In-process Python dict + disk-backed JSON for the Memory Agent

This is a deliberate architectural constraint. A team that can run this system with only Azure OpenAI credentials is a team with a much lower barrier to entry, faster iteration, and no cloud resource provisioning overhead.

---

## 3. Microsoft Agent Framework 1.0 — What You Must Know

### 3.1 The Four Primitives

**Agent** — The core unit. Created once from a client, a name, a system prompt (instructions), and a list of tools. The agent wraps every LLM call with the middleware chain, the hooks, and the session state. An agent is stateless by design — state lives in the session.

**Session** — A stateful, persistent conversation between a user and an agent. The Framework manages the session object: it accumulates the message history, tracks pending tool calls, and persists key-value state that survives multiple `run()` calls. In multi-agent workflows, each agent invocation creates a child session scoped to that agent's stage.

**Workflow** — A graph of nodes where each node is either an Agent or a plain Python function. Edges define dependencies and routing conditions. The graph engine handles: sequential execution, fan-out (parallel branches), fan-in (merge results), conditional routing (if metric > threshold, go to Node A, else Node B), and human-in-the-loop gates (pause the graph until an external event arrives). Checkpointing is built into the graph engine — after each node completes, its output is persisted so the graph can resume after a crash.

**Middleware** — A pipeline that wraps every agent execution. Each middleware class implements `before_run`, `after_run`, `before_tool_call`, `after_tool_call`, and `on_error`. Middleware is registered on the client, not on individual agents, so it applies uniformly to every agent that the client creates. This is the right scope — per-client registration ensures no agent accidentally runs without the safety or logging middleware.

### 3.2 Package Installation Order

Install in this exact order to avoid dependency conflicts:

First: `pip install agent-framework-openai` — installs core + the Azure OpenAI/OpenAI connector. This is a stable released package.

Second: `pip install mcp` — installs MCP client support. Required for `MCPStreamableHTTPTool`.

Third: `pip install mcp[ws]` — adds WebSocket transport support. Required if any MCP server uses WebSocket (not needed for Streamable HTTP, but install anyway for flexibility).

Fourth: `pip install fastmcp` — the FastMCP framework for building MCP servers with minimal boilerplate.

Fifth: install all data science libraries pinned to exact versions. See Section 24 for the full sandbox requirements file.

### 3.3 Agent Framework — Azure OpenAI Client Construction

The Framework detects Azure OpenAI versus consumer OpenAI by presence of `AZURE_OPENAI_ENDPOINT` in the environment. When that variable is set, `agent_framework.openai.AzureOpenAIChatClient` is used automatically by the generic client factory.

Construct one client per deployment model. Do not construct one client per agent — that wastes connection pool resources. Construct one client for your primary reasoning model (e.g., `gpt-4o`) and one for your fast/cheap model (e.g., `gpt-4o-mini`). Reuse them across agents.

### 3.4 Tool Definition and ToolInvocation

All tools exposed to agents must be registered as `AIFunction` instances or MCP tool references. When an agent calls an MCP tool, the Framework handles the full MCP round-trip: serialise the tool call, POST to the MCP server, receive the `ToolResult`, deserialise, inject into the conversation. Your code never manually manages this round-trip.

Tool handlers receive a `ToolInvocation` dataclass with fields: `tool_name`, `tool_input` (a dict matching the tool's JSON Schema), `session_id`, `run_id`. Return `ToolResult` with fields: `result_type` (`"text"`, `"json"`, `"error"`), `text_result_for_llm` (a string that the LLM reads), and optionally `raw_result` (unprocessed data for programmatic use).

### 3.5 Context Providers and Chat History

The Framework provides pluggable context providers. For this system, use the built-in in-memory chat history provider (backed by a deque with a configurable max length). This gives each agent session a rolling window of conversation history. The Orchestrator session has a longer window (last 50 exchanges). Sub-agent sessions have shorter windows (last 20 exchanges) — they do not need the full pipeline history, only their stage-specific context.

The persistent key-value state in sessions is used to pass structured data between tool calls within a single agent's session: the cleaned dataset path, the feature manifest, the best model's metric scores, and so on.

---

## 4. Azure OpenAI Integration Layer

### 4.1 Deployment Strategy

Create these deployments in your Azure OpenAI resource before starting implementation:

**Primary reasoning deployment** — use `gpt-4o` (latest stable version available in your region). This deployment is used by: the Orchestrator, the Debug Agent, the Explainability Agent, the Report Agent, and any agent whose task requires extended multi-step reasoning.

**Fast execution deployment** — use `gpt-4o-mini`. This deployment is used by: the Ingestion Agent, the EDA Agent (stats computation prompts, not the narrative), the Cleaning Agent's transformation selection, and the Feature Engineering Agent's quick lookups.

**Embedding deployment** — use `text-embedding-3-large`. This is used exclusively by the Memory Agent's RAG query interface. One embedding deployment is sufficient.

Set the token-per-minute quota for the primary deployment to the maximum available in your subscription tier. Under-provisioned deployments become pipeline bottlenecks.

### 4.2 Authentication — Managed Identity Only

In production, use Azure Managed Identity. No API keys. This means:

- The service (Azure Container Instance, Azure App Service, or local machine with `az login`) has a managed identity assigned
- The managed identity is granted `Cognitive Services OpenAI User` role on the Azure OpenAI resource
- The `DefaultAzureCredential` class from `azure-identity` automatically uses the managed identity in Azure-hosted environments and falls back to `az login` on developer machines

In local development only, `az login` is acceptable. Never commit `AZURE_OPENAI_API_KEY` to version control. If you are using API keys during early development, rotate them before first deployment and switch to managed identity.

### 4.3 Deployment Name vs Model ID

Azure OpenAI uses deployment names, not model IDs, in API calls. Your deployment name is what you set in the Azure portal (e.g., `my-gpt4o-prod`). The model version is associated with the deployment. Your environment configuration must store the deployment name, not `gpt-4o`.

This distinction matters in the Framework: the `model` parameter in client construction is the deployment name, not the model ID string.

### 4.4 Retry and Rate Limit Strategy for Azure OpenAI

Azure OpenAI returns HTTP 429 (Too Many Requests) when you exceed your token quota. The correct response is exponential backoff with jitter. The Framework's Retry Middleware handles this automatically if you configure it to treat 429 as a retryable status code.

Additionally, plan for regional capacity constraints. If your primary region returns capacity errors (HTTP 503 with a specific error message), implement a secondary region fallback in the client construction layer — not in the middleware. This is a client-level concern.

---

## 5. MCP Design — Streamable HTTP Sessions

### 5.1 Two MCP Servers — Design Rationale

Two servers, not one, for a deliberate reason: the Tracking MCP Server must never go down, even if the DS Tools MCP Server has a bug and crashes. Separate servers means separate processes, separate crash domains, and independent restartability. The Tracking Server is the system's spine. Treat it as infrastructure.

### 5.2 Tracking MCP Server

**Process:** standalone Python process, runs with FastMCP, listens on port `8100`.

**Persistence:** SQLite database file at a configurable path. Write-ahead logging (WAL) mode enabled. This gives you: durable writes, concurrent readers, and sub-millisecond write latency for logging workloads.

**Tools it exposes (all via MCP Streamable HTTP):**

`record_start` — logs agent name, run ID, stage name, session ID, start timestamp, and input artifact paths. Returns the assigned log entry ID.

`record_end` — logs agent name, run ID, stage name, end timestamp, duration milliseconds, status (success or failure), and output artifact paths.

`record_tool_call` — logs tool name, agent name, input hash (SHA-256 of the tool input JSON, not the raw input), output hash, duration, and status.

`record_artifact` — logs artifact path, artifact type (dataset, model, report, chart, feature_manifest, etc.), producing agent, run ID, and a metadata JSON blob.

`record_error` — logs agent name, error class, error message, truncated traceback (first 2000 characters), attempt number, and whether the Debug Agent was invoked.

`record_checkpoint` — marks a stage as complete for a given run ID. The Orchestrator queries this before invoking any stage to implement resumable pipelines.

`query_run_status` — returns all stages and their statuses for a given run ID. The Orchestrator calls this when resuming after a crash.

`query_best_artifact` — accepts artifact type, task type, and metric name. Returns the artifact path of the historically best artifact matching those criteria, sorted by the metric. This powers the Memory Agent's retrospective queries.

`query_artifact_lineage` — accepts an artifact path. Returns all upstream artifacts (the full dependency chain) and all downstream artifacts (anything derived from this artifact).

### 5.3 DS Tools MCP Server

**Process:** separate Python process, runs with FastMCP, listens on port `8101`.

**Persistence:** none. This server is stateless. All state lives in files, in the sandbox, or in the Tracking MCP Server.

**Tools it exposes:**

`read_file` — the file ingestion entry point. Accepts a file path. Runs the full file type detection pipeline (Section 6). Returns a structured parse result. This is the most complex tool in the system. It does not just read bytes — it detects, validates, normalises, and returns a typed data object the agent can reason about.

`execute_code` — submits Python source code to the Docker sandbox. Returns stdout, stderr, exit code, and a list of output file paths written to `/workspace/output`. This is the only way code runs in this system.

`get_data_sample` — returns the first N rows of a dataset (CSV, Parquet, or JSON) as a JSON array. Used by agents to inspect data without loading the full dataset.

`write_output` — writes a string (Markdown, JSON, HTML) to a named file in the current run's output directory. Returns the file path.

`search_docs` — invokes Browser-Use to crawl documentation. Returns extracted text. Maximum 4,000 characters.

`web_research` — invokes DuckDuckGo search. Returns structured result list. Maximum 10 results.

`embed_text` — calls Azure OpenAI's embedding deployment. Accepts a string or list of strings. Returns embedding vectors. Used by the Memory Agent.

`semantic_search` — searches the in-process Memory Agent's index. Returns matching artifact records with scores.

`log_metrics` — writes a JSON metrics object to the current run's metrics file. Appends, does not overwrite. This is the system's MLflow substitute.

### 5.4 MCP Server — FastMCP Implementation Pattern

Use `fastmcp.FastMCP()` to construct each server. Mount the server's Streamable HTTP app into a parent FastAPI application. This gives you the MCP protocol endpoint at `/mcp/mcp` and the ability to add REST health-check endpoints at `/health` on the same server.

The health check endpoint must return: server name, version, uptime seconds, number of tools registered, and database connection status (for the Tracking Server). This health check is called by the Orchestrator at startup to verify both MCP servers are ready before beginning any pipeline run.

### 5.5 Session Lifecycle for MCP Tool Calls

Every MCP tool call goes through a session. The Framework's `MCPStreamableHTTPTool` manages this: on first use, it initialises an MCP session (receiving an `Mcp-Session-Id`), and it echoes that session ID on all subsequent calls. This means all tool calls within one agent session share one MCP session. Do not create a new `MCPStreamableHTTPTool` instance per tool call — create one instance per agent at agent construction time and reuse it.

---

## 6. The read\_file Tool — File Type Detection Pipeline

### 6.1 Architectural Importance

Every data science pipeline starts with a file. The quality of the downstream agents depends entirely on the quality of the parse. A misidentified file type produces corrupt data silently — no error, just wrong results. A wrong delimiter on a CSV, a wrong sheet on an Excel file, a wrong encoding on a text file: all produce data that looks valid but is not.

This tool is the most important tool in the system. Design it carefully.

### 6.2 Detection Strategy — Three-Layer Approach

Do not rely on file extension alone. Use three layers in sequence, stopping when confident:

**Layer 1 — Magika (primary detector)**

Run Magika's `identify_path()` on the file. Magika is AI-powered, trained on 100M+ files, and achieves ~99% accuracy across 200+ content types including binary formats, text formats, code files, documents, archives, images, audio, and video. Its inference time is ~5ms on CPU. It returns: a content type label (e.g., `csv`, `parquet`, `pdf`, `png`, `json`, `xlsx`, `sqlite`, `zip`, `python`, `mp3`), a MIME type, a confidence score, and a category.

Use Magika's output as the primary type signal.

**Layer 2 — Magic Bytes Validation (secondary check)**

For confidence scores below 0.85 or for file types where Magika historically has higher error rates (plain text formats that share magic bytes), run `filetype.guess()` from the `filetype` library against the first 261 bytes. This is a pure magic-number check — no AI, no heuristics beyond bytes. Use it to confirm or challenge Magika's result.

**Layer 3 — Content Heuristics (for ambiguous text files)**

When both Magika and `filetype` indicate a generic text type (e.g., `text/plain`), examine the content structure: count commas, tabs, and pipes in the first 10 lines to identify delimiter-separated values; look for `{` or `[` as first character to identify JSON or JSONL; look for XML/HTML tags; attempt to parse as a Python literal with `ast.literal_eval`. This layer disambiguates CSV, TSV, JSON, JSONL, XML, YAML, Markdown, and plaintext.

### 6.3 Complete File Type Routing Table

After detection, route to the appropriate reader based on the detected type. The routing table must cover every type an agent might encounter:

**Tabular Data — Structured**

`csv` or `text/csv` → read with `pandas.read_csv()`. Before reading, detect encoding with `chardet.detect()` on the first 10,000 bytes. Detect delimiter by running `csv.Sniffer().sniff()` on the first 2,048 bytes. Detect whether the first row is a header by examining if all values are strings and none are parseable as numbers. Return: schema (column names + dtypes), row count, sample (first 5 rows as JSON), and detected encoding, delimiter, and header flag.

`parquet` or `application/vnd.apache.parquet` → read schema with `pyarrow.parquet.read_schema()` first (zero data read). Then read metadata (row count, column statistics). Then optionally read a sample with `pandas.read_parquet(path, nrows=5)`. Return: schema, row count, sample.

`xlsx` or `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` → read with `openpyxl.load_workbook()`. Enumerate all sheets. For each sheet, read the first row to detect headers and the dimensions (max_row, max_column). Let the agent decide which sheet(s) to process. Return: list of sheets with their dimensions and first-row headers.

`xls` → read with `xlrd.open_workbook()`. Same sheet enumeration logic. Return same structure.

`tsv` → same as CSV pipeline with tab delimiter forced.

`json` → parse with `json.loads()`. If the top level is a list of dicts, treat as tabular. If the top level is a dict, treat as a single record. Return: detected JSON shape (tabular or record), key list (for records) or column list (for tabular), row count (for tabular), sample.

`jsonl` → read first 10 lines. Parse each as a JSON object. Check consistency of keys across lines. Return: key list, total line count (via a fast line counter, not full parse), sample.

`sqlite` or `application/x-sqlite3` → connect with `sqlite3.connect()`. Enumerate tables with `SELECT name FROM sqlite_master WHERE type='table'`. For each table, return column names, types, and row count. Let the agent choose which tables to query.

**Documents — Unstructured**

`pdf` or `application/pdf` → extract text with `pymupdf.open()` (faster than PyPDF2 and handles more edge cases). Detect if it is a scanned PDF (no extractable text) by checking if extracted text is shorter than 100 characters per page. If scanned, flag it as requiring OCR and set `requires_ocr=True` in the result. Return: page count, extracted text (first 3,000 characters), table of contents if present, `requires_ocr` flag.

`docx` or `application/vnd.openxmlformats-officedocument.wordprocessingml.document` → extract with `python-docx`. Return: paragraph count, extracted text (first 3,000 characters), table count, embedded image count.

`pptx` → extract with `python-pptx`. Return: slide count, text per slide (first 500 chars each), shape types present.

`txt` or `text/plain` → read directly. Return: character count, line count, detected encoding, first 3,000 characters.

`md` or `text/markdown` → read as text. Additionally extract headings, code blocks, and link count. Return the same structure as txt plus the metadata.

**Images**

`png`, `jpeg`, `jpg`, `gif`, `bmp`, `webp` or their MIME equivalents → use `Pillow.Image.open()`. Return: width, height, mode (RGB, RGBA, L, etc.), file size, EXIF data summary if present. Do not embed the image bytes in the result — return the file path. The agent will pass the path back to the sandbox for any processing.

**Audio and Video**

`mp3`, `wav`, `mp4`, `webm`, `avi` → use `mutagen` for metadata extraction. Return: duration seconds, bitrate, codec, sample rate (audio), resolution (video). Flag as requiring specialised processing (Whisper for transcription, or a vision model for video frames). The system does not transcribe or process audio/video in-process — it flags the file type and lets the Orchestrator decide whether to route it to a generative AI scenario.

**Archives**

`zip` → list contents with `zipfile.ZipFile`. Return: member count, member names and sizes, and whether any member is itself an archive (nested ZIP detection). Do not auto-extract. Let the agent decide which members to extract and process.

`tar`, `gz`, `bz2` → same: list contents, return structure, do not auto-extract.

**ML and Code Artefacts**

`pickle` or detected as Python pickle (magic bytes: `\x80\x04`) → attempt `pickle.loads()` in a restricted environment. Detect the top-level type of the unpickled object. If it is a sklearn estimator (has `predict` and `fit_params_` attributes), flag as `sklearn_model`. If it is a dict with keys matching a PyTorch state dict pattern, flag as `pytorch_state_dict`. Never unpickle in the main process from an untrusted source — send to the sandbox.

`safetensors` → use `safetensors.safe_open()`. Return: metadata dict, tensor names and shapes.

`onnx` → use `onnx.load()`. Return: graph inputs (name, shape, type), graph outputs (name, shape, type), opset version, model IR version.

`python` (`.py`) → do not execute. Read as text. Return: line count, first 2,000 characters, detected imports.

**Unknown or Unsupported**

If detection fails or the type is not in the routing table: return `type=unknown`, `mime=<detected mime>`, `raw_bytes_hex_preview=<first 64 bytes as hex>`, and a flag `supported=false`. The Orchestrator receives this and presents a human-in-the-loop gate asking the user how to proceed.

### 6.4 Downstream Routing Based on read\_file Output

After `read_file` returns a typed result, the Ingestion Agent must route accordingly:

If `supported=false` → human-in-the-loop gate.

If `requires_ocr=true` (scanned PDF) → route to a specialised OCR sub-task in the sandbox using `pytesseract` or a vision model via Azure OpenAI's vision capability on the PDF pages rendered as images.

If the detected type is audio or video → check the user's task description. If the task is transcription, route to the Whisper scenario in the DL coverage matrix. If the task is something else, surface an unsupported warning.

If the detected type is an ML artefact (pickle, safetensors, ONNX) → the task is evaluation or deployment of an existing model. Skip to the Evaluation Agent, passing the model artefact. The Training Agent is not needed.

If the detected type is tabular → proceed through the standard pipeline: EDA → Cleaning → Feature Engineering → Training.

If the detected type is a document (PDF, DOCX with text) → route to a text-based ML scenario. The Feature Engineering Agent will use embeddings, not tabular transforms.

If the detected type is an image → route to an image-based ML scenario.

If the detected type is code → route to a code analysis scenario.

---

## 7. Agent Catalogue and System Prompts

### 7.1 Orchestrator Agent

**Model deployment:** primary reasoning model.

**Role:** owns the workflow graph, reads from the Tracking MCP Server, routes failures to the Debug Agent, manages human-in-the-loop gates, aggregates the final artifact manifest.

**System prompt design principles:** the system prompt must establish the Orchestrator's identity as a director, not an executor. It must include the full list of available stage names so the Orchestrator can construct a valid graph. It must include the routing rules for each `read_file` output type. It must include the escalation protocol (what to do when the Debug Agent exhausts all attempts). It must include the checkpoint-resume protocol.

**What the Orchestrator must never do:** call `execute_code`, perform statistical analysis, or generate training code. If you find the Orchestrator generating code, your system prompt is wrong.

### 7.2 Ingestion Agent

**Model deployment:** fast model.

**Role:** receives the raw file path from the Orchestrator, calls `read_file`, validates the parse result against the user's task description, confirms the routing decision, and records the raw artifact in the Tracking MCP Server.

**Validation logic the agent must apply:** if the task says "classify customer churn" but `read_file` returns an image, the agent must flag a routing mismatch. If the task says "train a time series model" but the data has no datetime column, the agent must flag that. These are signals the Orchestrator uses to either prompt the user for clarification or proceed with a modified plan.

### 7.3 EDA Agent

**Model deployment:** fast model for stat computation prompts; primary model for the narrative generation call.

**Role:** generates a complete statistical profile of the dataset. Produces charts via `execute_code`. Calls Azure OpenAI to generate a plain-English narrative. Records all chart file paths in the Tracking MCP Server.

**Non-negotiable outputs:** summary statistics for every column, null analysis, cardinality analysis for categoricals, distribution shapes for numerics, correlation analysis, class balance for target columns, and a data quality flag list. Any agent that does not produce all of these is incomplete.

### 7.4 Cleaning Agent

**Model deployment:** primary model.

**Role:** applies a strict chain-of-thought cleaning protocol. Documents every decision. Applies transformations via `execute_code`. Versions the output dataset with a content hash.

**Chain-of-thought protocol the system prompt must enforce (in order):**

Step one — enumerate all quality issues found in the EDA report (read it from session state, not by re-running analysis).

Step two — for each missing value column, classify the missingness pattern and select an imputation strategy with justification.

Step three — for each identified outlier, propose a handling strategy with justification. Present to the human-in-the-loop gate before applying.

Step four — apply all approved transformations via `execute_code`.

Step five — validate: re-compute null counts and compare to pre-cleaning counts. Confirm row count delta is documented.

Step six — write the transformation log to `write_output`. Hash the cleaned dataset and record the hash.

**Critical design constraint:** the Cleaning Agent must never drop a column without explicit justification recorded in the transformation log. Columns should only be dropped when they are: identifiers with no predictive value, columns with >95% missing values with no viable imputation, or columns with zero variance.

### 7.5 Feature Engineering Agent

**Model deployment:** primary model.

**Role:** researches domain-specific features via DuckDuckGo, generates and applies feature transforms via `execute_code`, runs a quick importance assessment, produces the feature manifest.

**Research-first protocol:** before generating any feature, the agent must use `web_research` to search for domain-specific feature engineering best practices. The search query is constructed from the dataset's domain (inferred from column names and the user's task description). The research results are included in the context before the feature generation prompt. This prevents the agent from applying generic transforms when domain-specific ones would be far more effective.

**Transforms to implement (all via execute_code, not imported libraries in the agent):**

For numeric columns: log transform for right-skewed distributions (detected by skewness > 2.0), square root transform for count data, standardisation (z-score), min-max scaling, polynomial features (degree 2, interacting pairs only — not full expansion), binning with automated bin count selection.

For categorical columns: one-hot encoding for cardinality ≤ 20, target encoding for cardinality > 20 (with leave-one-out to prevent leakage), ordinal encoding for ordered categories (requires the user to specify the order).

For text columns: character count, word count, average word length, punctuation density as numeric features, plus sentence embeddings via the `embed_text` MCP tool.

For datetime columns: year, month, day of week, hour, is_weekend, days_since_epoch, cyclical encoding (sin/cos) for periodic features.

Cross-column features: pairwise ratios for numeric columns with domain meaning (e.g., clicks / impressions = CTR), interaction terms for the top 5 most important numeric features (assessed by correlation with the target).

### 7.6 Model Selection and Training Agent

**Model deployment:** primary model.

**Role:** detects the task type, selects algorithms, generates training code via `execute_code` using the appropriate template, records all results in the Tracking MCP Server.

**Task type detection:** examine the target column (if classification: ≤20 unique values or dtype object; if regression: numeric with high cardinality), the data structure (image paths → image ML; text columns → NLP; datetime index → time series), and the user's explicit description.

**Classical ML algorithm selection:** for any tabular classification or regression task, the agent must run at minimum three algorithms — a linear baseline, a gradient boosting model, and an ensemble. The linear baseline establishes a sanity floor. If the linear model outperforms the gradient boosting model on validation, that is a signal that the data has a linear relationship and the feature engineering can be simplified.

**Deep Learning routing:** if the task type maps to any DL scenario in Section 14's coverage matrix, the agent must use the appropriate DL template and apply every item in the training loop checklist.

### 7.7 Hyperparameter Tuning Agent

**Model deployment:** fast model (tuning is iterative and the prompts are short).

**Role:** defines the search space appropriate to the winning model, runs Optuna inside `execute_code`, captures results, produces the `best_params` artifact.

**Search space design principle:** the search space must be model-specific. A generic search space is worse than no tuning — it wastes compute on parameters that do not matter. The agent must reason about the model type and define ranges that reflect the meaningful parameter landscape for that specific model.

**Termination criteria:** the tuning session ends when either the time budget (configurable, default 60 minutes) is exhausted or the improvement over the best observed trial drops below 0.001 (configurable) for 20 consecutive trials. The agent must detect this plateau and stop rather than continuing to consume compute.

### 7.8 Evaluation and Validation Agent

**Model deployment:** primary model.

**Role:** computes all metrics, runs fairness analysis, produces the evaluation report. This agent makes the deployment recommendation — proceed, conditional proceed (proceed after human review of flagged issues), or do not deploy.

**Metric selection by task type:**

Binary classification: accuracy, precision, recall, F1 (macro and weighted), ROC-AUC, PR-AUC, Matthews Correlation Coefficient, confusion matrix.

Multi-class classification: all of the above (OvR for ROC-AUC), Cohen's Kappa, per-class F1.

Regression: RMSE, MAE, R², MAPE, Huber Loss, residual distribution statistics.

Time series: MASE, SMAPE, coverage (for interval forecasts), DTW distance (for sequence similarity).

Deep learning: additionally training/validation loss curves, gradient norm evolution, and for NLP models: perplexity.

Fairness: demographic parity difference, equalized odds difference using Fairlearn. These metrics are mandatory for any classification model and for any regression model where the output affects people.

**Deployment recommendation logic:** the agent must produce a structured deployment recommendation object with fields: `recommend` (proceed / conditional / block), `blocking_issues` (list of issues that must be resolved before deployment), `advisory_issues` (list of issues to monitor post-deployment). If `recommend` is `block`, the `before_deploy` hook must prevent deployment from proceeding.

### 7.9 Explainability Agent

**Model deployment:** primary model.

**Role:** produces SHAP values, LIME explanations, attention maps (for DL), and a plain-English narrative. Saves all plots via `execute_code` and records file paths.

**Explainer selection by model type:** tree-based models use SHAP `TreeExplainer`. Linear models use SHAP `LinearExplainer`. Neural networks use SHAP `DeepExplainer` and Captum's `IntegratedGradients`. Image models use `GradCAM` via Captum. Text models use LIME `TextExplainer`.

**Narrative generation protocol:** the agent collects the top 10 SHAP features by mean absolute value, constructs a prompt with the feature names, their importance values, their direction (positive or negative contribution), and the model's overall performance, then calls Azure OpenAI to produce a non-technical narrative explaining why the model makes its predictions.

**Browser-Use trigger condition:** if the Explainability Agent's `execute_code` call fails with a SHAP or Captum API error, invoke `search_docs` pointing to the SHAP or Captum documentation before attempting any fix. Do not rely on training knowledge for API compatibility — SHAP's API has changed across major versions.

### 7.10 Report Agent

**Model deployment:** primary model.

**Role:** assembles all artifact paths from the Tracking MCP Server (via `record_artifact` query), reads each artifact's metadata, and produces the final report in Markdown and HTML.

**Report sections (all mandatory):** executive summary (3 paragraphs, non-technical), dataset overview (source, size, schema summary), EDA highlights (key findings from the EDA narrative), data quality summary (what was cleaned and why), feature engineering summary (features created, features selected, features dropped), model comparison table (all algorithms tried, their cross-validation metrics, their training time), best model detail (architecture, hyperparameters, final metrics), fairness analysis (protected attributes tested, metrics, recommendation status), SHAP explanation (top 10 features with inline chart), deployment recommendation, and appendix (full feature manifest, full metrics table, artifact paths for reproducibility).

**HTML rendering requirement:** all charts embedded as inline base64. The HTML file must render correctly in a browser without any external dependencies. Use a single `<style>` block for CSS. No JavaScript frameworks. This requirement exists so the report can be attached to an email and read without internet access.

### 7.11 Deployment Agent

**Model deployment:** fast model.

**Role:** packages the model, writes the FastAPI wrapper template (as a file, not as executed code), builds the Docker container, runs the container locally for a smoke test, reports the endpoint URL.

**Serialisation by model type:** sklearn models use joblib (not pickle — joblib is faster for large numpy arrays). PyTorch models use `torch.save(state_dict)` for weights plus a separate file for the model architecture class. All models also export to ONNX for portability — this is not optional. An ONNX export that fails is a signal that the model architecture is not ONNX-compatible and must be redesigned.

**Smoke test protocol:** after the container starts, the agent sends 10 prediction requests (sampled from the held-out test set), verifies each response matches the expected schema, computes the round-trip latency P99, and flags if P99 exceeds 500ms. Smoke test results are recorded in the Tracking MCP Server.

### 7.12 Memory Agent

**Model deployment:** embedding deployment for indexing, fast model for query synthesis.

**Role:** not in the linear pipeline. Subscribes to the Tracking MCP Server as an observer. After each `record_artifact` call, indexes the artifact. Exposes natural-language query capability via `semantic_search`.

**Index structure:** a simple in-process index backed by a JSON file and NumPy vectors. On startup, load the index from disk. After each new artifact, update the index and write to disk. This keeps the Memory Agent durable across restarts without requiring any vector database service.

**Query protocol:** embed the query using `embed_text`, compute cosine similarity against all indexed artifact embeddings, retrieve top-5, pass them to the fast model with a prompt that asks for a grounded answer citing the specific artifact paths.

### 7.13 Debug Agent

**Model deployment:** primary model.

**Role:** receives failing code, an error message, a traceback, and an attempt number. Returns repaired code. Never modifies pipeline state. Never calls tools other than `search_docs` and `web_research`.

**Resolution strategy sequence (see full details in Section 11):** pattern match → Browser-Use documentation crawl → DuckDuckGo search → first-principles LLM reasoning → complete rewrite. Each attempt is logged to the Tracking MCP Server via `record_error`.

---

## 8. Workflow Graph and Pipeline Orchestration

### 8.1 Graph Structure

The workflow graph is defined as a Python data structure in `workflows/pipeline_graph.py`. It is not generated dynamically. It is defined once, statically, and parameterised at runtime with the run ID and the input artifact paths. Static graph definitions are easier to reason about, easier to test, and easier to checkpoint.

Nodes in the graph:

`health_check` → `read_file` → `ingest` → `eda` → [human gate: review EDA] → `clean` → [human gate: approve outlier handling] → `feature_eng` → `train` → `tune` → `evaluate` → [human gate: review fairness] → `explain` → `report` → [human gate: approve deployment] → `deploy`

In parallel with the entire linear sequence, the Memory Agent runs as an observer node that fires asynchronously after each stage node completes. It does not block the linear sequence.

### 8.2 Conditional Routing

Between `read_file` and `ingest`, insert a routing node that examines the `read_file` result and selects the correct downstream pipeline variant. The three variants are:

Variant A (tabular): the standard linear pipeline above.

Variant B (document/image/text): skip Feature Engineering's tabular transforms; use embedding-based features; route to the appropriate DL scenario in the Training Agent.

Variant C (existing model artefact): skip Ingestion, EDA, Cleaning, Feature Engineering, and Training; route directly to Evaluation, then Explainability, then Report, then Deployment.

### 8.3 Checkpointing

After each node completes successfully, the graph engine calls `record_checkpoint` on the Tracking MCP Server with the run ID and node name. At the start of each pipeline run, the Orchestrator calls `query_run_status` to get the list of completed checkpoints. Any node whose name appears in the completed checkpoints list is skipped.

This makes the pipeline resumable after any failure. A crashed pipeline at the `tune` stage resumes from `tune`, not from `read_file`. All inputs to `tune` (the cleaned dataset, the feature manifest, the baseline model) are read from their artifact paths stored in the Tracking MCP Server.

### 8.4 Human-in-the-Loop Gates

Each gate pauses the graph and records a `pending_human_approval` status in the Tracking MCP Server. The Orchestrator polls this status before advancing.

In development mode (configurable), all gates auto-approve after a 2-second delay. This allows end-to-end testing without human interaction.

In production mode, gates block until the user sends an approval event. The mechanism for this is a REST endpoint on the Tracking MCP Server: `POST /approve/{run_id}/{gate_name}` with a JSON body containing the decision and any override parameters.

---

## 9. Middleware Pipeline — Design and Ordering

### 9.1 Registration Scope

Middleware is registered on the Azure OpenAI client, not on individual agents. One client — one middleware chain. This guarantees uniform application across all agents.

Create two clients (primary and fast models). Register the same middleware chain on both. The chain is identical — do not create a "lighter" chain for the fast model client. Telemetry and safety must apply to every call regardless of model tier.

### 9.2 Inbound Order (request path)

Register in this order. The first registered middleware wraps the outermost layer. Requests pass through in registration order. Responses pass through in reverse.

Position 1 — `LoggingMiddleware` — logs every call start with structured JSON. Fires before all other middleware so that even calls that are rejected (by safety) are logged.

Position 2 — `RateLimitMiddleware` — token bucket per agent per minute. Pauses on limit hit. Never drops — always waits and retries within the configured maximum wait.

Position 3 — `SafetyMiddleware` — inspects every tool call input against the blocked pattern list. Raises `SafetyViolationError` on match. Does not retry after a safety violation.

Position 4 — `RetryMiddleware` — exponential backoff with jitter for transient failures (HTTP 429, 502, 503, 504, timeouts). Non-transient failures (logic errors, HTTP 400, 401) pass through unretried.

Position 5 — `TelemetryMiddleware` — creates an OpenTelemetry span for every agent invocation. Attributes include run ID, session ID, model deployment name, agent name, and all tool names called. Exports to Azure Monitor (via the OTLP exporter pointed at your Application Insights connection string).

### 9.3 LoggingMiddleware — What to Capture

Every log entry is a JSON object with these fields: `timestamp` (ISO 8601 with milliseconds), `run_id`, `session_id`, `agent_name`, `event` (one of: `agent_start`, `agent_end`, `tool_call_start`, `tool_call_end`, `error`), `model_deployment`, `input_token_count`, `output_token_count`, `duration_ms`, `status` (success or error), `error_class` (if error), `error_message` (if error, first 200 characters).

Write log entries to stdout as JSONL. Do not write to disk in the agent process. In production, stdout is captured by the container runtime and forwarded to Azure Monitor Logs.

### 9.4 SafetyMiddleware — Blocked Pattern List

Maintain the blocked pattern list in `config/safety_patterns.py` as a versioned module. Never hardcode patterns inside the middleware class — the list must be editable without changing middleware logic.

The initial list must cover: shell commands that delete or move files outside `/workspace`, SQL DDL commands (`DROP TABLE`, `ALTER TABLE`, `CREATE TABLE`, `TRUNCATE`), environment variable access in generated code (`os.environ`, `os.getenv`), any outbound network call from inside the sandbox (the sandbox has `network_mode=none`, but defence-in-depth requires the pattern check too), and `__import__` used with restricted modules (`subprocess`, `socket`, `urllib`).

The list is reviewed and extended before each production release. Document who reviewed it and when.

### 9.5 RetryMiddleware — Transient vs Non-Transient Classification

**Transient** (retry with backoff): `HTTPStatusError` with status 429, 502, 503, 504; `asyncio.TimeoutError`; `aiohttp.ClientConnectorError` (network-level connection failure); Azure OpenAI capacity error (HTTP 503 with error code `capacity`).

**Non-transient** (do not retry, propagate immediately): `HTTPStatusError` with status 400, 401, 403, 404; `SafetyViolationError`; `ValidationError` (malformed tool input); `json.JSONDecodeError` (malformed model response); any `Exception` whose class is listed in a configurable non-retryable exception list.

Backoff formula: `min(base_delay * (2 ** attempt_number) + random.uniform(0, jitter_range), max_delay)`. Default: `base_delay=1.0`, `jitter_range=1.0`, `max_delay=60.0`. Maximum retry count: 3 for Azure OpenAI calls; 5 for MCP tool calls.

---

## 10. Hooks Registry — Lifecycle Events

### 10.1 Hook Semantics

A hook is a function registered for a named event. When the event fires, all hooks registered for that event execute in registration order. A hook can: pass through (return the event data unchanged), modify (return modified event data), or abort (raise `HookRejectionError`, which stops the event and propagates an error).

Hooks are finer-grained than middleware. Middleware wraps entire agent invocations. Hooks attach to specific named events within an invocation.

### 10.2 Required Hooks

`before_code_execute` — fires before any Python code is submitted to the sandbox. Input: the code string. Expected output: the (possibly modified) code string. Required action: run `bandit` (security linter) on the code. If any HIGH-severity finding, raise `HookRejectionError` with the finding detail. Run `ast.parse()` to verify the code is syntactically valid before submission. Return the code.

`after_code_execute` — fires after the sandbox returns a result, regardless of success or failure. Input: the result object (stdout, stderr, exit code, output file list). Required action: if success, upload output files to the current run's output directory and record their paths in the Tracking MCP Server via `record_artifact`. If failure, trigger the `on_code_error` hook.

`on_code_error` — fires when `execute_code` returns a non-zero exit code. Input: the code, the stderr, the exit code, the current attempt number. Required action: route to the Debug Agent (Section 11.2). Increment the attempt counter. Return the repaired code for re-execution.

`before_model_train` — fires before any training job is submitted via `execute_code`. Input: training code, dataset path, feature manifest path. Required action: verify the dataset path exists and is accessible. Verify the feature manifest path exists. Verify the training code contains all items in the training loop checklist (checked via keyword search for each checklist item's key function names). If any verification fails, raise `HookRejectionError`.

`after_model_train` — fires after a training job completes successfully. Input: the result object including the metrics JSON written to `/workspace/output`. Required action: call `log_metrics` MCP tool to persist the metrics. Call `record_artifact` to register the model file path. Record the metrics in the run's session state for downstream agents.

`before_deploy` — fires before the Deployment Agent submits its packaging job. Input: the model artifact path, the evaluation report path. Required action: verify the evaluation report path exists. Read the `recommend` field. If `recommend == "block"`, raise `HookRejectionError` with the blocking issues. If `recommend == "conditional"`, trigger a human-in-the-loop gate and wait for approval before returning.

`after_deploy` — fires after the deployment container is running. Input: the endpoint URL, the test sample paths. Required action: call the endpoint 10 times with sample inputs. Verify response schema on each call. Record P99 latency. If any call fails or P99 > 500ms, record an advisory warning in the Tracking MCP Server. Do not block deployment for P99 violations — record and continue.

`on_max_debug_attempts` — fires when the Debug Agent has exhausted all repair attempts. Input: original code, all repair attempts, all error messages. Required action: write the full debug report to `write_output`. Trigger the human-in-the-loop gate for this stage. Do not retry automatically.

`before_mcp_tool_call` — fires before any MCP tool call is dispatched. Input: tool name, tool input dict. Required action: apply the safety pattern check against the tool input (same patterns as the SafetyMiddleware, but at the granularity of individual tool inputs rather than entire completions). Log the tool call to the Tracking MCP Server via `record_tool_call`.

---

## 11. Code Sandbox — Execution, Auto-Fix, and Debug Loop

### 11.1 Sandbox Contract

The sandbox is a Docker container. Its contract is:

- Accepts Python source code via a mounted file at `/workspace/code.py`
- Has read access to data files mounted at `/workspace/data/`
- Has read-write access to `/workspace/output/` for artifacts
- Has no outbound network access (`--network none`)
- Runs as a non-root user (`uid=1000`)
- Is killed after the wall-clock timeout (default 300 seconds)
- Returns: exit code, stdout (max 50KB, truncated after), stderr (max 50KB), list of files in `/workspace/output/`

The sandbox image is built from `python:3.12-slim`. Do not use full images — they are too large for fast iteration. Install only the data science libraries required (see Section 24 for the exact requirements file). Pin every library to an exact version. Rebuild the image when any library is updated. Tag the image with the SHA-256 of the requirements file.

### 11.2 Execution Flow

The `execute_code` MCP tool receives source code. Before submitting to the sandbox:

The `before_code_execute` hook fires. If it raises, execution stops and the error is returned.

The code is written to a temporary file in the host's scratch directory.

The Docker container is started with the code file mounted at `/workspace/code.py` and the relevant data files mounted at `/workspace/data/` (read-only).

The container runs `python /workspace/code.py`. stdout and stderr are streamed in real time. If the wall-clock timeout expires, the container is killed and a `TimeoutError` result is returned.

After the container exits, output files from `/workspace/output/` are copied to the current run's output directory. The `after_code_execute` hook fires with the full result.

### 11.3 Auto-Fix Loop

The auto-fix loop runs inside the `on_code_error` hook. It calls the Debug Agent and submits the repaired code back to `execute_code`. The loop continues until success or `max_debug_attempts` is reached.

**Attempt 1 — Pattern Match (< 100ms)**

Check the error class against the Error Catalogue (Section 26). If a matching pattern exists, apply the pre-defined fix transformation directly (do not call the LLM). Pattern fixes are deterministic and fast. Examples: `ModuleNotFoundError: No module named 'X'` → prepend the correct import statement. `KeyError: 'column_name'` → look up the feature manifest and substitute the correct column name.

**Attempt 2 — Browser-Use Documentation Crawl (10–30 seconds)**

Infer the failing library from the traceback (look for the first non-`/workspace` file path in the traceback — the library is named in that path). Look up the documentation URL in the library registry. Call `search_docs` with the error message and the documentation URL. Present the extracted documentation text, the original code, and the error to the primary model. Ask it to return repaired code only.

**Attempt 3 — DuckDuckGo Search (5–15 seconds)**

Call `web_research` with a query constructed from the error class, error message, library name, and Python version. Present the top-5 results, the original code, and the error to the primary model. Ask it to return repaired code only.

**Attempt 4 — First-Principles Reasoning (no search)**

Present the original task description (not the failing code — start clean), the error, and the traceback to the primary model. Ask it to reason step-by-step about what the code is trying to accomplish, why the error occurs, and what the correct implementation is. Ask it to return repaired code only. This prompt must explicitly say "Do not repeat the failing code structure. Reason from first principles."

**Attempt 5 — Complete Rewrite**

Present the original task description only (no code, no error). Ask the primary model to solve the task using the simplest, most reliable approach available. The rewrite may use a different library or a different algorithmic approach entirely. This is an acceptable outcome — the goal is correctness, not adherence to the original approach.

**Escalation**

If Attempt 5 fails, fire `on_max_debug_attempts`. Write the full debug report. Trigger the human-in-the-loop gate. The pipeline pauses.

---

## 12. Browser-Use Integration — Docs and Error Crawling

### 12.1 When Browser-Use Is Invoked

Browser-Use is invoked in exactly two situations, both inside the `search_docs` MCP tool:

Situation 1: the Debug Agent requests documentation for a specific library error (Attempt 2 of the auto-fix loop).

Situation 2: the Explainability Agent encounters an API compatibility error with SHAP, LIME, or Captum.

Do not use Browser-Use for general search (that is DuckDuckGo's job) or for any task that does not require navigating a multi-page documentation site.

### 12.2 Configuration

Browser-Use runs in local headless mode. Use `ChatAzureOpenAI` as the LLM driver for the Browser-Use agent. This means Browser-Use's navigation reasoning also uses your Azure OpenAI deployment — consistent authentication, consistent billing, no additional API keys.

Set `max_steps=50`. Most documentation sites are reachable within 10 steps. 50 is a safety ceiling. Do not set it lower than 30 — complex API reference pages with multi-level navigation require more steps.

In the `search_docs` MCP tool's implementation, configure the Browser-Use agent with `flash_mode=False` — you need the full reasoning capability for documentation extraction, not maximum speed.

### 12.3 Library Documentation Registry

Maintain a dictionary mapping library names to their documentation root URLs. This registry is the only configuration needed for Browser-Use — the agent navigates autonomously from the root URL.

The registry must include: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `torch` (PyTorch), `torchvision`, `transformers` (Hugging Face), `datasets` (Hugging Face), `sentence-transformers`, `shap`, `lime`, `captum`, `optuna`, `pyarrow`, `openpyxl`, `pymupdf`, `pillow`, `magika`, `fastmcp`, `agent-framework`.

When the Debug Agent infers a library from a traceback but that library is not in the registry, fall through to DuckDuckGo search immediately (do not attempt Browser-Use with an unknown URL).

### 12.4 Output Handling

Browser-Use returns a result object. Access the `final_result` field. Truncate to 4,000 characters before including in any LLM prompt. If `final_result` is empty or contains content clearly unrelated to the query (e.g., the browser navigated to a login page), log a `browser_use_miss` event and skip to the next debug attempt. Do not retry Browser-Use on the same query within the same debug session.

---

## 13. DuckDuckGo Deep Research Integration

### 13.1 When DuckDuckGo Is Invoked

Three situations:

1. Debug Agent Attempt 3 — error message search against GitHub Issues and Stack Overflow.
2. Feature Engineering Agent — domain-specific feature research before generating transforms.
3. Model Selection Agent — algorithm and architecture research for unfamiliar problem domains.

### 13.2 Query Construction Rules

**For debugging queries:** include the exact Python error class (e.g., `ValueError`), the first line of the error message (truncated to 100 characters), the library name, and one of `site:stackoverflow.com OR site:github.com`. This scopes results to the most useful sources.

**For feature engineering research:** include the domain name inferred from column names and task description (e.g., `"customer churn prediction"`, `"financial time series"`, `"medical image classification"`), the word `features`, and the current year to bias toward recent work.

**For algorithm research:** include the task type (e.g., `"tabular classification"`, `"time series forecasting"`), relevant keywords from the user's constraints (e.g., `"imbalanced"`, `"small dataset"`, `"low latency inference"`), and terms like `"benchmark 2025 2026"` to get recent comparisons.

### 13.3 Rate Limiting

Apply a global rate limit of 30 DuckDuckGo searches per hour tracked in the Tracking MCP Server via `record_tool_call`. If the limit is reached, return a `rate_limit_exceeded` result from the `web_research` tool. The calling agent must proceed to its next strategy without waiting.

### 13.4 Result Processing

Return the top 10 results as a structured list. Each entry: `rank`, `title`, `url`, `snippet` (first 300 characters of the search snippet). Pass this structure to the LLM without modification — do not summarise before passing, as the LLM may use different parts of different snippets than you would expect.

---

## 14. Deep Learning and Generative AI Coverage Matrix

### 14.1 Scenario Detection

The Model Selection Agent must classify every incoming task into exactly one scenario. Detection order: explicit user statement overrides everything. If the user says "train a BERT model," that is the scenario. If the user is silent on architecture, infer from data characteristics.

The full scenario list and their primary detection signals:

| Scenario | Primary Detection Signals | Template |
|---|---|---|
| Binary Classification | Target has 2 unique values; tabular data | `templates/sklearn_train.py` |
| Multi-class Classification | Target has 3–100 unique values; tabular | `templates/sklearn_train.py` |
| Regression | Target is numeric; tabular data | `templates/sklearn_train.py` |
| Imbalanced Classification | Class ratio > 10:1 in the target | `templates/imbalanced_train.py` |
| Time Series Forecasting | Datetime index + numeric target | `templates/dl/time_series.py` |
| Text Classification | Text column is the primary feature | `templates/dl/bert_classify.py` |
| Named Entity Recognition | Text column; labels are BIO-tagged sequences | `templates/dl/bert_ner.py` |
| Text Generation / Fine-tuning | Task is language modelling or instruction following | `templates/dl/llm_finetune.py` |
| Image Classification | Image file paths as features | `templates/dl/cnn_classify.py` |
| Object Detection | Image paths + bounding box labels | `templates/dl/yolo_detect.py` |
| Image Generation | Task is generative; no labelled output | `templates/dl/diffusion.py` |
| Tabular Deep Learning | Tabular data; user specifies deep learning | `templates/dl/tabnet.py` |
| Clustering | No target column; task is grouping | `templates/sklearn_cluster.py` |
| Anomaly Detection | Highly imbalanced; task is outlier detection | `templates/anomaly.py` |
| RAG System | Document corpus; task is Q&A or retrieval | `templates/dl/rag.py` |
| Embedding Fine-tuning | Task is improving embeddings for retrieval | `templates/dl/embedding_finetune.py` |
| LLM Evaluation | Existing LLM outputs; task is evaluation | `templates/dl/llm_eval.py` |
| Audio Transcription | Audio files detected by `read_file` | `templates/dl/whisper.py` |
| Multimodal | Both image and text features | `templates/dl/multimodal.py` |

### 14.2 Deep Learning Training Loop Checklist

This checklist is verified by the `before_model_train` hook. Every item must be present in the generated code before training begins.

Data pipeline: `DataLoader` with configurable `batch_size`; appropriate augmentation for the data modality; normalisation matching the pre-trained model's expected statistics; class weighting or `WeightedRandomSampler` if class imbalance is detected; `num_workers` set to the number of available CPU cores minus one (minimum 1).

Model: clearly defined input shape assertion; clearly defined output shape; dropout for regularisation (rate configurable, default 0.1 for transformers, 0.3 for CNNs); batch normalisation for CNNs; correct attention mask handling for transformer models.

Loss function: task-appropriate. Binary classification: `BCEWithLogitsLoss`. Multi-class: `CrossEntropyLoss` with optional class weights. Regression: `MSELoss` or `HuberLoss`. Imbalanced: `FocalLoss`.

Optimiser: `AdamW` with `weight_decay` (default 0.01). Document deviation.

LR scheduler: one of `CosineAnnealingLR` (general purpose), `OneCycleLR` (when training from scratch), or `LinearSchedulerWithWarmup` from transformers (for pre-trained model fine-tuning). The warmup period must be 10% of total training steps for pre-trained model fine-tuning.

Mixed precision: `torch.autocast(device_type='cuda', dtype=torch.float16)` context manager around the forward pass and loss computation.

Gradient clipping: `torch.nn.utils.clip_grad_norm_` with `max_norm=1.0` before `optimizer.step()`.

Early stopping: patience configurable (default 10 epochs), minimum delta configurable (default 1e-4), monitored on validation loss.

Checkpointing: `torch.save(model.state_dict(), best_model_path)` when validation metric improves. Resume from best checkpoint at end of training.

Reproducibility: `torch.manual_seed`, `np.random.seed`, `random.seed`, and `torch.cuda.manual_seed_all` all set to the same seed value. Seed value logged to metrics file.

Logging: training loss, validation loss, primary metric, and learning rate written to `metrics.json` in `/workspace/output/` at every epoch.

GPU memory: `torch.cuda.empty_cache()` at the start of each epoch. Gradient accumulation enabled when the configured `batch_size * gradient_accumulation_steps` exceeds GPU memory.

### 14.3 Deep Learning Auto-Remediation Rules

These rules are applied by the Debug Agent's pattern matching (Attempt 1 of the auto-fix loop) for DL-specific errors:

`CUDA out of memory` → halve `batch_size`. If still OOM, enable `gradient_checkpointing_enable()` on transformer models, or manually add `torch.utils.checkpoint.checkpoint` calls for custom models. If still OOM, switch `device` to `cpu` and log a warning.

`loss is nan or inf at epoch 1` → the learning rate is too high or the input data contains NaN/Inf. Check data: `torch.isnan(batch).any()`. If data is clean, reduce LR by factor of 10.

`loss does not decrease for 5 epochs` → apply learning rate warmup over the first 10% of steps, verify weight initialisation is using PyTorch defaults, try reducing model depth by one layer.

`validation loss diverges from training loss` (overfitting signal after epoch 5) → increase dropout by 0.1, add `weight_decay=0.01` if not already set, apply stronger augmentation, reduce `patience` for early stopping to 5.

`training accuracy equals class prior` (model not learning) → verify labels are not all the same value, verify the loss function matches the task, verify the target tensor dtype is correct (`long` for cross-entropy, `float` for BCE).

`DataLoader worker process crashed` → reduce `num_workers` to 0 and retry. If that succeeds, gradually increase `num_workers` by 1 until the crash recurs. Set `num_workers` to one below the crash threshold.

`RuntimeError: NCCL` (distributed training) → reduce `nproc_per_node` to 1 (fallback to single-GPU). Log a warning that distributed training is unavailable.

`HuggingFace model load fails with KeyError` → invoke `search_docs` on the HuggingFace model card for the specific model. The architecture may have changed across versions. Fall back to a smaller variant of the same model family.

---

## 15. Session Management with Streamable HTTP

### 15.1 Session Hierarchy

Each pipeline run creates one Orchestrator session. Within that session, the Orchestrator creates child sessions for each stage agent. This hierarchy is:

```
Orchestrator Session (run_id: abc123)
├── Ingestion Session (stage: ingest, session_id: ingest-xyz)
├── EDA Session (stage: eda, session_id: eda-xyz)
├── Cleaning Session (stage: clean, session_id: clean-xyz)
... (one session per stage)
└── Memory Session (stage: memory, session_id: memory-xyz, async)
```

The `run_id` is a UUID generated by the Orchestrator at the start of every pipeline run. All sessions, all MCP tool calls, all artifact records, and all log entries carry this `run_id`. This is the primary correlation key for debugging and retrospective queries.

### 15.2 Session State Schema

Each agent session's key-value state must conform to a defined schema. This prevents agents from storing ad-hoc keys that other agents cannot rely on.

The canonical session state keys are:

`run_id` — string, the pipeline run UUID. Set by the Orchestrator, read-only for all other agents.

`stage` — string, the current stage name. Set by each agent for its own session.

`input_artifact_paths` — list of strings, the artifact paths passed to this agent from the Orchestrator.

`output_artifact_paths` — list of strings, populated by this agent as it produces artifacts.

`metrics` — dict, key-value metrics relevant to this stage. EDA agent writes statistics here. Training agent writes model metrics here.

`feature_manifest_path` — string, set by the Feature Engineering Agent, read by Training and Evaluation agents.

`model_artifact_path` — string, set by the Training Agent, read by Tuning, Evaluation, Explainability, and Deployment agents.

`best_params_path` — string, set by the HP Tuning Agent, read by Evaluation.

`evaluation_report_path` — string, set by the Evaluation Agent, read by Explainability and Deployment.

`deployment_recommendation` — dict with fields `recommend`, `blocking_issues`, `advisory_issues`. Set by Evaluation, read by Deployment.

### 15.3 Session Resumability Design

Each stage agent must be designed to be idempotent: running it twice with the same inputs produces the same output. This is the foundation of resumability.

Idempotency implementation: at the start of every stage, the agent calls `query_run_status` and checks if this stage already has a `record_end` entry with status `success`. If yes, retrieve the output artifact paths from that record and return them without re-executing. This is the fast path for resuming after a crash.

The slow path (actual execution) only runs when no successful checkpoint exists for this stage.

### 15.4 MCP Session Reuse

The `MCPStreamableHTTPTool` instances for the Tracking MCP Server and DS Tools MCP Server are constructed once at application startup and reused across all agent sessions. Do not create new `MCPStreamableHTTPTool` instances per pipeline run or per agent — this would create unnecessary MCP session overhead and slow down the startup of each stage.

---

## 16. In-Process Memory and Context Providers

### 16.1 Chat History Provider

Use the Framework's built-in in-memory chat history provider with a `max_messages` limit. Set `max_messages=40` for the Orchestrator session (it needs longer history to track the full pipeline status) and `max_messages=20` for all sub-agent sessions.

The chat history provider stores `ChatMessage` objects. Do not store large data payloads in chat messages. Store artifact paths (short strings), not artifact contents (potentially megabytes). When an agent needs to inspect an artifact's content, it calls `get_data_sample` or reads it via `execute_code` inside the sandbox.

### 16.2 Persistent Key-Value State

The session's key-value state (described in Section 15.2) is the mechanism for passing structured data between tool calls within a single agent's session. Use it for paths, metric values, and configuration decisions. Never use it for raw data (DataFrames, model weights, image tensors). Those live in files, accessed by path.

### 16.3 Memory Agent's Index

The Memory Agent maintains an in-process index backed by two files:

`memory/artifact_index.json` — a JSON array where each entry is an artifact record: artifact path, artifact type, producing agent, run ID, task type, dataset name, metric values, timestamp.

`memory/artifact_embeddings.npy` — a NumPy array of shape `(n_artifacts, embedding_dim)` where each row is the embedding of the corresponding artifact record's text representation.

On startup, the Memory Agent loads both files. On new artifact: embed the artifact record text (via `embed_text`), append to both files, write both files to disk. The write is synchronous within the Memory Agent but asynchronous relative to the main pipeline (the Memory Agent runs in a separate thread or process).

---

## 17. Phase 1 — Foundation, Environment, MCP Servers

### Goal

Both MCP servers are running, a hello-world agent call succeeds with the full middleware chain, the sandbox executes a test script, and the repository structure is established.

### Step 1.1 — Repository Initialisation

Create the Git repository. Add `.gitignore` covering `.env`, `__pycache__`, `.venv`, `*.pyc`, `*.egg-info`, `dist/`, `*.db` (SQLite files — never commit the Tracking MCP Server's database), `outputs/` (pipeline outputs — too large for git), and any file larger than 5MB. Create `pyproject.toml` with all direct dependencies pinned to exact versions.

### Step 1.2 — Azure OpenAI Resource and Deployments

In the Azure portal, create (or confirm you have access to) one Azure OpenAI resource. Create the three deployments: primary reasoning, fast execution, and embedding. Document the deployment names in `.env.example`. Grant your developer account the `Cognitive Services OpenAI User` role on the resource. Verify access with `az cognitiveservices account keys list` — not to use the key, but to confirm the resource is accessible.

### Step 1.3 — Python Environment

Create a virtual environment with Python 3.12. Install the packages in the order specified in Section 3.2. Verify: `python -c "import agent_framework; print(agent_framework.__version__)"` prints `1.0.x`. Verify: `python -c "import mcp; print(mcp.__version__)"` succeeds. Verify: `python -c "from agent_framework.openai import AzureOpenAIChatClient"` succeeds without import errors.

### Step 1.4 — Tracking MCP Server

Implement the Tracking MCP Server using FastMCP. Create the SQLite schema with WAL mode enabled. Expose all tools listed in Section 5.2. Mount the FastMCP app into a FastAPI parent app at `/mcp`. Add a `/health` endpoint. Start the server. Verify compliance:

`curl -I http://localhost:8100/` returns `405 Method Not Allowed` with `Allow: POST` header.

`curl -X HEAD http://localhost:8100/` returns `200 OK`.

`curl -X POST http://localhost:8100/mcp/mcp -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}},"id":1}'` returns a response with an `Mcp-Session-Id` header.

### Step 1.5 — DS Tools MCP Server (Stubs)

Implement the DS Tools MCP Server with stub implementations that return hardcoded success responses. All tools from Section 5.3 must be registered. Run the same three `curl` compliance checks on port `8101`.

### Step 1.6 — Middleware Chain

Implement all five middleware classes. Write unit tests for each:

`LoggingMiddleware`: after one agent call, exactly one JSON log entry appears in stdout containing all required fields.

`RateLimitMiddleware`: when the bucket is exhausted, the next call waits (do not fail immediately). Verify with a sleep-based test.

`SafetyMiddleware`: a blocked pattern in a tool input raises `SafetyViolationError`. A clean tool input passes through unchanged.

`RetryMiddleware`: an HTTP 429 response triggers a retry with a delay. An HTTP 400 response propagates immediately without retry.

`TelemetryMiddleware`: an OpenTelemetry span is created and closed after each agent call. Verify the span has the required attributes.

### Step 1.7 — Hooks Registry

Implement the hooks registry and all hooks from Section 10.2. Write unit tests verifying each hook fires at the correct lifecycle event.

### Step 1.8 — Docker Sandbox Image

Write `Dockerfile.sandbox` starting from `python:3.12-slim`. Install only the packages in `sandbox/requirements.sandbox.txt` (see Section 24 for contents). Create a non-root user `sandbox` with `uid=1000`. Set `USER sandbox`. The entry point is `python /workspace/code.py`.

Build the image. Tag it with the SHA-256 of `requirements.sandbox.txt`. Verify:

A script that prints `hello` executes and returns exit code 0.

A script that does `import requests; requests.get("http://example.com")` raises a `ConnectionError` (no outbound network).

A script that tries to write to `/etc/passwd` raises a `PermissionError` (non-root user).

### Step 1.9 — Hello-World Integration Test

Create a minimal agent using `AzureOpenAIChatClient` (pointing to your fast execution deployment), registered with the full middleware chain and both MCP servers. Send it the message `"What is 2 + 2?"`. Verify:

The response contains `4`.

The Tracking MCP Server's `record_start` tool was called (check the SQLite database).

A log entry appears in stdout.

An OpenTelemetry span was exported (check the console exporter output).

### Phase 1 Exit Gate

- Both MCP servers pass the three `curl` compliance checks
- All middleware unit tests pass
- All hook unit tests pass
- Docker sandbox passes all three verification scripts
- Hello-world integration test passes end-to-end
- CI pipeline runs lint (`ruff`), type check (`mypy`), and all tests on every push

---

## 18. Phase 2 — read\_file Tool and Data Ingestion

### Goal

The `read_file` tool correctly detects and reads every file type in the routing table. The Ingestion Agent uses it and records the raw artifact.

### Step 2.1 — Magika and filetype Installation

Install `magika` and `filetype` in the main environment (not the sandbox). Verify Magika loads its model: `from magika import Magika; m = Magika(); r = m.identify_path('any_test_file'); print(r.output.label)`. The first call loads the model (~5ms overhead). Subsequent calls are fast.

### Step 2.2 — FileTypeDetector Class

Implement `tools/file_type_detector.py` with the three-layer detection strategy from Section 6.2. This class is a pure Python utility — no LLM calls. It returns a `FileTypeResult` dataclass with fields: `detected_type`, `mime_type`, `confidence`, `category`, `supported`, `requires_ocr`, `detection_layers_used` (which of the three layers was needed).

Write comprehensive unit tests covering:

Every file type in the routing table (use small test files). Confirm the correct `detected_type` is returned.

A CSV file renamed as `.json` (extension mismatch). Confirm the true type is detected.

A file with corrupted magic bytes. Confirm the fallback to Layer 3 works.

An empty file. Confirm a graceful `unsupported` result is returned.

### Step 2.3 — Reader Implementations

Implement one reader class per file type category. Each reader is a pure Python class with a single method: `read(path: str) -> FileReadResult`. The `FileReadResult` dataclass carries the type-specific fields described in the routing table (Section 6.3).

Implement readers in this priority order (most common data science types first): CSV, Parquet, JSON, JSONL, Excel (xlsx + xls), SQLite, PDF, DOCX, PNG/JPEG/image types, ZIP, pickle, ONNX, safetensors.

For each reader, write integration tests with real (small) sample files that cover: normal files, empty files, files with encoding issues (for text formats), and malformed files.

### Step 2.4 — read\_file MCP Tool

Connect the `FileTypeDetector` and the reader implementations into the `read_file` MCP tool implementation in the DS Tools MCP Server. The tool:

Runs `FileTypeDetector.detect(path)`.

Routes to the appropriate reader based on `detected_type`.

If `supported=false`, returns the `unknown` result without calling any reader.

If `requires_ocr=true`, returns the OCR-required result without attempting text extraction.

Records the detection result in the Tracking MCP Server via `record_tool_call`.

Returns the `FileReadResult` serialised as JSON via `ToolResult`.

### Step 2.5 — Ingestion Agent

Implement the Ingestion Agent (Section 7.2). Connect to both MCP servers. Register the middleware chain and hooks. Write the system prompt establishing validation logic.

Test with five different file types: CSV, Parquet, PDF (text-based), XLSX, and JSON. For each, verify: the correct `detected_type` is returned, the correct reader is used, the artifact is recorded in the Tracking MCP Server, and the routing decision is logged in the Orchestrator session state.

Test with one intentionally mistyped file (rename a Parquet file as `.csv`). Verify the correct type is still detected.

### Phase 2 Exit Gate

- `FileTypeDetector` passes all unit tests including edge cases
- All reader implementations pass their integration tests
- `read_file` MCP tool correctly handles all five test files and the mistyped file
- Ingestion Agent produces a correct artifact record in the Tracking MCP Server for all test files
- A CSV with mixed encodings is correctly ingested with the detected encoding

---

## 19. Phase 3 — EDA, Cleaning, Feature Engineering

### Goal

The first three analytical pipeline agents run end-to-end on a real dataset, producing real artifacts, with the auto-fix loop verified.

### Step 3.1 — execute_code MCP Tool

Replace the stub `execute_code` implementation with the real Docker sandbox integration. The tool must: accept source code, write it to a temp file, mount it and the relevant data files into the sandbox container, run with resource limits and timeout, collect output files, and return the full result.

Test with three scripts: one that succeeds and writes an output file, one that raises a Python exception, and one that runs for longer than the timeout. Verify all three return the correct result structure.

### Step 3.2 — EDA Agent

Implement the EDA Agent (Section 7.3). Create the EDA code template in `templates/eda_template.py`. The template is parameterised by column names, dtypes, and target column. The agent's job is to fill in the parameters and submit the template via `execute_code`.

The template must produce: a `stats.json` file in `/workspace/output/` containing all statistics, and PNG chart files for all required visualisations. The agent records all output file paths in the Tracking MCP Server.

Test with the Titanic dataset. Verify the `stats.json` exists and contains all required fields. Verify all expected chart files are produced.

**Auto-fix loop injection test:** remove one import from the template (e.g., `import seaborn`). Verify the auto-fix loop adds the import back, the code re-executes successfully, and the attempt count reaches exactly 1 (pattern match catch).

### Step 3.3 — Cleaning Agent

Implement the Cleaning Agent (Section 7.4) and its cleaning code template. The template must handle: median/mode imputation, KNN imputation (using `sklearn.impute.KNNImputer`), IQR outlier detection, and Z-score outlier detection.

The human-in-the-loop gate for outlier handling must be implemented. In development mode, it auto-approves. In production mode, it blocks.

Test with the Titanic dataset (which has missing `Age` and `Cabin` values and outliers in `Fare`). Verify: `Age` is imputed with median (MCAR classification), `Cabin` is dropped (>95% missing), `Fare` outliers are flagged in the transformation log. Verify the output cleaned dataset has zero null values in the columns that were imputed.

### Step 3.4 — Feature Engineering Agent

Implement the Feature Engineering Agent (Section 7.5) and the feature engineering code template.

**DuckDuckGo research integration test:** before the first feature generation, verify that `web_research` is called with a domain-relevant query. Verify the results are included in the feature generation prompt.

Test with the Titanic dataset. Verify the feature manifest lists: all original columns with their transforms, at minimum one interaction term, one-hot encoding for `Sex` and `Embarked`, ordinal encoding for `Pclass`, and SHAP-based importance scores from the quick RandomForest run. Verify no column appears in the manifest without a transform entry.

### Step 3.5 — Workflow Integration (Phases 1–3)

Wire Ingestion → EDA → Cleaning → Feature Engineering into the workflow graph with checkpoint nodes between each stage. Run the full sub-pipeline on the Titanic dataset. Verify:

All four stages complete successfully.

The Tracking MCP Server contains `record_start` and `record_end` entries for all four stages.

Simulate a crash at the Cleaning stage (raise an exception mid-run). Verify the pipeline resumes from Cleaning on the next run (not from Ingestion).

### Phase 3 Exit Gate

- `execute_code` handles success, exception, and timeout correctly
- EDA Agent produces all required outputs for Titanic
- Auto-fix loop recovers from one injected error in the EDA template
- Cleaning Agent produces a null-free cleaned dataset for Titanic
- Feature Engineering Agent produces a complete feature manifest with DuckDuckGo research logged
- Full Ingestion → Feature Engineering pipeline completes and resumes correctly after a simulated crash

---

## 20. Phase 4 — Model Training, Tuning, and Evaluation

### Goal

Classical ML and at least two DL scenarios run end-to-end. All metrics are logged. The Evaluation Agent produces a complete report including fairness metrics.

### Step 4.1 — Training Templates

Write `templates/sklearn_train.py`. This template accepts: dataset path, feature list, target column, algorithm name, and algorithm hyperparameters. It trains the algorithm, computes cross-validation metrics, and writes the trained model file and metrics JSON to `/workspace/output/`.

Write `templates/dl/bert_classify.py`. This template must satisfy every item in the DL training loop checklist from Section 14.2 without exception. Every checklist item must be explicitly present in the template as implemented code, not as a comment.

Write `templates/dl/lstm_timeseries.py`. Same checklist requirement.

### Step 4.2 — Model Selection Agent

Implement the Model Selection Agent (Section 7.6). The agent must: detect the task type, select templates, fill in template parameters, submit via `execute_code`, capture results, and record the best model in the Tracking MCP Server.

Test with Titanic (classification): verify at minimum three algorithms are tried. Verify a RandomForest outperforms logistic regression on this dataset (if not, the feature engineering is incorrect — fix it before advancing).

Test with Air Passengers (time series): verify the LSTM template is selected, the training checklist is satisfied (verified by the `before_model_train` hook), and the model file is produced.

### Step 4.3 — Auto-Remediation for DL

Inject the OOM error into the LSTM training by setting a batch size that exceeds the sandbox memory limit. Verify the auto-remediation halves the batch size. Inject a NaN loss by using an extreme learning rate. Verify the auto-remediation detects NaN and reduces the LR.

### Step 4.4 — HP Tuning Agent

Implement the HP Tuning Agent (Section 7.7). The Optuna study must run inside `execute_code`. The agent defines the search space in its prompt, generates the Optuna study code, submits it, and captures the `best_params.json` from `/workspace/output/`.

Test with Titanic. Verify the tuned model's cross-validation AUC is at least 0.01 higher than the baseline model's AUC. If not, the search space is too narrow — expand it.

### Step 4.5 — Evaluation Agent

Implement the Evaluation Agent (Section 7.8). The evaluation code template must: load the tuned model and the test set, compute all task-appropriate metrics, run Fairlearn analysis with `Sex` as the protected attribute on Titanic, detect and report data drift between training and test distributions.

Test with Titanic. Verify all required metrics are present in the output `evaluation_report.json`. Verify the Fairlearn analysis runs and produces a `fairness_metrics` section. Verify the deployment recommendation object has all three required fields.

### Phase 4 Exit Gate

- Classical ML pipeline produces a tuned model with at least 0.80 ROC-AUC on Titanic
- LSTM trains on Air Passengers without manual intervention
- OOM auto-remediation fires correctly when batch size is oversized
- NaN loss auto-remediation fires correctly
- Evaluation report contains all required sections
- Fairlearn analysis runs on Titanic with `Sex` as protected attribute

---

## 21. Phase 5 — Explainability, Reporting, Deployment

### Goal

SHAP values are computed, the HTML report is self-contained, and the deployment container passes the smoke test.

### Step 5.1 — Explainability Agent

Implement the Explainability Agent (Section 7.9). The SHAP computation runs via `execute_code`. The agent selects the correct SHAP explainer based on the model type read from session state.

**Browser-Use trigger test:** intentionally use an incorrect SHAP API call in the template (e.g., call `shap.Explainer` with incorrect arguments for a tree model). Verify that Browser-Use crawls the SHAP documentation and the Debug Agent repairs the call on attempt 2.

Test with Titanic's RandomForest model. Verify: a beeswarm plot PNG is produced, a bar plot PNG is produced, a `shap_values.npy` file is produced, and a `narrative.md` containing the plain-English explanation is produced.

### Step 5.2 — Report Agent

Implement the Report Agent (Section 7.10). The report template is a Python string template (not Jinja, not Mako — a plain f-string template is sufficient and has no external dependencies). The agent fills in the template with data from session state and artifact files.

Render the HTML report. Open it in a browser (local development) and verify: all sections are present, all charts are embedded as base64, no external URLs are referenced (check with a network inspector), and the document renders correctly when saved to disk and opened without internet access.

### Step 5.3 — Deployment Agent

Implement the Deployment Agent (Section 7.11). The FastAPI wrapper template is a Python file written to `/workspace/output/app.py` via `write_output`. The Dockerfile is written to `/workspace/output/Dockerfile`.

The Deployment Agent must not use `execute_code` to run Docker commands inside the sandbox. Docker build and run happen in the host environment, not in the sandbox. The agent calls `write_output` to produce the files, then the host-side `deploy_endpoint` MCP tool runs `docker build` and `docker run`.

Implement the `deploy_endpoint` MCP tool in the DS Tools MCP Server. It reads the output files, builds the Docker image, starts the container, waits for the health check endpoint to return `200`, then returns the endpoint URL.

Run the smoke test. Verify 10 prediction requests succeed.

### Phase 5 Exit Gate

- SHAP beeswarm and bar plots are produced for Titanic model
- Browser-Use documentation crawl fires correctly on the injected SHAP API error
- HTML report is self-contained (verified with offline browser test)
- Deployment container starts and smoke test passes 10/10 predictions
- `before_deploy` hook correctly blocks deployment when evaluation recommendation is `block`

---

## 22. Phase 6 — Memory Agent and Retrospection

### Goal

The Memory Agent indexes all artifacts from a completed pipeline run. Three retrospective natural-language queries return correct, grounded answers.

### Step 6.1 — Memory Agent Implementation

Implement the Memory Agent (Section 7.12). Set it up as a subscriber to the Tracking MCP Server: poll `query_artifact_lineage` every 30 seconds and index any artifact not yet in the in-process index.

Alternatively, implement direct event-driven notification: when the Tracking MCP Server's `record_artifact` is called, it makes a POST to a webhook registered by the Memory Agent. The Memory Agent's FastAPI app receives the webhook and indexes the artifact in real time.

The webhook approach is preferred — it has lower latency and avoids unnecessary polling. Implement both and use the webhook by default, with polling as a fallback.

### Step 6.2 — embed\_text Tool

Implement the `embed_text` MCP tool in the DS Tools MCP Server. It calls Azure OpenAI's embedding deployment. Cache embeddings in a local dict keyed by content hash — identical text produces identical embeddings, no point in calling the API twice. The cache persists in memory only; it resets on server restart.

### Step 6.3 — semantic\_search Tool

Implement the `semantic_search` MCP tool. It calls `embed_text` to embed the query, then computes cosine similarity against all indexed embeddings using NumPy, returns the top-k results sorted by score.

### Step 6.4 — Retrospective Query Tests

Run the full Titanic pipeline twice with different hyperparameter seeds to create two different model artifacts in the index. Then verify these three queries return correct, grounded answers citing specific artifact paths:

Query 1: "What was the highest ROC-AUC we achieved on a binary classification task?"

Query 2: "Show me all feature manifests produced in this run."

Query 3: "Which model artifact should I use for deployment?"

### Phase 6 Exit Gate

- Memory Agent indexes all artifacts from the Titanic pipeline run
- All three retrospective queries return correct answers with cited artifact paths
- `embed_text` caching is verified (two calls with identical text make one Azure OpenAI call)

---

## 23. Phase 7 — Hardening, Observability, Production Gate

### Goal

All safety and observability requirements are met. Five benchmark datasets run end-to-end. Performance targets are achieved.

### Step 7.1 — Safety Penetration Tests

Write and run a suite of 25 injection tests against the SafetyMiddleware and the `before_mcp_tool_call` hook. All 25 must be blocked. The suite must cover: shell injection via `execute_code` input, SQL injection via `query_sql` input (if implemented), path traversal attempts, environment variable access attempts, and `__import__` bypass attempts.

Document the results in `docs/security_review.md`.

### Step 7.2 — Distributed Tracing Verification

Run a full pipeline. Open Azure Monitor Application Insights. Find the trace for the run using the `run_id` as the correlation ID. Verify: the trace contains spans for all stages, all MCP tool calls appear as child spans, all sandbox executions appear as child spans, and no spans are orphaned (missing parent).

### Step 7.3 — Performance Benchmarks

Run all five benchmark scenarios and record wall-clock time. Required targets:

Titanic (binary classification): under 20 minutes end-to-end.

California Housing (regression): under 20 minutes end-to-end.

Credit Card Fraud (imbalanced classification): under 25 minutes (fairness analysis adds time).

IMDB Sentiment (BERT fine-tuning): under 90 minutes with a GPU in the sandbox. If no GPU is available in the sandbox, this benchmark is excluded and documented as GPU-required.

Air Passengers (LSTM time series): under 45 minutes with GPU.

If any target is missed, profile the pipeline to find the bottleneck. Common causes: Magika model loading on every `read_file` call (fix: initialise Magika once at server startup), excessive Azure OpenAI calls in the EDA narrative generation (fix: reduce the prompt size), sandbox startup overhead (fix: reuse containers with volume remounting instead of starting a new container per execution).

### Step 7.4 — End-to-End No-Intervention Tests

Run each benchmark dataset with all human-in-the-loop gates set to auto-approve. Verify that no stage requires manual intervention. If any stage fails during automated testing and requires a human gate, the auto-fix loop did not handle the failure — investigate and fix.

### Step 7.5 — Documentation

Write the `README.md` covering: system overview, prerequisites (Python 3.12, Docker, Azure OpenAI resource with three deployments, `az login`), installation steps, how to run a pipeline (`python main.py --file path/to/data.csv --task "your task description"`), how to query the Memory Agent, how to add a new agent, and how to extend the tool registry.

Write a Mermaid architecture diagram in `docs/architecture.md`.

Write `docs/error_catalogue.md` documenting every known error pattern and its fix.

Write `docs/decision_log.md` documenting every architectural decision made during implementation, the alternatives considered, and the rationale for the chosen approach.

### Phase 7 Exit Gate

- All 25 safety penetration tests pass (all blocked)
- Distributed trace is complete and correct in Application Insights
- All three classical ML performance targets are met
- DL benchmarks meet targets (or GPU-required exclusion is documented)
- All five benchmark datasets complete without manual intervention
- README is complete and accurate
- All documentation files exist and are current

---

## 24. Directory and File Map

```
ds-agent/
│
├── main.py                        -- CLI entry point; accepts --file and --task
│
├── pyproject.toml                 -- all direct dependencies, exact versions
│
├── .env.example                   -- all environment variable keys with comments
│
├── agents/
│   ├── __init__.py
│   ├── base.py                    -- shared agent construction helper
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
│   ├── __init__.py
│   ├── chain.py                   -- assembles middleware in correct order
│   ├── logging_mw.py
│   ├── rate_limit_mw.py
│   ├── safety_mw.py
│   ├── retry_mw.py
│   └── telemetry_mw.py
│
├── hooks/
│   ├── __init__.py
│   └── registry.py                -- all hooks defined and registered here
│
├── mcp_servers/
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── server.py              -- FastMCP + FastAPI app, port 8100
│   │   ├── tools.py               -- all tracking tool implementations
│   │   └── db.py                  -- SQLite layer with WAL mode
│   └── ds_tools/
│       ├── __init__.py
│       ├── server.py              -- FastMCP + FastAPI app, port 8101
│       ├── tools.py               -- all DS tool implementations
│       └── registry.py            -- tool discovery
│
├── tools/
│   ├── __init__.py
│   ├── file_type_detector.py      -- Magika + filetype + heuristics
│   ├── readers/
│   │   ├── __init__.py
│   │   ├── csv_reader.py
│   │   ├── parquet_reader.py
│   │   ├── json_reader.py
│   │   ├── excel_reader.py
│   │   ├── sqlite_reader.py
│   │   ├── pdf_reader.py
│   │   ├── docx_reader.py
│   │   ├── image_reader.py
│   │   ├── archive_reader.py
│   │   └── model_reader.py        -- pickle, ONNX, safetensors
│   ├── sandbox_executor.py        -- Docker sandbox integration
│   ├── browser_use_tool.py        -- Browser-Use wrapper
│   ├── duckduckgo_tool.py         -- DuckDuckGo wrapper
│   └── memory_index.py            -- in-process index + embedding search
│
├── sandbox/
│   ├── Dockerfile.sandbox
│   └── requirements.sandbox.txt   -- pinned, minimal, for the image only
│
├── templates/
│   ├── eda_template.py
│   ├── cleaning_template.py
│   ├── feature_template.py
│   ├── sklearn_train.py
│   ├── imbalanced_train.py
│   ├── sklearn_cluster.py
│   ├── anomaly.py
│   ├── eval_template.py
│   └── dl/
│       ├── bert_classify.py
│       ├── bert_ner.py
│       ├── llm_finetune.py
│       ├── cnn_classify.py
│       ├── yolo_detect.py
│       ├── diffusion.py
│       ├── time_series.py
│       ├── tabnet.py
│       ├── rag.py
│       ├── embedding_finetune.py
│       ├── llm_eval.py
│       ├── whisper.py
│       └── multimodal.py
│
├── workflows/
│   ├── __init__.py
│   ├── pipeline_graph.py          -- static workflow graph definition
│   ├── routing.py                 -- read_file output → pipeline variant routing
│   └── checkpointing.py           -- checkpoint read/write helpers
│
├── memory/
│   ├── __init__.py
│   ├── agent.py                   -- Memory Agent
│   ├── indexer.py                 -- artifact indexing logic
│   └── query.py                   -- RAG query interface
│
├── config/
│   ├── __init__.py
│   ├── settings.py                -- Pydantic BaseSettings model
│   ├── safety_patterns.py         -- versioned blocked pattern list
│   ├── library_registry.py        -- library name → docs URL mapping
│   └── scenario_routing.py        -- scenario name → template mapping
│
├── tests/
│   ├── unit/
│   │   ├── test_file_type_detector.py
│   │   ├── test_readers.py
│   │   ├── test_middleware.py
│   │   ├── test_hooks.py
│   │   ├── test_sandbox_executor.py
│   │   └── test_mcp_compliance.py
│   ├── integration/
│   │   ├── test_tracking_mcp_server.py
│   │   ├── test_ds_tools_mcp_server.py
│   │   ├── test_ingestion_agent.py
│   │   ├── test_auto_fix_loop.py
│   │   ├── test_browser_use_trigger.py
│   │   └── test_memory_agent.py
│   └── e2e/
│       ├── test_titanic.py
│       ├── test_housing.py
│       ├── test_fraud.py
│       ├── test_imdb.py
│       └── test_air_passengers.py
│
├── docs/
│   ├── architecture.md            -- Mermaid diagram
│   ├── error_catalogue.md
│   ├── library_registry.md
│   ├── decision_log.md
│   └── security_review.md
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 25. Configuration Reference — Azure OpenAI Only

All configuration is stored as environment variables. The `settings.py` module defines a Pydantic `BaseSettings` model that reads them. This is the complete list — do not add configuration that is not documented here without updating this document and `decision_log.md`.

**Azure OpenAI**

`AZURE_OPENAI_ENDPOINT` — the endpoint URL for your Azure OpenAI resource. Format: `https://<resource-name>.openai.azure.com/`. This variable's presence is what triggers the Framework to use Azure OpenAI instead of consumer OpenAI.

`AZURE_OPENAI_PRIMARY_DEPLOYMENT` — deployment name for the primary reasoning model (e.g., `my-gpt4o-prod`). Used by: Orchestrator, Debug Agent, Explainability, Report Agent, and any DL scenario that requires extended reasoning.

`AZURE_OPENAI_FAST_DEPLOYMENT` — deployment name for the fast execution model (e.g., `my-gpt4o-mini-prod`). Used by: Ingestion, EDA (stat prompts), HP Tuning, Memory Agent (query synthesis).

`AZURE_OPENAI_EMBEDDING_DEPLOYMENT` — deployment name for the embedding model (e.g., `my-embedding-prod`). Used exclusively by the `embed_text` tool and the Memory Agent.

`AZURE_OPENAI_API_VERSION` — API version string (e.g., `2025-03-01-preview`). Use the latest stable version available.

**Agent Framework**

`AGENT_FRAMEWORK_DEFAULT_MAX_TOKENS` — integer, default 2048. Maximum tokens per completion. The Orchestrator and Report Agent may need higher values (set to 4096 for those agents individually).

`AGENT_FRAMEWORK_DEFAULT_TEMPERATURE` — float, default 0.1. Low temperature for code generation and reasoning agents. The Report Agent's narrative sections may use 0.3 for more natural prose.

**MCP Servers**

`TRACKING_MCP_URL` — full URL of the Tracking MCP Server. Default: `http://localhost:8100`. In production, use the internal Docker network hostname.

`DS_TOOLS_MCP_URL` — full URL of the DS Tools MCP Server. Default: `http://localhost:8101`.

`TRACKING_MCP_DB_PATH` — filesystem path for the SQLite database file. Default: `./data/tracking.db`. Ensure this path is on a durable volume in production.

**Sandbox**

`SANDBOX_IMAGE` — Docker image name and tag. Default: `ds-agent-sandbox:latest`. In production, use a fully qualified image name with a specific SHA tag.

`SANDBOX_TIMEOUT_SECONDS` — integer, default 300. The wall-clock timeout for any single sandbox execution.

`SANDBOX_MEMORY_LIMIT` — string, default `4g`. Docker memory limit.

`SANDBOX_CPU_QUOTA` — float, default 2.0. Number of CPU cores.

`SANDBOX_SCRATCH_DIR` — host path for temporary code and data files mounted into the sandbox. Default: `/tmp/ds-agent-scratch`. Must be on a fast local disk (not NFS).

`MAX_AUTO_FIX_ATTEMPTS` — integer, default 5. Maximum repair attempts before escalation.

**Research Tools**

`BROWSER_USE_AZURE_DEPLOYMENT` — deployment name to use for Browser-Use's internal LLM. Use `AZURE_OPENAI_FAST_DEPLOYMENT` value. Browser-Use requires the same `AZURE_OPENAI_ENDPOINT` environment variable to be set.

`BROWSER_USE_MAX_STEPS` — integer, default 50.

`BROWSER_USE_HEADLESS` — boolean, default `true`.

`DUCKDUCKGO_MAX_RESULTS` — integer, default 10.

`DUCKDUCKGO_RATE_LIMIT_PER_HOUR` — integer, default 30.

**Pipeline Behaviour**

`PIPELINE_HUMAN_IN_THE_LOOP` — boolean, default `true`. Set to `false` in development for auto-approve of all gates.

`PIPELINE_CHECKPOINT_ENABLED` — boolean, default `true`. Set to `false` only for debugging.

`PIPELINE_TOKEN_BUDGET` — integer, default 500000. Total Azure OpenAI token budget per pipeline run. The Telemetry Middleware tracks cumulative usage.

`MEMORY_INDEX_PATH` — path to the `artifact_index.json` file. Default: `./data/memory_index.json`.

`MEMORY_EMBEDDINGS_PATH` — path to the `artifact_embeddings.npy` file. Default: `./data/memory_embeddings.npy`.

---

## 26. Error Catalogue and Auto-Remediation Map

This catalogue is the lookup table for the Debug Agent's Attempt 1 (pattern match). Maintain it in `docs/error_catalogue.md` and load it into the Debug Agent's session state at startup.

| Error Class | Error Pattern | Library | Auto-Remediation |
|---|---|---|---|
| `ModuleNotFoundError` | `No module named 'X'` | any | Identify the correct import for `X` from the sandbox requirements and prepend the import statement |
| `KeyError` | `'<column_name>'` | pandas | Look up the feature manifest; substitute the correct column name |
| `ValueError` | `could not convert string to float` | pandas/numpy | Add `.astype(float)` conversion with error handling before the failing operation |
| `ValueError` | `Input contains NaN` | sklearn | Add `SimpleImputer` or `fillna()` before the failing step |
| `MemoryError` | (any) | any | Switch to chunked processing using `pandas.read_csv(chunksize=10000)` or equivalent |
| `TimeoutError` | (any) | any | Sample the dataset to 10% of rows and re-run as a validation step; log that full dataset run requires extended timeout |
| `RuntimeError` | `CUDA out of memory` | PyTorch | Halve `batch_size`; if still OOM enable gradient checkpointing; if still OOM move to CPU |
| `RuntimeError` | `loss is nan` | PyTorch | Check for NaN in input tensors; reduce learning rate by factor of 10 |
| `RuntimeError` | `Expected all tensors to be on the same device` | PyTorch | Add `.to(device)` to all tensors and the model before the forward pass |
| `AttributeError` | `'NoneType' object has no attribute` | any | Add a null check before the failing attribute access |
| `FileNotFoundError` | (any) | any | Verify the path against the session state artifact paths; substitute the correct path |
| `OSError` | `[Errno 28] No space left on device` | any | Clear temporary files from `/workspace/output/`; reduce output file size (e.g., compress charts) |
| `ImportError` | `cannot import name 'X' from 'Y'` | any | Check the library version in the sandbox requirements; the API may have changed — invoke Browser-Use on the library docs |
| `TypeError` | `got an unexpected keyword argument` | any | The API signature has changed — invoke Browser-Use on the library docs |
| `DataLoaderError` | `worker process crashed` | PyTorch | Set `num_workers=0` and retry |
| `OSError` | `NCCL error` | PyTorch distributed | Reduce to single-GPU training; set `nproc_per_node=1` |

---

## 27. Security and Responsible AI Obligations

### 27.1 Security Obligations (non-negotiable)

The sandbox runs as non-root with no outbound network — this must be verified in the Phase 1 exit gate and re-verified before every production release. A sandbox that can reach the network is not a sandbox.

No secret (API key, connection string, personal data) may appear in: log files, MCP tool call records, artifact files, or the Tracking MCP Server's SQLite database. The logging middleware must sanitise outputs — scrub values that match patterns for API keys (Base64 strings > 40 characters), connection strings (containing `AccountKey=` or similar), and email addresses.

Managed identity, not API keys, in production. No exceptions.

### 27.2 Responsible AI Obligations (non-negotiable)

Fairness analysis (Fairlearn) runs on every classification model. No classification model is deployed without a `fairness_metrics` section in the evaluation report.

The `before_deploy` hook blocks deployment when the evaluation recommendation is `block`. This is a code-enforced constraint — the Deployment Agent cannot override it.

Every deployed model endpoint exposes a `/model_card` route that returns the model card JSON from the evaluation report. Downstream consumers of the endpoint can read the model card to understand training data, known limitations, and intended use.

SHAP explanations are produced for every deployed model. A model without an explanation is not a deployed model — it is a black box. This system does not deploy black boxes.

### 27.3 PII Detection

The Ingestion Agent, after reading the file, must run a PII scan on the first 1,000 rows of any tabular dataset. Use a heuristic scan: check column names against a list of PII-indicative names (`name`, `email`, `phone`, `ssn`, `dob`, `address`, `ip_address`, `user_id`, and domain equivalents), and apply pattern matching on column values for email format, phone number format, and US SSN format.

When PII is detected: log the finding to the Tracking MCP Server, present the finding to the user via a human-in-the-loop gate, and proceed only with the user's explicit confirmation that the data is authorised for ML processing.

---

## 28. Verification Gates per Phase

Use this table as the final checklist before signing off each phase. Every row must be checked before the phase is considered complete. Update the date column when you sign off.

| Phase | Name | All Exit Criteria Met | Reviewer | Date |
|---|---|---|---|---|
| 1 | Foundation, Environment, MCP Servers | | | |
| 2 | read\_file Tool and Data Ingestion | | | |
| 3 | EDA, Cleaning, Feature Engineering | | | |
| 4 | Model Training, Tuning, Evaluation | | | |
| 5 | Explainability, Reporting, Deployment | | | |
| 6 | Memory Agent and Retrospection | | | |
| 7 | Hardening, Observability, Production Gate | | | |

---

## 29. Reference Documents

Read these before implementing each relevant section. When this document conflicts with an official source, the official source takes precedence — update this document to reflect the current truth.

`learn.microsoft.com/en-us/agent-framework/overview` — Microsoft Agent Framework overview and quickstart.

`learn.microsoft.com/en-us/agent-framework/support/upgrade/python-2026-significant-changes` — all breaking changes and package restructuring in the 1.0 release. Read before writing any import statements.

`learn.microsoft.com/en-us/python/api/agent-framework-core/agent_framework.mcpstreamablehttptool` — `MCPStreamableHTTPTool` API reference. Read before implementing MCP client connections.

`github.com/microsoft/agent-framework/issues/5317` — the known issue with legacy SSE GET in `MCPStreamableHTTPTool`. Read before testing MCP tool connections.

`learn.microsoft.com/en-us/azure/app-service/tutorial-ai-model-context-protocol-server-python` — the canonical example for mounting a FastMCP server into a FastAPI app.

`github.com/google/magika` — Magika documentation, supported content types, and accuracy benchmarks.

`docs.browser-use.com/llms-full.txt` — the full Browser-Use SDK reference in a single file optimised for LLMs. Read this file, not individual doc pages, when designing the `search_docs` tool.

`docs.browser-use.com/customize/supported-models` — confirmed support for `ChatAzureOpenAI`. Verify the `AZURE_OPENAI_ENDPOINT` environment variable is set before Browser-Use initialisation.

`fairlearn.org/main/user_guide/index.html` — Fairlearn user guide. Read the section on `MetricFrame` before implementing the Evaluation Agent's fairness analysis.

`shap.readthedocs.io` — SHAP documentation. Pay particular attention to the explainer selection guide — the wrong explainer for a model type produces incorrect values.

---

*This document is the primary implementation reference. All architectural decisions are recorded in `docs/decision_log.md`. Changes to this document require a pull request with review from at least one person who has read the entire document. Do not approve partial reviews.*

---

## 30. Sandbox Requirements File — Authoritative Package List

### 30.1 Purpose and Rules

`sandbox/requirements.sandbox.txt` is the single source of truth for what is available inside the code execution sandbox. Every package is pinned to an exact version using `==`. No floating ranges. No `>=`. No `~=`. Floating ranges produce non-reproducible sandboxes — a pipeline that works today breaks tomorrow because a patch release changed an internal API.

The sandbox image is tagged with the SHA-256 hash of this file. When the file changes, rebuild the image. When you add a new library to the system's templates, add it here first and rebuild before writing the template.

The sandbox has no outbound network. You cannot `pip install` at runtime. If a template references a library that is not in this file, the sandbox will produce a `ModuleNotFoundError` that the auto-fix loop will unsuccessfully attempt to resolve. Add the library here instead.

### 30.2 Core Python Utilities

Install in the sandbox: `python-dateutil`, `pytz`, `tzdata`, `chardet`, `charset-normalizer`, `tqdm`, `joblib`, `cloudpickle`, `dill`. These are low-level utilities required by virtually every data science library. Installing them explicitly ensures the correct versions are pinned rather than relying on transitive installs.

### 30.3 Data I/O and Formats

Install: `pandas`, `numpy`, `pyarrow`, `fastparquet`, `openpyxl`, `xlrd`, `xlwt`, `odfpy`, `sqlite-utils`, `pymupdf` (for PDF), `python-docx`, `python-pptx`, `Pillow`, `imageio`, `tifffile`, `soundfile`, `mutagen`, `zipfile36`.

Pin `pandas` to the latest stable 2.x release. Pin `numpy` to a version that is compatible with both your pandas version and your sklearn version — numpy 2.x has breaking changes relative to 1.x; check compatibility before pinning.

### 30.4 Classical Machine Learning

Install: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn`, `statsmodels`, `scipy`, `optuna`, `optuna-integration`, `hyperopt`.

Pin `scikit-learn` and `xgboost` to versions that are mutually compatible — check both changelogs. Pin `catboost` to a version that matches the Python 3.12 wheel; CatBoost has historically lagged Python version support.

### 30.5 Deep Learning

Install: `torch`, `torchvision`, `torchaudio`, `transformers`, `datasets`, `tokenizers`, `accelerate`, `peft`, `trl`, `sentence-transformers`, `timm`, `einops`, `safetensors`.

For the sandbox image, install the CPU-only variant of PyTorch by default. If GPU support is needed, build a separate `Dockerfile.sandbox.gpu` that installs the CUDA variant. Keep the CPU and GPU images version-locked — the same `transformers` version, the same `torch` minor version, different CUDA builds.

Pin `transformers` carefully. Hugging Face releases frequently and model-loading APIs change between minor versions.

### 30.6 Explainability and Evaluation

Install: `shap`, `lime`, `captum`, `fairlearn`, `alibi`, `eli5`, `yellowbrick`, `dtreeviz`.

Pin `shap` to a version that is compatible with your `xgboost` and `sklearn` versions — SHAP's `TreeExplainer` uses internal APIs of both. Check the SHAP changelog before upgrading.

### 30.7 Visualisation

Install: `matplotlib`, `seaborn`, `plotly`, `kaleido` (for Plotly static image export), `bokeh`, `altair`.

`kaleido` is required for rendering Plotly charts as PNG inside the sandbox. Without it, `fig.write_image()` fails silently or raises a cryptic error. This is a common source of confusion — document it in `docs/error_catalogue.md`.

### 30.8 Statistical and Time Series

Install: `statsmodels`, `pmdarima`, `prophet`, `neuralprophet`, `tslearn`, `sktime`, `stumpy`, `ruptures`.

### 30.9 NLP Utilities

Install: `nltk`, `spacy`, `gensim`, `textblob`, `langdetect`, `ftfy`, `unidecode`.

After installing `spacy`, the sandbox startup script must run `python -m spacy download en_core_web_sm` to download the default English model. This must be baked into the Docker image build — it cannot run at execution time (no network).

### 30.10 Metrics and Experiment Tracking

Install: `mlflow`, `evaluate` (Hugging Face evaluate library), `rouge-score`, `sacrebleu`, `ragas`, `bert-score`.

`mlflow` is the in-sandbox experiment tracking library. It writes to a local `mlruns/` directory inside `/workspace/output/`. The `log_metrics` MCP tool reads from this directory after execution.

### 30.11 Utilities and Static Analysis

Install: `bandit`, `pylint`, `ast-grep-cli`, `psutil`, `memory-profiler`.

`bandit` and `pylint` are installed in the sandbox because the `before_code_execute` hook runs them via subprocess before submitting to Docker. Having them in the image allows the hook to also run them inside the sandbox as a post-execution audit step.

---

## 31. Template Authoring Guide

### 31.1 What a Template Is

A template is a Python source file stored in `templates/` that is parameterised by the agent. Templates are not executed by the agents directly. They are text files that the agent reads, fills in the parameter placeholders, and submits via `execute_code`.

Templates use a simple placeholder syntax: `{PLACEHOLDER_NAME}`. The agent's prompt includes a description of each placeholder and its expected value. The agent fills them in by string replacement. Do not use Jinja or any templating engine — string replacement is sufficient and eliminates a dependency.

Every template must begin with a comment block that documents: the template's purpose, all placeholders and their types, the expected output files (with paths relative to `/workspace/output/`), and the expected runtime (in seconds, on a typical CPU).

### 31.2 Template Authoring Rules

Every template must satisfy these rules without exception. Violations cause silent failures or incorrect results that reach deployment.

**Self-contained:** the template must not import from any module in the `ds-agent` project. It runs inside the sandbox, which has no knowledge of the project's code. It imports only from the libraries in `sandbox/requirements.sandbox.txt`.

**Deterministic output paths:** all output files are written to paths under `/workspace/output/` that are deterministic and documented. The agent must know exactly which files to expect after execution. Variable output paths make the `after_code_execute` hook unreliable.

**Structured metrics output:** every template that trains a model, computes statistics, or produces evaluation results must write a `metrics.json` file to `/workspace/output/metrics.json`. The JSON schema of `metrics.json` is defined per template and documented in the template's header comment. Never write metrics to stdout — stdout is for human-readable progress reporting.

**Graceful partial failure:** if a template can produce multiple outputs (e.g., charts for each column), and one chart fails, the template must catch the exception, log it to a `errors.json` file in `/workspace/output/`, and continue producing the remaining outputs. A template that crashes on a single bad column fails the entire EDA stage.

**Progress reporting to stdout:** print progress statements to stdout in a structured format: `[STAGE] message`. The sandbox executor collects stdout and the agent reads it to understand what happened. For example: `[DATA] Loaded 10842 rows, 28 columns` or `[TRAIN] Epoch 3/20: loss=0.342, val_loss=0.389, lr=1e-4`.

**Seed at the top:** every template that uses any randomness (train-test split, model initialisation, sampling) must set all relevant seeds in the first executable line, before any imports that trigger random initialisation.

### 31.3 EDA Template — Required Outputs

The EDA template must produce these files in `/workspace/output/`:

`stats.json` — a JSON object with these top-level keys: `shape` (rows, columns), `dtypes` (column name → dtype string), `describe` (the full `DataFrame.describe(include='all')` output as a dict), `null_counts` (column name → null count), `null_percentages` (column name → percentage), `cardinality` (column name → unique count, for all columns), `skewness` (column name → skewness value, for numeric columns), `kurtosis` (column name → kurtosis value, for numeric columns), `target_distribution` (if a target column is identified: class counts and percentages), `correlation_matrix` (numeric columns → correlation values as a dict of dicts).

`charts/histogram_{column_name}.png` — one histogram per numeric column.

`charts/boxplot_{column_name}.png` — one box plot per numeric column.

`charts/correlation_heatmap.png` — one correlation heatmap covering all numeric columns.

`charts/class_balance.png` — one bar chart showing target class distribution (only if a target column is identified).

`charts/missing_values_heatmap.png` — a heatmap showing the null pattern across all columns and all rows (use `seaborn.heatmap` on `df.isnull()`).

`errors.json` — a list of any columns or charts that failed during the template's execution, with the error message for each.

### 31.4 Cleaning Template — Required Outputs

`cleaned_dataset.parquet` — the cleaned dataset written as Parquet (not CSV — Parquet preserves dtypes, is faster to read, and compresses better).

`transformation_log.json` — a JSON array where each entry is a transformation record with fields: `column`, `transformation_type` (imputation, outlier_cap, outlier_remove, dtype_cast, drop_column), `original_stats` (null count, min, max, mean before transform), `post_stats` (same after transform), `justification` (a string explaining why this transformation was applied), `rows_affected` (count).

`cleaning_summary.json` — a compact summary: `rows_before`, `rows_after`, `columns_before`, `columns_after`, `null_cells_before`, `null_cells_after`, `outliers_flagged`, `outliers_removed`, `columns_dropped`.

`errors.json` — any columns that could not be processed.

### 31.5 Feature Engineering Template — Required Outputs

`features_train.parquet` — the feature-engineered training set.

`features_test.parquet` — the feature-engineered test set (same transforms applied, fit on training only — no leakage).

`feature_manifest.json` — a JSON array where each entry is a feature record with fields: `feature_name`, `source_column` (the original column it derives from, or a list if it is a cross-column feature), `transform_type` (one-hot, target_encode, log_transform, polynomial, embedding, interaction, original), `transform_params` (the parameters used, e.g., the degree for polynomial), `importance_score` (from the quick RandomForest run, or null if the feature was added but not yet ranked), `retained` (boolean — whether the feature passed the importance threshold).

`importance_chart.png` — a bar chart of feature importances from the quick RandomForest run.

`errors.json` — any features that could not be generated.

### 31.6 Training Template — Required Outputs

`model.joblib` (for sklearn models) or `model_state_dict.pt` + `model_config.json` (for PyTorch models) or `model.onnx` (ONNX export, always produced in addition to the primary format).

`metrics.json` — the metrics object with all task-appropriate metrics. Schema differs by task type but must always include: `task_type`, `algorithm`, `train_metric` (primary metric on training set), `val_metric` (primary metric on validation set), `test_metric` (primary metric on test set if a held-out test set was provided), `cv_scores` (list of per-fold scores from cross-validation), `cv_mean`, `cv_std`, `training_time_seconds`, `model_size_bytes`, `seed`.

For deep learning training templates additionally: `loss_curve.json` — a JSON array of `{epoch, train_loss, val_loss, lr}` objects, one per epoch.

`training_summary.md` — a brief human-readable summary of the training run, written to stdout by the template. The agent copies this into the session state as the training narrative.

`errors.json` — any recoverable errors encountered during training (non-fatal warnings, fallbacks applied).

### 31.7 Deep Learning Template Additional Requirements

Every DL template must satisfy all items in the training loop checklist from Section 14.2 plus these additional structural requirements:

The model class definition must appear before the training loop. Do not define the model inside the training loop — this causes issues with checkpointing and export.

The data loading section must validate that the dataset files exist and are readable before initialising the model. A template that initialises a large model and then discovers the data is missing wastes memory and time.

The train-validation split must be performed before any data preprocessing. Preprocessing (normalisation, tokenisation) must be fit on the training split only. Applying preprocessing to the full dataset before splitting is a data leakage bug. This requirement is enforced by the `before_model_train` hook, which checks for the presence of a split operation before any preprocessing operation in the template code.

The ONNX export section must appear after the training loop, not before. It must verify the export is correct by running a sample inference on the ONNX model and comparing outputs to the PyTorch model outputs. If the outputs diverge by more than 1e-4 (relative), write a warning to `errors.json` but do not abort — the PyTorch model is still valid.

---

## 32. Agent System Prompt Authoring Guide

### 32.1 Principles

Every agent system prompt must satisfy these properties. An agent that does not satisfy them will produce inconsistent results, hallucinate tool calls, or overstep its authority.

**Single responsibility statement:** the first sentence of every system prompt must state the agent's single responsibility in 20 words or fewer. "You are the EDA Agent. Your sole responsibility is to produce a complete statistical profile of a dataset." If you cannot state the responsibility in 20 words, the agent has too many responsibilities.

**Explicit tool list with usage descriptions:** after the responsibility statement, list every tool the agent is authorised to use, with a one-sentence description of when to use it. The agent must not call tools not on its list. If a tool it needs is missing from its list, that is a system design error — fix the design, not the prompt.

**Explicit output contract:** state exactly what the agent must produce. Name the files. Specify the JSON keys. State where to write them. The agent must not consider its job done until every item in the output contract is produced and recorded in the Tracking MCP Server.

**Explicit escalation protocol:** state exactly what the agent does when it encounters an error it cannot resolve. "If `execute_code` returns a non-zero exit code, call the `on_code_error` hook by notifying the Orchestrator via the Tracking MCP Server's `record_error` tool. Do not attempt to fix code errors yourself — that is the Debug Agent's responsibility."

**Explicit boundary conditions:** state what the agent must not do. "You must not train any model. You must not call `deploy_endpoint`. You must not read files that are not in your input artifact list."

### 32.2 System Prompt Structure

Every system prompt must follow this structure in order:

Section 1 — Identity and single responsibility (2–3 sentences).

Section 2 — Authorised tools and when to use each (bullet list, one tool per bullet).

Section 3 — Input contract: what you receive and from where (reference the session state keys from Section 15.2).

Section 4 — Processing protocol: the step-by-step sequence the agent must follow. Number the steps. The agent executes them in order. If a step fails, it follows the escalation protocol — it does not skip to the next step.

Section 5 — Output contract: exactly what files to produce, where to write them, and what to record in the Tracking MCP Server.

Section 6 — Escalation protocol: what to do on failure.

Section 7 — Boundary conditions: explicit list of things the agent must not do.

### 32.3 Processing Protocol Design

The processing protocol (Section 4 of the prompt) must be specific enough that two different LLMs reading the same prompt would produce the same sequence of tool calls on the same input. Vague instructions produce inconsistent behaviour across model versions.

Bad: "Analyse the dataset and produce statistics."

Good: "Step 1: Call `get_data_sample` with `n_rows=5` to inspect the first five rows. Step 2: Examine the column names and dtypes from the session state's `input_artifact_paths[0]` schema field. Step 3: Generate the EDA template code by substituting the column names and dtypes into the `eda_template.py` template. Step 4: Call `execute_code` with the generated code and the dataset path. Step 5: If exit code is non-zero, call `record_error` and wait for the Debug Agent response via the Orchestrator. Step 6: If exit code is zero, read `stats.json` from the output artifact list. Step 7: Call `write_output` with the stats. Step 8: Call `record_artifact` for each output file. Step 9: Call the Azure OpenAI completion endpoint with the stats to generate the narrative. Step 10: Call `write_output` with the narrative as `eda_narrative.md`."

### 32.4 Orchestrator Prompt Special Requirements

The Orchestrator prompt has additional requirements that sub-agent prompts do not:

The prompt must include the complete workflow graph in a compact text representation so the Orchestrator knows the valid stage sequence and the dependencies between stages.

The prompt must include the checkpoint-resume protocol explicitly: before invoking any stage, query `query_run_status` and skip stages that already have a successful checkpoint.

The prompt must include the complete routing table from Section 6.4 so the Orchestrator can route `read_file` outputs to the correct pipeline variant.

The prompt must include the human-in-the-loop gate conditions with explicit trigger conditions and the pause/resume mechanism.

The prompt must include the escalation protocol for when the Debug Agent exhausts all five attempts: write the debug report, trigger the human gate, and record the pipeline status as `blocked_awaiting_human` in the Tracking MCP Server.

The prompt must include the token budget protocol: the Orchestrator monitors cumulative token usage from the Telemetry Middleware's reports and halts the pipeline if the budget is exceeded.

### 32.5 Debug Agent Prompt Special Requirements

The Debug Agent prompt must make it absolutely clear that the agent returns only repaired code — never explanations, never apologies, never markdown fences around the code. The output is code that will be written directly to a file and executed. Any non-code text in the output causes a syntax error.

The prompt must include the attempt number in every invocation. The attempt number determines which resolution strategy to use. The agent must not apply a more expensive strategy (Browser-Use) when a cheaper strategy (pattern match) has not yet been attempted.

The prompt must state that the Debug Agent never modifies pipeline state, never calls tools other than `search_docs` and `web_research`, and never changes the algorithm or approach unless on Attempt 5.

---

## 33. CI/CD Pipeline Specification

### 33.1 Philosophy

Continuous integration must be fast enough to not be skipped. If CI takes more than 10 minutes, developers bypass it. Design CI to complete in under 8 minutes for the standard push workflow and under 30 minutes for the full integration test suite (run only on pull requests to `main`).

### 33.2 Standard Push Workflow (Target: under 8 minutes)

Trigger: every push to any branch.

**Step 1 — Lint (target: under 60 seconds)**

Run `ruff check .` for style and import errors. Run `ruff format --check .` for formatting. Run `mypy --strict` for type checking. Fail fast — if lint or type checking fails, do not proceed to tests.

Maintain a `pyproject.toml` section for both `ruff` and `mypy` configuration. Do not use separate config files — consolidate configuration in one place.

**Step 2 — Unit Tests (target: under 3 minutes)**

Run `pytest tests/unit/` with a 30-second timeout per test. Unit tests must not make any network calls. Mock Azure OpenAI, MCP servers, Docker, and the filesystem. Any unit test that makes a real network call is incorrectly written — fix it.

Measure and enforce code coverage. Minimum coverage threshold: 85% for all modules in `agents/`, `middleware/`, `hooks/`, and `tools/`. Coverage below this threshold fails CI.

**Step 3 — Sandbox Image Build Verification (target: under 3 minutes)**

Build the sandbox Docker image. Verify that `python -c "import pandas, torch, transformers, shap, optuna"` succeeds inside the image. This catches package installation failures before they reach integration tests.

Only rebuild the image if `sandbox/requirements.sandbox.txt` has changed (use hash-based caching in the CI workflow). Most pushes will skip this step due to the cache hit.

**Step 4 — MCP Compliance Tests (target: under 2 minutes)**

Start both MCP servers in the CI environment. Run `tests/unit/test_mcp_compliance.py` against them. These tests verify the three `curl` compliance checks and confirm tool registration is correct.

### 33.3 Pull Request to Main Workflow (Target: under 30 minutes)

Trigger: pull request targeting `main` branch.

Run all steps from the standard push workflow first.

**Step 5 — Integration Tests (target: under 10 minutes)**

Run `pytest tests/integration/` with a 120-second timeout per test. Integration tests may use the real Docker sandbox but must not make real Azure OpenAI calls — use the `pytest-recording` library to replay recorded Azure OpenAI responses. Record new responses when intentionally changing behaviour, not on every run.

**Step 6 — Auto-Fix Loop Tests (target: under 5 minutes)**

Run `tests/integration/test_auto_fix_loop.py`. These tests inject known errors into templates and verify the auto-fix loop repairs them correctly. They must run against the real sandbox Docker image.

**Step 7 — Security Scan (target: under 3 minutes)**

Run `bandit -r agents/ tools/ mcp_servers/ -ll` (medium and high severity only). Any HIGH severity finding fails CI. MEDIUM severity findings produce a warning but do not fail CI unless there are more than 5 new ones compared to the previous run (detected by storing the count in a CI artefact).

Run `safety check --json` to check all installed dependencies against the safety vulnerability database. Any known vulnerability with a severity of HIGH or CRITICAL fails CI.

### 33.4 Deployment Workflow

Trigger: a tag pushed with the format `v*.*.*` (semantic version).

This workflow runs after the pull request workflow succeeds. It additionally:

Builds and pushes the sandbox Docker image to your container registry, tagged with both the semantic version and the `requirements.sandbox.txt` SHA.

Runs `pytest tests/e2e/test_titanic.py` as a smoke test against the staging environment with real Azure OpenAI credentials (stored as CI secrets). This is the only CI step that makes real API calls.

If the smoke test passes, creates a GitHub Release with the changelog and the signed image digest.

### 33.5 CI Environment Configuration

Store these secrets in your CI provider (GitHub Actions Secrets, or equivalent): `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_PRIMARY_DEPLOYMENT`, `AZURE_OPENAI_FAST_DEPLOYMENT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`, `CONTAINER_REGISTRY_URL`, `CONTAINER_REGISTRY_USERNAME`, `CONTAINER_REGISTRY_PASSWORD`.

Do not store `AZURE_OPENAI_API_KEY` in CI secrets. Use a dedicated service principal with `Cognitive Services OpenAI User` role for CI, not a user API key. Rotate the service principal credentials on a 90-day schedule.

---

## 34. Operational Runbook

### 34.1 Starting the System

The system has three processes that must start in order: the Tracking MCP Server, the DS Tools MCP Server, and then the main application. Starting the application before the MCP servers are healthy causes immediate failure.

Start the Tracking MCP Server first. After starting, poll its `/health` endpoint every 2 seconds until it returns `{"status": "healthy"}` or until a 30-second timeout expires. If the timeout expires, the Tracking MCP Server has not started — check the logs for database connection errors.

Start the DS Tools MCP Server next. Apply the same polling and timeout logic. The DS Tools Server starts faster (no database initialisation) but should still be verified before the application starts.

Start the main application. The application's own startup sequence calls both `/health` endpoints as its first action. If either returns unhealthy, the application logs the failure and exits cleanly with a non-zero exit code. Do not start the application without both MCP servers healthy.

### 34.2 Running a Pipeline

The entry point is `python main.py`. It accepts two required arguments: `--file` (path to the input file) and `--task` (a natural-language description of the data science task). It accepts one optional argument: `--resume-run-id` (a run ID from a previous run to resume from the last checkpoint).

Before starting a pipeline run, the application generates a new `run_id` UUID and logs it to stdout. Record this `run_id` — you will need it to query the Memory Agent, to resume after a crash, and to find the run's trace in Application Insights.

The application outputs pipeline progress to stdout as structured JSONL. Each line is a JSON object with `run_id`, `stage`, `status`, `timestamp`, and optional `message`. Redirect stdout to a log file for long-running pipelines.

### 34.3 Resuming After a Crash

If the application exits with a non-zero exit code mid-pipeline, run `python main.py --file <same_file> --task "<same_task>" --resume-run-id <run_id_from_previous_run>`.

The Orchestrator queries `query_run_status` for the provided `run_id`, identifies the last successful checkpoint, and resumes from the next stage. All previous stage outputs are retrieved from the Tracking MCP Server's artifact records.

If the resume fails (the Tracking MCP Server's database is corrupted or the checkpoint records are missing), do not attempt to resume. Start a fresh run with a new `run_id`.

### 34.4 Querying the Memory Agent

The Memory Agent exposes a query endpoint on the DS Tools MCP Server via the `semantic_search` tool. From the command line: `python -c "from memory.query import query; print(query('<your question>'))"`.

For example: `python -c "from memory.query import query; print(query('best model for binary classification this month'))"`.

The query returns a JSON object with `answer` (the LLM-synthesised answer), `sources` (the top artifact records used), and `confidence` (a score from 0 to 1 based on the similarity of the top result).

### 34.5 Monitoring a Running Pipeline

During a pipeline run, monitor progress by tailing the application's JSONL stdout log. Every stage start and end produces a log entry. Every tool call produces a log entry. Every error produces a log entry.

For the Tracking MCP Server's full audit log, query the SQLite database directly: `sqlite3 data/tracking.db "SELECT stage, status, started_at, ended_at, duration_ms FROM agent_runs WHERE run_id = '<run_id>' ORDER BY started_at;"`.

In Application Insights, find the distributed trace by filtering for the `run_id` attribute in the trace search. The trace shows the full execution timeline with all spans nested correctly.

### 34.6 Upgrading the Sandbox Image

When `sandbox/requirements.sandbox.txt` changes:

Step 1: compute the new SHA-256 of the file: `sha256sum sandbox/requirements.sandbox.txt`.

Step 2: update `SANDBOX_IMAGE` in `.env` to use the new SHA tag: `ds-agent-sandbox:<new_sha>`.

Step 3: build the new image: `docker build -f sandbox/Dockerfile.sandbox -t ds-agent-sandbox:<new_sha> .`.

Step 4: run the sandbox verification tests: `pytest tests/unit/test_sandbox_executor.py -v`.

Step 5: run the integration tests: `pytest tests/integration/ -v`.

Step 6: if all tests pass, delete the old image: `docker rmi ds-agent-sandbox:<old_sha>`.

Never delete the old image before the new one is verified. Keep the last two versions of the sandbox image at all times to enable rollback.

### 34.7 Rotating Azure OpenAI Credentials

If you are using API keys (development only): generate a new key in the Azure portal. Update the `.env` file. Restart all three processes. Verify with a hello-world agent call. Delete the old key in the Azure portal.

If you are using managed identity (production): managed identity credentials are rotated automatically by Azure — no action required on a normal rotation schedule. If you need to revoke access (security incident), remove the managed identity's `Cognitive Services OpenAI User` role assignment in the Azure portal. The system will stop making Azure OpenAI calls immediately.

### 34.8 Debugging a Failed Stage

When a stage fails and the auto-fix loop exhausts all five attempts, the system writes a debug report to `outputs/<run_id>/debug_report_<stage>_<timestamp>.md` and triggers the human-in-the-loop gate.

To investigate manually:

Step 1: read the debug report. It contains the original code, all five repair attempts, all error messages, and the documentation and search results the Debug Agent found.

Step 2: query the Tracking MCP Server for the error records: `sqlite3 data/tracking.db "SELECT attempt_number, error_class, error_message FROM error_logs WHERE run_id = '<run_id>' AND stage = '<stage>' ORDER BY attempt_number;"`.

Step 3: if the error is a library version incompatibility (common when upgrading packages), update `sandbox/requirements.sandbox.txt` and rebuild the image.

Step 4: if the error is a model output that produces unexpected data (wrong column types, unexpected nulls), trace back to the Cleaning Agent's transformation log and verify the transformation was applied correctly.

Step 5: if the error is a template logic error, fix the template in `templates/`, rebuild the sandbox image, and resume the pipeline.

Step 6: after fixing, approve the human gate: `curl -X POST http://localhost:8100/approve/<run_id>/<stage> -H "Content-Type: application/json" -d '{"decision": "retry", "override_params": {}}'`.

---

## 35. Upgrade and Migration Guide

### 35.1 Upgrading Microsoft Agent Framework

The Microsoft Agent Framework 1.0 has a long-term support commitment and a documented upgrade path. Before upgrading to any new minor or major version:

Read `learn.microsoft.com/en-us/agent-framework/support/upgrade/python-2026-significant-changes` for the Python significant changes document. This document lists every breaking change, removed feature, and migration step.

Run the unit tests against the new version in an isolated virtual environment before updating the main environment.

Pay particular attention to: changes to `ToolInvocation` field names (breaking in some releases), changes to `ToolResult` field names (breaking in some releases), changes to middleware hook signatures, and changes to the `MCPStreamableHTTPTool` handshake behaviour.

After upgrading, re-run the MCP compliance tests. The Framework's MCP client implementation is updated frequently and compliance behaviour can change.

### 35.2 Upgrading Azure OpenAI API Version

Azure OpenAI API versions are date-versioned. Each version adds new capabilities and deprecates old behaviours. Before upgrading `AZURE_OPENAI_API_VERSION`:

Read the Azure OpenAI changelog for the target version.

Check whether the structured output format your agents expect has changed (response format, tool call format, streaming format).

Run the hello-world integration test against the new version before changing the production environment.

### 35.3 Upgrading Sandbox Packages

Upgrade sandbox packages one at a time, not in bulk. Upgrading five packages simultaneously makes it impossible to identify which package introduced a regression.

For each package upgrade: update its version in `requirements.sandbox.txt`, rebuild the image, run the full unit and integration test suite, and only then move to the next package.

### 35.4 Adding a New Agent

When adding a new agent to the system:

Step 1: define the agent's single responsibility clearly. If you cannot state it in 20 words, split the responsibility.

Step 2: identify which existing session state keys the agent reads and which new keys it writes. Add the new keys to the session state schema in Section 15.2.

Step 3: identify which MCP tools the agent uses. Add any new tools to the DS Tools MCP Server.

Step 4: write the system prompt following Section 32.2's structure.

Step 5: write the agent's Python module in `agents/`.

Step 6: add the agent as a node in the workflow graph in `workflows/pipeline_graph.py`. Define its dependencies (which node must complete before it starts) and its outputs (which session state keys it writes).

Step 7: add any new hooks in `hooks/registry.py`.

Step 8: write unit tests for the agent module and integration tests for its full interaction with the MCP servers and sandbox.

Step 9: update this document's Section 7 with the new agent's entry.

### 35.5 Adding a New File Type to read\_file

When adding support for a new file type:

Step 1: add the Magika content type label and its expected MIME type to `tools/file_type_detector.py`.

Step 2: write the reader class in `tools/readers/`.

Step 3: add the routing entry in `tools/file_type_detector.py`'s routing logic.

Step 4: add a downstream routing entry in `workflows/routing.py` specifying which pipeline variant this file type triggers.

Step 5: add unit tests with sample files for the new reader.

Step 6: update Section 6.3 of this document with the new file type's entry.

Step 7: update `docs/library_registry.md` if the new reader uses a library that is not already in the sandbox requirements.

---

## 36. Decision Log — Pre-Populated Entries

This section pre-populates the `docs/decision_log.md` file with the architectural decisions that were made during the design of this specification. Every entry follows the format: **Decision**, **Alternatives Considered**, **Rationale**, **Date**.

### Decision 1 — Azure OpenAI as the Exclusive Model Provider

**Decision:** use Azure OpenAI exclusively. Do not configure any other provider.

**Alternatives considered:** multi-provider setup with Azure OpenAI as primary and OpenAI consumer API as fallback; Ollama for local development.

**Rationale:** multi-provider adds authentication complexity, billing complexity, rate limit management complexity, and debugging complexity. A single provider simplifies every layer of the stack. Ollama for local development would require maintaining separate model download and management processes. The productivity cost outweighs the cost-saving benefit for a team working on production ML pipelines.

### Decision 2 — SQLite for the Tracking MCP Server

**Decision:** use SQLite with WAL mode for the Tracking MCP Server's persistence.

**Alternatives considered:** PostgreSQL, Redis, in-memory only.

**Rationale:** SQLite with WAL mode handles the write patterns of this system (append-heavy with occasional read queries) without network overhead or external service management. The database file can be inspected directly with standard SQLite tooling, which simplifies debugging. For the scale of this system (one pipeline at a time), SQLite is correct. If the system scales to concurrent pipelines, migrate to PostgreSQL.

### Decision 3 — In-Process Memory Agent Index

**Decision:** use a JSON file + NumPy array as the Memory Agent's index, not a vector database service.

**Alternatives considered:** Chroma, Qdrant, pgvector.

**Rationale:** vector database services are operational overhead (process management, connection pooling, version upgrades) that is not justified for the size of index this system produces (hundreds to low thousands of artifacts). A JSON file and NumPy cosine similarity computation runs in milliseconds for indexes of this size. If the index grows beyond 10,000 artifacts, migrate to Chroma (zero-deployment option, runs in-process).

### Decision 4 — Templates as Plain Text Files, Not Dynamic Code Generation

**Decision:** use parameterised text templates for all generated code, not LLM-generated code from scratch on every run.

**Alternatives considered:** LLM generates the full code from scratch on every invocation; code generation via a specialised code-generation model.

**Rationale:** templates are deterministic, testable, and debuggable. LLM-generated code from scratch produces different code every run, which makes debugging non-reproducible. Templates make the auto-fix loop work: the Debug Agent repairs a known structure, not a random structure. The trade-off is reduced flexibility — templates cannot handle edge cases that were not anticipated. Mitigate this by making templates parameterised enough to cover the common cases, and use the Debug Agent for the long tail.

### Decision 5 — No Azure Blob Storage, No Azure ML Workspace

**Decision:** use local filesystem for all artifact storage. Use MLflow (in-process, local) for experiment tracking.

**Alternatives considered:** Azure Blob Storage for artifacts, Azure ML Workspace for experiment tracking, Cosmos DB for lineage.

**Rationale:** adding Azure Blob Storage, Azure ML, and Cosmos DB adds three more services to provision, authenticate, and manage. This raises the barrier to entry for the system from "have an Azure OpenAI resource" to "have five Azure services correctly configured." For a first deployment, local storage is correct. The architecture is designed so that switching to Azure Blob Storage requires only changing the `write_output` and `read_artifact` tool implementations — the rest of the system is unaffected.

### Decision 6 — Magika for File Type Detection

**Decision:** use Google's Magika as the primary file type detector.

**Alternatives considered:** extension-based detection only, `python-magic` (libmagic bindings), `filetype` library only.

**Rationale:** extension-based detection fails silently on misnamed files. `python-magic` requires a C library (`libmagic`) that adds OS-level dependencies and fails on Windows without manual DLL installation. `filetype` is pure Python but relies on magic bytes, which are ambiguous for many text formats. Magika is AI-powered, trained on 100M+ files, achieves ~99% accuracy, and runs in ~5ms on CPU. It is the correct tool for this job. Its single drawback is a ~1MB model load on first call — mitigate by initialising it once at server startup.

### Decision 7 — Docker Sandbox with Network Isolation

**Decision:** execute all generated code in a Docker container with `--network none`.

**Alternatives considered:** subprocess execution on the host, a Python `exec()` sandbox, an Azure Container Instance per execution.

**Rationale:** subprocess execution on the host provides no isolation — a generated script with a bug (or an adversarial prompt injection) can access the host filesystem and network. A Python `exec()` sandbox is incomplete — it does not prevent access to OS-level resources. Azure Container Instance per execution has ~10-second cold start overhead. Docker with `--network none` provides full filesystem and network isolation, starts in under 2 seconds on a warm host, and is the industry standard for this pattern.

---

## 37. Glossary

This glossary defines every term used in this document with a precise, system-specific definition. Use these definitions consistently in code comments, commit messages, and team communication.

**Agent** — in the context of this system, an instance of `agent_framework.Agent` configured with a system prompt, a set of MCP tool references, and the middleware chain. Each agent is responsible for exactly one pipeline stage.

**Artifact** — any file produced by a pipeline stage that is consumed by a downstream stage or stored for retrospective querying. Artifacts include: datasets (raw, cleaned, feature-engineered), model files (joblib, PyTorch state dict, ONNX), metrics files, charts, reports, manifests, and transformation logs. Every artifact has a deterministic path and is registered in the Tracking MCP Server.

**Auto-fix loop** — the mechanism that automatically attempts to repair failing sandbox code using the Debug Agent, Browser-Use, and DuckDuckGo. The loop runs up to five times before escalating to a human gate.

**Checkpoint** — a record in the Tracking MCP Server's database that a specific pipeline stage completed successfully for a specific run ID. Checkpoints enable pipeline resumption after a crash.

**Content type label** — the string Magika returns identifying a file's true type, based on content analysis. Examples: `csv`, `parquet`, `pdf`, `png`, `sqlite`. Distinct from MIME type.

**Debug Agent** — a specialised agent invoked by the auto-fix loop. It receives failing code and an error, applies repair strategies in sequence, and returns repaired code. It has no authority over pipeline state.

**DS Tools MCP Server** — the MCP server that exposes data science tools as callable endpoints. Runs on port 8101. Stateless — all state lives in files or in the Tracking MCP Server.

**Feature manifest** — a JSON file produced by the Feature Engineering Agent listing every feature (original and engineered) with its transform type, source column, and importance score. The feature manifest is the contract between the Feature Engineering stage and all downstream stages.

**Human-in-the-loop gate** — a workflow pause point that requires an explicit human approval event before the pipeline continues. Gates are defined at fixed points in the workflow graph.

**MCP session** — a session established between the `MCPStreamableHTTPTool` client and an MCP server, identified by an `Mcp-Session-Id` header. One MCP session is shared across all tool calls within one agent invocation.

**Pipeline run** — one execution of the full workflow graph from start to finish (or from resume point to finish). Each pipeline run has a unique `run_id` UUID.

**Sandbox** — the Docker container in which all generated Python code is executed. The sandbox has no outbound network, limited memory and CPU, a configurable wall-clock timeout, and runs as a non-root user.

**Scenario** — a data science task classification used to select the appropriate training template. Examples: binary classification, BERT text classification, LSTM time series. The full scenario list is in Section 14.1.

**Session** — an `agent_framework.Session` object that persists conversation history and key-value state across multiple tool calls within a single agent invocation. Sessions are created by the Framework when an agent runs.

**Stage** — one node in the pipeline workflow graph. Each stage corresponds to one agent invocation. Stage names are: `health_check`, `read_file`, `ingest`, `eda`, `clean`, `feature_eng`, `train`, `tune`, `evaluate`, `explain`, `report`, `deploy`.

**Template** — a parameterised Python source file in `templates/` that the agent fills in and submits to the sandbox. Templates are the mechanism by which agents generate code deterministically.

**Tracking MCP Server** — the MCP server that records all pipeline events, artifacts, errors, and checkpoints. Runs on port 8100. Backed by a SQLite database. This is the system's audit log and coordination backbone.

**`ToolInvocation`** — the dataclass passed to every MCP tool handler by the Framework. Contains: `tool_name`, `tool_input` (dict), `session_id`, `run_id`.

**`ToolResult`** — the dataclass returned from every MCP tool handler. Contains: `result_type` (text, json, error), `text_result_for_llm` (string), optionally `raw_result`.

**Workflow graph** — the directed graph of pipeline stage nodes defined in `workflows/pipeline_graph.py`. Edges represent dependencies and routing conditions. The graph engine handles sequential execution, fan-out, fan-in, conditional routing, checkpointing, and human-in-the-loop gates.

---

## 38. Frequently Asked Questions

### Q: Why are there two MCP servers instead of one?

Two separate processes means two separate crash domains. The Tracking MCP Server must never go down during a pipeline run — it is the audit log. If the DS Tools MCP Server crashes mid-pipeline (because a new tool implementation has a bug), the Tracking MCP Server continues recording events and the pipeline can resume after the DS Tools Server is restarted. A single combined server would take both down together.

### Q: Why does the system use Docker for the sandbox rather than Azure Container Instances?

Docker starts in under 2 seconds on a warm host. Azure Container Instances take 10–15 seconds to cold start. For a pipeline that runs 10–20 sandbox executions per stage and has 8–10 stages, that is 100–200 sandbox starts per pipeline. At 15 seconds per start, the overhead is 25–50 minutes — longer than some entire pipelines. Docker is correct for latency-sensitive workloads.

### Q: Can the system process multiple pipeline runs concurrently?

In the current design, the system processes one pipeline run at a time. The Tracking MCP Server's SQLite database supports concurrent readers but has serialised writes. The sandbox executor can run multiple containers concurrently, but the MCP server file paths and the Memory Agent's index are not designed for concurrent writes from multiple runs. To support concurrent runs, replace SQLite with PostgreSQL, add a run-scoped file path namespace, and add a write lock to the Memory Agent's index. Document this as a known future work item in `docs/decision_log.md`.

### Q: Why is the Memory Agent an observer rather than a stage in the linear pipeline?

The Memory Agent's job is to index artifacts, not to produce them. Placing it in the linear pipeline would add latency to every stage (each stage must wait for the Memory Agent to finish indexing before the next stage starts). As an asynchronous observer, indexing happens in parallel with the next stage's execution. The only risk is that a pipeline crash between a stage completing and the Memory Agent finishing its indexing produces an incomplete index — mitigate by having the Memory Agent re-index any artifacts not yet in the index on startup.

### Q: What happens if Azure OpenAI is unavailable?

The Retry Middleware catches HTTP 503 with a capacity error and retries with exponential backoff. The maximum retry duration is `base_delay * (2 ** max_retries) + max_delay`, which with default settings is approximately 67 seconds. If Azure OpenAI is unavailable for longer than this, the agent call fails and the auto-fix loop is not invoked (the failure is not a code error). The Orchestrator receives the failure and triggers the human-in-the-loop gate for that stage with a message indicating the Azure OpenAI service is unavailable.

### Q: How does the system handle a file with mixed encodings in a CSV?

The CSV reader uses `chardet.detect()` on the first 10,000 bytes to detect the dominant encoding. It then reads the CSV with `pandas.read_csv(encoding=detected_encoding, encoding_errors='replace')`. The `encoding_errors='replace'` argument substitutes the Unicode replacement character for any byte that cannot be decoded with the detected encoding. The reader records the count of replaced characters in the `FileReadResult`. If the replacement character count is greater than 0.1% of total characters, the reader adds a data quality flag indicating mixed encoding.

### Q: Can the system train a model on a dataset too large to fit in the sandbox's memory?

Yes, with chunked processing. The cleaning template supports chunked `pandas.read_csv(chunksize=N)` processing. The feature engineering template can process chunks independently and concatenate at the end. The training template uses PyTorch's `DataLoader`, which never loads the full dataset into memory — it loads batches on demand. For datasets that do not fit on disk inside the sandbox, the pipeline must be redesigned to reference the data from external storage — this is a known limitation of the current design and is documented in `docs/decision_log.md`.

---

*End of specification. Version history is maintained in Git. This document was authored from the perspective of a Senior AI Architect and Engineer with production experience building multi-agent data science systems. All decisions are documented in Section 36. All terms are defined in Section 37.*

---

## 39. Inter-Agent Contract Tables

### 39.1 What Inter-Agent Contracts Are

Every agent in the pipeline has an explicit, typed contract: what it receives from the previous stage, what it produces, and what it records in the Tracking MCP Server. These contracts are the connective tissue of the system. Violating a contract — an agent that produces a different output schema than documented, or reads a session state key that a previous agent did not write — produces failures that are hard to trace because they appear several stages after the actual error.

These tables are the authoritative source of truth for inter-agent data flow. When an agent's implementation deviates from its contract, fix the implementation, not the contract. If a contract genuinely needs to change, update this table, update the agent's system prompt, update the receiving agent's system prompt, and document the change in `docs/decision_log.md`.

### 39.2 Contract Table Format

Each contract table has four columns:

**From** — the session state key or artifact path the current agent reads as input.

**Type** — the Python type of the value.

**Produced by** — which upstream agent or system component writes this value.

**Used for** — what the current agent does with this value.

### 39.3 Orchestrator Agent — Input Contract

| From | Type | Produced by | Used for |
|---|---|---|---|
| `--file` CLI argument | string | user input | path to input file passed to `read_file` |
| `--task` CLI argument | string | user input | natural-language task description used for routing and agent prompting |
| `--resume-run-id` (optional) | string or None | user input | if set, query `query_run_status` and skip completed checkpoints |

### 39.4 Orchestrator Agent — Output Contract

| To | Type | Written to | Consumed by |
|---|---|---|---|
| `run_id` | string (UUID) | session state | all sub-agents, all MCP tool calls, all log entries |
| `task_description` | string | session state | all sub-agents receive it in session initialisation |
| `pipeline_variant` | string (A, B, or C) | session state | determines which stages are active in the workflow graph |
| `human_gate_results` | dict (gate name → decision) | session state | agents that follow human gates read the decision |

### 39.5 Ingestion Agent — Input Contract

| From | Type | Produced by | Used for |
|---|---|---|---|
| `input_artifact_paths[0]` | string (file path) | Orchestrator (from CLI `--file`) | passed to `read_file` |
| `task_description` | string | Orchestrator | validates routing alignment between task and file type |

### 39.6 Ingestion Agent — Output Contract

| To | Type | Written to | Consumed by |
|---|---|---|---|
| `input_artifact_paths[0]` (updated) | string (path to raw copy) | session state | EDA Agent reads the raw snapshot |
| `file_type_result` | dict (FileReadResult JSON) | session state | Orchestrator uses it to confirm pipeline variant |
| `schema` | dict (column name → dtype) | session state | EDA, Cleaning, Feature Engineering agents |
| Tracking MCP `record_artifact` | artifact record | Tracking MCP Server | Memory Agent indexes it; Orchestrator queries on resume |

### 39.7 EDA Agent — Input Contract

| From | Type | Produced by | Used for |
|---|---|---|---|
| `input_artifact_paths[0]` | string (dataset path) | Ingestion Agent | dataset path passed into the EDA template |
| `schema` | dict | Ingestion Agent | column names and dtypes used to parameterise the EDA template |
| `task_description` | string | Orchestrator | used to identify the target column |

### 39.8 EDA Agent — Output Contract

| To | Type | Written to | Consumed by |
|---|---|---|---|
| `eda_report_path` | string (path to `stats.json`) | session state | Cleaning Agent reads it to enumerate quality issues |
| `eda_narrative_path` | string (path to `eda_narrative.md`) | session state | Report Agent includes it in the final report |
| `chart_uris` | list of strings (file paths) | session state | Report Agent embeds them in the HTML report |
| `data_quality_flags` | list of dicts | session state | Cleaning Agent reads them to prioritise cleaning steps |
| Tracking MCP `record_artifact` (one per output file) | artifact records | Tracking MCP Server | Memory Agent indexes them |

### 39.9 Cleaning Agent — Input Contract

| From | Type | Produced by | Used for |
|---|---|---|---|
| `input_artifact_paths[0]` | string (raw dataset path) | Ingestion Agent | the dataset to clean |
| `eda_report_path` | string | EDA Agent | read to enumerate quality issues; must exist before Cleaning starts |
| `data_quality_flags` | list of dicts | EDA Agent | prioritises which issues to address first |
| `schema` | dict | Ingestion Agent | used to verify dtypes after cleaning |

### 39.10 Cleaning Agent — Output Contract

| To | Type | Written to | Consumed by |
|---|---|---|---|
| `cleaned_dataset_path` | string (path to `cleaned_dataset.parquet`) | session state | Feature Engineering, Training, Evaluation agents |
| `transformation_log_path` | string | session state | Report Agent includes a summary; Evaluation Agent references it for drift analysis |
| `cleaning_summary` | dict | session state | Report Agent includes it in the data quality section |
| `content_hash` | string (SHA-256 of cleaned_dataset.parquet) | session state | used as version identifier for the cleaned dataset |
| Tracking MCP `record_artifact` | artifact records | Tracking MCP Server | Memory Agent indexes them |

### 39.11 Feature Engineering Agent — Input Contract

| From | Type | Produced by | Used for |
|---|---|---|---|
| `cleaned_dataset_path` | string | Cleaning Agent | the dataset to engineer features from |
| `schema` | dict | Ingestion Agent | column types inform which transforms are applicable |
| `task_description` | string | Orchestrator | used to identify the target column; also used in DuckDuckGo domain research query |
| `transformation_log_path` | string | Cleaning Agent | read to avoid applying transforms to already-transformed columns |

### 39.12 Feature Engineering Agent — Output Contract

| To | Type | Written to | Consumed by |
|---|---|---|---|
| `features_train_path` | string (path to `features_train.parquet`) | session state | Training Agent reads it for model training |
| `features_test_path` | string (path to `features_test.parquet`) | session state | Evaluation Agent reads it for final evaluation |
| `feature_manifest_path` | string (path to `feature_manifest.json`) | session state | Training, Evaluation, Explainability, Report agents |
| `target_column` | string | session state | all downstream agents need to know the target column name |
| `task_type` | string (scenario label from Section 14.1) | session state | Training Agent uses it to select the correct template |
| Tracking MCP `record_artifact` | artifact records | Tracking MCP Server | Memory Agent indexes them |

### 39.13 Model Selection and Training Agent — Input Contract

| From | Type | Produced by | Used for |
|---|---|---|---|
| `features_train_path` | string | Feature Engineering Agent | training data path passed into the training template |
| `feature_manifest_path` | string | Feature Engineering Agent | read to get the feature list and target column |
| `target_column` | string | Feature Engineering Agent | passed into training template as the target |
| `task_type` | string | Feature Engineering Agent | used to select the correct training template |
| `task_description` | string | Orchestrator | used in DuckDuckGo algorithm research query |

### 39.14 Model Selection and Training Agent — Output Contract

| To | Type | Written to | Consumed by |
|---|---|---|---|
| `model_artifact_path` | string (path to model file) | session state | HP Tuning, Evaluation, Explainability, Deployment agents |
| `model_type` | string (sklearn, pytorch, xgboost, etc.) | session state | all downstream agents use it to select the right explainer, serialiser, and evaluator |
| `baseline_metrics` | dict | session state | HP Tuning Agent verifies improvement; Report Agent includes it in comparison table |
| `algorithm_comparison` | list of dicts | session state | Report Agent uses it to populate the model comparison table |
| `onnx_model_path` | string (path to `model.onnx`) | session state | Deployment Agent uses ONNX for portability |
| Tracking MCP `record_artifact` and `log_metrics` | artifact and metric records | Tracking MCP Server | Memory Agent indexes them |

### 39.15 HP Tuning Agent — Input Contract

| From | Type | Produced by | Used for |
|---|---|---|---|
| `model_artifact_path` | string | Training Agent | the baseline model to tune |
| `model_type` | string | Training Agent | used to define a model-type-appropriate search space |
| `features_train_path` | string | Feature Engineering Agent | training data for tuning trials |
| `baseline_metrics` | dict | Training Agent | used to verify that tuning improves on the baseline |

### 39.16 HP Tuning Agent — Output Contract

| To | Type | Written to | Consumed by |
|---|---|---|---|
| `model_artifact_path` | string (updated to tuned model path) | session state | replaces the baseline model path for Evaluation, Explainability, Deployment |
| `best_params_path` | string (path to `best_params.json`) | session state | Report Agent includes it in the best model section |
| `tuning_metrics` | dict | session state | Report Agent includes it; compared to `baseline_metrics` to show improvement |
| Tracking MCP `record_artifact` and `log_metrics` | artifact and metric records | Tracking MCP Server | Memory Agent indexes them |

### 39.17 Evaluation Agent — Input Contract

| From | Type | Produced by | Used for |
|---|---|---|---|
| `model_artifact_path` | string | HP Tuning Agent | the tuned model to evaluate |
| `features_test_path` | string | Feature Engineering Agent | held-out test set for final evaluation |
| `feature_manifest_path` | string | Feature Engineering Agent | used to identify protected attributes for fairness analysis |
| `model_type` | string | Training Agent | used to select the correct evaluation approach |
| `task_type` | string | Feature Engineering Agent | used to select the correct metrics |
| `target_column` | string | Feature Engineering Agent | the column being predicted |

### 39.18 Evaluation Agent — Output Contract

| To | Type | Written to | Consumed by |
|---|---|---|---|
| `evaluation_report_path` | string (path to `evaluation_report.json`) | session state | Explainability, Report, Deployment agents |
| `deployment_recommendation` | dict (`recommend`, `blocking_issues`, `advisory_issues`) | session state | `before_deploy` hook reads this; Report Agent includes it |
| `fairness_metrics` | dict | session state | Report Agent includes it in the fairness section |
| `drift_report` | dict | session state | Report Agent includes it; Deployment Agent uses it to configure monitoring thresholds |
| Tracking MCP `record_artifact` and `log_metrics` | artifact and metric records | Tracking MCP Server | Memory Agent indexes them |

### 39.19 Explainability Agent — Input Contract

| From | Type | Produced by | Used for |
|---|---|---|---|
| `model_artifact_path` | string | HP Tuning Agent | the model to explain |
| `features_test_path` | string | Feature Engineering Agent | the dataset to compute SHAP values on |
| `feature_manifest_path` | string | Feature Engineering Agent | feature names for SHAP plot labels |
| `model_type` | string | Training Agent | used to select the correct SHAP explainer |
| `evaluation_report_path` | string | Evaluation Agent | read to get top-performing metrics for the narrative |

### 39.20 Explainability Agent — Output Contract

| To | Type | Written to | Consumed by |
|---|---|---|---|
| `shap_values_path` | string (path to `shap_values.npy`) | session state | Report Agent includes the plots |
| `shap_plot_paths` | list of strings (chart file paths) | session state | Report Agent embeds them in the HTML report |
| `explanation_narrative_path` | string (path to `narrative.md`) | session state | Report Agent includes it in the explainability section |
| Tracking MCP `record_artifact` | artifact records | Tracking MCP Server | Memory Agent indexes them |

### 39.21 Report Agent — Input Contract

| From | Type | Produced by | Used for |
|---|---|---|---|
| `eda_narrative_path` | string | EDA Agent | EDA section of the report |
| `eda_report_path` | string | EDA Agent | statistics for the dataset overview section |
| `chart_uris` | list of strings | EDA Agent | embedded in the HTML report |
| `transformation_log_path` | string | Cleaning Agent | data quality section |
| `cleaning_summary` | dict | Cleaning Agent | data quality section |
| `feature_manifest_path` | string | Feature Engineering Agent | feature engineering section |
| `algorithm_comparison` | list of dicts | Training Agent | model comparison table |
| `best_params_path` | string | HP Tuning Agent | best model section |
| `evaluation_report_path` | string | Evaluation Agent | metrics section and deployment recommendation |
| `fairness_metrics` | dict | Evaluation Agent | fairness section |
| `shap_plot_paths` | list of strings | Explainability Agent | explainability section |
| `explanation_narrative_path` | string | Explainability Agent | explainability section |
| `deployment_recommendation` | dict | Evaluation Agent | deployment recommendation section |

### 39.22 Report Agent — Output Contract

| To | Type | Written to | Consumed by |
|---|---|---|---|
| `report_md_path` | string (path to `final_report.md`) | session state | Memory Agent indexes it |
| `report_html_path` | string (path to `final_report.html`) | session state | delivered to the user as the final deliverable |
| Tracking MCP `record_artifact` | artifact records | Tracking MCP Server | Memory Agent indexes them |

### 39.23 Deployment Agent — Input Contract

| From | Type | Produced by | Used for |
|---|---|---|---|
| `model_artifact_path` | string | HP Tuning Agent | the model to deploy |
| `onnx_model_path` | string | Training Agent | ONNX export for portability |
| `model_type` | string | Training Agent | used to select serialisation and serving approach |
| `feature_manifest_path` | string | Feature Engineering Agent | used to generate the prediction endpoint's input schema |
| `deployment_recommendation` | dict | Evaluation Agent | `before_deploy` hook reads this; deployment blocked if `recommend == "block"` |
| `drift_report` | dict | Evaluation Agent | used to set monitoring thresholds |
| `report_md_path` | string | Report Agent | model card section extracted for the endpoint's `/model_card` route |

### 39.24 Deployment Agent — Output Contract

| To | Type | Written to | Consumed by |
|---|---|---|---|
| `endpoint_url` | string | session state | delivered to the user; Memory Agent records it |
| `smoke_test_results` | dict (10 predictions with latencies) | session state | `after_deploy` hook checks P99 |
| Tracking MCP `record_artifact` | artifact records (container image, endpoint URL, smoke test results) | Tracking MCP Server | Memory Agent indexes them |

---

## 40. Testing Strategy and Test Case Taxonomy

### 40.1 Testing Philosophy

Tests in this system serve three distinct purposes and are organised into three distinct categories. Do not mix categories in a single test file. A test that does multiple things at once is a test that is hard to fix when it fails.

**Unit tests** verify that a single function or class behaves correctly in isolation. They use mocks for all external dependencies. They run in milliseconds. If a unit test takes more than 1 second, it is making a real network call or real file I/O — fix it.

**Integration tests** verify that two or more components work correctly together. They may use real Docker containers and real file I/O, but they use recorded Azure OpenAI responses (not live API calls). They run in seconds to low tens of seconds.

**End-to-end tests** verify that the entire pipeline runs correctly from a real input file to a real deployed endpoint. They make real Azure OpenAI calls. They run in minutes. They run only on pull requests to `main` and on the release deployment workflow.

### 40.2 Unit Test Taxonomy

For each module, write unit tests covering these categories in order of priority:

**Happy path** — normal successful execution with valid inputs. This is the first test written. If you cannot write a passing happy path test, you do not understand what the function is supposed to do.

**Empty inputs** — what happens with an empty string, empty list, empty dict, or None. The function must either handle these gracefully or raise a clearly named exception with a descriptive message.

**Invalid inputs** — wrong type, value out of range, or a value violating a business rule. The function must raise a clearly named exception.

**Boundary conditions** — the exact edge of the valid input range. For `read_file`: a file with exactly one row (header only), a file with no columns, a file at exactly the maximum size. For EDA template: a dataset with exactly one numeric column, only categorical columns, all null values in one column.

**Error propagation** — when a dependency raises an exception, does the function propagate it correctly, wrap it in a domain-specific exception, or swallow it? Swallowing is always wrong. Test all three scenarios.

### 40.3 Unit Tests for FileTypeDetector

Write the following test cases for `tools/file_type_detector.py`:

A CSV file with a `.csv` extension — confirm `detected_type == "csv"` and `confidence > 0.9`.

A Parquet file with a `.parquet` extension — confirm `detected_type == "parquet"`.

A CSV file renamed as `.json` (extension mismatch) — confirm `detected_type == "csv"` despite the wrong extension. This test validates that Magika's content-based detection overrides extension-based detection.

A PDF file — confirm `detected_type == "pdf"`.

A PyTorch state dict pickle file — confirm `detected_type == "pickle"` and the result includes `pickle_top_level_type` indicating a dict.

A ZIP archive — confirm `detected_type == "zip"` and `member_count > 0`.

An empty file (0 bytes) — confirm `supported == False` and no exception is raised.

A file with corrupted first 16 bytes — confirm the detector falls back gracefully without crashing.

A scanned PDF (no extractable text) — confirm `requires_ocr == True`.

A plain text file with JSON content — confirm `detected_type == "json"`, not `"txt"`.

### 40.4 Unit Tests for Middleware

**LoggingMiddleware** — after one call, exactly one log entry appears in captured stdout. The entry is valid JSON. It contains all required fields. `input_token_count` is populated. `duration_ms` is greater than zero.

**RateLimitMiddleware** — with a bucket size of 2 tokens and a refill rate of 1 token per second, three consecutive calls: the first two succeed immediately, the third is delayed by approximately 1 second. Measure delay with `time.monotonic()`. The delay must be between 0.8 and 1.5 seconds.

**SafetyMiddleware** — 10 blocked patterns in separate tests, each confirming `SafetyViolationError` is raised. 5 clean inputs sharing keywords with blocked patterns but not matching — confirm they pass through unchanged.

**RetryMiddleware** — mock a dependency that returns HTTP 429 twice then HTTP 200. Confirm exactly two retries and the successful response is returned. Mock HTTP 400 — confirm no retry. Confirm exponential backoff is applied by measuring total elapsed time.

**TelemetryMiddleware** — after one call, exactly one OpenTelemetry span is created and finished. The span has the correct name, `run_id` attribute, and `agent_name` attribute. Use an in-memory span exporter for the test.

### 40.5 Unit Tests for Hooks

`before_code_execute` — pass code containing a file deletion on a non-workspace path. Confirm `HookRejectionError`. Pass clean code. Confirm returned unchanged. Pass code with a syntax error — confirm `SyntaxError` from `ast.parse()` is raised before `bandit` runs.

`after_code_execute` — pass a success result with two output files. Confirm `record_artifact` is called twice on the Tracking MCP Server mock. Pass a failure result. Confirm `on_code_error` fires and `record_artifact` is not called.

`before_model_train` — pass training code missing `torch.manual_seed`. Confirm `HookRejectionError` with a message identifying the missing seed. Pass complete training code. Confirm the hook passes through.

`before_deploy` — pass `deployment_recommendation` with `recommend == "block"` and one blocking issue. Confirm `HookRejectionError`. Pass `recommend == "conditional"` — confirm human-in-the-loop gate is triggered. Pass `recommend == "proceed"` — confirm no exception.

### 40.6 Integration Test Taxonomy

**test_tracking_mcp_server.py** — start the Tracking MCP Server. Call `record_start`, `record_end`, `record_artifact`. Call `query_run_status` and confirm the run has one entry with correct fields. Call `query_best_artifact` with a metric name and confirm it returns the artifact with the highest metric value.

**test_ds_tools_mcp_server.py** — start the DS Tools MCP Server. Call `read_file` with a test CSV and confirm the returned schema matches the file's actual schema. Call `execute_code` with a script writing a metrics file. Confirm the file appears in the artifact list. Call `web_research` with a simple query and confirm 10 results are returned.

**test_ingestion_agent.py** — start both MCP servers. Create the Ingestion Agent with mocked Azure OpenAI responses. Pass a test CSV. Confirm the agent calls `read_file`, validates the result, and calls `record_artifact`. Test the routing mismatch case: pass an image file with a regression task description. Confirm the mismatch is flagged.

**test_auto_fix_loop.py** — run `execute_code` with a deliberate `ModuleNotFoundError`. Confirm the auto-fix loop fires Attempt 1 (pattern match), repairs the code, re-runs, and returns success. Run with a deliberate `KeyError` referencing a column in the feature manifest. Confirm Attempt 1 looks up the correct column name.

**test_browser_use_trigger.py** — run the Explainability Agent's SHAP execution with a deliberately incorrect SHAP API call. Confirm Attempt 2 fires Browser-Use, documentation is retrieved, and the repaired code uses the correct current-version API.

**test_memory_agent.py** — run the Memory Agent after inserting three test artifact records into the Tracking MCP Server. Confirm all three are indexed. Call `semantic_search` with a query matching one specifically. Confirm the top result is the correct artifact.

### 40.7 End-to-End Test Structure

Each e2e test in `tests/e2e/` must follow this structure in order:

Step 1 — Setup: start both MCP servers, verify health, copy the test dataset to a temp directory.

Step 2 — Run: execute the pipeline with `--pipeline_human_in_the_loop=false`. Capture the `run_id`.

Step 3 — Verify artifacts: query the Tracking MCP Server for all artifacts. Verify every expected artifact type is present (dataset, model, evaluation_report, explanation, report_html).

Step 4 — Verify metrics: query logged metrics. Verify the primary metric meets the minimum threshold from Section 40.8.

Step 5 — Verify deployment: call the deployed endpoint's `/predict` route with a sample input. Verify response schema matches the expected output.

Step 6 — Verify memory: query the Memory Agent with a task-specific query. Verify the answer references artifacts from this run.

Step 7 — Teardown: stop the deployed endpoint container, delete the temp directory.

### 40.8 Minimum Acceptable Metric Thresholds per Benchmark Dataset

| Dataset | Task Type | Primary Metric | Minimum Threshold |
|---|---|---|---|
| Titanic | Binary Classification | ROC-AUC | 0.83 |
| California Housing | Regression | R² | 0.78 |
| Credit Card Fraud | Imbalanced Classification | PR-AUC | 0.65 |
| IMDB Sentiment | Text Classification | Accuracy | 0.91 |
| Air Passengers | Time Series | SMAPE | less than 12.0 |

---

## 41. Performance Optimisation Guide

### 41.1 Identifying Bottlenecks

Before optimising, measure. Every pipeline run produces a distributed trace in Application Insights. Identify the longest span. That is the bottleneck. Common bottlenecks and their causes:

Azure OpenAI call latency — caused by large prompts, long conversation history, or under-provisioned deployments.

Sandbox cold start — caused by the Docker image not being cached on the host, or a new container being started for every execution call.

Magika model load — caused by initialising Magika on every `read_file` call rather than once at server startup.

MCP round-trip latency — caused by MCP session overhead on the first call, or large tool result payloads being serialised and deserialised.

### 41.2 Azure OpenAI Latency Reduction

Reduce prompt size. Every token adds latency. The EDA Agent's statistics prompt should contain only column names, dtypes, and summary statistics — not raw data. Pass summary statistics, not the full `describe()` output.

Use `max_tokens` correctly. Set it to the minimum value that produces correct outputs. Overprovisioning adds latency even when the model produces fewer tokens than the limit.

Use the fast deployment for agents that do not need extended reasoning. Review deployment assignments once per quarter.

Use streaming for long completions. The Report Agent's narrative generation is a long completion — stream it so the application can begin processing before it is complete.

### 41.3 Sandbox Startup Latency Reduction

Use container reuse. Maintain a pool of warm containers (configurable, default 3). Each warm container runs a supervisor process that waits for a code file, executes it, writes results, and signals readiness for the next execution. The `execute_code` tool claims a container from the pool, mounts the code, triggers execution, waits for completion, and returns the container to the pool.

This reduces the per-execution overhead from 2 seconds to under 200ms.

### 41.4 Memory Footprint Reduction

Use `python:3.12-slim` as the base image. Install only the libraries in `sandbox/requirements.sandbox.txt`. Use multi-stage Docker builds to separate build dependencies from runtime. Delete `.pyc` files and `__pycache__` directories in the final image layer.

### 41.5 MCP Round-Trip Latency Reduction

Initialise `MCPStreamableHTTPTool` instances at application startup and reuse them across all agent sessions. Do not create new instances per pipeline run or per agent.

For the `execute_code` tool, do not include stdout in the tool result when stdout exceeds 10KB — write stdout to a file and include the file path instead. For `read_file`, do not include file content in the result — include only schema, sample rows, and metadata.

### 41.6 Parallelising Independent Stages

After Feature Engineering completes, model selection trial runs (three algorithms) can run in parallel. Each training job is a separate `execute_code` call. Issue all three concurrently using `asyncio.gather()`. Collect results and select the best. This reduces the Training stage's wall-clock time by approximately 60%.

Implement parallel trial execution after the sequential version is verified correct. Never parallelise prematurely.

---

## 42. Known Limitations and Future Roadmap

### 42.1 Current Known Limitations

**Single concurrent pipeline** — the system processes one pipeline run at a time. To support concurrent runs: replace SQLite with PostgreSQL, add run-scoped file path namespacing, and add a write lock to the Memory Agent's index.

**Local file storage only** — all artifacts are stored on the local filesystem. The architecture supports migrating to remote storage by replacing the `write_output` tool's implementation only — the rest of the system is unaffected.

**No real-time streaming of pipeline progress** — the CLI outputs JSONL to stdout but there is no WebSocket or SSE endpoint for a web UI.

**Audio and video processing is flagged, not executed** — the `read_file` tool detects these file types and flags them but does not automatically route to a transcription or vision pipeline.

**No formal model version registry** — the system assigns content hashes to artifacts but does not de-duplicate or formally version models across runs.

**No drift monitoring after deployment** — the Deployment Agent configures the endpoint but does not run ongoing drift checks automatically.

**The Debug Agent cannot fix architectural errors** — the auto-fix loop handles syntactic errors and API changes. Fundamental architectural errors (data leakage bugs, wrong loss function for task type) require human investigation.

### 42.2 Future Roadmap (Priority Order)

**Concurrent pipeline support** — PostgreSQL, run-scoped namespacing, thread-safe index. Estimated effort: 2 weeks.

**Web UI for pipeline progress and human gates** — FastAPI WebSocket backend, React frontend. Estimated effort: 3 weeks.

**Drift monitoring scheduler** — daily drift check comparing live prediction log to the training distribution baseline. Estimated effort: 1 week.

**Federated Memory Agent** — query across multiple index files from different users or teams. Estimated effort: 1 week.

**Remote storage migration** — replace `write_output` local implementation with a remote storage implementation. No other changes required. Estimated effort: 2 days.

**A2A protocol compliance** — implement A2A so the system's agents can be discovered and invoked by agents in other frameworks. Estimated effort: 1 week after A2A specification is finalised.

**Streaming report generation** — stream HTML report generation to a browser in real time. Estimated effort: 3 days.

**Model card API endpoint** — expose the model card JSON at `/model_card` on the deployed endpoint. Estimated effort: 1 day.

---

## 43. New Engineer Onboarding Guide

### 43.1 Prerequisites

Set up these tools before your first day:

Python 3.12 installed via `pyenv`. Docker Desktop (macOS/Windows) or Docker Engine (Linux). Azure CLI with `az login` completed. Git with SSH key configured. VS Code with the Python extension.

### 43.2 Day One Setup

Follow these steps in order. Do not skip or reorder.

Step 1 — Clone the repository. Read the README completely.

Step 2 — Read this document. Read Sections 1, 2, 3, and 4 first. Then read the phase you have been assigned to implement.

Step 3 — Create the virtual environment: `python3.12 -m venv .venv && source .venv/bin/activate`.

Step 4 — Install dependencies: `pip install -e ".[dev]"`.

Step 5 — Copy `.env.example` to `.env`. Fill in all Azure OpenAI values from the team's shared secrets manager.

Step 6 — Build the sandbox image: `docker build -f sandbox/Dockerfile.sandbox -t ds-agent-sandbox:latest .`. This takes 5–10 minutes on first build.

Step 7 — Run unit tests: `pytest tests/unit/ -v`. All tests must pass.

Step 8 — Start both MCP servers in separate terminals. Verify their `/health` endpoints return `{"status": "healthy"}`.

Step 9 — Run the hello-world integration test: `pytest tests/integration/test_tracking_mcp_server.py::test_hello_world -v`.

Step 10 — Run a full pipeline on the Titanic fixture: `python main.py --file tests/fixtures/titanic.csv --task "Predict passenger survival. Binary classification. Optimise for ROC-AUC." --pipeline_human_in_the_loop=false`. Open `outputs/<run_id>/final_report.html` in a browser when complete.

### 43.3 Understanding the Codebase — Reading Order

Read source files in this order to build understanding layer by layer:

1. `config/settings.py`
2. `config/safety_patterns.py`
3. `middleware/chain.py`
4. `middleware/logging_mw.py`
5. `hooks/registry.py`
6. `tools/file_type_detector.py`
7. `mcp_servers/tracking/tools.py`
8. `mcp_servers/ds_tools/tools.py`
9. `workflows/pipeline_graph.py`
10. `agents/orchestrator.py`
11. The agent file for your assigned work
12. The template file(s) your agent uses

### 43.4 Making Your First Change

Step 1 — Read the contract tables in Section 39 for the agent you are modifying.

Step 2 — Read the system prompt in the agent's `agents/*.py` file.

Step 3 — Write a failing unit test demonstrating the intended change.

Step 4 — Make the change.

Step 5 — Verify the unit test passes.

Step 6 — Run the full unit test suite: `pytest tests/unit/ -v`.

Step 7 — If you changed a template, run the relevant integration test.

Step 8 — If you changed a contract, update Section 39's contract tables and both the producing and receiving agent's system prompts.

Step 9 — Commit with the format: `<type>(<scope>): <description>`. Types: `feat`, `fix`, `docs`, `test`, `refactor`.

### 43.5 Common Mistakes and How to Avoid Them

**Storing data in session state instead of files** — session state is for paths, metadata, and small scalar values only. Never store a DataFrame, NumPy array, or model object in session state.

**Calling Azure OpenAI directly from a tool implementation** — tools must not call Azure OpenAI completions. If a tool needs LLM reasoning, that reasoning belongs in the agent's prompt. The sole exception is the `embed_text` tool.

**Adding retry inside a tool implementation** — the RetryMiddleware handles retries. Tool-level retry creates double-retry with exponentially multiplied wait times.

**Writing to paths outside `/workspace/output/` inside a template** — the sandbox's non-root user can only write to `/workspace/output/`. All template output must go there.

**Skipping the `before_model_train` hook check** — the hook verifies training loop checklist completeness. Missing items produce models that overfit, are non-reproducible, or fail on production edge cases.

**Assigning the primary model to agents that do not need it** — misassignment wastes quota and causes rate limiting that slows the entire pipeline.

---

## 44. Monitoring and Alerting Specification

### 44.1 Application-Level Metrics to Collect

Collect these custom metrics in Application Insights beyond the standard OpenTelemetry traces:

`pipeline.run.count` — increment on each pipeline run start. Dimension: `pipeline_variant`.

`pipeline.run.duration_seconds` — record on each pipeline run end. Dimension: `pipeline_variant`, `final_status`.

`pipeline.stage.duration_seconds` — record on each stage completion. Dimension: `stage_name`, `status`.

`pipeline.auto_fix.attempt_count` — histogram of auto-fix attempts needed before success. Distribution skewed toward higher values signals outdated sandbox libraries or systematic template issues.

`pipeline.auto_fix.escalation_count` — count of auto-fix escalations (all five attempts failed). Must be near zero in a healthy system.

`pipeline.token_usage.total` — cumulative Azure OpenAI tokens per pipeline run. Dimension: `deployment_name`.

`pipeline.sandbox.execution_duration_seconds` — histogram of sandbox execution times. A long tail indicates templates running close to the timeout.

`read_file.detection_confidence` — histogram of Magika's confidence scores. Significant mass below 0.85 means the system is frequently encountering unusual file types.

### 44.2 Alert Conditions

`auto_fix.escalation_count > 0` in any 1-hour window — escalations signal a systemic problem. Alert immediately. Priority: high.

`pipeline.run.duration_seconds p95 > 3600` — a pipeline exceeding one hour has a bottleneck or an infinite loop. Alert after 30 minutes. Priority: medium.

`pipeline.token_usage.total > 800000` for a single run — a run at 800,000 tokens has either bypassed the budget check or has a runaway prompt. Priority: medium.

`pipeline.sandbox.execution_duration_seconds p99 > 250` — templates running close to the 300-second timeout. Priority: low, trend-watch.

`read_file.detection_confidence p10 < 0.7` — 10% of files detected with low confidence. May indicate a new file type not handled well. Priority: low, investigate.

Azure OpenAI HTTP 429 rate exceeding 5 per hour — deployment quota being hit frequently. Increase quota or reduce token usage. Priority: medium.

---

## 45. Data Contract Validation

### 45.1 Why Runtime Validation Matters

The inter-agent contracts in Section 39 define what each agent must produce. But a contract defined only in a document is not enforced at runtime. An agent that produces an `evaluation_report.json` with a missing `fairness_metrics` key does not fail immediately — it fails silently when the Report Agent tries to access that key several stages later, producing a confusing error that appears to be a Report Agent bug.

Implement runtime contract validation at the boundary of every stage transition. When an agent finishes and writes to session state, validate that all promised output keys are present and have the correct types. Fail immediately with a clear error identifying which agent produced the violation.

### 45.2 Validation Implementation

Implement `workflows/contract_validator.py` as a pure validation module. It contains one validator class per agent, each implementing a `validate_output(session_state: dict) -> list[ValidationError]` method. The method returns an empty list on success and a list of `ValidationError` objects on failure.

Each `ValidationError` has fields: `producing_agent`, `key`, `expected_type`, `actual_value` (or None if missing), `severity` (blocking or advisory).

The Workflow Graph engine calls the validator after each stage node completes. If any blocking `ValidationError` is returned, the graph halts and routes to the human-in-the-loop gate. Advisory errors are logged but do not halt the pipeline.

### 45.3 Critical Validations to Implement

After Ingestion Agent: `schema` key is present and is a dict with at least one key. `input_artifact_paths[0]` is a string pointing to an existing file.

After EDA Agent: `eda_report_path` is a string pointing to an existing `stats.json`. The file is valid JSON containing `describe`, `null_counts`, and `correlation_matrix` keys.

After Cleaning Agent: `cleaned_dataset_path` points to an existing `.parquet` file. The file is readable with a non-zero row count. `transformation_log_path` exists.

After Feature Engineering Agent: `features_train_path` and `features_test_path` both exist. `feature_manifest_path` exists. `target_column` is a non-empty string. `task_type` is one of the valid scenario labels from Section 14.1.

After Training Agent: `model_artifact_path` points to an existing file. `model_type` is a non-empty string. `baseline_metrics` contains the primary metric key appropriate for the task type.

After HP Tuning Agent: `model_artifact_path` has been updated (differs from the baseline model path). `best_params_path` exists.

After Evaluation Agent: `evaluation_report_path` exists. The evaluation report JSON contains `recommend`, `blocking_issues`, `fairness_metrics`, and `drift_report` keys. `deployment_recommendation` has all three required fields.

After Explainability Agent: `shap_values_path` exists. `shap_plot_paths` is a list with at least two entries. `explanation_narrative_path` exists.

---

## 46. Complete Table of Contents Addendum

The following sections were added in the continuation of this document beyond the original 29 sections. Update the Table of Contents at the document top to include entries 30 through 46.

- Section 30 — Sandbox Requirements File — Authoritative Package List
- Section 31 — Template Authoring Guide
- Section 32 — Agent System Prompt Authoring Guide
- Section 33 — CI/CD Pipeline Specification
- Section 34 — Operational Runbook
- Section 35 — Upgrade and Migration Guide
- Section 36 — Decision Log — Pre-Populated Entries
- Section 37 — Glossary
- Section 38 — Frequently Asked Questions
- Section 39 — Inter-Agent Contract Tables
- Section 40 — Testing Strategy and Test Case Taxonomy
- Section 41 — Performance Optimisation Guide
- Section 42 — Known Limitations and Future Roadmap
- Section 43 — New Engineer Onboarding Guide
- Section 44 — Monitoring and Alerting Specification
- Section 45 — Data Contract Validation
- Section 46 — Complete Table of Contents Addendum

---

*End of specification. Total sections: 46. Total guidance pages: complete. Version history is maintained in Git. This document was authored from the perspective of a Senior AI Architect and Engineer with production experience building multi-agent data science systems. All architectural decisions are documented in Section 36. All terms are defined in Section 37. All inter-agent contracts are defined in Section 39. All test cases are defined in Section 40. Do not implement any agent, tool, template, or middleware class without first reading the section of this document that governs it.*
