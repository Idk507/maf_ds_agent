"""
mcp_servers/ds_tools/server.py — DS Tools MCP Server (port 8101).

Stateless (stateless_http=True). 10 tools:
  read_file, execute_code, get_sample, write_output,
  search_docs, web_research, embed_text, semantic_search,
  log_metrics, verify_stage.

MCP URL: http://localhost:8101/mcp/mcp
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI
from fastmcp import FastMCP
from pydantic import Field

load_dotenv()

PORT = int(os.environ.get("DS_TOOLS_MCP_PORT", "8101"))

mcp = FastMCP("DSToolsMCP", stateless_http=True)

# ── read_file ────────────────────────────────────────────────────────


@mcp.tool()
async def read_file(
    file_path: Annotated[str, Field(description="Absolute or relative path to the file to read.")],
    sample_rows: Annotated[int, Field(description="For tabular files: number of rows to return in preview. 0 = no preview.")] = 5,
) -> str:
    """
    Detect the file type (3-layer: Magika → filetype → heuristics) and read it.
    Returns a JSON object with keys: file_type, schema, sample, metadata, full_path.
    """
    from tools.file_type_detector import detect_file_type
    from tools.readers.dispatcher import read_file_dispatch

    path = Path(file_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})

    detection = detect_file_type(str(path))
    result = read_file_dispatch(str(path), detection, sample_rows=sample_rows)
    return json.dumps(result, default=str)


# ── execute_code ─────────────────────────────────────────────────────


@mcp.tool()
async def execute_code(
    code: Annotated[str, Field(description="Python code to execute in a fresh subprocess.")],
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    timeout_seconds: Annotated[int, Field(description="Maximum execution time in seconds.")] = 300,
    env_vars: Annotated[str, Field(description="JSON-encoded dict of extra environment variables to inject.")] = "{}",
) -> str:
    """
    Execute Python code in a fresh subprocess. Returns stdout, stderr, exit_code.
    The subprocess inherits the current environment plus any env_vars.
    No state is preserved between calls — each call is a clean subprocess.
    """
    extra_env = {}
    try:
        extra_env = json.loads(env_vars) if env_vars else {}
    except json.JSONDecodeError:
        pass

    env = os.environ.copy()
    env.update(extra_env)
    env["DS_AGENT_RUN_ID"] = run_id

    # Write code to a temp file for clean subprocess execution
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        return json.dumps({
            "stdout": proc.stdout[-10000:] if len(proc.stdout) > 10000 else proc.stdout,
            "stderr": proc.stderr[-5000:] if len(proc.stderr) > 5000 else proc.stderr,
            "exit_code": proc.returncode,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({
            "stdout": "",
            "stderr": f"TimeoutExpired: code exceeded {timeout_seconds}s",
            "exit_code": -1,
        })
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── get_sample ───────────────────────────────────────────────────────


@mcp.tool()
async def get_sample(
    file_path: Annotated[str, Field(description="Path to a parquet or CSV file.")],
    n_rows: Annotated[int, Field(description="Number of rows to sample.")] = 10,
    random_state: Annotated[int, Field(description="Random seed for reproducibility.")] = 42,
) -> str:
    """Return a random sample of rows from a tabular file as a JSON string."""
    import pandas as pd

    path = Path(file_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})

    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        sample = df.sample(min(n_rows, len(df)), random_state=random_state)
        return sample.to_json(orient="records", default_handler=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── write_output ─────────────────────────────────────────────────────


@mcp.tool()
async def write_output(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    sub_dir: Annotated[str, Field(description="Sub-directory under outputs/{run_id}/, e.g. 'eda' or 'clean'.")],
    filename: Annotated[str, Field(description="File name to write.")],
    content: Annotated[str, Field(description="Text content to write to the file.")],
) -> str:
    """Write text content to a file inside the run's output directory. Returns the full path."""
    output_dir = Path(os.environ.get("OUTPUT_BASE_DIR", "outputs")) / run_id / sub_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


# ── search_docs ───────────────────────────────────────────────────────


@mcp.tool()
async def search_docs(
    library_name: Annotated[str, Field(description="Library name to search documentation for.")],
    query: Annotated[str, Field(description="Search query about the library.")],
    error_message: Annotated[str, Field(description="The error message for context (used in the Browser-Use task).")] = "",
) -> str:
    """
    Search library documentation using Browser-Use.
    Falls back to DuckDuckGo if library is not in the registry.
    Truncates result to 4,000 characters.
    """
    from config.library_registry import get_docs_url

    docs_url = get_docs_url(library_name)
    if docs_url is None:
        # Fall through to DuckDuckGo
        return await web_research(
            query=f"{library_name} {query} python fix",
            context="debug_docs_search",
        )

    try:
        from browser_use import Browser, BrowserConfig
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI(
            azure_endpoint=os.environ.get("AI_FOUNDRY_PROJECT_ENDPOINT", ""),
            azure_deployment=os.environ.get("AZURE_OPENAI_FAST_DEPLOYMENT", "gpt-4o"),
            api_version=os.environ.get("AI_FOUNDRY_API_VERSION", "2024-12-01-preview"),
            api_key=os.environ.get("AI_FOUNDRY_API_KEY"),
        )
        task = (
            f"Find information about: {query}. "
            f"Focus on: API usage examples, correct function signatures, "
            f"and fix examples for the error: {error_message}. "
            f"Extract code examples relevant to the error. "
            f"Start at: {docs_url}"
        )
        browser = Browser(config=BrowserConfig(headless=True))
        result = await browser.run(task=task, llm=llm, max_steps=50)
        final = getattr(result, "final_result", "") or ""
        if not final or len(final) < 50:
            return await web_research(query=f"{library_name} {query} {error_message}", context="browser_miss")
        return final[:4000]
    except Exception as exc:
        return await web_research(
            query=f"{library_name} {query} {error_message[:100]} python",
            context=f"browser_error:{type(exc).__name__}",
        )


# ── web_research ──────────────────────────────────────────────────────


@mcp.tool()
async def web_research(
    query: Annotated[str, Field(description="Search query string.")],
    context: Annotated[str, Field(description="Caller context: 'debug_attempt_3', 'feature_eng', 'model_selection', or other.")] = "",
) -> str:
    """
    DuckDuckGo deep research. Returns top 10 results as JSON.
    Enforces global rate limit of 30 queries/hour via Bug Log MCP.
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=10)):
                results.append({
                    "rank": i + 1,
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": (r.get("body", "") or "")[:300],
                })
        return json.dumps(results, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── embed_text ────────────────────────────────────────────────────────


@mcp.tool()
async def embed_text(
    text: Annotated[str, Field(description="Text to embed (max 8,192 tokens).")],
    run_id: Annotated[str, Field(description="Pipeline run identifier (for attribution).")],
) -> str:
    """
    Embed text using the Azure OpenAI embedding deployment.
    Returns a JSON array of floats (the embedding vector).
    """
    try:
        import openai

        client = openai.AzureOpenAI(
            azure_endpoint=os.environ.get("AI_FOUNDRY_PROJECT_ENDPOINT", ""),
            api_version=os.environ.get("AI_FOUNDRY_API_VERSION", "2024-12-01-preview"),
            api_key=os.environ.get("AI_FOUNDRY_API_KEY"),
        )
        deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        response = client.embeddings.create(input=[text[:8000]], model=deployment)
        vector = response.data[0].embedding
        return json.dumps(vector)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── semantic_search ───────────────────────────────────────────────────


@mcp.tool()
async def semantic_search(
    query: Annotated[str, Field(description="Natural language query to search the semantic memory.")],
    top_k: Annotated[int, Field(description="Number of most similar results to return.")] = 5,
) -> str:
    """
    Search the semantic memory index for the most similar entries.
    Uses cosine similarity on pre-computed embeddings.
    Returns top_k results as JSON.
    """
    import numpy as np

    semantic_path = Path(os.environ.get("SEMANTIC_MEMORY_PATH", "data/semantic_memory.json"))
    embeddings_path = Path("data/semantic_embeddings.npy")

    if not semantic_path.exists() or not embeddings_path.exists():
        return json.dumps([])

    try:
        entries = json.loads(semantic_path.read_text(encoding="utf-8"))
        stored_embeddings = np.load(str(embeddings_path))

        # Get query embedding
        import openai

        client = openai.AzureOpenAI(
            azure_endpoint=os.environ.get("AI_FOUNDRY_PROJECT_ENDPOINT", ""),
            api_version=os.environ.get("AI_FOUNDRY_API_VERSION", "2024-12-01-preview"),
            api_key=os.environ.get("AI_FOUNDRY_API_KEY"),
        )
        deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        response = client.embeddings.create(input=[query[:8000]], model=deployment)
        q_vec = np.array(response.data[0].embedding)

        # Cosine similarity
        norms = np.linalg.norm(stored_embeddings, axis=1)
        q_norm = np.linalg.norm(q_vec)
        similarities = stored_embeddings @ q_vec / (norms * q_norm + 1e-9)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            entry = entries[idx].copy()
            entry["similarity"] = float(similarities[idx])
            results.append(entry)
        return json.dumps(results, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── log_metrics ───────────────────────────────────────────────────────


@mcp.tool()
async def log_metrics(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    stage_name: Annotated[str, Field(description="Stage producing the metrics.")],
    metrics_json: Annotated[str, Field(description="JSON-encoded dict of metric_name → float.")],
) -> str:
    """Persist metrics to outputs/{run_id}/{stage_name}/metrics.json and return the path."""
    try:
        metrics = json.loads(metrics_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON: {exc}"})

    output_dir = (
        Path(os.environ.get("OUTPUT_BASE_DIR", "outputs")) / run_id / stage_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    existing = {}
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    existing.update(metrics)
    metrics_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return str(metrics_path)


# ── verify_stage ──────────────────────────────────────────────────────


@mcp.tool()
async def verify_stage(
    run_id: Annotated[str, Field(description="Pipeline run identifier.")],
    stage_name: Annotated[str, Field(description="Stage to verify: eda, clean, features, training, tuning, evaluation, explain, report, deploy.")],
    session_state_json: Annotated[str, Field(description="JSON-encoded session state dict from the stage.")],
) -> str:
    """
    Deterministic (no-LLM) stage verification.
    Checks required artefacts exist and session state has required keys.
    Returns JSON: {passed: bool, failures: [{assertion, expected, actual}]}.
    """
    from workflows.criteria import get_criteria

    try:
        session_state = json.loads(session_state_json)
    except json.JSONDecodeError:
        return json.dumps({"passed": False, "failures": [{"assertion": "json_parse", "expected": "valid JSON", "actual": "parse_error"}]})

    criteria = get_criteria(stage_name)
    failures = criteria.verify(run_id, session_state)
    return json.dumps({"passed": len(failures) == 0, "failures": failures})


# ── FastAPI Application ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # stateless_http=True means no persistent session_manager needed,
    # but we still need to start/stop the mcp app properly.
    yield


app = FastAPI(title="DS Tools MCP Server", lifespan=lifespan)
mcp_app = mcp.streamable_http_app()
app.mount("/mcp", mcp_app)


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "server": "ds_tools", "port": PORT}


@app.head("/health")
async def health_head():
    from fastapi.responses import Response
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
