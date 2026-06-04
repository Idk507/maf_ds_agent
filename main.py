"""
main.py — Entry point for the MAF DS Agent pipeline.

Usage:
    python main.py --file /path/to/data.csv --task "Predict customer churn"
    python main.py --file /path/to/model.pkl --task "Evaluate and report on this model"

Environment:
    Copy .env.example to .env and fill in Azure OpenAI credentials before running.
    MCP servers must be started separately (see README §7).

Harness Engineering:
    - Pipeline is gated at each stage (Ralph Loop exit criteria)
    - Debug Agent is invoked automatically on stage failure
    - PIPELINE_HUMAN_IN_THE_LOOP=true enables interactive gates between stages
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def _configure_file_logging() -> None:
    """Add file handler once logs/ directory exists."""
    Path("logs").mkdir(exist_ok=True)
    fh = logging.FileHandler("logs/pipeline.log", mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    logging.getLogger().addHandler(fh)


async def _run(task: str, file_path: str, run_id: str | None) -> dict:
    """Import and run the orchestrator pipeline."""
    from agents.orchestrator import build_orchestrator

    orchestrator = build_orchestrator()
    result = await orchestrator.run_pipeline(
        task_description=task,
        file_path=file_path,
        run_id=run_id,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MAF DS Agent — Automated ML Pipeline powered by Microsoft Agent Framework"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the input dataset or model file.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help='Natural language description of the ML task (e.g. "Predict customer churn").',
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit pipeline run ID (UUID). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the final session_state JSON. Defaults to stdout.",
    )
    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.file).exists():
        logger.error("Input file not found: %s", args.file)
        sys.exit(1)

    # Ensure logs directory exists and configure file logging
    Path("data/artefacts").mkdir(parents=True, exist_ok=True)
    _configure_file_logging()

    logger.info("Starting MAF DS Agent pipeline")
    logger.info("  Task: %s", args.task)
    logger.info("  File: %s", args.file)

    # Run the async pipeline
    result = asyncio.run(_run(args.task, args.file, args.run_id))

    # Output result
    result_json = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(result_json, encoding="utf-8")
        logger.info("Session state written to %s", args.output)
    else:
        print("\n=== PIPELINE COMPLETE ===")
        print(result_json)

    # Print summary
    print("\n=== PIPELINE SUMMARY ===")
    print(f"  Run ID              : {result.get('run_id', 'N/A')}")
    print(f"  Pipeline Variant    : {result.get('pipeline_variant', 'N/A')}")
    print(f"  Model Type          : {result.get('model_type', 'N/A')}")
    print(f"  Deployment Status   : {result.get('deployment_recommendation', 'N/A')}")
    print(f"  Report              : {result.get('report_html_path', 'N/A')}")
    print(f"  Endpoint URL        : {result.get('endpoint_url', 'N/A')}")
    baseline = result.get("baseline_metrics", {})
    if baseline:
        print("  Metrics:")
        for k, v in baseline.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
