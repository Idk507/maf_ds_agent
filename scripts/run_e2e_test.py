"""
scripts/run_e2e_test.py — End-to-end real-world integration tester.

Runs the full MAF DS Agent pipeline against multiple real datasets:
  1. Titanic (binary classification)
  2. Wine Quality Red (regression)
  3. Student Scores (simple linear regression)

Usage:
    python scripts/run_e2e_test.py
    python scripts/run_e2e_test.py --datasets titanic wine
    python scripts/run_e2e_test.py --datasets titanic --verbose

Requirements:
  - MCP servers running (python scripts/start_servers.py)
  - .env with real Azure OpenAI credentials
  - data/titanic.csv, data/winequality_red.csv, data/student_scores.csv

Harness Engineering:
  - Each run uses a unique run_id with dataset prefix
  - Results are stored in outputs/{dataset}/
  - A summary report is printed after all runs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── Dataset registry ─────────────────────────────────────────────────

DATASETS: dict[str, dict[str, str]] = {
    "titanic": {
        "file": "data/titanic.csv",
        "task": (
            "Build a binary classifier to predict passenger survival on the Titanic dataset. "
            "The target column is 'Survived' (1 = survived, 0 = did not survive). "
            "Use features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked. "
            "Drop columns: PassengerId, Name, Ticket, Cabin. "
            "Handle missing Age values by median imputation. "
            "Encode Sex and Embarked as one-hot or label encoded. "
            "Train a Random Forest classifier. Evaluate using ROC-AUC and accuracy. "
            "The dataset has 891 rows and 12 columns."
        ),
        "variant": "tabular",
    },
    "wine": {
        "file": "data/winequality_red.csv",
        "task": (
            "Build a regression model to predict red wine quality scores from physicochemical properties. "
            "The target column is 'quality' (integer 3-8). "
            "All other columns are numeric features: fixed acidity, volatile acidity, citric acid, "
            "residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, "
            "sulphates, alcohol. "
            "Train a Gradient Boosting Regressor. Evaluate using RMSE and R-squared. "
            "The dataset has 1599 rows and 12 columns with a semicolon separator."
        ),
        "variant": "tabular",
    },
    "student": {
        "file": "data/student_scores.csv",
        "task": (
            "Build a simple linear regression model to predict student exam scores based on study hours. "
            "The target column is 'Scores' and the single feature is 'Hours'. "
            "Evaluate using Mean Squared Error and R-squared. "
            "The dataset has 25 rows and 2 columns."
        ),
        "variant": "tabular",
    },
}


# ── Health check ─────────────────────────────────────────────────────

async def _check_mcp_servers() -> bool:
    """Return True if all three MCP servers are healthy."""
    import httpx

    urls = [
        "http://localhost:8100/health",
        "http://localhost:8101/health",
        "http://localhost:8102/health",
    ]
    async with httpx.AsyncClient(timeout=5.0) as client:
        for url in urls:
            try:
                resp = await client.get(url)
                if resp.status_code not in (200, 204):
                    print(f"[WARN] MCP server unhealthy: {url} → {resp.status_code}")
                    return False
            except Exception as exc:
                print(f"[ERROR] MCP server unreachable: {url} → {exc}")
                return False
    return True


# ── Single pipeline run ───────────────────────────────────────────────

async def _run_single(
    dataset_name: str,
    config: dict[str, str],
    run_id: str,
    output_dir: Path,
    verbose: bool,
) -> dict[str, Any]:
    """Run the full pipeline for one dataset. Returns a result summary dict."""
    from agents.orchestrator import build_orchestrator

    print(f"\n{'='*70}")
    print(f"  DATASET : {dataset_name.upper()}")
    print(f"  FILE    : {config['file']}")
    print(f"  RUN ID  : {run_id}")
    print(f"  OUTPUT  : {output_dir}")
    print(f"{'='*70}")

    file_path = Path(config["file"])
    if not file_path.exists():
        return {
            "dataset": dataset_name,
            "run_id": run_id,
            "status": "skipped",
            "reason": f"File not found: {file_path}",
            "duration_s": 0,
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    status = "unknown"
    result: dict[str, Any] = {}

    try:
        orchestrator = build_orchestrator()
        result = await orchestrator.run_pipeline(
            task_description=config["task"],
            file_path=str(file_path.resolve()),
            run_id=run_id,
        )
        status = result.get("status", "success")
        print(f"\n[OK] Pipeline completed — status: {status}")

        if verbose:
            print(json.dumps(result, indent=2, default=str))

        # Save full result as JSON
        out_json = output_dir / f"{run_id}_result.json"
        out_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"[OK] Result saved → {out_json}")

    except KeyboardInterrupt:
        status = "interrupted"
        print("[WARN] Pipeline interrupted by user")
    except Exception as exc:
        status = "error"
        print(f"[ERROR] Pipeline failed: {exc}")
        if verbose:
            import traceback
            traceback.print_exc()

    duration = time.monotonic() - start
    return {
        "dataset": dataset_name,
        "run_id": run_id,
        "status": status,
        "duration_s": round(duration, 1),
        "stages_completed": result.get("stages_completed", []),
        "stages_failed": result.get("stages_failed", []),
        "artefacts": result.get("artefacts", []),
        "metrics": result.get("metrics", {}),
    }


# ── Main ──────────────────────────────────────────────────────────────

async def _main(datasets: list[str], verbose: bool) -> int:
    print("\n" + "=" * 70)
    print("  MAF DS Agent — Real-World End-to-End Test Runner")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Health check
    print("\n[1/3] Checking MCP server health ...")
    healthy = await _check_mcp_servers()
    if not healthy:
        print(
            "\n[ERROR] One or more MCP servers are not running.\n"
            "Start them first:\n"
            "    python scripts/start_servers.py\n"
        )
        return 1
    print("[OK] All 3 MCP servers are healthy (ports 8100/8101/8102)")

    # Filter datasets
    requested = {k: DATASETS[k] for k in datasets if k in DATASETS}
    missing = [k for k in datasets if k not in DATASETS]
    if missing:
        print(f"[WARN] Unknown datasets (skipped): {missing}")
    if not requested:
        print("[ERROR] No valid datasets to run.")
        return 1

    print(f"\n[2/3] Running pipelines for: {list(requested.keys())}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries: list[dict[str, Any]] = []

    for name, config in requested.items():
        run_id = f"{name}-{timestamp}"
        output_dir = Path("outputs") / name / run_id
        summary = await _run_single(name, config, run_id, output_dir, verbose)
        summaries.append(summary)

    # Print summary table
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'DATASET':<15} {'STATUS':<12} {'DURATION':>10}  {'STAGES OK':<12} {'STAGES FAIL'}")
    print(f"  {'-'*60}")
    all_ok = True
    for s in summaries:
        ok_count = len(s.get("stages_completed", []))
        fail_count = len(s.get("stages_failed", []))
        status = s["status"]
        if status not in ("success",):
            all_ok = False
        print(
            f"  {s['dataset']:<15} {status:<12} {s['duration_s']:>9.1f}s  "
            f"{ok_count:<12} {fail_count}"
        )
    print(f"{'='*70}\n")

    # Save overall summary
    summary_path = Path("outputs") / f"e2e_summary_{timestamp}.json"
    summary_path.parent.mkdir(exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    print(f"[3/3] Full summary saved → {summary_path}")

    return 0 if all_ok else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="MAF DS Agent — E2E test runner")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["titanic"],
        choices=list(DATASETS.keys()),
        help="Which datasets to test (default: titanic)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full result JSON")
    args = parser.parse_args()

    exit_code = asyncio.run(_main(args.datasets, args.verbose))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
