"""
scripts/start_servers.py — Launch all three MCP servers as concurrent subprocesses.

Usage:
    python scripts/start_servers.py            # start all three
    python scripts/start_servers.py tracking   # start one server by name
    python scripts/start_servers.py --check    # health-check only (no start)

Harness Engineering (Canary gate):
    After starting, the script performs health checks on all servers with
    exponential back-off. If any server fails to become healthy within 30 s,
    the whole process exits non-zero — acting as a pipeline gate.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

SERVERS = {
    "tracking": {
        "module": "mcp_servers.tracking.server",
        "port": int(os.environ.get("TRACKING_MCP_PORT", "8100")),
        "health_url": "http://localhost:{port}/health",
    },
    "ds_tools": {
        "module": "mcp_servers.ds_tools.server",
        "port": int(os.environ.get("DS_TOOLS_MCP_PORT", "8101")),
        "health_url": "http://localhost:{port}/health",
    },
    "bug_log": {
        "module": "mcp_servers.bug_log.server",
        "port": int(os.environ.get("BUG_LOG_MCP_PORT", "8102")),
        "health_url": "http://localhost:{port}/health",
    },
}


def _build_cmd(server_name: str, cfg: dict) -> list[str]:
    """Build the uvicorn launch command for a given server."""
    return [
        sys.executable,
        "-m",
        "uvicorn",
        f"{cfg['module']}:app",
        "--host", "0.0.0.0",
        "--port", str(cfg["port"]),
        "--log-level", "info",
        "--no-access-log",
    ]


async def _wait_healthy(name: str, port: int, timeout: float = 30.0) -> bool:
    """Poll /health with exponential back-off until healthy or timeout."""
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    delay = 0.5
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    print(f"  ✓ {name} healthy on port {port}")
                    return True
            except Exception:
                pass
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 5.0)
    print(f"  ✗ {name} failed to become healthy within {timeout}s", file=sys.stderr)
    return False


async def _check_only(names: list[str]) -> bool:
    """Health-check named servers without starting anything."""
    tasks = [
        _wait_healthy(name, SERVERS[name]["port"], timeout=5.0)
        for name in names
    ]
    results = await asyncio.gather(*tasks)
    return all(results)


async def _start_servers(names: list[str]) -> None:
    """Start named servers as subprocesses and wait until all healthy."""
    procs: list[subprocess.Popen] = []
    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    print(f"Starting MCP servers: {', '.join(names)}")
    for name in names:
        cfg = SERVERS[name]
        cmd = _build_cmd(name, cfg)
        print(f"  → {name} on port {cfg['port']}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        procs.append(proc)

    # Health check all servers concurrently
    tasks = [
        _wait_healthy(name, SERVERS[name]["port"])
        for name in names
    ]
    results = await asyncio.gather(*tasks)

    if not all(results):
        print("\n[FATAL] One or more MCP servers failed the health gate.", file=sys.stderr)
        for proc in procs:
            proc.terminate()
        sys.exit(1)

    print("\nAll MCP servers are healthy. Press Ctrl+C to stop.\n")

    # Forward SIGINT/SIGTERM to all children
    def _stop(sig, frame):
        print("\nShutting down MCP servers...")
        for proc in procs:
            proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Wait for all subprocesses
    for proc in procs:
        proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start MAF DS Agent MCP servers")
    parser.add_argument(
        "servers",
        nargs="*",
        default=None,
        help=f"Which servers to start (choices: {', '.join(SERVERS.keys())}; default: all)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Health-check running servers without starting new ones",
    )
    args = parser.parse_args()

    names = [s for s in (args.servers or list(SERVERS.keys())) if s in SERVERS]
    invalid = [s for s in (args.servers or []) if s not in SERVERS]
    if invalid:
        parser.error(f"invalid server(s): {invalid}. Choose from: {list(SERVERS.keys())}")

    if args.check:
        ok = asyncio.run(_check_only(names))
        sys.exit(0 if ok else 1)
    else:
        asyncio.run(_start_servers(names))


if __name__ == "__main__":
    main()
