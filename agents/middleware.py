"""
agents/middleware.py — AgentMiddleware stack for the DS Agent pipeline.

Five middleware classes:
  LoggingMiddleware    — Structured log of every agent invocation (always outermost)
  RateLimitMiddleware  — Token-bucket per agent type to prevent LLM cost spikes
  SafetyMiddleware     — Block unsafe code patterns before execution
  RetryMiddleware      — Retry transient Azure/MCP failures with exponential back-off
  TelemetryMiddleware  — OpenTelemetry span creation per agent call (innermost)

Pipeline agents:    Logging → RateLimit → Safety → Retry → Telemetry
Observer agents:    Logging → Retry → Telemetry
Debug Agent:        Logging → Retry → Telemetry  (NO safety — must inspect bad code)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, Callable, Awaitable

from agent_framework import AgentMiddleware, AgentRunContext

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────

_UNSAFE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bos\.system\b"),
    re.compile(r"\bsubprocess\.Popen\b.*shell\s*=\s*True"),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"__import__\s*\(\s*['\"]os['\"]"),
    re.compile(r"shutil\.rmtree"),
    re.compile(r"\brm\s+-rf\b"),
    re.compile(r"DROP\s+TABLE", re.IGNORECASE),
    re.compile(r"DELETE\s+FROM\s+\w+\s*;", re.IGNORECASE),
]


def _extract_code_blocks(text: str) -> list[str]:
    """Return all ```python ... ``` blocks from a text string."""
    return re.findall(r"```python\s*(.*?)```", text, re.DOTALL)


def _has_unsafe_patterns(text: str) -> tuple[bool, str]:
    """Check code text for unsafe patterns. Returns (is_unsafe, matched_pattern)."""
    for pat in _UNSAFE_PATTERNS:
        if pat.search(text):
            return True, pat.pattern
    return False, ""


# ── LoggingMiddleware ─────────────────────────────────────────────────


class LoggingMiddleware(AgentMiddleware):
    """
    Structured logging at the agent invocation boundary.
    Logs agent name, run_id, token counts (if available), and latency.
    """

    async def process(
        self,
        context: AgentRunContext,
        next_middleware: Callable[[AgentRunContext], Awaitable[Any]],
    ) -> Any:
        agent_name = getattr(context, "agent_name", "unknown")
        run_id = (context.kwargs or {}).get("run_id", "none")
        start = time.monotonic()
        logger.info(
            "agent.start",
            extra={"agent": agent_name, "run_id": run_id},
        )
        try:
            result = await next_middleware(context)
            elapsed = time.monotonic() - start
            logger.info(
                "agent.done",
                extra={"agent": agent_name, "run_id": run_id, "elapsed_s": round(elapsed, 2)},
            )
            return result
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "agent.error",
                extra={"agent": agent_name, "run_id": run_id, "elapsed_s": round(elapsed, 2), "error": str(exc)},
            )
            raise


# ── RateLimitMiddleware ───────────────────────────────────────────────


class RateLimitMiddleware(AgentMiddleware):
    """
    Token-bucket rate limiter.
    Limits each agent type to `calls_per_minute` concurrent/rapid calls.
    Defaults: pipeline agents → 10 RPM; can be overridden via env.
    """

    def __init__(self, calls_per_minute: int = 10) -> None:
        self._cpm = calls_per_minute
        self._window: dict[str, list[float]] = defaultdict(list)

    async def process(
        self,
        context: AgentRunContext,
        next_middleware: Callable[[AgentRunContext], Awaitable[Any]],
    ) -> Any:
        agent_name = getattr(context, "agent_name", "unknown")
        now = time.monotonic()
        window = self._window[agent_name]
        # Evict timestamps older than 60 s
        self._window[agent_name] = [t for t in window if now - t < 60]
        if len(self._window[agent_name]) >= self._cpm:
            sleep_for = 60 - (now - self._window[agent_name][0]) + 0.1
            logger.warning(
                "rate_limit.sleeping",
                extra={"agent": agent_name, "sleep_s": round(sleep_for, 1)},
            )
            await asyncio.sleep(max(0, sleep_for))
        self._window[agent_name].append(time.monotonic())
        return await next_middleware(context)


# ── SafetyMiddleware ──────────────────────────────────────────────────


class SafetyMiddleware(AgentMiddleware):
    """
    Pre-execution safety gate.
    Scans the outgoing message for unsafe Python patterns in code blocks.
    Raises ValueError (which the Ralph Loop treats as a failed attempt)
    if a pattern is matched.

    NOT applied to the Debug Agent — it must be able to inspect bad code.
    """

    async def process(
        self,
        context: AgentRunContext,
        next_middleware: Callable[[AgentRunContext], Awaitable[Any]],
    ) -> Any:
        # Inspect the last user message if present
        messages = getattr(context, "messages", []) or []
        for msg in messages[-2:]:
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str):
                for block in _extract_code_blocks(content):
                    is_unsafe, pattern = _has_unsafe_patterns(block)
                    if is_unsafe:
                        raise ValueError(
                            f"SafetyMiddleware: blocked unsafe pattern '{pattern}' in generated code."
                        )
        return await next_middleware(context)


# ── RetryMiddleware ───────────────────────────────────────────────────

_RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    OSError,
)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RetryMiddleware(AgentMiddleware):
    """
    Exponential back-off retry for transient Azure / MCP failures.
    Retries up to `max_retries` times with jitter.
    Gives up immediately for ValueError (logic errors, not transient).
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0) -> None:
        self._max_retries = max_retries
        self._base_delay = base_delay

    async def process(
        self,
        context: AgentRunContext,
        next_middleware: Callable[[AgentRunContext], Awaitable[Any]],
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await next_middleware(context)
            except ValueError:
                raise  # Logic errors are not retried
            except Exception as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    break
                delay = self._base_delay * (2 ** attempt) + (time.monotonic() % 0.5)
                logger.warning(
                    "retry.sleeping",
                    extra={"attempt": attempt + 1, "delay_s": round(delay, 2), "error": str(exc)},
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]


# ── TelemetryMiddleware ───────────────────────────────────────────────


class TelemetryMiddleware(AgentMiddleware):
    """
    OpenTelemetry span per agent invocation.
    Falls back gracefully if opentelemetry is not installed or OTEL_EXPORTER_OTLP_ENDPOINT is unset.
    """

    async def process(
        self,
        context: AgentRunContext,
        next_middleware: Callable[[AgentRunContext], Awaitable[Any]],
    ) -> Any:
        agent_name = getattr(context, "agent_name", "unknown")
        run_id = (context.kwargs or {}).get("run_id", "none")

        try:
            from opentelemetry import trace

            tracer = trace.get_tracer("ds_agent")
            with tracer.start_as_current_span(
                name=f"agent.{agent_name}",
                attributes={"run_id": run_id, "agent": agent_name},
            ):
                return await next_middleware(context)
        except ImportError:
            return await next_middleware(context)
        except Exception:
            # Never let telemetry break the pipeline
            return await next_middleware(context)


# ── Middleware stack factories ─────────────────────────────────────────


def pipeline_stack() -> list[AgentMiddleware]:
    """Middleware for pipeline agents: Logging → RateLimit → Safety → Retry → Telemetry."""
    return [
        LoggingMiddleware(),
        RateLimitMiddleware(calls_per_minute=int(os.environ.get("AGENT_RATE_LIMIT_RPM", "10"))),
        SafetyMiddleware(),
        RetryMiddleware(max_retries=3),
        TelemetryMiddleware(),
    ]


def observer_stack() -> list[AgentMiddleware]:
    """Middleware for observer agents: Logging → Retry → Telemetry."""
    return [
        LoggingMiddleware(),
        RetryMiddleware(max_retries=3),
        TelemetryMiddleware(),
    ]


def debug_stack() -> list[AgentMiddleware]:
    """Middleware for Debug Agent: Logging → Retry → Telemetry. NO safety check."""
    return [
        LoggingMiddleware(),
        RetryMiddleware(max_retries=3),
        TelemetryMiddleware(),
    ]
