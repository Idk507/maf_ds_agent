"""
agents/clients.py — Azure OpenAI client singletons.

Two clients only:
  PRIMARY_CLIENT — gpt-4o for complex reasoning agents
  FAST_CLIENT    — gpt-4o fast track for data/stat agents

Auth:
  Dev:  AI_FOUNDRY_API_KEY env var (key-based)
  Prod: DefaultAzureCredential (managed identity, no key)

Clients are lazily initialised on first access to allow importing the module
without Azure credentials (e.g. during tests or offline development).
"""
from __future__ import annotations

import functools
import os

from agent_framework.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


def _resolve_deployment(primary_key: str) -> str:
    """Resolve deployment name from primary env var or fall back to AI_FOUNDRY_DEPLOYMENT_NAME."""
    return (
        os.environ.get(primary_key)
        or os.environ.get("AI_FOUNDRY_DEPLOYMENT_NAME")
        or "gpt-4o"
    )


def _make_client(deployment_env_key: str) -> OpenAIChatCompletionClient:
    """Construct an OpenAIChatCompletionClient (Azure routing) for the given deployment.

    Supports both legacy Azure OpenAI (*.openai.azure.com) and Azure AI Foundry
    (*.services.ai.azure.com) endpoint formats.
    """
    endpoint = os.environ["AI_FOUNDRY_PROJECT_ENDPOINT"].strip().rstrip("/")
    deployment = _resolve_deployment(deployment_env_key)
    api_version = os.environ.get("AI_FOUNDRY_API_VERSION", "2024-12-01-preview").strip()
    api_key = os.environ.get("AI_FOUNDRY_API_KEY", "").strip() or None

    # Azure AI Foundry (services.ai.azure.com) exposes the OpenAI-compatible API
    # under the same /openai/deployments/… path as classic Azure OpenAI.
    if api_key:
        return OpenAIChatCompletionClient(
            model=deployment,
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
        )

    # Production: use DefaultAzureCredential (managed identity)
    from azure.identity import DefaultAzureCredential

    return OpenAIChatCompletionClient(
        model=deployment,
        azure_endpoint=endpoint,
        api_version=api_version,
        credential=DefaultAzureCredential(),
    )


@functools.lru_cache(maxsize=1)
def _get_primary_client() -> OpenAIChatCompletionClient:
    return _make_client("AZURE_OPENAI_PRIMARY_DEPLOYMENT")


@functools.lru_cache(maxsize=1)
def _get_fast_client() -> OpenAIChatCompletionClient:
    return _make_client("AZURE_OPENAI_FAST_DEPLOYMENT")


class _LazyClient:
    """Lazy proxy: instantiates the underlying client only when first attribute is accessed."""

    def __init__(self, factory) -> None:
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_client", None)

    def _resolve(self) -> OpenAIChatCompletionClient:
        client = object.__getattribute__(self, "_client")
        if client is None:
            factory = object.__getattribute__(self, "_factory")
            client = factory()
            object.__setattr__(self, "_client", client)
        return client

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)

    def __repr__(self) -> str:
        return f"<LazyClient wrapping {object.__getattribute__(self, '_factory').__name__}>"


# Module-level singletons — lazily resolved on first access.
PRIMARY_CLIENT: OpenAIChatCompletionClient = _LazyClient(_get_primary_client)  # type: ignore[assignment]
FAST_CLIENT: OpenAIChatCompletionClient = _LazyClient(_get_fast_client)  # type: ignore[assignment]

# Agent → client assignment reference (for documentation purposes):
#
#   PRIMARY_CLIENT:
#     Orchestrator, Debug, Cleaning, Model Selection,
#     Evaluation, Explainability, Report, Bug Log Observer
#
#   FAST_CLIENT:
#     Ingestion, EDA (stat prompts), Feature Engineering,
#     HP Tuning, Memory, Artefact Tracking Observer
