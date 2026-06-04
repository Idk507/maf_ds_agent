"""
agents/clients.py — Azure OpenAI client singletons.

Two clients only:
  PRIMARY_CLIENT — gpt-4o for complex reasoning agents
  FAST_CLIENT    — gpt-4o fast track for data/stat agents

Auth:
  Dev:  AI_FOUNDRY_API_KEY env var (key-based)
  Prod: DefaultAzureCredential (managed identity, no key)
"""
from __future__ import annotations

import os

from agent_framework.openai import AzureOpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


def _make_client(deployment_env_key: str) -> AzureOpenAIChatCompletionClient:
    """Construct an AzureOpenAIChatCompletionClient for the given deployment."""
    endpoint = os.environ["AI_FOUNDRY_PROJECT_ENDPOINT"]
    deployment = os.environ[deployment_env_key]
    api_version = os.environ.get("AI_FOUNDRY_API_VERSION", "2024-12-01-preview")
    api_key = os.environ.get("AI_FOUNDRY_API_KEY")

    if api_key:
        return AzureOpenAIChatCompletionClient(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
            api_key=api_key,
        )

    # Production: use DefaultAzureCredential (managed identity)
    from azure.identity import DefaultAzureCredential

    return AzureOpenAIChatCompletionClient(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        credential=DefaultAzureCredential(),
    )


# Singletons — module-level, evaluated once at import time.
PRIMARY_CLIENT: AzureOpenAIChatCompletionClient = _make_client("AZURE_OPENAI_PRIMARY_DEPLOYMENT")
FAST_CLIENT: AzureOpenAIChatCompletionClient = _make_client("AZURE_OPENAI_FAST_DEPLOYMENT")

# Agent → client assignment reference (for documentation purposes):
#
#   PRIMARY_CLIENT:
#     Orchestrator, Debug, Cleaning, Model Selection,
#     Evaluation, Explainability, Report, Bug Log Observer
#
#   FAST_CLIENT:
#     Ingestion, EDA (stat prompts), Feature Engineering,
#     HP Tuning, Memory, Artefact Tracking Observer
