"""
workflows/pipeline_graph.py — Static pipeline stage graphs per file type.

The pipeline variant is selected based on the detected file type from
tools/file_type_detector.py. Each variant defines the ordered list of
stage names that the orchestrator will execute.

Stage names must match keys in workflows/criteria.py get_criteria() registry.

Variants:
  tabular        — Full ML pipeline for structured data (CSV, Parquet, etc.)
  document_text  — NLP pipeline (classification, summarisation, RAG)
  image          — Vision pipeline (CNN, ViT, CLIP)
  existing_model — Evaluation-only for pre-trained model files (.pkl, .h5, etc.)
"""
from __future__ import annotations

from typing import Literal

# ── File type → ordered stage names ──────────────────────────────────

PIPELINE_GRAPH: dict[str, list[str]] = {
    "tabular": [
        "ingestion",
        "eda",
        "cleaning",
        "feature_engineering",
        "training",
        "tuning",
        "evaluation",
        "explainability",
        "report",
        "deployment",
    ],
    "document_text": [
        "ingestion",
        "eda",
        "cleaning",
        "feature_engineering",
        "training",
        "tuning",
        "evaluation",
        "explainability",
        "report",
        "deployment",
    ],
    "image": [
        "ingestion",
        "feature_engineering",
        "training",
        "tuning",
        "evaluation",
        "explainability",
        "report",
        "deployment",
    ],
    "existing_model": [
        "ingestion",
        "evaluation",
        "explainability",
        "report",
        "deployment",
    ],
}

# Canonical set of all stage names used across any variant
ALL_STAGES: frozenset[str] = frozenset(
    stage for stages in PIPELINE_GRAPH.values() for stage in stages
)

PipelineVariant = Literal["tabular", "document_text", "image", "existing_model"]


def get_pipeline_stages(variant: str) -> list[str]:
    """Return the ordered list of stage names for the given pipeline variant.

    Args:
        variant: One of 'tabular', 'document_text', 'image', 'existing_model'.

    Returns:
        Ordered list of stage name strings.

    Raises:
        KeyError: If variant is not recognised.
    """
    if variant not in PIPELINE_GRAPH:
        valid = ", ".join(sorted(PIPELINE_GRAPH.keys()))
        raise KeyError(
            f"Unknown pipeline variant '{variant}'. Valid options: {valid}"
        )
    return list(PIPELINE_GRAPH[variant])


def select_variant(file_type_category: str) -> str:
    """Map a FileTypeResult category to a pipeline variant name.

    Args:
        file_type_category: Category string from FileTypeResult
                            ('tabular', 'document_text', 'image', 'existing_model').

    Returns:
        Pipeline variant string (same as category for direct match).
    """
    if file_type_category in PIPELINE_GRAPH:
        return file_type_category
    # Fallback: unknown types go through the full tabular pipeline
    return "tabular"
