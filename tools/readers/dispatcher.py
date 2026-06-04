"""
tools/readers/dispatcher.py — Route file reads to the correct reader.

Delegates to:
  tabular       → readers.tabular.read_tabular
  document_text → readers.document.read_document
  image         → readers.image.read_image
  existing_model → readers.model.read_model
  unknown       → basic metadata only
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def read_file_dispatch(
    file_path: str,
    detection,  # FileTypeResult
    sample_rows: int = 5,
) -> dict[str, Any]:
    """
    Dispatch file reading based on detected category.
    Returns a dict with: file_type, mime_type, confidence, source, schema, sample, metadata.
    """
    from tools.file_type_detector import FileTypeResult

    base = {
        "full_path": str(Path(file_path).resolve()),
        "file_type": detection.category,
        "mime_type": detection.mime_type,
        "confidence": detection.confidence,
        "detection_source": detection.source,
        "schema": None,
        "sample": None,
        "metadata": {},
    }

    try:
        if detection.category == "tabular":
            from tools.readers.tabular import read_tabular

            return {**base, **read_tabular(file_path, sample_rows=sample_rows)}

        if detection.category == "document_text":
            from tools.readers.document import read_document

            return {**base, **read_document(file_path)}

        if detection.category == "image":
            from tools.readers.image import read_image

            return {**base, **read_image(file_path)}

        if detection.category == "existing_model":
            from tools.readers.model import read_model

            return {**base, **read_model(file_path)}

        # Unknown — return only size and path
        path = Path(file_path)
        base["metadata"] = {"size_bytes": path.stat().st_size if path.exists() else 0}
        return base

    except Exception as exc:
        base["error"] = str(exc)
        return base
