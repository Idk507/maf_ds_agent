"""
tools/file_type_detector.py — 3-layer file type detection.

Layer 1: Magika (Google ML-based, confidence threshold 0.85)
Layer 2: filetype (magic bytes) when Magika confidence < 0.85
Layer 3: Content heuristics for ambiguous text files

Returns a FileTypeResult dataclass with:
  category: str   — one of: tabular, document_text, image, existing_model, unknown
  mime_type: str  — e.g. text/csv, application/pdf
  confidence: float
  source: str     — "magika", "filetype", or "heuristics"
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ── Module-level Magika singleton ─────────────────────────────────────
# Instantiated once to avoid repeated model loading (~500 ms each).
_MAGIKA_INSTANCE: "Magika | None" = None  # type: ignore[name-defined]


def _get_magika() -> "Magika":  # type: ignore[name-defined]
    global _MAGIKA_INSTANCE
    if _MAGIKA_INSTANCE is None:
        from magika import Magika

        _MAGIKA_INSTANCE = Magika()
    return _MAGIKA_INSTANCE


# ── Category mapping ─────────────────────────────────────────────────

_TABULAR_EXTENSIONS = {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".json", ".jsonl", ".feather"}
_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
_MODEL_EXTENSIONS = {".onnx", ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".h5", ".keras", ".safetensors"}

_MAGIKA_LABEL_TO_CATEGORY: dict[str, str] = {
    "csv": "tabular",
    "tsv": "tabular",
    "json": "tabular",
    "jsonl": "tabular",
    "parquet": "tabular",
    "xlsx": "tabular",
    "xls": "tabular",
    "pdf": "document_text",
    "docx": "document_text",
    "txt": "document_text",
    "markdown": "document_text",
    "html": "document_text",
    "htm": "document_text",
    "jpeg": "image",
    "jpg": "image",
    "png": "image",
    "gif": "image",
    "bmp": "image",
    "tiff": "image",
    "webp": "image",
    "onnx": "existing_model",
    "pickle": "existing_model",
    "pkl": "existing_model",
    "pt": "existing_model",
    "pth": "existing_model",
    "h5": "existing_model",
}

_MIME_TO_CATEGORY: dict[str, str] = {
    "text/csv": "tabular",
    "text/tab-separated-values": "tabular",
    "application/json": "tabular",
    "application/vnd.ms-excel": "tabular",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "tabular",
    "application/pdf": "document_text",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document_text",
    "text/plain": "document_text",
    "text/html": "document_text",
    "image/jpeg": "image",
    "image/png": "image",
    "image/gif": "image",
    "image/bmp": "image",
    "image/tiff": "image",
    "image/webp": "image",
}


# ── Result type ───────────────────────────────────────────────────────


@dataclass
class FileTypeResult:
    category: str       # tabular | document_text | image | existing_model | unknown
    mime_type: str
    confidence: float
    source: str         # magika | filetype | heuristics | extension


# ── Heuristics (Layer 3) ─────────────────────────────────────────────


def _heuristic_detect(path: Path) -> FileTypeResult:
    ext = path.suffix.lower()
    if ext in _TABULAR_EXTENSIONS:
        return FileTypeResult("tabular", f"application/{ext.lstrip('.')}", 0.7, "extension")
    if ext in _DOCUMENT_EXTENSIONS:
        return FileTypeResult("document_text", f"text/{ext.lstrip('.')}", 0.7, "extension")
    if ext in _IMAGE_EXTENSIONS:
        return FileTypeResult("image", f"image/{ext.lstrip('.')}", 0.7, "extension")
    if ext in _MODEL_EXTENSIONS:
        return FileTypeResult("existing_model", f"application/{ext.lstrip('.')}", 0.7, "extension")

    # Try to read first 2 KB and look for structural indicators
    try:
        snippet = path.read_bytes()[:2048]
        text = snippet.decode("utf-8", errors="replace")
        if text.count(",") > 10 and "\n" in text:
            return FileTypeResult("tabular", "text/csv", 0.6, "heuristics")
        if text.strip().startswith("{") or text.strip().startswith("["):
            return FileTypeResult("tabular", "application/json", 0.65, "heuristics")
        if snippet[:4] in (b"\x50\x4b\x03\x04", b"\x89PNG", b"\xff\xd8\xff"):
            return FileTypeResult("image", "image/unknown", 0.7, "heuristics")
        if b"%PDF" in snippet[:1024]:
            return FileTypeResult("document_text", "application/pdf", 0.85, "heuristics")
    except OSError:
        pass
    return FileTypeResult("unknown", "application/octet-stream", 0.0, "heuristics")


# ── Main detection function ───────────────────────────────────────────


MAGIKA_CONFIDENCE_THRESHOLD = 0.85


def detect_file_type(file_path: str) -> FileTypeResult:
    """
    Run 3-layer detection and return a FileTypeResult.

    Layer 1: Magika ML (if confidence >= 0.85)
    Layer 2: filetype magic bytes (if Magika confidence < threshold)
    Layer 3: Extension/content heuristics (fallback)
    """
    path = Path(file_path)

    # ── Layer 1: Magika ───────────────────────────────────────────────
    try:
        magika = _get_magika()
        result = magika.identify_path(path)
        label = result.output.ct_label
        confidence = float(result.output.score)
        mime_type = result.output.mime_type or ""
        if confidence >= MAGIKA_CONFIDENCE_THRESHOLD:
            category = _MAGIKA_LABEL_TO_CATEGORY.get(label, "unknown")
            if category == "unknown":
                category = _MIME_TO_CATEGORY.get(mime_type, "unknown")
            return FileTypeResult(category, mime_type, confidence, "magika")
        # Confidence too low — fall through to Layer 2
    except Exception:
        pass

    # ── Layer 2: filetype (magic bytes) ──────────────────────────────
    try:
        import filetype as ft

        kind = ft.guess(str(path))
        if kind is not None:
            mime_type = kind.mime
            category = _MIME_TO_CATEGORY.get(mime_type, "unknown")
            if category == "unknown" and mime_type.startswith("image/"):
                category = "image"
            return FileTypeResult(category, mime_type, 0.80, "filetype")
    except Exception:
        pass

    # ── Layer 3: Heuristics ───────────────────────────────────────────
    return _heuristic_detect(path)
