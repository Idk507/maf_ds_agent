"""tools/readers/image.py — Read image metadata and optional thumbnail."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def read_image(file_path: str) -> dict[str, Any]:
    """Return image dimensions, mode, and base64-encoded 128x128 thumbnail."""
    import base64
    import io

    from PIL import Image

    path = Path(file_path)
    size_bytes = path.stat().st_size
    metadata: dict[str, Any] = {"size_bytes": size_bytes, "extension": path.suffix.lower()}
    schema: dict[str, Any] = {"type": "image"}

    try:
        with Image.open(str(path)) as img:
            schema["width"] = img.width
            schema["height"] = img.height
            schema["mode"] = img.mode
            schema["format"] = img.format

            # Create tiny thumbnail for preview without loading full image into context
            thumb = img.copy()
            thumb.thumbnail((128, 128))
            buf = io.BytesIO()
            fmt = img.format or "PNG"
            if fmt not in ("PNG", "JPEG", "GIF", "BMP", "WEBP"):
                fmt = "PNG"
            thumb.save(buf, format=fmt)
            thumbnail_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        schema["read_error"] = str(exc)
        thumbnail_b64 = ""

    return {
        "schema": schema,
        "sample": thumbnail_b64,
        "metadata": metadata,
    }
