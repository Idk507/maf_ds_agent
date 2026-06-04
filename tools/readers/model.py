"""tools/readers/model.py — Inspect existing ML model artefacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def read_model(file_path: str) -> dict[str, Any]:
    """
    Detect and describe an existing model file.
    Supports: .pkl/.pickle (sklearn/joblib), .pt/.pth (PyTorch), .h5/.keras (Keras), .onnx.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    size_bytes = path.stat().st_size
    schema: dict[str, Any] = {"type": "existing_model", "extension": ext, "size_bytes": size_bytes}
    metadata: dict[str, Any] = {"size_bytes": size_bytes}
    description = ""

    try:
        if ext in (".pkl", ".pickle"):
            import pickle

            with open(path, "rb") as f:
                obj = pickle.load(f)
            schema["class"] = type(obj).__name__
            schema["module"] = type(obj).__module__
            # Sklearn models: expose params
            if hasattr(obj, "get_params"):
                schema["params"] = obj.get_params()
            description = f"Pickle model: {type(obj).__name__}"

        elif ext == ".joblib":
            import joblib

            obj = joblib.load(str(path))
            schema["class"] = type(obj).__name__
            schema["module"] = type(obj).__module__
            if hasattr(obj, "get_params"):
                schema["params"] = obj.get_params()
            description = f"Joblib model: {type(obj).__name__}"

        elif ext in (".pt", ".pth"):
            import torch

            obj = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(obj, dict) and "state_dict" in obj:
                schema["pytorch_type"] = "checkpoint"
                schema["keys"] = list(obj.keys())
            elif isinstance(obj, dict):
                schema["pytorch_type"] = "state_dict"
                schema["layer_count"] = len(obj)
            else:
                schema["pytorch_type"] = str(type(obj).__name__)
            description = "PyTorch model checkpoint"

        elif ext in (".h5", ".keras"):
            import tensorflow as tf

            model = tf.keras.models.load_model(str(path), compile=False)
            schema["input_shape"] = str(model.input_shape)
            schema["output_shape"] = str(model.output_shape)
            schema["layer_count"] = len(model.layers)
            description = "Keras/TF model"

        elif ext == ".onnx":
            import onnx

            model = onnx.load(str(path))
            schema["opset"] = model.opset_import[0].version if model.opset_import else None
            schema["inputs"] = [inp.name for inp in model.graph.input]
            schema["outputs"] = [out.name for out in model.graph.output]
            description = "ONNX model"

        elif ext == ".safetensors":
            import safetensors

            with safetensors.safe_open(str(path), framework="pt") as f:
                schema["keys"] = list(f.keys())[:20]
            description = "SafeTensors model"

        else:
            description = f"Unknown model format: {ext}"

    except Exception as exc:
        schema["read_error"] = str(exc)
        description = f"Error reading model: {exc}"

    schema["description"] = description

    return {
        "schema": schema,
        "sample": description,
        "metadata": metadata,
    }
