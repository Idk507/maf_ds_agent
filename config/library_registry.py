"""config/library_registry.py — Maps library names → documentation root URLs.

Used by Browser-Use inside search_docs. When a library is NOT in this registry,
search_docs falls through to DuckDuckGo immediately.
"""

LIBRARY_REGISTRY: dict[str, str] = {
    "pandas": "https://pandas.pydata.org/docs/",
    "numpy": "https://numpy.org/doc/stable/",
    "scikit-learn": "https://scikit-learn.org/stable/",
    "sklearn": "https://scikit-learn.org/stable/",
    "xgboost": "https://xgboost.readthedocs.io/en/stable/",
    "lightgbm": "https://lightgbm.readthedocs.io/en/stable/",
    "catboost": "https://catboost.ai/en/docs/",
    "torch": "https://pytorch.org/docs/stable/",
    "pytorch": "https://pytorch.org/docs/stable/",
    "torchvision": "https://pytorch.org/vision/stable/",
    "transformers": "https://huggingface.co/docs/transformers/",
    "datasets": "https://huggingface.co/docs/datasets/",
    "sentence-transformers": "https://www.sbert.net/docs/",
    "shap": "https://shap.readthedocs.io/en/stable/",
    "lime": "https://lime-ml.readthedocs.io/en/latest/",
    "captum": "https://captum.ai/api/",
    "optuna": "https://optuna.readthedocs.io/en/stable/",
    "pyarrow": "https://arrow.apache.org/docs/python/",
    "openpyxl": "https://openpyxl.readthedocs.io/en/stable/",
    "pymupdf": "https://pymupdf.readthedocs.io/en/latest/",
    "fitz": "https://pymupdf.readthedocs.io/en/latest/",
    "pillow": "https://pillow.readthedocs.io/en/stable/",
    "PIL": "https://pillow.readthedocs.io/en/stable/",
    "fastmcp": "https://gofastmcp.com/",
    "agent-framework": "https://learn.microsoft.com/en-us/azure/ai-studio/",
    "magika": "https://google.github.io/magika/",
    "filetype": "https://pypi.org/project/filetype/",
    "scipy": "https://docs.scipy.org/doc/scipy/",
    "statsmodels": "https://www.statsmodels.org/stable/",
    "matplotlib": "https://matplotlib.org/stable/",
    "seaborn": "https://seaborn.pydata.org/",
    "plotly": "https://plotly.com/python/",
    "fairlearn": "https://fairlearn.org/v0.11/",
    "onnx": "https://onnx.ai/onnx/",
    "onnxruntime": "https://onnxruntime.ai/docs/",
}


def get_docs_url(library_name: str) -> str | None:
    """Return the documentation URL for the given library, or None if unknown."""
    return LIBRARY_REGISTRY.get(library_name.lower()) or LIBRARY_REGISTRY.get(library_name)
