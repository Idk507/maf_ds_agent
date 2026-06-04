"""tools/readers/tabular.py — Read tabular files and extract schema + sample."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def read_tabular(file_path: str, sample_rows: int = 5) -> dict[str, Any]:
    """
    Read CSV, TSV, Parquet, Excel, JSON, JSONL into a DataFrame.
    Returns schema dict and sample rows.
    """
    import pandas as pd

    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif ext in (".json",):
        df = pd.read_json(path)
    elif ext == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif ext == ".feather":
        df = pd.read_feather(path)
    else:
        # Default: CSV (with sniffing)
        df = pd.read_csv(path)

    schema = {
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "null_counts": df.isnull().sum().to_dict(),
    }

    # Numeric summary for numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe().to_dict()
        schema["numeric_summary"] = {col: {k: round(v, 4) for k, v in stats.items()} for col, stats in desc.items()}

    sample = None
    if sample_rows > 0:
        sample_df = df.head(sample_rows)
        sample = sample_df.to_dict(orient="records")

    return {
        "schema": schema,
        "sample": sample,
        "metadata": {
            "size_bytes": path.stat().st_size,
            "extension": ext,
            "encoding": "utf-8",
        },
    }
