"""
workflows/criteria.py — Stage exit-gate verification.

Each pipeline stage has deterministic criteria that must pass before the
Ralph Loop can checkpoint and advance. If any criterion fails, the loop
injects a structured failure feedback message and retries.

Usage (by ds_tools/server.py verify_stage tool):
    from workflows.criteria import get_criteria
    criteria = get_criteria(stage_name)
    failures = criteria.verify(run_id, session_state)
    # failures: list[dict] — each: {assertion, expected, actual}
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


# ── Base class ────────────────────────────────────────────────────────


class StageCriteria(ABC):
    """Abstract base for per-stage exit-gate verification."""

    @abstractmethod
    def verify(self, run_id: str, session_state: dict[str, Any]) -> list[dict[str, str]]:
        """
        Run all assertions.
        Returns list of failures. Empty list = all passed.
        Each failure: {assertion: str, expected: str, actual: str}
        """
        ...

    # ── Helper methods ────────────────────────────────────────────────

    def _require_key(
        self,
        key: str,
        session_state: dict,
        failures: list[dict[str, str]],
    ) -> bool:
        """Assert a key is present and non-empty in session_state."""
        val = session_state.get(key)
        if not val:
            failures.append(
                {
                    "assertion": f"session_state.{key} is set",
                    "expected": "non-empty value",
                    "actual": repr(val),
                }
            )
            return False
        return True

    def _require_file(
        self,
        key: str,
        session_state: dict,
        failures: list[dict[str, str]],
        run_id: str,
    ) -> bool:
        """Assert a file path stored in session_state[key] actually exists on disk."""
        path_val = session_state.get(key)
        if not path_val:
            failures.append(
                {
                    "assertion": f"session_state.{key} references an existing file",
                    "expected": "file path set and exists",
                    "actual": "key missing or empty",
                }
            )
            return False
        p = Path(str(path_val))
        if not p.exists():
            failures.append(
                {
                    "assertion": f"file at session_state.{key} exists",
                    "expected": f"file exists at {p}",
                    "actual": "file not found",
                }
            )
            return False
        return True

    def _require_metrics_key(
        self,
        metrics_key: str,
        session_state: dict,
        failures: list[dict[str, str]],
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> bool:
        """Assert a numeric metric is present and within bounds."""
        metrics = session_state.get("baseline_metrics") or session_state.get("tuning_metrics", {})
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except Exception:
                metrics = {}
        val = metrics.get(metrics_key)
        if val is None:
            failures.append(
                {
                    "assertion": f"metrics.{metrics_key} is set",
                    "expected": "numeric value",
                    "actual": "not found",
                }
            )
            return False
        if min_value is not None and float(val) < min_value:
            failures.append(
                {
                    "assertion": f"metrics.{metrics_key} >= {min_value}",
                    "expected": f">= {min_value}",
                    "actual": str(val),
                }
            )
            return False
        if max_value is not None and float(val) > max_value:
            failures.append(
                {
                    "assertion": f"metrics.{metrics_key} <= {max_value}",
                    "expected": f"<= {max_value}",
                    "actual": str(val),
                }
            )
            return False
        return True


# ── Stage criteria implementations ────────────────────────────────────


class IngestionCriteria(StageCriteria):
    """Ingestion stage: file read, schema extracted, artefact saved."""

    def verify(self, run_id: str, session_state: dict) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        self._require_key("file_type_result", session_state, failures)
        self._require_key("schema", session_state, failures)
        self._require_key("pipeline_variant", session_state, failures)
        self._require_file("input_artefact_path", session_state, failures, run_id)
        return failures


class EDACriteria(StageCriteria):
    """EDA stage: report written, narrative produced, charts generated."""

    def verify(self, run_id: str, session_state: dict) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        self._require_key("eda_narrative_path", session_state, failures)
        self._require_file("eda_narrative_path", session_state, failures, run_id)
        self._require_key("data_quality_flags", session_state, failures)
        # eda_report_path is optional (some variants skip HTML report)
        # chart_paths may be empty for text datasets — only require key
        if "chart_paths" not in session_state:
            failures.append(
                {
                    "assertion": "session_state.chart_paths is set",
                    "expected": "list (may be empty)",
                    "actual": "key missing",
                }
            )
        return failures


class CleaningCriteria(StageCriteria):
    """Cleaning stage: cleaned dataset written, transformation log saved."""

    def verify(self, run_id: str, session_state: dict) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        self._require_file("cleaned_dataset_path", session_state, failures, run_id)
        self._require_file("transformation_log_path", session_state, failures, run_id)
        self._require_key("cleaning_summary", session_state, failures)
        self._require_key("content_hash", session_state, failures)
        return failures


class FeatureEngineeringCriteria(StageCriteria):
    """Feature engineering: train/test splits written, feature manifest saved."""

    def verify(self, run_id: str, session_state: dict) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        self._require_file("features_train_path", session_state, failures, run_id)
        self._require_file("features_test_path", session_state, failures, run_id)
        self._require_file("feature_manifest_path", session_state, failures, run_id)
        self._require_key("target_column", session_state, failures)
        self._require_key("task_type", session_state, failures)
        # task_type must be one of: classification, regression, clustering
        task_type = session_state.get("task_type", "")
        if task_type not in ("classification", "regression", "clustering"):
            failures.append(
                {
                    "assertion": "session_state.task_type is valid",
                    "expected": "classification | regression | clustering",
                    "actual": repr(task_type),
                }
            )
        return failures


class TrainingCriteria(StageCriteria):
    """Training: model artefact and ONNX export saved, baseline metrics recorded."""

    def verify(self, run_id: str, session_state: dict) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        self._require_file("model_artefact_path", session_state, failures, run_id)
        self._require_key("model_type", session_state, failures)
        self._require_key("baseline_metrics", session_state, failures)
        self._require_key("algorithm_comparison", session_state, failures)
        # ONNX export is required for deployment — warn if missing
        if not session_state.get("onnx_model_path"):
            failures.append(
                {
                    "assertion": "session_state.onnx_model_path is set",
                    "expected": "ONNX export path",
                    "actual": "not set",
                }
            )
        return failures


class TuningCriteria(StageCriteria):
    """HP Tuning: best params written, tuning metrics recorded."""

    def verify(self, run_id: str, session_state: dict) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        self._require_file("best_params_path", session_state, failures, run_id)
        self._require_key("tuning_metrics", session_state, failures)
        return failures


class EvaluationCriteria(StageCriteria):
    """Evaluation: report written, deployment recommendation made, fairness checked."""

    def verify(self, run_id: str, session_state: dict) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        self._require_file("evaluation_report_path", session_state, failures, run_id)
        self._require_key("deployment_recommendation", session_state, failures)
        # Fairness metrics must be present (may be empty dict for non-classification)
        if "fairness_metrics" not in session_state:
            failures.append(
                {
                    "assertion": "session_state.fairness_metrics is set",
                    "expected": "dict (may be empty)",
                    "actual": "key missing",
                }
            )
        return failures


class ExplainabilityCriteria(StageCriteria):
    """Explainability: SHAP values computed, plots saved, narrative written."""

    def verify(self, run_id: str, session_state: dict) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        self._require_file("shap_values_path", session_state, failures, run_id)
        self._require_key("explanation_narrative_path", session_state, failures)
        # shap_plot_paths must be a list with at least one entry
        plot_paths = session_state.get("shap_plot_paths", [])
        if not isinstance(plot_paths, list) or len(plot_paths) == 0:
            failures.append(
                {
                    "assertion": "session_state.shap_plot_paths has at least one plot",
                    "expected": "list with >= 1 path",
                    "actual": repr(plot_paths),
                }
            )
        return failures


class ReportCriteria(StageCriteria):
    """Report: Markdown and HTML report files written."""

    def verify(self, run_id: str, session_state: dict) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        self._require_file("report_md_path", session_state, failures, run_id)
        self._require_file("report_html_path", session_state, failures, run_id)
        return failures


class DeploymentCriteria(StageCriteria):
    """Deployment: endpoint URL set, all 10/10 smoke tests passed."""

    def verify(self, run_id: str, session_state: dict) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        self._require_key("endpoint_url", session_state, failures)
        smoke = session_state.get("smoke_test_results", {})
        if isinstance(smoke, str):
            try:
                smoke = json.loads(smoke)
            except Exception:
                smoke = {}
        passed = smoke.get("passed", 0)
        total = smoke.get("total", 10)
        if int(passed) < int(total):
            failures.append(
                {
                    "assertion": f"smoke_tests {passed}/{total} all pass",
                    "expected": f"{total}/{total} passed",
                    "actual": f"{passed}/{total} passed",
                }
            )
        return failures


# ── Registry ─────────────────────────────────────────────────────────

_CRITERIA_REGISTRY: dict[str, StageCriteria] = {
    "ingestion": IngestionCriteria(),
    "eda": EDACriteria(),
    "clean": CleaningCriteria(),
    "features": FeatureEngineeringCriteria(),
    "feature_eng": FeatureEngineeringCriteria(),
    "feature_eng_text": FeatureEngineeringCriteria(),
    "feature_eng_image": FeatureEngineeringCriteria(),
    "training": TrainingCriteria(),
    "train": TrainingCriteria(),
    "tuning": TuningCriteria(),
    "tune": TuningCriteria(),
    "evaluation": EvaluationCriteria(),
    "evaluate": EvaluationCriteria(),
    "explain": ExplainabilityCriteria(),
    "explainability": ExplainabilityCriteria(),
    "report": ReportCriteria(),
    "deploy": DeploymentCriteria(),
    "deployment": DeploymentCriteria(),
}


class _PassAllCriteria(StageCriteria):
    """Used for unknown stage names — always passes (no criteria defined)."""

    def verify(self, run_id: str, session_state: dict) -> list[dict[str, str]]:
        return []


def get_criteria(stage_name: str) -> StageCriteria:
    """
    Retrieve the StageCriteria for a given stage name (case-insensitive).
    Returns a pass-all no-op for unrecognized stages.
    """
    return _CRITERIA_REGISTRY.get(stage_name.lower(), _PassAllCriteria())
