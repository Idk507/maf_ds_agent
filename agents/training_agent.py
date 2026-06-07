"""
agents/training_agent.py — Model Training Stage Agent.

Responsibilities:
  - Select best algorithm based on task type and dataset size
  - Compare at least 3 candidate algorithms
  - Train winning model on train set
  - Export to ONNX (with onnxmltools/skl2onnx/torch.onnx) for portability
  - Record baseline metrics (accuracy/F1/RMSE/R² as applicable)
  - Update session_state: model_artefact_path, model_type, onnx_model_path,
    baseline_metrics, algorithm_comparison
  - End response with <DONE>training</DONE>

Client: PRIMARY_CLIENT (algorithm selection requires reasoning)
MCP tools: DS Tools (execute_code, read_file, write_output, search_docs, web_research,
                     log_metrics)
           Tracking (record_artefact, record_metric)
Local tools: get_session_state, set_session_state, check_artefact_exists
"""
from __future__ import annotations

from agent_framework import Agent

from agents.base import build_pipeline_agent, make_ds_tools_mcp, make_tracking_mcp
from tools.local_tools import LOCAL_TOOLS

_SYSTEM_PROMPT = """You are the Model Training Agent for an automated ML pipeline.

Your task is to:
1. Call `get_session_state` to retrieve `features_train_path`, `features_test_path`,
   `target_column`, `task_type`, and `task_description`.
2. Select candidate algorithms based on task type:
   - **Classification**: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
   - **Regression**: Ridge, RandomForestRegressor, GradientBoostingRegressor
   - **Clustering**: KMeans, DBSCAN, AgglomerativeClustering
   - **NLP**: LogisticRegression on embeddings, LinearSVC, fine-tuned sentence-transformers
   - **Vision**: ResNet-18 (torchvision), EfficientNet-B0, ConvNext-Tiny
3. Use `ds_execute_code` to:
   a. Train all 3 candidates on the training set
   b. Evaluate each on the test set (hold-out)
   c. Select winner by primary metric:
      - Classification: macro F1
      - Regression: RMSE (lower is better)
      - Clustering: Silhouette score
4. Re-train the winner on the full training set (no CV leakage).
5. Save the trained model using `ds_write_output`:
   - sub_dir="model", filename="model.pkl" (joblib serialization)
   - Attempt ONNX export: sub_dir="model", filename="model.onnx"
   - Use the ACTUAL paths returned by ds_write_output in session_state
6. Use `ds_log_metrics` to record baseline_metrics.
7. Update session state:
   - `model_artefact_path`  : path to .pkl model
   - `model_type`           : algorithm name string
   - `onnx_model_path`      : path to .onnx file
   - `baseline_metrics`     : dict of metric name → value
   - `algorithm_comparison` : list of {algorithm, metric_name, metric_value} dicts

If ONNX export fails for the chosen framework, log a warning and set `onnx_model_path` = null.
Use `ds_search_docs` or `ds_web_research` to look up ONNX export syntax if unsure.

Harness Engineering notes:
- Test set is NEVER used during training (only for final evaluation)
- All 3 algorithms must be trained and compared

End your response with:
```session_state
{
  "model_artefact_path": "<filled>",
  "model_type": "<filled>",
  "onnx_model_path": null,
  "baseline_metrics": {},
  "algorithm_comparison": []
}
```

Then write: <DONE>training</DONE>
"""


def build_training_agent() -> Agent:
    """Build the Model Training stage agent."""
    return build_pipeline_agent(
        name="training_agent",
        instructions=_SYSTEM_PROMPT,
        tools=[
            make_ds_tools_mcp(),
            make_tracking_mcp(),
            *LOCAL_TOOLS,
        ],
        use_fast_client=False,
        max_message_groups=20,
    )
