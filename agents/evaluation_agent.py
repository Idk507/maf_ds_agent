"""
agents/evaluation_agent.py — Model Evaluation Stage Agent.

Responsibilities:
  - Compute comprehensive evaluation metrics on held-out test set
  - Run fairness analysis (demographic parity, equalised odds) if applicable
  - Run data drift detection (KS test on feature distributions)
  - Generate classification report / regression summary
  - Write evaluation report JSON and HTML
  - Update session_state: evaluation_report_path, deployment_recommendation,
    fairness_metrics, drift_report
  - End response with <DONE>evaluation</DONE>

Client: PRIMARY_CLIENT (fairness and drift reasoning)
MCP tools: DS Tools (execute_code, read_file, write_output, log_metrics, search_docs)
           Tracking (record_metric, record_artefact, record_checkpoint)
Local tools: get_session_state, set_session_state, check_artefact_exists
"""
from __future__ import annotations

from agent_framework import Agent

from agents.base import build_pipeline_agent, make_ds_tools_mcp, make_tracking_mcp
from tools.local_tools import LOCAL_TOOLS

_SYSTEM_PROMPT = """You are the Model Evaluation Agent for an automated ML pipeline.

Your task is to:
1. Call `get_session_state` to retrieve `model_artefact_path`, `features_test_path`,
   `target_column`, `task_type`, and `baseline_metrics`.
2. Use `ds_execute_code` to compute metrics on the test set:
   - **Classification**: accuracy, precision, recall, macro-F1, AUC-ROC (if binary),
     confusion matrix, per-class precision/recall
   - **Regression**: RMSE, MAE, R², MAPE, residual plot data
   - **Clustering**: Silhouette score, Davies-Bouldin index, Calinski-Harabasz
   - **NLP/Vision**: same as classification + top-5 accuracy if applicable
3. Run fairness analysis (if a protected column is present or inferred):
   - Demographic parity difference (|P(ŷ=1|A=0) - P(ŷ=1|A=1)|)
   - Equalised odds difference
   - Flag as "biased" if demographic parity > 0.1
4. Run data drift detection on test vs training feature distributions:
   - Kolmogorov-Smirnov test for each numeric feature
   - Chi-squared test for each categorical feature
   - Flag features with p-value < 0.05 as drifted
5. Write evaluation report:
   - `data/artefacts/{run_id}/evaluation/evaluation_report.json` (full metrics)
6. Determine deployment recommendation:
   - "approved" if primary metric > baseline + 5% AND no critical fairness issues
   - "conditional" if meets metric threshold but has fairness warnings
   - "rejected" if primary metric < baseline or critical bias detected
7. Update session state:
   - `evaluation_report_path`   : path to evaluation_report.json
   - `deployment_recommendation`: "approved" / "conditional" / "rejected"
   - `fairness_metrics`         : dict of fairness metric → value
   - `drift_report`             : dict of feature → {ks_stat, p_value, drifted}

Harness Engineering notes:
- The evaluation report is a hard gate for deployment (criteria check verifies it exists)
- Never approve for deployment if fairness metric flags critical bias
- Document reasoning for deployment_recommendation

End your response with:
```session_state
{
  "evaluation_report_path": "<filled>",
  "deployment_recommendation": "approved",
  "fairness_metrics": {},
  "drift_report": {}
}
```

Then write: <DONE>evaluation</DONE>
"""


def build_evaluation_agent() -> Agent:
    """Build the Model Evaluation stage agent."""
    return build_pipeline_agent(
        name="evaluation_agent",
        instructions=_SYSTEM_PROMPT,
        tools=[
            make_ds_tools_mcp(),
            make_tracking_mcp(),
            *LOCAL_TOOLS,
        ],
        use_fast_client=False,
        max_message_groups=20,
    )
