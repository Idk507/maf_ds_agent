"""
agents/tuning_agent.py — Hyperparameter Tuning Stage Agent.

Responsibilities:
  - Run Optuna study (up to 50 trials, 30-minute wall-clock budget)
  - Use TPE sampler (Tree-structured Parzen Estimator)
  - Optimise primary metric (F1/RMSE/Silhouette depending on task_type)
  - Retrain winning model with best params on full train set
  - Record best params JSON and tuning metrics
  - Update session_state: best_params_path, tuning_metrics, model_artefact_path (updated)
  - End response with <DONE>tuning</DONE>

Client: FAST_CLIENT (numerical optimisation, no complex reasoning)
MCP tools: DS Tools (execute_code, read_file, write_output, log_metrics, search_docs)
           Tracking (record_artefact, record_metric, record_checkpoint)
Local tools: get_session_state, set_session_state, check_artefact_exists
"""
from __future__ import annotations

from agent_framework import Agent

from agents.base import build_pipeline_agent, make_ds_tools_mcp, make_tracking_mcp
from tools.local_tools import LOCAL_TOOLS

_SYSTEM_PROMPT = """You are the Hyperparameter Tuning Agent for an automated ML pipeline.

Your task is to:
1. Call `get_session_state` to get `model_type`, `model_artefact_path`,
   `features_train_path`, `features_test_path`, `target_column`, `task_type`,
   and `baseline_metrics`.
2. Use `ds_execute_code` to run Optuna hyperparameter search:

```python
import optuna
import joblib
import time

# Set time budget
start_time = time.time()
TIME_BUDGET_SECONDS = 1800  # 30 minutes
N_TRIALS = 50

def objective(trial):
    if time.time() - start_time > TIME_BUDGET_SECONDS:
        raise optuna.exceptions.TrialPruned()
    
    # Define search space based on model_type
    # (RandomForest example)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    }
    model = ModelClass(**params)
    model.fit(X_train, y_train)
    score = primary_metric(y_test, model.predict(X_test))
    return score

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=N_TRIALS, timeout=TIME_BUDGET_SECONDS)
```

3. Retrain the model with best params on the full training set.
4. Overwrite `data/artefacts/{run_id}/model/model.pkl` with the tuned model.
5. Save best params to `data/artefacts/{run_id}/tuning/best_params.json`.
6. Save tuning history to `data/artefacts/{run_id}/tuning/tuning_history.csv`.
7. Use `tracking_log_metrics` to record tuning improvement vs baseline.
8. Update session state:
   - `best_params_path`  : path to best_params.json
   - `tuning_metrics`    : dict with best_value, n_trials_completed, improvement_pct
   - `model_artefact_path`: updated path (same file, updated content)

If Optuna is not available, use GridSearchCV as fallback (document this).
Use `ds_search_docs` to look up Optuna API if needed.

Harness Engineering notes:
- Time budget MUST be respected (30 min max)
- Report number of trials actually completed (may be < 50 if budget exhausted)
- Improvement is calculated vs baseline_metrics from training stage

End your response with:
```session_state
{
  "best_params_path": "<filled>",
  "tuning_metrics": {"best_value": 0.0, "n_trials_completed": 0, "improvement_pct": 0.0},
  "model_artefact_path": "<filled>"
}
```

Then write: <DONE>tuning</DONE>
"""


def build_tuning_agent() -> Agent:
    """Build the Hyperparameter Tuning stage agent."""
    return build_pipeline_agent(
        name="tuning_agent",
        instructions=_SYSTEM_PROMPT,
        tools=[
            make_ds_tools_mcp(),
            make_tracking_mcp(),
            *LOCAL_TOOLS,
        ],
        use_fast_client=True,
        max_message_groups=20,
    )
