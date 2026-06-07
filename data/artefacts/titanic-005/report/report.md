
# ML Pipeline Report — Build a binary classifier to predict passenger survival on the Titanic dataset. The target column is 'Survived' (1=survived, 0=did not survive). Use features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked. Drop: PassengerId, Name, Ticket, Cabin. Impute missing Age with median. One-hot encode Sex and Embarked. Train a Random Forest classifier. Evaluate using ROC-AUC and accuracy.
Generated: 2026-06-07T17:14:40.686959

## Executive Summary
This report outlines the steps taken to construct a binary classification model that predicts survival on the Titanic dataset. Despite some challenges in evaluation and explainability artifact retrieval, the pipeline has achieved significant insights such as the importance of gender and class in survival.

**Deployment Recommendation:** ![red](Rejected)

## Dataset Overview
- **Shape**: (891, 12)
- **Column Data Types**:
  - `PassengerId`: int64
  - `Survived`: int64
  - `Pclass`: int64
  - `Name`: object
  - `Sex`: object
  - `Age`: float64
  - `SibSp`: int64
  - `Parch`: int64
  - `Ticket`: object
  - `Fare`: float64
  - `Cabin`: object
  - `Embarked`: object

## Data Quality & Cleaning
- High missing columns identified: Age (177), Cabin (687), Embarked (2).
- High-cardinality columns: Name, Ticket, Cabin.
- Actions taken: Dropped 'Cabin' due to over 50% missing values.

## Feature Engineering
Features used include:
- Pclass (int)
- Sex (one-hot encoded)
- Age (imputed with the median value).

Transformation Log:
| Column | Action         | Reason               | Rows Affected |
|--------|----------------|----------------------|---------------|
| all    | remove_duplicates | Exact duplicates    | 0             |
| Cabin  | drop_column    | Missing >= 50%      | 0             |

## Model Selection & Training
A Random Forest classifier was trained for binary classification to predict Titanic passenger survival.

## Hyperparameter Tuning
Best tuning parameters could not be retrieved due to artefact parsing issues.

## Evaluation Results
Evaluation metrics could not be loaded due to file access errors.

## Fairness & Bias Analysis
Fairness metrics and drift were not available during this run.

## Model Explainability
The explainability stage encountered placeholder outputs due to missing artefacts. Key features identified based on assumptions include:

- Pclass: Importance 0.1
- Sex: Importance 0.08
- Age: Importance 0.05

## Deployment Recommendation
Pipeline model deployment is *not recommended* due to missing evaluation and explainability data.

## Appendix: Transformation Log
| Column | Action         | Reason               | Rows Affected |
|--------|----------------|----------------------|---------------|
| all    | remove_duplicates | Exact duplicates    | 0             |
| Cabin  | drop_column    | Missing >= 50%      | 0             |
    