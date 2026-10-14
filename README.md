# Suspended Sediment Particle Size Prediction

A reproducible comparison of Random Forest, Support Vector Regression (SVR), and XGBoost for estimating median particle diameter (`d50`) and particle-size-distribution variance (`sigma2`). The primary notebook preserves the supplied step-by-step workflow and its result table.

## Dataset and provenance

`data/synthetic_data_improved.csv` is the supplied synthetic dataset. It contains 1,648 hourly rows, eight environmental predictor columns, and two targets (`d50`, `sigma2`). Predictor values include missing entries. This is synthetic data, not the original Stanford dataset or measured field observations.

## Workflow used in the original notebook

1. Remove the time column and drop incomplete rows (1,402 records remain).
2. Remove rows at or above the 99th percentile for either target (1,384 remain).
3. Split randomly into 70% train and 30% test using `random_state=42` (968 train, 416 test).
4. Use the four predictors `S`, `ub`, `np`, and `T`.
5. Fit Random Forest, tuned SVR (5-fold grid search with standard scaling), and XGBoost separately for each target.
6. Evaluate on the same held-out test rows using R², RMSE, and MAE.

The outlier thresholds are calculated before the split in the supplied notebook. This means test-target values affect which records are retained, which can make the reported scores optimistic. The results should be interpreted as a reproducible classroom comparison on synthetic data, not validated real-world performance. Small differences in the last decimal may occur across library versions.

## Project files

- `notebooks/ML mini project.ipynb` — supplied step-by-step notebook with executed outputs, metrics table, and charts
- `run_project.py` — executable version of the same model workflow
- `data/synthetic_data_improved.csv` — supplied data
- `results/` — generated metrics, test predictions, feature importance, run metadata, and chart
- `Suspended_Sediment_Particle_Size_Project_Report.pdf` — two-page summary using the same split and workflow

## Setup and run

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, install dependencies, and run the script:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_project.py
```

```bash
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
python run_project.py
```

Open the step-by-step notebook from the project directory with:

```bash
jupyter notebook "notebooks/ML mini project.ipynb"
```

The script writes `results/model_metrics.csv`, `results/test_predictions.csv`, `results/feature_importance.csv`, `results/run_metadata.csv`, and `results/r2_comparison.png`.

## Private GitHub repository

This project is hosted in the private repository `DARKDRAGON02/suspended-sediment-particle-size-prediction`. Collaborators can clone it and follow the setup steps above.
