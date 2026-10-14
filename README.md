# Suspended Sediment Particle Size Prediction

A reproducible comparison of Random Forest, Support Vector Regression (SVR), and XGBoost for estimating median particle diameter (`d50`) and particle-size-distribution variance (`sigma2`). The notebook preserves the original step-by-step structure but corrects the evaluation procedure to avoid target leakage.

## Dataset and provenance

`data/synthetic_data_improved.csv` is the supplied synthetic dataset. It contains 1,648 hourly rows, eight environmental predictor columns, and two targets (`d50`, `sigma2`). Predictor values include missing entries. This is synthetic data, not verified field observations or the original Stanford dataset.

## Corrected evaluation workflow

1. Remove the time column and rows with missing values (1,402 complete rows remain).
2. Split the data into 70% training and 30% test rows using `random_state=42` (981 train, 421 test).
3. Fit Random Forest feature importance using training data only and select the four highest ranked predictors. In the current run these are `ub`, `S`, `chl_a`, and `T`.
4. Fit Random Forest, standardized SVR with a five-fold grid search, and XGBoost separately for each target. Feature selection and hyperparameter tuning use training data only.
5. Evaluate once on the held-out test rows with R², RMSE, and MAE.

No rows are removed based on target values. This avoids using held-out target labels to choose which observations remain. The random split measures performance on similar rows from this synthetic dataset; it does not establish performance on future time periods or real field samples. Scores may vary slightly across library versions.

## Project files

- `notebooks/ML mini project.ipynb` — step-by-step executed notebook with the comparison table and plots
- `run_project.py` — executable version of the same workflow
- `data/synthetic_data_improved.csv` — supplied dataset
- `results/` — metrics, test predictions, training-only feature importance, run metadata, and chart
- `Suspended_Sediment_Particle_Size_Project_Report.pdf` — two-page report generated from the same results

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

Open the notebook from the project directory:

```bash
jupyter notebook "notebooks/ML mini project.ipynb"
```

## Repository

The project is hosted in the private repository `DARKDRAGON02/suspended-sediment-particle-size-prediction`.
