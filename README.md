# Suspended Sediment Particle Size Prediction

This project compares Random Forest, Support Vector Regression (SVR), and XGBoost for estimating median particle diameter (`d50`) and distribution variance (`sigma2`) from the supplied synthetic dataset.

## Data

`data/synthetic_data_improved.csv` is the dataset supplied for the project. It contains 1,648 rows with eight predictors and two targets. The data are synthetic; results should not be interpreted as validation on field measurements.

## Notebook and reported workflow

`notebooks/ML mini project.ipynb` is the original notebook provided for this project. It is preserved byte-for-byte, including its saved cell outputs and comparison table. A copy of the supplied CSV is placed beside the notebook because the notebook loads `synthetic_data_improved.csv` by a relative filename.

The notebook removes the time column and incomplete rows (1,402 remain), then removes rows at or above either target's 99th percentile (1,384 remain). It uses a shuffled 70/30 split with `random_state=42` (968 training rows, 416 test rows), eight input variables for a Random Forest importance table, and a fixed four-feature set (`S`, `ub`, `np`, `T`) for the three regressors. SVR uses a standardized pipeline and five-fold grid search.

### Evaluation caveats

- The 99th-percentile cutoffs are calculated from all complete rows before the train/test split. Since they use target values from rows later assigned to the test set, the reported scores may be optimistic for an untouched future sample.
- The feature-importance table ranks `ub`, `S`, `T`, and `chl_a` highest in the saved notebook output. The notebook's manually specified model inputs are `S`, `ub`, `np`, and `T`. The reported results use that fixed list; the discrepancy is documented here because the notebook is preserved unchanged.
- This is a random holdout from synthetic data. It does not measure future-period forecasting performance or performance on field samples.
- Re-running with different library versions can change scores slightly. The PDF and saved `results/model_metrics.csv` report the values in the original notebook's saved comparison table.

## Notebook-reported test metrics

| Model | d50 R² | d50 RMSE | d50 MAE | sigma2 R² | sigma2 RMSE | sigma2 MAE |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.839729 | 15.769309 | 12.557175 | 0.904587 | 7.931298 | 6.251205 |
| SVR | 0.854116 | 15.044842 | 11.969561 | 0.914782 | 7.495579 | 5.801557 |
| XGBoost | 0.849638 | 15.274025 | 12.284159 | 0.913948 | 7.532165 | 5.880333 |

## Files

- `notebooks/ML mini project.ipynb` — original step-by-step notebook with saved outputs and final comparison table
- `notebooks/synthetic_data_improved.csv` — same supplied CSV, beside the notebook for its relative data path
- `run_project.py` — runnable script that follows the notebook workflow and writes files in `results/`
- `data/synthetic_data_improved.csv` — project dataset
- `results/` — feature importance, model metrics, metadata, test predictions, and chart
- `Suspended_Sediment_Particle_Size_Project_Report.pdf` — two-page report matching the notebook's saved comparison table and documenting the evaluation caveats

## Setup and run

Python 3.10 or newer is recommended. From the project root:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```powershell
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

To open the preserved notebook from the project root:

```bash
jupyter notebook "notebooks/ML mini project.ipynb"
```
