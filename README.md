# Suspended Sediment Particle Size Prediction

A reproducible comparison of Random Forest, Support Vector Regression (SVR), and XGBoost for estimating median sediment particle diameter (`d50`) and particle-size-distribution variance (`sigma2`).

## Dataset and provenance

`data/synthetic_data_improved.csv` is the supplied synthetic dataset. It contains 1,648 hourly rows, eight environmental predictor columns, and two targets (`d50`, `sigma2`). Predictor values include missing entries; targets are present. This is **synthetic data**, not the original Stanford dataset or measured field observations. Results describe performance on this synthetic dataset only and should not be presented as a reproduction of Stanford's reported results.

The timestamp is used only to sort the records and make a chronological holdout. It is not a model feature. The first 80% of rows are used for training and the final 20% for testing. Median imputation is fitted on training data only. SVR additionally uses standard scaling. Models are evaluated separately for each target with R², RMSE, and MAE.

## Project files

- `run_project.py` — complete executable training/evaluation pipeline
- `notebooks/01_model_comparison.ipynb` — notebook interface to the pipeline and dataset checks
- `data/synthetic_data_improved.csv` — included data
- `results/` — generated metrics, predictions, and chart (created when the pipeline runs)

## Setup and run

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

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

To use the notebook, start Jupyter from this project directory:

```bash
jupyter notebook notebooks/01_model_comparison.ipynb
```

The pipeline writes `results/model_metrics.csv`, `results/test_predictions.csv`, and `results/r2_comparison.png`.

## Interpretation and limitations

This dataset is synthetic, relatively small, and may encode simplified or artificial relationships. A chronological split reduces the chance of placing adjacent time records in both sets, but does not establish performance on real sediment samples or on a different monitoring period. The project is a model-comparison exercise, not a validated environmental monitoring tool. Check the data source and obtain faculty approval before describing it as field data or the Stanford dataset.

## Private GitHub repository

Create a new **private** repository, upload this project folder, and invite your faculty/TA and collaborator `tejuro` under **Settings → Collaborators**. Keep the repository private. Verify that each invitation is accepted before relying on access for review.
