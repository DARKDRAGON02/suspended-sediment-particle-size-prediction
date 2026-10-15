"""Run the model workflow recorded in notebooks/ML mini project.ipynb.

This script intentionally mirrors the notebook. In particular, target-based
99th-percentile filtering occurs before the train/test split, so reported scores
describe the filtered synthetic sample and may be optimistic.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic_data_improved.csv"
RESULTS = ROOT / "results"
FEATURES = ["S", "ub", "np", "T", "a676_a650", "a450_a676", "chl_a", "u"]
TOP_FEATURES = ["S", "ub", "np", "T"]
TARGETS = ["d50", "sigma2"]


def main():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from xgboost import XGBRegressor

    RESULTS.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(DATA)
    data = source.drop(columns=["time"], errors="ignore").dropna()

    # Match the supplied notebook: calculate target cutoffs before splitting.
    d50_limit = data["d50"].quantile(0.99)
    sigma2_limit = data["sigma2"].quantile(0.99)
    data = data[(data["d50"] < d50_limit) & (data["sigma2"] < sigma2_limit)]

    X = data[FEATURES]
    y_d50 = data["d50"]
    y_sigma2 = data["sigma2"]
    X_train, X_test, y_train_d50, y_test_d50 = train_test_split(
        X, y_d50, test_size=0.30, random_state=42
    )
    X_train2, X_test2, y_train_sigma2, y_test_sigma2 = train_test_split(
        X, y_sigma2, test_size=0.30, random_state=42
    )

    importance_model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    importance_model.fit(X_train, y_train_d50)
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance_model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    importance.to_csv(RESULTS / "feature_importance.csv", index=False)

    # The notebook explicitly uses this fixed set. It differs from the four
    # highest entries in the importance table; preserve that for comparability.
    Xtr_d50, Xte_d50 = X_train[TOP_FEATURES], X_test[TOP_FEATURES]
    Xtr_sigma2, Xte_sigma2 = X_train2[TOP_FEATURES], X_test2[TOP_FEATURES]

    rf_d50 = RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, random_state=42, oob_score=True, n_jobs=-1
    )
    rf_sigma2 = RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, random_state=42, oob_score=True, n_jobs=-1
    )
    rf_d50.fit(Xtr_d50, y_train_d50)
    rf_sigma2.fit(Xtr_sigma2, y_train_sigma2)
    pred_rf_d50 = rf_d50.predict(Xte_d50)
    pred_rf_sigma2 = rf_sigma2.predict(Xte_sigma2)

    svr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf")),
    ])
    param_grid = {
        "svr__C": [1, 10, 100, 500, 1000, 2000],
        "svr__epsilon": [0.1, 0.5, 1, 2],
    }
    grid_d50 = GridSearchCV(svr_pipe, param_grid, scoring="r2", cv=5, n_jobs=1)
    grid_sigma2 = GridSearchCV(svr_pipe, param_grid, scoring="r2", cv=5, n_jobs=1)
    grid_d50.fit(Xtr_d50, y_train_d50)
    grid_sigma2.fit(Xtr_sigma2, y_train_sigma2)
    pred_svr_d50 = grid_d50.best_estimator_.predict(Xte_d50)
    pred_svr_sigma2 = grid_sigma2.best_estimator_.predict(Xte_sigma2)

    xgb_params = dict(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=1,
    )
    xgb_d50 = XGBRegressor(**xgb_params).fit(Xtr_d50, y_train_d50)
    xgb_sigma2 = XGBRegressor(**xgb_params).fit(Xtr_sigma2, y_train_sigma2)
    pred_xgb_d50 = xgb_d50.predict(Xte_d50)
    pred_xgb_sigma2 = xgb_sigma2.predict(Xte_sigma2)

    models = [
        ("Random Forest", pred_rf_d50, pred_rf_sigma2),
        ("SVR", pred_svr_d50, pred_svr_sigma2),
        ("XGBoost", pred_xgb_d50, pred_xgb_sigma2),
    ]
    targets = {
        "d50": (y_test_d50, [m[1] for m in models]),
        "sigma2": (y_test_sigma2, [m[2] for m in models]),
    }
    metric_rows = []
    for target, (actual, predictions) in targets.items():
        for (name, _, _), predicted in zip(models, predictions):
            metric_rows.append({
                "target": target,
                "model": name,
                "R2": r2_score(actual, predicted),
                "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
                "MAE": mean_absolute_error(actual, predicted),
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "model_metrics.csv", index=False)

    predictions = pd.DataFrame({
        "d50_actual": y_test_d50.to_numpy(),
        "sigma2_actual": y_test_sigma2.to_numpy(),
        "d50_random_forest": pred_rf_d50,
        "sigma2_random_forest": pred_rf_sigma2,
        "d50_svr": pred_svr_d50,
        "sigma2_svr": pred_svr_sigma2,
        "d50_xgboost": pred_xgb_d50,
        "sigma2_xgboost": pred_xgb_sigma2,
    })
    predictions.to_csv(RESULTS / "test_predictions.csv", index=False)

    metadata = pd.DataFrame([{
        "original_rows": len(source),
        "complete_rows": len(source.drop(columns=["time"], errors="ignore").dropna()),
        "filtered_rows": len(data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "split": "train_test_split(test_size=0.30, random_state=42)",
        "fixed_features_used": ", ".join(TOP_FEATURES),
        "target_filtering": "remove rows at/above each full-data 99th percentile before splitting",
        "svr_best_params_d50": str(grid_d50.best_params_),
        "svr_best_params_sigma2": str(grid_sigma2.best_params_),
        "rf_oob_r2_d50": rf_d50.oob_score_,
        "rf_oob_r2_sigma2": rf_sigma2.oob_score_,
    }])
    metadata.to_csv(RESULTS / "run_metadata.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target, label in zip(axes, TARGETS, ["d50", "sigma²"]):
        part = metrics[metrics["target"] == target]
        ax.bar(part["model"], part["R2"], color=["#2563eb", "#14b8a6", "#f97316"])
        ax.set_title(f"{label}: test-set R²")
        ax.set_ylabel("R² (higher is better)")
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_comparison.png", dpi=180)
    plt.close(fig)

    print(f"Rows: original={len(source)}, complete={len(source.drop(columns=['time'], errors='ignore').dropna())}, filtered={len(data)}")
    print(f"Split: train={len(X_train)}, test={len(X_test)} (70/30, random_state=42)")
    print(f"Fixed features from notebook: {TOP_FEATURES}")
    print(f"SVR best parameters: d50={grid_d50.best_params_}; sigma2={grid_sigma2.best_params_}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved results to {RESULTS}")


if __name__ == "__main__":
    main()
