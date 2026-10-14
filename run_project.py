from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic_data_improved.csv"
RESULTS = ROOT / "results"
FEATURES = ["S", "ub", "np", "T", "a676_a650", "a450_a676", "chl_a", "u"]
TOP_FEATURES = ["S", "ub", "np", "T"]
TARGETS = ["d50", "sigma2"]


def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from xgboost import XGBRegressor

    RESULTS.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    if "time" in data.columns:
        data = data.drop(columns=["time"])
    data = data.dropna().copy()
    d50_limit = data["d50"].quantile(0.99)
    sigma2_limit = data["sigma2"].quantile(0.99)
    data = data[(data["d50"] < d50_limit) & (data["sigma2"] < sigma2_limit)].copy()

    X = data[FEATURES]
    y_d50 = data["d50"]
    y_sigma2 = data["sigma2"]
    X_train, X_test, y_train_d50, y_test_d50 = train_test_split(
        X, y_d50, test_size=0.30, random_state=42
    )
    X_train2, X_test2, y_train_sigma2, y_test_sigma2 = train_test_split(
        X, y_sigma2, test_size=0.30, random_state=42
    )

    # The original notebook explicitly selects these four predictors after
    # inspecting Random Forest feature importance on the training data.
    importance_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    importance_model.fit(X_train, y_train_d50)
    importance = pd.DataFrame({"feature": FEATURES, "importance": importance_model.feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(RESULTS / "feature_importance.csv", index=False)
    Xtr_d50, Xte_d50 = X_train[TOP_FEATURES], X_test[TOP_FEATURES]
    Xtr_sigma2, Xte_sigma2 = X_train2[TOP_FEATURES], X_test2[TOP_FEATURES]

    rf_d50 = RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_split=2,
                                   min_samples_leaf=1, random_state=42, oob_score=True, n_jobs=-1)
    rf_sigma2 = RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1, random_state=42, oob_score=True, n_jobs=-1)
    rf_d50.fit(Xtr_d50, y_train_d50)
    rf_sigma2.fit(Xtr_sigma2, y_train_sigma2)

    svr_pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    param_grid = {"svr__C": [1, 10, 100, 500, 1000, 2000],
                  "svr__epsilon": [0.1, 0.5, 1, 2]}
    grid_d50 = GridSearchCV(svr_pipe, param_grid, scoring="r2", cv=5, n_jobs=1)
    grid_sigma2 = GridSearchCV(svr_pipe, param_grid, scoring="r2", cv=5, n_jobs=1)
    grid_d50.fit(Xtr_d50, y_train_d50)
    grid_sigma2.fit(Xtr_sigma2, y_train_sigma2)

    xgb_params = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      objective="reg:squarederror", random_state=42, n_jobs=1)
    xgb_d50 = XGBRegressor(**xgb_params)
    xgb_sigma2 = XGBRegressor(**xgb_params)
    xgb_d50.fit(Xtr_d50, y_train_d50)
    xgb_sigma2.fit(Xtr_sigma2, y_train_sigma2)

    models = {
        "d50": {"Random Forest": (rf_d50, Xte_d50, y_test_d50),
                "SVR": (grid_d50.best_estimator_, Xte_d50, y_test_d50),
                "XGBoost": (xgb_d50, Xte_d50, y_test_d50)},
        "sigma2": {"Random Forest": (rf_sigma2, Xte_sigma2, y_test_sigma2),
                   "SVR": (grid_sigma2.best_estimator_, Xte_sigma2, y_test_sigma2),
                   "XGBoost": (xgb_sigma2, Xte_sigma2, y_test_sigma2)},
    }
    rows = []
    predictions = pd.DataFrame(index=range(len(y_test_d50)))
    for target, fitted in models.items():
        predictions[f"{target}_actual"] = fitted["Random Forest"][2].to_numpy()
        for model_name, (model, X_test_part, y_test) in fitted.items():
            pred = model.predict(X_test_part)
            rows.append({"target": target, "model": model_name,
                         "R2": r2_score(y_test, pred),
                         "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                         "MAE": mean_absolute_error(y_test, pred)})
            predictions[f"{target}_{model_name.lower().replace(' ', '_')}"] = pred
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "model_metrics.csv", index=False)
    predictions.to_csv(RESULTS / "test_predictions.csv", index=False)
    metadata = pd.DataFrame([{
        "original_rows": 1648,
        "rows_after_dropna": int(pd.read_csv(DATA).drop(columns=["time"], errors="ignore").dropna().shape[0]),
        "rows_after_outlier_filter": len(data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "split": "train_test_split(test_size=0.30, random_state=42)",
        "features_used": ", ".join(TOP_FEATURES),
        "svr_best_params_d50": str(grid_d50.best_params_),
        "svr_best_params_sigma2": str(grid_sigma2.best_params_),
        "rf_oob_r2_d50": rf_d50.oob_score_,
        "rf_oob_r2_sigma2": rf_sigma2.oob_score_,
    }])
    metadata.to_csv(RESULTS / "run_metadata.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target, label in zip(axes, TARGETS, ["d50", "sigma²"]):
        part = metrics[metrics.target == target]
        ax.bar(part.model, part.R2, color=["#2563eb", "#14b8a6", "#f97316"])
        ax.set_title(f"{label}: test-set R²")
        ax.set_ylabel("R² (higher is better)")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Rows: original={len(pd.read_csv(DATA))}, complete={int(pd.read_csv(DATA).drop(columns=['time'], errors='ignore').dropna().shape[0])}, after outlier filter={len(data)}")
    print(f"Split: train={len(X_train)}, test={len(X_test)} (70/30, random_state=42); features={TOP_FEATURES}")
    print(f"SVR best parameters: d50={grid_d50.best_params_}; sigma2={grid_sigma2.best_params_}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved results to {RESULTS}")


if __name__ == "__main__":
    main()
