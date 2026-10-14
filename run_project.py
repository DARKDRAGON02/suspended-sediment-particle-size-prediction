from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic_data_improved.csv"
RESULTS = ROOT / "results"
FEATURES = ["S", "ub", "np", "T", "a676_a650", "a450_a676", "chl_a", "u"]
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
    frame = pd.read_csv(DATA)
    if "time" in frame.columns:
        frame = frame.drop(columns=["time"])
    frame = frame.dropna().reset_index(drop=True)
    X = frame[FEATURES]
    y = frame[TARGETS]

    # Split before learning any feature ranking or model parameters.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    importance_model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    importance_model.fit(X_train, y_train["d50"])
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": importance_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance.to_csv(RESULTS / "feature_importance.csv", index=False)
    top_features = importance.head(4)["feature"].tolist()
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    rf_models = {
        target: RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, random_state=42, oob_score=True, n_jobs=-1
        ) for target in TARGETS
    }
    for target in TARGETS:
        rf_models[target].fit(X_train, y_train[target])

    svr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf")),
    ])
    param_grid = {
        "svr__C": [1, 10, 100, 500, 1000, 2000],
        "svr__epsilon": [0.1, 0.5, 1, 2],
    }
    svr_models = {}
    for target in TARGETS:
        search = GridSearchCV(
            svr_pipe, param_grid, scoring="r2", cv=5, n_jobs=1
        )
        search.fit(X_train, y_train[target])
        svr_models[target] = search

    xgb_params = dict(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=1,
    )
    xgb_models = {target: XGBRegressor(**xgb_params) for target in TARGETS}
    for target in TARGETS:
        xgb_models[target].fit(X_train, y_train[target])

    fitted = {
        "Random Forest": rf_models,
        "SVR": {target: svr_models[target].best_estimator_ for target in TARGETS},
        "XGBoost": xgb_models,
    }
    rows = []
    predictions = pd.DataFrame({
        f"{target}_actual": y_test[target].to_numpy() for target in TARGETS
    })
    for model_name, target_models in fitted.items():
        for target in TARGETS:
            pred = target_models[target].predict(X_test)
            predictions[f"{target}_{model_name.lower().replace(' ', '_')}"] = pred
            rows.append({
                "target": target,
                "model": model_name,
                "R2": r2_score(y_test[target], pred),
                "RMSE": np.sqrt(mean_squared_error(y_test[target], pred)),
                "MAE": mean_absolute_error(y_test[target], pred),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "model_metrics.csv", index=False)
    predictions.to_csv(RESULTS / "test_predictions.csv", index=False)

    metadata = pd.DataFrame([{
        "original_rows": len(pd.read_csv(DATA)),
        "complete_rows": len(frame),
        "train_rows": len(X_train) + 0,  # set below from split ratio
        "test_rows": len(X_test),
        "split": "train_test_split(test_size=0.30, random_state=42)",
        "features_selected_on_training_data": ", ".join(top_features),
        "outlier_filtering": "none; all complete rows retained",
        "svr_best_params_d50": str(svr_models["d50"].best_params_),
        "svr_best_params_sigma2": str(svr_models["sigma2"].best_params_),
        "rf_oob_r2_d50": rf_models["d50"].oob_score_,
        "rf_oob_r2_sigma2": rf_models["sigma2"].oob_score_,
    }])
    # Replace the training-row entry after feature selection, which preserves row counts.
    metadata.loc[0, "train_rows"] = len(y_train)
    metadata.to_csv(RESULTS / "run_metadata.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target, label in zip(axes, TARGETS, ["d50", "sigma²"]):
        part = metrics[metrics.target == target]
        ax.bar(part.model, part.R2, color=["#2563eb", "#14b8a6", "#f97316"])
        ax.set_title(f"{label}: test-set R²")
        ax.set_ylabel("R² (higher is better)")
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Rows: original={len(pd.read_csv(DATA))}, complete={len(frame)}; no target-based filtering.")
    print(f"Split: train={len(y_train)}, test={len(y_test)} (70/30, random_state=42)")
    print(f"Top 4 features selected from training data: {top_features}")
    print(f"SVR best parameters: d50={svr_models['d50'].best_params_}; sigma2={svr_models['sigma2'].best_params_}")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved results to {RESULTS}")


if __name__ == "__main__":
    main()
