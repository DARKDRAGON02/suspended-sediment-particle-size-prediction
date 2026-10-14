from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic_data_improved.csv"
RESULTS = ROOT / "results"
TARGETS = ["d50", "sigma2"]
FEATURES = ["S", "ub", "np", "T", "a676_a650", "a450_a676", "chl_a", "u"]


def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from xgboost import XGBRegressor

    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA, parse_dates=["time"])
    missing_targets = frame[TARGETS].isna().any(axis=1)
    frame = frame.loc[~missing_targets].sort_values("time").reset_index(drop=True)
    X = frame[FEATURES]
    y = frame[TARGETS]
    split = int(len(frame) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    rows = []
    predictions = pd.DataFrame({"time": frame["time"].iloc[split:].values})

    for target in TARGETS:
        y_train, y_test = y[target].iloc[:split], y[target].iloc[split:]
        estimators = {
            "Random Forest": make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
            ),
            "SVR": make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                SVR(C=100, epsilon=5, kernel="rbf"),
            ),
            "XGBoost": make_pipeline(
                SimpleImputer(strategy="median"),
                XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.04,
                             subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
                             random_state=42, n_jobs=-1),
            ),
        }
        for model_name, model in estimators.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rows.append({"target": target, "model": model_name,
                         "R2": r2_score(y_test, pred),
                         "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                         "MAE": mean_absolute_error(y_test, pred)})
            predictions[f"{target}_{model_name.lower().replace(' ', '_')}"] = pred
        predictions[f"{target}_actual"] = y_test.to_numpy()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "model_metrics.csv", index=False)
    predictions.to_csv(RESULTS / "test_predictions.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target, label in zip(axes, TARGETS, ["Median particle diameter d50", "PSD variance sigma2"]):
        subset = metrics[metrics.target == target]
        ax.bar(subset.model, subset.R2, color=["#2563eb", "#14b8a6", "#f97316"])
        ax.set_title(f"{label}: test-set R²")
        ax.set_ylabel("R² (higher is better)")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_comparison.png", dpi=180)
    plt.close(fig)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved results to {RESULTS}")


if __name__ == "__main__":
    main()
