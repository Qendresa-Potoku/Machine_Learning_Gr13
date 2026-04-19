from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def _print_training_step(step_no: int, title: str) -> None:
    print("\n" + "=" * 60)
    print(f"PHASE 2 - STEP {step_no}: {title}")
    print("=" * 60)


def evaluate_regression_outlier_experiments(
    df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    random_state: int = 42,
) -> dict:
    """Train and compare regression models with and without outlier handling.

    Experiment A: Train using all available rows.
    Experiment B: Train after removing suspicious/invalid rows and target IQR/Z-score extremes.

    Evaluation is done on the same untouched holdout test set for fair comparison.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    if not np.issubdtype(df[target_col].dtype, np.number):
        raise ValueError("Regression evaluation requires a numeric target column.")

    model_dir = output_dir / "model_evaluation"
    model_dir.mkdir(parents=True, exist_ok=True)

    y = df[target_col].astype(float)
    feature_df = df.drop(columns=[target_col]).copy()

    outlier_labels = None
    if "outlier_type" in feature_df.columns:
        outlier_labels = feature_df["outlier_type"].astype(str).copy()
        feature_df = feature_df.drop(columns=["outlier_type"])

    numeric_features = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric_features.empty:
        raise ValueError("No numeric features available for model training.")

    X_train, X_test, y_train, y_test = train_test_split(
        numeric_features,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    outlier_train = None
    if outlier_labels is not None:
        outlier_train = outlier_labels.loc[X_train.index]

    baseline_model = RandomForestRegressor(
        n_estimators=350,
        random_state=random_state,
        n_jobs=-1,
    )
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_metrics = _regression_metrics(y_test, baseline_pred)

    q1 = float(y_train.quantile(0.25))
    q3 = float(y_train.quantile(0.75))
    iqr = q3 - q1
    iqr_low = q1 - 1.5 * iqr
    iqr_high = q3 + 1.5 * iqr
    iqr_mask = (y_train >= iqr_low) & (y_train <= iqr_high)

    std = float(y_train.std(ddof=0))
    if std <= 0:
        z_mask = pd.Series(True, index=y_train.index)
    else:
        z_scores = ((y_train - float(y_train.mean())) / std).abs()
        z_mask = z_scores <= 3.0

    contextual_mask = pd.Series(True, index=y_train.index)
    if outlier_train is not None:
        contextual_mask = ~outlier_train.isin(["invalid", "suspicious"])

    keep_mask = iqr_mask & z_mask & contextual_mask

    X_train_clean = X_train.loc[keep_mask]
    y_train_clean = y_train.loc[keep_mask]

    cleaned_model = RandomForestRegressor(
        n_estimators=350,
        random_state=random_state,
        n_jobs=-1,
    )
    cleaned_model.fit(X_train_clean, y_train_clean)
    cleaned_pred = cleaned_model.predict(X_test)
    cleaned_metrics = _regression_metrics(y_test, cleaned_pred)

    metrics_df = pd.DataFrame(
        [
            {"experiment": "with_outliers", **baseline_metrics},
            {"experiment": "without_outliers", **cleaned_metrics},
        ]
    )
    metrics_df.to_csv(model_dir / "regression_outlier_comparison.csv", index=False)

    plt.figure(figsize=(8, 5))
    x = np.arange(3)
    width = 0.35
    metric_names = ["mae", "rmse", "r2"]
    baseline_vals = [baseline_metrics[m] for m in metric_names]
    cleaned_vals = [cleaned_metrics[m] for m in metric_names]

    plt.bar(x - width / 2, baseline_vals, width=width, label="with_outliers", color="#4C78A8")
    plt.bar(x + width / 2, cleaned_vals, width=width, label="without_outliers", color="#54A24B")
    plt.xticks(x, [m.upper() for m in metric_names])
    plt.title("Model Performance: With vs Without Outliers")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "metrics_with_vs_without_outliers.png", dpi=300)
    plt.close()

    removed_total = int((~keep_mask).sum())
    removed_by_iqr = int((~iqr_mask).sum())
    removed_by_z = int((~z_mask).sum())
    removed_by_context = int((~contextual_mask).sum())

    summary = {
        "task": "regression",
        "target": target_col,
        "test_size": 0.2,
        "features_used": int(numeric_features.shape[1]),
        "rows": {
            "train_before": int(len(X_train)),
            "train_after_outlier_handling": int(len(X_train_clean)),
            "test": int(len(X_test)),
            "removed_from_train_total": removed_total,
        },
        "outlier_detection_methods": {
            "iqr": {
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "low": round(iqr_low, 4),
                "high": round(iqr_high, 4),
                "removed_count": removed_by_iqr,
            },
            "z_score": {
                "threshold": 3.0,
                "removed_count": removed_by_z,
            },
            "contextual_outlier_type": {
                "applied": bool(outlier_train is not None),
                "removed_labels": ["invalid", "suspicious"],
                "removed_count": removed_by_context,
            },
        },
        "metrics": {
            "with_outliers": baseline_metrics,
            "without_outliers": cleaned_metrics,
            "delta_without_minus_with": {
                "mae": round(cleaned_metrics["mae"] - baseline_metrics["mae"], 4),
                "rmse": round(cleaned_metrics["rmse"] - baseline_metrics["rmse"], 4),
                "r2": round(cleaned_metrics["r2"] - baseline_metrics["r2"], 4),
            },
        },
        "artifacts": {
            "metrics_csv": str(model_dir / "regression_outlier_comparison.csv"),
            "metrics_plot": str(model_dir / "metrics_with_vs_without_outliers.png"),
        },
    }

    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)
    print("Experiment A (with outliers):", baseline_metrics)
    print("Experiment B (without outliers):", cleaned_metrics)
    print("Training rows removed in Experiment B:", removed_total)
    print("Saved model evaluation artifacts to:", model_dir)

    return summary


def train_final_regression_model(
    df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    random_state: int = 42,
) -> dict:
    """Train the final regression model and export artifacts.

    The pipeline keeps all rows and applies sample weighting using outlier labels:
    - normal / valid: full weight
    - suspicious: reduced weight
    - invalid: strongly reduced weight
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    if not np.issubdtype(df[target_col].dtype, np.number):
        raise ValueError("Final training requires a numeric target column.")

    train_dir = output_dir / "final_model"
    train_dir.mkdir(parents=True, exist_ok=True)

    _print_training_step(1, "Input Validation")
    print(f"Rows received: {len(df):,}")
    print(f"Columns received: {len(df.columns)}")
    print(f"Target column: {target_col}")

    y = df[target_col].astype(float)
    features_df = df.drop(columns=[target_col]).copy()

    outlier_labels = None
    if "outlier_type" in features_df.columns:
        outlier_labels = features_df["outlier_type"].astype(str).copy()
        features_df = features_df.drop(columns=["outlier_type"])

    X = features_df.select_dtypes(include=[np.number]).copy()
    if X.empty:
        raise ValueError("No numeric features found for final model training.")

    _print_training_step(2, "Train/Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Numeric features used: {X.shape[1]}")

    sample_weight = np.ones(len(X_train), dtype=float)
    weight_distribution: dict[str, int] = {}
    if outlier_labels is not None:
        train_labels = outlier_labels.loc[X_train.index].fillna("normal")
        weight_map = {
            "normal": 1.0,
            "valid": 1.0,
            "suspicious": 0.65,
            "invalid": 0.35,
        }
        sample_weight = train_labels.map(lambda x: weight_map.get(str(x), 1.0)).astype(float).to_numpy()
        weight_distribution = {k: int(v) for k, v in train_labels.value_counts().to_dict().items()}

    _print_training_step(3, "Model Training")
    model = RandomForestRegressor(
        n_estimators=450,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    print("Model: RandomForestRegressor")
    print("Training complete.")

    _print_training_step(4, "Evaluation")
    y_pred = model.predict(X_test)
    test_metrics = _regression_metrics(y_test, y_pred)
    print("Test metrics:", test_metrics)

    _print_training_step(5, "Artifacts Export")
    predictions_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "abs_error": (y_test - y_pred).abs(),
        },
        index=y_test.index,
    ).reset_index(drop=True)
    predictions_path = train_dir / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    fi_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    fi_path = train_dir / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)

    plt.figure(figsize=(10, 6))
    top_fi = fi_df.head(15).iloc[::-1]
    plt.barh(top_fi["feature"], top_fi["importance"], color="#4C78A8")
    plt.title("Top 15 Feature Importances (Final Model)")
    plt.xlabel("importance")
    plt.ylabel("feature")
    plt.tight_layout()
    fi_plot_path = train_dir / "feature_importance_top15.png"
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.6, s=24, color="#54A24B")
    min_axis = float(min(y_test.min(), y_pred.min()))
    max_axis = float(max(y_test.max(), y_pred.max()))
    plt.plot([min_axis, max_axis], [min_axis, max_axis], "r--", linewidth=1.2)
    plt.title("Actual vs Predicted Delay (Test Set)")
    plt.xlabel("actual delay_min")
    plt.ylabel("predicted delay_min")
    plt.tight_layout()
    scatter_path = train_dir / "actual_vs_predicted.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    model_path = train_dir / "final_random_forest_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    print(f"Saved model: {model_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved feature importance CSV: {fi_path}")
    print(f"Saved feature importance plot: {fi_plot_path}")
    print(f"Saved actual-vs-predicted plot: {scatter_path}")

    summary = {
        "target": target_col,
        "rows": {
            "total": int(len(X)),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
        },
        "features_used": int(X.shape[1]),
        "model": {
            "name": "RandomForestRegressor",
            "params": {
                "n_estimators": 450,
                "max_depth": None,
                "min_samples_leaf": 2,
                "random_state": random_state,
            },
        },
        "sample_weighting": {
            "applied": bool(outlier_labels is not None),
            "label_distribution_train": weight_distribution,
            "weights": {
                "normal": 1.0,
                "valid": 1.0,
                "suspicious": 0.65,
                "invalid": 0.35,
            },
        },
        "test_metrics": test_metrics,
        "artifacts": {
            "model": str(model_path),
            "predictions_csv": str(predictions_path),
            "feature_importance_csv": str(fi_path),
            "feature_importance_plot": str(fi_plot_path),
            "actual_vs_predicted_plot": str(scatter_path),
        },
    }

    return summary
