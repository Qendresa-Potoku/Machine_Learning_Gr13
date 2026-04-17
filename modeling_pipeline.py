from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
DATA_PATH = Path("outputs/cleaned_dataset_regression.csv")
OUTLIER_TYPES = ["normal", "valid", "suspicious"]
EXTREME_TYPES = ["valid", "suspicious"]


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_cleaned_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    return df


def infer_outlier_type(df: pd.DataFrame) -> pd.Series:
    if "delay_min" not in df.columns:
        raise ValueError("delay_min is required to infer outlier_type.")

    delay_series = df["delay_min"]
    q1 = float(delay_series.quantile(0.25))
    q3 = float(delay_series.quantile(0.75))
    iqr = q3 - q1
    iqr_high = q3 + 1.5 * iqr
    q99 = float(delay_series.quantile(0.99))

    rush_col = df["is_rush_hour"] if "is_rush_hour" in df.columns else pd.Series(0, index=df.index)
    weather_col = df["is_bad_weather"] if "is_bad_weather" in df.columns else pd.Series(0, index=df.index)

    speed_invalid_mask = pd.Series(False, index=df.index)
    if "speed_normal" in df.columns:
        speed_invalid_mask = df["speed_normal"] <= 0

    duration_invalid_mask = pd.Series(False, index=df.index)
    if "duration_normal_min" in df.columns:
        duration_invalid_mask = df["duration_normal_min"] == 0

    invalid_mask = (delay_series < -3) | speed_invalid_mask | duration_invalid_mask
    high_delay_mask = (delay_series > iqr_high) | (delay_series >= q99)

    outlier_type = pd.Series("normal", index=df.index, dtype="object")
    outlier_type.loc[invalid_mask] = "invalid"
    outlier_type.loc[high_delay_mask & ((rush_col == 1) | (weather_col == 1)) & (~invalid_mask)] = "valid"
    outlier_type.loc[high_delay_mask & (rush_col == 0) & (weather_col == 0) & (~invalid_mask)] = "suspicious"

    return outlier_type


def ensure_outlier_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "outlier_type" not in out.columns:
        print("outlier_type column not found; inferring it from the cleaned dataset rules.")
        out["outlier_type"] = infer_outlier_type(out)

    return out


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "outlier_type" not in df.columns:
        raise ValueError("outlier_type column is required before filtering invalid rows.")

    filtered = df[df["outlier_type"] != "invalid"].copy().reset_index(drop=True)
    return filtered


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_cols if col not in {"delay_min"}]

    if "outlier_type" in features:
        features.remove("outlier_type")

    if not features:
        raise ValueError("No numeric features found for modeling.")

    return features


def get_classifier_feature_columns(df: pd.DataFrame) -> list[str]:
    # Contextual-only features for non-trivial routing.
    base_contextual = [
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        "is_rush_hour",
        "is_bad_weather",
        "temperature",
        "wind",
        "rain",
    ]

    contextual = [c for c in base_contextual if c in df.columns]
    route_features = [c for c in df.columns if c.startswith("route_")]

    classifier_features = contextual + route_features
    if not classifier_features:
        raise ValueError("No contextual features found for classifier training.")

    return classifier_features


def add_target_class(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_class"] = np.where(out["outlier_type"] == "normal", 0, 1)
    return out


def prepare_xy(df: pd.DataFrame, feature_columns: Iterable[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df.loc[:, list(feature_columns)].copy()
    y = df["delay_min"].copy()
    return X, y


def train_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)
    return model


def train_classifier(X_train: pd.DataFrame, y_class_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_class_train)
    return model


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
    }


def print_metrics(title: str, metrics: dict[str, float], row_count: int | None = None) -> None:
    suffix = f" ({row_count:,} rows)" if row_count is not None else ""
    print(f"\n{title}{suffix}")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²  : {metrics['r2']:.4f}")


def evaluate_by_outlier_type(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series, outlier_test: pd.Series) -> dict[str, dict[str, float]]:
    group_metrics: dict[str, dict[str, float]] = {}

    for outlier_type in OUTLIER_TYPES:
        mask = outlier_test == outlier_type
        if mask.sum() == 0:
            print(f"\n{outlier_type.title()} subset: no rows in test split")
            continue

        metrics = evaluate_predictions(y_test.loc[mask], model.predict(X_test.loc[mask]))
        group_metrics[outlier_type] = metrics
        print_metrics(f"Baseline on {outlier_type}", metrics, int(mask.sum()))

    return group_metrics


def split_for_modeling(df: pd.DataFrame, feature_columns: list[str]) -> dict:
    X, y = prepare_xy(df, feature_columns)
    y_class = df["target_class"].copy()
    outlier_type = df["outlier_type"].copy()

    X_train, X_test, y_train, y_test, y_class_train, y_class_test, outlier_train, outlier_test = train_test_split(
        X,
        y,
        y_class,
        outlier_type,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_class,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_class_train": y_class_train,
        "y_class_test": y_class_test,
        "outlier_train": outlier_train,
        "outlier_test": outlier_test,
    }


def fit_baseline_model(split_data: dict) -> dict:
    print_section("BASELINE - SINGLE RANDOM FOREST")

    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]
    outlier_test = split_data["outlier_test"]

    model = train_regressor(X_train, y_train)
    y_pred = model.predict(X_test)

    overall_metrics = evaluate_predictions(y_test, y_pred)
    print_metrics("Baseline overall test performance", overall_metrics, len(X_test))
    group_metrics = evaluate_by_outlier_type(model, X_test, y_test, outlier_test)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "outlier_test": outlier_test,
        "predictions": y_pred,
        "metrics": overall_metrics,
        "group_metrics": group_metrics,
    }


def predict_pipeline(
    classifier: RandomForestClassifier,
    model_normal: RandomForestRegressor,
    model_extreme: RandomForestRegressor,
    X_test_classifier: pd.DataFrame,
    X_test_regression: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_class = classifier.predict(X_test_classifier)

    final_predictions = np.empty(len(X_test_regression), dtype=float)
    normal_mask = predicted_class == 0
    extreme_mask = predicted_class == 1

    if normal_mask.any():
        final_predictions[normal_mask] = model_normal.predict(X_test_regression.loc[normal_mask])
    if extreme_mask.any():
        final_predictions[extreme_mask] = model_extreme.predict(X_test_regression.loc[extreme_mask])

    return final_predictions, predicted_class


def train_dual_model(split_data: dict) -> dict:
    print_section("IMPROVED DUAL MODEL - CLASSIFIER ROUTING")

    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]
    y_class_train = split_data["y_class_train"]
    y_class_test = split_data["y_class_test"]
    outlier_train = split_data["outlier_train"]
    outlier_test = split_data["outlier_test"]

    print_section("STEP 1 - CLASSIFICATION TARGET")
    print("target_class = 0 for normal, 1 for valid/suspicious")
    print(f"Training class distribution: {y_class_train.value_counts().to_dict()}")

    print_section("STEP 2 - TRAIN CLASSIFIER")
    classifier_feature_columns = get_classifier_feature_columns(X_train)
    print(f"Classifier contextual features used: {len(classifier_feature_columns)}")
    classifier = train_classifier(X_train[classifier_feature_columns], y_class_train)

    y_class_pred = classifier.predict(X_test[classifier_feature_columns])
    classifier_accuracy = accuracy_score(y_class_test, y_class_pred)
    classifier_precision = precision_score(y_class_test, y_class_pred, zero_division=0)
    classifier_recall = recall_score(y_class_test, y_class_pred, zero_division=0)
    classifier_f1 = f1_score(y_class_test, y_class_pred, zero_division=0)
    classifier_cm = confusion_matrix(y_class_test, y_class_pred)

    print("Classifier evaluation:")
    print(f"  Accuracy : {classifier_accuracy:.4f}")
    print(f"  Precision: {classifier_precision:.4f}")
    print(f"  Recall   : {classifier_recall:.4f}")
    print(f"  F1-score : {classifier_f1:.4f}")
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(classifier_cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_string())

    print_section("STEP 3 - TRAIN REGRESSORS")
    train_df = X_train.copy()
    train_df["delay_min"] = y_train.values
    train_df["outlier_type"] = outlier_train.values

    df_normal = train_df[train_df["outlier_type"] == "normal"].copy()
    df_extreme = train_df[train_df["outlier_type"].isin(EXTREME_TYPES)].copy()

    if df_normal.empty:
        raise ValueError("Normal training subset is empty; cannot train Model A.")
    if df_extreme.empty:
        raise ValueError("Extreme training subset is empty; cannot train Model B.")

    model_normal = train_regressor(df_normal.drop(columns=["delay_min", "outlier_type"]), df_normal["delay_min"])
    model_extreme = train_regressor(df_extreme.drop(columns=["delay_min", "outlier_type"]), df_extreme["delay_min"])

    print_section("STEP 4 - PREDICTION PIPELINE")
    final_predictions, predicted_class = predict_pipeline(
        classifier,
        model_normal,
        model_extreme,
        X_test[classifier_feature_columns],
        X_test,
    )

    overall_metrics = evaluate_predictions(y_test, final_predictions)
    print_metrics("Dual-model overall test performance", overall_metrics, len(X_test))
    print(f"\nPredicted routing counts: normal={int((predicted_class == 0).sum()):,}, extreme={int((predicted_class == 1).sum()):,}")

    return {
        "classifier": classifier,
        "model_normal": model_normal,
        "model_extreme": model_extreme,
        "X_test": X_test,
        "y_test": y_test,
        "outlier_test": outlier_test,
        "predictions": final_predictions,
        "metrics": overall_metrics,
        "predicted_class": predicted_class,
        "classifier_features": classifier_feature_columns,
    }


def compare_models(baseline_metrics: dict[str, float], dual_metrics: dict[str, float]) -> None:
    print_section("STEP 5 - COMPARISON")

    comparison = pd.DataFrame(
        [
            {"model": "Single Random Forest", **baseline_metrics},
            {"model": "Dual-model system", **dual_metrics},
        ]
    )

    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    mae_delta = baseline_metrics["mae"] - dual_metrics["mae"]
    rmse_delta = baseline_metrics["rmse"] - dual_metrics["rmse"]
    r2_delta = dual_metrics["r2"] - baseline_metrics["r2"]

    print("\nImprovement vs baseline:")
    print(f"  MAE  improvement: {mae_delta:.4f}")
    print(f"  RMSE improvement: {rmse_delta:.4f}")
    print(f"  R²   improvement: {r2_delta:.4f}")


def main() -> None:
    print_section("TRAFFIC DELAY MODELING PIPELINE")
    df = load_cleaned_dataset(DATA_PATH)
    df = ensure_outlier_type(df)
    df = remove_invalid_rows(df)
    df = add_target_class(df)

    feature_columns = get_feature_columns(df)
    split_data = split_for_modeling(df, feature_columns)

    print(f"Loaded dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Outlier distribution: {df['outlier_type'].value_counts(dropna=False).to_dict()}")

    baseline_result = fit_baseline_model(split_data)
    dual_result = train_dual_model(split_data)

    compare_models(baseline_result["metrics"], dual_result["metrics"])


if __name__ == "__main__":
    main()