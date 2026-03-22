import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from math import pi, sin, cos
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skewness_utils import analyze_skewness_with_graphics
from visualizations import visualize


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def choose_dataset_scope(df: pd.DataFrame, sample_n: int = 5000) -> pd.DataFrame:
    print_section("DATASET SCOPE SELECTION")
    print(f"\nOptions:")
    print(f"  [1] Process FULL dataset ({len(df):,} rows)")
    print(f"  [2] Process SAMPLE ({min(sample_n, len(df)):,} rows)")
    
    while True:
        choice = input("\nEnter choice [1/2] (default: 2 - sample): ").strip().lower()
        
        if choice in ["1", "full", "f"]:
            print(f"\n>> Processing full dataset: {len(df):,} rows")
            return df
        elif choice in ["2", "sample", "s", ""]:
            n = min(sample_n, len(df))
            sampled = df.sample(n=n, random_state=42).reset_index(drop=True)
            print(f"\n>> Processing sample: {n:,} rows (random)")
            return sampled
        else:
            print("Invalid choice. Please enter '1' (full) or '2' (sample).")


def analyze_data_types(df: pd.DataFrame) -> dict:
    print_section("1) DATA TYPES")

    groups = {
        "numeric": [
            "distance_km",
            "duration_normal_min",
            "duration_traffic_min",
            "delay_min",
            "temperature",
            "wind",
        ],
        "categorical": ["origin", "destination"],
        "temporal": ["timestamp"],
        "binary": ["is_weekend", "rain"],
        "discrete": ["hour", "day_of_week"],
    }

    for group_name, columns in groups.items():
        existing = [c for c in columns if c in df.columns]
        missing = [c for c in columns if c not in df.columns]
        print(f"\n{group_name.upper()}:")
        print(f"  existing: {existing}")
        if missing:
            print(f"  missing: {missing}")

    print("\nCurrent dtype of each column:")
    for col in df.columns:
        print(f"  - {col:25} -> {df[col].dtype}")

    return groups


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print_section("2) FEATURE ENGINEERING")
    out = df.copy()

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["hour"] = out["timestamp"].dt.hour
        out["day_of_week"] = out["timestamp"].dt.dayofweek
        out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

    if {"origin", "destination"}.issubset(out.columns):
        out["route"] = out["origin"].astype(str) + " → " + out["destination"].astype(str)

    if "hour" in out.columns:
        out["is_rush_hour"] = (
            (out["hour"].isin(range(7, 10))) | (out["hour"].isin(range(16, 18)))
        ).astype(int)

    if {"rain", "wind"}.issubset(out.columns):
        out["is_bad_weather"] = ((out["rain"] > 0) | (out["wind"] > 8)).astype(int)

    if {"distance_km", "duration_normal_min"}.issubset(out.columns):
        epsilon = 1e-5
        out["speed_normal"] = out["distance_km"] / (out["duration_normal_min"] + epsilon)

    if "hour" in out.columns:
        out["hour_sin"] = np.sin(2 * pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * pi * out["hour"] / 24)

    out = out.replace([np.inf, -np.inf], np.nan)

    print("Created: hour, day_of_week, is_weekend, route, is_rush_hour, is_bad_weather, speed_normal, hour_sin, hour_cos")
    return out


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print_section("3) DATA CLEANING")
    cleaned = df.copy()

    before_rows = len(cleaned)
    before_null = int(cleaned.isna().sum().sum())

    # Fill missing values with median for numeric attributes.
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    median_imputed_cols: list[str] = []
    for col in numeric_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            median_imputed_cols.append(col)

    # Remove duplicate rows after imputation.
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    after_rows = len(cleaned)
    after_null = int(cleaned.isna().sum().sum())

    columns_to_check_iqr = [
        "delay_min",
        "speed_normal",
        "distance_km",
        "duration_normal_min",
    ]

    # Check IQR outlier counts for selected continuous columns.
    iqr_outlier_counts_selected: dict[str, int] = {}
    for col in columns_to_check_iqr:
        if col not in cleaned.columns:
            continue
        series = cleaned[col]
        if series.nunique(dropna=True) <= 10:
            iqr_outlier_counts_selected[col] = 0
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            iqr_outlier_counts_selected[col] = 0
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        iqr_outlier_counts_selected[col] = int(((series < low) | (series > high)).sum())

    # Winsorization (percentile clipping) instead of row deletion.
    winsorized_columns: dict[str, dict[str, float | int]] = {}
    for col in ["delay_min", "speed_normal"]:
        if col not in cleaned.columns:
            continue

        series = cleaned[col]
        p01 = float(series.quantile(0.01))
        p99 = float(series.quantile(0.99))
        before_clip = series.copy()
        cleaned[col] = series.clip(lower=p01, upper=p99)

        changed_count = int((before_clip != cleaned[col]).sum())
        winsorized_columns[col] = {
            "p01": round(p01, 6),
            "p99": round(p99, 6),
            "values_clipped": changed_count,
        }

    print(f"Rows before cleaning: {before_rows}")
    print(f"Rows after median-imputation+drop_duplicates: {after_rows}")
    print(f"Nulls before: {before_null}, nulls after: {after_null}")
    print(f"Median-imputed columns: {median_imputed_cols if median_imputed_cols else 'None'}")
    if iqr_outlier_counts_selected:
        print("IQR outlier counts (selected columns):")
        for col, count in iqr_outlier_counts_selected.items():
            print(f"  - {col:25}: {count}")
    if winsorized_columns:
        print("Winsorization applied (1%-99% clipping):")
        for col, info in winsorized_columns.items():
            print(
                f"  - {col:25}: clipped={info['values_clipped']}, "
                f"p01={info['p01']}, p99={info['p99']}"
            )
    print("IQR row removal: disabled")

    summary = {
        "rows_before": before_rows,
        "rows_after_impute_dedup": after_rows,
        "null_before": before_null,
        "null_after": after_null,
        "median_imputed_columns": median_imputed_cols,
        "iqr_outlier_counts_selected": iqr_outlier_counts_selected,
        "iqr_outlier_rows_removed": 0,
        "winsorized_columns": winsorized_columns,
        "rows_final": len(cleaned),
    }
    return cleaned, summary


def create_target(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, str]:
    print_section("4) TARGET")
    out = df.copy()

    if "delay_min" not in out.columns:
        raise ValueError("Column delay_min is missing. Target cannot be created.")

    task = task.lower()
    if task == "regression":
        target_col = "delay_min"
        print("Selected target: delay_min (regression)")
        return out, target_col

    if task == "classification":
        bins = [-np.inf, 3, 7, np.inf]
        labels = ["Low", "Medium", "High"]
        out["traffic_level"] = pd.cut(out["delay_min"], bins=bins, labels=labels, right=False)
        target_col = "traffic_level"
        print("Selected target: traffic_level (classification: Low/Medium/High)")
        print("Rule: delay < 3 -> Low, delay 3-7 -> Medium, delay > 7 -> High")
        return out, target_col

    raise ValueError("Task must be one of: regression or classification")

def normalize_features(df: pd.DataFrame, target_col: str, method: str = "standard") -> pd.DataFrame:
    print_section("6) NORMALIZATION")

    out = df.copy()

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()

    numeric_cols = [col for col in numeric_cols if col != target_col]
    numeric_cols = [
        col
        for col in numeric_cols
        if not col.startswith("route_") and out[col].nunique(dropna=True) > 2
    ]

    print(f"Columns to normalize: {numeric_cols}")

    if not numeric_cols:
        print("No numeric columns found for normalization.")
        return out

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        print("No normalization applied.")
        return out

    out[numeric_cols] = scaler.fit_transform(out[numeric_cols])

    print(f"Normalization applied using: {method}")
    return out


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    print_section("ENCODE FEATURES")
    out = df.copy()

    if "route" in out.columns:
        route_encoded = pd.get_dummies(
            out["route"],
            prefix="route",
            drop_first=True,
            dtype=np.int8,
        )
        out = pd.concat([out, route_encoded], axis=1)
        print(f"Created {len(route_encoded.columns)} route encoding columns")

    return out


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not needed for modeling (prevent data leakage and reduce dimensionality)."""
    print_section("DROP UNUSED COLUMNS")
    out = df.copy()

    columns_to_drop = [
        "timestamp",      # Not needed after extracting temporal features
        "origin",         # Not needed after creating route
        "destination",    # Not needed after creating route
        "route",          # Drop after one-hot encoding
        "hour",           # Not needed after sin/cos encoding
        "duration_traffic_min",  # IMPORTANT: Remove to prevent data leakage
    ]

    existing_to_drop = [col for col in columns_to_drop if col in out.columns]
    out = out.drop(columns=existing_to_drop)

    print(f"Dropped columns: {existing_to_drop}")
    return out


def analyze_data_quality(df: pd.DataFrame) -> dict:
    print_section("5) DATA QUALITY")
    total_missing = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    print(f"Total missing values: {total_missing}")
    print(f"Total duplicate rows: {duplicate_count}")

    print("\nMissing values per column:")
    for col in df.columns:
        n = int(df[col].isna().sum())
        p = (n / len(df)) * 100 if len(df) > 0 else 0
        print(f"  - {col:25}: {n:6} ({p:.2f}%)")

    quality_score = 100.0
    if len(df) > 0 and len(df.columns) > 0:
        quality_score = 100 - ((total_missing / (len(df) * len(df.columns))) * 100)

    print(f"\nData quality score: {quality_score:.2f}%")
    return {
        "total_missing": total_missing,
        "duplicate_rows": duplicate_count,
        "quality_score": round(float(quality_score), 2),
    }


def profile_completeness(df: pd.DataFrame, label: str = "dataset") -> dict:
    print_section(f"COMPLETENESS PROFILE ({label.upper()})")

    rows, cols = df.shape
    total_cells = rows * cols
    total_null = int(df.isna().sum().sum())
    total_complete = int(total_cells - total_null)

    print(f"Rows: {rows}, Columns: {cols}")
    print(f"Total cells: {total_cells}")
    print(f"Complete cells: {total_complete}")
    print(f"Null cells: {total_null}")

    return {
        "rows": rows,
        "columns": cols,
        "total_cells": total_cells,
        "complete_cells": total_complete,
        "null_cells": total_null,
        "column_nulls": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "column_completeness_pct": ((1 - df.isna().mean()) * 100).round(2).to_dict(),
    }


def detect_outliers_iqr(
    df: pd.DataFrame,
    min_unique_for_continuous: int = 10,
    exclude_prefixes: tuple[str, ...] = ("route_",),
    exclude_columns: set[str] | None = None,
    exclude_keywords: tuple[str, ...] = ("_bin", "_flag", "_encoded"),
) -> dict[str, Any]:
    print_section("OUTLIER DETECTION")

    outlier_counts: dict[str, int] = {}
    skipped_columns: dict[str, str] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = exclude_columns or set()

    for col in numeric_cols:
        if col in excluded:
            skipped_columns[col] = "excluded_by_name"
            continue
        if any(col.startswith(prefix) for prefix in exclude_prefixes):
            skipped_columns[col] = "excluded_by_prefix"
            continue
        if any(keyword in col for keyword in exclude_keywords):
            skipped_columns[col] = "excluded_by_keyword"
            continue

        series = df[col].dropna()
        if series.empty:
            outlier_counts[col] = 0
            continue

        if series.nunique(dropna=True) <= min_unique_for_continuous:
            skipped_columns[col] = "low_cardinality"
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            skipped_columns[col] = "zero_iqr"
            continue

        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outlier_counts[col] = int(((series < low) | (series > high)).sum())

    if outlier_counts:
        print("Continuous columns analyzed:")
        for col, count in outlier_counts.items():
            print(f"  - {col:25}: {count}")
    else:
        print("No continuous numeric columns qualified for IQR outlier analysis.")

    if skipped_columns:
        print(f"Skipped columns: {len(skipped_columns)}")

    return {
        "method": "iqr",
        "iqr_multiplier": 1.5,
        "min_unique_for_continuous": int(min_unique_for_continuous),
        "exclude_prefixes": list(exclude_prefixes),
        "exclude_keywords": list(exclude_keywords),
        "analyzed_column_count": len(outlier_counts),
        "analyzed_columns": list(outlier_counts.keys()),
        "outlier_counts": outlier_counts,
        "skipped_columns": skipped_columns,
    }



def print_full_terminal_report(original_df: pd.DataFrame, final_df: pd.DataFrame, target_col: str, task: str) -> dict:
    print_section("7) FULL TERMINAL REPORT")

    print("Data shape:")
    print(f"  - Original: {original_df.shape[0]:,} rows, {original_df.shape[1]} columns")
    print(f"  - Processed: {final_df.shape[0]:,} rows, {final_df.shape[1]} columns")

    print("\nColumns in processed dataset:")
    for col in final_df.columns:
        print(f"  - {col}")


    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    continuous_numeric_cols = [
        col for col in numeric_cols if final_df[col].nunique(dropna=True) > 10 and not col.startswith("route_")
    ]
    if len(continuous_numeric_cols) > 0:
        print("\nNumeric column summary (continuous features):")
        print(final_df[continuous_numeric_cols].describe().to_string())

    categorical_cols = final_df.select_dtypes(exclude=[np.number]).columns
    if len(categorical_cols) > 0:
        print("\nCategorical column unique counts:")
        for col in categorical_cols:
            print(f"  - {col:25}: {final_df[col].nunique(dropna=False)} unique values")

    indicator_cols = [
        col
        for col in numeric_cols
        if final_df[col].nunique(dropna=True) <= 2
    ]
    if indicator_cols:
        print(f"\nBinary/indicator numeric columns: {len(indicator_cols)}")

    if task == "classification" and target_col in final_df.columns:
        print("\nTarget class distribution:")
        target_counts = final_df[target_col].value_counts(dropna=False)
        target_pct = (target_counts / len(final_df) * 100).round(2)
        for cls, count in target_counts.items():
            print(f"  - {cls}: {count} ({target_pct[cls]}%)")

    return {
        "original_shape": [int(original_df.shape[0]), int(original_df.shape[1])],
        "processed_shape": [int(final_df.shape[0]), int(final_df.shape[1])],
        "processed_columns": final_df.columns.tolist(),
    }


def save_outputs(df: pd.DataFrame, target_col: str, output_dir: Path, task: str, report: dict) -> None:
    print_section("8) OUTPUT EXPORT")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / f"cleaned_dataset_{task}.csv"
    report_path = output_dir / f"cleaned_report_{task}.json"

    df.to_csv(dataset_path, index=False)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved dataset: {dataset_path}")
    print(f"Saved report: {report_path}")
    print(f"Target column: {target_col}")





def main() -> None:
    input_file = "traffic_dataset.csv"
    task = "regression"
    output_dir = "outputs"
    scaling = "standard"
    outlier_min_unique = 10

    print_section("PREPROCESSING")
    print(f"Input: {input_file}")
    print(f"Task: {task}")
    print(f"Output dir: {output_dir}")

    df = pd.read_csv(input_file)
    
    df = choose_dataset_scope(df, sample_n=5000)

    type_groups = analyze_data_types(df)
    completeness_original = profile_completeness(df, label="original")

    df_fe = feature_engineering(df)
    total_missing_after_fe = int(df_fe.isna().sum().sum())
    if total_missing_after_fe > 0:
        print(f"Missing values detected after feature engineering: {total_missing_after_fe}")
        print("Missing values will be filled with median in clean_data.")
    else:
        print("No missing values detected before cleaning.")

    df_pre_clean = df_fe

    df_clean, clean_summary = clean_data(df_pre_clean)
    
    visualize(df_clean, Path(output_dir))

    df_encoded = encode_features(df_clean)

    duplicates_after_encoding = int(df_encoded.duplicated().sum())
    print(f"Duplicate rows after encoding: {duplicates_after_encoding}")
    if duplicates_after_encoding > 0:
        df_encoded = df_encoded.drop_duplicates().reset_index(drop=True)
        print(f"Removed duplicates after encoding: {duplicates_after_encoding}")

    df_final = drop_unused_columns(df_encoded)
    df_final, target_col = create_target(df_final, task)

    outlier_input_df = df_final.copy()

    if scaling != "none":
        df_final = normalize_features(df_final, target_col, method=scaling)

    completeness_final = profile_completeness(df_final, label="final")
    quality_summary = analyze_data_quality(df_final)
    skewness_summary = analyze_skewness_with_graphics(df_final, Path(output_dir))
    outlier_exclude_cols = {
        target_col,
        "day_of_week",
        "is_weekend",
        "rain",
        "is_rush_hour",
        "is_bad_weather",
    }
    outlier_summary = detect_outliers_iqr(
        outlier_input_df,
        min_unique_for_continuous=outlier_min_unique,
        exclude_prefixes=("route_",),
        exclude_columns=outlier_exclude_cols,
        exclude_keywords=("_bin", "_flag", "_encoded"),
    )

    sampling_summary: dict[str, Any] = {"method": "none", "note": "Dataset scope chosen interactively at start"}

    terminal_summary = print_full_terminal_report(df, df_final, target_col, task)

    report = {
        "task": task,
        "input_file": input_file,
        "target_col": target_col,
        "shape_final": [int(df_final.shape[0]), int(df_final.shape[1])],
        "type_groups": type_groups,
        "completeness": {
            "original": completeness_original,
            "final": completeness_final,
        },
        "cleaning": clean_summary,
        "quality": quality_summary,
        "skewness": skewness_summary,
        "outliers_iqr": outlier_summary,
        "sampling": sampling_summary,
        "terminal_summary": terminal_summary,
        "created_features": [
            "hour",
            "day_of_week",
            "is_weekend",
            "route",
            "is_rush_hour",
            "is_bad_weather",
            "speed_normal",
            "hour_sin",
            "hour_cos",
        ],
    }

    save_outputs(df_final, target_col, Path(output_dir), task, report)

    print_section("COMPLETED")


if __name__ == "__main__":
    main()
