import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from math import pi, sin, cos
from skewness_utils import analyze_skewness_with_graphics


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


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

    # Temporal features (no data leakage)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["hour"] = out["timestamp"].dt.hour
        out["day_of_week"] = out["timestamp"].dt.dayofweek
        out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

    # Route feature: concatenate origin and destination
    if {"origin", "destination"}.issubset(out.columns):
        out["route"] = out["origin"].astype(str) + " → " + out["destination"].astype(str)

    # Rush hour feature (hours 7-9 and 16-18)
    if "hour" in out.columns:
        out["is_rush_hour"] = (
            (out["hour"].isin(range(7, 10))) | (out["hour"].isin(range(16, 19)))
        ).astype(int)

    # Bad weather feature
    if {"rain", "wind"}.issubset(out.columns):
        out["is_bad_weather"] = ((out["rain"] > 0) | (out["wind"] > 8)).astype(int)

    # Speed normal: distance / duration_normal_min (no leakage, only uses normal duration)
    if {"distance_km", "duration_normal_min"}.issubset(out.columns):
        epsilon = 1e-5
        out["speed_normal"] = out["distance_km"] / (out["duration_normal_min"] + epsilon)

    # Cyclic encoding for hour
    if "hour" in out.columns:
        out["hour_sin"] = np.sin(2 * pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * pi * out["hour"] / 24)

    out = out.replace([np.inf, -np.inf], np.nan)

    print("Created: hour, day_of_week, is_weekend, route, is_rush_hour, is_bad_weather, speed_normal, hour_sin, hour_cos")
    return out


def clean_data(df: pd.DataFrame, remove_delay_outliers: bool = True) -> tuple[pd.DataFrame, dict]:
    print_section("3) DATA CLEANING")
    cleaned = df.copy()

    before_rows = len(cleaned)
    before_null = int(cleaned.isna().sum().sum())

    cleaned = cleaned.dropna().drop_duplicates().reset_index(drop=True)

    after_rows = len(cleaned)
    after_null = int(cleaned.isna().sum().sum())

    # Domain-based filtering (real-world constraints), not statistical IQR filtering.
    # We keep realistic heavy traffic behavior (large positive delays, long distances,
    # and duration variation) and only remove impossible/noisy records.
    outlier_removed = 0
    negative_delay_removed = 0
    low_speed_removed = 0
    if remove_delay_outliers:
        if "delay_min" in cleaned.columns:
            before_negative_delay = len(cleaned)
            cleaned = cleaned[cleaned["delay_min"] > -5]
            negative_delay_removed = before_negative_delay - len(cleaned)

        if "speed_normal" in cleaned.columns:
            before_low_speed = len(cleaned)
            cleaned = cleaned[cleaned["speed_normal"] > 0.05]
            low_speed_removed = before_low_speed - len(cleaned)

        cleaned = cleaned.reset_index(drop=True)
        outlier_removed = negative_delay_removed + low_speed_removed

    print(f"Rows before cleaning: {before_rows}")
    print(f"Rows after dropna+drop_duplicates: {after_rows}")
    print(f"Nulls before: {before_null}, nulls after: {after_null}")
    print(f"Domain-filtered rows removed (total): {outlier_removed}")
    print(f"  - delay_min <= -5 removed: {negative_delay_removed}")
    print(f"  - speed_normal <= 0.05 removed: {low_speed_removed}")

    summary = {
        "rows_before": before_rows,
        "rows_after_dropna_dedup": after_rows,
        "null_before": before_null,
        "null_after": after_null,
        "domain_filtered_rows_removed": outlier_removed,
        "negative_delay_removed": negative_delay_removed,
        "low_speed_removed": low_speed_removed,
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


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply one-hot encoding to categorical features."""
    print_section("ENCODE FEATURES")
    out = df.copy()

    # One-hot encode route column
    if "route" in out.columns:
        route_encoded = pd.get_dummies(out["route"], prefix="route", drop_first=True)
        out = pd.concat([out, route_encoded], axis=1)
        print(f"Created {len(route_encoded.columns)} route encoding columns")

    return out


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not needed for modeling (prevent data leakage and reduce dimensionality)."""
    print_section("DROP UNUSED COLUMNS")
    out = df.copy()

    # Columns to drop
    columns_to_drop = [
        "timestamp",      # Not needed after extracting temporal features
        "origin",         # Not needed after creating route
        "destination",    # Not needed after creating route
        "route",          # Drop after one-hot encoding
        "hour",           # Not needed after sin/cos encoding
        "duration_traffic_min",  # IMPORTANT: Remove to prevent data leakage
    ]

    # Drop only columns that exist in the dataframe
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


def print_full_terminal_report(original_df: pd.DataFrame, final_df: pd.DataFrame, target_col: str, task: str) -> dict:
    print_section("7) FULL TERMINAL REPORT")

    original_mem_mb = original_df.memory_usage(deep=True).sum() / (1024**2)
    final_mem_mb = final_df.memory_usage(deep=True).sum() / (1024**2)

    print("Data shape:")
    print(f"  - Original: {original_df.shape[0]:,} rows, {original_df.shape[1]} columns")
    print(f"  - Processed: {final_df.shape[0]:,} rows, {final_df.shape[1]} columns")

    print("\nMemory usage:")
    print(f"  - Original: {original_mem_mb:.2f} MB")
    print(f"  - Processed: {final_mem_mb:.2f} MB")

    print("\nColumns in processed dataset:")
    for col in final_df.columns:
        print(f"  - {col}")

    print("\nPer-column memory usage (MB):")
    col_mem = (final_df.memory_usage(deep=True) / (1024**2)).sort_values(ascending=False)
    for col, mem_mb in col_mem.items():
        print(f"  - {col:25}: {mem_mb:10.4f} MB")

    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("\nNumeric column summary:")
        print(final_df[numeric_cols].describe().to_string())

    categorical_cols = final_df.select_dtypes(exclude=[np.number]).columns
    if len(categorical_cols) > 0:
        print("\nCategorical column unique counts:")
        for col in categorical_cols:
            print(f"  - {col:25}: {final_df[col].nunique(dropna=False)} unique values")

    if task == "classification" and target_col in final_df.columns:
        print("\nTarget class distribution:")
        target_counts = final_df[target_col].value_counts(dropna=False)
        target_pct = (target_counts / len(final_df) * 100).round(2)
        for cls, count in target_counts.items():
            print(f"  - {cls}: {count} ({target_pct[cls]}%)")

    return {
        "original_shape": [int(original_df.shape[0]), int(original_df.shape[1])],
        "processed_shape": [int(final_df.shape[0]), int(final_df.shape[1])],
        "original_memory_mb": round(float(original_mem_mb), 2),
        "processed_memory_mb": round(float(final_mem_mb), 2),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 dataset preparation for ML")
    parser.add_argument("--input", default="traffic_dataset.csv", help="Path to input CSV dataset")
    parser.add_argument(
        "--task",
        default="regression",
        choices=["regression", "classification"],
        help="Target type: regression or classification",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Folder where prepared dataset and report are saved",
    )
    parser.add_argument(
        "--keep-outliers",
        action="store_true",
        help="If set, delay_min outliers are kept",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_section("PREPROCESSING")
    print(f"Input: {args.input}")
    print(f"Task: {args.task}")
    print(f"Output dir: {args.output_dir}")

    df = pd.read_csv(args.input)

    type_groups = analyze_data_types(df)
    df_fe = feature_engineering(df)
    df_clean, clean_summary = clean_data(df_fe, remove_delay_outliers=(not args.keep_outliers))
    df_encoded = encode_features(df_clean)

    duplicates_after_encoding = int(df_encoded.duplicated().sum())
    print(f"Duplicate rows after encoding: {duplicates_after_encoding}")
    if duplicates_after_encoding > 0:
        df_encoded = df_encoded.drop_duplicates().reset_index(drop=True)
        print(f"Removed duplicates after encoding: {duplicates_after_encoding}")

    df_final = drop_unused_columns(df_encoded)
    df_final, target_col = create_target(df_final, args.task)
    quality_summary = analyze_data_quality(df_final)
    skewness_summary = analyze_skewness_with_graphics(df_final, Path(args.output_dir))
    terminal_summary = print_full_terminal_report(df, df_final, target_col, args.task)

    report = {
        "task": args.task,
        "input_file": args.input,
        "target_col": target_col,
        "shape_final": [int(df_final.shape[0]), int(df_final.shape[1])],
        "type_groups": type_groups,
        "cleaning": clean_summary,
        "quality": quality_summary,
        "skewness": skewness_summary,
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

    save_outputs(df_final, target_col, Path(args.output_dir), args.task, report)

    print_section("COMPLETED")


if __name__ == "__main__":
    main()
