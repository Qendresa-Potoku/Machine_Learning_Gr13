from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _safe_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100


def _corr_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        return pd.DataFrame()

    cols = [c for c in numeric_df.columns if numeric_df[c].nunique(dropna=False) > 1 and not c.startswith("route_")]
    if len(cols) < 2:
        return pd.DataFrame()

    return numeric_df[cols].corr()


def analyze_true_outliers(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    required = ["delay_min"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for outlier analysis: {missing}")

    print("\n" + "=" * 60)
    print("TRUE OUTLIER ANALYSIS")
    print("=" * 60)

    delay_series = out["delay_min"]
    q1 = float(delay_series.quantile(0.25))
    q3 = float(delay_series.quantile(0.75))
    iqr = q3 - q1
    iqr_low = q1 - 1.5 * iqr
    iqr_high = q3 + 1.5 * iqr
    q99 = float(delay_series.quantile(0.99))

    iqr_delay_mask = (delay_series < iqr_low) | (delay_series > iqr_high)
    top1_delay_mask = delay_series >= q99
    high_delay_mask = (delay_series > iqr_high) | top1_delay_mask

    rush_col = out["is_rush_hour"] if "is_rush_hour" in out.columns else pd.Series(0, index=out.index)
    weather_col = out["is_bad_weather"] if "is_bad_weather" in out.columns else pd.Series(0, index=out.index)

    speed_invalid_mask = pd.Series(False, index=out.index)
    if "speed_normal" in out.columns:
        speed_invalid_mask = out["speed_normal"] <= 0

    duration_invalid_mask = pd.Series(False, index=out.index)
    if "duration_normal_min" in out.columns:
        duration_invalid_mask = out["duration_normal_min"] == 0

    invalid_mask = (delay_series < -3) | speed_invalid_mask | duration_invalid_mask

    valid_mask = high_delay_mask & ((rush_col == 1) | (weather_col == 1)) & (~invalid_mask)
    suspicious_mask = high_delay_mask & (rush_col == 0) & (weather_col == 0) & (~invalid_mask)

    outlier_mask = iqr_delay_mask | top1_delay_mask | invalid_mask
    uncategorized_outlier_mask = outlier_mask & (~invalid_mask) & (~valid_mask) & (~suspicious_mask)

    out["outlier_type"] = "normal"
    out.loc[invalid_mask, "outlier_type"] = "invalid"
    out.loc[valid_mask, "outlier_type"] = "valid"
    out.loc[suspicious_mask | uncategorized_outlier_mask, "outlier_type"] = "suspicious"

    outliers_df = out[out["outlier_type"] != "normal"].copy()
    normal_df = out[out["outlier_type"] == "normal"].copy()

    total_rows = len(out)
    outlier_rows = len(outliers_df)

    valid_count = int((out["outlier_type"] == "valid").sum())
    suspicious_count = int((out["outlier_type"] == "suspicious").sum())
    invalid_count = int((out["outlier_type"] == "invalid").sum())

    print("Step 1 - Outlier Detection")
    print(f"  IQR bounds (delay_min): low={iqr_low:.4f}, high={iqr_high:.4f}")
    print(f"  Top 1% threshold (delay_min q99): {q99:.4f}")
    print(f"  IQR outliers (delay_min): {int(iqr_delay_mask.sum()):,}")
    print(f"  Top 1% outliers (delay_min): {int(top1_delay_mask.sum()):,}")

    print("\nStep 2 - Outlier Classification")
    print(f"  valid: {valid_count:,} ({_safe_pct(valid_count, total_rows):.2f}% of full dataset)")
    print(f"  suspicious: {suspicious_count:,} ({_safe_pct(suspicious_count, total_rows):.2f}% of full dataset)")
    print(f"  invalid: {invalid_count:,} ({_safe_pct(invalid_count, total_rows):.2f}% of full dataset)")
    print(f"  total outliers: {outlier_rows:,} ({_safe_pct(outlier_rows, total_rows):.2f}% of full dataset)")

    outlier_rush_count = int((outliers_df.get("is_rush_hour", pd.Series(dtype=int)) == 1).sum())
    outlier_weather_count = int((outliers_df.get("is_bad_weather", pd.Series(dtype=int)) == 1).sum())
    outlier_rush_pct = _safe_pct(outlier_rush_count, outlier_rows)
    outlier_weather_pct = _safe_pct(outlier_weather_count, outlier_rows)

    hour_distribution = {}
    if "hour" in outliers_df.columns and len(outliers_df) > 0:
        hour_distribution = (
            outliers_df["hour"]
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )

    outlier_top_routes = {}
    normal_top_routes = {}
    if "route" in out.columns and "delay_min" in out.columns:
        if len(outliers_df) > 0:
            outlier_top_routes = (
                outliers_df.groupby("route", dropna=False)["delay_min"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .round(4)
                .to_dict()
            )
        if len(normal_df) > 0:
            normal_top_routes = (
                normal_df.groupby("route", dropna=False)["delay_min"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .round(4)
                .to_dict()
            )

    print("\nStep 3 - Relationship Analysis (Outliers Only)")
    print(f"  % of outliers in rush hour: {outlier_rush_pct:.2f}%")
    print(f"  % of outliers in bad weather: {outlier_weather_pct:.2f}%")
    print(f"  Outlier hour distribution available: {'yes' if hour_distribution else 'no'}")
    print(f"  Top outlier routes available: {'yes' if outlier_top_routes else 'no'}")

    outlier_dir = output_dir / "outlier_analysis"
    outlier_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    if "is_rush_hour" in out.columns:
        plt.figure(figsize=(9, 6))
        sns.boxplot(data=out, x="is_rush_hour", y="delay_min")
        plt.title("Delay vs Rush Hour")
        plt.xlabel("is_rush_hour")
        plt.ylabel("delay_min")
        plt.tight_layout()
        plt.savefig(outlier_dir / "boxplot_delay_vs_rush_hour.png", dpi=300)
        plt.close()

    if "is_bad_weather" in out.columns:
        plt.figure(figsize=(9, 6))
        sns.boxplot(data=out, x="is_bad_weather", y="delay_min")
        plt.title("Delay vs Bad Weather")
        plt.xlabel("is_bad_weather")
        plt.ylabel("delay_min")
        plt.tight_layout()
        plt.savefig(outlier_dir / "boxplot_delay_vs_bad_weather.png", dpi=300)
        plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(normal_df["delay_min"], bins=40, stat="density", color="#4C78A8", alpha=0.55, label="normal")
    if len(outliers_df) > 0:
        sns.histplot(outliers_df["delay_min"], bins=40, stat="density", color="#E45756", alpha=0.55, label="outliers")
    plt.title("Delay Distribution: Outliers vs Normal")
    plt.xlabel("delay_min")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outlier_dir / "hist_delay_outliers_vs_normal.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    order = ["normal", "valid", "suspicious", "invalid"]
    counts = out["outlier_type"].value_counts().reindex(order, fill_value=0)
    counts_df = pd.DataFrame({"outlier_type": counts.index, "count": counts.values})
    sns.barplot(
        data=counts_df,
        x="outlier_type",
        y="count",
        hue="outlier_type",
        palette=["#4C78A8", "#54A24B", "#F58518", "#E45756"],
        legend=False,
    )
    plt.title("Outlier Type Counts")
    plt.xlabel("outlier_type")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outlier_dir / "bar_outlier_type_counts.png", dpi=300)
    plt.close()

    if "speed_normal" in out.columns:
        plt.figure(figsize=(9, 6))
        scatter_df = out[["delay_min", "speed_normal", "outlier_type"]].copy()
        if len(scatter_df) > 12000:
            scatter_df = scatter_df.sample(n=12000, random_state=42)
        sns.scatterplot(
            data=scatter_df,
            x="speed_normal",
            y="delay_min",
            hue="outlier_type",
            hue_order=["normal", "valid", "suspicious", "invalid"],
            alpha=0.6,
            s=25,
        )
        plt.title("Delay vs Speed Normal")
        plt.xlabel("speed_normal")
        plt.ylabel("delay_min")
        plt.tight_layout()
        plt.savefig(outlier_dir / "scatter_delay_vs_speed_normal.png", dpi=300)
        plt.close()

    full_corr = _corr_for_analysis(out)
    outlier_corr = _corr_for_analysis(outliers_df)

    corr_diff_top_changes = []
    if not full_corr.empty:
        plt.figure(figsize=(12, 9))
        sns.heatmap(full_corr, cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True)
        plt.title("Correlation Matrix - Full Dataset")
        plt.tight_layout()
        plt.savefig(outlier_dir / "correlation_full_dataset.png", dpi=300)
        plt.close()

    if not outlier_corr.empty:
        plt.figure(figsize=(12, 9))
        sns.heatmap(outlier_corr, cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True)
        plt.title("Correlation Matrix - Outliers Only")
        plt.tight_layout()
        plt.savefig(outlier_dir / "correlation_outliers_only.png", dpi=300)
        plt.close()

    common_cols = [c for c in full_corr.columns if c in outlier_corr.columns]
    if len(common_cols) >= 2:
        diff = outlier_corr.loc[common_cols, common_cols] - full_corr.loc[common_cols, common_cols]
        abs_diff = diff.abs().where(np.triu(np.ones(diff.shape), k=1).astype(bool))

        stacked = abs_diff.stack().sort_values(ascending=False).head(10)
        for (c1, c2), delta in stacked.items():
            corr_diff_top_changes.append(
                {
                    "feature_a": c1,
                    "feature_b": c2,
                    "abs_delta": round(float(delta), 4),
                    "full_corr": round(float(full_corr.loc[c1, c2]), 4),
                    "outlier_corr": round(float(outlier_corr.loc[c1, c2]), 4),
                }
            )

    suspicious_pct_outliers = _safe_pct(suspicious_count, outlier_rows)
    invalid_pct_outliers = _safe_pct(invalid_count, outlier_rows)

    print("\nStep 6 - Final Insights")
    print(f"  {outlier_rush_pct:.2f}% of outliers occur during rush hour.")
    print(f"  {suspicious_pct_outliers:.2f}% of outliers are unexplained (suspicious).")
    print(f"  {invalid_pct_outliers:.2f}% of outliers are invalid data points.")

    if invalid_pct_outliers >= 20:
        interpretation = (
            "A meaningful share of outliers looks data-quality related. "
            "Flag invalid rows for quality review before modeling decisions."
        )
    elif suspicious_pct_outliers >= 40:
        interpretation = (
            "Many outliers are not explained by rush hour or bad weather. "
            "They may be real rare events or missing explanatory features."
        )
    else:
        interpretation = (
            "Most outliers are consistent with realistic traffic conditions. "
            "Keep them, but preserve outlier flags for robust model evaluation."
        )

    print(f"  Interpretation: {interpretation}")
    print(f"Saved outlier analysis artifacts to: {outlier_dir}")

    summary = {
        "detection": {
            "iqr": {
                "q1": round(q1, 6),
                "q3": round(q3, 6),
                "iqr": round(iqr, 6),
                "low_bound": round(iqr_low, 6),
                "high_bound": round(iqr_high, 6),
                "outlier_count": int(iqr_delay_mask.sum()),
            },
            "top_1pct": {
                "q99_threshold": round(q99, 6),
                "outlier_count": int(top1_delay_mask.sum()),
            },
        },
        "counts": {
            "total_rows": int(total_rows),
            "outlier_rows": int(outlier_rows),
            "normal_rows": int(len(normal_df)),
            "valid": valid_count,
            "suspicious": suspicious_count,
            "invalid": invalid_count,
        },
        "percentages": {
            "outliers_of_dataset_pct": round(_safe_pct(outlier_rows, total_rows), 2),
            "valid_of_outliers_pct": round(_safe_pct(valid_count, outlier_rows), 2),
            "suspicious_of_outliers_pct": round(suspicious_pct_outliers, 2),
            "invalid_of_outliers_pct": round(invalid_pct_outliers, 2),
            "outliers_in_rush_hour_pct": round(outlier_rush_pct, 2),
            "outliers_in_bad_weather_pct": round(outlier_weather_pct, 2),
        },
        "hour_distribution_outliers": {str(k): int(v) for k, v in hour_distribution.items()},
        "top_routes": {
            "outliers": outlier_top_routes,
            "normal": normal_top_routes,
        },
        "delay_summary": {
            "outliers": {
                "mean": round(float(outliers_df["delay_min"].mean()), 4) if len(outliers_df) else None,
                "median": round(float(outliers_df["delay_min"].median()), 4) if len(outliers_df) else None,
            },
            "normal": {
                "mean": round(float(normal_df["delay_min"].mean()), 4) if len(normal_df) else None,
                "median": round(float(normal_df["delay_min"].median()), 4) if len(normal_df) else None,
            },
        },
        "correlation": {
            "full_shape": [int(full_corr.shape[0]), int(full_corr.shape[1])],
            "outliers_shape": [int(outlier_corr.shape[0]), int(outlier_corr.shape[1])],
            "top_abs_differences": corr_diff_top_changes,
        },
        "insights": {
            "rush_hour_outlier_statement": f"{outlier_rush_pct:.2f}% of outliers occur during rush hour",
            "suspicious_outlier_statement": f"{suspicious_pct_outliers:.2f}% of outliers are unexplained (suspicious)",
            "invalid_outlier_statement": f"{invalid_pct_outliers:.2f}% of outliers are invalid data points",
            "interpretation": interpretation,
        },
        "artifacts_dir": str(outlier_dir),
    }

    return out, summary
