from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_skewness_with_graphics(df: pd.DataFrame, output_dir: Path, max_plots: int = 10) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found for skewness analysis.")
        return {"plotted_columns": [], "plot_dir": None}

    candidate_cols = [
        col
        for col in numeric_cols
        if df[col].nunique(dropna=True) > 10 and not col.startswith("route_")
    ]
    if not candidate_cols:
        candidate_cols = numeric_cols

    skew_series = df[candidate_cols].skew(numeric_only=True).sort_values(key=lambda s: s.abs(), ascending=False)
    cols_to_plot = skew_series.head(max_plots).index.tolist()

    plot_dir = output_dir / "skewness_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("Skewness (sorted by absolute value):")
    for col, skew_val in skew_series.items():
        print(f"  - {col:25}: {float(skew_val):8.4f}")

    summary = {
        "plot_dir": str(plot_dir),
        "plotted_columns": cols_to_plot,
        "skewness": {},
        "outlier_counts_iqr": {},
    }

    for col in cols_to_plot:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outlier_count = int(((series < low) | (series > high)).sum())
        skew_value = float(series.skew())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(series, bins=30)
        axes[0].set_title(f"{col} histogram")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")

        axes[1].boxplot(series, vert=False)
        axes[1].set_title(f"{col} boxplot")
        axes[1].set_xlabel(col)

        fig.suptitle(
            f"{col} | skew={skew_value:.3f} | IQR outliers={outlier_count}",
            fontsize=11,
        )
        fig.tight_layout()

        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in col)
        plot_path = plot_dir / f"{safe_name}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        summary["skewness"][col] = round(skew_value, 4)
        summary["outlier_counts_iqr"][col] = outlier_count

    print(f"Saved skewness/outlier plots to: {plot_dir}")
    return summary
