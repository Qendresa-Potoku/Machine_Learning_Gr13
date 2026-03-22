import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

def visualize(df: pd.DataFrame, output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("VISUALIZATIONS")
    print("=" * 60)
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    if "hour" in df.columns and "delay_min" in df.columns:
        plt.figure(figsize=(12, 6))
        
        sns.lineplot(data=df, x="hour", y="delay_min", errorbar="sd", marker="o", linewidth=2.5, color="#e74c3c")
        
        plt.title("Impact of Time on Traffic Delay", fontsize=16, fontweight='bold')
        plt.xlabel("Hour of Day (0-23)", fontsize=12)
        plt.ylabel("Delay (minutes)", fontsize=12)
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle="--", alpha=0.7)
        
        plt.axvspan(7, 9, color='orange', alpha=0.2, label='Morning Rush')
        plt.axvspan(16, 18, color='orange', alpha=0.2, label='Evening Rush')
        plt.legend()
        
        plt.tight_layout()
        save_path = viz_dir / "delay_vs_hour.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")
        
    if "distance_km" in df.columns and "delay_min" in df.columns:
        plt.figure(figsize=(12, 6))
        
        if len(df) > 1000:
            plt.hexbin(df["distance_km"], df["delay_min"], gridsize=20, cmap="Blues", mincnt=1)
            plt.colorbar(label="Count")
            sns.regplot(data=df, x="distance_km", y="delay_min", scatter=False, color="red", line_kws={"linestyle": "--"})
        else:
            sns.scatterplot(data=df, x="distance_km", y="delay_min", alpha=0.6, color="#3498db")
            sns.regplot(data=df, x="distance_km", y="delay_min", scatter=False, color="red", line_kws={"linestyle": "--"})
            
        plt.title("Correlation: Distance vs Delay", fontsize=16, fontweight='bold')
        plt.xlabel("Distance (km)", fontsize=12)
        plt.ylabel("Delay (minutes)", fontsize=12)
        
        plt.tight_layout()
        save_path = viz_dir / "delay_vs_distance.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

    if "route" in df.columns and "delay_min" in df.columns:
        top_routes = df.groupby("route")["delay_min"].mean().sort_values(ascending=False).head(15)
        
        plt.figure(figsize=(14, 8))
        barplot = sns.barplot(x=top_routes.values, y=top_routes.index, palette="viridis", hue=top_routes.index, legend=False)
        
        plt.title("Top 15 Most Congested Routes (Avg Delay)", fontsize=16, fontweight='bold')
        plt.xlabel("Average Delay (min)", fontsize=12)
        plt.ylabel("Route", fontsize=12)
        
        for i, v in enumerate(top_routes.values):
            barplot.text(v + 0.1, i, f"{v:.1f}m", va='center', fontsize=10)
            
        plt.tight_layout()
        save_path = viz_dir / "top_routes_delay.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")
    
    if "is_weekend" in df.columns and "delay_min" in df.columns:
        plt.figure(figsize=(8, 6))
        df_copy = df.copy()
        df_copy["Day Type"] = df_copy["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        
        sns.boxplot(data=df_copy, x="Day Type", y="delay_min", palette="Set2", hue="Day Type")
        
        plt.title("Traffic Delay Distribution: Weekday vs Weekend", fontsize=16, fontweight='bold')
        plt.ylabel("Delay (min)", fontsize=12)
        
        plt.tight_layout()
        save_path = viz_dir / "delay_weekend_vs_weekday.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

