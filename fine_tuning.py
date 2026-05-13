"""
Phase 3A: Hyperparameter Tuning for RandomForest
Fine-tunes RandomForest parameters using GridSearchCV and cross-validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


class RandomForestTuner:
    """Fine-tune RandomForest hyperparameters"""

    def __init__(self, output_dir: Path = Path("outputs/fine_tuning")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model: Optional[RandomForestRegressor] = None
        self.best_params: Optional[dict] = None
        self.grid_search: Optional[GridSearchCV] = None
        self.baseline_metrics: Optional[dict] = None
        self.tuned_metrics: Optional[dict] = None

    @staticmethod
    def _print_section(title: str) -> None:
        """Print formatted section header"""
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)

    @staticmethod
    def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate evaluation metrics"""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

    def measure_baseline(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> dict:
        """Measure baseline RandomForest performance"""
        self._print_section("BASELINE: Current RandomForest (450 trees, default params)")

        # Train baseline model
        baseline_model = RandomForestRegressor(
            n_estimators=450, random_state=42, n_jobs=-1
        )
        baseline_model.fit(X_train, y_train, sample_weight=sample_weight)

        # Evaluate
        y_pred = baseline_model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)

        # Cross-validation score
        cv_scores = cross_val_score(
            baseline_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
        )
        cv_rmse = np.sqrt(-cv_scores.mean())

        print(f"Test RMSE:     {metrics['RMSE']:.4f} minutes")
        print(f"Test MAE:      {metrics['MAE']:.4f} minutes")
        print(f"Test R²:       {metrics['R2']:.4f}")
        print(f"CV RMSE (5x):  {cv_rmse:.4f} minutes (mean ± std)")
        print(f"CV std dev:    {np.sqrt(-cv_scores.std()):.4f}")

        self.baseline_metrics = metrics
        return metrics

    def hyperparameter_search(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> dict:
        """Search optimal hyperparameters using GridSearchCV"""
        self._print_section("HYPERPARAMETER GRID SEARCH (OPTIMIZED)")

        # Define SMALLER parameter grid for faster optimization
        # Focus on most impactful parameters
        param_grid = {
            "n_estimators": [450, 600, 800],        # Trees
            "max_depth": [20, 30, None],             # Depth
            "min_samples_leaf": [2, 5],              # Overfitting prevention
        }

        print(f"\nTesting {self._count_combinations(param_grid)} parameter combinations...")
        print("Using 3-fold cross-validation (faster than 5-fold)\n")

        # GridSearchCV with reduced CV folds for speed
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

        self.grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,  # Reduced from 5 to 3 for speed
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )

        # Fit grid search
        self.grid_search.fit(X_train, y_train, sample_weight=sample_weight)

        # Best parameters
        self.best_params = self.grid_search.best_params_
        print("\n" + "=" * 70)
        print("BEST PARAMETERS FOUND:")
        print("=" * 70)
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")

        # Best CV score
        best_cv_rmse = np.sqrt(-self.grid_search.best_score_)
        print(f"\nBest CV RMSE: {best_cv_rmse:.4f} minutes")

        return self.best_params

    def evaluate_tuned_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> dict:
        """Train and evaluate model with best parameters"""
        self._print_section("TUNED MODEL: Evaluating with best parameters")

        # Train final model with best params
        self.best_model = RandomForestRegressor(
            **self.best_params, random_state=42, n_jobs=-1
        )
        self.best_model.fit(X_train, y_train, sample_weight=sample_weight)

        # Evaluate
        y_pred = self.best_model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)

        # Cross-validation score
        cv_scores = cross_val_score(
            self.best_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
        )
        cv_rmse = np.sqrt(-cv_scores.mean())

        print(f"Test RMSE:     {metrics['RMSE']:.4f} minutes")
        print(f"Test MAE:      {metrics['MAE']:.4f} minutes")
        print(f"Test R²:       {metrics['R2']:.4f}")
        print(f"CV RMSE (5x):  {cv_rmse:.4f} minutes")
        print(f"CV std dev:    {np.sqrt(-cv_scores.std()):.4f}")

        self.tuned_metrics = metrics
        return metrics

    def compare_results(self) -> pd.DataFrame:
        """Compare baseline vs tuned model"""
        self._print_section("COMPARISON: Baseline vs Tuned")

        comparison = pd.DataFrame(
            [
                {"Model": "Baseline (450 trees, default)", **self.baseline_metrics},
                {"Model": "Tuned (optimized params)", **self.tuned_metrics},
            ]
        )

        print("\n" + comparison.to_string(index=False))

        # Calculate improvements
        mae_improvement = (
            (self.baseline_metrics["MAE"] - self.tuned_metrics["MAE"])
            / self.baseline_metrics["MAE"]
            * 100
        )
        rmse_improvement = (
            (self.baseline_metrics["RMSE"] - self.tuned_metrics["RMSE"])
            / self.baseline_metrics["RMSE"]
            * 100
        )
        r2_improvement = (
            (self.tuned_metrics["R2"] - self.baseline_metrics["R2"])
            / self.baseline_metrics["R2"]
            * 100
        )

        print(f"\nImprovement:")
        print(f"  MAE improvement:  {mae_improvement:+.2f}%")
        print(f"  RMSE improvement: {rmse_improvement:+.2f}%")
        print(f"  R² improvement:   {r2_improvement:+.2f}%")

        # Save comparison
        comparison.to_csv(self.output_dir / "baseline_vs_tuned.csv", index=False)
        print(f"\n[OK] Saved: {self.output_dir / 'baseline_vs_tuned.csv'}")

        return comparison

    def save_artifacts(self) -> None:
        """Save best model and results"""
        self._print_section("SAVING ARTIFACTS")

        # Save model
        model_path = self.output_dir / "tuned_random_forest_model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(self.best_model, f)
        print(f"[OK] Saved model: {model_path}")

        # Save best parameters
        params_df = pd.DataFrame([self.best_params])
        params_path = self.output_dir / "best_hyperparameters.csv"
        params_df.to_csv(params_path, index=False)
        print(f"[OK] Saved parameters: {params_path}")

        # Save feature importances
        feature_importance = pd.DataFrame(
            {
                "feature": self.best_model.feature_names_in_,
                "importance": self.best_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        
        importance_path = self.output_dir / "feature_importance_tuned.csv"
        feature_importance.to_csv(importance_path, index=False)
        print(f"[OK] Saved feature importance: {importance_path}")

    def create_visualizations(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Create comparison visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Metrics comparison
        ax1 = axes[0, 0]
        metrics_names = ["MAE", "RMSE", "R2"]
        baseline_vals = [
            self.baseline_metrics["MAE"],
            self.baseline_metrics["RMSE"],
            self.baseline_metrics["R2"],
        ]
        tuned_vals = [
            self.tuned_metrics["MAE"],
            self.tuned_metrics["RMSE"],
            self.tuned_metrics["R2"],
        ]

        x = np.arange(len(metrics_names))
        width = 0.35
        ax1.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#4C78A8")
        ax1.bar(x + width / 2, tuned_vals, width, label="Tuned", color="#54A24B")
        ax1.set_ylabel("Value")
        ax1.set_title("Metrics Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. RMSE improvement percentage
        ax2 = axes[0, 1]
        improvements = {
            "MAE": (self.baseline_metrics["MAE"] - self.tuned_metrics["MAE"])
            / self.baseline_metrics["MAE"]
            * 100,
            "RMSE": (self.baseline_metrics["RMSE"] - self.tuned_metrics["RMSE"])
            / self.baseline_metrics["RMSE"]
            * 100,
            "R2": (self.tuned_metrics["R2"] - self.baseline_metrics["R2"])
            / self.baseline_metrics["R2"]
            * 100,
        }
        colors = ["#54A24B" if v > 0 else "#E15759" for v in improvements.values()]
        ax2.barh(list(improvements.keys()), list(improvements.values()), color=colors)
        ax2.set_xlabel("Improvement (%)")
        ax2.set_title("Performance Improvement")
        ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax2.grid(True, alpha=0.3)

        # 3. Top 10 features
        ax3 = axes[1, 0]
        feature_importance = pd.DataFrame(
            {
                "feature": self.best_model.feature_names_in_,
                "importance": self.best_model.feature_importances_,
            }
        ).sort_values("importance", ascending=True).tail(10)
        
        ax3.barh(feature_importance["feature"], feature_importance["importance"], color="#4C78A8")
        ax3.set_xlabel("Importance")
        ax3.set_title("Top 10 Most Important Features")
        ax3.grid(True, alpha=0.3)

        # 4. Grid search results heatmap (top params)
        ax4 = axes[1, 1]
        results_df = pd.DataFrame(self.grid_search.cv_results_)
        top_results = results_df.nsmallest(10, "rank_test_score")[["param_n_estimators", "mean_test_score"]]
        top_results["RMSE"] = np.sqrt(-top_results["mean_test_score"])
        
        ax4.plot(
            range(len(top_results)),
            top_results["RMSE"],
            marker="o",
            linewidth=2,
            markersize=8,
            color="#4C78A8",
        )
        ax4.set_ylabel("RMSE (minutes)")
        ax4.set_xlabel("Top 10 Parameter Combinations")
        ax4.set_title("Grid Search Results (Top 10)")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / "tuning_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[OK] Saved visualization: {plot_path}")

    @staticmethod
    def _count_combinations(param_grid: dict) -> int:
        """Count total parameter combinations"""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count


def run_phase3a_hyperparameter_tuning(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    output_dir: Path = Path("outputs/fine_tuning"),
) -> tuple[RandomForestRegressor, dict]:
    """
    Run complete Phase 3A: Hyperparameter Tuning
    """
    tuner = RandomForestTuner(output_dir)

    # Step 1: Baseline
    tuner.measure_baseline(X_train, X_test, y_train, y_test, sample_weight)

    # Step 2: Grid search
    tuner.hyperparameter_search(X_train, X_test, y_train, y_test, sample_weight)

    # Step 3: Evaluate tuned model
    tuner.evaluate_tuned_model(X_train, X_test, y_train, y_test, sample_weight)

    # Step 4: Compare
    tuner.compare_results()

    # Step 5: Save artifacts
    tuner.save_artifacts()

    # Step 6: Visualizations
    tuner.create_visualizations(X_test, y_test)

    return tuner.best_model, tuner.best_params
