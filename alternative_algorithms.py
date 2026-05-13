"""
Alternative Algorithms Module
Implements 4 algorithms for traffic delay prediction:
1. LightGBM (Gradient Boosting)
2. SVR (Support Vector Regression)
3. KNN (K-Nearest Neighbors)
4. K-Means (Unsupervised Clustering)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class AlgorithmComparison:
    """Manage comparison of multiple regression algorithms"""

    def __init__(self, output_dir: Path = Path("outputs/algorithm_comparison")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_before_clusters: Optional[pd.DataFrame] = None
        self.results_after_clusters: Optional[pd.DataFrame] = None
        self.kmeans_model: Optional[KMeans] = None
        self.scalers: dict = {}

    @staticmethod
    def _print_step(step_num: int, title: str) -> None:
        """Print formatted step header"""
        print("\n" + "=" * 70)
        print(f"STEP {step_num}: {title}")
        print("=" * 70)

    @staticmethod
    def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate evaluation metrics"""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

    @staticmethod
    def _clean_feature_names(X: pd.DataFrame) -> pd.DataFrame:
        """Clean feature names for LightGBM compatibility (no special JSON characters)"""
        X_clean = X.copy()
        X_clean.columns = [
            col.replace("[", "_").replace("]", "_").replace("{", "_").replace("}", "_")
            .replace('"', "_").replace("'", "_").replace(":", "_").replace(" ", "_")
            for col in X_clean.columns
        ]
        return X_clean

    def step1_independent_testing(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        STEP 1: Train all algorithms independently on original data
        """
        self._print_step(1, "Independent Algorithm Testing")

        results = {}
        models = {}

        # RandomForest (Baseline)
        print("\n  Training RandomForest...")
        rf_model = RandomForestRegressor(n_estimators=450, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train, sample_weight=sample_weight)
        rf_pred = rf_model.predict(X_test)
        results["RandomForest"] = self._calculate_metrics(y_test, rf_pred)
        models["RandomForest"] = rf_model

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            print("  Training LightGBM...")
            
            lgb_model = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=40,
                max_depth=15,
                random_state=42,
                verbose=-1,
            )
            # Use numpy arrays to avoid feature name issues
            lgb_model.fit(X_train.values, y_train.values, sample_weight=sample_weight)
            lgb_pred = lgb_model.predict(X_test.values)
            results["LightGBM"] = self._calculate_metrics(y_test, lgb_pred)
            models["LightGBM"] = lgb_model
        else:
            print("LightGBM not available (install: pip install lightgbm)")

        # SVR (requires scaling)
        print("  Training SVR...")
        X_train_scaled = StandardScaler().fit_transform(X_train)
        X_test_scaled = StandardScaler().fit(X_train).transform(X_test)
        self.scalers["SVR"] = StandardScaler().fit(X_train)

        svr_model = SVR(kernel="rbf", C=100, epsilon=1.0)
        svr_model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
        svr_pred = svr_model.predict(X_test_scaled)
        results["SVR"] = self._calculate_metrics(y_test, svr_pred)
        models["SVR"] = svr_model

        # KNN
        print("  Training KNN...")
        knn_model = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)
        knn_model.fit(X_train, y_train)
        knn_pred = knn_model.predict(X_test)
        results["KNN"] = self._calculate_metrics(y_test, knn_pred)
        models["KNN"] = knn_model

        results_df = pd.DataFrame(results).T
        self.results_before_clusters = results_df
        self.models = models
        self.X_train_orig = X_train
        self.X_test_orig = X_test
        self.y_test = y_test

        print("\n  Results:")
        print(results_df.to_string())

        return results_df

    def step2_discover_clusters(
        self,
        X_train: pd.DataFrame,
        n_clusters: int = 5,
    ) -> tuple[KMeans, np.ndarray]:
        """
        STEP 2: Discover traffic patterns using K-Means clustering
        """
        self._print_step(2, "Discover Traffic Patterns with K-Means")

        print(f"\n  Finding {n_clusters} traffic clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_train)
        self.kmeans_model = kmeans
        self.train_clusters = clusters

        # Analyze clusters
        print(f"\n  Cluster Analysis:")
        cluster_info = []
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            count = mask.sum()
            avg_delay = self.y_train[mask].mean() if hasattr(self, 'y_train') else 0
            print(f"    Cluster {cluster_id}: {count:,} routes")
            cluster_info.append({"cluster_id": cluster_id, "route_count": count})

        return kmeans, clusters

    def step3_retrain_with_clusters(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        train_clusters: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        STEP 3: Add cluster feature and retrain all algorithms
        """
        self._print_step(3, "Retrain with Cluster Features")

        # Add cluster feature
        X_train_enhanced = X_train.copy()
        X_train_enhanced["traffic_cluster"] = train_clusters

        test_clusters = self.kmeans_model.predict(X_test)
        X_test_enhanced = X_test.copy()
        X_test_enhanced["traffic_cluster"] = test_clusters

        results = {}

        # RandomForest
        print("\n  Training RandomForest (with clusters)...")
        rf_model = RandomForestRegressor(n_estimators=450, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_enhanced, y_train, sample_weight=sample_weight)
        rf_pred = rf_model.predict(X_test_enhanced)
        results["RandomForest"] = self._calculate_metrics(y_test, rf_pred)

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            print("  Training LightGBM (with clusters)...")
            
            lgb_model = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=40,
                max_depth=15,
                random_state=42,
                verbose=-1,
            )
            # Use numpy arrays to avoid feature name issues
            lgb_model.fit(X_train_enhanced.values, y_train.values, sample_weight=sample_weight)
            lgb_pred = lgb_model.predict(X_test_enhanced.values)
            results["LightGBM"] = self._calculate_metrics(y_test, lgb_pred)

        # SVR (refit scaler with enhanced features)
        print("  Training SVR (with clusters)...")
        scaler_enhanced = StandardScaler()
        X_train_scaled = scaler_enhanced.fit_transform(X_train_enhanced)
        X_test_scaled = scaler_enhanced.transform(X_test_enhanced)

        svr_model = SVR(kernel="rbf", C=100, epsilon=1.0)
        svr_model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
        svr_pred = svr_model.predict(X_test_scaled)
        results["SVR"] = self._calculate_metrics(y_test, svr_pred)

        # KNN
        print("  Training KNN (with clusters)...")
        knn_model = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)
        knn_model.fit(X_train_enhanced, y_train)
        knn_pred = knn_model.predict(X_test_enhanced)
        results["KNN"] = self._calculate_metrics(y_test, knn_pred)

        results_df = pd.DataFrame(results).T
        self.results_after_clusters = results_df

        print("\n  Results (with cluster feature):")
        print(results_df.to_string())

        return results_df

    def step4_generate_report(self) -> None:
        """
        STEP 4: Generate comprehensive comparison report
        """
        self._print_step(4, "Generate Comparison Report")

        # Save comparison tables
        before_path = self.output_dir / "algorithms_before_clusters.csv"
        after_path = self.output_dir / "algorithms_after_clusters.csv"

        self.results_before_clusters.to_csv(before_path)
        self.results_after_clusters.to_csv(after_path)

        print(f"\n  [OK] Saved: {before_path}")
        print(f"  [OK] Saved: {after_path}")

        # Create comparison visualization
        comparison_df = pd.DataFrame({
            "Before Clusters": self.results_before_clusters["RMSE"],
            "After Clusters": self.results_after_clusters["RMSE"],
        })
        comparison_df["Improvement %"] = (
            (comparison_df["Before Clusters"] - comparison_df["After Clusters"])
            / comparison_df["Before Clusters"]
            * 100
        )

        print("\n  Performance Improvement Summary:")
        print(comparison_df.to_string())

        # Generate visualization
        self._create_comparison_plots()

        print(f"\n  [OK] Report complete. Outputs saved to: {self.output_dir}")

    def _create_comparison_plots(self) -> None:
        """Create comparison visualizations"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Before clusters
        ax1 = axes[0]
        self.results_before_clusters["RMSE"].sort_values().plot(
            kind="barh", ax=ax1, color="#4C78A8"
        )
        ax1.set_title("Algorithm Performance (Before Clusters)")
        ax1.set_xlabel("RMSE (minutes)")
        ax1.invert_yaxis()

        # After clusters
        ax2 = axes[1]
        self.results_after_clusters["RMSE"].sort_values().plot(
            kind="barh", ax=ax2, color="#54A24B"
        )
        ax2.set_title("Algorithm Performance (After Clusters)")
        ax2.set_xlabel("RMSE (minutes)")
        ax2.invert_yaxis()

        plt.tight_layout()
        plot_path = self.output_dir / "algorithm_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  [OK] Saved plot: {plot_path}")


def run_algorithm_comparison(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    output_dir: Path = Path("outputs/algorithm_comparison"),
) -> AlgorithmComparison:
    """
    Run complete 4-step algorithm comparison pipeline
    """
    comparison = AlgorithmComparison(output_dir)

    # Store for later use
    comparison.y_train = y_train
    comparison.y_test = y_test

    comparison.step1_independent_testing(X_train, X_test, y_train, y_test, sample_weight)

    kmeans, train_clusters = comparison.step2_discover_clusters(X_train, n_clusters=5)

    comparison.step3_retrain_with_clusters(
        X_train, X_test, y_train, y_test, train_clusters, sample_weight
    )

    comparison.step4_generate_report()

    return comparison
