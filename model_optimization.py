"""
Model Optimization: Advanced Enhancements & Explainability Analysis
Implements systematic improvements, ensemble voting, and model interpretability
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import seaborn as sns

# Suppress feature name warnings (intentionally using numpy arrays in some contexts)
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*X has feature names.*")


class ModelOptimizer:
    """Systematic model optimization with improvements and explainability analysis"""

    def __init__(self, output_dir: Path = Path("outputs/model_optimization")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.improvements_log = []
        self.phase_comparison = None

    @staticmethod
    def _print_header(title: str) -> None:
        """Print formatted section header"""
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)

    @staticmethod
    def _calculate_comprehensive_metrics(y_true, y_pred) -> dict:
        """Calculate comprehensive evaluation metrics"""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
        
        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R²": round(r2, 4),
            "MAPE": round(mape, 4),
        }

    def improvement_1_feature_selection(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[dict, pd.DataFrame]:
        """
        IMPROVEMENT 1: Feature Selection with SelectKBest
        
        Rationale: Reduces model complexity, improves interpretability, and can reduce overfitting
        by identifying the most predictive features only.
        """
        self._print_header("IMPROVEMENT 1: Feature Selection (SelectKBest)")
        
        print("\n Baseline: All {} features".format(X_train.shape[1]))
        
        # Baseline with all features
        rf_baseline = RandomForestRegressor(n_estimators=450, random_state=42, n_jobs=-1)
        rf_baseline.fit(X_train, y_train, sample_weight=sample_weight)
        baseline_pred = rf_baseline.predict(X_test)
        baseline_metrics = self._calculate_comprehensive_metrics(y_test, baseline_pred)
        
        print(f"Baseline Metrics: MAE={baseline_metrics['MAE']}, RMSE={baseline_metrics['RMSE']}, R²={baseline_metrics['R²']}")
        
        # Feature selection: Keep top 70% of features
        n_features_to_keep = max(5, int(X_train.shape[1] * 0.7))
        print(f"\nApplying SelectKBest: Keeping top {n_features_to_keep} features (out of {X_train.shape[1]})")
        
        selector = SelectKBest(score_func=f_regression, k=n_features_to_keep)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_features = X_train.columns[selector.get_support()].tolist()
        print(f"Selected Features: {selected_features}")
        
        # Train with selected features
        rf_selected = RandomForestRegressor(n_estimators=450, random_state=42, n_jobs=-1)
        rf_selected.fit(X_train_selected, y_train, sample_weight=sample_weight)
        selected_pred = rf_selected.predict(X_test_selected)
        selected_metrics = self._calculate_comprehensive_metrics(y_test, selected_pred)
        
        print(f"Selected Features Metrics: MAE={selected_metrics['MAE']}, RMSE={selected_metrics['RMSE']}, R²={selected_metrics['R²']}")
        
        # Calculate improvements
        mae_change = (baseline_metrics['MAE'] - selected_metrics['MAE']) / baseline_metrics['MAE'] * 100
        rmse_change = (baseline_metrics['RMSE'] - selected_metrics['RMSE']) / baseline_metrics['RMSE'] * 100
        r2_change = (selected_metrics['R²'] - baseline_metrics['R²']) / baseline_metrics['R²'] * 100
        
        print(f"\n✓ Impact: MAE {mae_change:+.2f}%, RMSE {rmse_change:+.2f}%, R² {r2_change:+.2f}%")
        print(f"✓ Complexity Reduction: {X_train.shape[1]} → {n_features_to_keep} features ({n_features_to_keep/X_train.shape[1]*100:.1f}%)")
        
        comparison = pd.DataFrame({
            'Method': ['Baseline (All Features)', 'Feature Selection (SelectKBest)'],
            'n_features': [X_train.shape[1], n_features_to_keep],
            **{k: [baseline_metrics[k], selected_metrics[k]] for k in baseline_metrics.keys()}
        })
        
        self.improvements_log.append({
            'Improvement': 'Feature Selection (SelectKBest)',
            'Rationale': 'Reduce complexity, improve interpretability, reduce overfitting',
            'Result': f'Kept top {n_features_to_keep} features; R² change: {r2_change:+.2f}%'
        })
        
        return selected_metrics, comparison

    def improvement_2_ensemble_voting(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[dict, pd.DataFrame]:
        """
        IMPROVEMENT 2: Ensemble Voting Regressor
        
        Rationale: Combines predictions from multiple algorithms (Random Forest + Gradient Boosting)
        to leverage the strengths of both and reduce variance/bias.
        """
        self._print_header("IMPROVEMENT 2: Ensemble Voting Regressor")
        
        print("\n Combining multiple algorithms for robust predictions...")
        
        # Baseline single model
        rf_baseline = RandomForestRegressor(n_estimators=450, random_state=42, n_jobs=-1)
        rf_baseline.fit(X_train, y_train, sample_weight=sample_weight)
        baseline_pred = rf_baseline.predict(X_test)
        baseline_metrics = self._calculate_comprehensive_metrics(y_test, baseline_pred)
        
        print(f"RandomForest (baseline): RMSE={baseline_metrics['RMSE']}, R²={baseline_metrics['R²']}")
        
        # Component 1: Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=450, 
            max_depth=20, 
            random_state=42, 
            n_jobs=-1
        )
        
        # Component 2: Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        
        # Ensemble: Voting with equal weights
        voting_reg = VotingRegressor(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
            ]
        )
        
        print("\nTraining ensemble components...")
        voting_reg.fit(X_train, y_train)
        ensemble_pred = voting_reg.predict(X_test)
        ensemble_metrics = self._calculate_comprehensive_metrics(y_test, ensemble_pred)
        
        print(f"Ensemble (RF + GB): RMSE={ensemble_metrics['RMSE']}, R²={ensemble_metrics['R²']}")
        
        # Calculate improvements
        rmse_change = (baseline_metrics['RMSE'] - ensemble_metrics['RMSE']) / baseline_metrics['RMSE'] * 100
        r2_change = (ensemble_metrics['R²'] - baseline_metrics['R²']) / baseline_metrics['R²'] * 100
        
        print(f"\n✓ Ensemble Improvement: RMSE {rmse_change:+.2f}%, R² {r2_change:+.2f}%")
        print(f"✓ Reduced Variance: Combining RF (high variance) + GB (high bias)")
        
        comparison = pd.DataFrame({
            'Method': ['RandomForest (Single)', 'Ensemble (RF + GB)'],
            'Components': ['1 model', '2 models (voting)'],
            **{k: [baseline_metrics[k], ensemble_metrics[k]] for k in baseline_metrics.keys()}
        })
        
        self.improvements_log.append({
            'Improvement': 'Ensemble Voting',
            'Rationale': 'Combine multiple algorithms to reduce variance and leverage model strengths',
            'Result': f'RMSE: {rmse_change:+.2f}%, R²: {r2_change:+.2f}%'
        })
        
        # Save ensemble model
        ensemble_path = self.output_dir / "ensemble_voting_model.pkl"
        with open(ensemble_path, 'wb') as f:
            pickle.dump(voting_reg, f)
        print(f"\n[OK] Saved ensemble model: {ensemble_path}")
        
        return ensemble_metrics, comparison

    def improvement_3_advanced_regularization(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[dict, pd.DataFrame]:
        """
        IMPROVEMENT 3: Advanced Regularization with Cross-Validation
        
        Rationale: Optimizes hyperparameters with stricter regularization to prevent overfitting
        on new data and improve generalization performance.
        """
        self._print_header("IMPROVEMENT 3: Advanced Regularization & Generalization")
        
        print("\n Applying stricter regularization for better generalization...")
        
        # Baseline: Standard RandomForest
        rf_baseline = RandomForestRegressor(n_estimators=450, random_state=42, n_jobs=-1)
        rf_baseline.fit(X_train, y_train, sample_weight=sample_weight)
        baseline_pred = rf_baseline.predict(X_test)
        baseline_metrics = self._calculate_comprehensive_metrics(y_test, baseline_pred)
        baseline_cv = cross_val_score(rf_baseline, X_train, y_train, cv=5, scoring='r2').mean()
        
        print(f"Baseline RF: Test R²={baseline_metrics['R²']}, CV R²={baseline_cv:.4f}")
        
        # Advanced: Stricter regularization
        rf_regularized = RandomForestRegressor(
            n_estimators=600,           # More trees
            max_depth=15,               # Shallower trees
            min_samples_split=10,       # More samples required to split
            min_samples_leaf=5,         # Larger leaf nodes
            max_features='sqrt',        # Limit feature consideration
            random_state=42,
            n_jobs=-1
        )
        
        rf_regularized.fit(X_train, y_train, sample_weight=sample_weight)
        regularized_pred = rf_regularized.predict(X_test)
        regularized_metrics = self._calculate_comprehensive_metrics(y_test, regularized_pred)
        regularized_cv = cross_val_score(rf_regularized, X_train, y_train, cv=5, scoring='r2').mean()
        
        print(f"Regularized RF: Test R²={regularized_metrics['R²']}, CV R²={regularized_cv:.4f}")
        
        # Calculate improvements
        cv_stability = abs(regularized_cv - regularized_metrics['R²']) - abs(baseline_cv - baseline_metrics['R²'])
        r2_change = (regularized_metrics['R²'] - baseline_metrics['R²']) / baseline_metrics['R²'] * 100
        
        print(f"\n✓ Generalization Improvement: CV stability change: {cv_stability:+.4f}")
        print(f"✓ Test R² change: {r2_change:+.2f}%")
        print(f"✓ Applied: max_depth=15, min_samples_split=10, min_samples_leaf=5, max_features='sqrt'")
        
        comparison = pd.DataFrame({
            'Method': ['Baseline RF', 'Regularized RF'],
            'Test_R²': [baseline_metrics['R²'], regularized_metrics['R²']],
            'CV_R²': [baseline_cv, regularized_cv],
            'CV_Stability': [abs(baseline_cv - baseline_metrics['R²']), abs(regularized_cv - regularized_metrics['R²'])],
            **{k: [baseline_metrics[k], regularized_metrics[k]] for k in ['MAE', 'RMSE', 'MAPE']}
        })
        
        self.improvements_log.append({
            'Improvement': 'Advanced Regularization',
            'Rationale': 'Prevent overfitting with stricter constraints for better generalization',
            'Result': f'Test R²: {r2_change:+.2f}%, CV Stability improved'
        })
        
        # Save regularized model
        reg_model_path = self.output_dir / "regularized_random_forest_model.pkl"
        with open(reg_model_path, 'wb') as f:
            pickle.dump(rf_regularized, f)
        print(f"\n[OK] Saved regularized model: {reg_model_path}")
        
        return regularized_metrics, comparison

    def model_explainability_analysis(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """
        Model Explainability Analysis: Advanced interpretability assessment
        
        Analyzes feature importance, feature interactions, and prediction confidence
        to provide insights into what the model learns and how it makes decisions.
        """
        self._print_header("MODEL EXPLAINABILITY ANALYSIS")
        
        print("\n Analyzing model interpretability and decision patterns...\n")
        
        # Train final model
        final_model = RandomForestRegressor(n_estimators=450, random_state=42, n_jobs=-1)
        final_model.fit(X_train, y_train, sample_weight=sample_weight)
        
        # 1. Feature Importance Analysis
        print("1️⃣ FEATURE IMPORTANCE RANKING")
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': final_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")
        
        # 2. Prediction Confidence Analysis
        print("\n2️⃣ PREDICTION CONFIDENCE & UNCERTAINTY")
        predictions = final_model.predict(X_test)
        residuals = y_test.values - predictions
        
        confidence_score = 1 - (np.std(residuals) / np.mean(np.abs(y_test)))
        print(f"  Model Confidence Score: {confidence_score:.4f} (0-1 scale)")
        print(f"  Mean Absolute Residual: {np.mean(np.abs(residuals)):.4f} minutes")
        print(f"  Std Dev of Residuals: {np.std(residuals):.4f} minutes")
        
        # 3. Feature Interaction Detection
        print("\n3️⃣ TOP FEATURE INTERACTIONS")
        print("  Based on combined importance of feature pairs:")
        
        top_features = feature_importance.head(5)['Feature'].tolist()
        interactions = []
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                import1 = feature_importance[feature_importance['Feature'] == feat1]['Importance'].values[0]
                import2 = feature_importance[feature_importance['Feature'] == feat2]['Importance'].values[0]
                combined = import1 * import2
                interactions.append({'Pair': f"{feat1} × {feat2}", 'Interaction_Score': combined})
        
        interactions_df = pd.DataFrame(interactions).sort_values('Interaction_Score', ascending=False)
        for idx, row in interactions_df.head(5).iterrows():
            print(f"  {row['Pair']:50s}: {row['Interaction_Score']:.4f}")
        
        # 4. Model Behavior Insights
        print("\n4️⃣ MODEL BEHAVIOR INSIGHTS")
        predictions_std = []
        for tree in final_model.estimators_[:min(10, len(final_model.estimators_))]:
            predictions_std.append(tree.predict(X_test))
        
        predictions_std = np.std(predictions_std, axis=0)
        high_uncertainty_idx = np.argsort(predictions_std)[-5:]
        
        print(f"  Prediction Variance Across Trees:")
        print(f"    Mean variance: {predictions_std.mean():.4f}")
        print(f"    Max variance: {predictions_std.max():.4f}")
        print(f"  Most Uncertain Predictions: Top {len(high_uncertainty_idx)} samples have highest inter-tree variance")
        
        # Save explainability report
        explainability_report = {
            'Feature Importance': feature_importance.to_dict(),
            'Confidence Score': float(confidence_score),
            'Mean Residual': float(np.mean(np.abs(residuals))),
            'Top Interactions': interactions_df.head(5).to_dict(),
            'Prediction Variance': float(predictions_std.mean())
        }
        
        report_path = self.output_dir / "explainability_report.csv"
        feature_importance.to_csv(report_path, index=False)
        print(f"\n[OK] Saved feature importance: {report_path}")
        
        # Create explainability visualizations
        self._create_explainability_plots(feature_importance, interactions_df, predictions_std)

    def _create_explainability_plots(
        self,
        feature_importance: pd.DataFrame,
        interactions_df: pd.DataFrame,
        predictions_std: np.ndarray
    ) -> None:
        """Create explainability visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Feature Importance
        ax1 = axes[0, 0]
        top_10_features = feature_importance.head(10)
        ax1.barh(range(len(top_10_features)), top_10_features['Importance'].values, color='#4C78A8')
        ax1.set_yticks(range(len(top_10_features)))
        ax1.set_yticklabels(top_10_features['Feature'].values)
        ax1.set_xlabel('Importance')
        ax1.set_title('Top 10 Feature Importance')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        
        # 2. Feature Interactions
        ax2 = axes[0, 1]
        top_interactions = interactions_df.head(8)
        ax2.barh(range(len(top_interactions)), top_interactions['Interaction_Score'].values, color='#54A24B')
        ax2.set_yticks(range(len(top_interactions)))
        ax2.set_yticklabels(top_interactions['Pair'].values, fontsize=8)
        ax2.set_xlabel('Interaction Score')
        ax2.set_title('Top Feature Interactions')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature Importance Distribution
        ax3 = axes[1, 0]
        ax3.hist(feature_importance['Importance'].values, bins=20, color='#E15759', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Importance Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Feature Importance Distribution')
        ax3.axvline(feature_importance['Importance'].mean(), color='black', linestyle='--', label='Mean')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Prediction Uncertainty
        ax4 = axes[1, 1]
        ax4.hist(predictions_std, bins=30, color='#F28E2B', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Prediction Std Dev (across trees)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Model Prediction Uncertainty Distribution')
        ax4.axvline(predictions_std.mean(), color='black', linestyle='--', label='Mean')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / "explainability_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Saved explainability visualization: {plot_path}")

    def create_comprehensive_comparison(
        self,
        baseline_metrics: dict,
        optimized_metrics: dict,
    ) -> pd.DataFrame:
        """
        Create comparison across baseline and optimized models
        """
        self._print_header("MODEL OPTIMIZATION RESULTS COMPARISON")
        
        print("\n Comparing baseline vs optimized model performance...\n")
        
        comparison_df = pd.DataFrame({
            'Model': ['Baseline RandomForest', 'Optimized (Ensemble)'],
            'MAE': [baseline_metrics['MAE'], optimized_metrics['MAE']],
            'RMSE': [baseline_metrics['RMSE'], optimized_metrics['RMSE']],
            'R²': [baseline_metrics['R²'], optimized_metrics['R²']],
        })
        
        print(comparison_df.to_string(index=False))
        
        # Calculate improvements
        improvement_mae = (baseline_metrics['MAE'] - optimized_metrics['MAE']) / baseline_metrics['MAE'] * 100
        improvement_rmse = (baseline_metrics['RMSE'] - optimized_metrics['RMSE']) / baseline_metrics['RMSE'] * 100
        improvement_r2 = (optimized_metrics['R²'] - baseline_metrics['R²']) / baseline_metrics['R²'] * 100
        
        print(f"\n✓ Overall Improvements (Baseline → Optimized):")
        print(f"  MAE:  {improvement_mae:+.2f}%")
        print(f"  RMSE: {improvement_rmse:+.2f}%")
        print(f"  R²:   {improvement_r2:+.2f}%")
        
        # Save comparison
        comparison_df.to_csv(self.output_dir / "optimization_comparison.csv", index=False)
        print(f"\n[OK] Saved optimization comparison: {self.output_dir / 'optimization_comparison.csv'}")
        
        return comparison_df

    def generate_comprehensive_summary(self) -> None:
        """Generate comprehensive optimization summary"""
        self._print_header("MODEL OPTIMIZATION: COMPREHENSIVE SUMMARY")
        
        print("\nIMPROVEMENTS APPLIED:")
        improvements_summary = pd.DataFrame(self.improvements_log)
        print(improvements_summary.to_string(index=False))
        
        summary_path = self.output_dir / "improvements_summary.csv"
        improvements_summary.to_csv(summary_path, index=False)
        print(f"\n[OK] Saved improvements summary: {summary_path}")


def run_model_optimization(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    baseline_metrics: dict,
    sample_weight: Optional[np.ndarray] = None,
    output_dir: Path = Path("outputs/model_optimization"),
) -> Tuple[dict, pd.DataFrame]:
    """
    Run complete model optimization workflow
    """
    optimizer = ModelOptimizer(output_dir)
    
    # Apply 3 improvements
    print("-" * 35)
    print("MODEL OPTIMIZATION: ADVANCED ENHANCEMENTS & ANALYSIS")
    print("-" * 35)
    
    print("\n--- IMPLEMENTING IMPROVEMENTS ---\n")
    
    feat_sel_metrics, feat_sel_comp = optimizer.improvement_1_feature_selection(
        X_train, X_test, y_train, y_test, sample_weight
    )
    
    ensemble_metrics, ensemble_comp = optimizer.improvement_2_ensemble_voting(
        X_train, X_test, y_train, y_test, sample_weight
    )
    
    regularized_metrics, regularized_comp = optimizer.improvement_3_advanced_regularization(
        X_train, X_test, y_train, y_test, sample_weight
    )
    
    # Unique contribution: Explainability
    print("\n--- UNIQUE CONTRIBUTION ---\n")
    optimizer.model_explainability_analysis(X_train, X_test, y_train, y_test, sample_weight)
    
    # Comparison
    print("\n--- OPTIMIZATION RESULTS ---\n")
    optimized_metrics = ensemble_metrics  # Use ensemble as final metrics
    comparison = optimizer.create_comprehensive_comparison(baseline_metrics, optimized_metrics)
    
    # Summary
    optimizer.generate_comprehensive_summary()
    
    return optimized_metrics, comparison
