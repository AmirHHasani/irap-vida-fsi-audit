"""
stage1_tuning.py

Hyperparameter tuning for Stage 1 models with BY_ROAD cross-validation
to prevent data leakage. This module maintains the same StratifiedGroupKFold structure
used in the main pipeline.

Features:
- Prevents data leakage with proper BY_ROAD CV
- Tracks parameter importance
- Saves tuning results and visualizations
- Compatible with SHAP analysis (uses simple, tree-friendly parameters)
- Supports CatBoost, LightGBM, and XGBoost
"""

# Set environment flags before numpy/pandas imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Also try this for Intel MKL
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import sys
import warnings
warnings.filterwarnings('ignore')

# Now import everything else
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_contour,
    plot_parallel_coordinate
)
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import stage1_config as cfg
from stage1_data_loader import load_data
from stage1_feature_engineering import prepare_features, fit_transform_preprocessor
from stage1_utils import TargetTransformer
from pathlib import Path
import joblib
import json
from datetime import datetime
import time
import re

# Import model libraries with specific order (tree models first).
lgb = None
ctb = None
xgb = None

# Import tree-based models first
try:
    import lightgbm as lgb
    print("[INFO] LightGBM loaded successfully")
except Exception as e:
    print(f"[WARN] Could not load LightGBM: {e}")

try:
    import catboost as ctb
    print("[INFO] CatBoost loaded successfully")
except Exception as e:
    print(f"[WARN] Could not load CatBoost: {e}")

try:
    import xgboost as xgb
    print("[INFO] XGBoost loaded successfully")
except Exception as e:
    print(f"[WARN] Could not load XGBoost: {e}")



class OptunaModelTuner:
    """
    Hyperparameter tuning using Optuna with BY_ROAD cross-validation.
    
    This class ensures no data leakage by using the same StratifiedGroupKFold
    structure as the main Stage 1 pipeline.
    """
    
    def __init__(self, model_type, n_trials=150, cv_folds=5, random_state=42,
                 output_dir=None, study_name=None):
        """
        Initialize the tuner.
        
        Args:
            model_type: Model to tune ('CatBoost', 'LightGBM', 'XGBoost')
            n_trials: Number of trials
            cv_folds: Number of CV folds (default: 5)
            random_state: Random seed
            output_dir: Directory to save results
            study_name: Custom study name (default: auto-generated)
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        # Cache sanitized categorical feature names to match prepare_features output
        self.sanitized_categorical_feature_names = self._sanitize_feature_list(cfg.CATEGORICAL_FEATURES)
        self.sanitized_numeric_feature_names = self._sanitize_feature_list(cfg.NUMERICAL_FEATURES)
        
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_dir = Path(f'stage1_tuning/{timestamp}_{model_type}')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Study name
        if study_name is None:
            study_name = f'{model_type}_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.study_name = study_name
        
        # Data containers (filled during tune())
        self.X = None
        self.y = None
        self.road_col = None
        self.stratify_key = None
        self.study = None
        self.best_params = None
        self.target_transformer = None
        
        print(f"\n{'='*70}")
        print(f"Hyperparameter Tuning for {model_type}")
        print(f"{'='*70}")
        print(f"Study name: {study_name}")
        print(f"Trials: {n_trials}")
        print(f"CV Folds: {cv_folds}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")

    @staticmethod
    def _sanitize_feature_name(name: str) -> str:
        """Apply same sanitation logic as prepare_features for config columns."""
        name = str(name).lower()
        name = re.sub(r'[\s\(\)\-\/]+', '_', name)
        name = re.sub(r'[^a-z0-9_]+', '', name)
        name = re.sub(r'[_]+', '_', name)
        return name.strip('_')

    def _sanitize_feature_list(self, names):
        seen = set()
        sanitized = []
        for n in names:
            s = self._sanitize_feature_name(n)
            if s and s not in seen:
                sanitized.append(s)
                seen.add(s)
        return sanitized
    
    def load_and_prepare_data(self):
        """
        Load data and prepare for CV (same as main pipeline).
        Maintains the BY_ROAD grouping structure.
        """
        print("Loading and preparing data...")
        
        # Load data
        df = load_data(cfg.SEGMENTS_DATA_CSV)

        # Prepare features
        X, y, metadata = prepare_features(
            df,
            target_col=cfg.TARGET_COL,
            metadata_cols=cfg.METADATA_COLS,
            feature_exclusions=cfg.FEATURE_EXCLUSIONS
        )

        # Align with main pipeline: use configured TargetTransformer (log_offset, log1p, etc.)
        original_min, original_max = y.min(), y.max()
        target_transformer_cfg = getattr(cfg, 'TARGET_TRANSFORMATION', {})
        self.target_transformer = TargetTransformer(target_transformer_cfg)
        y = pd.Series(self.target_transformer.fit_transform(y), index=y.index, name=y.name)
        transformed_min, transformed_max = y.min(), y.max()
        print(
            f"   [INFO] Target transformed using method='{self.target_transformer.method}'. "
            f"Original range [{original_min:.4f}, {original_max:.4f}] -> transformed range [{transformed_min:.4f}, {transformed_max:.4f}]"
        )
        
        # Extract road column for grouping
        road_col = metadata[cfg.ROAD_COLUMN_NAME].astype(str)
        
        # Create stratification key (same as main pipeline)
        # Stratify by road-level mean risk (quantile binning)
        road_mean = y.groupby(road_col).mean()
        try:
            stratify_key = pd.qcut(road_mean, q=self.cv_folds, labels=False, duplicates='drop')
        except ValueError:
            bins = min(self.cv_folds, max(2, len(road_mean.unique()) - 1))
            stratify_key = pd.cut(road_mean, bins=bins, labels=False, duplicates='drop')
        
        # Align stratification key to segments
        aligned_key = road_col.map(stratify_key).fillna(int(stratify_key.median())).astype(int)
        
        print(f"Data loaded: {len(X)} segments, {X.shape[1]} features")
        print(f"Unique roads: {road_col.nunique()}")
        print(f"Target range ({self.target_transformer.method}): [{y.min():.4f}, {y.max():.4f}]")
        
        self.X = X
        self.y = y
        self.road_col = road_col
        self.stratify_key = aligned_key
        
        return X, y, road_col, aligned_key
    
    def suggest_catboost_params(self, trial):
        """Suggest CatBoost hyperparameters (SHAP-compatible, GPU-compatible)."""
        # CatBoost: subsample requires bootstrap_type='Bernoulli' or 'MVS'
        # Default 'Bayesian' doesn't support subsample parameter
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        
        params = {
            'iterations': trial.suggest_int('iterations', 500, 3000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'depth': trial.suggest_int('depth', 6, 12),  # Expanded: 6-12 (includes 10, allows deeper)
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
            'random_seed': self.random_state,
            'verbose': 0,
            'loss_function': 'RMSE'
        }
        
        # Add rsm (column sampling) only if NOT using GPU
        # GPU only supports rsm for pairwise ranking, not regression
        if not cfg.USE_GPU:
            params['rsm'] = trial.suggest_float('rsm', 0.5, 1.0)  # colsample_bylevel
        
        # Add subsample with compatible bootstrap_type
        if subsample < 1.0:
            params['subsample'] = subsample
            params['bootstrap_type'] = 'Bernoulli'  # Required for subsample
        
        return params
    
    def suggest_lightgbm_params(self, trial):
        """Suggest LightGBM hyperparameters (SHAP-compatible)."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 128),  # Focused: 31-128 (standard range)
            'max_depth': trial.suggest_int('max_depth', 6, 15),  # Expanded: 6-15 (allows deeper trees)
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Tightened: 0.6-1.0 (avoid too sparse)
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # Tightened: 0.6-1.0
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),  # Expanded: 0-2 (more regularization)
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),  # Adjusted: 10-100
            'random_state': self.random_state,
            'verbosity': -1  # Changed from 'verbose' to 'verbosity'
        }
    
    def suggest_xgboost_params(self, trial):
        """Suggest XGBoost hyperparameters (SHAP-compatible)."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 6, 15),  # Expanded: 6-15 (allows deeper trees)
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Tightened: 0.6-1.0
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # Tightened: 0.6-1.0
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),  # Expanded: 0-2
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'random_state': self.random_state,
            'objective': 'reg:squarederror',
            'enable_categorical': True,
            'tree_method': 'hist'
        }
    
    def objective(self, trial):
        """
        Objective function with BY_ROAD cross-validation.
        
        This ensures no road appears in both train and test, preventing data leakage.
        """
        # Suggest parameters based on model type
        if self.model_type == 'CatBoost':
            params = self.suggest_catboost_params(trial)
        elif self.model_type == 'LightGBM':
            params = self.suggest_lightgbm_params(trial)
        elif self.model_type == 'XGBoost':
            params = self.suggest_xgboost_params(trial)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Run BY_ROAD cross-validation
        splitter = StratifiedGroupKFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        fold_scores = []
        fold_mae = []
        fold_rmse = []
        
        for fold_id, (train_idx, test_idx) in enumerate(
            splitter.split(self.X, self.stratify_key, groups=self.road_col), 1
        ):
            # Split data
            X_train_fold = self.X.iloc[train_idx]
            X_test_fold = self.X.iloc[test_idx]
            y_train_fold = self.y.iloc[train_idx]
            y_test_fold = self.y.iloc[test_idx]
            
            # Standard preprocessing for tree models (CatBoost, LightGBM, XGBoost)
            X_train_proc, X_test_proc, preprocessor, feature_names = fit_transform_preprocessor(
                X_train_fold, X_test_fold,
                cfg.NUMERICAL_FEATURES, cfg.CATEGORICAL_FEATURES,
                for_tree_models=True, scale_numeric=False
            )
            
            # Train model based on type
            try:
                if self.model_type == 'CatBoost':
                    # Add GPU support for CatBoost
                    if cfg.USE_GPU:
                        params['task_type'] = 'GPU'
                        params['devices'] = '0'  # Use first GPU
                    model = ctb.CatBoostRegressor(**params)
                    model.fit(X_train_proc, y_train_fold, verbose=False)
                    
                elif self.model_type == 'LightGBM':
                    # Add GPU support for LightGBM
                    if cfg.USE_GPU:
                        params['device'] = 'gpu'
                        params['gpu_platform_id'] = 0
                        params['gpu_device_id'] = 0
                    model = lgb.LGBMRegressor(**params)
                    # LightGBM uses 'verbosity' parameter, not 'verbose'
                    model.fit(X_train_proc, y_train_fold)
                    
                elif self.model_type == 'XGBoost':
                    # Add GPU support for XGBoost
                    if cfg.USE_GPU:
                        params['tree_method'] = 'gpu_hist'
                        params['gpu_id'] = 0
                    # Note: XGBoost needs categorical data as integers for enable_categorical
                    model = xgb.XGBRegressor(**params)
                    model.fit(X_train_proc, y_train_fold, verbose=False)
                    
                else:
                    raise ValueError(f"Unsupported model: {self.model_type}")
                
                # Predict and compute metrics
                y_pred = model.predict(X_test_proc)
                r2 = r2_score(y_test_fold, y_pred)
                mae = mean_absolute_error(y_test_fold, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
                
                fold_scores.append(r2)
                fold_mae.append(mae)
                fold_rmse.append(rmse)
                
            except Exception as e:
                print(f"  Trial {trial.number} Fold {fold_id} failed: {e}")
                # Return a bad score to indicate failure
                return -999.0
        
        # Return mean R² across folds (Maximizes by default)
        mean_r2 = np.mean(fold_scores)
        mean_mae = np.mean(fold_mae)
        mean_rmse = np.mean(fold_rmse)
        
        # Store additional metrics for reporting
        trial.set_user_attr('mean_r2', mean_r2)
        trial.set_user_attr('mean_mae', mean_mae)
        trial.set_user_attr('mean_rmse', mean_rmse)
        trial.set_user_attr('fold_r2_scores', fold_scores)
        
        print(f"  Trial {trial.number}: R²={mean_r2:.4f}, MAE={mean_mae:.5f}, RMSE={mean_rmse:.5f}")
        
        return mean_r2  # will maximize this
    
    def tune(self):
        """
        Run hyperparameter tuning.
        
        Returns:
            best_params: Dictionary of best hyperparameters
            study: Study object
        """
        # Load data first
        if self.X is None:
            self.load_and_prepare_data()
        
        # Create study
        print(f"\nStarting tuning with {self.n_trials} trials...")
        print(f"This may take a while depending on your hardware and data size.\n")
        
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',  # Maximize R²
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            catch=(Exception,)  # Don't stop on individual trial failures
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        best_value = self.study.best_value
        
        print(f"\n{'='*70}")
        print(f"Tuning Complete!")
        print(f"{'='*70}")
        print(f"Best R²: {best_value:.4f}")
        print(f"\nBest Parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"{'='*70}\n")
        
        # Save results
        self.save_results()
        
        # Generate visualizations
        self.generate_visualizations()
        
        return self.best_params, self.study
    
    def save_results(self):
        """Save tuning results to disk."""
        print("Saving tuning results...")
        
        # Save best parameters as JSON
        best_params_path = self.output_dir / 'best_params.json'
        with open(best_params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        print(f"  [OK] Best parameters: {best_params_path}")
        
        # Save study object
        study_path = self.output_dir / 'study.pkl'
        joblib.dump(self.study, study_path)
        print(f"  [OK] Study object: {study_path}")
        
        # Save trials dataframe
        trials_df = self.study.trials_dataframe()
        trials_csv_path = self.output_dir / 'trials_dataframe.csv'
        trials_df.to_csv(trials_csv_path, index=False)
        print(f"  [OK] Trials dataframe: {trials_csv_path}")
        
        # Save parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            importance_df = pd.DataFrame([
                {'parameter': k, 'importance': v}
                for k, v in importance.items()
            ]).sort_values('importance', ascending=False)
            
            importance_path = self.output_dir / 'parameter_importance.csv'
            importance_df.to_csv(importance_path, index=False)
            print(f"  [OK] Parameter importance: {importance_path}")
        except Exception as e:
            print(f"  [WARN] Could not compute parameter importance: {e}")
        
        # Save summary report
        summary_path = self.output_dir / 'tuning_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Hyperparameter Tuning Summary\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Model Type: {self.model_type}\n")
            f.write(f"Study Name: {self.study_name}\n")
            f.write(f"Number of Trials: {len(self.study.trials)}\n")
            f.write(f"Best Trial: {self.study.best_trial.number}\n")
            f.write(f"Best R²: {self.study.best_value:.4f}\n\n")
            
            f.write(f"Best Parameters:\n")
            f.write(f"{'-'*70}\n")
            for param, value in self.best_params.items():
                f.write(f"  {param}: {value}\n")
            
            f.write(f"\n{'-'*70}\n")
            f.write(f"Cross-Validation Details:\n")
            f.write(f"  CV Strategy: BY_ROAD StratifiedGroupKFold\n")
            f.write(f"  Number of Folds: {self.cv_folds}\n")
            f.write(f"  Total Segments: {len(self.X)}\n")
            f.write(f"  Unique Roads: {self.road_col.nunique()}\n")
            
            # Best trial metrics
            best_trial = self.study.best_trial
            f.write(f"\n{'-'*70}\n")
            f.write(f"Best Trial Metrics:\n")
            
            # Safe formatting with fallback for string values
            mean_r2 = best_trial.user_attrs.get('mean_r2', 'N/A')
            mean_mae = best_trial.user_attrs.get('mean_mae', 'N/A')
            mean_rmse = best_trial.user_attrs.get('mean_rmse', 'N/A')
            
            if isinstance(mean_r2, (int, float)):
                f.write(f"  Mean R²: {mean_r2:.4f}\n")
            else:
                f.write(f"  Mean R²: {mean_r2}\n")
                
            if isinstance(mean_mae, (int, float)):
                f.write(f"  Mean MAE: {mean_mae:.5f}\n")
            else:
                f.write(f"  Mean MAE: {mean_mae}\n")
                
            if isinstance(mean_rmse, (int, float)):
                f.write(f"  Mean RMSE: {mean_rmse:.5f}\n")
            else:
                f.write(f"  Mean RMSE: {mean_rmse}\n")
            
            fold_scores = best_trial.user_attrs.get('fold_r2_scores', [])
            if fold_scores:
                f.write(f"\n  Fold R² Scores:\n")
                for i, score in enumerate(fold_scores, 1):
                    f.write(f"    Fold {i}: {score:.4f}\n")
        
        print(f"  [OK] Summary report: {summary_path}")
        print()
    
    def generate_visualizations(self):
        """Generate visualization plots."""
        print("Generating visualizations...")
        
        try:
            # Optimization history
            fig = plot_optimization_history(self.study)
            fig.write_html(str(self.output_dir / 'optimization_history.html'))
            print(f"  [OK] Optimization history")
            
            # Parameter importance
            fig = plot_param_importances(self.study)
            fig.write_html(str(self.output_dir / 'parameter_importance.html'))
            print(f"  [OK] Parameter importance")
            
            # Slice plot
            fig = plot_slice(self.study)
            fig.write_html(str(self.output_dir / 'parameter_slice.html'))
            print(f"  [OK] Parameter slice plot")
            
            # Contour plot (top 2 important parameters)
            try:
                fig = plot_contour(self.study)
                fig.write_html(str(self.output_dir / 'parameter_contour.html'))
                print(f"  [OK] Parameter contour plot")
            except:
                print(f"  [WARN] Could not generate contour plot")
            
            # Parallel coordinate plot
            fig = plot_parallel_coordinate(self.study)
            fig.write_html(str(self.output_dir / 'parallel_coordinate.html'))
            print(f"  [OK] Parallel coordinate plot")
            
        except Exception as e:
            print(f"  [WARN] Some visualizations failed: {e}")
        
        print()


def run_tuning_for_model(
    model_type,
    n_trials=100,
    cv_folds=5,
    random_state=42
):
    """
    Convenience function to run tuning for a single model.
    
    Args:
        model_type: 'CatBoost', 'LightGBM', or 'XGBoost'
        n_trials: Number of trials
        cv_folds: Number of CV folds
        random_state: Random seed
    
    Returns:
        best_params: Dictionary of best hyperparameters
        study: study object
    """
    tuner = OptunaModelTuner(
        model_type=model_type,
        n_trials=n_trials,
        cv_folds=cv_folds,
        random_state=random_state
    )
    
    best_params, study = tuner.tune()
    
    return best_params, study


def run_tuning_for_all_models(
    n_trials=100,
    cv_folds=5,
    random_state=42
):
    """
    Run tuning for all available models.
    
    Args:
        n_trials: Number of trials per model
        cv_folds: Number of CV folds
        random_state: Random seed
    
    Returns:
        results: Dictionary with model names as keys and (best_params, study) as values
    """
    results = {}
    
    # Check which models are available
    available_models = []
    if ctb is not None:
        available_models.append('CatBoost')
    if lgb is not None:
        available_models.append('LightGBM')
    if xgb is not None:
        available_models.append('XGBoost')
    
    print(f"Available models for tuning: {available_models}")
    print()
    
    for model_type in available_models:
        print(f"\n{'#'*70}")
        print(f"# Tuning {model_type}")
        print(f"{'#'*70}\n")
        
        try:
            best_params, study = run_tuning_for_model(
                model_type=model_type,
                n_trials=n_trials,
                cv_folds=cv_folds,
                random_state=random_state
            )
            results[model_type] = (best_params, study)
        except Exception as e:
            print(f"ERROR: Tuning failed for {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    combined_output_dir = Path('stage1_tuning/combined_results')
    combined_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all best parameters in one JSON
    all_best_params = {
        model: params 
        for model, (params, _) in results.items()
    }
    with open(combined_output_dir / 'all_best_params.json', 'w') as f:
        json.dump(all_best_params, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"All Tuning Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: stage1_tuning/")
    print(f"Combined results: {combined_output_dir}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == '__main__':
    """
    Example usage:
    
    # Tune a single model
    python stage1_tuning.py --model CatBoost --trials 50
    
    # Tune all models
    python stage1_tuning.py --all --trials 50
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument('--model', type=str, default=None,
                        help='Model to tune: CatBoost, LightGBM, XGBoost')
    parser.add_argument('--all', action='store_true',
                        help='Tune all available models')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of trials (default: 100)')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if args.all:
        # Tune all models
        run_tuning_for_all_models(
            n_trials=args.trials,
            cv_folds=args.folds,
            random_state=args.seed
        )
    elif args.model:
        # Tune single model
        run_tuning_for_model(
            model_type=args.model,
            n_trials=args.trials,
            cv_folds=args.folds,
            random_state=args.seed
        )
    else:
        print("Please specify --model <ModelName> or --all")
        print("Example: python stage1_tuning.py --model CatBoost --trials 50")
