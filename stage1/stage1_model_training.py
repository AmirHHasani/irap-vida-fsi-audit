# stage1_model_training.py
"""
Model training and evaluation for the Stage 1 analysis.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import catboost as ctb
except Exception:
    ctb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
import numpy as np # ADDED: Import numpy
from sklearn.preprocessing import StandardScaler
import stage1_config as cfg
from pathlib import Path
import joblib
from stage1_utils import build_stratification_key


# Import the new preprocessing function
# Only import the split preprocessor
# Import the new unified preprocessing helpers
from stage1_feature_engineering import preprocess_data_split, fit_transform_preprocessor
# Import feature lists from config to pass to the preprocessor
from stage1_config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

# For Cross-Validation strategies
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, GroupKFold
import numpy as np


def train_and_evaluate_model(X, y, model_type, split_strategy, target_col, road_col_name, test_size, random_state):
    """
    Orchestrates the model training and evaluation process, including the three-mode split logic.
    Handles 'RANDOM' and 'DIAGNOSTIC' strategies. 'BY_ROAD' is handled by run_by_road_cross_validation.
    """
    print("   Splitting data into training and testing sets...")
    # Drop the road column from features if it exists, but keep it for grouping
    X_features = X.drop(columns=[road_col_name], errors='ignore')

    if split_strategy == 'DIAGNOSTIC':
        full_df = pd.concat([X, y], axis=1)
        n_test = int(len(full_df) * test_size)
        test_indices = full_df.nlargest(n_test, target_col).index
        train_indices = full_df.index.difference(test_indices)
        X_train, X_test = X_features.loc[train_indices], X_features.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        print(f"   Using High-Risk Diagnostic Split: Test set contains the top {len(X_test)} riskiest segments.")
    elif split_strategy == 'RANDOM':
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=random_state
        )
        print(f"   Using Standard Random Split: {len(X_train)} train / {len(X_test)} test.")
    else:
        raise ValueError(f"Unsupported split strategy '{split_strategy}' in this function.")


    # --- Step B: Preprocessing (Post-Split to Prevent Leakage) ---
    print("   Building unified preprocessor and transforming splits...")
    # Fit preprocessor on training split only and transform both splits. Keep original X_train/X_test
    X_train_proc, X_test_proc, preprocessor, feature_names = fit_transform_preprocessor(
        X_train, X_test, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, for_tree_models=True, scale_numeric=False
    )

    # --- Step C: Model Definition ---
    print(f"   Defining model: {model_type}")
    model = None
    if model_type == 'CatBoost':
        model = ctb.CatBoostRegressor(
            # Tuned hyperparameters from Optuna (2025-10-16 run, R²=0.699)
            iterations=2700,
            learning_rate=0.057180837980277475,
            depth=10,
            l2_leaf_reg=1.910118305911902,
            min_data_in_leaf=30,
            subsample=0.5770241989636643,
            loss_function='RMSE',
            random_seed=random_state,
            verbose=0,
            cat_features=CATEGORICAL_FEATURES,
            bootstrap_type='Bernoulli'
        )

    elif model_type == 'LightGBM':
        model = lgb.LGBMRegressor(
            # Tuned hyperparameters from Optuna (2025-10-16 run, R²=0.689)
            n_estimators=1600,
            learning_rate=0.013102532483752814,
            num_leaves=38,
            max_depth=15,
            subsample=0.808191937618305,
            colsample_bytree=0.8161071000274498,
            reg_alpha=0.4844922766155489,
            reg_lambda=9.632501050596495,
            min_child_samples=28,
            random_state=random_state,
            verbosity=-1
        )

    elif model_type == 'XGBoost':
        model = xgb.XGBRegressor(
            # Tuned hyperparameters from Optuna (2025-10-16 run, R²=0.647)
            n_estimators=2900,
            learning_rate=0.031267285253120426,
            max_depth=8,
            subsample=0.6946000930203216,
            colsample_bytree=0.7944888223337778,
            reg_alpha=1.5556540652500264,
            reg_lambda=3.9075969867624596,
            min_child_weight=3,
            gamma=0.0009420866279373377,
            objective='reg:squarederror',
            random_state=random_state,
            enable_categorical=True
        )

    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")

    # --- Step D: Training ---
    if model:
        print("   Training model...")
        # With OrdinalEncoder preprocessing, treat encoded categorical columns as numeric
        # to avoid mismatched dtype issues with LightGBM's native categorical handling.
        model.fit(X_train_proc, y_train)
        print("   Model training complete.")
    elif not model:
        print(f"   Skipping evaluation for {model_type} as it could not be initialized.")
        return None, {}, None, None, None, None

    # --- Step E: Evaluation ---
    print("   Evaluating model on the test set...")
    # Use processed test matrix for prediction
    y_pred = model.predict(X_test_proc)
    
    metrics = {
        "Model Type": model_type,
        "Split Strategy": split_strategy,
        "Test R2": r2_score(y_test, y_pred),
        "Test MAE": mean_absolute_error(y_test, y_pred),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    # --- Step F: Return ---
    # Return processed feature splits so downstream SHAP and explanation code use the exact model inputs
    return model, metrics, X_train_proc, y_train, X_test_proc, y_test


def get_model_instance(model_type, random_state, cat_features=None):
    """Factory that returns an unfitted model instance for the given type."""
    if cat_features is None:
        cat_features = []
    return _get_model_instance_impl(model_type, random_state, cat_features)
def train_final_model(X, y, model_type, random_state, cat_features=None):
    """
    Fits a final model on all provided data with consistent preprocessing.
    Args:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target data.
        model_type (str): The type of model to fit.
        random_state (int): The random state for reproducibility.
        cat_features (list, optional): List of categorical features for models that need it.
    Returns:
        tuple: (model, X_processed)
    """
    # Determine features to use
    cfg_num = NUMERICAL_FEATURES if NUMERICAL_FEATURES else []
    cfg_cat = (cat_features if cat_features is not None else CATEGORICAL_FEATURES) or []
    present_num = [c for c in cfg_num if c in X.columns]
    present_cat = [c for c in cfg_cat if c in X.columns]
    if not present_num and not present_cat:
        # Derive from X dtypes if config features are not present
        derived = [c for c in X.columns if c not in {'segment_id', 'road_id', 'latitude', 'longitude'}]
        present_num = [c for c in derived if X[c].dtype.kind in 'fiubc']
        present_cat = [c for c in derived if c not in present_num]

    # Sanitize column names to align with fit_transform_preprocessor expectations
    import re
    def _sanitize(name: str) -> str:
        name = str(name).lower()
        name = re.sub(r'[\s\(\)\-\/]+', '_', name)
        name = re.sub(r'[^a-z0-9_]+', '', name)
        name = re.sub(r'[_]+', '_', name)
        return name.strip('_')

    X_san = X.copy()
    X_san.rename(columns={c: _sanitize(c) for c in X_san.columns}, inplace=True)
    present_num_san = [_sanitize(c) for c in present_num if _sanitize(c) in X_san.columns]
    present_cat_san = [_sanitize(c) for c in present_cat if _sanitize(c) in X_san.columns]
    # If sanitation caused lists to be empty, fall back to using derived lists on sanitized X
    if not present_num_san and not present_cat_san:
        derived = [c for c in X_san.columns if c not in {'segment_id', 'road_id', 'latitude', 'longitude'}]
        present_num_san = [c for c in derived if X_san[c].dtype.kind in 'fiubc']
        present_cat_san = [c for c in derived if c not in present_num_san]

    # Preprocess the data using the sanitized feature lists
    X_processed, _, preprocessor, feature_names = fit_transform_preprocessor(
        X_san, X_san, present_num_san, present_cat_san, for_tree_models=True, scale_numeric=False
    )
    
    # Get the model instance
    model = get_model_instance(model_type, random_state, present_cat_san)
    
    # Fit the model
    model.fit(X_processed, y)
    
    return model, X_processed
    # Heuristic: for small datasets skip GPU for tree models (overhead > gain)
    use_gpu_tree = cfg.USE_GPU and (cat_features is not None)  # baseline condition
    # We cannot infer row count here; caller may set dynamically later if needed.

    if model_type == 'CatBoost':
        params = dict(
            random_state=random_state, 
            verbose=0, 
            cat_features=cat_features,
            # Tuned hyperparameters from Optuna (2025-10-16 run, R²=0.699)
            iterations=2700,
            learning_rate=0.057180837980277475,
            depth=10,
            l2_leaf_reg=1.910118305911902,
            min_data_in_leaf=30,
            subsample=0.5770241989636643,
            bootstrap_type='Bernoulli'
        )
        if cfg.USE_GPU and cfg.GPU_ENABLE_CATBOOST:
            params.update(task_type='GPU', devices=str(cfg.GPU_DEVICE_ID))
        if ctb is None:
            print("   [WARNING] catboost not installed; cannot instantiate CatBoost.")
            return None
        return ctb.CatBoostRegressor(**params)
    elif model_type == 'LightGBM':
        params = dict(
            random_state=random_state,
            # Tuned hyperparameters from Optuna (2025-10-16 run, R²=0.689)
            n_estimators=1600,
            learning_rate=0.013102532483752814,
            num_leaves=38,
            max_depth=15,
            subsample=0.808191937618305,
            colsample_bytree=0.8161071000274498,
            reg_alpha=0.4844922766155489,
            reg_lambda=9.632501050596495,
            min_child_samples=28
        )
        if cfg.USE_GPU and cfg.GPU_ENABLE_LIGHTGBM:
            params.update(device_type='gpu')
        if lgb is None:
            print("   [WARNING] lightgbm not installed; cannot instantiate LightGBM.")
            return None
        return lgb.LGBMRegressor(**params)
    elif model_type == 'XGBoost':
        params = dict(
            random_state=random_state, 
            enable_categorical=False, 
            verbosity=0,
            # Tuned hyperparameters from Optuna (2025-10-16 run, R²=0.647)
            n_estimators=2900,
            learning_rate=0.031267285253120426,
            max_depth=8,
            subsample=0.6946000930203216,
            colsample_bytree=0.7944888223337778,
            reg_alpha=1.5556540652500264,
            reg_lambda=3.9075969867624596,
            min_child_weight=3,
            gamma=0.0009420866279373377
        )
        if cfg.USE_GPU and cfg.GPU_ENABLE_XGBOOST:
            params.update(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=cfg.GPU_DEVICE_ID)
        if xgb is None:
            print("   [WARNING] xgboost not installed; cannot instantiate XGBoost.")
            return None
        return xgb.XGBRegressor(**params)
    else:
        raise ValueError(f"Unsupported model type for instantiation: {model_type}")


def _get_model_instance_impl(model_type, random_state, cat_features):
    """Internal implementation for model instantiation."""
    # Heuristic: for small datasets skip GPU for tree models (overhead > gain)
    use_gpu_tree = cfg.USE_GPU and (cat_features is not None)
    if model_type == 'CatBoost':
        params = dict(
            random_state=random_state, 
            verbose=0, 
            cat_features=cat_features,
            # Tuned hyperparameters from Optuna (2025-10-16 run, R²=0.699)
            iterations=2700,
            learning_rate=0.057180837980277475,
            depth=10,
            l2_leaf_reg=1.910118305911902,
            min_data_in_leaf=30,
            subsample=0.5770241989636643,
            bootstrap_type='Bernoulli'
        )
        if cfg.USE_GPU and cfg.GPU_ENABLE_CATBOOST:
            params.update(task_type='GPU', devices=str(cfg.GPU_DEVICE_ID))
        if ctb is None:
            print("   [WARNING] catboost not installed; cannot instantiate CatBoost.")
            return None
        return ctb.CatBoostRegressor(**params)
    elif model_type == 'LightGBM':
        params = dict(
            random_state=random_state,
            # Tuned hyperparameters from Optuna (2025-10-16 run, R²=0.689)
            n_estimators=1600,
            learning_rate=0.013102532483752814,
            num_leaves=38,
            max_depth=15,
            subsample=0.808191937618305,
            colsample_bytree=0.8161071000274498,
            reg_alpha=0.4844922766155489,
            reg_lambda=9.632501050596495,
            min_child_samples=28
        )
        if cfg.USE_GPU and cfg.GPU_ENABLE_LIGHTGBM:
            params.update(device_type='gpu')
        if lgb is None:
            print("   [WARNING] lightgbm not installed; cannot instantiate LightGBM.")
            return None
        return lgb.LGBMRegressor(**params)
    elif model_type == 'XGBoost':
        params = dict(
            random_state=random_state, 
            enable_categorical=False, 
            verbosity=0,
            # Tuned hyperparameters from Optuna (2025-10-16 run, R²=0.647)
            n_estimators=2900,
            learning_rate=0.031267285253120426,
            max_depth=8,
            subsample=0.6946000930203216,
            colsample_bytree=0.7944888223337778,
            reg_alpha=1.5556540652500264,
            reg_lambda=3.9075969867624596,
            min_child_weight=3,
            gamma=0.0009420866279373377
        )
        if cfg.USE_GPU and cfg.GPU_ENABLE_XGBOOST:
            params.update(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=cfg.GPU_DEVICE_ID)
        if xgb is None:
            print("   [WARNING] xgboost not installed; cannot instantiate XGBoost.")
            return None
        return xgb.XGBRegressor(**params)
    else:
        raise ValueError(f"Unsupported model type for instantiation: {model_type}")

def _report_gpu_status(model, model_type: str):
    """Best-effort reporting of whether a model is using GPU.
    Returns a string label: 'GPU','CPU','UNKNOWN'."""
    try:
        if model_type == 'CatBoost':
            task_type = model.get_params().get('task_type')
            return 'GPU' if task_type == 'GPU' else 'CPU'
        if model_type == 'LightGBM':
            if hasattr(model, 'booster_') and hasattr(model.booster_, 'params'):
                dev = model.booster_.params.get('device_type')
                return 'GPU' if dev == 'gpu' else 'CPU'
        if model_type == 'XGBoost':
            params = model.get_params()
            if params.get('tree_method') == 'gpu_hist':
                return 'GPU'
            return 'CPU'
    except Exception:
        return 'UNKNOWN'
    return 'UNKNOWN'


# NEW FUNCTION: Leave-One-Group-Out Cross-Validation by Road

# REPLACEMENT: Robust global categorical mapping for all models


# --- CV Fold Model Fitting Function ---
def fit_and_evaluate_cv_fold(X_train, y_train, X_test, y_test, model_type, random_state, fold_num=None):
    """
    Fits and evaluates a model on a single CV fold with pre-split data.
    This function is specifically designed for cross-validation where data splitting
    is handled externally (e.g., by GroupKFold). It ensures consistent preprocessing
    and prevents any data leakage.
    Args:
        X_train (pd.DataFrame): Training features (already split)
        y_train (pd.Series): Training targets (already split)
        X_test (pd.DataFrame): Test features (already split)
        y_test (pd.Series): Test targets (already split)
        model_type (str): Model type to train
        random_state (int): Random state for reproducibility
        fold_num (int, optional): Fold number for logging
    Returns:
        tuple: (trained_model, metrics_dict, y_predictions, fold_dir)
    """
    if fold_num is not None:
        print(f"   Training {model_type} on fold {fold_num}...")
    else:
        print(f"   Training {model_type}...")

    # Build and fit a fold-level preprocessor and transform the fold splits to avoid leakage
    from stage1_config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    # Defensive fallback: if config feature lists are empty or none of the listed features exist
    # in the provided DataFrames, fall back to using all columns except index for numerical features.
    cfg_num = NUMERICAL_FEATURES if NUMERICAL_FEATURES else []
    cfg_cat = CATEGORICAL_FEATURES if CATEGORICAL_FEATURES else []
    # Sanitize configured names to match sanitized feature columns in X_train
    import re as _re
    def _san(name: str) -> str:
        s = str(name).lower()
        s = _re.sub(r'[\s\(\)\-\/]+', '_', s)
        s = _re.sub(r'[^a-z0-9_]+', '', s)
        s = _re.sub(r'[_]+', '_', s)
        return s.strip('_')
    san_cols = {c: _san(c) for c in X_train.columns}
    x_cols_set = set(san_cols.values()) if san_cols else set([_san(c) for c in X_train.columns])
    cfg_num_san = [_san(c) for c in cfg_num]
    cfg_cat_san = [_san(c) for c in cfg_cat]
    # Check presence
    present_num = [c for c in cfg_num_san if c in x_cols_set]
    present_cat = [c for c in cfg_cat_san if c in x_cols_set]
    if not present_num and not present_cat:
        # Derive features from DataFrame columns (exclude obvious metadata if present)
        derived = [c for c in X_train.columns if c not in {'segment_id', 'road_id', 'latitude', 'longitude'}]
        # Heuristic: treat string/object columns as categorical
        derived_num = [c for c in derived if X_train[c].dtype.kind in 'fiubc']
        derived_cat = [c for c in derived if c not in derived_num]
        print("   [WARN] Configured feature lists not found in X_train; deriving features from DataFrame.")
        present_num = derived_num
        present_cat = derived_cat

    X_train_proc, X_test_proc, preprocessor, feature_names = fit_transform_preprocessor(
        X_train, X_test, present_num, present_cat, for_tree_models=True, scale_numeric=False
    )
    # Debugging: if the preprocessor returned no features (empty feature_names or zero columns),
    # fall back to a safer, simpler preprocessing that preserves any numeric/categorical columns
    # discovered above. This prevents LightGBM from receiving an empty dataset which causes
    # the "maximum feature index is -1" fatal error.
    if (not feature_names) or (getattr(X_train_proc, 'shape', (None, 0))[1] == 0):
        print("   [WARN] Preprocessor produced no features. Falling back to robust CV preprocessing.")
        try:
            # Import locally to avoid circular import at module import time
            from stage1_feature_engineering import preprocess_for_cv_fold

            X_train_proc, X_test_proc = preprocess_for_cv_fold(X_train, X_test, present_num, present_cat)
            feature_names = list(X_train_proc.columns)
            print(f"   [INFO] Fallback preprocessing produced features: {feature_names}")
        except Exception as e_fallback:
            print(f"   [ERROR] Fallback preprocessing failed: {e_fallback}")
            # Let downstream code attempt to proceed and produce a clearer error
            pass
    # After this, X_train_proc and X_test_proc are DataFrames with consistent columns and safe encodings

    # --- MODEL INSTANTIATION ---
    # After preprocessing, determine which of the configured categorical features are present
    final_cat_features = [col for col in CATEGORICAL_FEATURES if col in X_train_proc.columns]
    # Use the top-level guarded imports (lgb, ctb, xgb). They will be None if missing.
    if model_type == 'CatBoost':
        if ctb is None:
            raise RuntimeError('CatBoost is not available in this environment')
        model = ctb.CatBoostRegressor(random_state=random_state, verbose=0, cat_features=final_cat_features)
    elif model_type == 'LightGBM':
        if lgb is None:
            raise RuntimeError('LightGBM is not available in this environment')
        model = lgb.LGBMRegressor(random_state=random_state, verbose=0)
    elif model_type == 'XGBoost':
        if xgb is None:
            raise RuntimeError('XGBoost is not available in this environment')
        model = xgb.XGBRegressor(random_state=random_state, enable_categorical=False, verbosity=0)  # Disable categorical since we converted to codes
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # --- TRAINING ---
    # Print debug/info about processed matrices to aid debugging when tests fail
    try:
        print(f"   [DEBUG] X_train_proc shape: {getattr(X_train_proc, 'shape', None)}, columns: {list(X_train_proc.columns) if hasattr(X_train_proc, 'columns') else None}")
        print(f"   [DEBUG] X_test_proc shape: {getattr(X_test_proc, 'shape', None)}")
    except Exception:
        pass

    # Fit model on processed matrices; categorical columns have been encoded numerically
    model.fit(X_train_proc, y_train)

    # --- EVALUATION ---
    y_pred = model.predict(X_test_proc)
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    metrics = {
        "Model Type": model_type,
        "Test R2": r2_score(y_test, y_pred),
        "Test MAE": mean_absolute_error(y_test, y_pred),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Fold": fold_num
    }
    if fold_num is not None:
        print(f"     -> Fold {fold_num} R²: {metrics['Test R2']:.4f}, MAE: {metrics['Test MAE']:.6f}, RMSE: {metrics['Test RMSE']:.6f}")
    # === New: Persist per-fold artifacts so explanations can be deterministic ===
    try:
        artifact_root = Path(getattr(cfg, 'OUTPUT_DIR', Path('.'))) / 'cv_artifacts' / str(model_type)
        fold_dir = artifact_root / f'fold_{fold_num or 0}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save fitted preprocessor (if present)
        try:
            joblib.dump(preprocessor, fold_dir / 'preprocessor.joblib')
        except Exception as e_dump_pre:
            print(f"   [WARN] Could not joblib.dump preprocessor: {e_dump_pre}")

        # Save trained model (use joblib when possible, fallback to model-specific save)
        try:
            joblib.dump(model, fold_dir / 'model.joblib')
        except Exception as e_model_dump:
            # Try model-specific save methods
            try:
                if hasattr(model, 'save_model'):
                    model.save_model(str(fold_dir / 'model.cbm'))
                elif hasattr(model, 'save'):
                    model.save(str(fold_dir / 'model'))
                else:
                    print(f"   [WARN] Unable to persist model for fold {fold_num}: {e_model_dump}")
            except Exception as e_spec:
                print(f"   [WARN] Model-specific save also failed: {e_spec}")

        # Save processed train/test splits, raw test index and predictions for audit
        try:
            X_train_proc.to_csv(fold_dir / 'X_train_proc.csv', index=True)
            X_test_proc.to_csv(fold_dir / 'X_test_proc.csv', index=True)
            pd.Series(y_test.values, index=y_test.index, name='y_test').to_csv(fold_dir / 'y_test.csv')
            pd.Series(y_pred, index=X_test_proc.index, name='y_pred').to_csv(fold_dir / 'y_pred.csv')
        except Exception as e_files:
            print(f"   [WARN] Could not save fold CSV artifacts: {e_files}")

        # Append to a master OOF mapping file: segment_id, fold, model_type, y_pred
        try:
            master_file = artifact_root / 'oof_predictions_master.csv'
            # CRITICAL FIX: Use X_test_proc.index (same as saved files) instead of X_test.index
            seg_ids = list(X_test_proc.index)
            # Include artifact_dir (absolute) so downstream aggregation can locate fold artifacts
            artifact_dir_str = None
            try:
                artifact_dir_str = str(Path(fold_dir).resolve()) if fold_dir is not None else None
            except Exception:
                artifact_dir_str = str(fold_dir) if fold_dir is not None else None

            # Add canonical index column if configured so downstream code can reliably map artifacts
            try:
                from stage1_config import CANONICAL_INDEX_COL
                canonical_col_name = CANONICAL_INDEX_COL
            except Exception:
                canonical_col_name = 'global_index'

            master_rows = pd.DataFrame({
                'segment_id': seg_ids,
                canonical_col_name: seg_ids,
                'fold': [fold_num] * len(seg_ids),
                'model_type': [model_type] * len(seg_ids),
                'y_pred': list(y_pred),
                'artifact_dir': [artifact_dir_str] * len(seg_ids)
            })
            if master_file.exists():
                master_rows.to_csv(master_file, mode='a', header=False, index=False)
            else:
                master_rows.to_csv(master_file, mode='w', header=True, index=False)
        except Exception as e_master:
            print(f"   [WARN] Could not update master OOF mapping: {e_master}")
    except Exception as e_art:
        print(f"   [WARN] Exception while saving CV artifacts: {e_art}")

    # Return trained model, metrics, predictions and the artifact directory path (Path or None)
    try:
        _ = fold_dir  # may or may not exist
    except Exception:
        fold_dir = None
    return model, metrics, y_pred, fold_dir

# --- NEW, SCALABLE VALIDATION FUNCTION ---
def run_stratified_road_kfold_cv(X, y, road_column, model_type, random_state, n_splits=5):
    """
    Performs Stratified Group K-Fold Cross-Validation.
    This is the robust, scalable method for large datasets with many groups (roads).
    It ensures each fold has a representative distribution of road-level risk while
    maintaining the critical property that road segments from the same road never
    appear in both training and test sets.
    Args:
        X (pd.DataFrame): Feature DataFrame, MUST contain the road_column.
        y (pd.Series): Target Series.
        road_column (str): The name of the column identifying the road.
        model_type (str): The name of the model to train.
        random_state (int): The global random state.
        n_splits (int): The number of folds (K).
    Returns:
        dict: A dictionary of the mean and std of performance metrics.
    """
    print(f"--- Running Stratified {n_splits}-Fold CV for model: {model_type} ---")

    strat_mode = getattr(cfg, 'CV_STRATIFICATION_MODE', 'target_mean')
    strat_bins = getattr(cfg, 'CV_STRATIFICATION_BINS', n_splits)
    stratify_series = build_stratification_key(
        metadata_df=X,
        y_transformed=y,
        mode=strat_mode,
        road_column=road_column,
        n_bins=strat_bins,
        feature_df=X
    )
    use_stratified = stratify_series is not None and stratify_series.nunique(dropna=True) > 1
    if use_stratified:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        aligned_stratify_key = stratify_series.astype(int)
        print(f"   Stratification mode '{strat_mode}' active with {aligned_stratify_key.nunique()} bins.")
    else:
        splitter = GroupKFold(n_splits=n_splits)
        aligned_stratify_key = np.zeros(len(X))
        if strat_mode != 'none':
            print(f"   Stratification mode '{strat_mode}' unavailable; using GroupKFold without stratification.")

    # --- 2. Setup the CV Splitter ---
    from stage1_config import METADATA_COLS, FEATURE_EXCLUSIONS
    groups = X[road_column]
    cols_to_drop = set(METADATA_COLS) | set(FEATURE_EXCLUSIONS) | {road_column}
    X_features = X.drop(columns=[col for col in cols_to_drop if col in X.columns], errors='ignore')

    test_r2_scores, test_mae_scores, test_rmse_scores = [], [], []

    for i, (train_idx, test_idx) in enumerate(splitter.split(X_features, aligned_stratify_key, groups)):
        print(f"\n   Fold {i+1}/{n_splits}...")
        X_train_fold, X_test_fold = X_features.iloc[train_idx], X_features.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        test_roads = groups.iloc[test_idx].unique()
        print(f"     Test roads: {len(test_roads)} roads, {len(test_idx)} segments")

        # Use the new CV fold function for fitting and evaluation
        model, fold_metrics, y_pred, fold_dir = fit_and_evaluate_cv_fold(
            X_train_fold, y_train_fold, X_test_fold, y_test_fold,
            model_type, random_state, fold_num=i+1
        )
        test_r2_scores.append(fold_metrics['Test R2'])
        test_mae_scores.append(fold_metrics['Test MAE'])
        test_rmse_scores.append(fold_metrics['Test RMSE'])

    metrics = {
        "Model Type": model_type,
        "Split Strategy": f"Stratified Group {n_splits}-Fold CV",
        "Test R2 Mean": np.mean(test_r2_scores),
        "Test R2 Std": np.std(test_r2_scores),
        "Test MAE Mean": np.mean(test_mae_scores),
        "Test MAE Std": np.std(test_mae_scores),
        "Test RMSE Mean": np.mean(test_rmse_scores),
        "Test RMSE Std": np.std(test_rmse_scores),
        "Test R2": np.mean(test_r2_scores),
        "Test MAE": np.mean(test_mae_scores),
        "Test RMSE": np.mean(test_rmse_scores),
        "Folds": n_splits
    }
    print(f"\n   Cross-validation complete. Mean R²: {metrics['Test R2 Mean']:.4f} ± {metrics['Test R2 Std']:.4f}")
    print(f"   Mean MAE: {metrics['Test MAE Mean']:.6f} ± {metrics['Test MAE Std']:.6f}")
    print(f"   Mean RMSE: {metrics['Test RMSE Mean']:.6f} ± {metrics['Test RMSE Std']:.6f}")
    # Ensure consistent 4-tuple return: (model, metrics_dict, y_pred, fold_dir)
    # fold_dir is optional; return None if not available to keep callers stable.
    fold_dir = locals().get('fold_dir', None)
    try:
        # y_pred may be in local scope as y_pred or y_pred (fold variable); ensure defined
        y_pred_out = locals().get('y_pred', locals().get('y_pred_fold', None))
    except Exception:
        y_pred_out = None
    return model, metrics, y_pred_out, fold_dir