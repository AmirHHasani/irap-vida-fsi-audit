# stage1_main.py
"""
Main orchestration script for the Stage 1 Interpretable Road Risk Modeling pipeline.

This script executes the end-to-end workflow for the research paper. It is designed
to be run from the command line and will produce all necessary artifacts, including
model metrics, SHAP plots, and a final summary report.

The pipeline follows these steps:
1.  Loads configuration from `stage1_config.py`.
2.  Loads and prepares the road segment data.
3.  Trains and evaluates the specified model using one of two strategies:
    a) General Performance: A standard random split for unbiased metrics.
    b) High-Risk Diagnostic: A deterministic split focusing on critical segments.
4.  Performs SHAP analysis to interpret the model's predictions globally.
5.  Conducts domain-specific analyses:
    - Supplementary statistical hypothesis testing on training data.
    - Local SHAP analysis to explain individual high-risk segments.
    - Validation of risk drivers against engineering countermeasures.
6.  Generates a final, consolidated markdown report with all findings.
"""
# ADDED: Workaround for OpenMP runtime conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import time
import sys
import json
import pandas as pd
import numpy as np
import torch
import traceback
import platform



pd.set_option('future.no_silent_downcasting', True)

from datetime import datetime

# Import the configuration file as a module
import stage1_config as cfg

# ADDED: Import the config variable for the countermeasure data path
from stage1_config import COUNTERMEASURE_DATA_CSV
# Import helper to choose preferred id column (prefers canonical index when present)
from stage1_utils import preferred_id_col, TargetTransformer, build_stratification_key

# Import missing imports that cause failures
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly
    print("[INFO] Plotly successfully imported for map generation")
except ImportError as e:
    print(f"[WARN] Plotly import failed: {e}")
    print("[INFO] Maps will be skipped. Install with: pip install plotly kaleido")
    px = None
    go = None
    plotly = None

# User guidance for map output
def print_map_guidance(map_dir):
    print("\n[INFO] Interactive HTML maps have been generated.")
    print(f"      Open the following files in your browser to view the results:")
    print(f"      - {map_dir / 'comprehensive_risk_map.html'}")
    print(f"      - {map_dir / f'top_{cfg.TOP_N_HOTSPOTS}_hotspots_map.html'}")
    print("      (You can share these HTML files directly with stakeholders.)\n")

# ADDED: Master results logging schema for cross-validated mapping
MASTER_RESULTS_COLUMNS = [
    'segment_id', 'road_id', 'latitude', 'longitude',
    'actual_risk', 'predicted_risk', 'fold_number',
    'prediction_confidence', 'model_type'
]

# ADDED: Directory structure setup for cross-validation outputs
from pathlib import Path
def setup_cv_output_structure(run_output_dir):
    """Create organized directory structure for CV results"""
    cv_dirs = {
        'maps': Path(run_output_dir) / 'maps',
        'fold_results': Path(run_output_dir) / 'fold_results',
    'road_explanations': Path(run_output_dir) / 'road_explanations',
    'segment_explanations': Path(run_output_dir) / 'segment_explanations',
    # NEW: hotspot-focused SHAP outputs (Step 3)
    'hotspot_shap': Path(run_output_dir) / cfg.HOTSPOT_SHAP_DIR_NAME
    }
    for dir_path in cv_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return cv_dirs

def transform_risk_to_size(risk_values, min_size=7, max_size=18):
    """Transform risk values to valid marker sizes, handling negative values from log transformation"""
    import numpy as np
    risk_values = np.array(risk_values)
    
    # Handle negative values from log transformation
    risk_min, risk_max = risk_values.min(), risk_values.max()
    if risk_min < 0:
        # Shift to positive range
        risk_shifted = risk_values - risk_min + 0.001
    else:
        risk_shifted = risk_values.copy()
    
    # Avoid division by zero
    if risk_shifted.max() == risk_shifted.min():
        return np.full_like(risk_shifted, (min_size + max_size) / 2)
    
    # Min-max scaling to size range
    risk_normalized = (risk_shifted - risk_shifted.min()) / (risk_shifted.max() - risk_shifted.min())
    sizes = min_size + risk_normalized * (max_size - min_size)
    
    # Validation - ensure all sizes are positive and within range
    sizes = np.clip(sizes, min_size, max_size)
    return sizes


# Import pipeline modules lazily to avoid heavy import-time dependencies when used as a library
# Import each module separately so a missing optional dependency does not disable the whole pipeline.
def _try_import(module_name, attrs):
    """Attempt to import given attributes from module_name. Returns dict of names->objects or None for missing."""
    results = {}
    try:
        mod = __import__(module_name, fromlist=['*'])
        for a in attrs:
            results[a] = getattr(mod, a)
        return results
    except Exception as _e:
        print(f"[WARN] Could not import {module_name}: {_e}")
        for a in attrs:
            results[a] = None
        return results

# Core data loader and feature prep (required)
_r = _try_import('stage1_data_loader', ['load_data'])
load_data = _r.get('load_data')
_r = _try_import('stage1_feature_engineering', ['prepare_features', 'fit_transform_preprocessor'])
prepare_features = _r.get('prepare_features')
fit_transform_preprocessor = _r.get('fit_transform_preprocessor')

# Model training utilities (may depend on optional libs)
_r = _try_import('stage1_model_training', ['train_and_evaluate_model', 'run_stratified_road_kfold_cv', 'fit_and_evaluate_cv_fold', 'get_model_instance', '_report_gpu_status'])
train_and_evaluate_model = _r.get('train_and_evaluate_model')
run_stratified_road_kfold_cv = _r.get('run_stratified_road_kfold_cv')
fit_and_evaluate_cv_fold = _r.get('fit_and_evaluate_cv_fold')
get_model_instance = _r.get('get_model_instance')
_report_gpu_status = _r.get('_report_gpu_status')

# Interpretability & downstream analysis
_r = _try_import('stage1_interpretability', ['run_shap_analysis'])
run_shap_analysis = _r.get('run_shap_analysis')
_r = _try_import('stage1_hypothesis_testing', ['perform_hypothesis_testing'])
perform_hypothesis_testing = _r.get('perform_hypothesis_testing')
_r = _try_import('stage1_high_risk_analysis', ['analyze_high_risk_segments'])
analyze_high_risk_segments = _r.get('analyze_high_risk_segments')
_r = _try_import('stage1_countermeasure_comparison', ['compare_with_countermeasures'])
compare_with_countermeasures = _r.get('compare_with_countermeasures')
_r = _try_import('stage1_reporting', ['generate_final_report'])
generate_final_report = _r.get('generate_final_report')

# Visualizations and explanations (optional)
_r = _try_import('stage1_visualizations', ['plot_model_comparison', 'plot_residual_analysis', 'generate_summary_maps', 'generate_road_comparison_map'])
plot_model_comparison = _r.get('plot_model_comparison')
plot_residual_analysis = _r.get('plot_residual_analysis')
generate_summary_maps = _r.get('generate_summary_maps')
generate_road_comparison_map = _r.get('generate_road_comparison_map')
_r = _try_import('stage1_individual_explanations', ['generate_all_individual_explanations'])
generate_all_individual_explanations = _r.get('generate_all_individual_explanations')


def main():
    """
    Executes the main data analysis pipeline from start to finish.
    """
    start_time = time.time()
    # --- NEW: Create a versioned output directory for this run ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir_name = f"{run_timestamp}_{cfg.SPLIT_STRATEGY}"
    run_output_dir = cfg.OUTPUT_DIR / output_dir_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Early configuration invariants / scientific safeguards
    try:
        if hasattr(cfg, 'HOTSPOT_K') and hasattr(cfg, 'HOTSPOT_K_LIST'):
            if cfg.HOTSPOT_K not in cfg.HOTSPOT_K_LIST:
                print(f"[WARN] HOTSPOT_K={cfg.HOTSPOT_K} not in HOTSPOT_K_LIST={cfg.HOTSPOT_K_LIST}; adding automatically for consistency.")
                cfg.HOTSPOT_K_LIST.append(cfg.HOTSPOT_K)
    except Exception as e_invariant:
        print(f"[WARN] Config invariant check failed: {e_invariant}")

    print("=====================================================================")
    print("  Starting Stage 1: Interpretable Road Risk Modeling Pipeline")
    print(f"  Strategy: {cfg.SPLIT_STRATEGY} | Outputting to: {run_output_dir}")
    # --- ADDED: Explicit GPU Configuration Logging ---
    if cfg.USE_GPU:
        import torch
        is_cuda_available = torch.cuda.is_available()
        print("---------------------------------------------------------------------")
        print(f"  GPU ACCELERATION IS ENABLED (USE_GPU=True)")
        print(f"  CUDA Available: {is_cuda_available}")
        if is_cuda_available:
            print(f"  PyTorch CUDA Version: {torch.version.cuda}")
            print(f"  Target Device: CUDA:{cfg.GPU_DEVICE_ID} ({torch.cuda.get_device_name(cfg.GPU_DEVICE_ID)})")
            print(f"  Models enabled for GPU: " +
                  f"CatBoost({cfg.GPU_ENABLE_CATBOOST}), " +
                  f"LightGBM({cfg.GPU_ENABLE_LIGHTGBM}), " +
                  f"XGBoost({cfg.GPU_ENABLE_XGBOOST})")
        else:
            print("  [WARNING] USE_GPU is True, but torch.cuda.is_available() is False.")
            print("            All models will fall back to CPU.")
        print("---------------------------------------------------------------------")
    else:
        print("  [INFO] GPU ACCELERATION IS DISABLED (USE_GPU=False)")
    print("=====================================================================")


    try:
        # --- STEP 1: DATA LOADING AND PREPARATION ---
        print("\n--- Step 1: Loading and Preparing Data ---")
        df = load_data(cfg.SEGMENTS_DATA_CSV)
        
        X, y, metadata_df = prepare_features(
            df=df,
            target_col=cfg.TARGET_COL,
            metadata_cols=cfg.METADATA_COLS,
            feature_exclusions=cfg.FEATURE_EXCLUSIONS
        )
        print("Data loading and feature preparation complete.")

        # Optional: create composite ID to avoid collisions across datasets
        try:
            if getattr(cfg, 'ENABLE_COMPOSITE_ID', False):
                ds_col = 'Dataset ID'
                base_id = getattr(cfg, 'ID_COL', 'Location ID')
                if (ds_col in df.columns) and (base_id in df.columns):
                    comp_col = getattr(cfg, 'COMPOSITE_ID_COL', 'composite_segment_id')
                    fmt = getattr(cfg, 'COMPOSITE_ID_FORMAT', '{dataset}:{loc}')
                    # Build composite id on metadata copy and propagate to X where applicable
                    meta_vals = df[[ds_col, base_id]].astype(str).fillna('NA')
                    metadata_df[comp_col] = [fmt.format(dataset=r[ds_col], loc=r[base_id]) for _, r in meta_vals.iterrows()]
                    # Prefer composite for downstream joins if requested
                    try:
                        # Add to X as a non-feature identifier for tracing only
                        X[comp_col] = metadata_df[comp_col]
                        print(f"[INFO] Composite ID column created: {comp_col}")
                    except Exception:
                        pass
                else:
                    print("[INFO] Composite ID disabled or required columns missing.")
        except Exception as e_comp:
            print(f"[WARN] Composite ID generation failed: {e_comp}")

        # Determine preferred ID column for this run (prefer canonical if present)
        try:
            id_col_use = preferred_id_col(metadata_df, prefer_canonical=True)
            print(f"[INFO] Preferred ID column for joins/selection: {id_col_use}")
        except Exception:
            id_col_use = getattr(cfg, 'ID_COL', 'segment_id')

        # --- NORMALIZE ROAD IDENTIFIERS ---
        try:
            road_col = cfg.ROAD_COLUMN_NAME
            if road_col in metadata_df.columns:
                print(f"[INFO] Normalizing road identifiers in column: {road_col}")
                # Coerce to string, strip whitespace, replace trailing .0 (e.g., '5.0' -> '5')
                metadata_df[road_col] = metadata_df[road_col].astype(str).fillna('nan').str.strip()
                metadata_df[road_col] = metadata_df[road_col].str.replace(r'\.0$', '', regex=True)
                # Normalize empty-like strings to a sentinel
                metadata_df.loc[metadata_df[road_col].str.len() == 0, road_col] = 'UNKNOWN_ROAD'
                # Optionally, fix common 'nan' artifacts to actual NaN for clarity
                metadata_df[road_col] = metadata_df[road_col].replace({'nan': None})
                print(f"[INFO] Road identifiers normalized; unique roads: {metadata_df[road_col].nunique()}")
            else:
                print(f"[WARN] Expected road column '{cfg.ROAD_COLUMN_NAME}' not found in metadata_df")
        except Exception as e_norm:
            print(f"[WARN] Road normalization failed: {e_norm}")

        # --- NEW: State-of-the-art transformations for skewed data ---
        print("\n--- Applying Target/AADT Transformations ---")
        target_transformer_cfg = getattr(cfg, 'TARGET_TRANSFORMATION', {})
        target_transformer = TargetTransformer(target_transformer_cfg)
        y = pd.Series(target_transformer.fit_transform(y.values), index=y.index, name=y.name)
        print(f"   Target variable transformed using method='{target_transformer.method}'.")

        def inverse_target(arr_like):
            try:
                return target_transformer.inverse_transform(np.asarray(arr_like, dtype=float))
            except Exception:
                return np.asarray(arr_like, dtype=float)

        # Log-transform the AADT feature(s) to handle wide distribution (robust to sanitized names)
        try:
            import re as _re
            def _san(name: str) -> str:
                s = str(name).lower()
                s = _re.sub(r'[\s\(\)\-\/]+', '_', s)
                s = _re.sub(r'[^a-z0-9_]+', '', s)
                s = _re.sub(r'[_]+', '_', s)
                return s.strip('_')
            aadt_candidates = []
            for col in X.columns:
                col_l = col.lower()
                if ('aadt' in col_l) or (col_l == 'vehicle flow (aadt)') or (_san(col) == 'vehicle_flow_aadt'):
                    aadt_candidates.append(col)
            if aadt_candidates:
                for c in aadt_candidates:
                    try:
                        if not np.issubdtype(X[c].dtype, np.number):
                            X[c] = pd.to_numeric(X[c], errors='coerce')
                        X[c] = np.log1p(X[c])
                    except Exception:
                        continue
                print(f"   AADT feature(s) transformed with np.log1p: {aadt_candidates}")
        except Exception as _e_aadt:
            print(f"   [WARN] AADT log-transform skipped due to: {_e_aadt}")
        # --- End of new transformations ---


        # FAST MODE: allow minimal run via env var or CLI flag
        fast_mode = (os.environ.get('STAGE1_FAST', '').strip() == '1') or ('--fast' in sys.argv)
        if fast_mode:
            print("[FAST] Fast mode enabled: limiting models/folds and skipping heavy steps")
            # Prefer CatBoost for fast mode
            cfg.CANDIDATE_MODELS = ['CatBoost']
            # Reduce number of folds
            try:
                cfg.CV_FOLDS = max(2, min(3, int(getattr(cfg, 'CV_FOLDS', 5))))
            except Exception:
                cfg.CV_FOLDS = 2
            # Skip hotspot SHAP and individual explanations to save time
            cfg.COMPUTE_HOTSPOT_SHAP = False
            cfg.GENERATE_ROAD_EXPLANATIONS = False
            cfg.GENERATE_SEGMENT_EXPLANATIONS = False
            # Avoid PNG export to skip kaleido
            setattr(cfg, 'SAVE_MAPS_AS_IMAGES', False)

        model_results = []
        best_model_for_shap = None
        X_train_for_shap, X_test_for_shap, y_test_for_shap = None, None, None
        oof_shap_summary_df = None  # NEW: placeholder for unbiased OOF SHAP feature importance

        # --- STEP 2: MODEL TRAINING & EVALUATION (STRATEGY-DEPENDENT) ---
        print(f"\n--- Step 2: Training & Evaluation using '{cfg.SPLIT_STRATEGY}' Strategy ---")


        if cfg.SPLIT_STRATEGY in ['RANDOM', 'DIAGNOSTIC']:
            # Filter candidate models to those available in the current environment
            available_models = []
            for mt in cfg.CANDIDATE_MODELS:
                try:
                    inst = get_model_instance(mt, random_state=cfg.RANDOM_STATE)
                    if inst is None:
                        print(f"[WARN] Skipping model {mt}: backend not available in this environment.")
                    else:
                        available_models.append(mt)
                except Exception as e_check:
                    print(f"[WARN] Could not instantiate model {mt}: {e_check}")
            if not available_models:
                raise RuntimeError('No candidate models available in this environment; install at least one of CatBoost/LightGBM/XGBoost')

            for model_type in available_models:
                print(f"\nTraining model: {model_type}")
                model, metrics, X_train, y_train, X_test, y_test = train_and_evaluate_model(
                    X=X, y=y, model_type=model_type,
                    split_strategy=cfg.SPLIT_STRATEGY,
                    target_col=cfg.TARGET_COL,
                    road_col_name=cfg.ROAD_COLUMN_NAME,
                    test_size=cfg.TEST_SIZE,
                    random_state=cfg.RANDOM_STATE
                )
                if model is None or X_test is None or y_test is None:
                    print(f"[ERROR] Model training failed for {model_type}. model: {model}, X_test: {type(X_test)}, y_test: {type(y_test)}")
                    continue
                model_results.append({'name': model_type, 'model': model, 'metrics': metrics,
                                     'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test})
                print(f"   Test R²: {metrics.get('Test R2', 'N/A'):.4f}, MAE: {metrics.get('Test MAE', 'N/A'):.6f}")

            if not model_results:
                raise RuntimeError("No models could be trained successfully. Check earlier error messages for details.")

            best_result = max(model_results, key=lambda x: x['metrics'].get('Test R2', float('-inf')))
            best_model_for_shap = best_result['model']
            # Robust: Reuse the splits from the best model result
            X_train_for_shap = best_result.get('X_train')
            X_test_for_shap = best_result.get('X_test')
            y_test_for_shap = best_result.get('y_test')
            if X_train_for_shap is None or X_test_for_shap is None or y_test_for_shap is None:
                print(f"[ERROR] Best model ({best_result['name']}) has invalid splits. Debug info:")
                print(f"X_train_for_shap: {type(X_train_for_shap)}")
                print(f"X_test_for_shap: {type(X_test_for_shap)}")
                print(f"y_test_for_shap: {type(y_test_for_shap)}")
                raise RuntimeError(
                    f"[ERROR] Best model ({best_result['name']}) failed to return valid data. "
                    f"X_train_for_shap: {type(X_train_for_shap)}, X_test_for_shap: {type(X_test_for_shap)}, y_test_for_shap: {type(y_test_for_shap)}"
                )


        elif cfg.SPLIT_STRATEGY == 'BY_ROAD':
            # ==================================================================
            # ENHANCED IMPLEMENTATION: MULTI-MODEL COMPARISON + PER-ROAD OOF PREDICTIONS & HOTSPOT SELECTION
            # ------------------------------------------------------------------
            # This implementation now supports:
            #  (a) All candidate models for comparison using StratifiedGroupKFold
            #  (b) Collection of OOF predictions per segment (log scale) and linear scale
            #  (c) Per-road Top-K hotspot selection using ONLY test-fold predictions (no leakage)
            #  (d) Computation of overlap@K style metrics (precision@K, recall@K)
            #  (e) Selection of best model based on CV performance for SHAP analysis
            # ==================================================================

            print(f"[INFO] Running BY_ROAD CV for all candidate models: {cfg.CANDIDATE_MODELS}")

            # --- NEW: Use the metadata_df for grouping and logging ---
            if cfg.ROAD_COLUMN_NAME not in metadata_df.columns:
                raise ValueError(f"Road column '{cfg.ROAD_COLUMN_NAME}' not found in metadata.")
            
            cv_dirs = setup_cv_output_structure(run_output_dir)

            # Build stratification key per configuration
            print("[INFO] Building stratification key for BY_ROAD CV...")
            strat_mode = getattr(cfg, 'CV_STRATIFICATION_MODE', 'target_mean')
            strat_bins = getattr(cfg, 'CV_STRATIFICATION_BINS', cfg.CV_FOLDS)
            road_col = metadata_df[cfg.ROAD_COLUMN_NAME].copy()
            if road_col.isna().any():
                print(f"  [WARN] Found {road_col.isna().sum()} NaN values in Road column, filling with 'UNKNOWN_ROAD'")
                road_col = road_col.fillna('UNKNOWN_ROAD')
            road_col = road_col.astype(str)
            stratify_series = build_stratification_key(
                metadata_df=metadata_df,
                y_transformed=y,
                mode=strat_mode,
                road_column=cfg.ROAD_COLUMN_NAME,
                n_bins=strat_bins,
                feature_df=X
            )

            aligned_key = None
            strat_array = None
            use_stratified = stratify_series is not None and stratify_series.nunique(dropna=True) > 1
            if use_stratified:
                aligned_key = stratify_series.astype(int)
                strat_array = aligned_key.values
                print(f"  [INFO] Stratification mode '{strat_mode}' active with {aligned_key.nunique()} bins.")
            else:
                strat_array = np.zeros(len(X))
                if strat_mode != 'none':
                    print(f"  [WARN] Stratification mode '{strat_mode}' unavailable; falling back to pure GroupKFold.")
                else:
                    print('  [INFO] Using pure GroupKFold (no stratification).')

            # Helper: prepare feature matrices (drop metadata & target-excluded columns)
            from stage1_config import METADATA_COLS, FEATURE_EXCLUSIONS
            cols_to_drop_set = set(METADATA_COLS) | set(FEATURE_EXCLUSIONS) | {cfg.ROAD_COLUMN_NAME}

            # Multi-model containers
            all_model_results = {}
            best_model_name = None
            best_model_cv_score = float('-inf')

            # CRITICAL: Save full feature matrix BEFORE any filtering
            # This is needed for per-dataset SHAP analysis which uses full OOF predictions
            X_full_unfiltered = X.copy()
            y_full_unfiltered = y.copy()
            metadata_full_unfiltered = metadata_df.copy()
            print(f"[INFO] Saved full unfiltered data: {len(X_full_unfiltered)} segments")

            # Optional: Downsample roads in fast mode for quicker iteration
            if fast_mode:
                try:
                    roads = metadata_df[cfg.ROAD_COLUMN_NAME].dropna().astype(str).unique().tolist()
                    # keep up to 20 roads for speed
                    keep_roads = set(roads[:20])
                    mask_keep = metadata_df[cfg.ROAD_COLUMN_NAME].astype(str).isin(keep_roads)
                    X = X.loc[mask_keep].reset_index(drop=True)
                    y = y.loc[mask_keep].reset_index(drop=True)
                    metadata_df = metadata_df.loc[mask_keep].reset_index(drop=True)
                    # Recompute stratification key with reduced set
                    stratify_series = build_stratification_key(
                        metadata_df=metadata_df,
                        y_transformed=y,
                        mode=strat_mode,
                        road_column=cfg.ROAD_COLUMN_NAME,
                        n_bins=strat_bins,
                        feature_df=X
                    )
                    use_stratified = stratify_series is not None and stratify_series.nunique(dropna=True) > 1
                    if use_stratified:
                        aligned_key = stratify_series.astype(int)
                        strat_array = aligned_key.values
                    else:
                        aligned_key = None
                        strat_array = np.zeros(len(X))
                    print(f"[FAST] Downsampled to {len(keep_roads)} roads, rows: {len(X)}")
                except Exception as e_fast_ds:
                    print(f"[FAST] Downsampling skipped due to: {e_fast_ds}")

            if use_stratified:
                from sklearn.model_selection import StratifiedGroupKFold
                splitter = StratifiedGroupKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
            else:
                from sklearn.model_selection import GroupKFold
                splitter = GroupKFold(n_splits=cfg.CV_FOLDS)

            # Determine available models in this environment
            available_models = []
            for mt in cfg.CANDIDATE_MODELS:
                try:
                    inst = get_model_instance(mt, random_state=cfg.RANDOM_STATE)
                    if inst is not None:
                        available_models.append(mt)
                    else:
                        print(f"[WARN] Skipping model {mt}: backend not available in this environment.")
                except Exception as e_chk:
                    print(f"[WARN] Could not instantiate model {mt}: {e_chk}")
            if not available_models:
                raise RuntimeError('No candidate models available for BY_ROAD CV; install one of CatBoost/LightGBM/XGBoost.')

            # In fast mode, pick a single model for speed
            if fast_mode and len(available_models) > 1:
                available_models = [available_models[0]]

            # Run CV for each available candidate model
            for model_type in available_models:
                print(f"\n=== Running {model_type} Cross-Validation ===")
                
                # Containers for this model's OOF results (reset for each model)
                master_pred_rows = []
                per_road_metric_rows = []
                fold_r2_scores = []
                fold_mae_log = []
                fold_mae_linear = []
                fold_rmse_log = []
                hierarchical_metrics_per_fold = []  # Initialize fresh for each model

                fold_id = 0
                split_labels = aligned_key if use_stratified else strat_array
                for train_idx, test_idx in splitter.split(X, split_labels, groups=road_col):
                    fold_id += 1
                    print(f"[CV] {model_type} Fold {fold_id}/{cfg.CV_FOLDS}")
                    X_train_fold = X.iloc[train_idx]
                    X_test_fold = X.iloc[test_idx]
                    y_train_fold = y.iloc[train_idx]
                    y_test_fold = y.iloc[test_idx]

                    # Drop metadata for modeling
                    X_train_features = X_train_fold.drop(columns=[c for c in cols_to_drop_set if c in X_train_fold.columns], errors='ignore')
                    X_test_features = X_test_fold.drop(columns=[c for c in cols_to_drop_set if c in X_test_fold.columns], errors='ignore')

                    # Use the shared fold helper to train and persist per-fold artifacts
                    try:
                        model_instance, metrics_fold, y_pred_fold, fold_dir = fit_and_evaluate_cv_fold(
                            X_train_features, y_train_fold, X_test_features, y_test_fold,
                            model_type, cfg.RANDOM_STATE, fold_num=fold_id
                        )
                    except Exception as e_fold_call:
                        print(f"[ERROR] fit_and_evaluate_cv_fold failed for fold {fold_id}, model {model_type}: {e_fold_call}")
                        # Fallback: attempt inline training for robustness if model backend is present
                        model_instance = get_model_instance(model_type, random_state=cfg.RANDOM_STATE)
                        if model_instance is None:
                            print(f"[WARN] Skipping fold {fold_id} for {model_type}: backend unavailable for fallback training.")
                            # produce dummy predictions to keep loop consistent
                            y_pred_fold = np.full_like(y_test_fold, fill_value=float(np.mean(y_train_fold)), dtype=float)
                            fold_dir = None
                        else:
                            model_instance.fit(X_train_features, y_train_fold)
                            y_pred_fold = model_instance.predict(X_test_features)
                            fold_dir = None

                    # Metrics (log scale)
                    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                    r2_val = r2_score(y_test_fold, y_pred_fold)
                    mae_log_val = mean_absolute_error(y_test_fold, y_pred_fold)
                    rmse_log_val = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
                    # Metrics (linear/original scale)
                    y_test_lin = inverse_target(y_test_fold)
                    y_pred_lin = inverse_target(y_pred_fold)
                    mae_lin_val = mean_absolute_error(y_test_lin, y_pred_lin)
                    fold_r2_scores.append(r2_val)
                    fold_mae_log.append(mae_log_val)
                    fold_mae_linear.append(mae_lin_val)
                    fold_rmse_log.append(rmse_log_val)
                    print(f"   Fold R2={r2_val:.4f} | MAE(log)={mae_log_val:.5f} | RMSE(log)={rmse_log_val:.5f} | MAE(orig)={mae_lin_val:.2f}")

                    # Store segment-level OOF predictions for this model
                    for i_local, global_idx in enumerate(test_idx):
                        segment_meta = metadata_df.iloc[global_idx]
                        # Normalize road_canon: string, strip, remove trailing .0
                        raw_road = str(segment_meta[cfg.ROAD_COLUMN_NAME]).strip()
                        road_canon = raw_road[:-2] if raw_road.endswith('.0') else raw_road
                        # Resolve a safe segment identifier using preferred id column when present
                        seg_id_val = None
                        try:
                            if id_col_use in segment_meta.index:
                                seg_id_val = segment_meta[id_col_use]
                            else:
                                seg_id_val = segment_meta.get(cfg.ID_COL, segment_meta.get('segment_id'))
                        except Exception:
                            seg_id_val = segment_meta.get(cfg.ID_COL, segment_meta.get('segment_id'))

                        master_pred_rows.append({
                            'segment_id': seg_id_val,
                            'road_id': segment_meta[cfg.ROAD_COLUMN_NAME],
                            'road_canon': road_canon,
                            'latitude': segment_meta.get('Latitude', None),
                            'longitude': segment_meta.get('Longitude', None),
                            'actual_risk': y_test_fold.iloc[i_local],
                            'predicted_risk': y_pred_fold[i_local],
                            'global_index': int(global_idx),
                            'fold_number': fold_id,
                            'prediction_confidence': -1,
                            'is_hotspot_per_road': False,
                            'hotspot_selection_method': None,
                            'model_type': model_type,
                            'artifact_dir': str(fold_dir) if ('fold_dir' in locals() and fold_dir is not None) else None,
                            'Dataset ID': segment_meta.get(cfg.DATASET_ID_COL, None)  # Add Dataset ID for heterogeneity analysis
                        })

                    # Per-road hotspot selection within this test fold ONLY
                    k = cfg.HOTSPOT_K
                    test_meta_df = metadata_df.iloc[test_idx].copy()
                    test_meta_df['actual_log'] = y_test_fold.values
                    test_meta_df['pred_log'] = y_pred_fold
                    test_meta_df['actual_linear'] = y_test_lin
                    test_meta_df['pred_linear'] = y_pred_lin
                    
                    # DEBUG: Count roads in this fold
                    unique_roads_in_fold = test_meta_df[cfg.ROAD_COLUMN_NAME].nunique()
                    total_segments_in_fold = len(test_meta_df)
                    print(f"  [DEBUG] Fold {fold_id}: {unique_roads_in_fold} unique roads, {total_segments_in_fold} segments")
                    
                    # Initialize hierarchical metrics accumulators for this fold
                    fold_hierarchical_metrics = {
                        'strict': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
                        'relaxed': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
                    }
                    
                    # Collect predicted hotspots across roads (per-road fixed-K: min(k, n_segments))
                    predicted_hotspots_all = set()
                    for road_id, df_road in test_meta_df.groupby(cfg.ROAD_COLUMN_NAME):
                        # Determine selection size for this road (min(k, n_segments))
                        n_seg = len(df_road)
                        k_sel = min(k, n_seg) if n_seg > 0 else 0
                        if k_sel == 0:
                            # nothing to select for this road
                            continue

                        # Get all segment IDs for this road
                        all_segments = set(df_road[id_col_use].astype(str))

                        # STRICT: Exact top-K matching
                        actual_hotspots_strict = set(df_road.nlargest(k_sel, 'actual_linear')[id_col_use].astype(str))
                        predicted_hotspots = set(df_road.nlargest(k_sel, 'pred_linear')[id_col_use].astype(str))
                        
                        # RELAXED: Top-K with tolerance (±2 ranks)
                        tolerance = 2
                        k_relaxed = min(k_sel + tolerance, n_seg)
                        actual_hotspots_relaxed = set(df_road.nlargest(k_relaxed, 'actual_linear')[id_col_use].astype(str))
                        
                        # Accumulate predicted hotspot ids
                        predicted_hotspots_all.update(predicted_hotspots)
                        
                        # Calculate STRICT metrics (current definition)
                        tp_strict = len(predicted_hotspots.intersection(actual_hotspots_strict))
                        fp_strict = len(predicted_hotspots - actual_hotspots_strict)
                        fn_strict = len(actual_hotspots_strict - predicted_hotspots)
                        # TN = 0 for top-K selection (we don't classify non-hotspots)
                        tn_strict = 0
                        
                        fold_hierarchical_metrics['strict']['TP'] += tp_strict
                        fold_hierarchical_metrics['strict']['FP'] += fp_strict
                        fold_hierarchical_metrics['strict']['FN'] += fn_strict
                        fold_hierarchical_metrics['strict']['TN'] += tn_strict
                        
                        # Calculate RELAXED metrics (top-K in top-K+2)
                        tp_relaxed = len(predicted_hotspots.intersection(actual_hotspots_relaxed))
                        fp_relaxed = len(predicted_hotspots - actual_hotspots_relaxed)
                        fn_relaxed = len(actual_hotspots_relaxed - predicted_hotspots)
                        # TN = 0 for top-K selection (we don't classify non-hotspots)
                        tn_relaxed = 0
                        
                        fold_hierarchical_metrics['relaxed']['TP'] += tp_relaxed
                        fold_hierarchical_metrics['relaxed']['FP'] += fp_relaxed
                        fold_hierarchical_metrics['relaxed']['FN'] += fn_relaxed
                        fold_hierarchical_metrics['relaxed']['TN'] += tn_relaxed
                        
                        # Original strict metrics for backward compatibility
                        precision_at_k = tp_strict / k_sel if k_sel > 0 else 0.0
                        recall_at_k = tp_strict / len(actual_hotspots_strict) if actual_hotspots_strict else 0.0

                        # Serialize hotspot lists as JSON strings for robust round-tripping
                        per_road_metric_rows.append({
                            'road_id': road_id,
                            'fold_id': fold_id,
                            'k': k_sel,
                            'precision_at_k': precision_at_k,
                            'recall_at_k': recall_at_k,
                            'num_segments': n_seg,
                            'pred_hotspots': json.dumps(list(predicted_hotspots)),
                            'actual_hotspots': json.dumps(list(actual_hotspots_strict)),
                            'fold': int(fold_id),
                            'model_type': model_type,
                            # Add hierarchical metrics per road
                            'tp_strict': tp_strict,
                            'fp_strict': fp_strict,
                            'fn_strict': fn_strict,
                            'tn_strict': tn_strict,
                            'tp_relaxed': tp_relaxed,
                            'fp_relaxed': fp_relaxed,
                            'fn_relaxed': fn_relaxed,
                            'tn_relaxed': tn_relaxed
                        })
                    
                    # Store fold-level hierarchical metrics
                    hierarchical_metrics_per_fold.append({
                        'fold_id': fold_id,
                        'model_type': model_type,
                        **{f'{metric_type}_{class_type}': count 
                           for metric_type, classes in fold_hierarchical_metrics.items() 
                           for class_type, count in classes.items()}
                    })

                    # Mark selected predicted hotspots in the cumulative master_pred_rows list for this fold
                    if predicted_hotspots_all:
                        for row in master_pred_rows:
                            try:
                                if row.get('fold_number') == fold_id and row.get('segment_id') in predicted_hotspots_all:
                                    row['is_hotspot_per_road'] = True
                                    row['hotspot_selection_method'] = 'PER_ROAD_TOP_K'
                            except Exception:
                                continue

                # Aggregate fold metrics for this model
                mean_r2 = float(np.mean(fold_r2_scores)) if fold_r2_scores else float('nan')
                std_r2 = float(np.std(fold_r2_scores)) if fold_r2_scores else float('nan')
                mean_mae_log = float(np.mean(fold_mae_log)) if fold_mae_log else float('nan')
                std_mae_log = float(np.std(fold_mae_log)) if fold_mae_log else float('nan')
                mean_mae_lin = float(np.mean(fold_mae_linear)) if fold_mae_linear else float('nan')
                std_mae_lin = float(np.std(fold_mae_linear)) if fold_mae_linear else float('nan')
                mean_rmse_log = float(np.mean(fold_rmse_log)) if fold_rmse_log else float('nan')
                std_rmse_log = float(np.std(fold_rmse_log)) if fold_rmse_log else float('nan')
                
                # Aggregate hierarchical metrics across all folds for this model
                hierarchical_summary_df = None
                if hierarchical_metrics_per_fold:
                    hierarchical_df = pd.DataFrame(hierarchical_metrics_per_fold)
                    
                    # DEBUG: Print per-fold breakdown
                    print(f"\n[DEBUG] Hierarchical metrics per fold:")
                    for idx, row in hierarchical_df.iterrows():
                        fold_total_predicted = row['strict_TP'] + row['strict_FP']
                        fold_total_actual = row['strict_TP'] + row['strict_FN']
                        print(f"  Fold {row['fold_id']}: Predicted={fold_total_predicted}, Actual_Strict={fold_total_actual}")
                    
                    # Calculate totals across all folds
                    total_metrics = {}
                    for metric_type in ['strict', 'relaxed']:
                        for class_type in ['TP', 'FP', 'FN', 'TN']:
                            col = f'{metric_type}_{class_type}'
                            total_metrics[col] = hierarchical_df[col].sum()
                    
                    # DEBUG: Print totals
                    total_predicted_strict = total_metrics['strict_TP'] + total_metrics['strict_FP']
                    total_actual_strict = total_metrics['strict_TP'] + total_metrics['strict_FN']
                    total_predicted_relaxed = total_metrics['relaxed_TP'] + total_metrics['relaxed_FP']
                    total_actual_relaxed = total_metrics['relaxed_TP'] + total_metrics['relaxed_FN']
                    print(f"\n[DEBUG] Aggregated totals:")
                    print(f"  Strict - Total Predicted: {total_predicted_strict}, Total Actual: {total_actual_strict}")
                    print(f"  Relaxed - Total Predicted: {total_predicted_relaxed}, Total Actual: {total_actual_relaxed}")
                    
                    # Calculate derived metrics for each definition
                    metrics_summary = []
                    for metric_type in ['strict', 'relaxed']:
                        tp = total_metrics[f'{metric_type}_TP']
                        fp = total_metrics[f'{metric_type}_FP']
                        fn = total_metrics[f'{metric_type}_FN']
                        
                        total_dangerous = tp + fn  # Actual dangerous segments (ground truth hotspots)
                        total_predicted = tp + fp  # Predicted dangerous segments (always = K × num_roads)
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        
                        # Definition names for clarity
                        definition_names = {
                            'strict': 'Exact Top-K Match',
                            'relaxed': 'Top-K in Top-(K+2)'
                        }
                        
                        metrics_summary.append({
                            'Definition': definition_names[metric_type],
                            'TP': int(tp),
                            'FP': int(fp),
                            'FN': int(fn),
                            'Total_Actual_Dangerous': int(total_dangerous),
                            'Total_Predicted': int(total_predicted),
                            'Precision': f'{precision:.3f}',
                            'Recall': f'{recall:.3f}',
                            'F1-Score': f'{f1:.3f}',
                            'Hit_Rate': f'{precision:.1%}'
                        })
                    
                    # Save hierarchical metrics summary
                    hierarchical_summary_df = pd.DataFrame(metrics_summary)
                    hierarchical_summary_path = run_output_dir / f'hierarchical_hotspot_metrics_{model_type}.csv'
                    hierarchical_summary_df.to_csv(hierarchical_summary_path, index=False)
                    
                    # Print summary table
                    print(f"\n{'='*80}")
                    print(f"HIERARCHICAL HOTSPOT METRICS - {model_type}")
                    print(f"{'='*80}")
                    print(hierarchical_summary_df.to_string(index=False))
                    print(f"{'='*80}\n")
                
                # Store model results
                all_model_results[model_type] = {
                    'name': model_type,
                    'model': None,  # No single fitted model for CV
                    'metrics': {
                        'Test R2': mean_r2,
                        'Test R2 Mean': mean_r2,
                        'Test R2 Std': std_r2,
                        'Test MAE': mean_mae_log,
                        'Test MAE (log)': mean_mae_log,
                        'Test MAE Std (log)': std_mae_log,
                        'Test MAE (linear)': mean_mae_lin,
                        'Test MAE Std (linear)': std_mae_lin,
                        'Test RMSE': mean_rmse_log,
                        'Test RMSE (log)': mean_rmse_log,
                        'Test RMSE Std (log)': std_rmse_log,
                        'CV_Folds': cfg.CV_FOLDS,
                        'Split Strategy': cfg.SPLIT_STRATEGY
                    },
                    'master_pred_df': pd.DataFrame(master_pred_rows),
                    'per_road_metrics_df': pd.DataFrame(per_road_metric_rows),
                    'hierarchical_metrics': hierarchical_summary_df
                }
                
                print(f"\n[RESULT] {model_type} CV: R2={mean_r2:.4f}±{std_r2:.4f} | MAE(log)={mean_mae_log:.5f}±{std_mae_log:.5f} | RMSE(log)={mean_rmse_log:.5f}±{std_rmse_log:.5f} | MAE(orig)={mean_mae_lin:.2f}")
                
                # Track best model
                if mean_r2 > best_model_cv_score:
                    best_model_cv_score = mean_r2
                    best_model_name = model_type

            # Optional: compute conformal intervals on OOF predictions (simple absolute residual calibration)
            try:
                if getattr(cfg, 'ENABLE_CONFORMAL_INTERVALS', True) and 'master_pred_df' in locals() and not master_pred_df.empty:
                    # Compute absolute residuals per fold (log scale), then global quantile
                    tmp = master_pred_df.copy()
                    if {'predicted_risk','actual_risk'}.issubset(tmp.columns):
                        tmp['abs_resid'] = (tmp['actual_risk'] - tmp['predicted_risk']).abs()
                        q = 1.0 - float(getattr(cfg, 'CONFORMAL_ALPHA', 0.1))
                        qhat = float(tmp['abs_resid'].quantile(q)) if not tmp['abs_resid'].isna().all() else None
                        if qhat is not None:
                            master_pred_df['pred_low'] = master_pred_df['predicted_risk'] - qhat
                            master_pred_df['pred_high'] = master_pred_df['predicted_risk'] + qhat
                            print(f"[INFO] Conformal intervals added (log scale) with q̂={qhat:.4f} (alpha={getattr(cfg,'CONFORMAL_ALPHA',0.1)})")
            except Exception as e_conf:
                print(f"[WARN] Conformal interval computation skipped: {e_conf}")

            # Select best model and use its results for downstream analysis
            if best_model_name:
                print(f"\n[BEST MODEL] {best_model_name} (R2={best_model_cv_score:.4f}) selected for downstream analysis")
                best_model_data = all_model_results[best_model_name]

                # Use best model's data for saving and downstream analysis
                master_pred_df = best_model_data['master_pred_df']
                per_road_metrics_df = best_model_data['per_road_metrics_df']

                # Export best model's segment-level OOF predictions
                master_pred_path = cv_dirs['fold_results'] / 'oof_predictions_segments.csv'
                # Ensure canonical index column exists before saving
                try:
                    from stage1_utils import ensure_canonical_index, resolve_artifact_dir
                    master_pred_df = ensure_canonical_index(master_pred_df)
                except Exception as e_canon:
                    print(f"[WARN] Could not enforce canonical index on master_pred_df: {e_canon}")

                # Quick validation: canonical index must be unique after ensure_canonical_index
                try:
                    if cfg.CANONICAL_INDEX_COL in master_pred_df.columns:
                        if not master_pred_df[cfg.CANONICAL_INDEX_COL].is_unique:
                            print(f"[WARN] {cfg.CANONICAL_INDEX_COL} is not unique in master_pred_df; duplicates will be trouble for artifact mapping.")
                except Exception:
                    pass

                master_pred_df.to_csv(master_pred_path, index=False)
                print(f"[INFO] Saved best model ({best_model_name}) OOF segment predictions: {master_pred_path}")

                # Also save fold artifact index to allow faithful OOF explanations later
                try:
                    if 'master_pred_df' in locals() and not master_pred_df.empty:
                        # Prefer explicit artifact_dir if present in master_pred_df
                        cols = [cfg.CANONICAL_INDEX_COL, 'segment_id', 'fold_number', 'model_type']
                        if 'artifact_dir' in master_pred_df.columns:
                            cols.append('artifact_dir')
                        # Include Dataset ID for heterogeneity analysis
                        if 'Dataset ID' in master_pred_df.columns:
                            cols.append('Dataset ID')
                        fold_index_df = master_pred_df[cols].copy()
                        # Normalize artifact_dir to absolute paths when present
                        if 'artifact_dir' in fold_index_df.columns:
                            fold_index_df['artifact_dir'] = fold_index_df['artifact_dir'].apply(lambda v: str(resolve_artifact_dir(v)) if pd.notna(v) and v is not None else None)
                        fold_index_path = cv_dirs['fold_results'] / 'fold_artifact_index.csv'
                        fold_index_df.to_csv(fold_index_path, index=False)
                        print(f"[INFO] Saved fold artifact index for OOF explanations: {fold_index_path}")
                except Exception as e_findex:
                    print(f"[WARN] Could not save fold_artifact_index.csv: {e_findex}")

                # Export best model's per-road hotspot metrics
                # Defensive: ensure hotspot list columns are JSON-serialized strings so downstream
                # code can parse them reliably with json.loads (avoid legacy eval formats).
                try:
                    import json as _json
                    for col in ['pred_hotspots', 'actual_hotspots']:
                        if col in per_road_metrics_df.columns:
                            # Convert lists or other iterables to JSON strings; leave existing JSON strings intact
                            def _to_json(val):
                                if pd.isna(val):
                                    return None
                                if isinstance(val, str):
                                    # assume already JSON or legacy string; keep as-is to avoid double-encode
                                    return val
                                try:
                                    return _json.dumps(list(val))
                                except Exception:
                                    try:
                                        return _json.dumps(val)
                                    except Exception:
                                        return str(val)
                            per_road_metrics_df[col] = per_road_metrics_df[col].apply(_to_json)
                except Exception as e_js:
                    print(f"[WARN] Could not JSON-serialize hotspot list columns: {e_js}")

                per_road_metrics_path = cv_dirs['fold_results'] / cfg.PER_ROAD_METRICS_CSV
                per_road_metrics_df.to_csv(per_road_metrics_path, index=False)
                print(f"[INFO] Saved best model ({best_model_name}) per-road hotspot metrics: {per_road_metrics_path}")

                # Add all models to model_results for reporting
                for model_name, model_data in all_model_results.items():
                    model_results.append(model_data)

                # Set primary model type to best performing model for SHAP
                cfg.PRIMARY_MODEL_TYPE = best_model_name
                primary_model_type = best_model_name

            else:
                raise RuntimeError("No valid models found during BY_ROAD cross-validation.")

            # --------------------------------------------------------------
            # STEP 2 EXTENDED RANKING METRICS
            # Compute multi-K ranking metrics (precision/recall/overlap, RR, nDCG, Spearman) per road.
            # Uses the full OOF segment-level table to avoid any leakage.
            # --------------------------------------------------------------
            try:
                from evaluation_utils import compute_ranking_metrics_for_road, ranking_results_to_long_df, aggregate_ranking_metrics
                k_list = sorted(set(cfg.HOTSPOT_K_LIST))
                print(f"[INFO] Computing extended ranking metrics for K values: {k_list}")
                # Merge needed columns into a DataFrame grouped by road
                # Use the resolved id_col_use (preferred canonical id when present)
                seg_df = master_pred_df.merge(
                    metadata_df[[id_col_use, cfg.ROAD_COLUMN_NAME]],
                    left_on='segment_id', right_on=id_col_use, how='left'
                )
                # Normalize road column name for grouping
                seg_df.rename(columns={cfg.ROAD_COLUMN_NAME: 'Road name'}, inplace=True)
                all_results = []
                for road_id, df_road in seg_df.groupby('Road name'):
                    if df_road.shape[0] < 2:
                        continue  # skip trivial roads
                    road_results = compute_ranking_metrics_for_road(
                        df_road=df_road,
                        k_list=k_list,
                        id_col=id_col_use,
                        pred_col='predicted_risk',  # FIX: Was 'pred_log'
                        actual_col='actual_risk'    # FIX: Was 'actual_log'
                    )
                    all_results.extend(road_results)
                if all_results:
                    long_df = ranking_results_to_long_df(all_results)
                    long_path = cv_dirs['fold_results'] / cfg.RANKING_METRICS_LONG_CSV
                    long_df.to_csv(long_path, index=False)
                    agg_metrics = aggregate_ranking_metrics(long_df)
                    agg_path = cv_dirs['fold_results'] / cfg.RANKING_METRICS_AGG_JSON
                    import json as _json
                    with open(agg_path, 'w') as f:
                        _json.dump(agg_metrics, f, indent=2)
                    print(f"[INFO] Extended ranking metrics saved: {long_path} & {agg_path}")
                else:
                    print('[WARN] No roads eligible for extended ranking metrics (insufficient segments).')
            except Exception as e:
                print(f"[WARN] Extended ranking metrics computation failed: {e}")

            # --------------------------------------------------------------
            # STEP 3: HOTSPOT-FOCUSED OOF SHAP
            # For each fold we:
            #   * Refit model on training fold.
            #   * Identify predicted hotspots per road in test fold (Top-K).
            #   * Collect those segments only and compute SHAP values (tree models only).
            #   * Aggregate mean|SHAP| across all hotspot segments for a global hotspot driver ranking.
            # NOTE: This avoids explaining non-priority segments and remains OOF (each segment explained only when it was in a test fold).
            # --------------------------------------------------------------
            hotspot_shap_importance = []  # list of (feature importances per fold) for aggregation
            # Collector for per-road high-risk profiles (rows: road_id + mean abs SHAP per feature)
            all_road_high_risk_profiles = []
            if cfg.COMPUTE_HOTSPOT_SHAP:
                print('[INFO] Computing hotspot-focused OOF SHAP values...')
                from stage1_config import METADATA_COLS, FEATURE_EXCLUSIONS
                # Removed redundant inner import of get_model_instance to avoid UnboundLocalError.
                from stage1_interpretability import _get_or_create_tree_explainer, safe_xgb_shap
                k_hot = cfg.HOTSPOT_K
                # Rebuild splitter based on configured stratification mode
                strat_mode_hot = getattr(cfg, 'CV_STRATIFICATION_MODE', 'target_mean')
                strat_bins_hot = getattr(cfg, 'CV_STRATIFICATION_BINS', cfg.CV_FOLDS)
                stratify_series2 = build_stratification_key(
                    metadata_df=metadata_df,
                    y_transformed=y,
                    mode=strat_mode_hot,
                    road_column=cfg.ROAD_COLUMN_NAME,
                    n_bins=strat_bins_hot,
                    feature_df=X
                )
                use_stratified_hot = stratify_series2 is not None and stratify_series2.nunique(dropna=True) > 1
                if use_stratified_hot:
                    from sklearn.model_selection import StratifiedGroupKFold
                    splitter2 = StratifiedGroupKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
                    aligned_key2 = stratify_series2.astype(int)
                else:
                    from sklearn.model_selection import GroupKFold
                    splitter2 = GroupKFold(n_splits=cfg.CV_FOLDS)
                    aligned_key2 = np.zeros(len(X))
                fold_counter = 0
                fold_feature_names = None
                per_fold_saved = []
                split_labels2 = aligned_key2 if use_stratified_hot else aligned_key2

                def _encode_lightgbm_objects(train_df: pd.DataFrame, test_df: pd.DataFrame):
                    """Ordinal-encode object/categorical columns for LightGBM."""
                    if primary_model_type != 'LightGBM':
                        return train_df, test_df
                    obj_cols = sorted(set(
                        train_df.select_dtypes(include=['object', 'category']).columns.tolist() +
                        test_df.select_dtypes(include=['object', 'category']).columns.tolist()
                    ))
                    if not obj_cols:
                        return train_df, test_df
                    train_df = train_df.copy()
                    test_df = test_df.copy()
                    for col in obj_cols:
                        combined = pd.concat([train_df[col], test_df[col]], axis=0)
                        cat = pd.Categorical(combined)
                        codes = pd.Series(cat.codes, index=combined.index)
                        train_df[col] = codes.iloc[:len(train_df)].values
                        test_df[col] = codes.iloc[len(train_df):].values
                    return train_df, test_df

                for tr_idx, te_idx in splitter2.split(X, split_labels2, groups=road_col):
                    fold_counter += 1
                    print(f"[HOTSPOT SHAP] Fold {fold_counter}")
                    X_tr = X.iloc[tr_idx]
                    X_te = X.iloc[te_idx]
                    y_tr = y.iloc[tr_idx]
                    y_te = y.iloc[te_idx]
                    cols_to_drop2 = set(METADATA_COLS) | set(FEATURE_EXCLUSIONS) | {cfg.ROAD_COLUMN_NAME}
                    X_tr_feat = X_tr.drop(columns=[c for c in cols_to_drop2 if c in X_tr.columns], errors='ignore')
                    X_te_feat = X_te.drop(columns=[c for c in cols_to_drop2 if c in X_te.columns], errors='ignore')
                    X_tr_feat, X_te_feat = _encode_lightgbm_objects(X_tr_feat, X_te_feat)
                    model_fold = get_model_instance(primary_model_type, random_state=cfg.RANDOM_STATE)
                    
                    # Train model on this fold
                    model_fold.fit(X_tr_feat, y_tr)
                    gpu_stat_fold = _report_gpu_status(model_fold, primary_model_type)
                    print(f"[GPU] {primary_model_type} hotspot fold {fold_counter} status: {gpu_stat_fold}")
                    y_pred_te = model_fold.predict(X_te_feat)
                    X_te_for_shap = X_te_feat

                    # Determine hotspots per road in this test fold
                    # NEW: Use metadata_df for grouping here as well
                    te_meta_df = metadata_df.iloc[te_idx].copy()
                    te_meta_df['pred_log'] = y_pred_te
                    te_meta_df['actual_log'] = y_te.values
                    
                    hotspot_mask_indices = []
                    for rd, sub in te_meta_df.groupby(cfg.ROAD_COLUMN_NAME):
                        sub_sorted = sub.sort_values('pred_log', ascending=False)
                        limit = k_hot if cfg.MAX_HOTSPOT_SEGMENTS_PER_ROAD is None else min(cfg.MAX_HOTSPOT_SEGMENTS_PER_ROAD, k_hot)
                        hotspot_ids = sub_sorted.head(limit)[id_col_use].tolist()

                        # Map ids to positional indices within the test set using preferred id
                        chosen_indices = sub[sub[id_col_use].isin(hotspot_ids)].index
                        hotspot_mask_indices.extend(chosen_indices)

                    if not hotspot_mask_indices:
                        print(f"  [INFO] No hotspot indices in fold {fold_counter}; skipping SHAP.")
                        continue
                    
                    # Convert global indices to positional indices relative to the test set (X_te)
                    pos_map = {idx: pos for pos, idx in enumerate(X_te.index)}
                    hotspot_positions = [pos_map[i] for i in hotspot_mask_indices if i in pos_map]
                    
                    if not hotspot_positions:
                        print(f"  [INFO] No matching hotspot positions found in fold {fold_counter}; skipping SHAP.")
                        continue

                    X_hotspot = X_te_for_shap.iloc[hotspot_positions]
                    
                    if X_hotspot.empty:
                        print(f"  [INFO] Empty hotspot feature frame fold {fold_counter}; skip.")
                        continue
                    # SHAP for hotspot subset (tree models only)
                    try:
                        if primary_model_type == 'XGBoost':
                            shap_vals, _base = safe_xgb_shap(model_fold, X_hotspot)
                        elif primary_model_type in ['CatBoost','LightGBM']:
                            expl = _get_or_create_tree_explainer(model_fold)
                            shap_vals = expl.shap_values(X_hotspot)
                        else:
                            print(f"  [WARN] SHAP not implemented for model {primary_model_type}; skipping fold.")
                            continue
                        # Aggregate importance for this fold (mean abs across all hotspots)
                        mean_abs = np.mean(np.abs(shap_vals), axis=0)
                        feature_names = list(X_hotspot.columns)
                        if fold_feature_names is None:
                            fold_feature_names = feature_names
                        hotspot_shap_importance.append(mean_abs)

                        # --- NEW: compute per-road high-risk profile (mean abs SHAP per road) ---
                        # Map hotspot rows back to global indices (the subset we used)
                        hotspot_global_indices = [i for i in hotspot_mask_indices if i in pos_map]
                        # Build DataFrame aligned to shap_vals rows
                        try:
                            shap_df = pd.DataFrame(np.abs(shap_vals), columns=feature_names, index=hotspot_global_indices)
                        except Exception:
                            # Fallback: use positional index if mapping failed
                            shap_df = pd.DataFrame(np.abs(shap_vals), columns=feature_names)

                        # For each road present in this test fold, compute mean SHAP for its hotspots (if any)
                        te_roads = te_meta_df.set_index(te_meta_df.index)[cfg.ROAD_COLUMN_NAME].to_dict()
                        # Group by road id using the global indices in shap_df.index
                        for gid in np.unique(shap_df.index) if hasattr(shap_df, 'index') else []:
                            pass
                        # iterate roads and compute mean for those indices
                        road_groups = {}
                        for global_idx in shap_df.index:
                            road_id = te_roads.get(global_idx, None)
                            if road_id is None:
                                continue
                            road_groups.setdefault(road_id, []).append(global_idx)

                        for road_id, gids in road_groups.items():
                            try:
                                rows = shap_df.loc[gids]
                                if isinstance(rows, pd.Series):
                                    mean_vec = rows.values
                                else:
                                    mean_vec = rows.mean(axis=0).values
                                # Build output row: road id + features
                                row_dict = {'road_id': road_id, 'fold': int(fold_counter), 'model_type': primary_model_type}
                                for fname, val in zip(feature_names, mean_vec):
                                    row_dict[fname] = float(val)
                                all_road_high_risk_profiles.append(row_dict)
                            except Exception as e_rprof:
                                print(f"  [WARN] Could not compute per-road profile for road {road_id} in fold {fold_counter}: {e_rprof}")
                        if cfg.SAVE_PER_FOLD_HOTSPOT_SHAP:
                            per_fold_df = pd.DataFrame({'feature': feature_names,'mean_abs_shap': mean_abs})
                            per_fold_file = cv_dirs['hotspot_shap'] / f'fold_{fold_counter}_hotspot_shap.csv'
                            per_fold_df.to_csv(per_fold_file, index=False)
                            per_fold_saved.append(str(per_fold_file))
                    except Exception as e:
                        print(f"  [WARN] Hotspot SHAP failed in fold {fold_counter}: {e}")
                # Aggregate across folds
                if hotspot_shap_importance and cfg.SAVE_AGGREGATED_HOTSPOT_SHAP:
                    stacked = np.vstack(hotspot_shap_importance)
                    global_mean = np.mean(stacked, axis=0)
                    global_std = np.std(stacked, axis=0)
                    agg_df = pd.DataFrame({
                        'feature': fold_feature_names,
                        'mean_abs_shap_hotspot': global_mean,
                        'std_abs_shap_hotspot': global_std,
                        'stability_ratio': global_std / (global_mean + 1e-12)
                    }).sort_values('mean_abs_shap_hotspot', ascending=False)
                    agg_path = cv_dirs['hotspot_shap'] / cfg.HOTSPOT_SHAP_FEATURE_IMPORTANCE_CSV
                    agg_df.to_csv(agg_path, index=False)
                    print(f"[INFO] Aggregated hotspot SHAP feature importance saved: {agg_path}")
                else:
                    print('[INFO] No hotspot SHAP values aggregated (possibly empty or all folds skipped).')

                # Optional: Per-road hotspot SHAP consolidation (only if saved per-fold & requested)
                if cfg.SAVE_PER_ROAD_HOTSPOT_SHAP and cfg.SAVE_PER_FOLD_HOTSPOT_SHAP and hotspot_shap_importance:
                    try:
                        per_fold_files = list((cv_dirs['hotspot_shap']).glob('fold_*_hotspot_shap.csv'))
                        if per_fold_files:
                            concat_df = []
                            for fpath in per_fold_files:
                                tmp = pd.read_csv(fpath)
                                tmp['fold_file'] = fpath.name
                                concat_df.append(tmp)
                            all_hotspot_shap = pd.concat(concat_df, ignore_index=True)
                            all_hotspot_shap.to_csv(cv_dirs['hotspot_shap'] / cfg.PER_ROAD_HOTSPOT_SHAP_CSV, index=False)
                            print(f"[INFO] Per-road hotspot SHAP consolidated: {cv_dirs['hotspot_shap'] / cfg.PER_ROAD_HOTSPOT_SHAP_CSV}")
                    except Exception as e_perroad:
                        print(f"[WARN] Could not build per-road hotspot SHAP export: {e_perroad}")

                # After hotspot SHAP computation finishes, save the collected per-road high-risk profiles
                try:
                    if all_road_high_risk_profiles:
                        profiles_df = pd.DataFrame(all_road_high_risk_profiles)
                        # Ensure consistent ordering of columns (road_id first)
                        cols = list(profiles_df.columns)
                        if 'road_id' in cols:
                            cols = ['road_id'] + [c for c in cols if c != 'road_id']
                        profiles_df = profiles_df[cols]
                        profiles_path = cv_dirs['fold_results'] / 'all_road_high_risk_profiles.csv'
                        profiles_df.to_csv(profiles_path, index=False)
                        print(f"[INFO] Saved all road high-risk profiles: {profiles_path}")
                    else:
                        print('[INFO] No per-road high-risk profiles were generated (empty).')
                except Exception as e_save_profiles:
                    print(f"[WARN] Could not save all_road_high_risk_profiles.csv: {e_save_profiles}")

            # --------------------------------------------------------------
            # STEP 4: Countermeasure Integration & Hotspot Overlay Map
            # Build TP/FP/FN classification, join countermeasures, compute coverage, generate map.
            # --------------------------------------------------------------
            if cfg.ENABLE_COUNTERMEASURE_OVERLAY:
                try:
                    print('[INFO] Building hotspot overlay with countermeasures...')
                    from hotspot_countermeasure_utils import (
                        build_hotspot_overlay,
                        integrate_countermeasures,
                        compute_countermeasure_coverage,
                        countermeasure_occurrence_counts
                    )
                    # Load countermeasure data (reuse existing path)
                    try:
                        countermeasure_df = pd.read_csv(COUNTERMEASURE_DATA_CSV)
                    except Exception as e_cm:
                        print(f'[WARN] Could not load countermeasure data: {e_cm}')
                        countermeasure_df = None

                    per_road_raw = pd.read_csv(per_road_metrics_path)
                    # Let the hotspot utility choose the preferred id column (pass None)
                    overlay_df = build_hotspot_overlay(
                        per_road_metrics_df=per_road_raw,
                        oof_segments_df=master_pred_df,
                        id_col='segment_id',  # Ensure we match the same ID type used in per_road_metrics
                        road_col=cfg.ROAD_COLUMN_NAME
                    )
                    if overlay_df.empty:
                        print('[WARN] Overlay dataframe empty; skipping countermeasure integration.')
                    else:
                        # Let integrate_countermeasures pick the preferred id column when possible
                        overlay_with_cm = integrate_countermeasures(
                            overlay_df=overlay_df,
                            countermeasure_df=countermeasure_df,
                            id_col=None
                        )
                        coverage_df = compute_countermeasure_coverage(overlay_with_cm)
                        freq_df = countermeasure_occurrence_counts(overlay_with_cm)
                        # Save artifacts
                        overlay_path = cv_dirs['fold_results'] / cfg.HOTSPOT_OVERLAY_CSV
                        overlay_with_cm.to_csv(overlay_path, index=False)
                        coverage_path = cv_dirs['fold_results'] / cfg.COUNTERMEASURE_COVERAGE_CSV
                        coverage_df.to_csv(coverage_path, index=False)
                        freq_path = cv_dirs['fold_results'] / cfg.COUNTERMEASURE_MIN_OCCURRENCES_LOG
                        freq_df.to_csv(freq_path, index=False)
                        print(f'[INFO] Saved overlay: {overlay_path}')
                        print(f'[INFO] Saved coverage metrics: {coverage_path}')
                        print(f'[INFO] Saved countermeasure frequencies: {freq_path}')

                        # Generate overlay map if location fields present
                        required_geo_cols = {'latitude','longitude'}
                        if required_geo_cols.issubset(set(master_pred_df.columns)):
                            try:
                                # Check if plotly is available for overlay map
                                try:
                                    import plotly.express as px_overlay
                                    plotly_overlay_available = True
                                except ImportError:
                                    plotly_overlay_available = False
                                    
                                if plotly_overlay_available:  # Check if plotly is available
                                    map_df = master_pred_df.merge(overlay_with_cm[['segment_id','class']], on='segment_id', how='left')
                                    # Add missing columns for hover data
                                    if 'pred_log' not in map_df.columns:
                                        map_df['pred_log'] = map_df['predicted_risk']
                                    if 'actual_log' not in map_df.columns:
                                        map_df['actual_log'] = map_df['actual_risk']
                                    
                                    # Assign color by class (TP/FP/FN/None)
                                    map_df['class'] = map_df['class'].fillna('OTHER')
                                    color_discrete_map = {'TP':'green','FP':'orange','FN':'red','OTHER':'lightgray'}
                                    # Marker sizing: keep points small on dense global maps
                                    # Previous sizing (7..22) was visually too large; scale ~1/4.
                                    try:
                                        map_df['marker_size'] = transform_risk_to_size(map_df['pred_log'], min_size=2, max_size=6)
                                    except Exception:
                                        map_df['marker_size'] = 3
                                    fig_map = px_overlay.scatter_mapbox(
                                        map_df,
                                        lat='latitude', lon='longitude',
                                        color='class',
                                        hover_data=['segment_id','road_id','pred_log','actual_log','class'],
                                        color_discrete_map=color_discrete_map,
                                        size='marker_size',
                                        size_max=6,
                                        zoom=7,
                                        title='Predicted vs Actual Hotspots (TP/FP/FN)'
                                    )
                                    fig_map.update_layout(mapbox_style='open-street-map')
                                    # Slightly increase opacity for better visibility
                                    try:
                                        fig_map.update_traces(marker=dict(opacity=0.85))
                                    except Exception:
                                        pass
                                    map_path = cv_dirs['maps'] / cfg.HOTSPOT_OVERLAY_MAP_HTML
                                    fig_map.write_html(str(map_path))
                                    print(f'[INFO] Hotspot overlay map saved: {map_path}')
                                    if getattr(cfg, 'SAVE_MAPS_AS_IMAGES', False):
                                        try:
                                            fig_map.write_image(str(cv_dirs['maps'] / cfg.HOTSPOT_OVERLAY_MAP_PNG), width=1400, height=900, scale=2)
                                            print(f"[INFO] Hotspot overlay PNG saved: {cv_dirs['maps'] / cfg.HOTSPOT_OVERLAY_MAP_PNG}")
                                        except Exception as e_img:
                                            print(f'[WARN] Could not save PNG overlay map. Exception: {e_img}')
                                            print('[WARN] Common causes: Kaleido not installed in this environment, Plotly/Kaleido version mismatch, or Mapbox tile/style access failure (HTTP 525).')
                                            print('[WARN] Fallback: HTML map saved; consider setting PLOTLY_MAPBOX_ACCESS_TOKEN or using open-street-map or headless-browser snapshot for PNGs.')
                                else:
                                    print('[WARN] Plotly not available for map generation')
                            except Exception as e_map:
                                print(f'[WARN] Failed to generate overlay map: {e_map}')
                        else:
                            print('[WARN] Missing latitude/longitude in master predictions for map overlay.')

                        # STEP 8: Enhanced layered map (if enabled)
                        if getattr(cfg, 'GENERATE_OVERLAY_LAYER_MAP', False):
                            try:
                                print('[INFO] Generating layered hotspot overlay map...')
                                import plotly.express as px
                                import plotly.graph_objects as go
                                layered_path = cv_dirs['maps'] / getattr(cfg, 'OVERLAY_LAYER_MAP_HTML', 'hotspot_overlay_layered.html')
                                # Build base dataframe merging coordinates & classification
                                base_df = master_pred_df.merge(
                                    overlay_with_cm[['segment_id','class','countermeasure']].drop_duplicates(),
                                    on='segment_id', how='left'
                                )
                                # Determine categories
                                categories = ['TP','FP','FN']
                                layers = []
                                color_map = {'TP':'green','FP':'orange','FN':'red'}
                                for cat in categories:
                                    sub = base_df[base_df['class'] == cat]
                                    if sub.empty:
                                        continue
                                    # Build hovertext safely, tolerating missing columns
                                    hover_texts = []
                                    for r in sub.itertuples(index=False):
                                        seg_id = getattr(r, 'segment_id', 'NA')
                                        pred_log = getattr(r, 'pred_log', None)
                                        actual_log = getattr(r, 'actual_log', None)
                                        pred_str = f"{pred_log:.3f}" if (pred_log is not None and not pd.isna(pred_log)) else 'NA'
                                        act_str = f"{actual_log:.3f}" if (actual_log is not None and not pd.isna(actual_log)) else 'NA'
                                        hover_texts.append(f"ID: {seg_id}<br>Pred log: {pred_str}<br>Actual log: {act_str}")
                                    layers.append(go.Scattermap(
                                        lat=sub.get('latitude', pd.Series(dtype=float)),
                                        lon=sub.get('longitude', pd.Series(dtype=float)),
                                        mode='markers',
                                        marker={'size':10,'color':color_map.get(cat,'gray'), 'opacity':0.7},
                                        name=f'{cat} segments ({len(sub)})',
                                        hovertext=hover_texts,
                                    ))
                                # Countermeasure layer (all segments with countermeasure regardless of class)
                                cm_segments = base_df[base_df['countermeasure'].notna()]
                                if not cm_segments.empty:
                                    layers.append(go.Scattermap(
                                        lat=cm_segments.get('latitude', pd.Series(dtype=float)),
                                        lon=cm_segments.get('longitude', pd.Series(dtype=float)),
                                        mode='markers',
                                        marker={'size':6,'color':'blue','opacity':0.5},
                                        name=f'Countermeasure present ({cm_segments.shape[0]})',
                                        hovertext=[f"ID: {r.segment_id}<br>CM: {str(r.countermeasure)[:40]}" for r in cm_segments.itertuples()],
                                    ))
                                # Map predicted risk heat as continuous layer (optional)
                                try:
                                    # Normalize predictions for color scaling
                                    risk_df = master_pred_df.copy()
                                    if {'latitude','longitude','pred_log'}.issubset(risk_df.columns):
                                        layers.append(go.Scattermap(
                                            lat=risk_df['latitude'],
                                            lon=risk_df['longitude'],
                                            mode='markers',
                                            marker={'size':4,'color':risk_df['pred_log'], 'colorscale':'Viridis', 'showscale':True},
                                            name='Pred log risk (all)',
                                                hovertext=[
                                                    (f"ID: {getattr(r, 'segment_id', 'NA')}<br>Pred log: {getattr(r, 'pred_log', 'NA'):.3f}" 
                                                     if (getattr(r, 'pred_log', None) is not None and not pd.isna(getattr(r, 'pred_log', None)))
                                                     else f"ID: {getattr(r, 'segment_id', 'NA')}<br>Pred log: NA")
                                                    for r in risk_df.itertuples()
                                                ],
                                            visible='legendonly'
                                        ))
                                except Exception as e_heat:
                                    print(f'[WARN] Could not add continuous risk layer: {e_heat}')

                                fig_layered = go.Figure(data=layers)
                                fig_layered.update_layout(
                                    mapbox={'style':'open-street-map','zoom':7},
                                    margin={'l':0,'r':0,'t':40,'b':0},
                                    title='Layered Hotspot / Countermeasure Map'
                                )
                                fig_layered.write_html(str(layered_path))
                                print(f"[INFO] Layered overlay map saved: {layered_path}")
                            except Exception as e_layer:
                                print(f'[WARN] Enhanced layered map generation failed: {e_layer}')
                except Exception as e_step4:
                    print(f'[WARN] Step 4 countermeasure integration failed: {e_step4}')

            # ========================================================
            # EXPERT FIX: CREATE ROBUST GLOBAL MODEL AND PREPROCESSOR
            # ========================================================
            # Part 1: Create a reliable global model for explanation fallback
            print(f"\n[INFO] Creating global representative {best_model_name} model for robust explanations...")
            
            # Prepare global preprocessor fitted on entire X dataset
            from sklearn.preprocessing import StandardScaler
            global_preprocessor = StandardScaler()
            
            # Remove metadata columns for model compatibility
            cols_to_drop_global = set(cfg.METADATA_COLS) | set(cfg.FEATURE_EXCLUSIONS) | {cfg.ROAD_COLUMN_NAME}
            X_features_global = X.drop(columns=[c for c in cols_to_drop_global if c in X.columns], errors='ignore')
            
            # Fit preprocessor on all data
            X_processed_global = global_preprocessor.fit_transform(X_features_global)
            
            # Create and fit global representative model on all data for robust explanations
            best_model_for_shap = get_model_instance(best_model_name, random_state=cfg.RANDOM_STATE)
            
            # Fit global model on all data for robust explanations
            print(f"[INFO] Fitting global model on {len(X_processed_global)} samples...")
            best_model_for_shap.fit(X_processed_global, y)
            print(f"[SUCCESS] Global {best_model_name} model fitted successfully")
            
            # Store preprocessor and feature names for explanations
            global_feature_names = list(X_features_global.columns)
            
            # Set placeholders for compatibility with RANDOM/DIAGNOSTIC strategy downstream code
            X_train_for_shap = X_features_global  # Use full feature set for training reference
            X_test_for_shap = None  # Will be set per-segment in explanations
            y_test_for_shap = None  # Will be set per-segment in explanations
            oof_shap_summary_df = None
        # --- END BY_ROAD BRANCH ---
        
        # --- STEP 2.1: GENERATE BASIC RISK MAPS (All Strategies) ---
        print("\n--- Step 2.1: Generating Basic Risk Maps ---")
        
        # DEBUG: Check conditions for map generation
        print(f"[DEBUG] SPLIT_STRATEGY: {cfg.SPLIT_STRATEGY}")
        print(f"[DEBUG] master_pred_df in locals(): {'master_pred_df' in locals()}")
        if 'master_pred_df' in locals():
            print(f"[DEBUG] master_pred_df.empty: {master_pred_df.empty}")
            print(f"[DEBUG] master_pred_df.shape: {master_pred_df.shape}")
        
        if cfg.SPLIT_STRATEGY == 'BY_ROAD' and 'master_pred_df' in locals() and not master_pred_df.empty:
            try:
                print(f"[INFO] Generating basic risk maps from {len(master_pred_df)} OOF predictions...")
                cv_dirs = setup_cv_output_structure(run_output_dir) if 'cv_dirs' not in locals() else cv_dirs
                
                # Check if plotly is available (import locally to avoid scope issues)
                try:
                    import plotly.express as px_local
                    import plotly.graph_objects as go_local
                    plotly_available = True
                    print("[INFO] Plotly available for map generation")
                except ImportError:
                    plotly_available = False
                    print("[WARN] Plotly not available - skipping map generation")
                
                if plotly_available:
                    # ADDED: Use the improved generate_summary_maps function
                    if generate_summary_maps is not None:
                        try:
                            print("[INFO] Generating improved summary maps...")
                            generate_summary_maps(master_pred_df, cv_dirs['maps'], top_n_hotspots=getattr(cfg, 'TOP_N_HOTSPOTS', 1000))
                        except Exception as e_summary:
                            print(f"[WARN] Summary maps generation failed: {e_summary}")
                    
                    # Basic scatter map of all predictions
                    map_df = master_pred_df.copy()
                    
                    # Ensure we have coordinate columns
                    if 'latitude' in map_df.columns and 'longitude' in map_df.columns:
                        # FIXED: Transform risk values to valid sizes for markers
                        actual_risk_sizes = transform_risk_to_size(map_df['actual_risk'], min_size=4, max_size=15)
                        
                        # Create color scale based on predicted risk
                        fig_basic = px_local.scatter_map(
                            map_df,
                            lat='latitude', 
                            lon='longitude',
                            color='predicted_risk',
                            size=actual_risk_sizes,  # FIXED: Use transformed sizes
                            hover_data=['segment_id', 'road_id', 'predicted_risk', 'actual_risk', 'fold_number'],
                            color_continuous_scale='Viridis',
                            size_max=15,
                            zoom=7,
                            title='Road Segment Risk Predictions (Out-of-Fold)'
                        )
                        fig_basic.update_layout(mapbox_style='open-street-map')
                        
                        # Save HTML map
                        basic_map_path = cv_dirs['maps'] / 'basic_risk_map.html'
                        fig_basic.write_html(str(basic_map_path))
                        print(f'[INFO] Basic risk map saved: {basic_map_path}')
                        
                        # Save PNG if enabled
                        if getattr(cfg, 'SAVE_MAPS_AS_IMAGES', False):
                            try:
                                png_path = cv_dirs['maps'] / 'basic_risk_map.png'
                                fig_basic.write_image(str(png_path), width=1400, height=900, scale=2)
                                print(f"[INFO] Basic risk map PNG saved: {png_path}")
                            except Exception as e_png:
                                print(f'[WARN] Could not save PNG map. Exception: {e_png}')
                                print('[WARN] If this is a Mapbox HTTP error (525), ensure PLOTLY_MAPBOX_ACCESS_TOKEN is set and network access to Mapbox is available.')
                                print('[WARN] Otherwise check that kaleido is installed into the active Python environment: python -m pip install --upgrade kaleido')
                        
                        # Create a second map showing actual vs predicted
                        sample_df = map_df.sample(n=min(5000, len(map_df)))  # Sample for performance
                        predicted_risk_sizes = transform_risk_to_size(sample_df['predicted_risk'], min_size=4, max_size=15)
                        
                        fig_comparison = px_local.scatter_map(
                            sample_df,
                            lat='latitude', 
                            lon='longitude',
                            color='actual_risk',
                            size=predicted_risk_sizes,  # FIXED: Use transformed sizes
                            hover_data=['segment_id', 'road_id', 'predicted_risk', 'actual_risk'],
                            color_continuous_scale='Reds',
                            size_max=15,
                            zoom=7,
                            title='Actual Risk vs Predicted Risk (Sample)'
                        )
                        fig_comparison.update_layout(mapbox_style='open-street-map')
                        
                        comparison_map_path = cv_dirs['maps'] / 'actual_vs_predicted_map.html'
                        fig_comparison.write_html(str(comparison_map_path))
                        print(f'[INFO] Comparison map saved: {comparison_map_path}')
                        
                        print(f"[SUCCESS] Generated 2 basic risk maps in {cv_dirs['maps']}")
                        
                    else:
                        print('[WARN] Missing latitude/longitude columns for map generation')
                else:
                    print('[WARN] Plotly not available - skipping map generation')
                    
            except Exception as e_maps:
                print(f'[WARN] Failed to generate basic risk maps: {e_maps}')
                traceback.print_exc()
        else:
            print('[INFO] Skipping basic maps - checking reasons:')
            if cfg.SPLIT_STRATEGY != 'BY_ROAD':
                print(f'  - Wrong strategy: {cfg.SPLIT_STRATEGY} (expected BY_ROAD)')
            if 'master_pred_df' not in locals():
                print('  - master_pred_df not available')
            elif master_pred_df.empty:
                print('  - master_pred_df is empty')
            
            # FALLBACK: Try to load from saved files
            try:
                print('[INFO] Attempting fallback map generation from saved files...')
                cv_dirs = setup_cv_output_structure(run_output_dir)
                oof_file = cv_dirs['fold_results'] / 'oof_predictions_segments.csv'
                
                if oof_file.exists():
                    print(f'[INFO] Loading OOF predictions from {oof_file}')
                    fallback_df = pd.read_csv(oof_file)
                    
                    # Check if plotly is available for fallback maps
                    try:
                        import plotly.express as px_fallback
                        plotly_fallback_available = True
                    except ImportError:
                        plotly_fallback_available = False
                    
                    if not fallback_df.empty and plotly_fallback_available:
                        print(f'[INFO] Generating fallback maps from {len(fallback_df)} records...')
                        
                        # Check coordinate columns
                        lat_col = 'latitude' if 'latitude' in fallback_df.columns else ('Latitude' if 'Latitude' in fallback_df.columns else None)
                        lon_col = 'longitude' if 'longitude' in fallback_df.columns else ('Longitude' if 'Longitude' in fallback_df.columns else None)
                        
                        if lat_col and lon_col:
                            # Sample for performance
                            sample_df = fallback_df.sample(n=min(3000, len(fallback_df)), random_state=42)
                            
                            # FIXED: Transform risk values to valid sizes for markers
                            fallback_actual_sizes = transform_risk_to_size(sample_df['actual_risk'], min_size=4, max_size=15)
                            
                            # Create basic map
                            fig_fallback = px_fallback.scatter_map(
                                sample_df,
                                lat=lat_col,
                                lon=lon_col,
                                color='predicted_risk',
                                size=fallback_actual_sizes,  # FIXED: Use transformed sizes
                                hover_data=['segment_id', 'road_id', 'predicted_risk', 'actual_risk'],
                                color_continuous_scale='Viridis',
                                size_max=15,
                                zoom=7,
                                title=f'Road Risk Predictions - Fallback Generation ({len(sample_df)} points)'
                            )
                            fig_fallback.update_layout(mapbox_style='open-street-map')
                            
                            fallback_map_path = cv_dirs['maps'] / 'fallback_risk_map.html'
                            fig_fallback.write_html(str(fallback_map_path))
                            print(f'[SUCCESS] Fallback map generated: {fallback_map_path}')
                            
                            # Create hotspots map
                            top_hotspots = sample_df.nlargest(100, 'predicted_risk')
                            hotspot_actual_sizes = transform_risk_to_size(top_hotspots['actual_risk'], min_size=6, max_size=20)
                            
                            fig_hotspots_fallback = px_fallback.scatter_map(
                                top_hotspots,
                                lat=lat_col,
                                lon=lon_col,
                                color='predicted_risk',
                                size=hotspot_actual_sizes,  # FIXED: Use transformed sizes
                                hover_data=['segment_id', 'road_id', 'predicted_risk', 'actual_risk'],
                                color_continuous_scale='Reds',
                                size_max=20,
                                zoom=8,
                                title=f'Top {len(top_hotspots)} Risk Hotspots - Fallback'
                            )
                            fig_hotspots_fallback.update_layout(mapbox_style='open-street-map')
                            
                            hotspots_fallback_path = cv_dirs['maps'] / 'fallback_hotspots_map.html'
                            fig_hotspots_fallback.write_html(str(hotspots_fallback_path))
                            print(f'[SUCCESS] Fallback hotspots map generated: {hotspots_fallback_path}')
                            
                            print_map_guidance(cv_dirs['maps'])
                        else:
                            print(f'[WARN] No coordinate columns found in fallback data. Columns: {list(fallback_df.columns)}')
                    else:
                        print('[WARN] Fallback data empty or plotly not available')
                else:
                    print(f'[WARN] No OOF predictions file found at {oof_file}')
                    
            except Exception as e_fallback:
                print(f'[WARN] Fallback map generation failed: {e_fallback}')
                traceback.print_exc()

        if not model_results:
            raise RuntimeError("No models could be trained successfully.")

        # --- STEP 2.5: GENERATE DIAGNOSTIC PLOTS ---
        print("\n--- Step 2.5: Generating Model Diagnostic Plots ---")
        plot_model_comparison(model_results, run_output_dir, cfg.SPLIT_STRATEGY)
        # Residual analysis only meaningful if we have a concrete fitted model & holdout split
        if cfg.SPLIT_STRATEGY in ['RANDOM', 'DIAGNOSTIC'] and best_model_for_shap is not None:
            y_pred_best = best_model_for_shap.predict(X_test_for_shap)
            plot_residual_analysis(y_true=y_test_for_shap, y_pred=y_pred_best,
                                   model_name=best_result['name'], output_dir=run_output_dir)
        elif cfg.SPLIT_STRATEGY == 'BY_ROAD':
            # If BY_ROAD CV was used, we can still produce residual diagnostics
            # from the aggregated out-of-fold predictions (master_pred_df)
            try:
                if 'master_pred_df' in locals() and (not master_pred_df.empty):
                    from stage1_visualizations import plot_oof_residual_analysis
                    plot_oof_residual_analysis(master_pred_df, run_output_dir)
                    # Optional: simple calibration curve (bin actual vs predicted on linear scale)
                    try:
                        if getattr(cfg, 'ENABLE_CALIBRATION_PLOT', True):
                            import matplotlib.pyplot as _plt
                            import numpy as _np
                            dfc = master_pred_df[['predicted_risk','actual_risk']].dropna().copy()
                            # back-transform if needed (if values look like log by having negatives, just expm1 both)
                            try:
                                dfc['pred_lin'] = inverse_target(dfc['predicted_risk'].astype(float).values)
                                dfc['act_lin'] = inverse_target(dfc['actual_risk'].astype(float).values)
                            except Exception:
                                dfc['pred_lin'] = dfc['predicted_risk']
                                dfc['act_lin'] = dfc['actual_risk']
                            # bin by predicted
                            bins = int(getattr(cfg, 'CALIBRATION_NUM_BINS', 10))
                            dfc['bin'] = pd.qcut(dfc['pred_lin'].rank(method='first'), q=bins, labels=False, duplicates='drop')
                            calib = dfc.groupby('bin').agg(pred_mean=('pred_lin','mean'), act_mean=('act_lin','mean')).reset_index(drop=True)
                            fig = _plt.figure(figsize=(6,6))
                            _plt.plot(calib['pred_mean'], calib['act_mean'], marker='o')
                            _plt.plot([calib['pred_mean'].min(), calib['pred_mean'].max()], [calib['pred_mean'].min(), calib['pred_mean'].max()], 'r--', label='Ideal')
                            _plt.xlabel('Predicted (linear)')
                            _plt.ylabel('Actual (linear)')
                            _plt.title('OOF Calibration Curve (binned)')
                            from stage1_config import save_plot as _save
                            _save(fig, 'oof_calibration_curve.png', directory=run_output_dir)
                            _plt.close(fig)
                            print('   Calibration curve saved.')
                    except Exception as e_cal:
                        print(f"[WARN] Calibration plot failed: {e_cal}")
                else:
                    print('[INFO] Skipping residual analysis (no aggregated OOF predictions available).')
            except Exception as e_res_oof:
                print(f"[WARN] OOF residual analysis failed: {e_res_oof}")
        print("Model diagnostic plots complete.")

        # --- STEP 3: RUN SHAP ANALYSIS (Feature Ranking Only) ---
        print("\n--- Step 3: Running SHAP Analysis for Feature Ranking ---")
        if cfg.SPLIT_STRATEGY == 'BY_ROAD' and oof_shap_summary_df is not None:
            print("[INFO] Using unbiased out-of-fold SHAP feature importance (OOF test folds only)")
            shap_summary_df = oof_shap_summary_df
        else:
            if cfg.SPLIT_STRATEGY == 'BY_ROAD':
                print('[INFO] Computing hotspot-focused OOF SHAP for feature importance...')
                
                # Check if hotspot SHAP was computed and use it for feature ranking
                hotspot_shap_file = cv_dirs.get('hotspot_shap', run_output_dir / 'hotspot_shap') / cfg.HOTSPOT_SHAP_FEATURE_IMPORTANCE_CSV
                if hotspot_shap_file.exists():
                    print(f"[INFO] Loading hotspot SHAP feature importance from {hotspot_shap_file}")
                    shap_summary_df = pd.read_csv(hotspot_shap_file)
                    # Rename columns for consistency
                    if 'mean_abs_shap_hotspot' in shap_summary_df.columns:
                        shap_summary_df.rename(columns={'mean_abs_shap_hotspot': 'mean_abs_shap'}, inplace=True)
                else:
                    print('[WARN] No hotspot SHAP found, creating empty summary.')
                    shap_summary_df = pd.DataFrame(columns=['feature','mean_abs_shap'])
            else:
                print('[INFO] Running standard SHAP (non BY_ROAD strategy).')
                from stage1_interpretability import run_shap_analysis
                shap_explanation, shap_summary_df = run_shap_analysis(
                    model=best_model_for_shap,
                    X_train=X_train_for_shap,
                    X_test=X_test_for_shap,
                    top_n=cfg.N_TOP_FEATURES_SHAP,
                    output_dir=run_output_dir,
                    plot_filename=None
                )
        print("SHAP feature ranking complete.")

        # --- STEP 3.5: GENERATE INDIVIDUAL EXPLANATIONS ---
        print("\n--- Step 3.5: Generating Individual Explanations ---")
        individual_explanation_results = {}
        
        if cfg.GENERATE_SEGMENT_EXPLANATIONS or cfg.GENERATE_ROAD_EXPLANATIONS:
            try:
                from stage1_individual_explanations import generate_all_individual_explanations
                
                if cfg.SPLIT_STRATEGY == 'BY_ROAD' and 'master_pred_df' in locals() and not master_pred_df.empty:
                    print(f"[INFO] Generating individual explanations for BY_ROAD strategy using OOF predictions...")
                    # === START: REPLACEMENT LOGIC FOR EXPLANATION MODEL ===
                    try:
                        print("[INFO] Preparing a consistently preprocessed model for explanations...")

                        # 1. Create and fit a preprocessor on the ENTIRE dataset.
                        #    This mirrors the process used inside the CV folds.
                        X_processed, _, preprocessor_for_exp, _ = fit_transform_preprocessor(
                            X, None, getattr(cfg, 'NUMERICAL_FEATURES', []), getattr(cfg, 'CATEGORICAL_FEATURES', [])
                        )

                        # Defensive diagnostic: ensure preprocessed X has features
                        if X_processed is None or X_processed.shape[1] == 0:
                            print("[ERROR] Preprocessed feature matrix for explanation model is empty. Skipping explanation model training.")
                            print("[DIAG] Original feature columns (first 50):", list(X.columns)[:50])
                            print("[DIAG] Config NUMERICAL_FEATURES (first 50):", getattr(cfg, 'NUMERICAL_FEATURES', [])[:50])
                            print("[DIAG] Config CATEGORICAL_FEATURES (first 50):", getattr(cfg, 'CATEGORICAL_FEATURES', [])[:50])
                            # set empty results so later code won't fail
                            individual_explanation_results = {}
                        else:

                                # 2. EXPERT FIX: Always pass global model and preprocessor for fallback
                                fold_index_path = cv_dirs['fold_results'] / 'fold_artifact_index.csv'
                                if fold_index_path.exists():
                                    print(f"[INFO] Using per-fold artifacts index for explanations: {fold_index_path}")
                                    print(f"[INFO] Global model fallback available: {best_model_for_shap is not None}")
                                    # Allow the explanations module to choose the preferred id column (pass None)
                                    individual_explanation_results = generate_all_individual_explanations(
                                        master_pred_df=master_pred_df,
                                        model=best_model_for_shap,  # EXPERT FIX: Pass global model as fallback
                                        X_features=X_full_unfiltered,  # Use FULL unfiltered features
                                        output_dir=run_output_dir,
                                        metadata_df=metadata_full_unfiltered,  # Use FULL unfiltered metadata
                                        id_col_name=None,
                                        preprocessor=global_preprocessor,  # EXPERT FIX: Pass global preprocessor as fallback
                                        fold_artifact_index=str(fold_index_path)
                                    )
                                    print(f"[SUCCESS] Individual explanations generated (using fold artifacts): {individual_explanation_results.get('total_files_created', 0)} files")
                                    
                                    # Generate dataset-level SHAP analysis
                                    if 'segment_explanations' in individual_explanation_results:
                                        try:
                                            from stage1_individual_explanations import generate_dataset_level_shap_analysis
                                            print("\n" + "="*80)
                                            print("STEP 3.5.1: Road-Based Dataset SHAP Analysis")
                                            print("="*80)
                                            print("[INFO] Analyzing datasets present in road-based top-K selection")
                                            print("[INFO] This shows which datasets contribute to highest-risk roads\n")
                                            dataset_analysis = generate_dataset_level_shap_analysis(
                                                segment_explanations=individual_explanation_results['segment_explanations'],
                                                master_pred_df=master_pred_df,
                                                output_dir=run_output_dir
                                            )
                                            print(f"[SUCCESS] Road-based dataset analysis: {len(dataset_analysis.get('files_created', []))} files created")
                                            print(f"[INFO] Output: dataset_shap_analysis/road_based_top_k/")
                                            print(f"[INFO] Note: May cover fewer than 12 datasets (road-based perspective)\n")
                                            
                                            # Generate per-dataset top-risk SHAP analysis
                                            try:
                                                from stage1_individual_explanations import generate_per_dataset_top_risk_shap
                                                print("\n" + "="*80)
                                                print("STEP 3.5.2: Per-Dataset Top-Risk SHAP Analysis")
                                                print("="*80)
                                                print("[INFO] Analyzing top 5% segments from FULL dataset for each dataset")
                                                print("[INFO] This ensures ALL 12 datasets are represented fairly\n")
                                                per_dataset_analysis = generate_per_dataset_top_risk_shap(
                                                    master_pred_df=master_pred_df,
                                                    model=best_model_for_shap,
                                                    X_features=X_full_unfiltered,  # Use FULL unfiltered features
                                                    preprocessor=global_preprocessor,
                                                    output_dir=run_output_dir,
                                                    top_pct=0.05,
                                                    fold_results_dir=cv_dirs['fold_results'],
                                                metadata_df=metadata_full_unfiltered  # Use FULL unfiltered metadata
                                                )
                                                print(f"[SUCCESS] Per-dataset analysis: {len(per_dataset_analysis.get('files_created', []))} files created")
                                                print(f"[INFO] Output: dataset_shap_analysis/per_dataset_top_risk/\n")
                                                
                                                # Generate regional SHAP aggregation
                                                try:
                                                    from stage1_individual_explanations import generate_regional_shap_analysis
                                                    from pathlib import Path
                                                    analysis_root = Path(run_output_dir) / 'dataset_shap_analysis'
                                                    per_dataset_dir = analysis_root / 'per_dataset_top_risk'
                                                    full_population_dir = analysis_root / 'per_dataset_full_population'
                                                    road_topk_dir = analysis_root / 'road_based_top_k'
                                                    if per_dataset_dir.exists():
                                                        print("\n" + "="*80)
                                                        print("STEP 3.5.3: Regional SHAP Aggregation")
                                                        print("="*80)
                                                        print("[INFO] Aggregating per-dataset results by geographic region")
                                                        print("[INFO] Creates bar and violin plots for 3 regions\n")
                                                        regional_analysis = generate_regional_shap_analysis(
                                                            per_dataset_dir=per_dataset_dir,
                                                            output_dir=run_output_dir,
                                                            full_population_dir=full_population_dir,
                                                            road_topk_dir=road_topk_dir
                                                        )
                                                        print(f"[SUCCESS] Regional analysis: {len(regional_analysis.get('files_created', []))} files created")
                                                        print(f"[INFO] Output: dataset_shap_analysis/regional_analysis/\n")
                                                    else:
                                                        print("[WARN] Per-dataset directory not found, skipping regional analysis")
                                                except Exception as e_regional:
                                                    print(f"[WARN] Regional SHAP analysis failed: {e_regional}")
                                                    traceback.print_exc()
                                            except Exception as e_per_dataset:
                                                print(f"[WARN] Per-dataset top-risk SHAP analysis failed: {e_per_dataset}")
                                                traceback.print_exc()
                                        except Exception as e_dataset:
                                            print(f"[WARN] Road-based dataset SHAP analysis failed: {e_dataset}")
                                            traceback.print_exc()
                                else:
                                    # Fallback: use global model directly
                                    print(f"[WARN] fold_artifact_index.csv not found in {cv_dirs['fold_results']}; using global model.")
                                    individual_explanation_results = generate_all_individual_explanations(
                                        master_pred_df=master_pred_df,
                                        model=best_model_for_shap,  # Use global model
                                        X_features=X_full_unfiltered,  # Use FULL unfiltered features
                                        output_dir=run_output_dir,
                                        metadata_df=metadata_full_unfiltered,  # Use FULL unfiltered metadata
                                        id_col_name=None,
                                        preprocessor=global_preprocessor  # Use global preprocessor
                                    )
                                    print(f"[SUCCESS] Individual explanations generated (global model): {individual_explanation_results.get('total_files_created', 0)} files")
                                    
                                    # Generate dataset-level SHAP analysis
                                    if 'segment_explanations' in individual_explanation_results:
                                        try:
                                            from stage1_individual_explanations import generate_dataset_level_shap_analysis
                                            print("\n" + "="*80)
                                            print("STEP 3.5.1: Road-Based Dataset SHAP Analysis")
                                            print("="*80)
                                            print("[INFO] Analyzing datasets present in road-based top-K selection")
                                            print("[INFO] This shows which datasets contribute to highest-risk roads\n")
                                            dataset_analysis = generate_dataset_level_shap_analysis(
                                                segment_explanations=individual_explanation_results['segment_explanations'],
                                                master_pred_df=master_pred_df,
                                                output_dir=run_output_dir
                                            )
                                            print(f"[SUCCESS] Road-based dataset analysis: {len(dataset_analysis.get('files_created', []))} files created")
                                            print(f"[INFO] Output: dataset_shap_analysis/road_based_top_k/")
                                            print(f"[INFO] Note: May cover fewer than 12 datasets (road-based perspective)\n")
                                            
                                            # Generate per-dataset top-risk SHAP analysis
                                            try:
                                                from stage1_individual_explanations import generate_per_dataset_top_risk_shap
                                                print("\n" + "="*80)
                                                print("STEP 3.5.2: Per-Dataset Top-Risk SHAP Analysis")
                                                print("="*80)
                                                print("[INFO] Analyzing top 5% segments from FULL dataset for each dataset")
                                                print("[INFO] This ensures ALL 12 datasets are represented fairly\n")
                                                per_dataset_analysis = generate_per_dataset_top_risk_shap(
                                                    master_pred_df=master_pred_df,
                                                    model=best_model_for_shap,
                                                    X_features=X_full_unfiltered,  # Use FULL unfiltered features
                                                    preprocessor=global_preprocessor,
                                                    output_dir=run_output_dir,
                                                    top_pct=0.05,
                                                    fold_results_dir=cv_dirs['fold_results'],
                                                metadata_df=metadata_full_unfiltered  # Use FULL unfiltered metadata
                                                )
                                                print(f"[SUCCESS] Per-dataset analysis: {len(per_dataset_analysis.get('files_created', []))} files created")
                                                print(f"[INFO] Output: dataset_shap_analysis/per_dataset_top_risk/\n")
                                                
                                                # Generate regional SHAP aggregation
                                                try:
                                                    from stage1_individual_explanations import generate_regional_shap_analysis
                                                    from pathlib import Path
                                                    analysis_root = Path(run_output_dir) / 'dataset_shap_analysis'
                                                    per_dataset_dir = analysis_root / 'per_dataset_top_risk'
                                                    full_population_dir = analysis_root / 'per_dataset_full_population'
                                                    road_topk_dir = analysis_root / 'road_based_top_k'
                                                    if per_dataset_dir.exists():
                                                        print("\n" + "="*80)
                                                        print("STEP 3.5.3: Regional SHAP Aggregation")
                                                        print("="*80)
                                                        print("[INFO] Aggregating per-dataset results by geographic region")
                                                        print("[INFO] Creates bar and violin plots for 3 regions\n")
                                                        regional_analysis = generate_regional_shap_analysis(
                                                            per_dataset_dir=per_dataset_dir,
                                                            output_dir=run_output_dir,
                                                            full_population_dir=full_population_dir,
                                                            road_topk_dir=road_topk_dir
                                                        )
                                                        print(f"[SUCCESS] Regional analysis: {len(regional_analysis.get('files_created', []))} files created")
                                                        print(f"[INFO] Output: dataset_shap_analysis/regional_analysis/\n")
                                                    else:
                                                        print("[WARN] Per-dataset directory not found, skipping regional analysis")
                                                except Exception as e_regional:
                                                    print(f"[WARN] Regional SHAP analysis failed: {e_regional}")
                                                    traceback.print_exc()
                                            except Exception as e_per_dataset:
                                                print(f"[WARN] Per-dataset top-risk SHAP analysis failed: {e_per_dataset}")
                                                traceback.print_exc()
                                        except Exception as e_dataset:
                                            print(f"[WARN] Road-based dataset SHAP analysis failed: {e_dataset}")
                                            traceback.print_exc()

                    except Exception as e_byroad:
                        print(f"[WARN] BY_ROAD individual explanations failed: {e_byroad}")
                        traceback.print_exc()
                        individual_explanation_results = {}
                    # === END: REPLACEMENT LOGIC ===
                        
                elif cfg.SPLIT_STRATEGY in ['RANDOM', 'DIAGNOSTIC'] and best_model_for_shap is not None and X_test_for_shap is not None:
                    print(f"[INFO] Generating individual explanations for {cfg.SPLIT_STRATEGY} strategy...")
                    
                    # For RANDOM/DIAGNOSTIC strategies, we have clean holdout data
                    # Create master_pred_df from test data
                    test_predictions = best_model_for_shap.predict(X_test_for_shap)
                    
                    # Create synthetic master_pred_df for explanations
                    master_synthetic = pd.DataFrame({
                        'segment_id': [f'test_{i}' for i in range(len(X_test_for_shap))],
                        'road_id': [f'road_{i//10}' for i in range(len(X_test_for_shap))],  # Group every 10 segments
                        'predicted_risk': test_predictions,
                        'actual_risk': y_test_for_shap.values if hasattr(y_test_for_shap, 'values') else y_test_for_shap
                    })
                    
                    individual_explanation_results = generate_all_individual_explanations(
                        master_pred_df=master_synthetic,
                        model=best_model_for_shap,
                        X_features=X_test_for_shap,
                        output_dir=run_output_dir
                    )
                    # Pass metadata and ID column to ensure robust id->position mapping
                    # If metadata_df is None this parameter is optional and ignored by the callee
                    individual_explanation_results = generate_all_individual_explanations(
                        master_pred_df=master_synthetic,
                        model=best_model_for_shap,
                        X_features=X_test_for_shap,
                        output_dir=run_output_dir,
                        metadata_df=metadata_full_unfiltered,  # Use FULL unfiltered metadata
                        id_col_name=id_col_use
                    )
                    
                    print(f"[SUCCESS] Individual explanations generated: {individual_explanation_results.get('total_files_created', 0)} files")
                    
                    # Generate dataset-level SHAP analysis (simplified for RANDOM/DIAGNOSTIC)
                    if 'segment_explanations' in individual_explanation_results:
                        try:
                            from stage1_individual_explanations import generate_dataset_level_shap_analysis
                            print("\n" + "="*80)
                            print("STEP 3.5.1: Dataset SHAP Analysis (Test Set)")
                            print("="*80)
                            print("[INFO] Analyzing test set segments (RANDOM/DIAGNOSTIC strategy)")
                            print("[INFO] Note: Per-dataset and regional analyses require BY_ROAD CV strategy\n")
                            dataset_analysis = generate_dataset_level_shap_analysis(
                                segment_explanations=individual_explanation_results['segment_explanations'],
                                master_pred_df=master_synthetic,
                                output_dir=run_output_dir
                            )
                            print(f"[SUCCESS] Test set dataset analysis: {len(dataset_analysis.get('files_created', []))} files created")
                            print(f"[INFO] Output: dataset_shap_analysis/road_based_top_k/")
                            print(f"[INFO] For comprehensive per-dataset analysis, use BY_ROAD CV strategy\n")
                            
                        except Exception as e_dataset:
                            print(f"[WARN] Dataset-level SHAP analysis failed: {e_dataset}")
                            traceback.print_exc()
                    
                else:
                    print("[INFO] Skipping individual explanations - insufficient data or disabled")
                    
            except ImportError as e_import:
                print(f"[WARN] Could not import individual explanations module: {e_import}")
            except Exception as e_explain:
                print(f"[WARN] Individual explanation generation failed: {e_explain}")
                traceback.print_exc()
                individual_explanation_results = {}
        else:
            print("[INFO] Individual explanations disabled in configuration")

        # --- STEP 4 & 5 (Downstream Analysis & Reporting) ---
        print("\n--- Step 4: Performing Domain-Specific Analyses & Validation ---")
        # Improved debug: show strategy and whether SHAP holdout splits are present
        print(f"[DEBUG] SPLIT_STRATEGY={cfg.SPLIT_STRATEGY}; X_test_for_shap present: {X_test_for_shap is not None}; y_test_for_shap present: {y_test_for_shap is not None}")
        if X_test_for_shap is not None:
            print(f"[DEBUG] X_test_for_shap shape: {X_test_for_shap.shape}")
        if y_test_for_shap is not None:
            try:
                print(f"[DEBUG] y_test_for_shap shape: {y_test_for_shap.shape}")
                print("[DEBUG] y_test_for_shap head:\n", y_test_for_shap.head())
            except Exception:
                print("[DEBUG] y_test_for_shap (non-pandas object)")

        # If we expect holdout splits (non-BY_ROAD) but they're missing, fail fast with clear message
        if cfg.SPLIT_STRATEGY not in ['BY_ROAD'] and (X_test_for_shap is None or y_test_for_shap is None):
            raise RuntimeError("[ERROR] Expected holdout X_test_for_shap/y_test_for_shap for SHAP analysis but they are None. Check earlier training/evaluation logs.")
        
        if cfg.SPLIT_STRATEGY == 'BY_ROAD':
            print('[INFO] Adapting analyze_high_risk_segments for BY_ROAD strategy using OOF predictions...')
            
            # Create synthetic test set from OOF predictions for high-risk analysis
            try:
                # Use the master predictions and metadata to create a synthetic test environment
                if 'master_pred_df' in locals() and not master_pred_df.empty:
                    # Select highest risk segments across all roads for analysis
                    top_segments = master_pred_df.nlargest(100, 'predicted_risk')  # Top 100 riskiest segments
                    
                    # Get corresponding features and metadata
                    segment_ids = top_segments['segment_id'].tolist()
                    # Use preferred id column to locate indices in metadata
                    meta_id_col = id_col_use if id_col_use in metadata_df.columns else cfg.ID_COL
                    segment_indices = metadata_df[metadata_df[meta_id_col].isin(segment_ids)].index
                    
                    if len(segment_indices) > 0:
                        # Create synthetic test data from these high-risk segments
                        X_test_synth = X.iloc[segment_indices]
                        y_test_synth = y.iloc[segment_indices]
                        
                        # Remove metadata columns for model compatibility
                        cols_to_drop_synth = set(cfg.METADATA_COLS) | set(cfg.FEATURE_EXCLUSIONS) | {cfg.ROAD_COLUMN_NAME}
                        X_test_features_synth = X_test_synth.drop(columns=[c for c in cols_to_drop_synth if c in X_test_synth.columns], errors='ignore')
                        
                        # Create a representative model for SHAP analysis (refit best model on full data sample)
                        print(f"[INFO] Creating representative {best_model_name} model for high-risk analysis...")
                        
                        # Sample training data to avoid memory issues
                        train_sample_size = min(5000, len(X))
                        train_indices = np.random.choice(len(X), size=train_sample_size, replace=False)
                        X_train_sample = X.iloc[train_indices]
                        y_train_sample = y.iloc[train_indices]
                        X_train_features_sample = X_train_sample.drop(columns=[c for c in cols_to_drop_synth if c in X_train_sample.columns], errors='ignore')
                        
                        # Fit representative model
                        repr_model = get_model_instance(best_model_name, random_state=cfg.RANDOM_STATE)
                        repr_model.fit(X_train_features_sample, y_train_sample)
                        
                        # Run high-risk analysis with synthetic data
                        high_risk_summary = analyze_high_risk_segments(
                            original_df=metadata_df.iloc[segment_indices],
                            X_test=X_test_features_synth,
                            y_test=y_test_synth,
                            model=repr_model,
                            shap_summary_df=shap_summary_df,
                            top_n=20,  # Analyze top 20 high-risk segments
                            output_dir=run_output_dir
                        )
                        
                        # Populate global SHAP holdout vars so downstream code can use them if needed
                        try:
                            # Align by ID using cfg.ID_COL if available
                            X_test_for_shap = X_test_features_synth.copy()
                            y_test_for_shap = y_test_synth.copy()
                        except Exception:
                            pass

                        print(f"[INFO] High-risk analysis completed for {len(segment_indices)} segments from OOF predictions")
                    else:
                        print("[WARN] No matching segments found for high-risk analysis")
                        high_risk_summary = pd.DataFrame()
                else:
                    print("[WARN] No master predictions available for high-risk analysis")
                    high_risk_summary = pd.DataFrame()
                    
            except Exception as e_hr:
                print(f"[WARN] High-risk analysis failed: {e_hr}")
                high_risk_summary = pd.DataFrame()
        else:
            high_risk_summary = analyze_high_risk_segments(
                original_df=metadata_df, # Use metadata_df which contains original IDs and coords
                X_test=X_test_for_shap,
                y_test=y_test_for_shap,
                model=best_model_for_shap,
                shap_summary_df=shap_summary_df,
                top_n=10,
                output_dir=run_output_dir
            )
        print("\n--- Step 5: Generating Final Consolidated Report (extended sections) ---")
        generate_final_report(
            metrics_results=model_results,
            shap_summary_df=shap_summary_df,
            countermeasure_summary=high_risk_summary,
            hypothesis_results=pd.DataFrame(),
            config_params={k: v for k, v in vars(cfg).items() if not k.startswith('__')},
            output_dir=run_output_dir,
            filename=cfg.FINAL_MARKDOWN_REPORT_FILE
        )
        print(f"Final report generated in: {run_output_dir}")

        # === STEP 9: Reproducibility Artifacts ===
        if cfg.WRITE_RUN_MANIFEST:
            try:
                import platform, pkgutil, importlib
                manifest = {
                    'timestamp': run_timestamp,
                    'strategy': cfg.SPLIT_STRATEGY,
                    'primary_model': getattr(cfg, 'PRIMARY_MODEL_TYPE', None),
                    'random_state': cfg.RANDOM_STATE,
                    'python_version': platform.python_version(),
                }
                if cfg.CAPTURE_LIBRARY_VERSIONS:
                    versions = {}
                    for pkg in ['pandas','numpy','scikit-learn','xgboost','lightgbm','catboost','shap']:
                        try:
                            m = importlib.import_module(pkg.replace('-', '_'))
                            versions[pkg] = getattr(m, '__version__', 'unknown')
                        except Exception:
                            versions[pkg] = 'not_installed'
                    manifest['library_versions'] = versions
                with open(run_output_dir / cfg.RUN_MANIFEST_FILE, 'w') as f:
                    json.dump(manifest, f, indent=2)
                print(f"[INFO] Run manifest saved: {run_output_dir / cfg.RUN_MANIFEST_FILE}")
            except Exception as e_manifest:
                print(f"[WARN] Failed to write run manifest: {e_manifest}")

        # === STEP 12: Publication Exports ===
        if cfg.EXPORT_PUBLICATION_TABLES:
            pub_dir = run_output_dir / cfg.PUBLICATION_EXPORT_DIR
            pub_dir.mkdir(parents=True, exist_ok=True)
            try:
                # Top features (prefer hotspot_SHAP else normal SHAP)
                hotspot_fi_path = run_output_dir / 'hotspot_shap' / cfg.HOTSPOT_SHAP_FEATURE_IMPORTANCE_CSV
                if hotspot_fi_path.exists():
                    hf = pd.read_csv(hotspot_fi_path).head(20)
                    hf.to_csv(pub_dir / cfg.PUB_TOP_FEATURES_CSV, index=False)
                elif 'shap_summary_df' in locals() and shap_summary_df is not None:
                    shap_summary_df.head(20).to_csv(pub_dir / cfg.PUB_TOP_FEATURES_CSV, index=False)
                # Ranking metrics aggregated
                agg_rank_path = run_output_dir / 'fold_results' / cfg.RANKING_METRICS_AGG_JSON
                if agg_rank_path.exists():
                    with open(agg_rank_path) as f:
                        import json as _json
                        agg_rank = _json.load(f)
                    # flatten
                    rows = []
                    for k, v in agg_rank.items():
                        row = {'K_or_Aggregate': k}
                        row.update(v)
                        rows.append(row)
                    pd.DataFrame(rows).to_csv(pub_dir / cfg.PUB_RANKING_AGG_CSV, index=False)
                # Coverage metrics
                cov_path = run_output_dir / 'fold_results' / cfg.COUNTERMEASURE_COVERAGE_CSV
                if cov_path.exists():
                    pd.read_csv(cov_path).to_csv(pub_dir / cfg.PUB_COVERAGE_CSV, index=False)
                print(f"[INFO] Publication export tables saved to {pub_dir}")
            except Exception as e_pub:
                print(f"[WARN] Publication exports failed: {e_pub}")

        # === STEP 11: Basic Test / Validation Summary ===
        if getattr(cfg, 'GENERATE_BASIC_TEST_SUMMARY', False):
            try:
                summary = { 'warnings': [] }
                from pathlib import Path as _Path
                fold_dir = run_output_dir / 'fold_results'
                hotspot_dir = run_output_dir / 'hotspot_shap'
                files_to_check = {
                    'oof_predictions': fold_dir / 'oof_predictions_segments.csv',
                    'ranking_long': fold_dir / getattr(cfg, 'RANKING_METRICS_LONG_CSV', 'per_road_hotspot_metrics_long.csv'),
                    'ranking_agg': fold_dir / getattr(cfg, 'RANKING_METRICS_AGG_JSON', 'hotspot_ranking_metrics_aggregated.json'),
                    'hotspot_fi': hotspot_dir / getattr(cfg, 'HOTSPOT_SHAP_FEATURE_IMPORTANCE_CSV', 'hotspot_shap_feature_importance.csv'),
                    'countermeasure_coverage': fold_dir / getattr(cfg, 'COUNTERMEASURE_COVERAGE_CSV', 'countermeasure_coverage_summary.csv')
                }
                file_status = {}
                import json as _json
                import pandas as _pd
                for label, path_obj in files_to_check.items():
                    file_status[label] = 'exists' if path_obj.exists() else 'missing'
                summary['file_status'] = file_status
                # OOF coverage
                if files_to_check['oof_predictions'].exists():
                    oof_df = _pd.read_csv(files_to_check['oof_predictions'])
                    seg_count = int(oof_df['segment_id'].nunique()) if 'segment_id' in oof_df.columns else None
                    summary['oof_prediction_coverage'] = {
                        'unique_segments_with_predictions': seg_count,
                        'total_rows': int(oof_df.shape[0])
                    }
                    if 'fold' in oof_df.columns:
                        fold_counts = oof_df.groupby('fold').size().to_dict()
                        summary['oof_prediction_coverage']['fold_counts'] = fold_counts
                # Ranking metrics quick stats
                if files_to_check['ranking_long'].exists():
                    rk_df = _pd.read_csv(files_to_check['ranking_long'])
                    for metric in ['precision@K','recall@K','overlap@K','nDCG@K','RR@K']:
                        if metric in rk_df.columns:
                            summary.setdefault('ranking_metrics_summary', {})[metric] = {
                                'mean': float(rk_df[metric].mean()),
                                'std': float(rk_df[metric].std())
                            }
                # Hotspot SHAP feature snapshot
                if files_to_check['hotspot_fi'].exists():
                    hf_df = _pd.read_csv(files_to_check['hotspot_fi']).head(10)
                    summary['hotspot_top_features'] = hf_df['feature'].tolist() if 'feature' in hf_df.columns else []
                # Countermeasure coverage rows
                if files_to_check['countermeasure_coverage'].exists():
                    cov_df = _pd.read_csv(files_to_check['countermeasure_coverage'])
                    summary['countermeasure_coverage_rows'] = int(cov_df.shape[0])
                # GPU configuration summary
                summary['gpu_config'] = {
                    'use_gpu': bool(getattr(cfg, 'USE_GPU', False)),
                    'device_id': int(getattr(cfg, 'GPU_DEVICE_ID', 0)),
                    'catboost_gpu': bool(getattr(cfg, 'GPU_ENABLE_CATBOOST', False)),
                    'lightgbm_gpu': bool(getattr(cfg, 'GPU_ENABLE_LIGHTGBM', False)),
                    'xgboost_gpu': bool(getattr(cfg, 'GPU_ENABLE_XGBOOST', False))
                }
                # Persist test summary
                test_sum_path = run_output_dir / getattr(cfg, 'TEST_SUMMARY_FILE', 'test_summary.json')
                with open(test_sum_path, 'w') as fts:
                    _json.dump(summary, fts, indent=2)
                print(f"[INFO] Basic test summary saved: {test_sum_path}")
            except Exception as e_test:
                print(f"[WARN] Could not generate basic test summary: {e_test}")

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    end_time = time.time()
    print("\n=====================================================================")
    print(f"  Pipeline Finished Successfully in {end_time - start_time:.2f} seconds")
    print("=====================================================================")

if __name__ == "__main__":
    main()