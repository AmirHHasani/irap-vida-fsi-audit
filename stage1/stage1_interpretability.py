import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import re
import traceback

# Import helper function for saving plots and config variables
from stage1_config import OUTPUT_DIR, N_TOP_FEATURES_SHAP, save_plot, COMPUTE_SHAP_IN_MODEL_SELECTION

# Heavy optional dependencies: import lazily or guard to avoid ImportError at module import time
try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None

try:
    # model libraries used in functions; import if available
    from catboost import CatBoostRegressor, Pool
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None
    Pool = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None

try:
    import xgboost as xgb
    XGBRegressor = getattr(xgb, 'XGBRegressor', None)
except Exception:  # pragma: no cover - optional dependency
    xgb = None
    XGBRegressor = None

# === NEW: Explainer cache to avoid repeated TreeExplainer creation ===
_EXPLAINER_CACHE = {}

def _get_or_create_tree_explainer(model):
    """Create or return a cached shap.TreeExplainer.

    Raises a clear ImportError if shap is not available.
    """
    if shap is None:
        raise ImportError("shap is required for explainers but is not installed in this environment")
    key = id(model)
    if key not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[key] = shap.TreeExplainer(model)
    return _EXPLAINER_CACHE[key]

# === NEW: Safe XGBoost SHAP using native pred_contribs to bypass SHAP binary parsing ===
def safe_xgb_shap(model, X_df: pd.DataFrame):
    """Compute SHAP values for an xgboost model using pred_contribs API.

    This function will try to use the xgboost native API. If xgboost is not
    available, raises ImportError.
    """
    if xgb is None:
        raise ImportError("xgboost is required for safe_xgb_shap but is not installed in this environment")
    booster = model.get_booster() if hasattr(model, 'get_booster') else model
    dmatrix = xgb.DMatrix(X_df)
    contribs = booster.predict(dmatrix, pred_contribs=True)  # (n_samples, n_features+1)
    # Last column is the expected value (base margin)
    base_values = contribs[:, -1]
    shap_values = contribs[:, :-1]
    return shap_values, base_values

def sanitize_dataframe_for_shap(df):
    """
    Sanitize DataFrame column names to prevent encoding errors in SHAP.
    This is a defensive function that ensures any DataFrame passed to SHAP
    has clean, ASCII-only column names.
    """
    def sanitize_col_name(name):
        # Convert to lowercase
        name = str(name).lower()
        # Replace special characters, parentheses, and spaces with a single underscore
        name = re.sub(r'[\s\(\)\-\/]+', '_', name)
        # Remove any other non-alphanumeric characters (except underscore)
        name = re.sub(r'[^a-z0-9_]+', '', name)
        # Collapse multiple underscores into one
        name = re.sub(r'[_]+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Ensure the name is ASCII
        return name.encode('ascii', 'ignore').decode('ascii')
    
    # Create a copy and sanitize column names
    df_clean = df.copy()
    sanitized_columns = {col: sanitize_col_name(col) for col in df_clean.columns}
    df_clean.rename(columns=sanitized_columns, inplace=True)
    
    return df_clean

def save_road_shap_summary_plot(model, X_road, output_path, top_n=N_TOP_FEATURES_SHAP):
    """
    Generate and save a SHAP summary bar plot for a single road's data.
    """
    import shap
    import matplotlib.pyplot as plt
    if X_road.shape[0] < 2:
        print(f"[INFO] Not enough segments for SHAP summary plot for this road.")
        return
    X_road_clean = sanitize_dataframe_for_shap(X_road)
    try:
        if isinstance(model, XGBRegressor):
            shap_values, base_vals = safe_xgb_shap(model, X_road_clean)
        else:
            explainer = _get_or_create_tree_explainer(model)
            shap_values = explainer.shap_values(X_road_clean)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_road_clean, plot_type='bar', max_display=top_n, show=False)
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"[INFO] Road SHAP summary plot saved: {output_path}")
    except Exception as e:
        print(f"[WARNING] Failed road SHAP for {output_path}: {e}")

# stage1_interpretability.py
"""
SHAP analysis for model interpretability in the Stage 1 risk modeling pipeline.

This module is responsible for "opening the black box" of the trained model
to understand which features are driving its predictions.
"""

def run_shap_analysis(model, X_train, X_test, top_n=N_TOP_FEATURES_SHAP, plot_filename="shap_summary_plot.png", output_dir=OUTPUT_DIR):
    """
    Computes SHAP values, saves enhanced summary plots, generates dependence plots,
    and returns the top features.
    """
    print("   Initializing SHAP explainer...")
    print(f"   [DEBUG] Input X_test shape: {X_test.shape}")
    print(f"   [DEBUG] Input X_test dtypes: {X_test.dtypes.value_counts()}")
    start_time = time.time()

    # CRITICAL: Sanitize the DataFrames before passing to SHAP
    X_test_clean = sanitize_dataframe_for_shap(X_test)
    print("   [INFO] DataFrame column names sanitized for SHAP compatibility.")

    # Use TreeExplainer for tree-based models, which is faster and more accurate
    print(f"   Detected model ({type(model).__name__}). Using shap.TreeExplainer.")
    print(f"   [DEBUG] Creating TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    print(f"   [DEBUG] TreeExplainer created successfully")
    
    # For consistent processing, ensure all data is numeric (matching training data format)
    print(f"   [DEBUG] Ensuring all data is numeric for SHAP calculation...")
    X_test_for_shap = X_test_clean.copy()
    
    # Convert any categorical columns to numeric codes if they exist
    cat_cols = X_test_for_shap.select_dtypes(include=['category']).columns
    if not cat_cols.empty:
        print(f"   Converting categorical columns to integer codes: {cat_cols.tolist()}")
        for col in cat_cols:
            X_test_for_shap[col] = X_test_for_shap[col].cat.codes
    
    # Ensure all columns are numeric, coercing errors and filling NaNs
    X_test_for_shap = X_test_for_shap.apply(pd.to_numeric, errors='coerce').fillna(0)
    print(f"   [DEBUG] Final data dtypes for SHAP:\n{X_test_for_shap.dtypes.value_counts()}")
    
    # Compute SHAP values
    print(f"   [DEBUG] Computing SHAP values...")
    try:
        if isinstance(model, XGBRegressor):
            shap_values, base_vals = safe_xgb_shap(model, X_test_for_shap)
            expected_value = np.mean(base_vals)
        else:
            explainer = _get_or_create_tree_explainer(model)
            shap_values = explainer.shap_values(X_test_for_shap)
            expected_value = explainer.expected_value
    except Exception as e:
        print(f"   [ERROR] SHAP computation failed: {e}")
        raise
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=X_test_for_shap,
        feature_names=X_test_clean.columns.tolist()  # Use clean column names
    )
    print(f"   [DEBUG] SHAP Explanation object created successfully")

    # --- 1. Generate SHAP Beeswarm + Violin Plots (Optional) ---
    if plot_filename is not None:
        print("   Generating SHAP summary plots (beeswarm + violin)...")

        # Helper: capture current figure after a SHAP plotting call
        def _save_shap_figure(fname):
            fig = plt.gcf()
            fig.set_size_inches(12, 0.4 * min(top_n, shap_explanation.values.shape[1]) + 2)
            plt.subplots_adjust(left=0.35)
            save_plot(fig, fname, directory=output_dir)
            plt.close(fig)

        # --- 1a. Beeswarm (dot) plot — canonical blue-red gradient ---
        try:
            plt.close('all')
            # Prefer modern shap.plots.beeswarm (SHAP ≥ 0.40)
            if hasattr(shap, 'plots') and hasattr(shap.plots, 'beeswarm'):
                shap.plots.beeswarm(shap_explanation, max_display=top_n, show=False)
            else:
                shap.summary_plot(shap_explanation, max_display=top_n, show=False,
                                  plot_type='dot')
            _save_shap_figure(plot_filename)
            print(f"   [OK] SHAP beeswarm plot saved: {plot_filename}")
        except Exception as e:
            print(f"   [WARN] Beeswarm plot failed ({e}); trying legacy summary_plot...")
            try:
                plt.close('all')
                shap.summary_plot(shap_explanation, max_display=top_n, show=False)
                _save_shap_figure(plot_filename)
            except Exception as e2:
                print(f"   [ERROR] All beeswarm attempts failed: {e2}")
                traceback.print_exc()

        # --- 1b. Violin plot — shows feature-value distributions ---
        violin_filename = plot_filename.replace('.png', '_violin.png')
        try:
            plt.close('all')
            if hasattr(shap, 'plots') and hasattr(shap.plots, 'violin'):
                shap.plots.violin(shap_explanation, max_display=top_n, show=False)
            else:
                shap.summary_plot(shap_explanation, max_display=top_n, show=False,
                                  plot_type='violin')
            _save_shap_figure(violin_filename)
            print(f"   [OK] SHAP violin plot saved: {violin_filename}")
        except Exception as e:
            print(f"   [WARN] Violin plot failed ({e}), skipping.")
    else:
        print("   Skipping global SHAP summary plot (plot_filename=None)")

    # --- 2. Feature Ranking (Always Generated) ---
    print("   Ranking features by mean absolute SHAP value...")
    mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)
    shap_summary_df = pd.DataFrame({
        'feature': X_test_clean.columns,  # Use clean column names
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    # --- 3. Generate SHAP Dependence Plots for Top Features (Optional) ---
    if plot_filename is not None:
        print("   Generating SHAP dependence plots for top features...")
        # Generate a plot for the top 5 features
        for feature in shap_summary_df['feature'].head(5).tolist():
            try:
                fig, ax = plt.subplots()
                shap.dependence_plot(feature, shap_values, X_test_for_shap, ax=ax, show=False)
                plt.tight_layout()
                # Create a safe filename
                safe_feature_name = "".join(c for c in feature if c.isalnum() or c in (' ', '_')).rstrip()
                dep_plot_filename = f"shap_dependence_{safe_feature_name.replace(' ', '_')}.png"
                save_plot(fig, dep_plot_filename, directory=output_dir)
                plt.close(fig)
            except Exception as e:
                print(f"   Warning: Could not generate dependence plot for '{feature}'. Reason: {e}")

    print(f"   Top 5 drivers: {shap_summary_df['feature'].head(5).tolist()}")
    return shap_explanation, shap_summary_df.head(top_n)

def save_segment_shap_explanation(model, X_segment, X_background, output_path, segment_id):
    """
    Generate and save a SHAP waterfall plot for a single segment.
    
    Args:
        model: Trained model
        X_segment: Single row DataFrame for the segment to explain
        X_background: Background data for SHAP explainer
        output_path: Path to save the plot
        segment_id: ID of the segment for the title
    """
    try:
        # CRITICAL: Sanitize the DataFrames before passing to SHAP
        X_segment_clean = sanitize_dataframe_for_shap(X_segment)
        if isinstance(model, XGBRegressor):
            shap_values_all, base_vals = safe_xgb_shap(model, X_segment_clean)
            shap_values = shap_values_all[0]
        else:
            explainer = _get_or_create_tree_explainer(model)
            shap_values_all = explainer.shap_values(X_segment_clean)
            shap_values = shap_values_all[0] if isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim > 1 else shap_values_all
        fig, ax = plt.subplots(figsize=(10, 8))
        feature_names = X_segment_clean.columns
        shap_df = pd.DataFrame({'feature': feature_names, 'shap_value': shap_values}).sort_values('shap_value', key=abs, ascending=False).head(15)
        colors = ['red' if x > 0 else 'blue' for x in shap_df['shap_value']]
        bars = plt.barh(range(len(shap_df)), shap_df['shap_value'], color=colors, alpha=0.7)
        plt.yticks(range(len(shap_df)), shap_df['feature'])
        plt.xlabel('SHAP Value (Impact on Prediction)')
        plt.title(f'Feature Impact for Segment {segment_id}')
        plt.grid(axis='x', alpha=0.3)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2, f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=8)
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"[INFO] Segment SHAP explanation saved: {output_path}")
    except Exception as e:
        print(f"[WARNING] Failed to create segment SHAP for {segment_id}: {e}")
        if 'fig' in locals():
            plt.close(fig)