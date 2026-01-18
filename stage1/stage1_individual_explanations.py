"""
Stage 1 Individual Explanations Module
Generates individual SHAP explanations for segments and roads
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib import colors
from pathlib import Path
import shap
from typing import Dict, List, Optional, Tuple, Any
import warnings
import re
warnings.filterwarnings('ignore')
import traceback
import joblib

# Import configuration and utilities
import stage1_config as cfg
from stage1_interpretability import sanitize_dataframe_for_shap, _get_or_create_tree_explainer, safe_xgb_shap

ROAD_HOTSPOT_POPULATION_KEY = 'top_396_segments'
ROAD_HOTSPOT_POPULATION_LABEL = 'Top 396 Segments (per-road top-K)'
TOP_RISK_POPULATION_KEY = 'dataset_top_5pct'
TOP_RISK_POPULATION_LABEL = 'Top 5% Risk (per dataset)'
FULL_POPULATION_KEY = 'all_segments'
FULL_POPULATION_LABEL = 'All Segments (full population)'
# Guard xgboost import to avoid import-time failure in minimal test envs
try:
    import xgboost as _xgb
    XGBRegressor = getattr(_xgb, 'XGBRegressor', None)
except Exception:
    XGBRegressor = None


def _ensure_coords_present_on_df(df: pd.DataFrame, metadata_df: Optional[pd.DataFrame] = None, id_col_name: Optional[str] = None) -> pd.DataFrame:
    """Ensure lowercase latitude/longitude exist on df by filling from metadata if necessary.

    - Renames capitalized Latitude/Longitude to lowercase.
    - If metadata_df and id_col_name provided, merges and fills missing coords from metadata.
    """
    out = df.copy()
    # rename capitalized if present
    if 'Latitude' in out.columns and 'latitude' not in out.columns:
        out = out.rename(columns={'Latitude': 'latitude'})
    if 'Longitude' in out.columns and 'longitude' not in out.columns:
        out = out.rename(columns={'Longitude': 'longitude'})

    if metadata_df is not None and id_col_name is not None and 'segment_id' in out.columns:
        lat_col = 'Latitude'
        lon_col = 'Longitude'
        if id_col_name in metadata_df.columns and lat_col in metadata_df.columns and lon_col in metadata_df.columns:
            meta = metadata_df[[id_col_name, lat_col, lon_col]].copy()
            meta = meta.rename(columns={id_col_name: 'segment_id', lat_col: 'latitude', lon_col: 'longitude'})
            joined = out.merge(meta, on='segment_id', how='left', suffixes=('', '_meta'))

            # fill or create latitude/longitude from meta
            if 'latitude_meta' in joined.columns:
                if 'latitude' in joined.columns:
                    joined['latitude'] = joined['latitude'].fillna(joined['latitude_meta'])
                else:
                    joined = joined.rename(columns={'latitude_meta': 'latitude'})
            if 'longitude_meta' in joined.columns:
                if 'longitude' in joined.columns:
                    joined['longitude'] = joined['longitude'].fillna(joined['longitude_meta'])
                else:
                    joined = joined.rename(columns={'longitude_meta': 'longitude'})

            for c in ['latitude_meta', 'longitude_meta']:
                if c in joined.columns:
                    joined = joined.drop(columns=[c])

            out = joined

    return out


def sanitize_road_name_for_filename(road_name: str) -> str:
    """
    Sanitize road name to be safe for use in filenames.
    
    Args:
        road_name: Original road name
        
    Returns:
        Sanitized road name safe for filenames
    """
    if not cfg.USE_ENHANCED_NAMING or not cfg.SANITIZE_ROAD_NAMES:
        return str(road_name)
    
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', str(road_name))
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


def create_enhanced_filename(prefix: str, road_id: str, segment_id: str = None, extension: str = 'png') -> str:
    """
    Create enhanced filename with road and segment identifiers for traceability.
    
    Args:
        prefix: File prefix (e.g., 'segment_waterfall', 'road_summary')
        road_id: Road identifier
        segment_id: Segment identifier (optional, for segment-level files)
        extension: File extension
        
    Returns:
        Enhanced filename following the convention: prefix_road_{road_id}_segment_{segment_id}.ext
    """
    if not cfg.USE_ENHANCED_NAMING:
        # Fall back to legacy naming
        if segment_id:
            return f"{prefix}{segment_id}.{extension}"
        else:
            return f"{prefix}{road_id}.{extension}"
    
    sanitized_road = sanitize_road_name_for_filename(road_id)
    
    if segment_id:
        return f"{prefix}_road_{sanitized_road}_segment_{segment_id}.{extension}"
    else:
        return f"{prefix}_road_{sanitized_road}.{extension}"

def generate_segment_explanations(master_pred_df: pd.DataFrame,
                                 model: Any,
                                 X_features: pd.DataFrame,
                                 output_dir: Path,
                                 top_n: int = None,
                                 metadata_df: Optional[pd.DataFrame] = None,
                                 id_col_name: Optional[str] = None,
                                 preprocessor: Optional[Any] = None,
                                 fold_artifact_index: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate individual SHAP explanations for top N segments.
    
    Args:
        master_pred_df: DataFrame with segment predictions and metadata
        model: Trained model for SHAP analysis
        X_features: Feature matrix aligned with predictions
        output_dir: Directory to save explanation files
        top_n: Number of top segments to explain (default from config)
        
    Returns:
        Dictionary with explanation summary and file paths
    """
    if top_n is None:
        top_n = cfg.TOP_N_SEGMENTS_FOR_EXPLANATION
    
    print(f"[INFO] Generating individual SHAP explanations for top {top_n} segments...")
    
    # Create segment explanations directory
    segment_dir = output_dir / 'segment_explanations'
    segment_dir.mkdir(exist_ok=True)
    
    # Normalize fold and road columns
    road_colname = getattr(cfg, 'ROAD_COLUMN_NAME', 'road_id')
    fold_colname = getattr(cfg, 'FOLD_COLUMN_NAME', None)
    # Ensure road column exists
    if road_colname not in master_pred_df.columns:
        if 'road_id' in master_pred_df.columns:
            master_pred_df = master_pred_df.rename(columns={'road_id': road_colname})
        else:
            print(f"[WARN] Road column '{road_colname}' not found in master_pred_df; using available columns.")
    # Ensure fold column exists if configured
    if fold_colname and fold_colname not in master_pred_df.columns:
        if 'fold' in master_pred_df.columns:
            master_pred_df = master_pred_df.rename(columns={'fold': fold_colname})
        elif 'fold_number' in master_pred_df.columns:
            master_pred_df = master_pred_df.rename(columns={'fold_number': fold_colname})
        else:
            print(f"[WARN] Fold column '{fold_colname}' not found in master_pred_df; using available columns.")

    # Prefer per-road top-K selection if available (is_hotspot_per_road)
    if 'is_hotspot_per_road' in master_pred_df.columns:
        top_segments = master_pred_df[master_pred_df['is_hotspot_per_road'] == True].copy()
        print(f"[INFO] Selected {len(top_segments)} segments for explanation (per-road top-K)")
    else:
        top_segments = master_pred_df.nlargest(top_n, 'predicted_risk').copy()
        print(f"[INFO] Selected {len(top_segments)} segments for explanation (global top-N)")
    if len(top_segments) == 0:
        print("[WARN] No segments found for explanation")
        return {'summary': pd.DataFrame(), 'files_created': []}
    # Align features with selected segments. Honor explicit id_col_name if provided.
    segment_id_col = id_col_name or getattr(cfg, 'ID_COL', 'segment_id')
    segment_ids = top_segments[segment_id_col if segment_id_col in top_segments.columns else 'segment_id'].tolist()
    
    # Find matching indices in feature matrix (this requires careful alignment)
    # For BY_ROAD strategy, we need to map segment_ids back to feature indices
    explanation_data = []
    files_created = []
    skipped_segments = []

    # --- Load overlay/countermeasure data for join ---
    overlay_cm_path = None
    overlay_cm_df = None
    try:
        # Try to find overlay/countermeasure file in expected location (same output_dir/fold_results)
        overlay_dir = output_dir.parent / 'fold_results'
        for fname in [getattr(cfg, 'HOTSPOT_OVERLAY_CSV', 'hotspot_prediction_overlay.csv'), 'hotspot_prediction_overlay.csv']:
            candidate = overlay_dir / fname
            if candidate.exists():
                overlay_cm_path = candidate
                break
        if overlay_cm_path is not None:
            overlay_cm_df = pd.read_csv(overlay_cm_path)
        else:
            print(f"[WARN] Could not find overlay/countermeasure file for segment explanations join.")
    except Exception as e:
        print(f"[WARN] Could not load overlay/countermeasure file: {e}")

    try:
        # Sanitize features for SHAP
        X_clean = sanitize_dataframe_for_shap(X_features)

        # Prefer canonical index mapping (global_index) when present
        from stage1_utils import ensure_canonical_index
        try:
            # This will add cfg.CANONICAL_INDEX_COL if missing (best effort)
            master_feat = ensure_canonical_index(X_features)
        except Exception:
            master_feat = X_features.copy()

        id_col = id_col_name or getattr(cfg, 'ID_COL', 'segment_id')
        id_to_pos: Dict[str, int] = {}

        # If canonical index is present in processed features, map from it
        if getattr(cfg, 'CANONICAL_INDEX_COL', None) in master_feat.columns:
            try:
                series = master_feat[cfg.CANONICAL_INDEX_COL]
                id_to_pos = {str(v): i for i, v in enumerate(series.values)}
            except Exception:
                id_to_pos = {}

        # Fallback: try explicit id column in X_features
        if not id_to_pos:
            X_cols = set(X_features.columns)
            alt_names = [id_col, getattr(cfg, 'ID_COL', None), 'segment_id', 'Location ID', 'LocationID']
            found_id_col = None
            for name in [n for n in alt_names if n]:
                if name in X_cols:
                    found_id_col = name
                    break

            if found_id_col is not None:
                id_series = X_features[found_id_col].astype(str)
                id_to_pos = {str(v): i for i, v in enumerate(id_series.values)}
                if found_id_col != id_col:
                    print(f"[INFO] Using feature ID column '{found_id_col}' for mapping (requested '{id_col}').")
            else:
                # Case: metadata_df provided and contains the identifier; align by metadata index -> X_features.index
                if metadata_df is not None and id_col in metadata_df.columns:
                    try:
                        positions = X_features.index.get_indexer(metadata_df.index)
                        meta_ids = metadata_df[id_col].astype(str).values
                        for meta_id, pos in zip(meta_ids, positions):
                            if int(pos) != -1:
                                id_to_pos[str(meta_id)] = int(pos)
                    except Exception:
                        pass

        # If mapping still empty, attempt fuzzy alignment by latitude/longitude if present
        if not id_to_pos:
            lat_cols = [c for c in X_features.columns if c.lower() in ('latitude', 'lat')]
            lon_cols = [c for c in X_features.columns if c.lower() in ('longitude', 'lon', 'long')]
            master_lat_cols = [c for c in top_segments.columns if c.lower() in ('latitude', 'lat')]
            master_lon_cols = [c for c in top_segments.columns if c.lower() in ('longitude', 'lon', 'long')]
            if lat_cols and lon_cols and master_lat_cols and master_lon_cols:
                # perform rounding-based join to build mapping
                try:
                    xf = X_features.copy()
                    mf = top_segments[[getattr(cfg, 'ID_COL', 'segment_id')] + master_lat_cols + master_lon_cols].copy()
                    xf['_lat_r'] = xf[lat_cols[0]].astype(float).round(6)
                    xf['_lon_r'] = xf[lon_cols[0]].astype(float).round(6)
                    mf['_lat_r'] = mf[master_lat_cols[0]].astype(float).round(6)
                    mf['_lon_r'] = mf[master_lon_cols[0]].astype(float).round(6)
                    merged_pos = xf.reset_index().merge(mf, on=['_lat_r','_lon_r'], how='inner')
                    for _, r in merged_pos.iterrows():
                        segid = str(r.get(getattr(cfg, 'ID_COL', 'segment_id')))
                        pos = int(r['index'])
                        id_to_pos[segid] = pos
                    if id_to_pos:
                        print(f"[INFO] Built id->position map using latitude/longitude fuzzy match for {len(id_to_pos)} segments.")
                except Exception:
                    pass

        # If mapping still empty, provide an informative error and guidance (don't silently fall back)
        if not id_to_pos:
            msg = (
                f"[ERROR] Could not build id->position map for identifier '{id_col}'.\n"
                "Possible causes: your features DataFrame is missing the ID column, or IDs are stored under a different name (e.g., 'Location ID').\n"
                "Suggested actions: ensure `X_features` contains the segment ID column (same type and values as `master_pred_df`),\n"
                "or run the provided debug script `debug_prediction_alignment.py` to produce an aligned features file.\n"
                "Example: python debug_prediction_alignment.py --master_pred <master_csv> --X_features <features_csv> --id_col segment_id --do_align\n"
            )
            raise RuntimeError(msg)

        # --- STRICT PER-FOLD ARTIFACT CHECK ---
        # If fold_artifact_index is provided, require that all folds for the selected segments have valid artifacts
        missing_folds = []
        fold_dirs: Dict[Tuple[int, str], Path] = {}
        force_global_mode = False
        if fold_artifact_index is not None:
            try:
                idx_df = pd.read_csv(fold_artifact_index)
                # Normalize fold column in index
                idx_fold_col = fold_colname if fold_colname and fold_colname in idx_df.columns else (
                    'fold' if 'fold' in idx_df.columns else 'fold_number' if 'fold_number' in idx_df.columns else None)
                if idx_fold_col is None:
                    raise FileNotFoundError(f"[FAIL] Could not find fold column in fold_artifact_index: {fold_artifact_index}")

                # Possible columns that may hold directory/path info
                dir_cols_priority = ['fold_dir', 'fold_path', 'artifact_dir', 'artifact_root', 'dir', 'path']
                file_cols_fallback = ['model_path', 'preprocessor_path', 'xtrain_proc_path', 'X_train_proc_path']

                # Build a unique set of fold/model_type pairs needed for these top segments
                needed_pairs = set()
                for _, segment_row in top_segments.iterrows():
                    fold_num = segment_row.get(fold_colname, None) if fold_colname else (
                        segment_row.get('fold', None) or segment_row.get('fold_number', None))
                    model_type = str(segment_row.get('model_type', 'model'))
                    if fold_num is not None:
                        needed_pairs.add((int(fold_num), model_type))

                for fold_num, model_type in sorted(needed_pairs):
                    candidates = idx_df[(idx_df[idx_fold_col] == int(fold_num))]
                    if 'model_type' in idx_df.columns:
                        candidates = candidates[candidates['model_type'].astype(str) == model_type]

                    # Resolve a fold directory path from candidates
                    fold_dir_path: Optional[Path] = None
                    if len(candidates) > 0:
                        # If an explicit artifact_dir column is present, prefer it
                        if 'artifact_dir' in candidates.columns:
                            val = candidates.iloc[0]['artifact_dir']
                            if isinstance(val, str) and val:
                                fold_dir_path = Path(val)
                        # Try directory-like columns next
                        if fold_dir_path is None:
                            for c in dir_cols_priority:
                                if c in candidates.columns:
                                    val = candidates.iloc[0][c]
                                    if isinstance(val, str) and val:
                                        p = Path(val)
                                        if c == 'artifact_root':
                                            p = p / f"fold_{int(fold_num)}"
                                        fold_dir_path = p
                                        break
                        # Fallback: infer from file paths (take parent)
                        if fold_dir_path is None:
                            for c in file_cols_fallback:
                                if c in candidates.columns:
                                    val = candidates.iloc[0][c]
                                    if isinstance(val, str) and val:
                                        fold_dir_path = Path(val).parent
                                        break

                    # If still unresolved, try the conventional artifact layout
                    if fold_dir_path is None:
                        default_root = Path(getattr(cfg, 'OUTPUT_DIR', Path('.'))) / 'cv_artifacts' / str(model_type)
                        tentative = default_root / f"fold_{int(fold_num)}"
                        if tentative.exists():
                            fold_dir_path = tentative

                    if fold_dir_path is None or not Path(fold_dir_path).exists():
                        missing_folds.append((int(fold_num), model_type))
                    else:
                        fold_dirs[(int(fold_num), model_type)] = Path(fold_dir_path)

            except Exception as e:
                available_cols = []
                try:
                    available_cols = list(idx_df.columns)  # type: ignore
                except Exception:
                    pass
                raise FileNotFoundError(
                    f"[FAIL] Could not read/resolve fold_artifact_index '{fold_artifact_index}': {e}.\n"
                    f"Available columns: {available_cols}"
                )
            if missing_folds:
                # De-duplicate and log once
                miss_uniq = sorted(set(missing_folds))
                if getattr(cfg, 'ALLOW_GLOBAL_PROXY_EXPLANATIONS', False):
                    print(f"[WARN] Missing or unresolved per-fold directories for folds: {miss_uniq}.\n"
                          f"Proxy explanations enabled; will use per-fold where available and global fallback otherwise.")
                else:
                    # Stay in per-fold mode for available folds; skip segments from missing folds
                    print(f"[WARN] Missing per-fold artifacts or unresolved paths for folds: {miss_uniq}.")
                    print(f"[WARN] Proxy explanations disabled; will SKIP segments whose folds are unresolved.")

        for _, segment_row in top_segments.iterrows():
            # Prefer configured id column but fall back to 'segment_id'
            segment_id = segment_row.get(segment_id_col, None) or segment_row.get('segment_id')
            if segment_id is None:
                print("[WARN] Top segment row missing identifier column; skipping row")
                continue
            segment_id = str(segment_id)

            try:
                # Use id->position map to find the correct row in X_features
                if segment_id in id_to_pos:
                    target_pos = id_to_pos[segment_id]
                    feature_row = X_features.iloc[[int(target_pos)]]
                else:
                    # Last resort: try to find by identifier column in X_features if present
                    if id_col in X_features.columns:
                        matches = X_features[X_features[id_col].astype(str) == segment_id]
                        if len(matches) >= 1:
                            feature_row = matches.head(1)
                        else:
                            print(f"[WARN] Segment ID {segment_id} not found in features; skipping.")
                            continue
                    else:
                        print(f"[WARN] Segment ID {segment_id} not found in id->position map and no id column in features; skipping.")
                        continue

                # Ensure columns match sanitized X_clean
                feature_row = feature_row.reindex(columns=X_clean.columns)

                # Determine fold and try to load per-fold artifacts if available
                fold_num = None
                if fold_colname and fold_colname in segment_row.index:
                    fold_num = segment_row[fold_colname]
                elif 'fold' in segment_row.index:
                    fold_num = segment_row['fold']
                elif 'fold_number' in segment_row.index:
                    fold_num = segment_row['fold_number']

                # If per-fold is active and allowed, choose the mapped fold dir
                fold_dir = None
                if (fold_artifact_index is not None) and (not force_global_mode) and (fold_num is not None):
                    model_type = str(segment_row.get('model_type', 'model'))
                    fold_dir = fold_dirs.get((int(fold_num), model_type), None)
                    if fold_dir is None:
                        if getattr(cfg, 'ALLOW_GLOBAL_PROXY_EXPLANATIONS', False):
                            print(f"[WARN] No fold directory resolved for fold {fold_num}, model_type {model_type} in index. FALLING BACK to global model for segment {segment_id} (proxy explanation mode enabled).")
                        else:
                            print(f"[WARN] No fold directory resolved for fold {fold_num}, model_type {model_type} in index. Skipping segment {segment_id} (proxy explanation mode disabled).")
                            skipped_segments.append({
                                'segment_id': segment_id,
                                'fold': fold_num,
                                'model_source': 'none',
                                'reason': 'missing_per_fold_artifact',
                                'error': 'No fold directory found and proxy explanations disabled.'
                            })
                            continue
                # Fallback to default artifact layout if no index or proxy mode
                if (fold_dir is None) and (fold_num is not None):
                    artifact_root = Path(getattr(cfg, 'OUTPUT_DIR', Path('.'))) / 'cv_artifacts' / str(segment_row.get('model_type', 'model'))
                    fold_dir = artifact_root / f'fold_{int(fold_num)}'

                # ========================================================
                # EXPERT FIX: ROBUST MODEL AND PREPROCESSOR LOADING WITH FALLBACK
                # ========================================================
                preproc_to_use = preprocessor  # Start with global preprocessor as fallback
                model_to_use = model           # Start with global model as fallback
                background_for_shap = None
                using_global_fallback = False
                used_exact_fold_processed = False
                fold_live_pred_value = None

                if fold_dir is not None and fold_dir.exists():
                    # Attempt to load per-fold artifacts, but fallback gracefully
                    try:
                        preproc_path = fold_dir / 'preprocessor.joblib'
                        model_path = fold_dir / 'model.joblib'
                        
                        fold_model_loaded = False
                        fold_preprocessor_loaded = False
                        
                        # Try to load per-fold preprocessor
                        if preproc_path.exists():
                            try:
                                fold_preprocessor = joblib.load(preproc_path)
                                preproc_to_use = fold_preprocessor
                                fold_preprocessor_loaded = True
                            except Exception as e_preproc:
                                print(f"[WARN] Could not load preprocessor for fold {fold_num}: {e_preproc}")
                        
                        # Try to load per-fold model
                        if model_path.exists():
                            try:
                                fold_model = joblib.load(model_path)
                                model_to_use = fold_model
                                fold_model_loaded = True
                            except Exception as e_model:
                                print(f"[WARN] Could not load model for fold {fold_num}: {e_model}")
                        
                        # Load X_train_proc as SHAP background if available
                        xtrain_proc_path = fold_dir / 'X_train_proc.csv'
                        if xtrain_proc_path.exists():
                            try:
                                background_for_shap = pd.read_csv(xtrain_proc_path, index_col=0)
                            except Exception as e_bg:
                                print(f"[WARN] Could not load SHAP background for fold {fold_num}: {e_bg}")

                        # Prefer using the exact processed test row and its saved prediction to avoid any mismatch
                        try:
                            xtest_proc_path = fold_dir / 'X_test_proc.csv'
                            ypred_path = fold_dir / 'y_pred.csv'
                            if xtest_proc_path.exists():
                                df_test_proc = pd.read_csv(xtest_proc_path, index_col=0)
                                # Primary key: global canonical index from master_pred_df row
                                can_col = getattr(cfg, 'CANONICAL_INDEX_COL', 'global_index')
                                gid_val = segment_row.get(can_col, None)
                                found_key = None
                                if gid_val is not None:
                                    keys_to_try = [gid_val]
                                    try:
                                        keys_to_try.append(int(gid_val))
                                    except Exception:
                                        pass
                                    # Look up by index label (df_test_proc index is saved as the fold test index)
                                    for k in keys_to_try:
                                        if k in df_test_proc.index:
                                            found_key = k
                                            break
                                        if str(k) in df_test_proc.index:
                                            found_key = str(k)
                                            break
                                # Secondary: if fold saved a dedicated global_index column, match on that
                                if found_key is None and can_col in df_test_proc.columns:
                                    try:
                                        match_rows = df_test_proc[df_test_proc[can_col].astype(str) == str(gid_val)] if gid_val is not None else pd.DataFrame()
                                        if len(match_rows) > 0:
                                            feature_row_processed = match_rows.head(1)
                                            used_exact_fold_processed = True
                                            found_key = feature_row_processed.index[0]
                                    except Exception:
                                        pass
                                # Tertiary: match by segment_id column in processed file (if present)
                                if found_key is None and 'segment_id' in df_test_proc.columns:
                                    try:
                                        m2 = df_test_proc[df_test_proc['segment_id'].astype(str) == segment_id]
                                        if len(m2) > 0:
                                            feature_row_processed = m2.head(1)
                                            used_exact_fold_processed = True
                                            found_key = feature_row_processed.index[0]
                                    except Exception:
                                        pass

                                if found_key is not None and not used_exact_fold_processed:
                                    feature_row_processed = df_test_proc.loc[[found_key]]
                                    used_exact_fold_processed = True

                                # Load the corresponding fold y_pred by the same key
                                if used_exact_fold_processed and ypred_path.exists():
                                    try:
                                        s_yp = pd.read_csv(ypred_path, index_col=0).iloc[:, 0]
                                        if found_key in s_yp.index:
                                            fold_live_pred_value = float(s_yp.loc[found_key])
                                        elif str(found_key) in s_yp.index:
                                            fold_live_pred_value = float(s_yp.loc[str(found_key)])
                                    except Exception as e_yp:
                                        print(f"[WARN] Could not load fold y_pred for {segment_id}: {e_yp}")

                                # Optional last-resort positional alignment (disabled by default)
                                if (not used_exact_fold_processed) and getattr(cfg, 'ALLOW_POSITIONAL_ALIGNMENT', False):
                                    pos_try = id_to_pos.get(segment_id)
                                    try:
                                        pos_try = int(pos_try) if pos_try is not None else None
                                    except Exception:
                                        pos_try = None
                                    if pos_try is not None and pos_try >= 0 and pos_try < len(df_test_proc):
                                        try:
                                            feature_row_processed = df_test_proc.iloc[[pos_try]]
                                            used_exact_fold_processed = True
                                            if ypred_path.exists():
                                                try:
                                                    s_yp = pd.read_csv(ypred_path, index_col=0).iloc[:, 0]
                                                    if pos_try < len(s_yp):
                                                        fold_live_pred_value = float(s_yp.iloc[pos_try])
                                                        print(f"[INFO] Used positional alignment to extract y_pred for segment {segment_id} (pos {pos_try})")
                                                except Exception as e_yp2:
                                                    print(f"[WARN] Could not load fold y_pred by position for {segment_id}: {e_yp2}")
                                        except Exception:
                                            pass
                        except Exception as e_proc_row:
                            print(f"[WARN] Could not use exact processed test row for {segment_id}: {e_proc_row}")
                        
                        # Log what was successfully loaded
                        if fold_model_loaded and fold_preprocessor_loaded:
                            print(f"[INFO] Successfully loaded per-fold artifacts for segment {segment_id} (fold {fold_num})")
                        elif fold_model_loaded or fold_preprocessor_loaded:
                            print(f"[WARN] Partially loaded per-fold artifacts for segment {segment_id} (fold {fold_num})")
                        else:
                            using_global_fallback = True
                            print(f"[WARN] Could not load per-fold artifacts for segment {segment_id} (fold {fold_num}). FALLING BACK to global model.")
                            
                    except Exception as e_load:
                        using_global_fallback = True
                        print(f"[WARN] Could not load artifacts for fold {fold_num}: {e_load}. FALLING BACK to global model for segment {segment_id}.")
                else:
                    using_global_fallback = True
                    if fold_num is not None:
                        print(f"[INFO] Fold directory not found for fold {fold_num}, using global model for segment {segment_id}")

                # CRITICAL: Ensure model_to_use is never None
                if model_to_use is None:
                    if getattr(cfg, 'ALLOW_GLOBAL_PROXY_EXPLANATIONS', False):
                        print(f"[CRITICAL] model_to_use is None for segment {segment_id}, but proxy explanations enabled. Skipping segment.")
                        skipped_segments.append({
                            'segment_id': segment_id,
                            'fold': fold_num,
                            'model_source': 'none',
                            'reason': 'no_model_available',
                            'error': 'No model available for proxy explanation.'
                        })
                        continue
                    else:
                        raise RuntimeError(f"[CRITICAL ERROR] model_to_use is None for segment {segment_id}. This should never happen after expert fix.")

                if using_global_fallback:
                    print(f"[INFO] Using global model and preprocessor for segment {segment_id}")
                else:
                    print(f"[INFO] Using per-fold model for segment {segment_id} (fold {fold_num})")

                # === Transform the feature row with the selected preprocessor ===
                if not used_exact_fold_processed:
                    if preproc_to_use is not None:
                        try:
                            feature_row_processed = preproc_to_use.transform(feature_row)
                        except Exception as e_tf:
                            print(f"[WARN] Preprocessor transform failed for {segment_id}: {e_tf}; trying without transform")
                            feature_row_processed = feature_row
                    else:
                        feature_row_processed = feature_row

                # Validate model prediction vs stored predicted risk (tolerant in proxy/global mode)
                stored_risk = segment_row.get('predicted_risk')
                prediction_used_for_plot = None
                rel_diff = None
                enforce_match = getattr(cfg, 'ENFORCE_PREDICTION_MATCH', True)
                try:
                    # Prefer the saved fold prediction to ensure exact match, else compute
                    if fold_live_pred_value is not None:
                        pred_from_model = fold_live_pred_value
                    else:
                        pred_from_model = float(np.squeeze(model_to_use.predict(feature_row_processed)))
                    tol = getattr(cfg, 'EXPLANATION_PREDICTION_MISMATCH_TOLERANCE', 0.01)
                    live_pred = float(pred_from_model)
                    stored_val = float(stored_risk) if stored_risk is not None else None
                    model_source = 'per-fold' if not using_global_fallback else 'global'

                    if stored_val is None:
                        # No stored prediction: treat as mismatch if strict
                        if enforce_match:
                            print(f"[WARN] No stored predicted risk for segment {segment_id}; skipping due to strict match policy.")
                            skipped_segments.append({
                                'segment_id': segment_id,
                                'fold': fold_num,
                                'model_source': model_source,
                                'reason': 'no_stored_prediction'
                            })
                            continue
                        else:
                            prediction_used_for_plot = live_pred
                    else:
                        # Compute relative difference when stored_val sizable, otherwise absolute
                        if abs(stored_val) > 1e-6:
                            rel_diff = abs(live_pred - stored_val) / max(abs(stored_val), 1e-12)
                        else:
                            rel_diff = abs(live_pred - stored_val)

                        if enforce_match and (rel_diff is not None and rel_diff > tol):
                            msg = f"Prediction mismatch: stored={stored_val:.6f}, live={live_pred:.6f}, rel_diff={rel_diff:.6f}, tol={tol}"
                            print(f"[WARN] Prediction mismatch for segment {segment_id} (Fold: {fold_num}, Model: {model_source}): {msg}. Skipping explanation.")
                            skipped_segments.append({
                                'segment_id': segment_id,
                                'fold': fold_num,
                                'model_source': model_source,
                                'reason': 'prediction_mismatch',
                                'stored_predicted_risk': stored_val,
                                'live_predicted_risk': live_pred,
                                'rel_diff': rel_diff
                            })
                            continue
                        # Match acceptable or non-strict: prefer stored value for continuity
                        prediction_used_for_plot = stored_val
                except Exception as e_pred:
                    print(f"[WARN] Could not validate prediction for {segment_id}: {e_pred}")
                    # When validation fails, continue using model prediction to allow explanation
                    try:
                        prediction_used_for_plot = float(np.squeeze(model_to_use.predict(feature_row_processed)))
                    except Exception:
                        continue

                # Generate SHAP values for this segment
                shap_values = None
                base_value = 0.0
                try:
                    # Try specialized XGBoost path first (use processed features)
                    if isinstance(model_to_use, XGBRegressor) and 'safe_xgb_shap' in globals() and callable(safe_xgb_shap):
                        shap_values, base_value = safe_xgb_shap(model_to_use, feature_row_processed)
                    else:
                        # === Use the PROCESSED data and fold background for the SHAP explainer ===
                        if background_for_shap is not None:
                            expl_background = background_for_shap
                        elif preproc_to_use is not None:
                            try:
                                expl_background = pd.DataFrame(preproc_to_use.transform(X_clean), columns=X_clean.columns)
                            except Exception:
                                expl_background = X_clean
                        else:
                            expl_background = X_clean

                        expl = shap.Explainer(model_to_use, expl_background)
                        sv = expl(feature_row_processed)
                        # sv.values shape: (n_rows, n_features) or (n_rows, n_output, n_features)
                        if hasattr(sv, 'values'):
                            arr = sv.values
                            shap_values = np.asarray(arr[0]).reshape(-1)
                        else:
                            shap_values = np.asarray(sv).reshape(-1)
                        if hasattr(sv, 'base_values'):
                            base_value = float(sv.base_values[0]) if np.ndim(sv.base_values) > 0 else float(sv.base_values)
                except Exception as e_shap:
                    print(f"[WARN] SHAP computation failed for {segment_id}: {e_shap}")
                    shap_values = np.zeros(len(X_clean.columns))

                # Create feature importance dataframe
                feature_importance = pd.DataFrame({
                    'feature': X_clean.columns,
                    'shap_value': shap_values,
                    'abs_shap_value': np.abs(shap_values)
                }).sort_values('abs_shap_value', ascending=False)

                # Extract top-N SHAP features (name, value, sign)
                N = getattr(cfg, 'TOP_N_SHAP_FEATURES_PER_SEGMENT', 5)
                topN = feature_importance.head(N)
                top_features = []
                for _, row in topN.iterrows():
                    top_features.append({
                        'feature': row['feature'],
                        'shap_value': row['shap_value'],
                        'sign': 'positive' if row['shap_value'] >= 0 else 'negative'
                    })

                # Join countermeasures for this segment (all unique countermeasures for this segment_id)
                countermeasures = []
                if overlay_cm_df is not None:
                    cm_rows = overlay_cm_df[overlay_cm_df['segment_id'].astype(str) == segment_id]
                    if 'countermeasure' in cm_rows.columns:
                        countermeasures = list(cm_rows['countermeasure'].dropna().unique())

                # Save individual explanation plot with enhanced naming
                road_id = segment_row.get(road_colname, segment_row.get('road_id', 'Unknown'))
                filename = create_enhanced_filename(
                    prefix=cfg.SEGMENT_EXPLANATION_PREFIX.rstrip('_'),
                    road_id=road_id,
                    segment_id=segment_id,
                    extension='png'
                )
                plot_path = segment_dir / filename
                _create_segment_waterfall_plot(
                    segment_id=segment_id,
                    feature_importance=feature_importance,
                    predicted_risk=prediction_used_for_plot if prediction_used_for_plot is not None else segment_row['predicted_risk'],
                    actual_risk=segment_row['actual_risk'],
                    base_value=base_value,
                    output_path=plot_path
                )
                files_created.append(str(plot_path))

                # Store explanation data: segment_id, top-N SHAP features, countermeasures
                explanation_data.append({
                    'segment_id': segment_id,
                    'road_id': road_id,
                    'predicted_risk': prediction_used_for_plot if prediction_used_for_plot is not None else segment_row['predicted_risk'],
                    'actual_risk': segment_row['actual_risk'],
                    **{f'top_feature_{i+1}_name': f['feature'] for i, f in enumerate(top_features)},
                    **{f'top_feature_{i+1}_shap': f['shap_value'] for i, f in enumerate(top_features)},
                    **{f'top_feature_{i+1}_sign': f['sign'] for i, f in enumerate(top_features)},
                    'countermeasures': "; ".join(countermeasures) if countermeasures else None,
                    'explanation_file': plot_path.name
                })
            except Exception as e:
                print(f"[WARN] Failed to generate explanation for segment {segment_id}: {e}")
                skipped_segments.append({
                    'segment_id': segment_id,
                    'fold': fold_num if 'fold_num' in locals() else None,
                    'model_source': 'unknown',
                    'reason': 'exception_during_explanation',
                    'error': str(e)
                })
                continue

    except Exception as e:
        print(f"[ERROR] Segment explanation generation failed: {e}")
        traceback.print_exc()
        return {'summary': pd.DataFrame(), 'files_created': []}

    # Create summary dataframe
    summary_df = pd.DataFrame(explanation_data)

    # Ensure the configured road column exists in the segment summary so downstream
    # road-level aggregation can group by the expected name (e.g. 'Road name').
    road_colname = getattr(cfg, 'ROAD_COLUMN_NAME', 'road_id')
    try:
        if not summary_df.empty and road_colname not in summary_df.columns:
            if 'road_id' in summary_df.columns:
                summary_df[road_colname] = summary_df['road_id']
            else:
                # Try to map from master_pred_df if it's available in locals()
                if 'master_pred_df' in locals() and getattr(master_pred_df, 'shape', (0,))[0] > 0:
                    id_col = getattr(cfg, 'ID_COL', 'Location ID')
                    if id_col in master_pred_df.columns and road_colname in master_pred_df.columns:
                        map_df = master_pred_df[[id_col, road_colname]].drop_duplicates()
                        summary_df = summary_df.merge(map_df, left_on='segment_id', right_on=id_col, how='left')
    except Exception as e_mapcol:
        print(f"[WARN] Could not ensure road column '{road_colname}' in segment summary: {e_mapcol}")

    # Save skipped segments log if any
    if skipped_segments:
        try:
            skipped_df = pd.DataFrame(skipped_segments)
            skipped_path = segment_dir / 'skipped_segments.csv'
            skipped_df.to_csv(skipped_path, index=False)
            files_created.append(str(skipped_path))
            print(f"[INFO] Saved skipped segments log: {skipped_path}")
        except Exception as e:
            print(f"[WARN] Could not save skipped segments log: {e}")

    if not summary_df.empty:
        # Save summary CSV
        summary_path = segment_dir / cfg.SEGMENT_EXPLANATION_SUMMARY_CSV
        summary_df.to_csv(summary_path, index=False)
        files_created.append(str(summary_path))

        print(f"[INFO] Generated {len(summary_df)} segment explanations")
        print(f"[INFO] Summary saved to: {summary_path}")

    return {
        'summary': summary_df,
        'files_created': files_created,
        'output_directory': str(segment_dir)
    }

def generate_road_explanations(master_pred_df: pd.DataFrame,
                              segment_explanations: Dict[str, Any],
                              output_dir: Path,
                              top_n: int = None) -> Dict[str, Any]:
    """
    Generate road-level aggregated SHAP explanations.
    
    Args:
        master_pred_df: DataFrame with segment predictions and metadata
        segment_explanations: Output from generate_segment_explanations
        output_dir: Directory to save explanation files
        top_n: Number of top roads to explain (default from config)
        
    Returns:
        Dictionary with road explanation summary and file paths
    """
    if top_n is None:
        top_n = cfg.TOP_N_ROADS_FOR_EXPLANATION
    
    print(f"[INFO] Generating road-level aggregated explanations for top {top_n} roads...")
    
    # Create road explanations directory
    road_dir = output_dir / 'road_explanations'
    road_dir.mkdir(exist_ok=True)
    
    # Aggregate segment data by road (use configured road column name)
    road_colname = getattr(cfg, 'ROAD_COLUMN_NAME', 'road_id')
    # Ensure column exists
    if road_colname not in master_pred_df.columns:
        if 'road_id' in master_pred_df.columns:
            master_pred_df = master_pred_df.rename(columns={'road_id': road_colname})
        else:
            print(f"[WARN] Road column '{road_colname}' not found in master_pred_df; available columns: {list(master_pred_df.columns)}")
            return {'summary': pd.DataFrame(), 'files_created': []}
    road_summary = master_pred_df.groupby(road_colname).agg({
        'predicted_risk': ['mean', 'max', 'count'],
        'actual_risk': ['mean', 'max'],
        'segment_id': 'first'  # Keep one segment_id for reference
    }).round(4)

    # Flatten column names
    road_summary.columns = ['_'.join(col).strip() for col in road_summary.columns]
    road_summary = road_summary.reset_index()

    # Select top roads by mean predicted risk
    if 'predicted_risk_mean' not in road_summary.columns:
        print(f"[WARN] 'predicted_risk_mean' not found in road_summary; available columns: {list(road_summary.columns)}")
        return {'summary': pd.DataFrame(), 'files_created': []}
    top_roads = road_summary.nlargest(top_n, 'predicted_risk_mean')
    
    if len(top_roads) == 0:
        print("[WARN] No roads found for explanation")
        return {'summary': pd.DataFrame(), 'files_created': []}
    
    # Create road explanations based on constituent segments
    segment_summary = segment_explanations.get('summary', pd.DataFrame())
    road_explanation_data = []
    files_created = []
    
    try:
        # EXPERT FIX: Handle empty segment summary to prevent IndexError
        if segment_summary.empty:
            print("[WARN] No segment explanations available for road-level aggregation")
            print("[INFO] Creating summary-only road explanations from prediction data")
            
            for _, road_row in top_roads.iterrows():
                road_id = road_row[road_colname]
                road_explanation_data.append({
                    'road_id': road_id,
                    'mean_predicted_risk': road_row['predicted_risk_mean'],
                    'max_predicted_risk': road_row['predicted_risk_max'],
                    'mean_actual_risk': road_row['actual_risk_mean'],
                    'segment_count': road_row['predicted_risk_count'],
                    'segments_explained': 0,
                    'top_feature': 'N/A',
                    'avg_top_feature_impact': 0.0,
                    'explanation_method': 'summary_only',
                    'explanation_file': 'N/A'
                })
        else:
            for _, road_row in top_roads.iterrows():
                road_id = road_row[road_colname]

                # Defaults to ensure variables always exist
                avg_impact = 0.0
                most_common_feature = 'N/A'
                road_segments = pd.DataFrame()

                # EXPERT FIX: Safe access to road segments with proper fallback
                try:
                    if road_colname in segment_summary.columns:
                        road_segments = segment_summary[segment_summary[road_colname] == road_id]
                    else:
                        print(f"[WARN] Road column '{road_colname}' not found in segment summary")
                        road_segments = pd.DataFrame()
                except Exception as e_road_match:
                    print(f"[WARN] Error matching road segments for {road_id}: {e_road_match}")
                    road_segments = pd.DataFrame()

                explanation_method = 'aggregated_segments' if len(road_segments) > 0 else 'summary_only'
                segments_explained = len(road_segments)

                if len(road_segments) > 0:
                    # Aggregate segment explanations for this road
                    if 'top_feature_impact' in road_segments.columns:
                        avg_impact = float(road_segments['top_feature_impact'].mean())
                    if 'top_feature' in road_segments.columns and len(road_segments) > 0:
                        try:
                            most_common_feature = road_segments['top_feature'].mode().iloc[0]
                        except Exception:
                            most_common_feature = 'Unknown'

                # Create road-level visualization with enhanced naming
                filename = create_enhanced_filename(
                    prefix=cfg.ROAD_EXPLANATION_PREFIX.rstrip('_'),  # Remove trailing underscore
                    road_id=road_id,
                    extension='png'
                )
                plot_path = road_dir / filename
                _create_road_summary_plot(
                    road_id=road_id,
                    road_segments=road_segments,
                    road_summary=road_row,
                    output_path=plot_path
                )
                files_created.append(str(plot_path))

                # Generate road validation map if enabled
                if cfg.GENERATE_ROAD_VALIDATION_MAPS:
                    try:
                        # Import here to avoid circular imports
                        from stage1_visualizations import generate_road_comparison_map
                        

                        # Get all segments for this road from master predictions
                        if road_colname in master_pred_df.columns:
                            road_segments_full = master_pred_df[master_pred_df[road_colname] == road_id].copy()
                        elif 'road_id' in master_pred_df.columns:
                            road_segments_full = master_pred_df[master_pred_df['road_id'] == road_id].copy()
                        else:
                            print(f"[WARN] Could not find road column '{road_colname}' or 'road_id' in master_pred_df for validation map generation of road {road_id}.")
                            road_segments_full = pd.DataFrame()

                        # Normalize/fill latitude/longitude using helper (safe if metadata not present)
                        try:
                            import sys
                            main_mod = sys.modules.get('__main__')
                            metadata_df = getattr(main_mod, 'metadata_df') if (main_mod and hasattr(main_mod, 'metadata_df')) else None
                            id_col = getattr(cfg, 'ID_COL', 'Location ID')
                            road_segments_full = _ensure_coords_present_on_df(road_segments_full, metadata_df=metadata_df, id_col_name=id_col)
                        except Exception as e:
                            print(f"[WARN] Could not ensure coords for road {road_id}: {e}")

                        if len(road_segments_full) > 0:
                            # Create validation maps directory
                            validation_maps_dir = output_dir / cfg.ROAD_VALIDATION_MAPS_DIR
                            validation_maps_dir.mkdir(exist_ok=True)

                            # Generate the validation map
                            map_filepath = generate_road_comparison_map(
                                road_df=road_segments_full,
                                road_id=road_id,
                                output_dir=validation_maps_dir,
                                top_k=cfg.VALIDATION_MAP_TOP_K or cfg.HOTSPOT_K
                            )

                            if map_filepath:
                                files_created.append(map_filepath)
                                print(f"[INFO] Road validation map created: {map_filepath}")
                        else:
                            print(f"[WARN] No segments found for road {road_id} validation map")
                            
                    except Exception as e_map:
                        print(f"[WARN] Failed to create validation map for road {road_id}: {e_map}")
                        # Continue processing other roads
                
                road_explanation_data.append({
                    'road_id': road_id,
                    'mean_predicted_risk': road_row['predicted_risk_mean'],
                    'max_predicted_risk': road_row['predicted_risk_max'],
                    'mean_actual_risk': road_row['actual_risk_mean'],
                    'segment_count': road_row['predicted_risk_count'],
                    'segments_explained': segments_explained,
                    'top_feature': most_common_feature,
                    'avg_top_feature_impact': avg_impact,
                    'explanation_method': explanation_method,
                    'explanation_file': plot_path.name if explanation_method == 'aggregated_segments' else 'N/A'
                })
    
    except Exception as e:
        print(f"[ERROR] Road explanation generation failed: {e}")
        traceback.print_exc()
        return {'summary': pd.DataFrame(), 'files_created': []}
    
    # Create summary dataframe
    road_summary_df = pd.DataFrame(road_explanation_data)
    
    if not road_summary_df.empty:
        # Save summary CSV
        summary_path = road_dir / cfg.ROAD_EXPLANATION_SUMMARY_CSV
        road_summary_df.to_csv(summary_path, index=False)
        files_created.append(str(summary_path))
        
        print(f"[INFO] Generated {len(road_summary_df)} road explanations")
        print(f"[INFO] Summary saved to: {summary_path}")
    
    return {
        'summary': road_summary_df,
        'files_created': files_created,
        'output_directory': str(road_dir)
    }

def _create_segment_waterfall_plot(segment_id: str,
                                  feature_importance: pd.DataFrame,
                                  predicted_risk: float,
                                  actual_risk: float,
                                  base_value: float,
                                  output_path: Path):
    """Create SHAP waterfall plot for individual segment."""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Select top 15 features for clarity
        top_features = feature_importance.head(15)
        
        # Create horizontal bar plot
        colors = ['red' if x > 0 else 'blue' for x in top_features['shap_value']]
        bars = ax.barh(range(len(top_features)), top_features['shap_value'], 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Calculate x-axis limits to make room for labels
        max_abs_val = max(abs(top_features['shap_value'].min()), abs(top_features['shap_value'].max()))
        x_margin = max_abs_val * 0.25  # 25% margin for labels
        ax.set_xlim(-max_abs_val - x_margin, max_abs_val + x_margin)
        
        # Customize plot
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=11)
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=13, fontweight='bold')
        ax.set_title(f'Feature Impact Explanation - Segment {segment_id}\n'
                    f'Predicted Target: {predicted_risk:.4f} | Actual Target: {actual_risk:.4f}', 
                    fontsize=14, fontweight='bold')
        
        # Add value labels OUTSIDE bars on the right side
        for i, (bar, value) in enumerate(zip(bars, top_features['shap_value'])):
            width = bar.get_width()
            # Place all labels on the right side (at the bar tip + offset)
            if abs(width) > 0.001:  # Only add label if bar is visible
                x_offset = max_abs_val * 0.02  # 2% of max value as offset
                # Always place at the rightmost point (bar tip)
                label_x = width + x_offset if width >= 0 else width - x_offset
                ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                       ha='left' if width >= 0 else 'right', va='center', 
                       fontsize=10, color='black', fontweight='bold')
        
        # Add baseline and prediction lines
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(axis='x', alpha=0.3)
        
        # Add explanation text (legend in top right)
        total_impact = top_features['shap_value'].sum()
        ax.text(0.98, 0.98, f'Base prediction: {base_value:.3f}\nTotal impact: {total_impact:.3f}', 
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        plt.tight_layout(pad=2.0)
        fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)
        
    except Exception as e:
        print(f"[WARN] Failed to create segment plot for {segment_id}: {e}")
        if 'fig' in locals():
            plt.close(fig)

def _create_road_summary_plot(road_id: str,
                             road_segments: pd.DataFrame,
                             road_summary: pd.Series,
                             output_path: Path):
    """Create road-level SHAP aggregation plot showing average feature impacts."""
    try:
        # Extract all SHAP values for this road's segments
        shap_columns = [col for col in road_segments.columns if col.startswith('top_feature_') and col.endswith('_shap')]
        name_columns = [col for col in road_segments.columns if col.startswith('top_feature_') and col.endswith('_name')]
        
        if not shap_columns or not name_columns or len(road_segments) == 0:
            # Fallback: create simple summary plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            summary_text = f"""Road {road_id} Summary:
            Segments: {road_summary.get('predicted_risk_count', 'N/A')}
            Avg Predicted Target: {road_summary.get('predicted_risk_mean', 0):.4f}
            Max Predicted Target: {road_summary.get('predicted_risk_max', 0):.4f}
            Avg Actual Target: {road_summary.get('actual_risk_mean', 0):.4f}
            
            No SHAP explanations available for this road."""
            ax.text(0.5, 0.5, summary_text, ha='center', va='center', 
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax.set_title(f'Road {road_id} Summary', fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return
        
        # Aggregate SHAP values across all segments for this road
        feature_impacts = {}
        for i, (name_col, shap_col) in enumerate(zip(name_columns, shap_columns)):
            for _, segment in road_segments.iterrows():
                feature_name = segment.get(name_col)
                shap_value = segment.get(shap_col)
                if pd.notna(feature_name) and pd.notna(shap_value):
                    if feature_name not in feature_impacts:
                        feature_impacts[feature_name] = []
                    feature_impacts[feature_name].append(float(shap_value))
        
        if not feature_impacts:
            print(f"[WARN] No valid SHAP features found for road {road_id}")
            return
        
        # Calculate average absolute SHAP impact per feature
        avg_impacts = {feat: np.mean(np.abs(vals)) for feat, vals in feature_impacts.items()}
        avg_signed_impacts = {feat: np.mean(vals) for feat, vals in feature_impacts.items()}
        
        # Sort by absolute impact
        sorted_features = sorted(avg_impacts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        feature_names = [f[0] for f in sorted_features]
        avg_values = [avg_signed_impacts[f[0]] for f in sorted_features]
        colors = ['red' if x > 0 else 'blue' for x in avg_values]
        
        bars = ax.barh(range(len(feature_names)), avg_values, 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Calculate x-axis limits to make room for labels
        max_abs_val = max(abs(min(avg_values)), abs(max(avg_values)))
        x_margin = max_abs_val * 0.25  # 25% margin for labels
        ax.set_xlim(-max_abs_val - x_margin, max_abs_val + x_margin)
        
        # Add value labels outside bars on the right side
        for i, (bar, value) in enumerate(zip(bars, avg_values)):
            width = bar.get_width()
            if abs(width) > 0.001:
                x_offset = max_abs_val * 0.02  # 2% of max value as offset
                # Always place at the rightmost point (bar tip)
                label_x = width + x_offset if width >= 0 else width - x_offset
                ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                       ha='left' if width >= 0 else 'right', va='center', 
                       fontsize=10, color='black', fontweight='bold')
        
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=11)
        ax.set_xlabel('Average SHAP Value (Impact on Prediction)', fontsize=13, fontweight='bold')
        ax.set_title(f'Road-Level Feature Impact Analysis - {road_id}\n'
                    f'Aggregated across {len(road_segments)} segments', 
                    fontsize=14, fontweight='bold')
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(axis='x', alpha=0.3)
        
        # Add summary statistics in top right
        summary_text = f"""Road Summary:
Segments: {len(road_segments)}
Avg Predicted: {road_summary.get('predicted_risk_mean', 0):.4f}
Avg Actual: {road_summary.get('actual_risk_mean', 0):.4f}"""
        
        ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        plt.tight_layout(pad=2.0)
        fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)
        
    except Exception as e:
        print(f"[WARN] Failed to create road plot for {road_id}: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals():
            plt.close(fig)

def _build_dataset_shap_longframe(feature_impacts: Dict[str, List[float]]) -> pd.DataFrame:
    """Convert dict(feature -> shap values) into a long-form DataFrame."""
    rows: List[Dict[str, float]] = []
    for feature_name, values in feature_impacts.items():
        if values is None:
            continue
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[~np.isnan(arr)]
        for value in arr:
            rows.append({'feature': feature_name, 'shap_value': float(value), 'abs_shap_value': abs(float(value))})
    return pd.DataFrame(rows)


def _create_dataset_violin_plots(dataset_feature_impacts: Dict[str, Dict[str, list]],
                                 segment_summary_with_dataset: pd.DataFrame,
                                 dataset_col: str,
                                 output_dir: Path) -> list:
    """Create density-style violin + strip plots that handle unequal SHAP lengths."""
    try:
        files_created: List[str] = []
        all_long_rows: List[pd.DataFrame] = []
        all_datasets = sorted(dataset_feature_impacts.keys())
        top_k = getattr(cfg, 'DATASET_VIOLIN_TOP_FEATURES', 15)

        print(f"[INFO] Creating density-style SHAP plots for {len(all_datasets)} datasets")

        for dataset_id in all_datasets:
            feature_impacts = dataset_feature_impacts[dataset_id]
            if not feature_impacts:
                print(f"[WARN] No SHAP data for dataset {dataset_id}, skipping")
                continue

            long_df = _build_dataset_shap_longframe(feature_impacts)
            if long_df.empty:
                print(f"[WARN] Dataset {dataset_id} produced 0 valid SHAP rows")
                continue

            long_df['dataset'] = str(dataset_id)
            all_long_rows.append(long_df)

            feature_order = (
                long_df.groupby('feature')['abs_shap_value']
                .mean()
                .sort_values(ascending=False)
                .head(top_k)
                .index.tolist()
            )

            if not feature_order:
                continue

            filtered_df = long_df[long_df['feature'].isin(feature_order)].copy()
            ordered = list(reversed(feature_order))
            filtered_df['feature'] = pd.Categorical(filtered_df['feature'], categories=ordered, ordered=True)
            filtered_df['polarity'] = np.where(filtered_df['shap_value'] >= 0, 'Risk ↑', 'Risk ↓')

            fig_height = max(6.0, len(ordered) * 0.55 + 3.0)
            fig, ax = plt.subplots(figsize=(18, fig_height))

            try:
                sns.violinplot(
                    data=filtered_df,
                    x='shap_value',
                    y='feature',
                    inner=None,
                    cut=0,
                    scale='width',
                    linewidth=0.8,
                    color='#d8e3f2',
                    ax=ax
                )
            except Exception as violin_exc:
                print(f"[WARN] Violin computation issue for dataset {dataset_id}: {violin_exc}")

            sns.stripplot(
                data=filtered_df,
                x='shap_value',
                y='feature',
                hue='polarity',
                dodge=False,
                jitter=0.15,
                size=3,
                alpha=0.75,
                palette={'Risk ↑': '#d62728', 'Risk ↓': '#1f77b4'},
                ax=ax
            )

            max_abs = filtered_df['shap_value'].abs().max()
            max_abs = max(max_abs, 1e-4)
            ax.set_xlim(-max_abs * 1.2, max_abs * 1.2)
            ax.axvline(0, color='#444444', linestyle='--', linewidth=1.0, alpha=0.7)
            ax.set_xlabel('SHAP Value (impact on risk prediction)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Feature', fontsize=13, fontweight='bold')
            ax.set_title(
                f'Dataset {dataset_id} - SHAP Density (Top {len(feature_order)} features)',
                fontsize=15,
                fontweight='bold',
                pad=14
            )
            ax.grid(True, axis='x', linestyle='--', alpha=0.2)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='upper right', framealpha=0.9, title='Direction')

            stats_text = (
                f"Segments: {filtered_df.shape[0]}\n"
                f"Features shown: {len(feature_order)}\n"
                f"Median |SHAP|: {filtered_df['abs_shap_value'].median():.4f}"
            )
            ax.text(
                0.02,
                0.02,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                va='bottom',
                ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

            plt.tight_layout(pad=1.4)
            plot_path = output_dir / f'dataset_{dataset_id}_shap_summary.png'
            fig.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            files_created.append(str(plot_path))
            print(f"[INFO] Saved density plot: {plot_path.name}")

        if len(all_long_rows) > 1:
            combined_df = pd.concat(all_long_rows, ignore_index=True)
            top_features_global = (
                combined_df.groupby('feature')['abs_shap_value']
                .mean()
                .sort_values(ascending=False)
                .head(top_k)
                .index.tolist()
            )

            if top_features_global:
                combined_filtered = combined_df[combined_df['feature'].isin(top_features_global)].copy()
                ordered_global = list(reversed(top_features_global))
                combined_filtered['feature'] = pd.Categorical(
                    combined_filtered['feature'], categories=ordered_global, ordered=True
                )
                combined_filtered['polarity'] = np.where(
                    combined_filtered['shap_value'] >= 0, 'Risk ↑', 'Risk ↓'
                )

                fig_height = max(6.0, len(ordered_global) * 0.5 + 3.0)
                fig, ax = plt.subplots(figsize=(18, fig_height))
                sns.violinplot(
                    data=combined_filtered,
                    x='shap_value',
                    y='feature',
                    inner=None,
                    cut=0,
                    scale='width',
                    linewidth=0.8,
                    color='#e4eaf5',
                    ax=ax
                )
                sns.stripplot(
                    data=combined_filtered,
                    x='shap_value',
                    y='feature',
                    hue='polarity',
                    dodge=False,
                    jitter=0.12,
                    size=2.5,
                    alpha=0.65,
                    palette={'Risk ↑': '#d62728', 'Risk ↓': '#1f77b4'},
                    ax=ax
                )
                ax.set_title('SHAP Density - All Datasets Combined', fontsize=15, fontweight='bold', pad=14)
                ax.set_xlabel('SHAP Value (impact on risk prediction)', fontsize=13, fontweight='bold')
                ax.set_ylabel('Feature', fontsize=13, fontweight='bold')
                ax.axvline(0, color='#444444', linestyle='--', linewidth=1.0, alpha=0.7)
                ax.grid(True, axis='x', linestyle='--', alpha=0.2)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, loc='upper right', framealpha=0.9, title='Direction')
                plt.tight_layout(pad=1.2)
                violin_path = output_dir / 'dataset_shap_violin_top10features.png'
                fig.savefig(violin_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                files_created.append(str(violin_path))
                print(f"[INFO] Saved combined density plot: {violin_path.name}")

        print("[INFO] Creating statistical summary of SHAP values...")
        stats_summary = []
        for dataset_id, feature_impacts in dataset_feature_impacts.items():
            for feature_name, shap_values in feature_impacts.items():
                shap_array = np.array(shap_values)
                if shap_array.size == 0:
                    continue
                stats_summary.append({
                    'Feature': feature_name,
                    'Dataset': str(dataset_id),
                    'Mean_SHAP': shap_array.mean(),
                    'Mean_Abs_SHAP': np.abs(shap_array).mean(),
                    'Median_SHAP': np.median(shap_array),
                    'Std_SHAP': shap_array.std(),
                    'Q25': np.percentile(shap_array, 25),
                    'Q75': np.percentile(shap_array, 75),
                    'IQR': np.percentile(shap_array, 75) - np.percentile(shap_array, 25),
                    'N_Segments': len(shap_array)
                })

        if stats_summary:
            stats_df = pd.DataFrame(stats_summary).sort_values('Mean_Abs_SHAP', ascending=False)
            stats_path = output_dir / 'dataset_shap_statistical_summary.csv'
            stats_df.to_csv(stats_path, index=False)
            files_created.append(str(stats_path))
            print(f"[INFO] Statistical summary saved: {stats_path.name}")

            print("[INFO] Creating dataset-feature heatmap...")
            top_features_global = stats_df.groupby('Feature')['Mean_Abs_SHAP'].mean().sort_values(ascending=False).head(15)
            top_feature_names_heatmap = top_features_global.index.tolist()
            heatmap_data = []
            for feature_name in top_feature_names_heatmap:
                feature_row = {'Feature': feature_name}
                feature_stats = stats_df[stats_df['Feature'] == feature_name]
                for dataset_id in sorted(feature_stats['Dataset'].unique()):
                    dataset_stat = feature_stats[feature_stats['Dataset'] == dataset_id]
                    feature_row[f'Dataset_{dataset_id}'] = dataset_stat['Mean_SHAP'].values[0] if not dataset_stat.empty else 0
                heatmap_data.append(feature_row)

            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data).set_index('Feature')
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                           cbar_kws={'label': 'Mean SHAP Value'}, ax=ax,
                           linewidths=0.5, linecolor='gray')
                ax.set_title('Feature Importance Heatmap Across Datasets\n(Mean SHAP Values - Top 15 Features)',
                            fontsize=14, fontweight='bold')
                ax.set_xlabel('Dataset ID', fontsize=12)
                ax.set_ylabel('Feature', fontsize=12)
                plt.tight_layout()
                heatmap_path = output_dir / 'dataset_feature_heatmap.png'
                fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                files_created.append(str(heatmap_path))
                print(f"[INFO] Heatmap saved: {heatmap_path.name}")

        return files_created

    except Exception as e:
        print(f"[WARN] Failed to create violin plots: {e}")
        traceback.print_exc()
        return []


def _extract_dataset_id_from_filename(stem: str) -> str:
    match = re.search(r'dataset_(.+?)(?:_|$)', stem)
    return match.group(1) if match else stem


def _prepare_regional_population_entries(per_dataset_dir: Path,
                                         full_population_dir: Optional[Path] = None,
                                         road_topk_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    def _register_population(key: str, label: str, directory: Optional[Path], file_patterns: List[str]) -> None:
        if directory is None:
            return
        if not directory.exists():
            print(f"[WARN] Population '{label}' skipped - directory missing: {directory}")
            return
        csv_files: List[Path] = []
        for pattern in file_patterns:
            csv_files.extend(directory.glob(pattern))
        if not csv_files:
            print(f"[WARN] Population '{label}' skipped - no files matching {file_patterns}")
            return

        dataset_shap_data: Dict[str, pd.DataFrame] = {}
        for csv_file in csv_files:
            dataset_id = _extract_dataset_id_from_filename(csv_file.stem)
            try:
                df = pd.read_csv(csv_file)
                dataset_shap_data[str(dataset_id)] = df
            except Exception as read_err:
                print(f"[WARN] Failed to load {csv_file.name}: {read_err}")

        if dataset_shap_data:
            entries.append({
                'key': key,
                'label': label,
                'dataset_shap_data': dataset_shap_data,
                'source_dir': directory
            })

    base_dir = per_dataset_dir.parent if per_dataset_dir else None
    inferred_full_dir = (base_dir / 'per_dataset_full_population') if base_dir else None
    inferred_road_dir = (base_dir / 'road_based_top_k') if base_dir else None

    _register_population(
        key=TOP_RISK_POPULATION_KEY,
        label=TOP_RISK_POPULATION_LABEL,
        directory=per_dataset_dir,
        file_patterns=['dataset_*_top*pct_shap.csv']
    )

    _register_population(
        key=ROAD_HOTSPOT_POPULATION_KEY,
        label=ROAD_HOTSPOT_POPULATION_LABEL,
        directory=road_topk_dir or inferred_road_dir,
        file_patterns=['dataset_*_road_hotspot_shap.csv']
    )

    _register_population(
        key=FULL_POPULATION_KEY,
        label=FULL_POPULATION_LABEL,
        directory=full_population_dir or inferred_full_dir,
        file_patterns=['dataset_*_all_segments_shap.csv', 'dataset_*_full_population_shap.csv']
    )

    return entries

def generate_dataset_level_shap_analysis(segment_explanations: Dict[str, Any],
                                         master_pred_df: pd.DataFrame,
                                         output_dir: Path) -> Dict[str, Any]:
    """
    Generate dataset-level SHAP aggregation showing feature impacts by Dataset ID.
    
    This provides effect modification analysis: how feature impacts vary across
    different contexts (country × road type combinations).
    
    Args:
        segment_explanations: Output from generate_segment_explanations with SHAP values
        master_pred_df: DataFrame with Dataset ID for each segment
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with dataset analysis results and file paths
    """
    print("[INFO] Generating road-based top-K dataset SHAP analysis...")
    print("[INFO] This analyzes datasets represented in road-based hotspot selection")
    
    try:
        # Create dataset analysis directory - road-based top-K perspective
        dataset_dir = output_dir / 'dataset_shap_analysis' / 'road_based_top_k'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        segment_summary = segment_explanations.get('summary', pd.DataFrame())
        
        if segment_summary.empty:
            print("[WARN] No segment explanations available for dataset-level analysis")
            return {'summary': pd.DataFrame(), 'files_created': []}
        
        # Merge with master_pred_df to get Dataset ID
        dataset_col = cfg.DATASET_ID_COL
        if dataset_col not in master_pred_df.columns:
            print(f"[WARN] Dataset ID column '{dataset_col}' not found in master_pred_df")
            print(f"[DEBUG] Available columns in master_pred_df: {list(master_pred_df.columns)}")
            return {'summary': pd.DataFrame(), 'files_created': []}
        
        # Merge on segment_id
        id_col = cfg.ID_COL
        
        # Determine the segment ID column name in segment_summary
        segment_id_col = None
        for possible_col in ['segment_id', id_col, 'Location ID', 'location_id']:
            if possible_col in segment_summary.columns:
                segment_id_col = possible_col
                break
        
        if segment_id_col is None:
            print(f"[WARN] Could not find segment ID column in segment_summary. Columns: {list(segment_summary.columns)}")
            return {'summary': pd.DataFrame(), 'files_created': []}
        
        print(f"[INFO] Using segment ID column: '{segment_id_col}' for merge")
        
        # Prepare merge data with proper column naming
        # Use the segment_id column that exists in master_pred_df (not cfg.ID_COL which may not exist)
        master_id_col = None
        for possible_col in ['segment_id', cfg.ID_COL, 'Location ID']:
            if possible_col in master_pred_df.columns:
                master_id_col = possible_col
                break
        
        if master_id_col is None:
            print(f"[WARN] Could not find segment ID column in master_pred_df. Columns: {list(master_pred_df.columns)}")
            return {'summary': pd.DataFrame(), 'files_created': []}
        
        print(f"[DEBUG] Using '{master_id_col}' from master_pred_df for merge with '{segment_id_col}' in segment_summary")
        
        # CRITICAL FIX: Ensure data types match for successful merge
        # segment_id might be stored as string in one df and int/object in another
        merge_data = master_pred_df[[master_id_col, dataset_col]].copy()
        merge_data = merge_data.rename(columns={master_id_col: segment_id_col})
        
        # Convert both to string to ensure merge works (segment IDs are identifiers, not numeric values)
        segment_summary[segment_id_col] = segment_summary[segment_id_col].astype(str)
        merge_data[segment_id_col] = merge_data[segment_id_col].astype(str)
        
        print(f"[DEBUG] Data types - segment_summary['{segment_id_col}']: {segment_summary[segment_id_col].dtype}, "
              f"merge_data['{segment_id_col}']: {merge_data[segment_id_col].dtype}")
        
        # CRITICAL DIAGNOSTIC: Check for duplicates and actual data before merge
        print(f"[DEBUG] merge_data has {len(merge_data)} rows, {merge_data[segment_id_col].nunique()} unique segment IDs")
        print(f"[DEBUG] segment_summary has {len(segment_summary)} rows, {segment_summary[segment_id_col].nunique()} unique segment IDs")
        print(f"[DEBUG] merge_data columns: {list(merge_data.columns)}")
        print(f"[DEBUG] Does merge_data have Dataset ID? {dataset_col in merge_data.columns}")
        if dataset_col in merge_data.columns:
            print(f"[DEBUG] merge_data Dataset ID sample: {merge_data[dataset_col].head().tolist()}")
            print(f"[DEBUG] merge_data Dataset ID nulls: {merge_data[dataset_col].isna().sum()}/{len(merge_data)}")
        
        # Merge on the identified segment_id column
        segment_summary_with_dataset = segment_summary.merge(merge_data, on=segment_id_col, how='left')
        
        # Check if merge succeeded - count how many got Dataset IDs
        n_matched = segment_summary_with_dataset[dataset_col].notna().sum()
        n_total = len(segment_summary_with_dataset)
        match_rate = (n_matched / n_total * 100) if n_total > 0 else 0
        
        print(f"[INFO] Merge success: {n_matched}/{n_total} segments ({match_rate:.1f}%) got Dataset IDs")
        
        if n_matched == 0:
            print(f"[WARN] Dataset ID merge failed - no matching segment IDs. Check column '{segment_id_col}'")
            print(f"[DEBUG] Sample segment IDs from summary: {segment_summary[segment_id_col].head().tolist()}")
            print(f"[DEBUG] Sample segment IDs from master: {merge_data[segment_id_col].head().tolist()}")
            print(f"[DEBUG] Are IDs in master unique? {merge_data[segment_id_col].is_unique}")
            print(f"[DEBUG] Checking overlap: {set(segment_summary[segment_id_col].head(10)) & set(merge_data[segment_id_col].head(10))}")
            # CRITICAL: Check what happened to Dataset ID column after merge
            print(f"[DEBUG] After merge, columns are: {list(segment_summary_with_dataset.columns)}")
            print(f"[DEBUG] Dataset ID column exists after merge? {dataset_col in segment_summary_with_dataset.columns}")
            return {'summary': pd.DataFrame(), 'files_created': []}
        
        if match_rate < 50:
            print(f"[WARN] Low merge success rate ({match_rate:.1f}%). Dataset analysis may be incomplete.")
        
        # Drop rows without Dataset ID (these segments aren't in master_pred_df)
        segment_summary_with_dataset = segment_summary_with_dataset[
            segment_summary_with_dataset[dataset_col].notna()
        ].copy()
        
        # Get unique datasets
        datasets = segment_summary_with_dataset[dataset_col].dropna().unique()
        print(f"[INFO] Road-based analysis covers {len(datasets)} unique datasets")
        
        # DIAGNOSTIC: Check which datasets are missing from road-based top-K
        all_datasets_in_data = master_pred_df[dataset_col].dropna().unique()
        missing_datasets = set(all_datasets_in_data) - set(datasets)
        if len(missing_datasets) > 0:
            print(f"\n[INFO] Road-based top-K includes {len(datasets)}/{len(all_datasets_in_data)} datasets")
            print(f"[INFO] {len(missing_datasets)} datasets not in road-based top-K selection:")
            for ds in sorted(missing_datasets):
                n_segs = len(master_pred_df[master_pred_df[dataset_col] == ds])
                print(f"    Dataset {ds}: {n_segs:,} total segments")
        else:
            print(f"[INFO] All {len(datasets)} datasets represented in road-based top-K selection")
        
        # Extract SHAP columns
        shap_columns = [col for col in segment_summary_with_dataset.columns 
                       if col.startswith('top_feature_') and col.endswith('_shap')]
        name_columns = [col for col in segment_summary_with_dataset.columns 
                       if col.startswith('top_feature_') and col.endswith('_name')]
        
        if not shap_columns or not name_columns:
            print("[WARN] No SHAP feature columns found in segment summary")
            return {'summary': pd.DataFrame(), 'files_created': []}
        
        # Aggregate SHAP values by dataset
        dataset_feature_impacts = {}
        
        for dataset_id in datasets:
            dataset_segments = segment_summary_with_dataset[
                segment_summary_with_dataset[dataset_col] == dataset_id
            ]
            
            feature_impacts = {}
            for name_col, shap_col in zip(name_columns, shap_columns):
                for _, segment in dataset_segments.iterrows():
                    feature_name = segment.get(name_col)
                    shap_value = segment.get(shap_col)
                    if pd.notna(feature_name) and pd.notna(shap_value):
                        if feature_name not in feature_impacts:
                            feature_impacts[feature_name] = []
                        feature_impacts[feature_name].append(float(shap_value))
            
            dataset_feature_impacts[dataset_id] = feature_impacts
        
        # Create visualization for each dataset
        files_created = []
        dataset_summary_data = []
        
        for dataset_id, feature_impacts in dataset_feature_impacts.items():
            if not feature_impacts:
                continue
            
            long_df = _build_dataset_shap_longframe(feature_impacts)
            if not long_df.empty:
                shap_stats = (
                    long_df.groupby('feature')
                    .agg(
                        mean_shap=('shap_value', 'mean'),
                        mean_abs_shap=('abs_shap_value', 'mean'),
                        median_shap=('shap_value', 'median'),
                        std_shap=('shap_value', 'std'),
                        n_segments=('shap_value', 'count')
                    )
                    .reset_index()
                )
                shap_stats['std_shap'] = shap_stats['std_shap'].fillna(0.0)
                shap_stats['dataset_id'] = dataset_id
                shap_stats['population_key'] = ROAD_HOTSPOT_POPULATION_KEY
                shap_stats['population_label'] = ROAD_HOTSPOT_POPULATION_LABEL
                csv_out = dataset_dir / f'dataset_{dataset_id}_road_hotspot_shap.csv'
                shap_stats.to_csv(csv_out, index=False)
                files_created.append(str(csv_out))
            else:
                print(f"[WARN] No SHAP samples for dataset {dataset_id}; skipping CSV export")

            # Calculate average impacts
            avg_impacts = {feat: np.mean(np.abs(vals)) for feat, vals in feature_impacts.items()}
            avg_signed_impacts = {feat: np.mean(vals) for feat, vals in feature_impacts.items()}
            
            # Sort by absolute impact and take top 10
            sorted_features = sorted(avg_impacts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Create plot with appropriate height for top 10
            # Use extra-wide figure for dataset-level analysis to accommodate legend
            fig, ax = plt.subplots(figsize=(20, 10))
            
            feature_names = [f[0] for f in sorted_features]
            avg_values = [avg_signed_impacts[f[0]] for f in sorted_features]
            colors = ['red' if x > 0 else 'blue' for x in avg_values]
            
            bars = ax.barh(range(len(feature_names)), avg_values,
                          color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Calculate x-axis limits to make room for labels AND legend
            max_abs_val = max(abs(min(avg_values)), abs(max(avg_values)))
            x_margin = max_abs_val * 0.35  # 35% margin for labels + legend space
            ax.set_xlim(-max_abs_val - x_margin, max_abs_val + x_margin)
            
            # Add value labels outside bars on the right side
            for i, (bar, value) in enumerate(zip(bars, avg_values)):
                width = bar.get_width()
                if abs(width) > 0.001:
                    x_offset = max_abs_val * 0.02  # 2% of max value as offset
                    # Always place at the rightmost point (bar tip)
                    label_x = width + x_offset if width >= 0 else width - x_offset
                    ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
                           ha='left' if width >= 0 else 'right', va='center',
                           fontsize=10, color='black', fontweight='bold')
            
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names, fontsize=11)
            ax.set_xlabel('Average SHAP Value (Impact on Prediction)', fontsize=13, fontweight='bold')
            
            # Get segment count for this dataset
            n_segments = len(segment_summary_with_dataset[
                segment_summary_with_dataset[dataset_col] == dataset_id
            ])
            
            ax.set_title(f'Dataset {dataset_id} - Feature Impact Analysis\n'
                        f'Top 10 Features | Aggregated across {n_segments} segments',
                        fontsize=14, fontweight='bold')
            
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(axis='x', alpha=0.3)
            
            # Add scientific summary statistics box - positioned to avoid bars
            stats_text = f"Statistical Summary (Top 10):\n"
            stats_text += f"Total Features: {len(feature_impacts)}\n"
            stats_text += f"Segments: {n_segments}\n"
            stats_text += f"Mean |SHAP|: {np.mean([avg_impacts[f[0]] for f in sorted_features]):.4f}\n"
            stats_text += f"\nTop 3 Risk Drivers:\n"
            for i, (feat, val) in enumerate(sorted_features[:3]):
                impact = avg_signed_impacts[feat]
                stats_text += f"{i+1}. {feat}: {impact:+.4f}\n"
            
            # Position legend in upper right with proper spacing
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), 
                   fontsize=9, family='monospace')
            
            plt.tight_layout(pad=2.0)  # Add more padding
            
            # Save plot with extra space
            plot_path = dataset_dir / f'dataset_{dataset_id}_shap_analysis.png'
            fig.savefig(plot_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
            plt.close(fig)
            files_created.append(str(plot_path))
            
            # Store summary data
            dataset_summary_data.append({
                'dataset_id': dataset_id,
                'n_segments': n_segments,
                'top_feature_1': sorted_features[0][0] if len(sorted_features) > 0 else None,
                'top_feature_1_impact': avg_signed_impacts[sorted_features[0][0]] if len(sorted_features) > 0 else None,
                'top_feature_2': sorted_features[1][0] if len(sorted_features) > 1 else None,
                'top_feature_2_impact': avg_signed_impacts[sorted_features[1][0]] if len(sorted_features) > 1 else None,
                'top_feature_3': sorted_features[2][0] if len(sorted_features) > 2 else None,
                'top_feature_3_impact': avg_signed_impacts[sorted_features[2][0]] if len(sorted_features) > 2 else None,
                'plot_file': plot_path.name
            })
        
        # Create summary DataFrame
        dataset_summary_df = pd.DataFrame(dataset_summary_data)
        
        # Save summary CSV
        if not dataset_summary_df.empty:
            summary_path = dataset_dir / 'dataset_shap_summary.csv'
            dataset_summary_df.to_csv(summary_path, index=False)
            print(f"[INFO] Dataset SHAP summary saved to: {summary_path}")
        
        # Create violin plots for top features across all datasets
        print("[INFO] Generating dataset-level SHAP violin plots...")
        violin_files = _create_dataset_violin_plots(
            dataset_feature_impacts, 
            segment_summary_with_dataset, 
            dataset_col, 
            dataset_dir
        )
        files_created.extend(violin_files)
        
        print(f"[INFO] Generated {len(files_created)} dataset-level SHAP analysis plots")
        
        # Write README explaining this analysis
        readme_path = dataset_dir / 'README.txt'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ROAD-BASED TOP-K DATASET SHAP ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            f.write("PURPOSE:\n")
            f.write("  This analysis shows SHAP feature importance for datasets\n")
            f.write("  represented in the road-based hotspot selection (top-K per road).\n\n")
            f.write("METHODOLOGY:\n")
            f.write("  1. Select top-K most dangerous segments per road\n")
            f.write("  2. Compute SHAP values for these selected segments\n")
            f.write("  3. Aggregate by Dataset ID to see which datasets dominate\n\n")
            f.write(f"COVERAGE:\n")
            f.write(f"  Datasets analyzed: {len(datasets)}/12\n")
            f.write(f"  Total segments: {len(segment_summary_with_dataset):,}\n\n")
            if len(missing_datasets) > 0:
                f.write(f"DATASETS NOT IN ROAD-BASED TOP-K:\n")
                for ds in sorted(missing_datasets):
                    f.write(f"  - Dataset {ds}\n")
                f.write("\n")
            f.write("INTERPRETATION:\n")
            f.write("  - Shows which datasets contribute to the highest-risk roads\n")
            f.write("  - Reveals geographic concentration of road safety issues\n")
            f.write("  - Complements per-dataset analysis (see ../per_dataset_top_risk/)\n\n")
            f.write("FOR COMPLETE DATASET COVERAGE (all 12 datasets):\n")
            f.write("  See: ../per_dataset_top_risk/\n\n")
            f.write("FILES IN THIS DIRECTORY:\n")
            f.write("  - dataset_shap_summary.csv: Feature importance by dataset\n")
            f.write("  - dataset_shap_violin_*.png: Distribution plots\n")
            f.write("  - dataset_feature_heatmap.png: Dataset-feature heatmap\n")
            f.write("  - dataset_shap_statistical_summary.csv: Statistical summary\n")
        files_created.append(str(readme_path))
        
        return {
            'summary': dataset_summary_df,
            'files_created': files_created,
            'output_directory': str(dataset_dir)
        }
        
    except Exception as e:
        print(f"[ERROR] Dataset-level SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {'summary': pd.DataFrame(), 'files_created': []}


def generate_per_dataset_top_risk_shap(
    master_pred_df: pd.DataFrame,
    model: Any,
    X_features: pd.DataFrame,
    preprocessor: Any,
    output_dir: Path,
    top_pct: float = 0.05,
    fold_results_dir: Optional[Path] = None,
    metadata_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Generate SHAP analysis for top N% most dangerous segments WITHIN each dataset.
    This ensures all 12 datasets get SHAP analysis regardless of road-based selection.
    
    Args:
        master_pred_df: Prediction DataFrame (may be filtered to top-K)
        model: Trained model
        X_features: Feature DataFrame (should have same row order as metadata_df)
        preprocessor: Fitted preprocessor
        output_dir: Base output directory
        top_pct: Percentage of top risk segments to analyze per dataset (default 5%)
        fold_results_dir: Directory containing full OOF predictions CSV (if None, uses master_pred_df)
        metadata_df: Optional metadata DataFrame for ID matching (should align with X_features rows)
        
    Returns:
        Dictionary with analysis results and file paths
    """
    print(f"\n{'='*80}")
    print(f"[INFO] Generating per-dataset top-{top_pct*100:.0f}% risk SHAP analysis...")
    print(f"{'='*80}\n")
    
    try:
        # Create subdirectory
        per_dataset_dir = output_dir / 'dataset_shap_analysis' / 'per_dataset_top_risk'
        per_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_col = cfg.DATASET_ID_COL
        
        # CRITICAL FIX: Load full OOF predictions if fold_results_dir provided
        # This ensures all datasets get representation, not just those in top-K
        if fold_results_dir is not None:
            oof_file = fold_results_dir / 'oof_predictions_segments.csv'
            if oof_file.exists():
                print(f"[INFO] Loading FULL OOF predictions from: {oof_file}")
                full_pred_df = pd.read_csv(oof_file)
                print(f"[INFO] Loaded {len(full_pred_df):,} total segments (full dataset)")
                
                # Standardize column names: OOF file uses 'predicted_risk', master_pred_df uses 'predictions'
                if 'predicted_risk' in full_pred_df.columns and 'predictions' not in full_pred_df.columns:
                    full_pred_df['predictions'] = full_pred_df['predicted_risk']
                    print(f"[INFO] Standardized column: 'predicted_risk' -> 'predictions'")
                
                # Ensure ID column exists for feature matching
                # OOF file has 'segment_id', but cfg.ID_COL might be 'Location ID'
                if cfg.ID_COL not in full_pred_df.columns and 'segment_id' in full_pred_df.columns:
                    full_pred_df[cfg.ID_COL] = full_pred_df['segment_id']
                    print(f"[INFO] Standardized ID column: 'segment_id' -> '{cfg.ID_COL}'")
                
                # Use full predictions instead of filtered master_pred_df
                working_df = full_pred_df
            else:
                print(f"[WARN] OOF file not found: {oof_file}")
                print(f"[WARN] Falling back to master_pred_df ({len(master_pred_df):,} segments)")
                working_df = master_pred_df
        else:
            print(f"[INFO] Using master_pred_df with {len(master_pred_df):,} segments")
            working_df = master_pred_df
        
        if dataset_col not in working_df.columns:
            print(f"[ERROR] Dataset ID column '{dataset_col}' not found")
            return {'summary': pd.DataFrame(), 'files_created': []}
        
        # Determine if model requires numeric-only input (LightGBM) and build global encoding maps
        model_name = type(model).__name__ if model is not None else ''
        is_lightgbm_model = model_name.lower().startswith('lgbm') or 'lgbm' in model_name.lower()
        lightgbm_category_maps: Dict[str, List[Any]] = {}
        if is_lightgbm_model and X_features is not None:
            obj_cols_global = X_features.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in obj_cols_global:
                try:
                    # Preserve observed order for stability; fall back to sorted unique
                    ordered = pd.Series(X_features[col].dropna().unique()).tolist()
                    if not ordered:
                        ordered = sorted(pd.Series(X_features[col].dropna().astype(str)).unique())
                    lightgbm_category_maps[col] = ordered
                except Exception:
                    continue

        def _encode_for_lightgbm(df_subset: pd.DataFrame) -> pd.DataFrame:
            """Ensure LightGBM receives purely numeric input by ordinal-encoding object columns."""
            if not is_lightgbm_model or df_subset is None or df_subset.empty:
                return df_subset
            encoded = df_subset.copy()
            obj_cols = encoded.select_dtypes(include=['object', 'category']).columns.tolist()
            if not obj_cols:
                return encoded
            for col in obj_cols:
                categories = lightgbm_category_maps.get(col)
                if not categories:
                    try:
                        categories = pd.Series(encoded[col].dropna().unique()).tolist()
                    except Exception:
                        categories = None
                if categories:
                    encoded[col] = pd.Categorical(encoded[col], categories=categories).codes
                else:
                    encoded[col] = pd.Categorical(encoded[col]).codes
            return encoded

        # Get all unique datasets
        all_datasets = sorted(working_df[dataset_col].dropna().unique())
        print(f"[INFO] Found {len(all_datasets)} datasets: {all_datasets}\n")
        
        files_created = []
        dataset_summaries = []
        
        # Process each dataset independently
        for dataset_id in all_datasets:
            print(f"\n--- Processing Dataset {dataset_id} ---")
            
            # Get all segments from this dataset
            dataset_mask = working_df[dataset_col] == dataset_id
            dataset_df = working_df[dataset_mask].copy()
            
            n_total = len(dataset_df)
            n_select = max(int(n_total * top_pct), 1)  # At least 1 segment
            
            print(f"  Total segments: {n_total:,}")
            print(f"  Selecting top {top_pct*100:.0f}%: {n_select:,} segments")
            
            # Select top N% by predicted risk
            dataset_df_sorted = dataset_df.nlargest(n_select, 'predictions')
            selected_ids = dataset_df_sorted[cfg.ID_COL].tolist()
            
            print(f"  Risk range: {dataset_df_sorted['predictions'].min():.4f} - {dataset_df_sorted['predictions'].max():.4f}")
            
            # Get features for these segments
            # Use global_index from OOF predictions to map back to original feature matrix rows
            if 'global_index' in dataset_df_sorted.columns:
                global_indices = dataset_df_sorted['global_index'].tolist()
                
                # Verify indices are valid
                max_idx = max(global_indices) if global_indices else -1
                if max_idx >= len(X_features):
                    print(f"  [ERROR] global_index out of range: max {max_idx} >= {len(X_features)} rows")
                    print(f"  [INFO] This means X_features doesn't have all segments. Need full unfiltered matrix.")
                    continue
                
                # Extract features using global indices
                X_dataset = X_features.iloc[global_indices].copy()
                print(f"  Matched {len(X_dataset)}/{len(global_indices)} segments using global_index")
                
            else:
                print(f"  [ERROR] No 'global_index' column in OOF predictions")
                print(f"  [INFO] Cannot match features - need global_index to map to feature matrix rows")
                continue
            
            if len(X_dataset) == 0:
                print(f"  [WARN] No features extracted for dataset {dataset_id}")
                continue
            
            print(f"  Computing SHAP values for {len(X_dataset)} segments...")
            
            # Compute SHAP values
            try:
                import shap
                
                X_dataset_prepped = _encode_for_lightgbm(X_dataset)

                # Create explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_dataset_prepped)
                
                # Get mean absolute SHAP per feature
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                feature_names = X_dataset_prepped.columns.tolist()
                
                # Create DataFrame
                shap_df = pd.DataFrame({
                    'feature': feature_names,
                    'mean_abs_shap': mean_abs_shap,
                    'mean_shap': shap_values.mean(axis=0)
                })
                shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
                
                # Save CSV
                csv_path = per_dataset_dir / f'dataset_{dataset_id}_top{int(top_pct*100)}pct_shap.csv'
                shap_df.to_csv(csv_path, index=False)
                files_created.append(str(csv_path))
                
                # Create visualization for top 15 features
                top_features = shap_df.head(15)
                
                fig, ax = plt.subplots(figsize=(16, 10))
                
                colors = ['red' if x > 0 else 'blue' for x in top_features['mean_shap']]
                bars = ax.barh(range(len(top_features)), top_features['mean_shap'],
                              color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Calculate x-axis limits to make room for labels
                max_abs_val = max(abs(top_features['mean_shap'].min()), abs(top_features['mean_shap'].max()))
                x_margin = max_abs_val * 0.25  # 25% margin for labels
                ax.set_xlim(-max_abs_val - x_margin, max_abs_val + x_margin)
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, top_features['mean_shap'])):
                    width = bar.get_width()
                    if abs(width) > 0.001:
                        x_offset = max_abs_val * 0.02  # 2% of max value as offset
                        label_x = width + x_offset if width >= 0 else width - x_offset
                        ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
                               ha='left' if width >= 0 else 'right', va='center',
                               fontsize=10, color='black', fontweight='bold')
                
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'], fontsize=11)
                ax.set_xlabel('Mean SHAP Value (Impact on Prediction)', fontsize=13, fontweight='bold')
                ax.set_title(f'Dataset {dataset_id} - Top {int(top_pct*100)}% Risk Segments\n'
                            f'Feature Impact Analysis | {n_select:,} segments ({top_pct*100:.1f}% of {n_total:,})',
                            fontsize=14, fontweight='bold')
                
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax.grid(axis='x', alpha=0.3)
                
                # Add stats box
                stats_text = f"Dataset {dataset_id} Statistics:\n"
                stats_text += f"Total segments: {n_total:,}\n"
                stats_text += f"Analyzed (top {top_pct*100:.0f}%): {n_select:,}\n"
                stats_text += f"Risk range: {dataset_df_sorted['predictions'].min():.3f}-{dataset_df_sorted['predictions'].max():.3f}\n"
                stats_text += f"\nTop 3 Risk Drivers:\n"
                for idx, row in top_features.head(3).iterrows():
                    stats_text += f"{row['feature']}: {row['mean_shap']:+.4f}\n"
                
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), 
                       fontsize=9, family='monospace')
                
                plt.tight_layout(pad=2.0)  # Add more padding
                
                # Save plot with extra space
                plot_path = per_dataset_dir / f'dataset_{dataset_id}_top{int(top_pct*100)}pct_shap.png'
                fig.savefig(plot_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
                plt.close(fig)
                files_created.append(str(plot_path))
                
                print(f"  ✓ SHAP analysis saved: {plot_path.name}")
                
                # Store summary
                dataset_summaries.append({
                    'dataset_id': dataset_id,
                    'total_segments': n_total,
                    'analyzed_segments': n_select,
                    'top_feature_1': top_features.iloc[0]['feature'],
                    'top_feature_1_shap': top_features.iloc[0]['mean_shap'],
                    'top_feature_2': top_features.iloc[1]['feature'],
                    'top_feature_2_shap': top_features.iloc[1]['mean_shap'],
                    'top_feature_3': top_features.iloc[2]['feature'],
                    'top_feature_3_shap': top_features.iloc[2]['mean_shap'],
                })
                
            except Exception as e:
                print(f"  [ERROR] SHAP computation failed for dataset {dataset_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create overall summary
        summary_df = pd.DataFrame(dataset_summaries)
        if not summary_df.empty:
            summary_path = per_dataset_dir / 'all_datasets_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            files_created.append(str(summary_path))
            print(f"\n[SUCCESS] Summary saved: {summary_path.name}")
        
        print(f"\n{'='*80}")
        print(f"[SUCCESS] Per-dataset SHAP analysis complete!")
        print(f"  Total datasets processed: {len(dataset_summaries)}/{len(all_datasets)}")
        print(f"  Total files created: {len(files_created)}")
        print(f"  Output directory: {per_dataset_dir}")
        print(f"{'='*80}\n")
        
        # Write README explaining this analysis
        readme_path = per_dataset_dir / 'README.txt'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("PER-DATASET TOP-RISK SHAP ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            f.write("PURPOSE:\n")
            f.write("  This analysis ensures ALL datasets are represented by analyzing\n")
            f.write("  the top-5% most dangerous segments WITHIN each dataset.\n\n")
            f.write("METHODOLOGY:\n")
            f.write("  1. Load FULL out-of-fold predictions (147,466 segments)\n")
            f.write("  2. For each dataset, select top 5% by predicted risk\n")
            f.write("  3. Compute SHAP values for selected segments per dataset\n")
            f.write("  4. Save individual CSV + visualization per dataset\n\n")
            f.write(f"COVERAGE:\n")
            f.write(f"  Datasets analyzed: {len(dataset_summaries)}/12 (target: all 12)\n")
            f.write(f"  Total segments analyzed: ~{sum([s['analyzed_segments'] for s in dataset_summaries]):,}\n\n")
            f.write("KEY DIFFERENCE FROM ROAD-BASED ANALYSIS:\n")
            f.write("  - Road-based: Selects top-K per ROAD, then groups by dataset\n")
            f.write("    -> Some datasets may have 0 segments if not in road top-K\n")
            f.write("  - Per-dataset: Selects top-5% per DATASET directly\n")
            f.write("    -> Guarantees all 12 datasets are analyzed fairly\n\n")
            f.write("INTERPRETATION:\n")
            f.write("  - Shows what drives risk WITHIN each dataset context\n")
            f.write("  - Enables comparison of feature effects across datasets\n")
            f.write("  - Supports regional analysis (see ../regional_analysis/)\n\n")
            f.write("FILES IN THIS DIRECTORY:\n")
            f.write("  - dataset_XXX_top5pct_shap.csv: Feature importance per dataset\n")
            f.write("  - dataset_XXX_top5pct_shap_chart.png: Visualization per dataset\n")
            f.write("  - all_datasets_summary.csv: Overview of all datasets\n\n")
            f.write("RELATED ANALYSES:\n")
            f.write("  - Road-based view: ../road_based_top_k/\n")
            f.write("  - Regional aggregation: ../regional_analysis/\n")
        files_created.append(str(readme_path))
        
        return {
            'summary': summary_df,
            'files_created': files_created,
            'output_directory': str(per_dataset_dir)
        }
        
    except Exception as e:
        print(f"[ERROR] Per-dataset top-risk SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {'summary': pd.DataFrame(), 'files_created': []}


def create_explanation_html_report(segment_explanations: Dict[str, Any],
                                  road_explanations: Dict[str, Any],
                                  output_dir: Path) -> str:
    """Create comprehensive HTML report linking all explanations."""
    
    report_path = output_dir / cfg.EXPLANATION_HTML_REPORT
    
    try:
        # Get summaries
        segment_summary = segment_explanations.get('summary', pd.DataFrame())
        road_summary = road_explanations.get('summary', pd.DataFrame())
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Road Risk Model Explanations Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .explanation-link {{ color: blue; text-decoration: underline; cursor: pointer; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Road Risk Model Explanations Report</h1>
                <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Segment Explanations: {len(segment_summary)} | Road Explanations: {len(road_summary)}</p>
            </div>
            
            <div class="section">
                <h2>Top Segment Explanations</h2>
        """
        
        if not segment_summary.empty:
            html_content += segment_summary.head(20).to_html(escape=False, classes='table')
        else:
            html_content += "<p>No segment explanations available.</p>"
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Road-Level Explanations</h2>
        """
        
        if not road_summary.empty:
            html_content += road_summary.to_html(escape=False, classes='table')
        else:
            html_content += "<p>No road explanations available.</p>"
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Files Generated</h2>
                <h3>Segment Explanation Files:</h3>
                <ul>
        """
        
        for file_path in segment_explanations.get('files_created', []):
            file_name = Path(file_path).name
            html_content += f"<li>{file_name}</li>"
        
        html_content += """
                </ul>
                <h3>Road Explanation Files:</h3>
                <ul>
        """
        
        for file_path in road_explanations.get('files_created', []):
            file_name = Path(file_path).name
            html_content += f"<li>{file_name}</li>"
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[INFO] HTML explanation report created: {report_path}")
        return str(report_path)
        
    except Exception as e:
        print(f"[ERROR] Failed to create HTML report: {e}")
        return ""

def generate_all_individual_explanations(master_pred_df: pd.DataFrame,
                                        model: Any,
                                        X_features: pd.DataFrame,
                                        output_dir: Path,
                                        metadata_df: Optional[pd.DataFrame] = None,
                                        id_col_name: Optional[str] = None,
                                        preprocessor: Optional[Any] = None,
                                        fold_artifact_index: Optional[str] = None) -> Dict[str, Any]:
    """
    Master function to generate all individual explanations - INTELLIGENT DISPATCHER.
    
    EXPERT FIX: This function now acts as an intelligent dispatcher that:
    1. Defaults to using the global model and preprocessor for robust explanations
    2. Only attempts per-fold mode if fold_artifact_index is valid and verified
    3. Provides clear fallback logic to prevent NoneType model errors
    
    Args:
        master_pred_df: DataFrame with segment predictions and metadata
        model: GLOBAL trained model for SHAP analysis (always valid fallback)
        X_features: Feature matrix aligned with predictions
        output_dir: Directory to save explanation files
        metadata_df: Optional metadata for coordinate mapping
        id_col_name: Optional ID column name for mapping
        preprocessor: GLOBAL preprocessor (always valid fallback)
        fold_artifact_index: Optional path to fold artifact index for per-fold explanations
        
    Returns:
        Dictionary with all explanation results
    """
    print("\n=== GENERATING INDIVIDUAL EXPLANATIONS ===")
    print(f"[INFO] Model provided: {model is not None}")
    print(f"[INFO] Preprocessor provided: {preprocessor is not None}")
    print(f"[INFO] Fold artifact index: {fold_artifact_index}")
    
    results = {
        'segment_explanations': {},
        'road_explanations': {},
        'dataset_shap_analysis': {},
        'html_report': '',
        'total_files_created': 0
    }
    
    # EXPERT FIX: Validate inputs to ensure we have valid fallbacks
    if model is None:
        raise ValueError("[CRITICAL] Global model is None - this should never happen after expert fix")
    
    try:
        # --- EXPERT FIX: INTELLIGENT DISPATCH LOGIC ---
        # Default to global model/preprocessor unless per-fold is explicitly validated
        use_per_fold = False
        validated_fold_index = None
        
        if fold_artifact_index is not None:
            print(f"[INFO] Checking fold artifact index for per-fold explanations: {fold_artifact_index}")
            try:
                # Verify fold artifact index exists and is readable
                idx_df = pd.read_csv(fold_artifact_index)
                print(f"[INFO] Fold artifact index loaded: {len(idx_df)} entries")

                # Normalize fold column in index for validation
                idx_fold_col = 'fold_number' if 'fold_number' in idx_df.columns else ('fold' if 'fold' in idx_df.columns else None)
                if idx_fold_col is None:
                    raise FileNotFoundError(f"[FAIL] Could not find fold column in fold_artifact_index for validation: {fold_artifact_index}")

                # Check that all folds in master_pred_df are present in the index
                folds_needed = set(master_pred_df['fold_number'].unique()) if 'fold_number' in master_pred_df.columns else set()
                if not folds_needed:
                    folds_needed = set(master_pred_df['fold'].unique()) if 'fold' in master_pred_df.columns else set()

                model_types_needed = set(master_pred_df['model_type'].unique()) if 'model_type' in master_pred_df.columns else set(['model'])

                missing_folds = []
                for fold in folds_needed:
                    for model_type in model_types_needed:
                        candidates = idx_df[(idx_df[idx_fold_col] == int(fold))]
                        if 'model_type' in idx_df.columns:
                            candidates = candidates[candidates['model_type'].astype(str) == str(model_type)]
                        if len(candidates) == 0:
                            missing_folds.append((fold, model_type))
                
                if missing_folds:
                    if getattr(cfg, 'ALLOW_GLOBAL_PROXY_EXPLANATIONS', False):
                        print(f"[WARN] Missing per-fold artifacts for folds: {missing_folds}. Using available per-fold and global fallback (proxy enabled).")
                        use_per_fold = True
                        validated_fold_index = fold_artifact_index
                    else:
                        print(f"[WARN] Missing per-fold artifacts for folds: {missing_folds}")
                        print(f"[WARN] FALLING BACK to global model for all explanations (proxy disabled)")
                        use_per_fold = False
                else:
                    print(f"[INFO] All required fold artifacts validated - using per-fold mode")
                    use_per_fold = True
                    validated_fold_index = fold_artifact_index
                    
            except Exception as e:
                print(f"[WARN] Could not validate fold_artifact_index '{fold_artifact_index}': {e}")
                print(f"[WARN] FALLING BACK to global model for all explanations")
                use_per_fold = False
        else:
            print(f"[INFO] No fold artifact index provided - using global model")
        
        print(f"[INFO] Explanation mode: {'Per-fold' if use_per_fold else 'Global model'}")

        # Generate segment explanations with appropriate mode
        segment_results = {}
        if cfg.GENERATE_SEGMENT_EXPLANATIONS:
            segment_results = generate_segment_explanations(
                master_pred_df=master_pred_df,
                model=model,  # Always pass the global model as fallback
                X_features=X_features,
                output_dir=output_dir,
                metadata_df=metadata_df,
                id_col_name=id_col_name,
                preprocessor=preprocessor,  # Always pass the global preprocessor as fallback
                fold_artifact_index=validated_fold_index if use_per_fold else None
            )
            results['segment_explanations'] = segment_results

        if cfg.GENERATE_ROAD_EXPLANATIONS:
            road_results = generate_road_explanations(
                master_pred_df=master_pred_df,
                segment_explanations=results['segment_explanations'],
                output_dir=output_dir
            )
            results['road_explanations'] = road_results
        
        # Generate dataset-level SHAP aggregation analysis
        if cfg.GENERATE_SEGMENT_EXPLANATIONS:  # Only if we have segment-level SHAP
            dataset_results = generate_dataset_level_shap_analysis(
                segment_explanations=results['segment_explanations'],
                master_pred_df=master_pred_df,
                output_dir=output_dir
            )
            results['dataset_shap_analysis'] = dataset_results

        if cfg.SAVE_EXPLANATION_HTML_REPORTS:
            html_report_path = create_explanation_html_report(
                segment_explanations=results['segment_explanations'],
                road_explanations=results['road_explanations'],
                output_dir=output_dir
            )
            results['html_report'] = html_report_path

        total_files = (len(results['segment_explanations'].get('files_created', [])) + 
                      len(results['road_explanations'].get('files_created', [])) + 
                      len(results['dataset_shap_analysis'].get('files_created', [])) +
                      (1 if results['html_report'] else 0))
        results['total_files_created'] = total_files

        print(f"\n=== EXPLANATION GENERATION COMPLETE ===")
        print(f"Total files created: {total_files}")
        print(f"Segment explanations: {len(results['segment_explanations'].get('summary', pd.DataFrame()))}")
        print(f"Road explanations: {len(results['road_explanations'].get('summary', pd.DataFrame()))}")
        print(f"Dataset-level SHAP analyses: {len(results['dataset_shap_analysis'].get('summary', pd.DataFrame()))}")

    except Exception as e:
        print(f"[ERROR] Individual explanation generation failed: {e}")
        traceback.print_exc()

    return results


def generate_regional_shap_analysis(
    per_dataset_dir: Path,
    output_dir: Path,
    full_population_dir: Optional[Path] = None,
    road_topk_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate regional SHAP analysis by aggregating per-dataset results.
    Creates bar plots and violin plots for each region.
    
    Args:
        per_dataset_dir: Directory containing per-dataset SHAP CSV files
        output_dir: Base output directory
        
    Returns:
        Dictionary with regional analysis results and file paths
    """
    print(f"\n{'='*80}")
    print(f"[INFO] Generating Regional SHAP Analysis...")
    print(f"{'='*80}\n")
    
    try:
        regional_dir = output_dir / 'dataset_shap_analysis' / 'regional_analysis'
        regional_dir.mkdir(parents=True, exist_ok=True)

        population_entries = _prepare_regional_population_entries(
            per_dataset_dir=per_dataset_dir,
            full_population_dir=full_population_dir,
            road_topk_dir=road_topk_dir
        )

        if not population_entries:
            print("[WARN] No population entries resolved for regional analysis")
            return {'summary': pd.DataFrame(), 'files_created': []}

        files_created: List[str] = []
        regional_summaries: List[Dict[str, Any]] = []

        for population_entry in population_entries:
            dataset_shap_data = population_entry['dataset_shap_data']
            population_key = population_entry['key']
            population_label = population_entry['label']

            print(f"\n{'='*80}")
            print(f"[INFO] Population: {population_label}")
            print(f"[INFO] Source directory: {population_entry['source_dir']}")
            print(f"[INFO] Datasets available: {len(dataset_shap_data)}")
            print(f"{'='*80}\n")

            for region_key, region_info in cfg.REGIONAL_GROUPINGS.items():
                region_name = region_info['name']
                region_datasets = region_info['datasets']

                print(f"\n{'-'*70}")
                print(f"Region: {region_name} | Population: {population_label}")
                print(f"Datasets expected: {region_datasets}")
                print(f"Countries: {region_info['countries']}")

                available_datasets = [d for d in region_datasets if d in dataset_shap_data]
                if not available_datasets:
                    print(f"  [WARN] No SHAP data available for region {region_name} in population {population_label}")
                    continue

                print(f"  Available datasets: {available_datasets}")

                all_features = set()
                for dataset_id in available_datasets:
                    all_features.update(dataset_shap_data[dataset_id]['feature'].tolist())

                aggregated_data = []
                for feature in all_features:
                    feature_shaps = []
                    for dataset_id in available_datasets:
                        df = dataset_shap_data[dataset_id]
                        feature_row = df[df['feature'] == feature]
                        if not feature_row.empty:
                            feature_shaps.append(feature_row['mean_shap'].iloc[0])

                    if feature_shaps:
                        aggregated_data.append({
                            'feature': feature,
                            'mean_shap': np.mean(feature_shaps),
                            'std_shap': np.std(feature_shaps),
                            'mean_abs_shap': np.mean(np.abs(feature_shaps)),
                            'n_datasets': len(feature_shaps)
                        })

                agg_df = pd.DataFrame(aggregated_data).sort_values('mean_abs_shap', ascending=False)
                print(f"  Aggregated {len(agg_df)} features")

                csv_path = regional_dir / f'{region_key}_{population_key}_aggregated_shap.csv'
                agg_df.to_csv(csv_path, index=False)
                files_created.append(str(csv_path))
                print(f"  ✓ Saved: {csv_path.name}")

                bar_path = _create_regional_bar_plot(
                    agg_df,
                    region_name,
                    region_info,
                    available_datasets,
                    regional_dir,
                    region_key,
                    population_key,
                    population_label
                )
                files_created.append(str(bar_path))

                violin_path = _create_regional_violin_plot(
                    dataset_shap_data,
                    available_datasets,
                    agg_df,
                    region_name,
                    region_info,
                    regional_dir,
                    region_key,
                    population_key,
                    population_label
                )
                files_created.append(str(violin_path))

                regional_summaries.append({
                    'region': region_name,
                    'region_key': region_key,
                    'population_key': population_key,
                    'population_label': population_label,
                    'n_datasets': len(available_datasets),
                    'n_features': len(agg_df),
                    'top_feature': agg_df.iloc[0]['feature'] if len(agg_df) > 0 else None,
                    'top_feature_impact': agg_df.iloc[0]['mean_abs_shap'] if len(agg_df) > 0 else None
                })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(regional_summaries)
        summary_path = regional_dir / 'regional_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        files_created.append(str(summary_path))
        
        print(f"\n{'='*80}")
        print(f"[SUCCESS] Regional SHAP Analysis Complete")
        print(f"{'='*80}")
        print(f"  Files created: {len(files_created)}")
        print(f"  Output directory: {regional_dir}")
        print(f"{'='*80}\n")
        
        # Write README explaining this analysis
        readme_path = regional_dir / 'README.txt'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("REGIONAL SHAP ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            f.write("PURPOSE:\n")
            f.write("  This analysis aggregates per-dataset SHAP results by geographic\n")
            f.write("  region to reveal regional patterns in road safety risk factors.\n\n")
            f.write("METHODOLOGY:\n")
            f.write("  1. Load per-dataset SHAP CSVs for each population cohort\n")
            f.write("  2. Group datasets by region using coordinate-based detection\n")
            f.write("  3. Aggregate SHAP values across datasets within each region\n")
            f.write("  4. Create bar charts (mean impact) and violin plots (distribution)\n\n")
            f.write("POPULATIONS COVERED:\n")
            for entry in population_entries:
                f.write(f"  - {entry['label']} ({entry['key']})\n")
            f.write("\n")
            f.write("REGIONAL GROUPINGS:\n")
            for region in regional_summaries:
                f.write(f"  {region['region']}:\n")
                f.write(f"    Datasets: {region['n_datasets']}\n")
                f.write(f"    Population: {region['population_label']}\n")
                if region.get('top_feature'):
                    f.write(f"    Top feature: {region['top_feature']}\n")
                f.write("\n")
            f.write("INTERPRETATION:\n")
            f.write("  - Bar charts: Show average feature importance per region\n")
            f.write("  - Violin plots: Show distribution of feature impacts across datasets\n")
            f.write("  - Enables regional policy recommendations\n")
            f.write("  - Reveals geographic heterogeneity in risk factors\n\n")
            f.write("FILES IN THIS DIRECTORY:\n")
            f.write("  - *_aggregated_shap.csv: Regional feature importance data (per population)\n")
            f.write("  - *_shap_bar_chart.png: Mean impact with error bars (per population)\n")
            f.write("  - *_shap_violin_plot.png: Distribution across datasets (per population)\n")
            f.write("  - regional_summary.csv: Overview of all regions\n\n")
            f.write("RELATED ANALYSES:\n")
            f.write("  - Source data: ../per_dataset_top_risk/\n")
            f.write("  - Road-based view: ../road_based_top_k/\n")
        files_created.append(str(readme_path))
        
        return {
            'summary': summary_df,
            'files_created': files_created,
            'output_dir': str(regional_dir)
        }
        
    except Exception as e:
        print(f"[ERROR] Regional SHAP analysis failed: {e}")
        traceback.print_exc()
        return {'summary': pd.DataFrame(), 'files_created': []}


def _create_regional_bar_plot(agg_df, region_name, region_info, datasets, output_dir, region_key,
                              population_key, population_label):
    """Create bar plot for regional aggregated SHAP values"""
    top_features = agg_df.head(15)
    if top_features.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'No feature data available for {region_name}\nPopulation: {population_label}',
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax.axis('off')
        plot_path = output_dir / f'{region_key}_{population_key}_shap_bar_top15.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"  [WARN] Skipped bar plot for {region_name} ({population_label}) due to empty data")
        return plot_path
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    y_pos = np.arange(len(top_features))
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_features['mean_shap']]
    
    bars = ax.barh(y_pos, top_features['mean_shap'], color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    max_abs_val = top_features['mean_shap'].abs().max()
    x_margin = max_abs_val * 0.35
    
    for i, (bar, row) in enumerate(zip(bars, top_features.itertuples())):
        value = row.mean_shap
        if value > 0:
            x_pos = value + max_abs_val * 0.02
            ha = 'left'
        else:
            x_pos = value - max_abs_val * 0.02
            ha = 'right'
        
        label = f'{value:.4f}\n(±{row.std_shap:.4f})'
        ax.text(x_pos, i, label, va='center', ha=ha, fontsize=10, fontweight='bold')
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'], fontsize=11)
    ax.set_xlabel('Mean SHAP Value (Impact on Risk Prediction)', fontsize=13, fontweight='bold')
    ax.set_title(f'{region_name} - {population_label}\n{region_info["description"]}',
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xlim([top_features['mean_shap'].min() - x_margin,
                 top_features['mean_shap'].max() + x_margin])
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.7, label='Increases Risk'),
        Patch(facecolor='#1f77b4', alpha=0.7, label='Decreases Risk')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.95)
    
    # Stats box
    stats_text = f"Region: {region_name}\n"
    stats_text += f"Datasets: {len(datasets)}\n"
    stats_text += f"Countries: {region_info['countries']}"
    
    stats_text += f"\nPopulation: {population_label}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(pad=2.0)
    plot_path = output_dir / f'{region_key}_{population_key}_shap_bar_top15.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Bar plot: {plot_path.name}")
    return plot_path


def _create_regional_violin_plot(dataset_shap_data, datasets, agg_df, region_name, 
                                 region_info, output_dir, region_key, population_key, population_label):
    """Create enhanced violin plot showing SHAP distribution across datasets in region"""
    top_features = agg_df.head(15)['feature'].tolist()

    violin_plot_data = []
    for feature in top_features:
        for dataset_id in datasets:
            df = dataset_shap_data[dataset_id]
            feature_row = df[df['feature'] == feature]
            if not feature_row.empty:
                shap_val = feature_row['mean_shap'].iloc[0]
                violin_plot_data.append({
                    'feature': feature,
                    'shap_value': shap_val,
                    'dataset_id': dataset_id
                })

    plot_df = pd.DataFrame(violin_plot_data)

    if not top_features or plot_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'No SHAP data available for {region_name}\nPopulation: {population_label}',
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax.axis('off')
        plot_path = output_dir / f'{region_key}_{population_key}_shap_violin_top15.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"  [WARN] Skipped violin plot for {region_name} ({population_label}) due to empty data")
        return plot_path

    selected_features = [feat for feat in top_features if feat in plot_df['feature'].unique()]
    if not selected_features:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'No SHAP data available for {region_name}\nPopulation: {population_label}',
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax.axis('off')
        plot_path = output_dir / f'{region_key}_{population_key}_shap_violin_top15.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"  [WARN] Skipped violin plot for {region_name} ({population_label}) due to empty data")
        return plot_path
    ordered_features = list(reversed(selected_features))
    plot_df = plot_df[plot_df['feature'].isin(ordered_features)].copy()
    plot_df['feature'] = pd.Categorical(plot_df['feature'], categories=ordered_features, ordered=True)

    fig, ax = plt.subplots(figsize=(16, 11))
    base_palette = ['#dfe7fd'] * len(ordered_features)
    feature_palette = {feat: base_palette[idx] for idx, feat in enumerate(ordered_features)}

    sns.violinplot(
        data=plot_df,
        y='feature',
        x='shap_value',
        hue='feature',
        palette=feature_palette,
        legend=False,
        order=ordered_features,
        inner=None,
        cut=0,
        scale='width',
        linewidth=0,
        ax=ax
    )

    norm = colors.TwoSlopeNorm(
        vcenter=0,
        vmin=plot_df['shap_value'].min(),
        vmax=plot_df['shap_value'].max()
    )
    cmap = plt.get_cmap('RdBu_r')

    feature_to_idx = {feat: idx for idx, feat in enumerate(ordered_features)}
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.12, 0.12, size=len(plot_df))
    y_positions = np.array([feature_to_idx[str(feat)] for feat in plot_df['feature']]) + jitter

    dataset_ids = sorted(plot_df['dataset_id'].unique())
    dataset_palette = sns.color_palette('husl', n_colors=len(dataset_ids)) if dataset_ids else []
    dataset_color_map = dict(zip(dataset_ids, dataset_palette))
    if dataset_color_map:
        outline_colors = plot_df['dataset_id'].map(dataset_color_map).fillna('#333333')
    else:
        outline_colors = '#333333'

    ax.scatter(
        plot_df['shap_value'],
        y_positions,
        c=cmap(norm(plot_df['shap_value'])),
        s=22,
        alpha=0.85,
        edgecolors=outline_colors,
        linewidths=0.6
    )

    ax.set_yticks(range(len(ordered_features)))
    ax.set_yticklabels(ordered_features)
    ax.axvline(0, color='#444444', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_title(f'{region_name} - {population_label}\nSHAP Value Distribution for Top {len(ordered_features)} Features',
                 fontsize=16, fontweight='bold', pad=18)
    ax.set_xlabel('SHAP Value (impact on risk prediction)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=13, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11)
    ax.set_facecolor('white')
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label('SHAP value (blue -> red)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    stats_text = (
        f"Region: {region_name}\n"
        f"Population: {population_label}\n"
        f"Datasets: {len(dataset_ids)}\n"
        f"Countries: {region_info['countries']}\n"
        f"Observations: {len(plot_df):,}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        va='top',
        ha='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    )

    if dataset_ids:
        dataset_handles = [
            Line2D(
                [0],
                [0],
                marker='o',
                linestyle='',
                markerfacecolor='white',
                markeredgecolor=dataset_color_map[ds],
                markeredgewidth=1.1,
                markersize=7,
                label=f'Dataset {ds}'
            )
            for ds in dataset_ids
        ]
        legend = ax.legend(
            handles=dataset_handles,
            title='Dataset ID (outline)',
            loc='lower right',
            frameon=True,
            fontsize=9,
            title_fontsize=10
        )
        legend.get_frame().set_alpha(0.9)

    ax.grid(axis='x', linestyle='--', alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=1.6)

    plot_path = output_dir / f'{region_key}_{population_key}_shap_violin_top15.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Enhanced violin plot: {plot_path.name}")
    return plot_path
