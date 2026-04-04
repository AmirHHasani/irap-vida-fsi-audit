"""hotspot_countermeasure_utils.py

Step 4 utilities: integrate predicted hotspot OOF outputs with countermeasure
data and generate overlay classification (TP / FP / FN) plus coverage metrics.

Definitions:
  Predicted hotspot set (per road): Top-K by predicted log risk.
  Actual hotspot set (per road): Top-K by actual log risk.
  TP: segment in both sets.
  FP: segment only in predicted set.
  FN: segment only in actual set (missed hotspot).

Outputs:
  - Overlay dataframe (segment-level classification for predicted & missed hotspots).
  - Coverage metrics summarizing fraction of predicted hotspots w/ existing countermeasures.
  - Frequency table of countermeasure occurrence among TP / FP / FN categories.

"""
from __future__ import annotations
from typing import Tuple, Dict, Any
from stage1_utils import preferred_id_col
import pandas as pd
import numpy as np


def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure prediction columns exist on the dataframe:
      - pred_log, pred_linear, actual_log, actual_linear
    If only one representation exists, derive the other using np.log1p / np.expm1.
    If only generic names exist (predicted_risk / actual_risk) derive both from them.

    Returns a new DataFrame with the required columns present (may contain NaNs).
    """
    out = df.copy()
    # Predicted
    try:
        if 'pred_log' in out.columns:
            if 'pred_linear' not in out.columns:
                out['pred_linear'] = np.expm1(out['pred_log']).astype(float)
        elif 'pred_linear' in out.columns:
            out['pred_log'] = np.log1p(out['pred_linear']).astype(float)
        elif 'predicted_risk' in out.columns:
            # Assume 'predicted_risk' is linear by default
            out['pred_linear'] = out['predicted_risk'].astype(float)
            out['pred_log'] = np.log1p(out['predicted_risk']).astype(float)
        else:
            out['pred_log'] = np.nan
            out['pred_linear'] = np.nan
    except Exception:
        out['pred_log'] = out.get('pred_log', pd.Series(dtype=float))
        out['pred_linear'] = out.get('pred_linear', pd.Series(dtype=float))

    # Actual
    try:
        if 'actual_log' in out.columns:
            if 'actual_linear' not in out.columns:
                out['actual_linear'] = np.expm1(out['actual_log']).astype(float)
        elif 'actual_linear' in out.columns:
            out['actual_log'] = np.log1p(out['actual_linear']).astype(float)
        elif 'actual_risk' in out.columns:
            out['actual_linear'] = out['actual_risk'].astype(float)
            out['actual_log'] = np.log1p(out['actual_risk']).astype(float)
        else:
            out['actual_log'] = np.nan
            out['actual_linear'] = np.nan
    except Exception:
        out['actual_log'] = out.get('actual_log', pd.Series(dtype=float))
        out['actual_linear'] = out.get('actual_linear', pd.Series(dtype=float))

    return out

def build_hotspot_overlay(per_road_metrics_df: pd.DataFrame, 
                         oof_segments_df: pd.DataFrame,
                         id_col: str = None,
                         road_col: str = 'road_id') -> pd.DataFrame:
    """
    Build hotspot overlay by classifying segments as TP/FP/FN based on per-road metrics.
    
    Args:
        per_road_metrics_df: DataFrame with per-road hotspot metrics including pred_hotspots/actual_hotspots
        oof_segments_df: DataFrame with segment-level OOF predictions
        id_col: Column name for segment ID
        road_col: Column name for road ID
        
    Returns:
        DataFrame with segments classified as TP/FP/FN with coordinates
    """
    # Determine id_col to use: prefer canonical if not provided
    id_col_use = id_col if id_col is not None else preferred_id_col(oof_segments_df, prefer_canonical=True)
    print(f"[INFO] Building hotspot overlay from {len(per_road_metrics_df)} road metrics and {len(oof_segments_df)} segments using id_col={id_col_use}")

    # Normalize key dtypes up-front to avoid mismatched comparisons
    # Coerce OOF segment id and road columns to string
    oof_seg = oof_segments_df.copy()
    if id_col_use in oof_seg.columns:
        try:
            oof_seg[id_col_use] = oof_seg[id_col_use].astype(str)
        except Exception:
            oof_seg[id_col_use] = oof_seg[id_col_use].apply(lambda x: str(x) if pd.notna(x) else '')
    if 'segment_id' in oof_seg.columns:
        try:
            oof_seg['segment_id'] = oof_seg['segment_id'].astype(str)
        except Exception:
            oof_seg['segment_id'] = oof_seg['segment_id'].apply(lambda x: str(x) if pd.notna(x) else '')
    # Normalize/resolve road column to use for segments frame
    seg_road_col = road_col if road_col in oof_seg.columns else (
        'road_id' if 'road_id' in oof_seg.columns else (
            'Road name' if 'Road name' in oof_seg.columns else None))
    if seg_road_col is None:
        raise KeyError(f"Could not find road column '{road_col}' in oof_segments_df and no fallback like 'road_id'/'Road name' present. Columns: {list(oof_seg.columns)}")
    if seg_road_col in oof_seg.columns:
        try:
            oof_seg[seg_road_col] = oof_seg[seg_road_col].astype(str)
        except Exception:
            oof_seg[seg_road_col] = oof_seg[seg_road_col].apply(lambda x: str(x) if pd.notna(x) else '')
    
    overlay_rows = []
    
    # Process each road's metrics
    for _, road_row in per_road_metrics_df.iterrows():
        road_id = road_row['road_id']
        pred_hotspots = road_row.get('pred_hotspots', [])
        actual_hotspots = road_row.get('actual_hotspots', [])
        
        # Handle case where hotspots are stored as strings (CSV serialization)
        # Prefer JSON-encoded lists; be defensive and coerce other formats into lists.
        if isinstance(pred_hotspots, str):
            try:
                import json as _json
                parsed = _json.loads(pred_hotspots)
                pred_hotspots = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                # Fallback: try to parse simple comma-separated or bracketed forms
                try:
                    s = pred_hotspots.strip()
                    if s.startswith('[') and s.endswith(']'):
                        s_inner = s[1:-1].strip()
                        if not s_inner:
                            pred_hotspots = []
                        else:
                            pred_hotspots = [x.strip().strip("'\"") for x in s_inner.split(',')]
                    elif ',' in s:
                        pred_hotspots = [x.strip().strip("'\"") for x in s.split(',')]
                    else:
                        pred_hotspots = [s]
                except Exception:
                    pred_hotspots = []
        if isinstance(actual_hotspots, str):
            try:
                import json as _json
                parsed = _json.loads(actual_hotspots)
                actual_hotspots = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                try:
                    s = actual_hotspots.strip()
                    if s.startswith('[') and s.endswith(']'):
                        s_inner = s[1:-1].strip()
                        if not s_inner:
                            actual_hotspots = []
                        else:
                            actual_hotspots = [x.strip().strip("'\"") for x in s_inner.split(',')]
                    elif ',' in s:
                        actual_hotspots = [x.strip().strip("'\"") for x in s.split(',')]
                    else:
                        actual_hotspots = [s]
                except Exception:
                    actual_hotspots = []
        
        # Coerce predicted/actual hotspot IDs to strings for robust set membership tests
        pred_set = set([str(x) for x in pred_hotspots]) if pred_hotspots else set()
        actual_set = set([str(x) for x in actual_hotspots]) if actual_hotspots else set()

        # Get segments for this road (normalize road id to string for comparison)
        road_id_str = str(road_id)
        road_segments = oof_seg[oof_seg[seg_road_col] == road_id_str]

        for _, segment in road_segments.iterrows():
            segment_id_raw = segment[id_col_use] if id_col_use in segment else segment.get('segment_id')
            segment_id = str(segment_id_raw) if segment_id_raw is not None else ''

            # Classify segment
            if segment_id in pred_set and segment_id in actual_set:
                classification = 'TP'  # True Positive
            elif segment_id in pred_set and segment_id not in actual_set:
                classification = 'FP'  # False Positive
            elif segment_id not in pred_set and segment_id in actual_set:
                classification = 'FN'  # False Negative
            else:
                classification = None  # True Negative (not included in overlay)

            if classification:  # Only include TP/FP/FN in overlay
                overlay_rows.append({
                    'segment_id': segment_id,
                    'road_id': road_id,
                    'latitude': segment.get('latitude', None),
                    'longitude': segment.get('longitude', None),
                    'predicted_risk': segment.get('predicted_risk', None),
                    'actual_risk': segment.get('actual_risk', None),
                    'class': classification,
                    'fold_number': segment.get('fold_number', None)
                })
    
    overlay_df = pd.DataFrame(overlay_rows)
    print(f"[INFO] Created overlay with {len(overlay_df)} hotspot segments (TP/FP/FN)")
    
    if not overlay_df.empty:
        class_counts = overlay_df['class'].value_counts()
        print(f"[INFO] Classification breakdown: {class_counts.to_dict()}")
    
    return overlay_df

def integrate_countermeasures(overlay_df: pd.DataFrame,
                            countermeasure_df: pd.DataFrame,
                            id_col: str) -> pd.DataFrame:
    """
    Integrate countermeasure data with hotspot overlay.

    This implementation is defensive: if the configured detail column is missing
    it falls back to other likely columns (text column, any column containing
    'counter') and always returns a frame with a 'countermeasure' column.
    """
    import stage1_config as cfg

    print(f"[INFO] Integrating countermeasures with {len(overlay_df) if overlay_df is not None else 0} overlay segments")

    # Strict mode: do not silently fall back if countermeasure data or expected
    # columns are missing. Fail fast with clear diagnostics so upstream code
    # fixes the data source instead of hiding schema drift.
    if overlay_df is None or overlay_df.empty:
        raise RuntimeError("overlay_df must be a non-empty DataFrame when integrating countermeasures")

    # Ensure the configured CSV path and dataframe availability
    cm_path = getattr(cfg, 'COUNTERMEASURE_DATA_CSV', None)
    if cm_path is None:
        raise RuntimeError("COUNTERMEASURE_DATA_CSV not set in configuration")

    # If a dataframe wasn't passed in, try to load from configured path — but still fail if unreadable
    if countermeasure_df is None:
        try:
            countermeasure_df = pd.read_csv(cm_path)
        except Exception as e:
            raise RuntimeError(f"Could not read countermeasure CSV at {cm_path}: {e}")

    if countermeasure_df.empty:
        raise RuntimeError(f"Countermeasure dataframe loaded from {cm_path} is empty — cannot integrate")

    out = overlay_df.copy()
    # Determine id column if not provided
    if id_col is None:
        try:
            from stage1_utils import preferred_id_col as _pref
            id_col = _pref(out, prefer_canonical=True)
        except Exception:
            id_col = 'segment_id' if 'segment_id' in out.columns else None
    # ensure merged keys are strings to avoid dtype mismatch
    if id_col and (id_col in out.columns):
        try:
            out[id_col] = out[id_col].astype(str)
        except Exception:
            out[id_col] = out[id_col].apply(lambda x: str(x) if pd.notna(x) else '')
    # Also ensure a canonical 'segment_id' column exists for downstream code
    if 'segment_id' not in out.columns and id_col and id_col != 'segment_id':
        out['segment_id'] = out[id_col]
    # Force 'segment_id' column to string dtype for safe merge
    try:
        out['segment_id'] = out['segment_id'].astype(str)
    except Exception:
        out['segment_id'] = out['segment_id'].apply(lambda x: str(x) if pd.notna(x) else '')

    cm_copy = countermeasure_df.copy()

    # Determine the ID column present in countermeasure dataframe
    cm_id_col = getattr(cfg, 'COUNTERMEASURE_ID_COL', None)
    if cm_id_col not in cm_copy.columns:
        # try case-insensitive match
        for c in cm_copy.columns:
            if cm_id_col and c.lower() == cm_id_col.lower():
                cm_id_col = c
                break
    if cm_id_col not in cm_copy.columns:
        raise RuntimeError(f"Configured COUNTERMEASURE_ID_COL='{getattr(cfg,'COUNTERMEASURE_ID_COL',None)}' not found in countermeasure CSV columns: {list(cm_copy.columns)}")

    # Identify a detail/text column to use for countermeasure description
    # Require the configured text/detail column to be present (strict, no fallbacks)
    detail_col = getattr(cfg, 'COUNTERMEASURE_DETAIL_COL', None)
    text_col = getattr(cfg, 'COUNTERMEASURE_TEXT_COL', None)
    # Prefer detail_col if set, else text_col
    chosen = detail_col if detail_col else text_col
    if chosen not in cm_copy.columns:
        # try case-insensitive match for chosen
        for c in cm_copy.columns:
            if chosen and c.lower() == chosen.lower():
                chosen = c
                break
    if chosen not in cm_copy.columns:
        # also allow the text_col if detail_col wasn't present but text_col is available
        if text_col and text_col in cm_copy.columns:
            chosen = text_col
        else:
            raise RuntimeError(f"Neither configured COUNTERMEASURE_DETAIL_COL='{getattr(cfg,'COUNTERMEASURE_DETAIL_COL',None)}' nor COUNTERMEASURE_TEXT_COL='{getattr(cfg,'COUNTERMEASURE_TEXT_COL',None)}' found in countermeasure CSV columns: {list(cm_copy.columns)}")

    # Coerce id column to string if available
    # Coerce id column to string
    cm_copy[cm_id_col] = cm_copy[cm_id_col].astype(str)

    # Avoid duplicate-column clashes: if overlay already contains the cm_id_col name,
    # drop it before merging to prevent pandas from producing awkward duplicate columns
    if cm_id_col in out.columns and cm_id_col != 'segment_id':
        out = out.drop(columns=[cm_id_col])

    cm_subset = cm_copy[[c for c in [cm_id_col, chosen] if c in cm_copy.columns]].drop_duplicates()
    # Perform merge; left_on uses 'segment_id' canonical name
    merged = out.merge(cm_subset, left_on='segment_id', right_on=cm_id_col, how='left')
    # If the right key column was introduced by the merge, drop it to keep frame tidy
    if cm_id_col in merged.columns and cm_id_col != 'segment_id':
        merged = merged.drop(columns=[cm_id_col])
    # Normalize column name
    if chosen in merged.columns:
        merged.rename(columns={chosen: 'countermeasure'}, inplace=True)
    else:
        merged['countermeasure'] = None

    # Robust assignment: compute boolean Series and assign by values to avoid
    # potential index-alignment or multi-index assignment problems.
    if 'countermeasure' in merged.columns:
        present_series = merged['countermeasure'].notna()
    else:
        present_series = merged.iloc[:, -1].notna() if merged.shape[1] > 0 else pd.Series([False]*len(merged))

    if len(present_series) != len(merged):
        print(f"[ERROR] Length mismatch before assigning countermeasure_present: merged {len(merged)} vs present_series {len(present_series)}")
        # Fall back to filling False to avoid crashing further downstream
        merged['countermeasure_present'] = False
        found = 0
    else:
        merged['countermeasure_present'] = present_series.values
        found = int(merged['countermeasure_present'].sum())
    print(f"[INFO] Found countermeasures for {found}/{len(merged)} segments ({(found/len(merged)*100) if len(merged) else 0:.1f}%)")
    return merged

def compute_countermeasure_coverage(overlay_with_cm: pd.DataFrame) -> pd.DataFrame:
    """Compute coverage metrics:
      - pct_pred_hotspots_with_cm (TP+FP predicted segments having any countermeasure recorded)
      - pct_tp_with_cm
      - pct_fn_with_cm
    Returns per-road rows plus a global aggregate row with road_id='__GLOBAL__'.
    """
    if overlay_with_cm.empty:
        return pd.DataFrame()
    rows = []
    for road_id, grp in overlay_with_cm.groupby('road_id'):
        def pct(mask):
            sub = grp[mask]
            if sub.empty:
                return 0.0
            return float((sub['countermeasure'].notna()).mean())
        row = {
            'road_id': road_id,
            'pct_pred_hotspots_with_cm': pct(grp['class'].isin(['TP','FP'])),
            'pct_tp_with_cm': pct(grp['class'] == 'TP'),
            'pct_fn_with_cm': pct(grp['class'] == 'FN'),
            'num_tp': int((grp['class'] == 'TP').sum()),
            'num_fp': int((grp['class'] == 'FP').sum()),
            'num_fn': int((grp['class'] == 'FN').sum())
        }
        rows.append(row)
    df_cov = pd.DataFrame(rows)
    if not df_cov.empty:
        global_row = {
            'road_id': '__GLOBAL__',
            'pct_pred_hotspots_with_cm': float(df_cov['pct_pred_hotspots_with_cm'].mean()),
            'pct_tp_with_cm': float(df_cov['pct_tp_with_cm'].mean()),
            'pct_fn_with_cm': float(df_cov['pct_fn_with_cm'].mean()),
            'num_tp': int(df_cov['num_tp'].sum()),
            'num_fp': int(df_cov['num_fp'].sum()),
            'num_fn': int(df_cov['num_fn'].sum())
        }
        df_cov = pd.concat([df_cov, pd.DataFrame([global_row])], ignore_index=True)
    return df_cov


def countermeasure_occurrence_counts(overlay_with_cm: pd.DataFrame) -> pd.DataFrame:
    """Frequency of countermeasure labels by class (TP/FP/FN)."""
    if overlay_with_cm.empty or 'countermeasure' not in overlay_with_cm.columns:
        return pd.DataFrame()
    df = overlay_with_cm.copy()
    df['has_cm'] = df['countermeasure'].notna()
    freq = df[df['has_cm']].groupby(['class','countermeasure']).size().reset_index(name='count')
    return freq.sort_values('count', ascending=False)
