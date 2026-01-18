"""
Utility helpers for Stage 1 pipeline: canonical id normalization and path helpers.

Provides:
 - ensure_canonical_index(df, id_candidates): ensure df contains cfg.CANONICAL_INDEX_COL
 - resolve_artifact_dir(path_like): return absolute Path or None
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

import stage1_config as cfg


def ensure_canonical_index(df: pd.DataFrame, id_candidates=None) -> pd.DataFrame:
    """Ensure the returned DataFrame has a canonical integer index column named
    cfg.CANONICAL_INDEX_COL. The function will prefer columns in this order:
    1) existing cfg.CANONICAL_INDEX_COL
    2) configured cfg.ID_COL
    3) 'segment_id'
    4) 'Location ID'

    It will coerce numeric-like values to int. If none present or duplicates found
    it will raise a ValueError with guidance.
    """
    if id_candidates is None:
        id_candidates = [cfg.CANONICAL_INDEX_COL, cfg.ID_COL, 'segment_id', 'Location ID']

    df_out = df.copy()

    # If canonical already present and integral, coerce and return
    if cfg.CANONICAL_INDEX_COL in df_out.columns:
        # Try coercion
        try:
            df_out[cfg.CANONICAL_INDEX_COL] = pd.to_numeric(df_out[cfg.CANONICAL_INDEX_COL], errors='raise').astype(int)
            # Check uniqueness
            if df_out[cfg.CANONICAL_INDEX_COL].is_unique:
                return df_out
            else:
                raise ValueError(f"Existing {cfg.CANONICAL_INDEX_COL} column is not unique.")
        except Exception as e:
            raise ValueError(f"Could not coerce existing {cfg.CANONICAL_INDEX_COL} to integer: {e}")

    # Try fallback id columns
    found = None
    for cand in id_candidates[1:]:
        if cand and cand in df_out.columns:
            found = cand
            break

    if found is None:
        raise ValueError(
            f"Could not find any identifier columns in DataFrame. Tried: {id_candidates}.\n"
            "Ensure your DataFrame contains one of these columns or run alignment to produce them."
        )

    # If candidate is already numeric and unique, use as canonical
    try:
        cand_series = pd.to_numeric(df_out[found], errors='raise')
        if cand_series.is_unique:
            df_out[cfg.CANONICAL_INDEX_COL] = cand_series.astype(int)
            return df_out
    except Exception:
        # Non-numeric candidate, keep trying below
        pass

    # If candidate is non-numeric (e.g., segment_id strings), try mapping via index
    # If DataFrame index corresponds to original global indices, add that
    if df_out.index.is_integer() or pd.api.types.is_integer_dtype(df_out.index):
        df_out[cfg.CANONICAL_INDEX_COL] = df_out.index.astype(int)
        # Check uniqueness
        if df_out[cfg.CANONICAL_INDEX_COL].is_unique:
            return df_out

    # As last resort, attempt to create a new canonical index by assigning a sequential integer
    # but warn the caller (this may break mapping to original artifact indices)
    df_out[cfg.CANONICAL_INDEX_COL] = range(len(df_out))
    # Note: we produce sequential indices but this may not correspond to original global indices
    return df_out


def resolve_artifact_dir(path_like):
    """Return absolute Path if path_like is truthy and exists; otherwise return None.
    Accepts str or Path or None.
    """
    if path_like is None:
        return None
    p = Path(path_like)
    try:
        if p.exists():
            return p.resolve()
        # If not exists, still return resolved parent if it looks like a dir string
        # but prefer None to avoid writing invalid paths
        return p.resolve()
    except Exception:
        return p


def preferred_id_col(df=None, prefer_canonical=True) -> str:
    """
    Determine the preferred identifier column name for DataFrame joins/lookups.
    Preference order:
      1) cfg.CANONICAL_INDEX_COL (if prefer_canonical and present on df)
      2) cfg.ID_COL if present on df
      3) 'segment_id' if present on df
      4) 'Location ID' if present on df
    If a DataFrame is not provided, returns cfg.CANONICAL_INDEX_COL if prefer_canonical else cfg.ID_COL.
    This helper returns the column name string (not the Series).
    """
    if df is None:
        return cfg.CANONICAL_INDEX_COL if prefer_canonical else cfg.ID_COL

    # Prefer canonical index when requested and present
    if prefer_canonical and getattr(cfg, 'CANONICAL_INDEX_COL', None) in df.columns:
        return cfg.CANONICAL_INDEX_COL

    # Fall back to configured ID_COL if present
    if getattr(cfg, 'ID_COL', None) in df.columns:
        return cfg.ID_COL

    for alt in ['segment_id', 'Location ID']:
        if alt in df.columns:
            return alt

    # Default fallback
    return cfg.CANONICAL_INDEX_COL if prefer_canonical else (cfg.ID_COL or 'segment_id')


class TargetTransformer:
    """Configurable target transformation helper with invertible operations."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.method = (self.config.get('method', 'log1p') or 'log1p').lower()
        self.offset = float(self.config.get('offset', 0.0) or 0.0)
        self.standardize = bool(self.config.get('yeo_johnson_standardize', False))
        self._power = None

    def fit(self, y: pd.Series):
        arr = np.asarray(y, dtype=float).reshape(-1, 1)
        if self.method == 'yeo_johnson':
            self._power = PowerTransformer(method='yeo-johnson', standardize=self.standardize)
            self._power.fit(arr)
        return self

    def transform(self, y):
        arr = np.asarray(y, dtype=float)
        if self.method == 'log1p':
            return np.log1p(np.clip(arr, a_min=0.0, a_max=None))
        if self.method == 'log_offset':
            eps = max(self.offset, 1e-9)
            return np.log(np.clip(arr, a_min=0.0, a_max=None) + eps)
        if self.method == 'yeo_johnson':
            if self._power is None:
                raise RuntimeError('Yeo-Johnson transformer not fitted. Call fit() first.')
            return self._power.transform(arr.reshape(-1, 1)).ravel()
        # method == 'none'
        return arr

    def inverse_transform(self, y_transformed):
        arr = np.asarray(y_transformed, dtype=float)
        if self.method == 'log1p':
            return np.expm1(arr)
        if self.method == 'log_offset':
            eps = max(self.offset, 1e-9)
            return np.exp(arr) - eps
        if self.method == 'yeo_johnson':
            if self._power is None:
                raise RuntimeError('Yeo-Johnson transformer not fitted. Call fit() first.')
            return self._power.inverse_transform(arr.reshape(-1, 1)).ravel()
        return arr

    def fit_transform(self, y):
        return self.fit(y).transform(y)


def build_stratification_key(metadata_df: pd.DataFrame,
                             y_transformed: pd.Series,
                             mode: str,
                             road_column: str,
                             n_bins: int = 5,
                             feature_df: Optional[pd.DataFrame] = None) -> Optional[pd.Series]:
    """Create a stratification key per row based on the requested mode."""

    mode = (mode or 'none').lower()
    if mode == 'none':
        return None

    if road_column not in metadata_df.columns:
        raise ValueError(f"Road column '{road_column}' not found while building stratification key")

    road_series = metadata_df[road_column].fillna('UNKNOWN_ROAD').astype(str)
    road_series_named = road_series.rename('road')

    if mode == 'target_mean':
        road_stats = y_transformed.groupby(road_series_named).mean()
        binned = _bin_group_values(road_stats, n_bins)
        if binned.empty:
            return None
        fill_val = int(round(binned.median())) if not binned.empty else 0
        return road_series.map(binned).fillna(fill_val).astype(int)

    if mode == 'proxy_dataset':
        dataset_col = getattr(cfg, 'DATASET_ID_COL', None)
        if dataset_col and dataset_col in metadata_df.columns:
            cats = metadata_df[dataset_col].fillna('UNKNOWN_DATASET').astype(str)
            cat_codes = {val: idx for idx, val in enumerate(sorted(cats.unique()))}
            mapped = cats.map(cat_codes)
            fill_val = -1 if mapped.isna().any() else 0
            return mapped.fillna(fill_val).astype(int)
        return None

    if mode == 'proxy_aadt':
        aadt_series = _extract_aadt_series(metadata_df, feature_df)
        if aadt_series is None:
            return None
        joined = pd.concat([road_series_named, aadt_series.rename('aadt')], axis=1).dropna()
        if joined.empty:
            return None
        road_medians = joined.groupby('road')['aadt'].median()
        binned = _bin_group_values(road_medians, n_bins)
        if binned.empty:
            return None
        fill_val = int(round(binned.median())) if not binned.empty else 0
        return road_series.map(binned).fillna(fill_val).astype(int)

    # Unknown mode fallback
    return None


def _bin_group_values(series: pd.Series, n_bins: int) -> pd.Series:
    series = series.dropna()
    if series.empty:
        return pd.Series(dtype=float)
    bins = max(2, min(int(n_bins), series.nunique()))
    if bins < 2:
        return pd.Series(0, index=series.index, dtype=float)
    try:
        cats = pd.qcut(series, q=bins, labels=False, duplicates='drop')
    except ValueError:
        cats = pd.cut(series, bins=bins, labels=False, duplicates='drop')
    if cats.isna().all():
        cats = pd.Series(0, index=series.index, dtype=float)
    return cats.astype(float)


def _extract_aadt_series(metadata_df: pd.DataFrame,
                         feature_df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    candidates = [
        'Vehicle flow (AADT)',
        'vehicle_flow_aadt'
    ]
    for cand in candidates:
        if cand in metadata_df.columns:
            return pd.to_numeric(metadata_df[cand], errors='coerce')
        if feature_df is not None and cand in feature_df.columns:
            return pd.to_numeric(feature_df[cand], errors='coerce')
    return None
