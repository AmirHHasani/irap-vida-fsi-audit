"""evaluation_utils.py

Extended hotspot ranking metrics utilities (self-contained copy for Stage 1).

Metrics: overlap@K, precision@K, recall@K, RR@K, DCG@K, nDCG@K, Spearman.
Also: Cohen's kappa (binary hot/not-hot) and bootstrap CIs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preferred_id_col(df: pd.DataFrame, *, prefer_canonical: bool = True) -> str:
    """Pick the best id column from a DataFrame."""
    if prefer_canonical and 'global_index' in df.columns:
        return 'global_index'
    if 'segment_id' in df.columns:
        return 'segment_id'
    if 'Location ID' in df.columns:
        return 'Location ID'
    return df.columns[0]


# ---------------------------------------------------------------------------
# Road-level ranking metrics
# ---------------------------------------------------------------------------

@dataclass
class RoadRankingResult:
    road_id: Any
    k: int
    overlap: int
    precision_at_k: float
    recall_at_k: float
    rr_at_k: float
    dcg_at_k: float
    ndcg_at_k: float
    spearman: float


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def discounted_cumulative_gain(binary_relevance: Sequence[int]) -> float:
    dcg = 0.0
    for i, rel in enumerate(binary_relevance):
        if rel:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def normalized_dcg(binary_relevance: Sequence[int]) -> float:
    actual_dcg = discounted_cumulative_gain(binary_relevance)
    ideal = sorted(binary_relevance, reverse=True)
    ideal_dcg = discounted_cumulative_gain(ideal)
    return _safe_div(actual_dcg, ideal_dcg)


def reciprocal_rank(pred_list: Sequence[Any], actual_set: set) -> float:
    for i, seg_id in enumerate(pred_list, start=1):
        if seg_id in actual_set:
            return 1.0 / i
    return 0.0


def compute_ranking_metrics_for_road(
    df_road: pd.DataFrame,
    k_list: Sequence[int],
    id_col: str | None = None,
    pred_col: str = 'predicted_risk',
    actual_col: str = 'actual_risk',
) -> List[RoadRankingResult]:
    results: List[RoadRankingResult] = []
    pred_sorted = df_road.sort_values(pred_col, ascending=False)
    actual_sorted = df_road.sort_values(actual_col, ascending=False)

    try:
        sp_corr, _ = spearmanr(pred_sorted[pred_col].values, actual_sorted[actual_col].values)
        if np.isnan(sp_corr):
            sp_corr = 0.0
    except Exception:
        sp_corr = 0.0

    id_col_use = id_col if id_col is not None else preferred_id_col(df_road, prefer_canonical=True)

    for k in k_list:
        k_eff = min(k, len(df_road))
        pred_top_ids = pred_sorted.head(k_eff)[id_col_use].tolist()
        actual_top_ids = actual_sorted.head(k_eff)[id_col_use].tolist()
        overlap = len(set(pred_top_ids) & set(actual_top_ids))
        precision_at_k = _safe_div(overlap, k_eff)
        recall_at_k = _safe_div(overlap, k_eff)
        rr = reciprocal_rank(pred_top_ids, set(actual_top_ids))
        relevance_binary = [1 if seg in actual_top_ids else 0 for seg in pred_top_ids]
        dcg = discounted_cumulative_gain(relevance_binary)
        ndcg = normalized_dcg(relevance_binary)
        results.append(RoadRankingResult(
            road_id=(df_road.iloc[0][df_road.columns.get_loc('Road name')]
                     if 'Road name' in df_road.columns else 'UNKNOWN'),
            k=k,
            overlap=overlap,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            rr_at_k=rr,
            dcg_at_k=dcg,
            ndcg_at_k=ndcg,
            spearman=sp_corr,
        ))
    return results


def ranking_results_to_long_df(results: List[RoadRankingResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def aggregate_ranking_metrics(long_df: pd.DataFrame) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for k, grp in long_df.groupby('k'):
        agg[str(k)] = {
            'mean_overlap': float(grp['overlap'].mean()),
            'mean_precision_at_k': float(grp['precision_at_k'].mean()),
            'mean_recall_at_k': float(grp['recall_at_k'].mean()),
            'mean_rr_at_k': float(grp['rr_at_k'].mean()),
            'mean_dcg_at_k': float(grp['dcg_at_k'].mean()),
            'mean_ndcg_at_k': float(grp['ndcg_at_k'].mean()),
            'mean_spearman': float(grp['spearman'].mean()),
        }
    agg['overall'] = {
        'MRR': float(long_df['rr_at_k'].mean()) if 'rr_at_k' in long_df else 0.0,
        'avg_spearman': float(long_df['spearman'].mean()) if 'spearman' in long_df else 0.0,
    }
    return agg


# ---------------------------------------------------------------------------
# Cohen's kappa for binary hotspot classification
# ---------------------------------------------------------------------------

def cohens_kappa_from_vectors(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Cohen's kappa for binary classification vectors (0/1)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    # Observed agreement
    po = np.mean(y_true == y_pred)
    # Expected agreement
    p_true_1 = np.mean(y_true == 1)
    p_pred_1 = np.mean(y_pred == 1)
    pe = p_true_1 * p_pred_1 + (1 - p_true_1) * (1 - p_pred_1)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return float((po - pe) / (1 - pe))


def compute_kappa_from_per_road_metrics(
    per_road_df: pd.DataFrame,
    k: int,
    id_col: str = 'segment_id',
    pred_col: str = 'predicted_risk',
    actual_col: str = 'actual_risk',
    road_col: str = 'Road name',
) -> float:
    """Compute Cohen's kappa across all roads for a given K.

    For each road, segments in the top-K by predicted risk are labelled 1
    (predicted hotspot) and likewise for actual risk (actual hotspot).
    Kappa is computed on the full segment-level binary vectors.
    """
    y_true_all = []
    y_pred_all = []

    for _, df_road in per_road_df.groupby(road_col):
        n = len(df_road)
        k_eff = min(k, n)
        if k_eff == 0:
            continue
        actual_top = set(df_road.nlargest(k_eff, actual_col)[id_col].tolist())
        pred_top = set(df_road.nlargest(k_eff, pred_col)[id_col].tolist())

        for seg_id in df_road[id_col]:
            y_true_all.append(1 if seg_id in actual_top else 0)
            y_pred_all.append(1 if seg_id in pred_top else 0)

    return cohens_kappa_from_vectors(np.array(y_true_all), np.array(y_pred_all))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _precompute_per_road_stats(
    per_road_df: pd.DataFrame,
    k: int,
    id_col: str,
    pred_col: str,
    actual_col: str,
    road_col: str,
) -> tuple:
    """Pre-compute per-road confusion-matrix counts and overlap ratios.

    Returns (road_ids, tp_arr, fp_arr, fn_arr, tn_arr, overlap_ratios)
    where each array is indexed by road position.  This avoids repeating
    expensive groupby/nlargest inside the bootstrap loop.
    """
    road_ids = []
    tp_list, fp_list, fn_list, tn_list = [], [], [], []
    overlap_list = []

    for road_id, df_r in per_road_df.groupby(road_col):
        n = len(df_r)
        k_eff = min(k, n)
        if k_eff == 0:
            continue
        actual_top = set(df_r.nlargest(k_eff, actual_col)[id_col].tolist())
        pred_top = set(df_r.nlargest(k_eff, pred_col)[id_col].tolist())
        all_ids = set(df_r[id_col].tolist())

        tp = len(pred_top & actual_top)
        fp = len(pred_top - actual_top)
        fn = len(actual_top - pred_top)
        tn = len(all_ids - pred_top - actual_top)

        road_ids.append(road_id)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        tn_list.append(tn)
        overlap_list.append(tp / k_eff)

    return (
        np.array(road_ids),
        np.array(tp_list, dtype=np.int64),
        np.array(fp_list, dtype=np.int64),
        np.array(fn_list, dtype=np.int64),
        np.array(tn_list, dtype=np.int64),
        np.array(overlap_list, dtype=np.float64),
    )


def _kappa_from_confusion(tp: int, fp: int, fn: int, tn: int) -> float:
    """Cohen's kappa from aggregated confusion-matrix counts."""
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    p_true1 = (tp + fn) / n
    p_pred1 = (tp + fp) / n
    pe = p_true1 * p_pred1 + (1 - p_true1) * (1 - p_pred1)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return float((po - pe) / (1 - pe))


def bootstrap_metric(
    per_road_df: pd.DataFrame,
    k: int,
    metric_fn: str = 'kappa',
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 42,
    id_col: str = 'segment_id',
    pred_col: str = 'predicted_risk',
    actual_col: str = 'actual_risk',
    road_col: str = 'Road name',
) -> Dict[str, float]:
    """Bootstrap roads to get CIs for kappa or mean overlap.

    Resamples *roads* (not individual segments) to respect the grouped
    structure.  Per-road confusion-matrix counts are pre-computed once and
    then resampled via integer indexing, making the loop ~100x faster than
    the naive pd.concat + groupby approach.

    Parameters
    ----------
    metric_fn : str
        'kappa' or 'overlap'.

    Returns
    -------
    dict with keys: point, ci_lower, ci_upper, se
    """
    # --- Pre-compute per-road statistics ONCE ---
    (_, tp_arr, fp_arr, fn_arr, tn_arr,
     overlap_arr) = _precompute_per_road_stats(
        per_road_df, k, id_col, pred_col, actual_col, road_col,
    )
    n_roads = len(tp_arr)
    if n_roads == 0:
        return {'point': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'se': 0.0}

    rng = np.random.default_rng(rng_seed)

    # --- Point estimate on original (un-resampled) data ---
    if metric_fn == 'kappa':
        point = _kappa_from_confusion(
            int(tp_arr.sum()), int(fp_arr.sum()),
            int(fn_arr.sum()), int(tn_arr.sum()),
        )
    elif metric_fn == 'overlap':
        point = float(overlap_arr.mean())
    else:
        raise ValueError(f"Unknown metric_fn: {metric_fn}")

    # --- Vectorised bootstrap (resample road indices only) ---
    # shape: (n_boot, n_roads)
    idx_matrix = rng.integers(0, n_roads, size=(n_boot, n_roads))

    if metric_fn == 'kappa':
        tp_boot = tp_arr[idx_matrix].sum(axis=1)
        fp_boot = fp_arr[idx_matrix].sum(axis=1)
        fn_boot = fn_arr[idx_matrix].sum(axis=1)
        tn_boot = tn_arr[idx_matrix].sum(axis=1)

        n_total = tp_boot + fp_boot + fn_boot + tn_boot
        po = np.where(n_total > 0, (tp_boot + tn_boot) / n_total, 0.0)
        p_true1 = np.where(n_total > 0, (tp_boot + fn_boot) / n_total, 0.0)
        p_pred1 = np.where(n_total > 0, (tp_boot + fp_boot) / n_total, 0.0)
        pe = p_true1 * p_pred1 + (1 - p_true1) * (1 - p_pred1)
        denom = 1.0 - pe
        boot_values = np.where(denom != 0, (po - pe) / denom, 0.0)
    else:  # overlap
        boot_values = overlap_arr[idx_matrix].mean(axis=1)

    lo = float(np.percentile(boot_values, 100 * alpha / 2))
    hi = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))
    se = float(np.std(boot_values, ddof=1))

    return {'point': point, 'ci_lower': lo, 'ci_upper': hi, 'se': se}
