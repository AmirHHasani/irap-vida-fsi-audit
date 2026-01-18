"""\
Stage 2: Hierarchical Causal Forest Pipeline
===========================================

This script trains EconML's CausalForestDML on *all* segments and then
exports segment/road/region results for the Stage 1 candidate hotspot set.

RUNNING GUIDE (recommended)
---------------------------

1) Baseline run (writes to a dedicated run folder)

    `conda activate stage2_env ; python stage2_hierarchical_cf.py --run-id 20260113_stage2_baseline_no_upgrade_cost`

     Outputs will be written under:
    `stage2_outputs/runs/<run-id>/...`

2) Sensitivity run examples (change only a few knobs)

     - Double causal-forest min_samples_leaf:
         `python stage2_hierarchical_cf.py --run-id 20260113_leaf_x2 --cf-min-samples-leaf 20`

     - Half causal-forest n_estimators:
         `python stage2_hierarchical_cf.py --run-id 20260113_cf_half --cf-n-estimators 1000`

     Each run writes a `stage2_run_metadata.json` containing the exact settings.

NOTES
-----
- This script expects Stage 1 outputs referenced in stage2_config.py.
- Runtime can be hours depending on settings and machine.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
import json
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

from stage2_treatment_codebook import apply_canonical_treatment_mapping, write_mapping_artifact
from stage2_treatment_codebook import get_spec

# Import configuration
from stage2_config import BINARY_TREATMENTS, ORDINAL_TREATMENTS, CONTROL_FEATURES


class DatasetStratifiedGroupKFold:
    """Road-grouped CV with Dataset-ID-balanced folds using scikit-learn.

    We want:
    - group integrity: all segments from a road are in the same fold
    - balanced survey composition: each fold has similar Dataset ID mix

    EconML passes the outcome as `y` into splitters; we *ignore* that `y` and
    instead stratify on the fixed `dataset_id` labels passed at initialization.
    """

    def __init__(
        self,
        n_splits: int,
        *,
        dataset_id: np.ndarray,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> None:
        from sklearn.model_selection import StratifiedGroupKFold

        self._dataset_id = np.asarray(dataset_id)
        self._inner = StratifiedGroupKFold(
            n_splits=int(n_splits),
            shuffle=bool(shuffle),
            random_state=int(random_state),
        )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self._inner.get_n_splits(X=X, y=y, groups=groups)

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups must be provided for DatasetStratifiedGroupKFold")
        if len(self._dataset_id) != len(groups):
            raise ValueError(
                f"dataset_id length ({len(self._dataset_id)}) must match groups length ({len(groups)})"
            )
        return self._inner.split(X, y=self._dataset_id, groups=groups)


def _default_stage2_output_root() -> Path:
    # Keep defaults backward compatible with stage2_config.OUTPUT_DIR when no CLI flags are provided.
    from stage2_config import OUTPUT_DIR

    return Path(OUTPUT_DIR)


def _runs_root() -> Path:
    return Path(__file__).resolve().parent / 'stage2_outputs' / 'runs'


def resolve_output_dir(*, output_dir: str | None, run_id: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    if run_id:
        return (_runs_root() / str(run_id)).resolve()
    return _default_stage2_output_root()


def write_run_metadata(*, output_dir: Path, args: argparse.Namespace) -> None:
    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(output_dir),
        "run_id": getattr(args, "run_id", None),
        "stage": "stage2_hierarchical_cf",
        "params": {
            "nuisance": {
                "n_estimators": args.nuisance_n_estimators,
                "max_depth": args.nuisance_max_depth,
                "min_samples_leaf": args.nuisance_min_samples_leaf,
                "max_features": "sqrt",
            },
            "causal_forest": {
                "n_estimators": args.cf_n_estimators,
                "max_depth": args.cf_max_depth,
                "min_samples_leaf": args.cf_min_samples_leaf,
                "mc_iters": args.cf_mc_iters,
                "inference": True,
                "cv": "GroupKFold(road_id)",
            },
        },
        "controls": {
            "n_controls": len(CONTROL_FEATURES),
            "contains_upgrade_cost": "Upgrade cost" in CONTROL_FEATURES,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'stage2_run_metadata.json').write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding='utf-8'
    )


class _Tee:
    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data):
        self._primary.write(data)
        self._secondary.write(data)

    def flush(self):
        self._primary.flush()
        self._secondary.flush()

    def isatty(self):
        return getattr(self._primary, "isatty", lambda: False)()


def _install_run_logger(output_dir: Path) -> Path:
    """Duplicate console output to a log file inside the run folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / 'stage2_console.log'
    log_fh = log_path.open('w', encoding='utf-8')

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, log_fh)
    sys.stderr = _Tee(old_err, log_fh)

    def _restore():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = old_out, old_err
        try:
            log_fh.flush()
            log_fh.close()
        except Exception:
            pass

    return log_path, _restore


def ensure_downstream_inputs(*, output_dir: Path) -> None:
    """Make the run folder self-contained for downstream scripts (Stage 2).

    Stage 2 prescriptions need `data/analysis_dataset.csv` to recover current treatment levels.
    We copy it from the default Stage 1 output root when running Stage 2 into a
    dedicated run folder.
    """
    try:
        src_root = _default_stage2_output_root()
        src = src_root / 'data' / 'analysis_dataset.csv'
        dst_dir = output_dir / 'data'
        dst = dst_dir / 'analysis_dataset.csv'
        if dst.exists():
            return
        if not src.exists():
            print(f"Warning: missing source analysis dataset: {src}")
            return
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied analysis dataset into run folder: {dst}")
    except Exception as exc:
        print(f"Warning: could not copy analysis_dataset.csv into run folder: {exc}")


def _write_outcome_spec(*, stage1_predictions: pd.DataFrame, output_dir: Path) -> None:
    """Persist outcome transformation metadata for downstream scripts.

    This prevents silent scale mismatches (e.g., applying linear thresholds to
    log-scale outcomes) by giving prescription generation a single source of truth.
    """
    from stage2_config import OUTCOME_EPSILON, OUTCOME_COL_ACTUAL, OUTCOME_COL_PREDICTED, TARGET_COL

    if OUTCOME_COL_ACTUAL not in stage1_predictions.columns:
        raise ValueError(f"Stage 1 predictions missing '{OUTCOME_COL_ACTUAL}'")

    actual = pd.to_numeric(stage1_predictions[OUTCOME_COL_ACTUAL], errors="coerce")
    # Heuristic for recording the transform used in Stage 1: negative values imply
    # log(FSI + offset) for small FSI.
    method = "log_offset" if (actual < 0).any() else "linear"

    spec = {
        "method": method,
        "offset": float(OUTCOME_EPSILON) if method == "log_offset" else 0.0,
        "outcome_col_actual": OUTCOME_COL_ACTUAL,
        "outcome_col_predicted": OUTCOME_COL_PREDICTED,
        "target_col_raw": TARGET_COL,
        "notes": "Stage 2 reads outcomes from Stage 1 OOF predictions; this file records the outcome scale for safe back-transforming.",
    }

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    outcome_path = data_dir / "outcome_spec.json"
    outcome_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  Outcome spec: {outcome_path}")

# ============================================================================
# STAGE A: DATA PREPARATION
# ============================================================================

def prepare_data(*, output_dir: Path):
    """Load and prepare data: train on ALL segments, report on hotspots."""
    print("\n" + "="*70)
    print("STAGE A: DATA PREPARATION")
    print("="*70)
    
    # Load Stage 1 hotspot identifications (from original Stage 1 output)
    from stage2_config import STAGE1_HOTSPOT_OVERLAY, STAGE1_OOF_PREDICTIONS, SEGMENTS_DATA_CSV
    stage1_hotspots = pd.read_csv(STAGE1_HOTSPOT_OVERLAY)
    print(f"Stage 1 hotspots overlay: {len(stage1_hotspots)} rows")
    
    # Load Stage 1 OOF predictions (predictions only, no features)
    stage1_predictions = pd.read_csv(STAGE1_OOF_PREDICTIONS, low_memory=False)
    print(f"Stage 1 predictions: {len(stage1_predictions):,} segments")

    # Persist outcome transformation metadata so downstream prescriptions cannot
    # silently mix log-scale and linear-scale rules.
    try:
        _write_outcome_spec(stage1_predictions=stage1_predictions, output_dir=output_dir)
    except Exception as e:
        print(f"Warning: could not write outcome spec metadata: {e}")
    
    # Load original data (has all features)
    original_data = pd.read_csv(SEGMENTS_DATA_CSV, low_memory=False)
    print(f"Original feature data: {len(original_data):,} segments")
    
    # Match original data with Stage 1 predictions using segment_id (same as Location ID)
    original_data['segment_id'] = original_data['Location ID']
    merge_columns = ['segment_id', 'predicted_risk', 'actual_risk']
    optional_columns = ['fold_number', 'road_id', 'road_canon']
    for col in optional_columns:
        if col in stage1_predictions.columns:
            merge_columns.append(col)
    stage1_subset = stage1_predictions[merge_columns].copy()
    
    analysis_data = original_data.merge(
        stage1_subset,
        on='segment_id',
        how='inner'
    )
    print(f"Merged dataset: {len(analysis_data):,} segments with features + predictions")
    
    # Merge hotspot metadata (class = TP/FP/FN from Stage 1 validation)
    if 'class' in stage1_hotspots.columns and 'segment_id' in stage1_hotspots.columns:
        analysis_data = analysis_data.merge(
            stage1_hotspots[['segment_id', 'class']].rename(columns={'class': 'hotspot_class'}),
            on='segment_id',
            how='left'
        )
    
    # Mark which segments are in the Stage 1 hotspot overlay (TP/FP/FN ledger)
    # NOTE: This is NOT the same as the *predicted candidate hotspot set* in the paper.
    hotspot_segment_ids = (
        stage1_hotspots['segment_id'].unique()
        if 'segment_id' in stage1_hotspots.columns
        else stage1_hotspots['Location ID'].unique()
    )
    analysis_data['is_hotspot'] = (
        analysis_data['segment_id'].isin(hotspot_segment_ids)
        if 'segment_id' in analysis_data.columns
        else analysis_data['Location ID'].isin(hotspot_segment_ids)
    )

    # Candidate hotspots for reporting/prescriptions (Stage 1 predicted hotspots): TP + FP
    if 'hotspot_class' in analysis_data.columns:
        analysis_data['is_candidate_hotspot'] = analysis_data['hotspot_class'].isin(['TP', 'FP'])
    else:
        analysis_data['is_candidate_hotspot'] = analysis_data['is_hotspot']
        print("Warning: 'hotspot_class' missing; treating overlay hotspots as candidate hotspots.")

    n_overlay = int(analysis_data['is_hotspot'].sum())
    n_candidates = int(analysis_data['is_candidate_hotspot'].sum())
    print(f"Hotspot overlay marked: {n_overlay:,} segments")
    print(f"Candidate hotspots (TP+FP): {n_candidates:,} segments")
    if 'hotspot_class' in analysis_data.columns:
        print(f"  TP (True Positive): {(analysis_data['hotspot_class'] == 'TP').sum()}")
        print(f"  FP (False Positive): {(analysis_data['hotspot_class'] == 'FP').sum()}")
        print(f"  FN (False Negative): {(analysis_data['hotspot_class'] == 'FN').sum()}")
    
    print(f"\nSTRATEGY: Train on ALL {len(analysis_data):,} segments (maximum power)")
    print(f"RESULTS: Report on {n_candidates:,} candidate hotspots (TP+FP) + aggregations at all levels")
    
    # Add country and regional mapping
    try:
        mapping = pd.read_csv('dataset_regional_mapping.csv')
        analysis_data = analysis_data.merge(
            mapping[['Dataset ID', 'Country Name', 'Country Code', 'Region Name']],
            on='Dataset ID',
            how='left'
        )
        print(f"Countries: {analysis_data['Country Name'].nunique()}")
        print(f"Regions: {analysis_data['Region Name'].nunique()}")
    except Exception as e:
        print(f"Warning: No regional mapping found: {e}")
        analysis_data['Country Name'] = analysis_data['Dataset ID'].astype(str)
        analysis_data['Region Name'] = None
    
    # Use Region Name from mapping file (not hardcoded)
    analysis_data['Region'] = analysis_data['Region Name']
    
    # Normalize road names to avoid duplicate IDs (e.g., "A2" vs "A2 ")
    def normalize_road_name(series):
        cleaned = series.fillna('').astype(str).str.strip()
        cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)
        cleaned = cleaned.str.replace('.0', '', regex=False)
        cleaned = cleaned.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        return cleaned
    
    if 'Road name' in analysis_data.columns:
        analysis_data['Road name'] = normalize_road_name(analysis_data['Road name'])
    if 'road_canon' in analysis_data.columns:
        analysis_data['road_canon'] = normalize_road_name(analysis_data['road_canon'])
    
    dataset_ids_clean = analysis_data['Dataset ID'].astype(str).str.strip()
    if 'road_id' in analysis_data.columns:
        analysis_data['road_id'] = normalize_road_name(analysis_data['road_id'])
    else:
        road_source = 'road_canon' if 'road_canon' in analysis_data.columns and analysis_data['road_canon'].notna().any() else 'Road name'
        analysis_data['__road_tmp'] = normalize_road_name(analysis_data[road_source])
        analysis_data['road_id'] = dataset_ids_clean + "_" + analysis_data['__road_tmp'].fillna('unknown')
        analysis_data['Road name'] = analysis_data['__road_tmp']
        analysis_data.drop(columns=['__road_tmp'], inplace=True)
    
    # Harmonize display name regardless of road_id source
    if 'road_canon' in analysis_data.columns and analysis_data['road_canon'].notna().any():
        analysis_data['Road name'] = analysis_data['road_canon']
    
    print(f"Regions: {analysis_data['Region'].nunique()}")
    print(f"Unique roads: {analysis_data['road_id'].nunique()}")
    
    # Create Dataset ID dummy variables
    dataset_dummies = pd.get_dummies(
        dataset_ids_clean,
        prefix='dataset',
        drop_first=False  # Keep all for interpretability
    )
    
    # Prepare feature matrix (ALL segments)
    available_controls = [
        col for col in CONTROL_FEATURES 
        if col in analysis_data.columns and col != 'Dataset ID'
    ]
    missing_controls = sorted(set(CONTROL_FEATURES) - set(available_controls) - {'Dataset ID'})
    if missing_controls:
        print(f"Warning: Missing control features: {missing_controls[:5]}{'...' if len(missing_controls) > 5 else ''}")
    X_controls = analysis_data[available_controls].copy().fillna(0)
    
    # Ensure all controls are numeric (convert any categorical to dummies if needed)
    X_controls_numeric = pd.DataFrame(index=analysis_data.index)
    for col in X_controls.columns:
        if X_controls[col].dtype == 'object' or X_controls[col].dtype.name == 'category':
            # Convert categorical to dummies
            dummies = pd.get_dummies(X_controls[col], prefix=col, drop_first=False)
            X_controls_numeric = pd.concat([X_controls_numeric, dummies], axis=1)
        else:
            X_controls_numeric[col] = pd.to_numeric(X_controls[col], errors='coerce')
    
    X_features = pd.concat([X_controls_numeric, dataset_dummies], axis=1)
    
    # Convert to pure float to avoid sklearn categorical issues
    X_features = X_features.astype(float)
    
    print(f"\nFeature summary:")
    print(f"  Configured controls (incl Dataset ID): {len(CONTROL_FEATURES)}")
    print(f"  Controls used (excluding Dataset ID): {len(available_controls)}")
    print(f"  Control columns after one-hot: {X_controls_numeric.shape[1]}")
    print(f"  Dataset dummies (Dataset ID one-hot): {dataset_dummies.shape[1]}")
    print(f"  Total X features: {X_features.shape[1]}")
    
    # Get viable treatments
    viable_treatments = BINARY_TREATMENTS + ORDINAL_TREATMENTS

    # Canonical treatment mapping (centralized): ensures "higher = better" and
    # prevents silent ordering bugs from raw numeric codes.
    mapping_report = apply_canonical_treatment_mapping(
        analysis_data,
        viable_treatments,
        strict=True,
        keep_raw=False,
    )
    mapped = list(mapping_report.get("treatments", {}).keys())
    if mapped:
        print(f"  Treatments canonical-mapped: {len(mapped)}")
    print(f"  Treatments: {len(viable_treatments)}")

    # Persist mapping metadata early so downstream stages can confirm consistency.
    try:
        data_dir = output_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = data_dir / 'treatment_level_mapping_used.json'
        write_mapping_artifact(mapping_report, mapping_path)
        print(f"  Mapping metadata: {mapping_path}")
    except Exception as e:
        print(f"  Warning: could not write mapping metadata: {e}")
    
    # Sample sizes by country (all data)
    print(f"\nData distribution by country:")
    country_counts = analysis_data['Country Name'].value_counts()
    for country, count in country_counts.items():
        overlay_count = analysis_data[(analysis_data['Country Name'] == country) & (analysis_data['is_hotspot'])].shape[0]
        candidate_count = analysis_data[(analysis_data['Country Name'] == country) & (analysis_data['is_candidate_hotspot'])].shape[0]
        print(f"  {country}: {count:,} total ({candidate_count} candidates; {overlay_count} overlay)")
    
    return analysis_data, X_features, viable_treatments


# ============================================================================
# PHASE B: CAUSAL FOREST ESTIMATION
# ============================================================================

def fit_causal_forest(data, X_features, treatment, *, params: dict | None = None, verbose=True):
    """
    Fit Causal Forest DML for a single treatment on ALL data.
    
    Args:
        data: Full dataset (all segments)
        X_features: Feature matrix (all segments)
        treatment: Treatment name
        verbose: Print progress
    
    Returns:
        model: Fitted CausalForestDML object
        contrast_results: List of dicts; one entry per explicit contrast with
            keys: name, treatment_base, t0, t1, contrast_type, baseline_level,
            levels, CATE_raw, CATE_ci_lower, CATE_ci_upper.
    """
    if verbose:
        print(f"    Preparing data...", end=' ')
    
    # Extract treatment and outcome
    Y = data['actual_risk'].values
    T = data[treatment].values
    X = X_features.values
    groups = data['road_id'].values  # For road-grouped cross-fitting
    
    # Remove missing values
    valid_mask = ~(pd.isna(Y) | pd.isna(T))
    Y_clean = Y[valid_mask]
    T_clean = T[valid_mask]
    X_clean = X[valid_mask]
    groups_clean = groups[valid_mask]

    # For discrete treatments, ensure integer-coded categories when possible.
    # This aligns with EconML's discrete_treatment handling (one-hot with dropped
    # baseline defined by lexicographic order of the categories).
    if np.all(np.isclose(T_clean, np.round(T_clean))):
        T_clean = np.round(T_clean).astype(int)
    
    # Check treatment variation
    unique_levels = np.unique(T_clean)
    n_unique = len(unique_levels)
    if n_unique < 2:
        if verbose:
            print(f"X No variation")
        return None, []
    
    # This pipeline treats both binary and ordinal treatments as discrete.
    # For ordinal treatments, we report *adjacent-level* contrasts using
    # effect(X, T0, T1) / effect_interval(X, T0, T1) to keep the estimand
    # unambiguous and persistent in outputs.
    is_binary = n_unique == 2
    is_discrete = True
    
    n_roads = len(np.unique(groups_clean))
    if n_roads < 2:
        if verbose:
            print("X Too few roads for grouped CV")
        return None, []

    if verbose:
        treatment_type = "binary" if is_binary else "ordinal"
        print(f"OK (n={len(Y_clean):,}, type={treatment_type}, values={n_unique}, roads={n_roads})")
        print(f"    Fitting Causal Forest DML (road-grouped CV)...", end=' ', flush=True)
    
    # Road-grouped cross-fitting, balanced by Dataset ID (survey composition)
    n_splits = min(5, n_roads)
    if 'Dataset ID' not in data.columns:
        raise KeyError("Dataset ID column is required to balance folds by survey composition")
    dataset_id_clean = data.loc[valid_mask, 'Dataset ID'].to_numpy()
    cv_splitter = DatasetStratifiedGroupKFold(
        n_splits=n_splits,
        dataset_id=dataset_id_clean,
        shuffle=True,
        random_state=42,
    )

    def _as_1d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[:, 0]
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D effect array, got shape {arr.shape}")
        return arr

    # Determine ordered treatment levels and explicit contrasts.
    levels_sorted = sorted(float(x) for x in unique_levels)
    spec = get_spec(treatment)
    if getattr(spec, "canonical_order", None):
        canonical_levels = [float(x) for x in spec.canonical_order]
        ordered_levels = [lvl for lvl in canonical_levels if lvl in set(levels_sorted)]
        if len(ordered_levels) < 2:
            ordered_levels = levels_sorted
    else:
        ordered_levels = levels_sorted

    baseline_level = min(ordered_levels)
    if is_binary:
        contrasts = [(ordered_levels[0], ordered_levels[1], "binary")]
    else:
        contrasts = [(ordered_levels[i], ordered_levels[i + 1], "adjacent") for i in range(len(ordered_levels) - 1)]

    if verbose and not is_binary:
        print(f"\n    Ordinal contrasts: {', '.join([f'{int(a)}→{int(b)}' for a, b, _ in contrasts])}")
        print(
            f"    Note: EconML encodes multi-valued discrete treatments internally with a dropped baseline; "
            f"baseline={baseline_level} (lexicographically smallest)."
        )

    params = params or {}
    nuisance_params = params.get("nuisance", {})
    forest_params = params.get("causal_forest", {})

    nuisance_n_estimators = int(nuisance_params.get("n_estimators", 500))
    nuisance_max_depth = int(nuisance_params.get("max_depth", 10))
    nuisance_min_samples_leaf = int(nuisance_params.get("min_samples_leaf", 5))

    cf_n_estimators = int(forest_params.get("n_estimators", 2000))
    cf_max_depth = int(forest_params.get("max_depth", 8))
    cf_min_samples_leaf = int(forest_params.get("min_samples_leaf", 10))
    cf_mc_iters = int(forest_params.get("mc_iters", 4))

    # Treatment model (propensity):
    # For discrete treatments, EconML expects a classifier-like model_t.
    model_t = RandomForestClassifier(
            n_estimators=nuisance_n_estimators,
            max_depth=nuisance_max_depth,
            min_samples_leaf=nuisance_min_samples_leaf,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    
    # Initialize Causal Forest
    cf_model = CausalForestDML(
        # Nuisance models (g(X) and e(X))
        model_y=RandomForestRegressor(
            n_estimators=nuisance_n_estimators,
            max_depth=nuisance_max_depth,
            min_samples_leaf=nuisance_min_samples_leaf,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        model_t=model_t,
        # Treat both binary and ordinal as discrete (multi-valued supported).
        discrete_treatment=is_discrete,
        
        # Causal forest parameters (tau(X) estimation)
        n_estimators=cf_n_estimators,          # Large forest for stability
        max_depth=cf_max_depth,                # Captures interactions
        min_samples_leaf=cf_min_samples_leaf,  # Prevents overfitting
        min_var_fraction_leaf=0.1,  # Honest inference (honesty parameter)
        min_var_leaf_on_val=True,   # Enable honesty
        
        # Cross-fitting (road-grouped for independence)
        cv=cv_splitter,             # Road-grouped K-fold
        mc_iters=cf_mc_iters,       # Monte Carlo iterations
        
        # Inference
        inference=True,
        random_state=42,
        n_jobs=-1
    )
    
    try:
        # Fit model on ALL data (with road-grouped cross-fitting)
        cf_model.fit(
            Y=Y_clean, 
            T=T_clean, 
            X=X_clean,
            W=None,  # All features can moderate effects
            groups=groups_clean  # Road groups for cross-fitting
        )
        
        if verbose:
            print("OK")
            print(f"    Estimating segment-specific effects...", end=' ', flush=True)

        contrast_results = []
        for t0, t1, contrast_type in contrasts:
            segment_cates_valid = _as_1d(cf_model.effect(X_clean, T0=t0, T1=t1))

            # Get confidence intervals
            try:
                ci_valid = cf_model.effect_interval(X_clean, T0=t0, T1=t1, alpha=0.05)
                ci_lower_valid = _as_1d(ci_valid[0])
                ci_upper_valid = _as_1d(ci_valid[1])
            except Exception:
                if verbose:
                    print(f"\n    Warning: CI estimation failed for {t0}->{t1}; using bootstrap (500 iterations)...")
                bootstrap_cates = []
                rng = np.random.default_rng(42)
                for _ in range(500):
                    idx = rng.choice(len(X_clean), len(X_clean), replace=True)
                    bootstrap_cates.append(_as_1d(cf_model.effect(X_clean[idx], T0=t0, T1=t1)))
                ci_lower_valid = np.percentile(bootstrap_cates, 2.5, axis=0)
                ci_upper_valid = np.percentile(bootstrap_cates, 97.5, axis=0)

            # Map back to full dataset (including invalid rows)
            segment_cates = np.full(len(data), np.nan, dtype=float)
            segment_cates[valid_mask] = segment_cates_valid

            ci_lower = np.full(len(data), np.nan, dtype=float)
            ci_upper = np.full(len(data), np.nan, dtype=float)
            ci_lower[valid_mask] = ci_lower_valid
            ci_upper[valid_mask] = ci_upper_valid

            contrast_label = f"{treatment} [{int(t0)}→{int(t1)}]"
            contrast_name = f"{treatment}__{int(t0)}_to_{int(t1)}"
            contrast_results.append(
                {
                    "name": contrast_name,
                    "label": contrast_label,
                    "treatment_base": treatment,
                    "t0": float(t0),
                    "t1": float(t1),
                    "contrast_type": contrast_type,
                    "baseline_level": float(baseline_level),
                    "levels": ordered_levels,
                    "CATE_raw": segment_cates,
                    "CATE_ci_lower": ci_lower,
                    "CATE_ci_upper": ci_upper,
                }
            )

            if verbose:
                print(f"\n    {contrast_label}: mean={np.nanmean(segment_cates):+.4f}, std={np.nanstd(segment_cates):.4f}")

        if verbose:
            print("OK")

        return cf_model, contrast_results
        
    except Exception as e:
        if verbose:
            print(f"X Failed: {str(e)[:50]}")
        return None, []


# ============================================================================
# PHASE C: COUNTRY-LEVEL AGGREGATION
# ============================================================================

def aggregate_by_country(data, all_results):
    """Aggregate CATEs to country level with dual statistics (all segments + hotspots)."""
    country_summaries = []
    
    for country in data['Country Name'].unique():
        country_mask = (data['Country Name'] == country)
        country_data = data[country_mask]
        country_hotspots = country_data[country_data['is_candidate_hotspot']]
        
        for treatment, results in all_results.items():
            if results['CATE_shrunk'] is None:
                continue

            treatment_base = results.get('treatment_base', treatment)
            t0 = results.get('t0', np.nan)
            t1 = results.get('t1', np.nan)
            contrast_type = results.get('contrast_type', None)
            
            # All segments
            country_indices = country_data.index.values
            country_cates = results['CATE_shrunk'][country_indices]
            valid_cates = country_cates[~np.isnan(country_cates)]
            
            if len(valid_cates) == 0:
                continue
            
            # Hotspots
            hotspot_indices = country_hotspots.index.values
            hotspot_cates = results['CATE_shrunk'][hotspot_indices]
            valid_hotspot_cates = hotspot_cates[~np.isnan(hotspot_cates)]
            
            country_summaries.append({
                'country': country,
                'region': country_data.iloc[0]['Region'],
                'treatment': treatment_base,
                'contrast': treatment,
                't0': t0,
                't1': t1,
                'contrast_type': contrast_type,
                
                # ALL SEGMENTS
                'n_segments_total': len(country_data),
                'n_segments_valid': len(valid_cates),
                'CATE_mean_all': valid_cates.mean(),
                'CATE_median_all': np.median(valid_cates),
                'CATE_std_all': valid_cates.std(),
                
                # HOTSPOTS
                'n_hotspots': len(country_hotspots),
                'n_hotspots_valid': len(valid_hotspot_cates),
                'CATE_mean_hotspots': valid_hotspot_cates.mean() if len(valid_hotspot_cates) > 0 else np.nan,
                'CATE_min_hotspots': valid_hotspot_cates.min() if len(valid_hotspot_cates) > 0 else np.nan,
                
                # EFFECT SIZE
                'percent_negative_all': (valid_cates < 0).mean() * 100,
                'percent_strong_effect_hotspots': (
                    (valid_hotspot_cates < -0.10).mean() * 100 
                    if len(valid_hotspot_cates) > 0 else np.nan
                )
            })
    
    return pd.DataFrame(country_summaries)


def aggregate_by_road(data, all_results):
    """Aggregate CATEs to road level with dual statistics (all segments + hotspots)."""
    road_summaries = []
    
    for road_id in data['road_id'].unique():
        road_mask = (data['road_id'] == road_id)
        road_data = data[road_mask]
        road_hotspots = road_data[road_data['is_candidate_hotspot']]
        
        for treatment, results in all_results.items():
            if results['CATE_shrunk'] is None:
                continue

            treatment_base = results.get('treatment_base', treatment)
            t0 = results.get('t0', np.nan)
            t1 = results.get('t1', np.nan)
            contrast_type = results.get('contrast_type', None)
            
            # All segments in road
            road_indices = road_data.index.values
            road_cates = results['CATE_shrunk'][road_indices]
            valid_cates = road_cates[~np.isnan(road_cates)]
            
            if len(valid_cates) == 0:
                continue
            
            # Hotspot segments in road
            hotspot_indices = road_hotspots.index.values
            hotspot_cates = results['CATE_shrunk'][hotspot_indices]
            valid_hotspot_cates = hotspot_cates[~np.isnan(hotspot_cates)]
            
            road_summaries.append({
                'road_id': road_id,
                'road_name': road_data.iloc[0]['Road name'],
                'dataset_id': road_data.iloc[0]['Dataset ID'],
                'country': road_data.iloc[0]['Country Name'],
                'region': road_data.iloc[0]['Region'],
                'treatment': treatment_base,
                'contrast': treatment,
                't0': t0,
                't1': t1,
                'contrast_type': contrast_type,
                
                # ALL SEGMENTS stats (complete picture)
                'n_segments_total': len(road_data),
                'n_segments_valid': len(valid_cates),
                'CATE_mean_all_segments': valid_cates.mean(),
                'CATE_median_all_segments': np.median(valid_cates),
                'CATE_std_all_segments': valid_cates.std(),
                
                # HOTSPOT stats (high-risk focus)
                'n_hotspots': len(road_hotspots),
                'n_hotspots_valid': len(valid_hotspot_cates),
                'CATE_mean_hotspots': valid_hotspot_cates.mean() if len(valid_hotspot_cates) > 0 else np.nan,
                'CATE_median_hotspots': np.median(valid_hotspot_cates) if len(valid_hotspot_cates) > 0 else np.nan,
                'CATE_min_hotspots': valid_hotspot_cates.min() if len(valid_hotspot_cates) > 0 else np.nan,
                
                # CONTEXT metrics
                'hotspot_percentage': (len(road_hotspots) / len(road_data) * 100) if len(road_data) > 0 else 0,
                'avg_predicted_risk_all': road_data['predicted_risk'].mean(),
                'avg_predicted_risk_hotspots': road_hotspots['predicted_risk'].mean() if len(road_hotspots) > 0 else np.nan,
                'risk_concentration_ratio': (
                    road_hotspots['predicted_risk'].mean() / road_data['predicted_risk'].mean()
                    if len(road_hotspots) > 0 and road_data['predicted_risk'].mean() > 0 else 1.0
                )
            })
    
    return pd.DataFrame(road_summaries)


def aggregate_by_region(data, all_results):
    """Aggregate CATEs to 3 macro-regions with dual statistics."""
    regional_summaries = []
    
    for region in data['Region'].dropna().unique():
        region_mask = (data['Region'] == region)
        region_data = data[region_mask]
        region_hotspots = region_data[region_data['is_candidate_hotspot']]
        
        for treatment, results in all_results.items():
            if results['CATE_shrunk'] is None:
                continue

            treatment_base = results.get('treatment_base', treatment)
            t0 = results.get('t0', np.nan)
            t1 = results.get('t1', np.nan)
            contrast_type = results.get('contrast_type', None)
            
            region_indices = region_data.index.values
            region_cates = results['CATE_shrunk'][region_indices]
            valid_cates = region_cates[~np.isnan(region_cates)]
            
            if len(valid_cates) == 0:
                continue
            
            # Hotspot CATEs
            hotspot_indices = region_hotspots.index.values
            hotspot_cates = results['CATE_shrunk'][hotspot_indices]
            valid_hotspot_cates = hotspot_cates[~np.isnan(hotspot_cates)]
            
            regional_summaries.append({
                'region': region,
                'treatment': treatment_base,
                'contrast': treatment,
                't0': t0,
                't1': t1,
                'contrast_type': contrast_type,
                
                # SAMPLE SIZE
                'n_segments_total': len(region_data),
                'n_segments_valid': len(valid_cates),
                'n_hotspots': len(region_hotspots),
                'n_hotspots_valid': len(valid_hotspot_cates),
                'n_countries': region_data['Country Name'].nunique(),
                'n_roads': region_data['road_id'].nunique(),
                
                # ALL SEGMENTS - CATE distribution
                'CATE_mean_all': valid_cates.mean(),
                'CATE_median_all': np.median(valid_cates),
                'CATE_std_all': valid_cates.std(),
                'CATE_p5_all': np.percentile(valid_cates, 5),
                'CATE_p95_all': np.percentile(valid_cates, 95),
                
                # HOTSPOTS - CATE distribution
                'CATE_mean_hotspots': valid_hotspot_cates.mean() if len(valid_hotspot_cates) > 0 else np.nan,
                'CATE_median_hotspots': np.median(valid_hotspot_cates) if len(valid_hotspot_cates) > 0 else np.nan,
                'CATE_min_hotspots': valid_hotspot_cates.min() if len(valid_hotspot_cates) > 0 else np.nan,
                
                # EFFECT MAGNITUDE (for paper Table 5)
                'percent_negative_all': (valid_cates < 0).mean() * 100,
                'percent_strong_effect_all': (valid_cates < -0.10).mean() * 100,
                'percent_moderate_effect_all': ((valid_cates < -0.05) & (valid_cates >= -0.10)).mean() * 100,
                'percent_strong_effect_hotspots': (
                    (valid_hotspot_cates < -0.10).mean() * 100 
                    if len(valid_hotspot_cates) > 0 else np.nan
                ),
                
                # RISK CONTEXT
                'avg_predicted_risk_all': region_data['predicted_risk'].mean(),
                'avg_actual_risk_all': region_data['actual_risk'].mean(),
                'avg_predicted_risk_hotspots': region_hotspots['predicted_risk'].mean() if len(region_hotspots) > 0 else np.nan
            })
    
    return pd.DataFrame(regional_summaries)


# ============================================================================
# PHASE D: HIERARCHICAL SHRINKAGE
# ============================================================================

def apply_road_level_shrinkage(data, segment_cates_full, shrinkage_k=20):
    """
    Apply road-level empirical Bayes shrinkage to stabilize estimates.
    
    Formula: weight = n_road / (n_road + k)
    Shrinkage target: Road-level means toward global mean
    
    Args:
        data: Full DataFrame (all segments)
        segment_cates_full: Full array of CATEs for ALL segments (147K)
        shrinkage_k: Shrinkage parameter (default 20)
    
    Returns:
        shrunken_cates_full: Full array with shrunk CATEs
        shrinkage_log: DataFrame with per-road shrinkage statistics
    """
    shrunken_cates_full = segment_cates_full.copy()
    
    # Global mean (shrinkage target for small roads)
    global_mean = np.nanmean(segment_cates_full)
    
    # Shrink each road
    roads = data['road_id'].unique()
    shrinkage_log = []
    
    for road_id in roads:
        road_mask = (data['road_id'] == road_id)
        road_indices = data.index[road_mask].values
        
        road_cates = segment_cates_full[road_indices]
        valid_mask = ~np.isnan(road_cates)
        n_road = valid_mask.sum()
        
        if n_road == 0:
            continue
        
        # Road mean (raw)
        road_mean_raw = np.nanmean(road_cates)
        
        # Shrinkage weight (James-Stein formula)
        shrinkage_weight = n_road / (n_road + shrinkage_k)
        
        # Shrink road mean toward global
        road_mean_shrunk = (
            shrinkage_weight * road_mean_raw +
            (1 - shrinkage_weight) * global_mean
        )
        
        # Shrink individual segments within road
        segment_deviations = road_cates - road_mean_raw
        road_cates_shrunk = (
            road_mean_shrunk + 
            shrinkage_weight * segment_deviations
        )
        
        # Update full array
        shrunken_cates_full[road_indices] = road_cates_shrunk
        
        # Log shrinkage statistics
        road_data = data[road_mask]
        shrinkage_log.append({
            'road_id': road_id,
            'road_name': road_data.iloc[0]['Road name'],
            'dataset_id': road_data.iloc[0]['Dataset ID'],
            'country': road_data.iloc[0]['Country Name'],
            'n_segments': n_road,
            'shrinkage_weight': shrinkage_weight,
            'shrinkage_pct': (1 - shrinkage_weight) * 100,
            'mean_raw': road_mean_raw,
            'mean_shrunk': road_mean_shrunk,
            'adjustment': road_mean_shrunk - road_mean_raw
        })
    
    return shrunken_cates_full, pd.DataFrame(shrinkage_log)


# ============================================================================
# PHASE E: SAVE COMPREHENSIVE RESULTS
# ============================================================================

def save_comprehensive_results(all_results, data, output_dir):
    """Save comprehensive results at all hierarchical levels."""
    # Create hierarchical output directory
    output_dir = Path(output_dir) / 'hierarchical_cf'
    subdirs = {
        'segment': output_dir / 'segment_level',
        'hotspot': output_dir / 'hotspot_level',
        'road': output_dir / 'road_level',
        'country': output_dir / 'country_level',
        'regional': output_dir / 'regional_level',
        'ate': output_dir / 'ate_results',
        'diagnostics': output_dir / 'diagnostics'
    }
    for subdir in subdirs.values():
        subdir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("SAVING COMPREHENSIVE RESULTS")
    print("="*70)
    
    # 1. SEGMENT LEVEL - ALL segments (wide format)
    print("\n[1/7] Segment-level data (all segments)...")
    
    # Select columns that exist in the data
    base_cols = ['segment_id', 'Dataset ID', 'Country Name', 'Region', 'road_id', 'is_hotspot', 'is_candidate_hotspot', 'actual_risk', 'predicted_risk']
    if 'Location ID' in data.columns:
        base_cols.insert(0, 'Location ID')
    if 'Road name' in data.columns:
        base_cols.insert(5, 'Road name')
    
    segment_data = data[base_cols].copy()
    
    if 'hotspot_class' in data.columns:
        segment_data['hotspot_class'] = data['hotspot_class']
    
    # Add CATE columns for each treatment (wide format)
    for treatment, results in all_results.items():
        if results['CATE_shrunk'] is None:
            continue
        
        safe_name = treatment.replace(' - ', '_').replace(' ', '_').lower()
        segment_data[f'{safe_name}_cate'] = results['CATE_shrunk']
        segment_data[f'{safe_name}_cate_raw'] = results['CATE_raw']
        segment_data[f'{safe_name}_ci_lower'] = results['CATE_ci_lower']
        segment_data[f'{safe_name}_ci_upper'] = results['CATE_ci_upper']
    
    segment_path = subdirs['segment'] / 'all_segments_cates_wide.csv'
    segment_data.to_csv(segment_path, index=False)
    print(f"    OK Saved: {segment_path.name} ({len(segment_data):,} rows)")
    
    # 2. HOTSPOT LEVEL
    # Candidate hotspots (TP+FP) are used for reporting/prescriptions; overlay hotspots (TP/FP/FN)
    # are exported separately for diagnostics.
    print("\n[2/7] Hotspot-level data...")
    candidate_hotspot_data = segment_data[segment_data['is_candidate_hotspot']].copy()
    candidate_hotspot_path = subdirs['hotspot'] / 'hotspot_segments_detailed.csv'
    candidate_hotspot_data.to_csv(candidate_hotspot_path, index=False)
    print(f"    OK Saved: {candidate_hotspot_path.name} ({len(candidate_hotspot_data):,} rows) [candidates TP+FP]")

    overlay_hotspot_data = segment_data[segment_data['is_hotspot']].copy()
    overlay_hotspot_path = subdirs['hotspot'] / 'hotspot_segments_overlay_detailed.csv'
    overlay_hotspot_data.to_csv(overlay_hotspot_path, index=False)
    print(f"    OK Saved: {overlay_hotspot_path.name} ({len(overlay_hotspot_data):,} rows) [overlay TP/FP/FN]")
    
    # 3. ROAD LEVEL
    print("\n[3/7] Road-level aggregation...")
    road_summaries = aggregate_by_road(data, all_results)
    road_path = subdirs['road'] / 'road_cate_summaries.csv'
    road_summaries.to_csv(road_path, index=False)
    print(f"    OK Saved: {road_path.name} ({len(road_summaries):,} rows)")
    print(f"    Contains: {road_summaries['road_id'].nunique()} unique roads")
    
    # 4. COUNTRY LEVEL
    print("\n[4/7] Country-level aggregation...")
    country_summaries = aggregate_by_country(data, all_results)
    country_path = subdirs['country'] / 'country_cate_summaries.csv'
    country_summaries.to_csv(country_path, index=False)
    print(f"    OK Saved: {country_path.name} ({len(country_summaries):,} rows)")
    print(f"    Countries: {', '.join(country_summaries['country'].unique())}")
    
    # 5. REGIONAL LEVEL
    print("\n[5/7] Regional-level aggregation...")
    regional_summaries = aggregate_by_region(data, all_results)
    regional_path = subdirs['regional'] / 'regional_cate_summaries.csv'
    regional_summaries.to_csv(regional_path, index=False)
    print(f"    OK Saved: {regional_path.name} ({len(regional_summaries):,} rows)")
    print(f"    Regions: {', '.join(regional_summaries['region'].unique())}")
    
    # 6. ATE RESULTS
    print("\n[6/7] ATE (Average Treatment Effect) summary...")
    ate_results = []
    for treatment, results in all_results.items():
        if results['CATE_shrunk'] is None:
            continue

        treatment_base = results.get('treatment_base', treatment)
        t0 = results.get('t0', np.nan)
        t1 = results.get('t1', np.nan)
        contrast_type = results.get('contrast_type', None)
        baseline_level = results.get('baseline_level', np.nan)
        
        cates = results['CATE_shrunk']
        valid_cates = cates[~np.isnan(cates)]
        
        candidate_mask = data['is_candidate_hotspot']
        candidate_cates = cates[candidate_mask.values]
        valid_candidate_cates = candidate_cates[~np.isnan(candidate_cates)]

        overlay_mask = data['is_hotspot']
        overlay_cates = cates[overlay_mask.values]
        valid_overlay_cates = overlay_cates[~np.isnan(overlay_cates)]
        
        ate_results.append({
            'treatment': treatment_base,
            'contrast': treatment,
            't0': t0,
            't1': t1,
            'contrast_type': contrast_type,
            'baseline_level': baseline_level,
            'ATE_all_segments': np.mean(valid_cates),
            'ATE_std_all': np.std(valid_cates),
            'ATE_se_all': np.std(valid_cates) / np.sqrt(len(valid_cates)),
            'ATE_ci_lower_all': np.percentile(valid_cates, 2.5),
            'ATE_ci_upper_all': np.percentile(valid_cates, 97.5),
            'percentile_5_all': np.percentile(valid_cates, 5),
            'percentile_95_all': np.percentile(valid_cates, 95),
            'ATE_candidate_hotspots': np.mean(valid_candidate_cates) if len(valid_candidate_cates) > 0 else np.nan,
            'ATE_std_candidate_hotspots': np.std(valid_candidate_cates) if len(valid_candidate_cates) > 0 else np.nan,
            'ATE_overlay_hotspots': np.mean(valid_overlay_cates) if len(valid_overlay_cates) > 0 else np.nan,
            'ATE_std_overlay_hotspots': np.std(valid_overlay_cates) if len(valid_overlay_cates) > 0 else np.nan,
            'n_total': len(cates),
            'n_valid_all': len(valid_cates),
            'n_candidate_hotspots': len(valid_candidate_cates),
            'n_overlay_hotspots': len(valid_overlay_cates),
            'percent_negative_all': (valid_cates < 0).mean() * 100,
            'percent_strong_effect_all': (valid_cates < -0.10).mean() * 100
        })
    
    ate_df = pd.DataFrame(ate_results)
    ate_path = subdirs['ate'] / 'ate_summary_table7.csv'
    ate_df.to_csv(ate_path, index=False)
    print(f"    OK Saved: {ate_path.name}")
    print(f"    Treatments analyzed: {len(ate_df)}")
    
    # 7. DIAGNOSTICS
    print("\n[7/7] Diagnostic files...")

    # Persist contrast spec so downstream code/paper tables can interpret
    # discrete multi-valued outputs unambiguously.
    try:
        contrast_spec = []
        for contrast_name, results in all_results.items():
            contrast_spec.append(
                {
                    "contrast": contrast_name,
                    "label": results.get("contrast_label"),
                    "treatment_base": results.get("treatment_base", contrast_name),
                    "t0": results.get("t0"),
                    "t1": results.get("t1"),
                    "contrast_type": results.get("contrast_type"),
                    "baseline_level": results.get("baseline_level"),
                    "levels": results.get("levels"),
                    "notes": (
                        "For discrete_treatment=True, EconML internally one-hot encodes multi-valued treatments "
                        "and drops the baseline (lexicographically smallest). This file records the explicit T0/T1 "
                        "used for reported effects to avoid K-1 dimension confusion."
                    ),
                }
            )

        contrast_path = subdirs['diagnostics'] / 'stage2_contrast_spec.json'
        contrast_path.write_text(json.dumps(contrast_spec, indent=2, sort_keys=True), encoding='utf-8')
        print(f"    OK Saved: {contrast_path.name}")
    except Exception as e:
        print(f"    Warning: could not write contrast spec: {e}")
    all_shrinkage = []
    for treatment, results in all_results.items():
        if 'shrinkage_log' in results and results['shrinkage_log'] is not None:
            log = results['shrinkage_log'].copy()
            log['treatment'] = treatment
            all_shrinkage.append(log)
    
    if all_shrinkage:
        shrinkage_df = pd.concat(all_shrinkage, ignore_index=True)
        shrinkage_path = subdirs['diagnostics'] / 'road_level_shrinkage_log.csv'
        shrinkage_df.to_csv(shrinkage_path, index=False)
        print(f"    OK Saved: {shrinkage_path.name} ({len(shrinkage_df):,} rows)")
    
    sample_sizes = pd.DataFrame([{
        'level': 'Segments (total)',
        'n': len(data),
        'description': 'All 100m road segments'
    }, {
        'level': 'Candidate hotspots (TP+FP)',
        'n': int(data['is_candidate_hotspot'].sum()),
        'description': 'Predicted high-risk segments used for reporting/prescriptions'
    }, {
        'level': 'Hotspot overlay (TP/FP/FN)',
        'n': int(data['is_hotspot'].sum()),
        'description': 'Evaluation/diagnostic overlay including false negatives'
    }, {
        'level': 'Roads',
        'n': data['road_id'].nunique(),
        'description': 'Unique roads across datasets'
    }, {
        'level': 'Countries',
        'n': data['Country Name'].nunique(),
        'description': 'National road networks'
    }, {
        'level': 'Regions',
        'n': data['Region'].nunique(),
        'description': 'Macro-regions (W Balkans, SE Europe, E Europe)'
    }])
    
    sample_path = subdirs['diagnostics'] / 'sample_sizes.csv'
    sample_sizes.to_csv(sample_path, index=False)
    print(f"    OK Saved: {sample_path.name}")
    
    print("\n" + "="*70)
    print(f"ALL RESULTS SAVED TO: {output_dir}")
    print("="*70)
    
    return output_dir


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(*, output_dir: Path, params: dict) -> None:
    """Execute complete hierarchical causal forest pipeline."""
    start_time = datetime.now()
    
    print("="*70)
    print("HIERARCHICAL CAUSAL FOREST - OPTIMAL SOLUTION")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Phase A: Prepare data
    data, X_features, viable_treatments = prepare_data(output_dir=output_dir)
    
    # Phase B: Estimate causal forests
    print("\n" + "="*70)
    print("PHASE B: CAUSAL FOREST ESTIMATION")
    print("="*70)
    print(f"Estimating {len(viable_treatments)} treatments...")
    print("(This will take 6-8 hours)")
    print(f"Strategy: Train/predict on ALL {len(data):,} segments, filter to hotspots for reporting")
    
    all_results = {}
    contrast_spec_for_run = []
    
    for i, treatment in enumerate(viable_treatments, 1):
        print(f"\n[{i}/{len(viable_treatments)}] {treatment}")

        cf_model, contrast_results = fit_causal_forest(data, X_features, treatment, params=params, verbose=True)

        if cf_model is None or not contrast_results:
            continue

        for cr in contrast_results:
            name = cr['name']
            all_results[name] = {
                'model': cf_model,
                'treatment_base': cr.get('treatment_base', treatment),
                't0': cr.get('t0'),
                't1': cr.get('t1'),
                'contrast_type': cr.get('contrast_type'),
                'contrast_label': cr.get('label'),
                'baseline_level': cr.get('baseline_level'),
                'levels': cr.get('levels'),
                'CATE_raw': cr['CATE_raw'],
                'CATE_ci_lower': cr['CATE_ci_lower'],
                'CATE_ci_upper': cr['CATE_ci_upper'],
                'CATE_shrunk': None,  # Will be filled in Phase D
            }

            contrast_spec_for_run.append(
                {
                    "contrast": name,
                    "label": cr.get("label"),
                    "treatment_base": cr.get("treatment_base", treatment),
                    "t0": cr.get("t0"),
                    "t1": cr.get("t1"),
                    "contrast_type": cr.get("contrast_type"),
                    "baseline_level": cr.get("baseline_level"),
                    "levels": cr.get("levels"),
                }
            )
    
    # Phase C: Road-level shrinkage (CHANGED from country-level)
    print("\n" + "="*70)
    print("PHASE C: ROAD-LEVEL SHRINKAGE")
    print("="*70)
    print("Applying empirical Bayes shrinkage to stabilize estimates")
    print("Formula: weight = n_road / (n_road + 20)")
    print("Shrinkage target: Road-level means toward global mean")
    
    all_shrinkage_logs = []
    
    for treatment in all_results.keys():
        print(f"  {treatment}...", end=' ', flush=True)
        shrunken_cates, shrinkage_log = apply_road_level_shrinkage(
            data,  # Use FULL dataset (not filtered hotspots)
            all_results[treatment]['CATE_raw'],
            shrinkage_k=20
        )
        all_results[treatment]['CATE_shrunk'] = shrunken_cates
        all_results[treatment]['shrinkage_log'] = shrinkage_log
        all_shrinkage_logs.append(shrinkage_log)
        print("OK")
    
    # Summary of shrinkage
    combined_log = pd.concat(all_shrinkage_logs, ignore_index=True)
    print(f"\nShrinkage applied to {combined_log['road_id'].nunique()} roads")
    print(f"Average shrinkage: {combined_log['shrinkage_pct'].mean():.1f}%")
    print(f"Range: {combined_log['shrinkage_pct'].min():.1f}% - {combined_log['shrinkage_pct'].max():.1f}%")
    
    # Phase D: Save comprehensive results
    # Persist per-run contrast metadata early (diagnostics + reproducibility)
    try:
        data_dir = output_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / 'stage2_contrast_spec.json').write_text(
            json.dumps(contrast_spec_for_run, indent=2, sort_keys=True),
            encoding='utf-8'
        )
        print(f"Saved contrast spec: {data_dir / 'stage2_contrast_spec.json'}")
    except Exception as e:
        print(f"Warning: could not write stage2 contrast spec: {e}")

    output_dir = save_comprehensive_results(all_results, data, output_dir)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"\nContrasts estimated: {len(all_results)}")
    print(f"Total segments analyzed: {len(data):,}")
    print(f"Candidate hotspots (TP+FP): {int(data['is_candidate_hotspot'].sum()):,}")
    print(f"Hotspot overlay (TP/FP/FN): {int(data['is_hotspot'].sum()):,}")
    print(f"Unique roads: {data['road_id'].nunique()}")
    print(f"Countries: {data['Country Name'].nunique()}")
    print(f"Regions: {data['Region'].nunique()}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 hierarchical causal forest (EconML CausalForestDML)")
    parser.add_argument("--run-id", type=str, default=None, help="Write outputs under stage2_outputs/runs/<run-id>")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit output directory (overrides --run-id)")

    # Sensitivity knobs (pre-specified; defaults match current baseline).
    parser.add_argument("--nuisance-n-estimators", type=int, default=500)
    parser.add_argument("--nuisance-max-depth", type=int, default=10)
    parser.add_argument("--nuisance-min-samples-leaf", type=int, default=5)
    parser.add_argument("--cf-n-estimators", type=int, default=2000)
    parser.add_argument("--cf-max-depth", type=int, default=8)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=10)
    parser.add_argument("--cf-mc-iters", type=int, default=4)

    args = parser.parse_args()

    output_dir = resolve_output_dir(output_dir=args.output_dir, run_id=args.run_id)
    log_path, _restore_logger = _install_run_logger(output_dir)
    try:
        print(f"Stage 2 log: {log_path}")
        write_run_metadata(output_dir=output_dir, args=args)
        ensure_downstream_inputs(output_dir=output_dir)

        params = {
            "nuisance": {
                "n_estimators": args.nuisance_n_estimators,
                "max_depth": args.nuisance_max_depth,
                "min_samples_leaf": args.nuisance_min_samples_leaf,
            },
            "causal_forest": {
                "n_estimators": args.cf_n_estimators,
                "max_depth": args.cf_max_depth,
                "min_samples_leaf": args.cf_min_samples_leaf,
                "mc_iters": args.cf_mc_iters,
            },
        }

        main(output_dir=output_dir, params=params)
    finally:
        _restore_logger()
