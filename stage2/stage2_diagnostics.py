"""stage2_diagnostics.py

Post-estimation diagnostics for Stage 2 Causal Forest pipeline.
Run automatically at the end of ``stage2_hierarchical_cf.main()`` or
standalone as ``python stage2_diagnostics.py``.

Produces:
  1. SRIP mapping table  (countermeasure -> 6 treatment classes)
  2. Per-treatment SRIP agreement (CF prescription vs iRAP recommendation)
  3. SMD / covariate-balance diagnostics across treatment arms
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. SRIP -> 6-treatment mapping table
# ---------------------------------------------------------------------------
# Maps every unique ``Countermeasure`` string in the iRAP countermeasures CSV
# to one of the six study treatments (or ``None`` if no match).
#
# Mapping logic follows the iRAP SRIP countermeasure naming convention:
#   - Each rule is (substring, treatment_name).
#   - First match wins (order matters for ambiguous names).
#   - Unmatched countermeasures are recorded but excluded from agreement.

SRIP_TO_TREATMENT: List[tuple[str, str]] = [
    # --- Centreline rumble strips ---
    ("centreline rumble strip", "Centreline rumble strips"),
    ("centre line rumble strip", "Centreline rumble strips"),
    ("centreline rumble", "Centreline rumble strips"),
    ("wide centreline", "Centreline rumble strips"),

    # --- Delineation ---
    ("improve delineation", "Delineation"),
    ("improve curve delineation", "Delineation"),
    ("delineation and signing", "Delineation"),
    # Note: "Central hatching" is delineation in iRAP vocabulary
    ("central hatching", "Delineation"),

    # --- Street lighting ---
    ("street lighting", "Street lighting"),
    ("lighting", "Street lighting"),

    # --- Road condition ---
    ("road surface rehabilitation", "Road condition"),
    ("skid resistance", "Road condition"),

    # --- Paved shoulder (driver-side) ---
    ("shoulder sealing driver side", "Paved shoulder - driver-side"),
    ("shoulder rumble strips", "Paved shoulder - driver-side"),
    # Shoulder rumble strips physically attach to the shoulder on the
    # driver side edge-line; map to the driver-side shoulder treatment.

    # --- Paved shoulder (passenger-side) ---
    ("shoulder sealing passenger side", "Paved shoulder - passenger-side"),
]


def build_srip_mapping_table(
    countermeasure_csv: Path | str,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Create a lookup table mapping every unique countermeasure name to a treatment.

    Returns a DataFrame with columns:
        countermeasure, countermeasure_summary_group, mapped_treatment

    If ``output_path`` is given the table is also written as CSV.
    """
    df = pd.read_csv(
        countermeasure_csv,
        usecols=["Countermeasure", "Countermeasure Summary Group"],
    )
    unique_cms = (
        df.drop_duplicates(subset=["Countermeasure"])
        .rename(columns={
            "Countermeasure": "countermeasure",
            "Countermeasure Summary Group": "countermeasure_summary_group",
        })
        .reset_index(drop=True)
    )

    def _match(cm_name: str) -> Optional[str]:
        lower = str(cm_name).strip().lower()
        for pattern, treatment in SRIP_TO_TREATMENT:
            if pattern in lower:
                return treatment
        return None

    unique_cms["mapped_treatment"] = unique_cms["countermeasure"].apply(_match)

    n_mapped = unique_cms["mapped_treatment"].notna().sum()
    n_total = len(unique_cms)
    print(f"  SRIP mapping: {n_mapped}/{n_total} countermeasure types mapped to 6 treatments")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        unique_cms.to_csv(out, index=False)
        print(f"  Saved: {out}")

    return unique_cms


# ---------------------------------------------------------------------------
# 2. Per-treatment SRIP agreement
# ---------------------------------------------------------------------------

def compute_srip_agreement(
    analysis_data: pd.DataFrame,
    countermeasure_csv: Path | str,
    treatments: Sequence[str],
    output_dir: Path | str,
    srip_mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare CF prescriptions against iRAP SRIP recommendations per treatment.

    For every hotspot segment that appears in the SRIP ledger and has a
    CF prescription, we create binary indicators:
        - ``srip_recommends``: 1 if SRIP has a countermeasure mapped to this treatment
        - ``cf_prescribes``:   1 if CF CATE is negative (risk-reducing)

    and compute per-treatment: TP, FP, FN, TN, precision, recall, F1, agreement rate.

    Parameters
    ----------
    analysis_data : DataFrame
        Full Stage 2 analysis DataFrame (all segments) with columns including
        ``segment_id``, ``is_candidate_hotspot``, and per-treatment CATE columns
        named ``CATE_shrunk_{treatment}`` or ``CATE_raw_{treatment}``.
    countermeasure_csv : path
        Path to iRAP countermeasures CSV.
    treatments : list of str
        The 6 treatment names.
    output_dir : path
        Where to write outputs.
    srip_mapping : DataFrame, optional
        Pre-built mapping table.  If None, builds it from ``countermeasure_csv``.

    Returns
    -------
    DataFrame with per-treatment agreement metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build SRIP mapping if not provided
    if srip_mapping is None:
        srip_mapping = build_srip_mapping_table(countermeasure_csv)

    # Step 2: Load countermeasures and map to treatments
    cm_df = pd.read_csv(countermeasure_csv, usecols=["Location ID", "Countermeasure"])
    cm_df = cm_df.rename(columns={"Location ID": "segment_id", "Countermeasure": "countermeasure"})

    # Map each countermeasure row to a treatment
    cm_mapped = cm_df.merge(
        srip_mapping[["countermeasure", "mapped_treatment"]],
        on="countermeasure",
        how="left",
    )
    # Keep only rows that map to one of our 6 treatments
    cm_mapped = cm_mapped[cm_mapped["mapped_treatment"].isin(treatments)]

    # Step 3: Build per-segment x per-treatment SRIP indicator
    # For each (segment_id, treatment), SRIP recommends = 1 if any countermeasure maps to it
    srip_indicators = (
        cm_mapped.groupby(["segment_id", "mapped_treatment"])
        .size()
        .reset_index(name="srip_count")
    )
    srip_indicators["srip_recommends"] = 1

    # Step 4: Determine CF prescription indicator
    # Identify available CATE columns
    cate_cols = {}
    for t in treatments:
        # Try shrunk first, then raw
        for prefix in ("CATE_shrunk_", "CATE_raw_"):
            col_name = f"{prefix}{t}"
            if col_name in analysis_data.columns:
                cate_cols[t] = col_name
                break

    # Focus on candidate hotspots
    hotspot_df = analysis_data[analysis_data["is_candidate_hotspot"]].copy()
    hotspot_ids = set(hotspot_df["segment_id"].unique())

    # Also restrict to segments that have SRIP coverage
    srip_covered_ids = set(cm_mapped["segment_id"].unique())
    overlap_ids = hotspot_ids & srip_covered_ids
    print(f"  SRIP agreement: {len(overlap_ids)} hotspot segments with SRIP coverage "
          f"(out of {len(hotspot_ids)} candidate hotspots)")

    # Step 5: Per-treatment agreement computation
    agreement_rows = []
    segment_level_rows = []

    for t in treatments:
        cate_col = cate_cols.get(t)

        # SRIP: which overlap segments have this treatment recommended?
        srip_for_t = set(
            srip_indicators[
                (srip_indicators["mapped_treatment"] == t)
                & (srip_indicators["segment_id"].isin(overlap_ids))
            ]["segment_id"].unique()
        )

        # CF: which overlap segments have negative CATE (risk-reducing)?
        if cate_col and cate_col in hotspot_df.columns:
            cf_for_t = set(
                hotspot_df[
                    (hotspot_df["segment_id"].isin(overlap_ids))
                    & (hotspot_df[cate_col] < 0)
                ]["segment_id"].unique()
            )
        else:
            cf_for_t = set()

        tp = len(srip_for_t & cf_for_t)
        fp = len(cf_for_t - srip_for_t)
        fn = len(srip_for_t - cf_for_t)
        tn = len(overlap_ids) - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        agreement = (tp + tn) / len(overlap_ids) if overlap_ids else 0.0

        agreement_rows.append({
            "treatment": t,
            "n_overlap_hotspots": len(overlap_ids),
            "srip_recommends": len(srip_for_t),
            "cf_prescribes": len(cf_for_t),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "F1": round(f1, 4),
            "agreement_rate": round(agreement, 4),
        })

        # Segment-level detail
        for seg_id in overlap_ids:
            segment_level_rows.append({
                "segment_id": seg_id,
                "treatment": t,
                "srip_recommends": 1 if seg_id in srip_for_t else 0,
                "cf_prescribes": 1 if seg_id in cf_for_t else 0,
            })

    result_df = pd.DataFrame(agreement_rows)
    result_df.to_csv(output_dir / "srip_agreement_per_treatment.csv", index=False)
    print(f"  Saved: {output_dir / 'srip_agreement_per_treatment.csv'}")

    seg_df = pd.DataFrame(segment_level_rows)
    seg_df.to_csv(output_dir / "srip_agreement_segment_level.csv", index=False)
    print(f"  Saved: {output_dir / 'srip_agreement_segment_level.csv'}")

    # Print summary
    print("\n  Per-treatment SRIP agreement:")
    print(f"  {'Treatment':<35s} {'SRIP':>5s} {'CF':>5s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>5s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    for _, row in result_df.iterrows():
        print(f"  {row['treatment']:<35s} {row['srip_recommends']:>5d} {row['cf_prescribes']:>5d} "
              f"{row['TP']:>4d} {row['FP']:>4d} {row['FN']:>4d} {row['TN']:>5d} "
              f"{row['precision']:>6.3f} {row['recall']:>6.3f} {row['F1']:>6.3f}")

    return result_df


# ---------------------------------------------------------------------------
# 3. SMD / covariate balance diagnostics
# ---------------------------------------------------------------------------

def standardized_mean_difference(
    x_treated: np.ndarray,
    x_control: np.ndarray,
) -> float:
    """Compute the absolute standardized mean difference (Austin, 2009).

    SMD = |mean(treated) - mean(control)| / sqrt((var_t + var_c) / 2)

    Returns 0 if both groups have zero variance.
    """
    mean_t = np.nanmean(x_treated)
    mean_c = np.nanmean(x_control)
    var_t = np.nanvar(x_treated, ddof=1) if len(x_treated) > 1 else 0.0
    var_c = np.nanvar(x_control, ddof=1) if len(x_control) > 1 else 0.0
    pooled_sd = np.sqrt((var_t + var_c) / 2)
    if pooled_sd < 1e-12:
        # When both groups have zero variance, if means differ the SMD is
        # conventionally infinite; we cap at a large sentinel value.
        if abs(mean_t - mean_c) < 1e-12:
            return 0.0
        return 999.0  # flag as extreme imbalance
    return float(abs(mean_t - mean_c) / pooled_sd)


def compute_covariate_balance(
    analysis_data: pd.DataFrame,
    X_features: pd.DataFrame,
    treatments: Sequence[str],
    output_dir: Path | str,
    smd_threshold: float = 0.1,
) -> pd.DataFrame:
    """Compute SMD for every covariate x treatment combination.

    For binary treatments (2 levels): treated vs. control.
    For ordinal treatments (>2 levels): highest level vs. lowest level.

    Saves:
        - ``covariate_balance_smd.csv`` -- full SMD table
        - ``covariate_balance_summary.json`` -- counts of imbalanced features

    Parameters
    ----------
    analysis_data : DataFrame
        Must contain treatment columns (canonically mapped).
    X_features : DataFrame
        The feature matrix used in DML (same row order as analysis_data).
    treatments : list of str
        The 6 treatment names.
    smd_threshold : float
        Threshold above which an SMD is flagged as imbalanced (default 0.10).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smd_rows = []
    summary: Dict[str, Any] = {}

    for treatment in treatments:
        if treatment not in analysis_data.columns:
            print(f"  [WARN] Treatment column '{treatment}' not found -- skipping SMD.")
            continue

        t_vals = analysis_data[treatment].dropna()
        levels = sorted(t_vals.unique())

        if len(levels) < 2:
            print(f"  [WARN] '{treatment}' has <2 levels -- skipping SMD.")
            continue

        # Define treated vs control
        # Binary: 1 vs 0.  Ordinal: highest vs lowest.
        control_level = levels[0]
        treated_level = levels[-1]

        mask_treated = analysis_data[treatment] == treated_level
        mask_control = analysis_data[treatment] == control_level

        n_treated = int(mask_treated.sum())
        n_control = int(mask_control.sum())

        n_imbalanced = 0
        for feat_col in X_features.columns:
            x_t = X_features.loc[mask_treated, feat_col].values.astype(float)
            x_c = X_features.loc[mask_control, feat_col].values.astype(float)
            smd_val = standardized_mean_difference(x_t, x_c)
            is_imbalanced = smd_val > smd_threshold

            smd_rows.append({
                "treatment": treatment,
                "treated_level": treated_level,
                "control_level": control_level,
                "n_treated": n_treated,
                "n_control": n_control,
                "covariate": feat_col,
                "smd": round(smd_val, 4),
                "imbalanced": is_imbalanced,
            })
            if is_imbalanced:
                n_imbalanced += 1

        n_feats = X_features.shape[1]
        summary[treatment] = {
            "n_treated": n_treated,
            "n_control": n_control,
            "treated_level": int(treated_level),
            "control_level": int(control_level),
            "n_covariates": n_feats,
            "n_imbalanced": n_imbalanced,
            "pct_imbalanced": round(100 * n_imbalanced / n_feats, 1) if n_feats else 0,
        }
        print(f"  {treatment}: {n_imbalanced}/{n_feats} covariates with SMD>{smd_threshold} "
              f"(n_treated={n_treated:,}, n_control={n_control:,})")

    smd_df = pd.DataFrame(smd_rows)
    smd_path = output_dir / "covariate_balance_smd.csv"
    smd_df.to_csv(smd_path, index=False)
    print(f"  Saved: {smd_path}")

    summary_path = output_dir / "covariate_balance_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    return smd_df


# ---------------------------------------------------------------------------
# Master runner (called from stage2_hierarchical_cf.main)
# ---------------------------------------------------------------------------

def run_all_diagnostics(
    analysis_data: pd.DataFrame,
    X_features: pd.DataFrame,
    treatments: Sequence[str],
    output_dir: Path | str,
    countermeasure_csv: Path | str,
) -> Dict[str, Any]:
    """Run all Stage 2 diagnostics and save outputs.

    Called at the end of the Stage 2 pipeline.

    Returns dict of results for optional downstream use.
    """
    output_dir = Path(output_dir)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    # --- 1. SRIP mapping table ---
    print("\n" + "-" * 60)
    print("DIAGNOSTIC 1: SRIP Mapping Table")
    print("-" * 60)
    srip_map = build_srip_mapping_table(
        countermeasure_csv,
        output_path=diag_dir / "srip_countermeasure_to_treatment_mapping.csv",
    )
    results["srip_mapping"] = srip_map

    # --- 2. Per-treatment SRIP agreement ---
    print("\n" + "-" * 60)
    print("DIAGNOSTIC 2: Per-Treatment SRIP Agreement")
    print("-" * 60)
    try:
        agreement = compute_srip_agreement(
            analysis_data=analysis_data,
            countermeasure_csv=countermeasure_csv,
            treatments=treatments,
            output_dir=diag_dir,
            srip_mapping=srip_map,
        )
        results["srip_agreement"] = agreement
    except Exception as e:
        print(f"  [WARN] SRIP agreement computation failed: {e}")
        import traceback; traceback.print_exc()

    # --- 3. SMD / covariate balance ---
    print("\n" + "-" * 60)
    print("DIAGNOSTIC 3: Covariate Balance (SMD)")
    print("-" * 60)
    try:
        smd_df = compute_covariate_balance(
            analysis_data=analysis_data,
            X_features=X_features,
            treatments=treatments,
            output_dir=diag_dir,
        )
        results["covariate_balance"] = smd_df
    except Exception as e:
        print(f"  [WARN] Covariate balance computation failed: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "-" * 60)
    print("All Stage 2 diagnostics complete.")
    print("-" * 60)

    return results


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Run diagnostics on existing Stage 2 outputs (no re-estimation needed)."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2 diagnostics (standalone)")
    parser.add_argument("--stage2-output-dir", type=str, required=True,
                        help="Path to an existing Stage 2 output directory")
    args = parser.parse_args()

    output_dir = Path(args.stage2_output_dir)
    data_dir = output_dir / "data"

    # Load prepared data
    from stage2_config import SEGMENTS_DATA_CSV, CONTROL_FEATURES, ALL_TREATMENTS
    from stage2_config import STAGE1_OOF_PREDICTIONS

    # ---- Strategy: load the segment-level CATE CSV (already saved) and
    #      merge CATEs onto the re-prepared analysis data so the SRIP
    #      agreement diagnostic can identify CF prescriptions. ----------
    seg_csv = (output_dir / "hierarchical_cf" / "segment_level"
               / "all_segments_cates_wide.csv")

    prep_path = data_dir / "stage2_prepared_data.csv"
    if prep_path.exists():
        print(f"Loading prepared data from {prep_path}")
        analysis_data = pd.read_csv(prep_path, low_memory=False)
    else:
        print("No prepared data found. Re-preparing from raw inputs.")
        from stage2_hierarchical_cf import prepare_data
        analysis_data, X_features, _ = prepare_data(output_dir=output_dir)

    # Attach CATEs from the segment-level CSV if available.
    # For ordinal treatments with multiple contrasts (e.g. 0->1, 1->2, 2->3),
    # combine using next-step logic: each segment gets the CATE from the
    # contrast where the segment's current canonical level equals t0.
    if seg_csv.exists():
        print(f"Loading CATEs from {seg_csv.name}")
        seg_df = pd.read_csv(seg_csv)
        for t in ALL_TREATMENTS:
            # Match the naming convention from save_comprehensive_results():
            # treatment.replace(' - ', '_').replace(' ', '_').lower()
            t_prefix = t.replace(' - ', '_').replace(' ', '_').lower()
            for suffix, dest_prefix in [("_cate_raw", "CATE_raw_"), ("_cate", "CATE_shrunk_")]:
                matches = [c for c in seg_df.columns
                           if c.startswith(t_prefix + "__") and c.endswith(suffix)]
                if suffix == "_cate":
                    matches = [c for c in matches if not c.endswith("_cate_raw")]
                if not matches:
                    continue
                dest_col = f"{dest_prefix}{t}"
                if len(matches) == 1:
                    # Binary treatment: single contrast column
                    analysis_data[dest_col] = seg_df[matches[0]].values
                    print(f"  Mapped {matches[0]} -> {dest_col}")
                else:
                    # Ordinal treatment: combine contrasts via next-step logic
                    combined = np.full(len(analysis_data), np.nan, dtype=float)
                    T_can = analysis_data[t].values
                    for col in sorted(matches):
                        # Extract t0 from "..._{t0}_to_{t1}{suffix}"
                        parts = col.replace(suffix, "").split("__")[-1].split("_to_")
                        t0 = int(parts[0])
                        seg_at_t0 = (np.round(T_can) == t0)
                        vals = seg_df[col].values
                        valid = seg_at_t0 & ~np.isnan(vals)
                        combined[valid] = vals[valid]
                    analysis_data[dest_col] = combined
                    print(f"  Mapped {len(matches)} contrasts -> {dest_col} (next-step)")
    else:
        print(f"[WARN] Segment CATE CSV not found: {seg_csv}")

    # Reconstruct X_features from columns
    available_controls = [c for c in CONTROL_FEATURES if c in analysis_data.columns and c != "Dataset ID"]
    X_controls = analysis_data[available_controls].fillna(0)
    X_numeric = pd.DataFrame(index=analysis_data.index)
    for col in X_controls.columns:
        if X_controls[col].dtype == "object":
            dummies = pd.get_dummies(X_controls[col], prefix=col, drop_first=False)
            X_numeric = pd.concat([X_numeric, dummies], axis=1)
        else:
            X_numeric[col] = pd.to_numeric(X_controls[col], errors="coerce")
    if "Dataset ID" in analysis_data.columns:
        ds_dummies = pd.get_dummies(analysis_data["Dataset ID"].astype(str), prefix="dataset", drop_first=False)
        X_features = pd.concat([X_numeric, ds_dummies], axis=1).astype(float)
    else:
        X_features = X_numeric.astype(float)

    from stage2_config import INPUT_DATA_DIR
    countermeasure_csv = INPUT_DATA_DIR / "countermeasures.csv"

    run_all_diagnostics(
        analysis_data=analysis_data,
        X_features=X_features,
        treatments=ALL_TREATMENTS,
        output_dir=output_dir,
        countermeasure_csv=countermeasure_csv,
    )
