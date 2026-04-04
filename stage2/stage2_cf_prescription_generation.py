"""
Stage 2 (CF-Aligned): Prescription Generator
=====================================================

Translates Stage 2 hierarchical causal forest outputs into actionable
segment- and road-level prescriptions, while explicitly reporting
which Stage 1 hotspots triggered the analysis.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from stage2_config import (
    OUTPUT_DIR,
    ALL_TREATMENTS,
    BINARY_TREATMENTS,
    ORDINAL_TREATMENTS,
    OUTCOME_EPSILON,
)

# ---------------------------------------------------------------------------
# RUN DIRECTORY OVERRIDE
# ---------------------------------------------------------------------------
# By default we read/write under stage2_config.OUTPUT_DIR. Set this to a
# specific run folder only if you intentionally want to target a non-default
# Stage 2 run without editing the global config.
#
# Example:
#   CUSTOM_STAGE2_OUTPUT_DIR = Path("stage2_outputs/from_stage1_2026-02-23/2026-02-24_18-47-18")
#
# Leave as None to keep automatic behaviour.
CUSTOM_STAGE2_OUTPUT_DIR: Path | None = None

ACTIVE_OUTPUT_DIR = CUSTOM_STAGE2_OUTPUT_DIR or OUTPUT_DIR
# analysis_dataset.csv lives one level up from the timestamped run folder
_analysis_candidate = ACTIVE_OUTPUT_DIR / "data" / "analysis_dataset.csv"
if not _analysis_candidate.exists():
    _analysis_candidate = ACTIVE_OUTPUT_DIR.parent / "data" / "analysis_dataset.csv"
ANALYSIS_DATASET_PATH = _analysis_candidate


def _normalize_treatment_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "treatment"


CURRENT_LEVEL_COLUMN_MAP: Dict[str, str] = {
    treatment: f"current_{_normalize_treatment_name(treatment)}"
    for treatment in ALL_TREATMENTS
}


TREATMENT_LEVEL_METADATA: Dict[str, Dict[str, object]] = {
    "Centreline rumble strips": {
        "order": [0, 1],
        "labels": {0: "Not present", 1: "Installed"},
    },
    "Delineation": {
        "order": [0, 1],
        "labels": {0: "Poor or missing", 1: "Adequate / installed"},
    },
    "Street lighting": {
        "order": [0, 1],
        "labels": {0: "Not present", 1: "Present"},
    },
    "Paved shoulder - driver-side": {
        "order": [0, 1, 2, 3],
        "labels": {
            0: "No paved shoulder",
            1: "<1 m paved shoulder",
            2: "1-2 m paved shoulder",
            3: ">=2 m paved shoulder",
        },
    },
    "Paved shoulder - passenger-side": {
        "order": [0, 1, 2, 3],
        "labels": {
            0: "No paved shoulder",
            1: "<1 m paved shoulder",
            2: "1-2 m paved shoulder",
            3: ">=2 m paved shoulder",
        },
    },
    "Road condition": {
        "order": [0, 1, 2],
        "labels": {
            0: "Poor surface",
            1: "Medium condition",
            2: "Good / rehabilitated surface",
        },
    },
}

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CF_HOTSPOT_FILE = (
    ACTIVE_OUTPUT_DIR
    / "hierarchical_cf"
    / "hotspot_level"
    / "hotspot_segments_detailed.csv"
)
STAGE2_CF_DIR = ACTIVE_OUTPUT_DIR / "stage2_cf_prescriptions"
STAGE2_CF_DIR.mkdir(parents=True, exist_ok=True)
REGIONAL_EXPORT_DIR = STAGE2_CF_DIR / "regions"
REGIONAL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MIN_ABSOLUTE_REDUCTION = 0.002  # absolute crash risk units (F+SI /100m/year)
MIN_PERCENT_REDUCTION = 5.0     # percent of baseline risk
TOP_TREATMENTS_PER_SEGMENT = 3
EPS = 1e-6

# ---------------------------------------------------------------------------
# REGION GROUPING (REPORTING)
# ---------------------------------------------------------------------------
# Optional: collapse multiple datasets into broader reporting regions.
# This affects downstream regional summaries/figures (e.g., Figure 4).
# If a dataset is not listed below, we keep its existing `Region` value.
# Keys use generic region labels (no country names per anonymisation constraint).
REGIONAL_GROUPINGS: Dict[str, Dict[str, object]] = {
    "Region_A": {
        "name": "EU Central/Adriatic",
        "datasets": ["1240", "1242", "1424", "1425", "1426"],
        "description": "5 datasets, 2 countries, Central/Adriatic",
    },
    "Region_B": {
        "name": "Western Balkans (non-EU)",
        "datasets": ["1246", "1247", "12008"],
        "description": "3 datasets, 2 countries, Western Balkans",
    },
    "Region_C": {
        "name": "EU Southeast Europe",
        "datasets": ["1398", "1400", "12983"],
        "description": "3 datasets, 3 countries, Southeast Europe",
    },
    "Region_D": {
        "name": "Eastern Europe",
        "datasets": ["980"],
        "description": "1 dataset, 1 country, Eastern Europe",
    },
}


def _normalize_dataset_id(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _apply_region_groupings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "Dataset ID" not in df.columns:
        return df

    dataset_to_group: Dict[str, str] = {}
    for meta in REGIONAL_GROUPINGS.values():
        name = str(meta.get("name", "")).strip()
        for ds in meta.get("datasets", []):
            dataset_to_group[str(ds)] = name

    out = df.copy()
    if "Region" in out.columns and "Region_original" not in out.columns:
        out["Region_original"] = out["Region"]

    grouped = out["Dataset ID"].apply(_normalize_dataset_id).map(dataset_to_group)
    if "Region" in out.columns:
        out["Region"] = grouped.fillna(out["Region"])
    else:
        out["Region"] = grouped.fillna("Unknown")

    return out

# ---------------------------------------------------------------------------
# CONTRAST-BASED CATE COLUMN HELPERS
# ---------------------------------------------------------------------------
# The hierarchical causal forest outputs one CATE column per adjacent-step
# contrast (e.g. 0→1, 1→2) rather than one per treatment.  Column names
# follow the pattern:  {treatment_slug}__{from}_to_{to}_cate
#   with matching _ci_lower / _ci_upper columns.

TREATMENT_CF_SLUG: Dict[str, str] = {
    "Centreline rumble strips": "centreline_rumble_strips",
    "Delineation": "delineation",
    "Street lighting": "street_lighting",
    "Paved shoulder - driver-side": "paved_shoulder_driver-side",
    "Paved shoulder - passenger-side": "paved_shoulder_passenger-side",
    "Road condition": "road_condition",
}


def _contrast_columns_for_treatment(
    treatment: str,
) -> Dict[Tuple[int, int], str]:
    """Return ``{(from_level, to_level): cate_column_name}`` for *treatment*."""
    slug = TREATMENT_CF_SLUG[treatment]
    meta = TREATMENT_LEVEL_METADATA[treatment]
    order: list = meta["order"]
    return {
        (order[i], order[i + 1]): f"{slug}__{order[i]}_to_{order[i + 1]}_cate"
        for i in range(len(order) - 1)
    }


def _all_expected_contrast_columns() -> List[str]:
    """Return every CATE column we expect to find in the hotspot CSV."""
    cols: List[str] = []
    for treatment in ALL_TREATMENTS:
        cols.extend(_contrast_columns_for_treatment(treatment).values())
    return cols


@dataclass
class Recommendation:
    segment_id: int
    location_id: int
    dataset_id: str
    country: str
    region: str
    road_name: str
    road_id: str
    hotspot_class: str
    treatment: str
    cate: float
    abs_reduction: float
    pct_reduction: float
    ci_lower: float | None
    ci_upper: float | None
    confidence: str
    priority: str
    recommendation_text: str
    current_level: int | None
    target_level: int | None
    current_level_label: str | None
    target_level_label: str | None
    level_change_label: str | None
    baseline_risk: float


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def load_current_treatment_levels() -> pd.DataFrame:
    if not ANALYSIS_DATASET_PATH.exists():
        raise FileNotFoundError(
            "analysis_dataset.csv not found. Run stage2_create_analysis_dataset.py before Stage 2 prescriptions."
        )

    usecols = ["Location ID"] + ALL_TREATMENTS
    data = pd.read_csv(ANALYSIS_DATASET_PATH, usecols=usecols, low_memory=False)
    rename_map = {"Location ID": "segment_id"}
    rename_map.update({name: CURRENT_LEVEL_COLUMN_MAP[name] for name in ALL_TREATMENTS})
    current_levels = data.rename(columns=rename_map)
    current_levels["segment_id"] = current_levels["segment_id"].astype(int)
    return current_levels


def describe_level(treatment: str, level: float | int | None) -> str | None:
    if level is None or pd.isna(level):
        return None
    try:
        level_int = int(level)
    except (ValueError, TypeError):
        return None

    meta = TREATMENT_LEVEL_METADATA.get(treatment, {})
    labels = meta.get("labels", {})
    return labels.get(level_int, f"Level {level_int}")


def get_next_better_level(treatment: str, level: float | int | None) -> int | None:
    if level is None or pd.isna(level):
        return None

    try:
        level_int = int(level)
    except (ValueError, TypeError):
        return None

    meta = TREATMENT_LEVEL_METADATA.get(treatment)
    if not meta:
        return None

    order = meta.get("order", [])
    if level_int not in order:
        return None

    idx = order.index(level_int)
    if idx >= len(order) - 1:
        return level_int  # already best level
    return order[idx + 1]


def format_level_change_label(
    treatment: str,
    current_level: int,
    target_level: int,
    current_label: str | None,
    target_label: str | None,
) -> str:
    text_current = current_label or f"Level {current_level}"
    text_target = target_label or f"Level {target_level}"
    return (
        f"{treatment} {current_level}->{target_level} "
        f"({text_current} -> {text_target})"
    )


def summarize_treatment_changes(recs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "treatment",
        "level_change_label",
        "current_level",
        "target_level",
        "current_level_label",
        "target_level_label",
        "n_prescriptions",
        "total_baseline_risk",
        "total_abs_reduction",
        "net_pct_reduction",
        "mean_abs_reduction",
        "mean_pct_reduction",
        "share_of_treatment_pct",
    ]

    if recs.empty or "level_change_label" not in recs.columns:
        return pd.DataFrame(columns=columns)

    change_recs = recs.dropna(subset=["level_change_label"]).copy()
    if change_recs.empty:
        return pd.DataFrame(columns=columns)

    summary = change_recs.groupby(
        [
            "treatment",
            "level_change_label",
            "current_level",
            "target_level",
            "current_level_label",
            "target_level_label",
        ]
    ).agg(
        n_prescriptions=("segment_id", "count"),
        total_baseline_risk=("baseline_risk", "sum"),
        total_abs_reduction=("abs_reduction", "sum"),
        total_pct_reduction=("pct_reduction", "sum"),
        mean_abs_reduction=("abs_reduction", "mean"),
        mean_pct_reduction=("pct_reduction", "mean"),
    ).reset_index()

    summary["net_pct_reduction"] = np.where(
        summary["total_baseline_risk"] > 0,
        summary["total_abs_reduction"] / summary["total_baseline_risk"] * 100.0,
        np.nan,
    )

    summary["share_of_treatment_pct"] = summary.groupby("treatment")["n_prescriptions"].transform(
        lambda counts: counts / counts.sum() * 100.0 if counts.sum() else 0.0
    )

    summary = summary.sort_values(
        ["n_prescriptions", "total_abs_reduction"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return summary[columns]

def load_hotspot_cf_outputs() -> pd.DataFrame:
    if not CF_HOTSPOT_FILE.exists():
        raise FileNotFoundError(
            f"Stage 2 hotspot file not found: {CF_HOTSPOT_FILE}\n"
            "Run stage2_hierarchical_cf.py first."
        )

    df = pd.read_csv(CF_HOTSPOT_FILE, low_memory=False)
    if "is_hotspot" not in df.columns:
        raise ValueError("Expected 'is_hotspot' column to flag Stage 1 selections.")

    hotspots = df[df["is_hotspot"] == True].copy()  # noqa: E712
    if hotspots.empty:
        raise ValueError("No hotspots found in CF output.")

    # Scientific alignment:
    # - Prescriptions should be generated for the predicted candidate hotspot set (TP+FP).
    # - False negatives (FN) are part of the validation overlay and should be diagnostics-only.
    if "hotspot_class" in hotspots.columns:
        candidates = hotspots[hotspots["hotspot_class"].isin(["TP", "FP"])].copy()
        if candidates.empty:
            print(
                "Warning: 'hotspot_class' present but no TP/FP rows found; "
                "falling back to all is_hotspot rows."
            )
        else:
            removed_fn = int((hotspots["hotspot_class"] == "FN").sum())
            if removed_fn:
                print(f"Excluding {removed_fn:,} FN rows (diagnostics-only).")
            hotspots = candidates
    else:
        print(
            "Warning: 'hotspot_class' missing; cannot exclude FN explicitly. "
            "Proceeding with all is_hotspot rows."
        )

    hotspots = _apply_region_groupings(hotspots)
    return hotspots


def confidence_label(ci_lower: float | None, ci_upper: float | None) -> str:
    if pd.isna(ci_lower) or pd.isna(ci_upper):
        return "Unknown"
    if ci_upper < 0:
        return "High (CI < 0)"
    if ci_lower < 0 < ci_upper:
        return "Medium (CI crosses 0)"
    return "Low"


def priority_label(pct_reduction: float) -> str:
    if pct_reduction >= 25:
        return "CRITICAL"
    if pct_reduction >= 15:
        return "HIGH"
    if pct_reduction >= 8:
        return "MEDIUM"
    return "LOW"


def build_recommendations(hotspots: pd.DataFrame) -> pd.DataFrame:
    recommendations: List[Recommendation] = []

    unknown_treatments = sorted(set(ALL_TREATMENTS) - set(TREATMENT_CF_SLUG.keys()))
    if unknown_treatments:
        raise ValueError(
            "Missing CF slug mapping for treatments: " + ", ".join(unknown_treatments)
        )

    expected_cols = _all_expected_contrast_columns()
    missing_cols = [col for col in expected_cols if col not in hotspots.columns]
    if missing_cols:
        raise ValueError(f"Missing CATE contrast columns in CF output: {missing_cols}")

    # Pre-compute contrast mappings once
    treatment_contrasts = {
        treatment: _contrast_columns_for_treatment(treatment)
        for treatment in ALL_TREATMENTS
    }

    for _, row in hotspots.iterrows():
        # actual_risk is on the log scale: log(FSI + eps).
        # Convert to FSI scale for absolute and relative reduction thresholds.
        baseline_log = float(row.get("actual_risk", 0.0))
        baseline_plus = math.exp(baseline_log)          # FSI + eps
        baseline_fsi = max(baseline_plus - OUTCOME_EPSILON, EPS)  # FSI
        for treatment in ALL_TREATMENTS:
            # ---- current level --------------------------------------------------
            current_col = CURRENT_LEVEL_COLUMN_MAP.get(treatment)
            current_val = row.get(current_col) if current_col else None
            current_level = None
            if current_val is not None and not pd.isna(current_val):
                try:
                    current_level = int(current_val)
                except (ValueError, TypeError):
                    current_level = None

            target_level = get_next_better_level(treatment, current_level)

            # Skip if current level is unknown or already at best
            if current_level is None or target_level is None or target_level == current_level:
                continue

            # ---- pick the adjacent-step contrast column -------------------------
            contrast_key = (current_level, target_level)
            cate_col = treatment_contrasts[treatment].get(contrast_key)
            if cate_col is None:
                continue

            cate_val = row.get(cate_col)
            if pd.isna(cate_val):
                continue

            abs_reduction = max(0.0, baseline_plus * (1.0 - math.exp(float(cate_val))))
            pct_reduction = abs_reduction / baseline_fsi * 100.0
            if abs_reduction < MIN_ABSOLUTE_REDUCTION or pct_reduction < MIN_PERCENT_REDUCTION:
                continue

            ci_lower = row.get(cate_col.replace("_cate", "_ci_lower"))
            ci_upper = row.get(cate_col.replace("_cate", "_ci_upper"))
            conf = confidence_label(ci_lower, ci_upper)
            priority = priority_label(pct_reduction)

            current_level_label = describe_level(treatment, current_level)
            target_level_label = describe_level(treatment, target_level)
            level_change_label = None
            if (
                current_level is not None
                and target_level is not None
                and target_level != current_level
            ):
                level_change_label = format_level_change_label(
                    treatment,
                    current_level,
                    target_level,
                    current_level_label,
                    target_level_label,
                )

            rec_text = (
                f"Stage 1 hotspot ({row.get('hotspot_class', 'NA')}). "
                f"Apply {treatment} for an estimated {pct_reduction:.1f}% "
                f"risk drop (~{abs_reduction:.3f} absolute). Confidence: {conf}."
            )

            recommendations.append(
                Recommendation(
                    segment_id=int(row["segment_id"]),
                    location_id=int(row.get("Location ID", row["segment_id"])),
                    dataset_id=str(row.get("Dataset ID", "")),
                    country=str(row.get("Country Name", "")),
                    region=str(row.get("Region", "")),
                    road_name=str(row.get("Road name", "")),
                    road_id=str(row.get("road_id", "")),
                    hotspot_class=str(row.get("hotspot_class", "")),
                    treatment=treatment,
                    cate=float(cate_val),
                    abs_reduction=abs_reduction,
                    pct_reduction=pct_reduction,
                    ci_lower=None if pd.isna(ci_lower) else float(ci_lower),
                    ci_upper=None if pd.isna(ci_upper) else float(ci_upper),
                    confidence=conf,
                    priority=priority,
                    recommendation_text=rec_text,
                    current_level=current_level,
                    target_level=target_level,
                    current_level_label=current_level_label,
                    target_level_label=target_level_label,
                    level_change_label=level_change_label,
                    baseline_risk=baseline_fsi,
                )
            )

    if not recommendations:
        print("Warning: no treatments met the benefit thresholds.")
        return pd.DataFrame()

    rec_df = pd.DataFrame([rec.__dict__ for rec in recommendations])
    rec_df["rank_within_segment"] = (
        rec_df.sort_values(["segment_id", "pct_reduction"], ascending=[True, False])
        .groupby("segment_id")
        .cumcount()
        + 1
    )
    return rec_df


def summarize_segments(hotspots: pd.DataFrame, recs: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "segment_id",
        "Location ID",
        "Dataset ID",
        "Country Name",
        "Region",
        "Road name",
        "road_id",
        "hotspot_class",
        "actual_risk",
        "predicted_risk",
    ]
    base = hotspots[base_cols].drop_duplicates(subset=["segment_id"]).copy()

    if recs.empty:
        base["n_recommendations"] = 0
        base["total_abs_reduction"] = 0.0
        base["total_pct_reduction"] = 0.0
        base["potential_new_risk"] = base["actual_risk"]
        return base

    agg = recs.groupby("segment_id").agg(
        n_recommendations=("treatment", "count"),
        total_abs_reduction=("abs_reduction", "sum"),
        total_pct_reduction=("pct_reduction", "sum"),
    ).reset_index()

    summaries = base.merge(agg, on="segment_id", how="left")
    summaries.loc[:, "n_recommendations"] = summaries["n_recommendations"].fillna(0)
    summaries.loc[:, "total_abs_reduction"] = summaries["total_abs_reduction"].fillna(0.0)
    summaries.loc[:, "total_pct_reduction"] = summaries["total_pct_reduction"].fillna(0.0)
    summaries.loc[:, "potential_new_risk"] = (
        summaries["actual_risk"] - summaries["total_abs_reduction"]
    ).clip(lower=0.0)
    summaries["n_recommendations"] = summaries["n_recommendations"].astype(int)

    # attach top-N recommendations per segment
    sorted_recs = recs.sort_values(["segment_id", "pct_reduction"], ascending=[True, False])
    for rank in range(1, TOP_TREATMENTS_PER_SEGMENT + 1):
        topk = sorted_recs[sorted_recs["rank_within_segment"] == rank]
        summaries = summaries.merge(
            topk[[
                "segment_id",
                "treatment",
                "pct_reduction",
                "recommendation_text",
            ]].rename(
                columns={
                    "treatment": f"top_treatment_{rank}",
                    "pct_reduction": f"top_reduction_{rank}_pct",
                    "recommendation_text": f"top_recommendation_{rank}",
                }
            ),
            on="segment_id",
            how="left",
        )

    return summaries


def summarize_roads(segment_summary: pd.DataFrame, recs: pd.DataFrame) -> pd.DataFrame:
    summary = segment_summary.copy()
    summary.loc[:, "road_id"] = summary["road_id"].fillna("unknown")

    road_groups = summary.groupby("road_id")
    road_df = road_groups.agg(
        road_name=("Road name", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        country=("Country Name", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        region=("Region", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        n_segments=("segment_id", "nunique"),
        mean_baseline_risk=("actual_risk", "mean"),
        total_abs_reduction=("total_abs_reduction", "sum"),
        total_pct_reduction=("total_pct_reduction", "sum"),
        n_segments_with_rec=("n_recommendations", lambda x: (x > 0).sum()),
    ).reset_index()

    if recs.empty:
        road_df["common_treatment"] = None
        road_df["road_priority"] = "LOW"
        return road_df

    change_recs = recs[recs["pct_reduction"] > 0].copy()
    if change_recs.empty:
        road_df["common_treatment"] = None
    else:
        common = (
            change_recs.groupby(["road_id", "treatment"])
            .size()
            .reset_index(name="count")
            .sort_values(["count"], ascending=False)
        )
        top_common = common.drop_duplicates("road_id")
        road_df["common_treatment"] = road_df["road_id"].map(
            top_common.set_index("road_id")["treatment"]
        )

    def classify_road(row: pd.Series) -> str:
        if row["total_pct_reduction"] >= 50:
            return "CRITICAL"
        if row["total_pct_reduction"] >= 30:
            return "HIGH"
        if row["total_pct_reduction"] >= 15 or row["n_segments_with_rec"] > 0:
            return "MEDIUM"
        return "LOW"

    road_df["road_priority"] = road_df.apply(classify_road, axis=1)
    return road_df


def summarize_regions(segment_summary: pd.DataFrame, recs: pd.DataFrame) -> pd.DataFrame:
    summary = segment_summary.copy()
    summary.loc[:, "Region"] = summary["Region"].fillna("Unknown")

    region_groups = summary.groupby("Region")
    region_df = region_groups.agg(
        n_segments=("segment_id", "nunique"),
        mean_baseline_risk=("actual_risk", "mean"),
        total_abs_reduction=("total_abs_reduction", "sum"),
        total_pct_reduction=("total_pct_reduction", "sum"),
        n_segments_with_rec=("n_recommendations", lambda x: (x > 0).sum()),
        countries=("Country Name", lambda x: ", ".join(sorted(x.dropna().unique()))),
    ).reset_index().rename(columns={"Region": "region"})

    if recs.empty:
        region_df["common_treatment"] = None
        region_df["region_priority"] = "LOW"
        return region_df

    change_recs = recs[recs["pct_reduction"] > 0].copy()
    if change_recs.empty:
        region_df["common_treatment"] = None
    else:
        change_recs.loc[:, "region"] = change_recs["region"].fillna("Unknown")
        region_map = (
            change_recs.groupby(["region", "treatment"])
            .size()
            .reset_index(name="count")
            .sort_values(["count"], ascending=False)
            .drop_duplicates("region")
            .set_index("region")["treatment"]
        )
        region_df["common_treatment"] = region_df["region"].map(region_map)

    def classify_region(row: pd.Series) -> str:
        if row["total_pct_reduction"] >= 200:
            return "CRITICAL"
        if row["total_pct_reduction"] >= 120:
            return "HIGH"
        if row["total_pct_reduction"] >= 60 or row["n_segments_with_rec"] >= 5:
            return "MEDIUM"
        return "LOW"

    region_df["region_priority"] = region_df.apply(classify_region, axis=1)
    return region_df


def summarize_region_treatments(recs: pd.DataFrame) -> pd.DataFrame:
    if recs.empty:
        return pd.DataFrame(columns=[
            "region",
            "treatment",
            "n_segments",
            "total_abs_reduction",
            "total_pct_reduction",
            "mean_abs_reduction",
            "mean_pct_reduction",
            "median_pct_reduction",
            "share_of_region_pct",
            "mean_confidence_score",
        ])

    recs_copy = recs.copy()
    recs_copy.loc[:, "region"] = recs_copy["region"].fillna("Unknown")

    confidence_scores = {
        "High (CI < 0)": 3,
        "Medium (CI crosses 0)": 2,
        "Low": 1,
        "Unknown": 0,
    }
    recs_copy.loc[:, "confidence_score"] = recs_copy["confidence"].map(confidence_scores).fillna(0)

    grouped = recs_copy.groupby(["region", "treatment"])
    summary = grouped.agg(
        n_segments=("segment_id", "nunique"),
        total_abs_reduction=("abs_reduction", "sum"),
        total_pct_reduction=("pct_reduction", "sum"),
        mean_abs_reduction=("abs_reduction", "mean"),
        mean_pct_reduction=("pct_reduction", "mean"),
        median_pct_reduction=("pct_reduction", "median"),
        mean_confidence_score=("confidence_score", "mean")
    ).reset_index()

    region_totals = summary.groupby("region")["total_pct_reduction"].transform("sum")
    summary["share_of_region_pct"] = np.where(
        region_totals > 0,
        summary["total_pct_reduction"] / region_totals * 100,
        np.nan,
    )

    # Map confidence score back to qualitative tiers for readability
    score_to_label = {3: "High", 2: "Medium", 1: "Low", 0: "Unknown"}
    summary["mean_confidence_label"] = summary["mean_confidence_score"].round().map(score_to_label)

    return summary.sort_values(["region", "total_pct_reduction"], ascending=[True, False])


def slugify_region(name: str) -> str:
    if not name or str(name).strip() == "":
        return "unknown"
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return slug or "unknown"


def export_region_recommendations(recs: pd.DataFrame, timestamp: str) -> List[Path]:
    exported_files: List[Path] = []
    if recs.empty:
        return exported_files

    recs_with_region = recs.copy()
    recs_with_region.loc[:, "region"] = recs_with_region["region"].fillna("Unknown")

    for region_name in sorted(recs_with_region["region"].unique()):
        region_recs = recs_with_region[recs_with_region["region"] == region_name]
        if region_recs.empty:
            continue

        # Order by largest percent reduction first within each segment.
        region_recs = region_recs.sort_values(
            ["segment_id", "pct_reduction"],
            ascending=[True, False]
        )
        slug = slugify_region(region_name)
        file_path = REGIONAL_EXPORT_DIR / f"{slug}_prescriptions_{timestamp}.csv"
        region_recs.to_csv(file_path, index=False)
        exported_files.append(file_path)

    return exported_files


def save_outputs(
    recs: pd.DataFrame,
    segment_summary: pd.DataFrame,
    road_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    region_treatment_summary: pd.DataFrame,
    treatment_change_summary: pd.DataFrame,
    hotspots: pd.DataFrame,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    rec_path = STAGE2_CF_DIR / f"stage2_segment_prescriptions_{timestamp}.csv"
    seg_path = STAGE2_CF_DIR / f"stage2_segment_summaries_{timestamp}.csv"
    road_path = STAGE2_CF_DIR / f"stage2_road_summaries_{timestamp}.csv"
    region_path = STAGE2_CF_DIR / f"stage2_region_summaries_{timestamp}.csv"
    region_treatment_path = STAGE2_CF_DIR / f"stage2_region_treatment_summaries_{timestamp}.csv"
    treatment_change_path = STAGE2_CF_DIR / f"stage2_treatment_change_summary_{timestamp}.csv"
    stage1_path = STAGE2_CF_DIR / f"stage1_hotspot_inventory_{timestamp}.csv"

    if not recs.empty:
        recs.to_csv(rec_path, index=False)
        print(f"[OK] Saved prescriptions: {rec_path}")
    else:
        print("No prescriptions saved (empty recommendations).")

    segment_summary.to_csv(seg_path, index=False)
    road_summary.to_csv(road_path, index=False)
    region_summary.to_csv(region_path, index=False)
    region_treatment_summary.to_csv(region_treatment_path, index=False)
    treatment_change_summary.to_csv(treatment_change_path, index=False)

    stage1_cols = [
        "segment_id",
        "Location ID",
        "Dataset ID",
        "Country Name",
        "Region",
        "Road name",
        "road_id",
        "hotspot_class",
        "actual_risk",
        "predicted_risk",
    ]
    stage1_df = hotspots[stage1_cols].drop_duplicates(subset=["segment_id"]).merge(
        segment_summary[[
            "segment_id",
            "n_recommendations",
            "total_abs_reduction",
            "total_pct_reduction",
            "potential_new_risk",
        ]],
        on="segment_id",
        how="left",
    )
    stage1_df.loc[:, "n_recommendations"] = stage1_df["n_recommendations"].fillna(0)
    stage1_df.loc[:, "total_abs_reduction"] = stage1_df["total_abs_reduction"].fillna(0.0)
    stage1_df.loc[:, "total_pct_reduction"] = stage1_df["total_pct_reduction"].fillna(0.0)
    stage1_df.loc[:, "potential_new_risk"] = stage1_df["potential_new_risk"].fillna(stage1_df["actual_risk"])
    stage1_df["n_recommendations"] = stage1_df["n_recommendations"].astype(int)
    stage1_df.to_csv(stage1_path, index=False)

    region_files = export_region_recommendations(recs, timestamp)

    print(f"[OK] Saved segment summaries: {seg_path}")
    print(f"[OK] Saved road summaries: {road_path}")
    print(f"[OK] Saved region summaries: {region_path}")
    print(f"[OK] Saved region-treatment summaries: {region_treatment_path}")
    if treatment_change_summary.empty:
        print("No level-change summary (insufficient upgrade information).")
    else:
        print(f"[OK] Saved treatment-change summary: {treatment_change_path}")
    print(f"[OK] Saved Stage 1 hotspot inventory: {stage1_path}")
    if region_files:
        print("Regional prescription files:")
        for path in region_files:
            print(f"  - {path}")


def run_pipeline() -> None:
    print("=" * 70)
    print("STAGE 2 CF: PRESCRIPTION GENERATION")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    hotspots = load_hotspot_cf_outputs()
    print(f"Stage 1 hotspots loaded: {len(hotspots):,}")

    current_levels = load_current_treatment_levels()
    hotspots = hotspots.merge(current_levels, on="segment_id", how="left")
    current_cols = list(CURRENT_LEVEL_COLUMN_MAP.values())
    missing_current = hotspots[current_cols].isna().any(axis=1).sum()
    if missing_current:
        print(f"Warning: missing current treatment levels for {missing_current} hotspots")

    recs = build_recommendations(hotspots)
    print(f"Recommendations generated: {len(recs):,}" if not recs.empty else "Recommendations generated: 0")

    segment_summary = summarize_segments(hotspots, recs)
    road_summary = summarize_roads(segment_summary, recs)
    region_summary = summarize_regions(segment_summary, recs)
    region_treatment_summary = summarize_region_treatments(recs)
    treatment_change_summary = summarize_treatment_changes(recs)

    save_outputs(
        recs,
        segment_summary,
        road_summary,
        region_summary,
        region_treatment_summary,
        treatment_change_summary,
        hotspots,
    )

    print("=" * 70)
    print("STAGE 2 CF COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 CF Prescription Generator")
    parser.add_argument(
        "--min-abs",
        type=float,
        default=MIN_ABSOLUTE_REDUCTION,
        help="Minimum absolute risk reduction to keep (default 0.002)",
    )
    parser.add_argument(
        "--min-pct",
        type=float,
        default=MIN_PERCENT_REDUCTION,
        help="Minimum percent risk reduction to keep (default 5%)",
    )
    args = parser.parse_args()

    MIN_ABSOLUTE_REDUCTION = args.min_abs
    MIN_PERCENT_REDUCTION = args.min_pct

    run_pipeline()
