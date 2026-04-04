"""
Unified configuration for all reporter scripts.

Default behavior:
- auto-detect the most recent Stage 2 timestamped run folder;
- allow override via the STAGE2_RUN_DIR environment variable;
- keep all other paths relative to the repository root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent          # reporters/reporters/
WORKSPACE_ROOT = _THIS_DIR.parent.parent             # project root


def _resolve_latest_stage2_run() -> Path:
    stage2_outputs_root = WORKSPACE_ROOT / "stage2" / "stage2_outputs"
    if not stage2_outputs_root.exists():
        raise FileNotFoundError(f"Stage 2 outputs root not found: {stage2_outputs_root}")

    candidates: list[Path] = []
    for parent in stage2_outputs_root.iterdir():
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            if child.is_dir():
                candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            f"No timestamped Stage 2 run folders found under: {stage2_outputs_root}"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


_stage2_run_dir_override = os.environ.get("STAGE2_RUN_DIR", "").strip()
STAGE2_RUN_DIR = (
    Path(_stage2_run_dir_override).expanduser().resolve()
    if _stage2_run_dir_override
    else _resolve_latest_stage2_run()
)

# Run-level data
CONTRAST_SPEC_JSON = STAGE2_RUN_DIR / "data" / "stage2_contrast_spec.json"
RUN_METADATA_JSON = STAGE2_RUN_DIR / "stage2_run_metadata.json"
OUTCOME_SPEC_JSON = STAGE2_RUN_DIR / "data" / "outcome_spec.json"
TREATMENT_MAP_JSON = STAGE2_RUN_DIR / "data" / "treatment_level_mapping_used.json"

# Hierarchical CF outputs
HOTSPOT_SEGMENTS_CSV = STAGE2_RUN_DIR / "hierarchical_cf" / "hotspot_level" / "hotspot_segments_detailed.csv"
HOTSPOT_OVERLAY_CSV = STAGE2_RUN_DIR / "hierarchical_cf" / "hotspot_level" / "hotspot_segments_overlay_detailed.csv"
ALL_SEGMENTS_CATES_CSV = STAGE2_RUN_DIR / "hierarchical_cf" / "segment_level" / "all_segments_cates_wide.csv"
ATE_SUMMARY_CSV = STAGE2_RUN_DIR / "hierarchical_cf" / "ate_results" / "ate_summary_table7.csv"
REGIONAL_SUMMARIES_CSV = STAGE2_RUN_DIR / "hierarchical_cf" / "regional_level" / "regional_cate_summaries.csv"
COUNTRY_SUMMARIES_CSV = STAGE2_RUN_DIR / "hierarchical_cf" / "country_level" / "country_cate_summaries.csv"
ROAD_SUMMARIES_CSV = STAGE2_RUN_DIR / "hierarchical_cf" / "road_level" / "road_cate_summaries.csv"

# Diagnostics
SRIP_AGREEMENT_CSV = STAGE2_RUN_DIR / "hierarchical_cf" / "diagnostics" / "srip_agreement_per_treatment.csv"
SRIP_SEGMENT_CSV = STAGE2_RUN_DIR / "hierarchical_cf" / "diagnostics" / "srip_agreement_segment_level.csv"
COVARIATE_BALANCE_CSV = STAGE2_RUN_DIR / "hierarchical_cf" / "diagnostics" / "covariate_balance_smd.csv"

# Shared analysis dataset (one level up from the timestamped run)
ANALYSIS_DATASET_CSV = STAGE2_RUN_DIR.parent / "data" / "analysis_dataset.csv"

# Prescriptions and reporter outputs
PRESCRIPTIONS_DIR = STAGE2_RUN_DIR / "stage2_cf_prescriptions"
REPORTS_DIR = STAGE2_RUN_DIR / "reports"

# External data paths
SEGMENTS_UNIQUE_CSV = WORKSPACE_ROOT / "input_data" / "segments_unique.csv"
IRAP_COUNTERMEASURES_CSV = WORKSPACE_ROOT / "combine_data" / "666902 Countermeasures - Excluding Overridden.csv"
CODEBOOK_CSV = WORKSPACE_ROOT / "stage2" / "codes_filled_analysis.csv"
REGIONAL_MAPPING_CSV = WORKSPACE_ROOT / "stage2" / "dataset_regional_mapping.csv"

# Stage 1 outputs
STAGE1_OUTPUT_DIR = WORKSPACE_ROOT / "stage1" / "stage1_outputs"


def _resolve_latest_stage1_run() -> Path:
    if not STAGE1_OUTPUT_DIR.exists():
        return STAGE1_OUTPUT_DIR
    candidates = sorted(
        [d for d in STAGE1_OUTPUT_DIR.iterdir() if d.is_dir() and not d.name.startswith((".", "cv"))],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else STAGE1_OUTPUT_DIR


STAGE1_RUN_DIR = _resolve_latest_stage1_run()
STAGE1_OOF_PREDICTIONS_CSV = STAGE1_RUN_DIR / "fold_results" / "oof_predictions_segments.csv"
STAGE1_HOTSPOT_OVERLAY_CSV = STAGE1_RUN_DIR / "fold_results" / "hotspot_prediction_overlay.csv"
STAGE1_HOTSPOT_PROFILES_CSV = STAGE1_RUN_DIR / "fold_results" / "all_road_high_risk_profiles.csv"

ALL_TREATMENTS = [
    "Centreline rumble strips",
    "Delineation",
    "Street lighting",
    "Paved shoulder - driver-side",
    "Paved shoulder - passenger-side",
    "Road condition",
]

TREATMENT_DISPLAY_ORDER = [
    "Road condition",
    "Paved shoulder (passenger-side)",
    "Paved shoulder (driver-side)",
    "Street lighting",
    "Delineation",
    "Centreline rumble strips",
]

TREATMENT_DISPLAY_TO_CANONICAL = {
    "Centreline rumble strips": "Centreline rumble strips",
    "Delineation": "Delineation",
    "Street lighting": "Street lighting",
    "Paved shoulder (driver-side)": "Paved shoulder - driver-side",
    "Paved shoulder (passenger-side)": "Paved shoulder - passenger-side",
    "Road condition": "Road condition",
}

TARGET_COL = "Total Fatal and Serious Injury Estimation per 100m per year"
OUTCOME_EPSILON = 0.001
RANDOM_STATE = 42

DML_CF_PARAMS = dict(
    n_estimators=2000,
    max_depth=8,
    min_samples_leaf=10,
    cv=5,
    mc_iters=4,
)

CONTROL_FEATURES = [
    "Vehicle flow (AADT)",
    "Bicycle peak hour flow",
    "Motorcycle %",
    "Pedestrian peak hour flow across the road",
    "Pedestrian peak hour flow along the road driver-side",
    "Pedestrian peak hour flow along the road passenger-side",
    "Operating Speed (85th percentile)",
    "Area type",
    "Carriageway",
    "Curvature",
    "Grade",
    "Quality of curve",
    "Number of lanes",
    "Sight distance",
    "Land use - driver-side",
    "Land use - passenger-side",
    "Roadside severity - driver-side distance",
    "Roadside severity - passenger-side distance",
    "Roadside severity - driver-side object",
    "Roadside severity - passenger-side object",
    "Sidewalk - driver-side",
    "Sidewalk - passenger-side",
    "Motorcycle observed flow",
    "Motorcycle speed limit",
    "Bicycle observed flow",
    "Intersecting road volume",
    "Pedestrian observed flow across the road",
    "Pedestrian observed flow along the road driver-side",
    "Pedestrian observed flow along the road passenger-side",
    "Service road",
    "Intersection quality",
    "Lane width",
    "Pedestrian crossing quality",
    "Vehicle parking",
    "Property access points",
    "Differential speed limits",
    "Shoulder rumble strips",
    "Dataset ID",
]

FIGURE_DPI = 300

COLORS = {
    "beneficial": "#009E73",
    "harmful": "#D55E00",
    "stage2": "#0072B2",
}

REGION_ORDER = [
    "EU Central/Adriatic",
    "Western Balkans (non-EU)",
    "EU Southeast Europe",
    "Eastern Europe",
]

REGION_COLORS = {
    "EU Central/Adriatic": "#1f77b4",
    "Western Balkans (non-EU)": "#d62728",
    "EU Southeast Europe": "#2ca02c",
    "Eastern Europe": "#9467bd",
}


def resolve_stage2_root(*, stage2_output_dir: str | None = None, run_id: str | None = None) -> Path:
    if stage2_output_dir:
        return Path(stage2_output_dir).resolve()
    if run_id:
        return _THIS_DIR / "stage2_outputs" / "runs" / str(run_id)
    return STAGE2_RUN_DIR.resolve()


def ensure_reports_dir(stage2_root: Path | None = None) -> Path:
    d = (stage2_root or STAGE2_RUN_DIR) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest_file(directory: Path, glob: str) -> Path | None:
    if not directory.exists():
        return None
    matches = sorted(directory.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def add_stage2_to_sys_path() -> None:
    stage2_dir = str(WORKSPACE_ROOT / "stage2")
    if stage2_dir not in sys.path:
        sys.path.insert(0, stage2_dir)


def validate() -> list[str]:
    warnings: list[str] = []
    checks = {
        "Stage 2 run directory": STAGE2_RUN_DIR,
        "Hotspot segments CSV": HOTSPOT_SEGMENTS_CSV,
        "Contrast spec JSON": CONTRAST_SPEC_JSON,
        "Analysis dataset CSV": ANALYSIS_DATASET_CSV,
        "Segments unique CSV": SEGMENTS_UNIQUE_CSV,
        "Stage 1 OOF predictions": STAGE1_OOF_PREDICTIONS_CSV,
    }
    for label, path in checks.items():
        if not path.exists():
            warnings.append(f"[WARN] {label} not found: {path}")
    return warnings


if __name__ == "__main__":
    print("Reporter configuration")
    print("=" * 70)
    print(f"  STAGE2_RUN_DIR         : {STAGE2_RUN_DIR}")
    print(f"  WORKSPACE_ROOT         : {WORKSPACE_ROOT}")
    print(f"  REPORTS_DIR            : {REPORTS_DIR}")
    print(f"  HOTSPOT_SEGMENTS_CSV   : {HOTSPOT_SEGMENTS_CSV}")
    print(f"  ALL_SEGMENTS_CATES_CSV : {ALL_SEGMENTS_CATES_CSV}")
    print(f"  ANALYSIS_DATASET_CSV   : {ANALYSIS_DATASET_CSV}")
    print(f"  SEGMENTS_UNIQUE_CSV    : {SEGMENTS_UNIQUE_CSV}")
    print(f"  CODEBOOK_CSV           : {CODEBOOK_CSV}")
    print(f"  IRAP_COUNTERMEASURES   : {IRAP_COUNTERMEASURES_CSV}")
    print(f"  STAGE1_RUN_DIR         : {STAGE1_RUN_DIR}")
    print("=" * 70)
    issues = validate()
    if issues:
        for warning in issues:
            print(warning)
    else:
        print("[OK] All key files found.")
