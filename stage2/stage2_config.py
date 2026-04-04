"""
Stage 2: Causal Analysis Configuration
VALIDATED CONFIGURATION (October 25, 2025 - iRAP SRIP VALIDATION)
6 Treatments
Codebook-verified, feasibility-assessed, engineering-validated
"""

import os
from pathlib import Path

# Canonical treatment mapping (single source of truth for recoding)
TREATMENT_CODEBOOK_MODULE = "stage2_treatment_codebook"

# ============================================================================
# Treatment Taxonomy - VALIDATED BY iRAP SRIP COMPARISON (Oct 25, 2025)
# ============================================================================


BINARY_TREATMENTS = [
    'Centreline rumble strips',  # iRAP: 11,176 recs | Codebook: "Centre line rumble strips"
    'Delineation',               # iRAP: 62,000+ recs | Codebook: 1=Adequate, 2=Poor
    'Street lighting'            # iRAP: ~200 recs | Codebook: 1=Not present, 2=Present
]

ORDINAL_TREATMENTS = [
    'Paved shoulder - driver-side',      # iRAP: 64,725 recs (2nd most common!)
    'Paved shoulder - passenger-side',   # iRAP: 40,929 recs (3rd most common!)
    'Road condition'                     # iRAP: 17,305 recs | Codebook: 1=Good, 3=Poor
]

# All validated treatments combined (6 total)
ALL_TREATMENTS = BINARY_TREATMENTS + ORDINAL_TREATMENTS

RECLASSIFIED_TO_CONTROLS = [
    # Not retrofit-able (new construction, not interventions)
    'Service road',  # iRAP: 0 recs | Requires land acquisition, $10M-50M/km
    
    # Too ambiguous (abstract ratings, not specific actions)
    'Intersection quality',      # iRAP has specific actions (signal, roundabout, delineation)
    'Lane width',                # iRAP specifies amounts (widen by 0.5m), not categories
    'Pedestrian crossing quality',  # iRAP has specific types (signalized, marked, grade-separated)
    
    # Policy/enforcement issues (not physical infrastructure)
    'Vehicle parking',           # Enforcement issue, ambiguous safety direction
    'Property access points',    # Legal barrier (property rights), iRAP: 0 recs
    'Differential speed limits', # Actually "Speed differential" (observational, not intervention)
    
    # Insufficient variation (excluded from analysis)
    'Shoulder rumble strips'     # 97% in one level (low positivity)
]

CONTROL_FEATURES = [
    # Exposure
    'Vehicle flow (AADT)',
    'Bicycle peak hour flow',
    'Motorcycle %',
    'Pedestrian peak hour flow across the road',
    'Pedestrian peak hour flow along the road driver-side',
    'Pedestrian peak hour flow along the road passenger-side',
    'Operating Speed (85th percentile)',
    
    # Geometric design (fixed properties)
    'Area type',
    'Carriageway',
    'Curvature',
    'Grade',
    'Quality of curve',
    'Number of lanes',
    'Sight distance',
    
    # Land use and context
    'Land use - driver-side',
    'Land use - passenger-side',
    
    # Roadside environment
    'Roadside severity - driver-side distance',
    'Roadside severity - passenger-side distance',
    'Roadside severity - driver-side object',  # 17 levels
    'Roadside severity - passenger-side object',  # 17 levels
    
    # Vulnerable road user infrastructure
    'Sidewalk - driver-side',  # 7 levels
    'Sidewalk - passenger-side',  # 7 levels
    
    # Traffic flows
    'Motorcycle observed flow',
    'Motorcycle speed limit',
    'Bicycle observed flow',
    'Intersecting road volume',
    'Pedestrian observed flow across the road',
    'Pedestrian observed flow along the road driver-side',
    'Pedestrian observed flow along the road passenger-side',
    
    # These are better as controls (not modifiable interventions)
    'Service road',               # Road type classification
    'Intersection quality',       # Abstract rating (control for existing quality)
    'Lane width',                 # Geometric feature (control for existing width)
    'Pedestrian crossing quality',# Abstract rating (control for existing quality)
    'Vehicle parking',            # Land use/policy context
    'Property access points',     # Land use context
    'Differential speed limits',  # Actually "Speed differential" - traffic behavior observation
    'Shoulder rumble strips',     # Low variation (97% vs 3%)
    
    # Dataset control
    'Dataset ID'  # CRITICAL: 12-level categorical control (geographic heterogeneity)
]

# Identifiers
ID_COL = 'Location ID'
ROAD_COL = 'Road name'
SEGMENT_ID_COL = 'segment_id'  # From Stage 1
DATASET_ID_COL = 'Dataset ID'

# ============================================================================
# DML Estimation Parameters
# ============================================================================

# Cross-fitting
DML_N_FOLDS = 5  # Number of folds for cross-fitting in DML

# Inference
DML_N_BOOTSTRAP = 1000  # Bootstrap iterations for road-clustered SEs
CONFIDENCE_LEVEL = 0.95

# Random seed
RANDOM_STATE = 42

# ============================================================================
# Outcome Variable Configuration
# ============================================================================

# Target variable (from Stage 1)
TARGET_COL = 'Total Fatal and Serious Injury Estimation per 100m per year'

# Outcome scale options
OUTCOME_SCALES = ['log', 'linear', 'rate']
# - 'log': log(risk + epsilon) for multiplicative effects
# - 'linear': absolute risk counts
# - 'rate': risk per km (requires segment length)

DEFAULT_OUTCOME = 'log'  # Preferred for interpretability
# IMPORTANT: Stage 1 uses a log-offset transform with offset=0.001 (see stage1_config.py).
# Keep this consistent across downstream scripts that back-transform model outputs.
OUTCOME_EPSILON = 0.001  # Offset used for log(FSI + offset)

# Outcome column names from Stage 1
OUTCOME_COL_ACTUAL = 'actual_risk'  # Ground truth from Stage 1 OOF predictions
OUTCOME_COL_PREDICTED = 'predicted_risk'  # Model predictions

# ============================================================================
# Paths
# ============================================================================

# Stage 1 outputs (OOF predictions - 147,466 segments)
STAGE1_OUTPUT_DIR = Path(__file__).parent.parent / 'stage1' / 'stage1_outputs'
LEGACY_STAGE1_OUTPUT_DIR = STAGE1_OUTPUT_DIR
STAGE1_BASE_DIR = STAGE1_OUTPUT_DIR

# Manual override: set a specific run name here. Leave empty ('') to
# auto-detect the most recent Stage 1 run directory at import time.
STAGE1_RUN_NAME = ''


def _resolve_stage1_run() -> Path:
    """Return the Stage 1 run directory.

    Priority:
      1. ``STAGE1_RUN_NAME`` (manual override) -- if non-empty and the dir exists.
      2. Most recently modified timestamped sub-directory in ``STAGE1_BASE_DIR``.
      3. Falls back to ``STAGE1_BASE_DIR`` itself (will likely fail downstream,
         but lets validation report a clear error).
    """
    # --- manual override ---
    if STAGE1_RUN_NAME:
        candidate = STAGE1_BASE_DIR / STAGE1_RUN_NAME
        if candidate.is_dir():
            return candidate
        print(f"[WARN] Manual STAGE1_RUN_NAME='{STAGE1_RUN_NAME}' not found; "
              f"falling back to auto-detection.")

    # --- auto-detect latest ---
    if STAGE1_BASE_DIR.exists():
        run_dirs = sorted(
            [d for d in STAGE1_BASE_DIR.iterdir()
             if d.is_dir() and not d.name.startswith('.')],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if run_dirs:
            print(f"[INFO] Auto-detected Stage 1 run: {run_dirs[0].name}")
            return run_dirs[0]

    # --- fallback ---
    print("[WARN] No Stage 1 run directories found; downstream reads will fail.")
    return STAGE1_BASE_DIR


STAGE1_RUN_DIR = _resolve_stage1_run()
# Expose the resolved name so downstream code can reference it
STAGE1_RUN_NAME_RESOLVED = STAGE1_RUN_DIR.name

# Key Stage 1 files
STAGE1_OOF_PREDICTIONS = STAGE1_RUN_DIR / 'fold_results' / 'oof_predictions_segments.csv'
STAGE1_HOTSPOT_OVERLAY = STAGE1_RUN_DIR / 'fold_results' / 'hotspot_prediction_overlay.csv'
STAGE1_HOTSPOT_PROFILES = STAGE1_RUN_DIR / 'fold_results' / 'all_road_high_risk_profiles.csv'
STAGE1_METRICS = STAGE1_RUN_DIR / 'fold_results' / 'per_road_hotspot_metrics.csv'

# Phase 2 data directory (feature data - 147,466 segments x 115 columns)
INPUT_DATA_DIR = Path(__file__).parent.parent / 'input_data'
PHASE2_DATA_DIR = INPUT_DATA_DIR
SEGMENTS_DATA_CSV = INPUT_DATA_DIR / 'segments_unique.csv'

# Stage 2 outputs -- organised by Stage 1 run for traceability
OUTPUT_DIR = Path(__file__).parent / 'stage2_outputs'
_run_label = STAGE1_RUN_NAME_RESOLVED if STAGE1_RUN_DIR != STAGE1_BASE_DIR else 'default'
_run_date = _run_label.split('_')[0] if '_' in _run_label else _run_label
OUTPUT_DIR = OUTPUT_DIR / f"from_stage1_{_run_date}"

# Output subdirectories
OUTPUT_SUBDIRS = {
    'ate_results': OUTPUT_DIR / 'ate_results',
    'dose_response': OUTPUT_DIR / 'dose_response',
    'heterogeneity': OUTPUT_DIR / 'heterogeneity',
    'diagnostics': OUTPUT_DIR / 'diagnostics',
    'prescriptions': OUTPUT_DIR / 'prescriptions',
    'reports': OUTPUT_DIR / 'reports',
    'data': OUTPUT_DIR / 'data'  # Preprocessed analysis datasets
}

# Stage 2 prepared data file path
STAGE2_DATA_PATH = OUTPUT_SUBDIRS['data'] / 'stage2_prepared_data.csv'

# ============================================================================
# Model Selection for Nuisance Functions (DML)
# ============================================================================

# Default nuisance models (can be customized per treatment)
DEFAULT_MODEL_T = 'RandomForest'  # For propensity score E[T|X,W]
DEFAULT_MODEL_Y = 'RandomForest'  # For outcome regression E[Y|X,W]

# Model hyperparameters
NUISANCE_MODEL_PARAMS = {
    'RandomForest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_leaf': 20,
        'random_state': RANDOM_STATE
    },
    'GradientBoosting': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    }
}

# ============================================================================
# Diagnostics Thresholds
# ============================================================================

# Overlap/positivity check
MIN_PROPENSITY_SCORE = 0.05  # Trim units with extreme propensity scores
MAX_PROPENSITY_SCORE = 0.95

# E-value thresholds for interpretation
EVALUE_THRESHOLDS = {
    'weak': 1.5,    # Modest unmeasured confounding
    'moderate': 2.0,
    'strong': 3.0
}

# ============================================================================
# Prescription/Optimization Parameters
# ============================================================================

# Budget optimization
DEFAULT_BUDGET = None  # Will require user input

# Cost estimates (EUR per unit) - PLACEHOLDER, requires user input
# Keys must match the canonical treatment names in ALL_TREATMENTS.
TREATMENT_COSTS = {
    'Centreline rumble strips': None,
    'Delineation': None,
    'Street lighting': None,
    'Paved shoulder - driver-side': None,
    'Paved shoulder - passenger-side': None,
    'Road condition': None,
}

# Road selection criteria
MIN_ROAD_LENGTH_KM = 0.5  # Minimum road length for prescription
MIN_EXPECTED_REDUCTION = 0.1  # Minimum risk reduction to consider

# ============================================================================
# Reporting Configuration
# ============================================================================

# Number of decimal places in tables
TABLE_DECIMALS = 3

# Figure DPI
FIGURE_DPI = 300

# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Color palette for datasets / reporting groups
# Keys are Dataset IDs (no country names to comply with anonymisation).
DATASET_COLORS = {
    '980': '#1f77b4',
    '1240': '#ff7f0e',
    '1242': '#2ca02c',
    '1246': '#d62728',
    '1247': '#9467bd',
    '1398': '#8c564b',
    '1400': '#e377c2',
    '1424': '#7f7f7f',
    '1425': '#bcbd22',
    '1426': '#17becf',
    '12008': '#aec7e8',
    '12983': '#ffbb78',
}

# ============================================================================
# Feature Name Mapping (Stage 1 sanitization)
# ============================================================================

# Display-friendly names for the 6 validated treatments
FEATURE_DISPLAY_NAMES = {
    'Centreline rumble strips': 'Centreline Rumble Strips',
    'Delineation': 'Delineation',
    'Street lighting': 'Street Lighting',
    'Paved shoulder - driver-side': 'Paved Shoulder (Driver)',
    'Paved shoulder - passenger-side': 'Paved Shoulder (Passenger)',
    'Road condition': 'Road Condition',
}

# ============================================================================
# Validation Flags
# ============================================================================

# Enable/disable specific validation steps
ENABLE_OVERLAP_PLOTS = True
ENABLE_EVALUE_COMPUTATION = True
ENABLE_COUNTERMEASURE_VALIDATION = True
ENABLE_PLACEBO_TESTS = True
ENABLE_LEAVE_ONE_OUT = True

# ============================================================================
# Helper Functions
# ============================================================================

def create_output_dirs():
    """Create all output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in OUTPUT_SUBDIRS.values():
        subdir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created output directories in {OUTPUT_DIR}")


def get_latest_stage1_run():
    """Auto-detect the most recent Stage 1 run directory.

    Returns Path or None.  Delegates to ``_resolve_stage1_run`` with the
    manual override intentionally bypassed.
    """
    if not STAGE1_BASE_DIR.exists():
        return None
    run_dirs = sorted(
        [d for d in STAGE1_BASE_DIR.iterdir()
         if d.is_dir() and not d.name.startswith('.')],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def set_stage1_version(version: str = None, custom_path: str = None):
    """Switch Stage 1 run at runtime and update all dependent paths.

    Parameters
    ----------
    version : str, optional
        Run-directory name inside ``STAGE1_BASE_DIR``.
    custom_path : str, optional
        Full path to an arbitrary Stage 1 output directory.
    """
    global STAGE1_RUN_DIR, STAGE1_RUN_NAME_RESOLVED, OUTPUT_DIR
    global STAGE1_OOF_PREDICTIONS, STAGE1_HOTSPOT_OVERLAY
    global STAGE1_HOTSPOT_PROFILES, STAGE1_METRICS

    if custom_path:
        STAGE1_RUN_DIR = Path(custom_path)
    elif version:
        STAGE1_RUN_DIR = STAGE1_BASE_DIR / version
    else:
        latest = get_latest_stage1_run()
        STAGE1_RUN_DIR = latest if latest else STAGE1_BASE_DIR

    STAGE1_RUN_NAME_RESOLVED = STAGE1_RUN_DIR.name

    # Refresh file paths
    STAGE1_OOF_PREDICTIONS = STAGE1_RUN_DIR / 'fold_results' / 'oof_predictions_segments.csv'
    STAGE1_HOTSPOT_OVERLAY = STAGE1_RUN_DIR / 'fold_results' / 'hotspot_prediction_overlay.csv'
    STAGE1_HOTSPOT_PROFILES = STAGE1_RUN_DIR / 'fold_results' / 'all_road_high_risk_profiles.csv'
    STAGE1_METRICS = STAGE1_RUN_DIR / 'fold_results' / 'per_road_hotspot_metrics.csv'

    # Refresh output directory
    _label = STAGE1_RUN_NAME_RESOLVED
    _date = _label.split('_')[0] if '_' in _label else _label
    OUTPUT_DIR = Path(__file__).parent / 'stage2_outputs' / f"from_stage1_{_date}"

    # Refresh subdirectories
    for key in OUTPUT_SUBDIRS:
        OUTPUT_SUBDIRS[key] = OUTPUT_DIR / key

    return STAGE1_RUN_DIR


def validate_config():
    """Validate configuration settings."""
    issues = []
    
    # Check treatment costs are set (for prescription)
    if any(cost is None for cost in TREATMENT_COSTS.values()):
        issues.append("[WARN] Treatment costs not set (required for prescription module)")
    
    # Check Stage 1 directory exists
    if not STAGE1_RUN_DIR.exists():
        issues.append(f"[ERR] Stage 1 run directory not found: {STAGE1_RUN_DIR}")
    elif not STAGE1_OOF_PREDICTIONS.exists():
        issues.append(f"[WARN] Stage 1 OOF predictions not found: {STAGE1_OOF_PREDICTIONS}")
    
    # Check confidence level is valid
    if not 0 < CONFIDENCE_LEVEL < 1:
        issues.append(f"[ERR] Invalid confidence level: {CONFIDENCE_LEVEL}")
    
    if issues:
        print("\n".join(issues))
    else:
        print("[OK] Configuration validated successfully")
    
    return len(issues) == 0


if __name__ == '__main__':
    print("Stage 2 Configuration")
    print("=" * 60)
    print(f"Stage 1 run (resolved): {STAGE1_RUN_NAME_RESOLVED}")
    print(f"  Manual override:      '{STAGE1_RUN_NAME}' (empty = auto-detect)")
    print(f"Binary Treatments ({len(BINARY_TREATMENTS)}): {BINARY_TREATMENTS}")
    print(f"Ordinal Treatments ({len(ORDINAL_TREATMENTS)}): {ORDINAL_TREATMENTS}")
    print(f"Total Viable Treatments: {len(ALL_TREATMENTS)}")
    print(f"Control Features: {len(CONTROL_FEATURES)}")
    print(f"DML Folds: {DML_N_FOLDS}")
    print(f"Bootstrap Iterations: {DML_N_BOOTSTRAP}")
    print(f"Default Outcome Scale: {DEFAULT_OUTCOME}")
    print(f"Stage 1 OOF: {STAGE1_OOF_PREDICTIONS}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    validate_config()
    create_output_dirs()
    
    if STAGE1_OOF_PREDICTIONS.exists():
        print(f"\n[OK] Stage 1 OOF predictions found")
    else:
        print(f"\n[WARN] Stage 1 OOF predictions not found at:")
        print(f"   {STAGE1_OOF_PREDICTIONS}")
