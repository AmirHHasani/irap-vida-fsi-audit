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
# Paths - VALIDATED OCTOBER 7, 2025
# ============================================================================

# Stage 1 outputs (OOF predictions - 147,466 segments)
# UPDATED: October 25, 2025 - Latest Stage 1 run
STAGE1_OUTPUT_DIR = Path(__file__).parent.parent / 'stage1_outputs'
LEGACY_STAGE1_OUTPUT_DIR = STAGE1_OUTPUT_DIR
STAGE1_BASE_DIR = STAGE1_OUTPUT_DIR
STAGE1_RUN_NAME = '2025-11-29_09-58-47_BY_ROAD'  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
STAGE1_RUN_DIR = STAGE1_BASE_DIR / STAGE1_RUN_NAME

# Key Stage 1 files
STAGE1_OOF_PREDICTIONS = STAGE1_RUN_DIR / 'fold_results' / 'oof_predictions_segments.csv'
STAGE1_HOTSPOT_OVERLAY = STAGE1_RUN_DIR / 'fold_results' / 'hotspot_prediction_overlay.csv'  # Segment-level (556 hotspots)
STAGE1_HOTSPOT_PROFILES = STAGE1_RUN_DIR / 'fold_results' / 'all_road_high_risk_profiles.csv'  # Road-level SHAP (132 roads)
STAGE1_METRICS = STAGE1_RUN_DIR / 'fold_results' / 'per_road_hotspot_metrics.csv'

# Phase 2 data directory (feature data - 147,466 segments × 115 columns)
INPUT_DATA_DIR = Path(__file__).parent.parent / 'input_data'
PHASE2_DATA_DIR = INPUT_DATA_DIR
SEGMENTS_DATA_CSV = INPUT_DATA_DIR / 'segments_unique.csv'

# Stage 2 outputs
# Organized by Stage 1 run name for traceability
OUTPUT_DIR = Path(__file__).parent / 'stage2_outputs'
if STAGE1_RUN_NAME:
    # Use shortened version of run name (date only)
    run_date = STAGE1_RUN_NAME.split('_')[0] if '_' in STAGE1_RUN_NAME else 'default'
    OUTPUT_DIR = OUTPUT_DIR / f"from_stage1_{run_date}"
else:
    OUTPUT_DIR = OUTPUT_DIR / "from_stage1_default"

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

# Cost estimates (€ per unit) - PLACEHOLDER, requires user input
TREATMENT_COSTS = {
    'speed_limit': None,      # € per sign per km
    'street_lighting': None,  # € per pole per km
    'sidewalks': None,        # € per m² or per km
    'crossings': None,        # € per crossing
    'rumble_strips': None     # € per km
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

# Color palette for countries
COUNTRY_COLORS = {
    'BiH': '#1f77b4',
    'Serbia': '#ff7f0e',
    'Macedonia': '#2ca02c',
    # Add more as needed
}

# ============================================================================
# Feature Name Mapping (Stage 1 sanitization)
# ============================================================================

# Stage 1 uses sanitized names (lowercase, underscores)
# This mapping helps reconnect to original feature names if needed
FEATURE_DISPLAY_NAMES = {
    'speed_limit': 'Speed Limit',
    'street_lighting': 'Street Lighting',
    'sidewalks': 'Sidewalks',
    'crossings': 'Pedestrian Crossings',
    'rumble_strips': 'Rumble Strips',
    'aadt': 'AADT (Traffic Volume)',
    'area_type': 'Area Type',
    'land_use': 'Land Use',
    'curvature': 'Curvature',
    'grade': 'Grade',
    'intersection_type': 'Intersection Type'
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
    print(f"✓ Created output directories in {OUTPUT_DIR}")


def get_latest_stage1_run():
    """
    Auto-detect the most recent Stage 1 run directory.
    Returns Path object or None if not found.
    """
    if not STAGE1_BASE_DIR.exists():
        return None
    
    # Look for timestamped run directories or version folders
    run_dirs = [d for d in STAGE1_BASE_DIR.iterdir() 
                if d.is_dir() and not d.name.startswith('.')]
    if not run_dirs:
        return None
    
    # Sort by modification time (most recent first)
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return run_dirs[0]


def set_stage1_version(version: str = None, custom_path: str = None):
    """
    Set Stage 1 version/run to use for analysis.
    
    Parameters
    ----------
    version : str, optional
        Version name (e.g., 'v1_xgboost', 'v2_tuned')
    custom_path : str, optional
        Full custom path to Stage 1 outputs
    
    Returns
    -------
    Path
        The Stage 1 run directory to use
    """
    global STAGE1_RUN_DIR, OUTPUT_DIR, STAGE1_VERSION
    
    if custom_path:
        STAGE1_RUN_DIR = Path(custom_path)
        STAGE1_VERSION = Path(custom_path).name
    elif version:
        STAGE1_RUN_DIR = STAGE1_BASE_DIR / version
        STAGE1_VERSION = version
    else:
        # Auto-detect latest
        latest = get_latest_stage1_run()
        if latest:
            STAGE1_RUN_DIR = latest
            STAGE1_VERSION = latest.name
        else:
            STAGE1_RUN_DIR = STAGE1_BASE_DIR
            STAGE1_VERSION = "default"
    
    # Update output directory to track Stage 1 version
    OUTPUT_DIR = Path(__file__).parent / 'stage2_outputs' / f"from_stage1_{STAGE1_VERSION}"
    
    # Update all output subdirectories
    OUTPUT_SUBDIRS['ate_results'] = OUTPUT_DIR / 'ate_results'
    OUTPUT_SUBDIRS['dose_response'] = OUTPUT_DIR / 'dose_response'
    OUTPUT_SUBDIRS['heterogeneity'] = OUTPUT_DIR / 'heterogeneity'
    OUTPUT_SUBDIRS['diagnostics'] = OUTPUT_DIR / 'diagnostics'
    OUTPUT_SUBDIRS['prescriptions'] = OUTPUT_DIR / 'prescriptions'
    OUTPUT_SUBDIRS['reports'] = OUTPUT_DIR / 'reports'
    OUTPUT_SUBDIRS['data'] = OUTPUT_DIR / 'data'
    
    return STAGE1_RUN_DIR


def validate_config():
    """Validate configuration settings."""
    issues = []
    
    # Check treatment costs are set (for prescription)
    if any(cost is None for cost in TREATMENT_COSTS.values()):
        issues.append("⚠️  Treatment costs not set (required for prescription module)")
    
    # Check Stage 1 directory exists
    if not STAGE1_BASE_DIR.exists():
        issues.append(f"⚠️  Stage 1 base directory not found: {STAGE1_BASE_DIR}")
    
    # Check confidence level is valid
    if not 0 < CONFIDENCE_LEVEL < 1:
        issues.append(f"❌ Invalid confidence level: {CONFIDENCE_LEVEL}")
    
    if issues:
        print("\n".join(issues))
    else:
        print("✓ Configuration validated successfully")
    
    return len(issues) == 0


if __name__ == '__main__':
    print("Stage 2 Configuration - USER APPROVED")
    print("=" * 60)
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
        print(f"\n✓ Stage 1 OOF predictions found")
    else:
        print(f"\n⚠️  Stage 1 OOF predictions not found")

