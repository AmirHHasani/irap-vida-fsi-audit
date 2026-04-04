"""
stage1_config.py
Central configuration for Stage 1: Interpretable Road Risk Modeling

Organized into logical sections: base paths, identifiers, features,
modeling, CV/hotspot settings, interpretability (SHAP), countermeasures,
outputs, and misc flags. Keep values stable across runs for reproducibility.
"""
from pathlib import Path

# -----------------------------------------------------------------------------
# 0. Base paths
# -----------------------------------------------------------------------------
_BASE_DIR = Path(__file__).parent.parent
# Canonical segments CSV path (input)
INPUT_DATA_DIR = _BASE_DIR / 'input_data'
SEGMENTS_DATA_CSV = INPUT_DATA_DIR / 'segments_unique.csv'
# Main outputs directory (each run creates a timestamped subfolder)
OUTPUT_DIR = Path('stage1_outputs/')

# -----------------------------------------------------------------------------
# 1. Identifiers & metadata
# -----------------------------------------------------------------------------
# Primary unique identifier in the dataset
ID_COL = 'Location ID'
# Canonical integer index used throughout artifacts and explanation pipelines
# This is the authoritative positional index (row in the original segments DataFrame)
CANONICAL_INDEX_COL = 'global_index'
# Column used to group data by road for grouped CV and reporting
ROAD_COLUMN_NAME = 'Road name'
# Target variable (authoritative name used across the pipeline)
TARGET_COL = 'Total Fatal and Serious Injury Estimation per 100m per year'
# Columns that are metadata/identifiers and should not be used as features
METADATA_COLS = [
    'Location ID', 'Road name', 'Section', 'Distance', 'Latitude', 'Longitude',
    'End Latitude', 'End Longitude', 'Smoothed Section ID', 'Road'
]

# -----------------------------------------------------------------------------
# 2. Feature lists & exclusions
# -----------------------------------------------------------------------------
NUMERICAL_FEATURES = [
    'Vehicle flow (AADT)'
]
CATEGORICAL_FEATURES = [
    'Dataset ID', 'Upgrade cost', 'Bicycle peak hour flow', 'Motorcycle %', 'Pedestrian peak hour flow across the road', 'Pedestrian peak hour flow along the road driver-side',
    'Pedestrian peak hour flow along the road passenger-side', 'Operating Speed (85th percentile)',
    'Area type', 'Carriageway', 'Centreline rumble strips', 'Curvature', 'Delineation', 'Differential speed limits',
    'Facilities for bicycles', 'Facilities for motorised two wheelers', 'Grade', 'Intersection channelisation',
    'Intersection quality', 'Quality of curve', 'Number of lanes', 'Intersection type', 'Land use - driver-side', 'Land use - passenger-side', 'Lane width',
    'Median type', 'Paved shoulder - driver-side', 'Paved shoulder - passenger-side', 'Pedestrian crossing facilities - inspected road',
    'Pedestrian crossing facilities - intersecting road', 'Pedestrian crossing quality', 'Pedestrian fencing',
    'Property access points', 'Road condition', 'Roadside severity - driver-side distance',
    'Roadside severity - driver-side object', 'Roadside severity - passenger-side distance', 'Roadside severity - passenger-side object',
    'Roadworks', 'School zone crossing supervisor', 'School zone warning', 'Service road', 'Shoulder rumble strips',
    'Sidewalk - driver-side', 'Sidewalk - passenger-side', 'Sight distance', 'Skid resistance / grip', 'Speed limit',
    'Speed management / traffic calming', 'Street lighting', 'Vehicle parking',
    'Motorcycle observed flow', 'Motorcycle speed limit', 'Bicycle observed flow', 'Intersecting road volume', 'Pedestrian observed flow across the road',
    'Pedestrian observed flow along the road driver-side', 'Pedestrian observed flow along the road passenger-side',
    'Truck speed limit'
]

#'Dataset ID',

# CATEGORICAL_FEATURES = [
#     'Upgrade cost', #'Bicycle peak hour flow', 'Motorcycle %', 'Pedestrian peak hour flow across the road', 'Pedestrian peak hour flow along the road driver-side',
#     #'Pedestrian peak hour flow along the road passenger-side', 'Operating Speed (85th percentile)',
#     'Area type', 'Carriageway', 'Centreline rumble strips', 'Curvature', 'Delineation', 'Differential speed limits',
#     'Facilities for bicycles', 'Facilities for motorised two wheelers', 'Grade', 'Intersection channelisation',
#     'Intersection quality', 'Quality of curve', 'Number of lanes', 'Intersection type', 'Land use - driver-side', 'Land use - passenger-side', 'Lane width',
#     'Median type', 'Paved shoulder - driver-side', 'Paved shoulder - passenger-side', 'Pedestrian crossing facilities - inspected road',
#     'Pedestrian crossing facilities - intersecting road', 'Pedestrian crossing quality', 'Pedestrian fencing',
#     'Property access points', 'Road condition', 'Roadside severity - driver-side distance',
#     'Roadside severity - driver-side object', 'Roadside severity - passenger-side distance', 'Roadside severity - passenger-side object',
#     'Roadworks', 'School zone crossing supervisor', 'School zone warning', 'Service road', 'Shoulder rumble strips',
#     'Sidewalk - driver-side', 'Sidewalk - passenger-side', 'Sight distance', 'Skid resistance / grip', 'Speed limit',
#     'Speed management / traffic calming', 'Street lighting', 'Vehicle parking',
#     'Motorcycle observed flow', 'Motorcycle speed limit', 'Bicycle observed flow', 'Intersecting road volume', 'Pedestrian observed flow across the road',
#     'Pedestrian observed flow along the road driver-side', 'Pedestrian observed flow along the road passenger-side',
#     'Truck speed limit'
# ]

# -----------------------------------------------------------------------------
# 3. Target transformation controls
# -----------------------------------------------------------------------------
# Method options: 'log1p', 'log_offset', 'yeo_johnson', 'none'
TARGET_TRANSFORMATION = {
    'method': 'log_offset',
    # Used when method == 'log_offset'; percentile-derived offsets recommended
    'offset': 0.001,
    # When using yeo_johnson, standardization can be toggled (False keeps scale)
    'yeo_johnson_standardize': False
}

# -----------------------------------------------------------------------------
# 4. Cross-validation stratification controls
# -----------------------------------------------------------------------------
# Options: 'target_mean' (current behavior), 'proxy_dataset', 'proxy_aadt', 'none'
CV_STRATIFICATION_MODE = 'target_mean'
CV_STRATIFICATION_BINS = 5  # number of quantile bins when applicable

FEATURE_EXCLUSIONS = [
    #'Vehicle flow (AADT)',
    'Operating Speed (mean)', 'Roads that cars can read', 'Length', 'Bicycle Star Rating Policy Target', 'Bicyclist Fatality Estimation Along per km per year',
    'Bicyclist Fatality Estimation Intersection per km per year ', 'Bicyclist Fatality Estimation Run-Off per km per year',
    'Bicyclist Fatality Estimation Total per km per year', 'Bicyclist SRS Along', 'Bicyclist SRS Intersection',
    'Bicyclist SRS Run-Off', 'Bicyclist SRS Total', 'Bicyclist SRS Total Smoothed', 'Bicyclist Star Rating Raw',
    'Bicyclist Star Rating Smoothed', 'Distance', 'End Latitude', 'End Longitude', 'Latitude',
    'Location ID', 'Longitude', 'Pedestrian Star Rating Policy Target', 'Pedestrian SRS Along', 'Pedestrian SRS Intersection',
    'Pedestrian SRS Run-Off', 'Pedestrian SRS Total', 'Pedestrian SRS Total Smoothed', 'Pedestrian Star Rating Raw',
    'Pedestrian Star Rating Smoothed', 'Road', 'Road name', 'Section', 'Smoothed Section ID', 'Star Rating (All)',
    'Star Rating (Bicycle)', 'Star Rating (Motorcycle)', 'Star Rating (Pedestrian)', 'Star Rating (Vehicle Occupant)',
    'Vehicle Occupant Star Rating Policy Target', 'Vehicle Occupant SRS Along', 'Vehicle Occupant SRS Intersection',
    'Vehicle Occupant SRS Run-Off', 'Vehicle Occupant SRS Total', 'Vehicle Occupant SRS Total Smoothed',
    'Vehicle Occupant Star Rating Raw', 'Vehicle Occupant Star Rating Smoothed', 'Motorcycle Star Rating Policy Target',
    'Motorcycle SRS Along', 'Motorcycle SRS Intersection', 'Motorcycle SRS Run-Off', 'Motorcycle SRS Total',
    'Motorcycle SRS Total Smoothed', 'Motorcycle Star Rating Raw', 'Motorcycle Star Rating Smoothed',
    'Star Rating Score (All)', 'Star Rating Score (Bicycle)', 'Star Rating Score (Motorcycle)',
    'Star Rating Score (Pedestrian)', 'Star Rating Score (Vehicle Occupant)',
    'Total Fatal and Serious Injury Estimation per 100m per year', 'Total Fatal and Serious Injury Estimation per km per year',
    'Total FSI (All)', 'Total FSI (Bicycle)', 'Total FSI (Motorcycle)', 'Total FSI (Pedestrian)', 'Total FSI (Vehicle Occupant)',
    'Total SRS (All)', 'Total SRS (Bicycle)', 'Total SRS (Motorcycle)', 'Total SRS (Pedestrian)', 'Total SRS (Vehicle Occupant)',
    'Total Fatality Estimation per 100m per year', 'Total Fatality Estimation per km per year',
    'Vulnerable Road User Fatality Estimation per km per year', 'Motorcyclist Fatality Estimation Total per km per year',
    'Pedestrian Fatality Estimation Total per km per year'
]

# -----------------------------------------------------------------------------
# 2.5 Dataset ID (Country × Road Type Fixed Effects)
# -----------------------------------------------------------------------------
# Dataset ID represents country × road type combinations and acts as a fixed effect
# to control for dataset-level heterogeneity (infrastructure standards, enforcement,
# driving culture, etc.). This improves model validity by ensuring feature effects
# are estimated "within-dataset" rather than confounded with country-level differences.
DATASET_ID_COL = 'Dataset ID'
# Set to True to include Dataset ID as a model feature (controls for dataset heterogeneity)
# Set to False to exclude it (keeps it only as metadata for analysis/grouping)
INCLUDE_DATASET_ID_AS_FEATURE = True
TREAT_DATASET_AS_CATEGORICAL = True
DATASET_ID_ENCODING = 'ordinal'  # For tree models; keeps memory efficient
# Exclude Dataset ID from SHAP feature importance displays (it's a control, not interpretable)
EXCLUDE_DATASET_FROM_SHAP_REPORTS = True

# -----------------------------------------------------------------------------
# 3. Modeling & evaluation parameters
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Candidate models (order for evaluation)
CANDIDATE_MODELS = [
    'CatBoost', 'LightGBM', 'XGBoost'
]

# Primary model used for some single-model flows
PRIMARY_MODEL_TYPE = 'CatBoost'

# -----------------------------------------------------------------------------
# 4. Hotspot selection & CV-specific settings
# -----------------------------------------------------------------------------
SPLIT_STRATEGY = 'BY_ROAD'  # Options: 'RANDOM','BY_ROAD','DIAGNOSTIC'

HOTSPOT_K = 3
HOTSPOT_K_LIST = [1, 3, 5]
MAX_HOTSPOT_SEGMENTS_PER_ROAD = None

# Bootstrap iterations for kappa / overlap CI computation (Step 2b)
BOOTSTRAP_N_ITERATIONS = 2000

# Directory name for hotspot SHAP outputs (under run output)
HOTSPOT_SHAP_DIR_NAME = 'hotspot_shap'

# -----------------------------------------------------------------------------
# 5. SHAP / interpretability settings
# -----------------------------------------------------------------------------
COMPUTE_HOTSPOT_SHAP = True
COMPUTE_SHAP_IN_MODEL_SELECTION = False  # kept for compatibility
SAVE_PER_FOLD_HOTSPOT_SHAP = True
SAVE_AGGREGATED_HOTSPOT_SHAP = True
HOTSPOT_SHAP_FEATURE_IMPORTANCE_CSV = 'hotspot_shap_feature_importance.csv'
N_TOP_FEATURES_SHAP = 20
TOP_N_SHAP_FEATURES_PER_SEGMENT = 20
TOP_N_SEGMENTS_FOR_EXPLANATION = 3
TOP_N_ROADS_FOR_EXPLANATION = 20
# Whether to attempt consolidation of per-fold hotspot SHAP into a per-road CSV
SAVE_PER_ROAD_HOTSPOT_SHAP = True


# Enhanced Artifact Naming for Traceability
USE_ENHANCED_NAMING = True  # Enable structured naming with road and segment IDs
SANITIZE_ROAD_NAMES = True  # Replace invalid filename characters in road names
# -----------------------------------------------------------------------------
# 6. Countermeasure overlay settings
# -----------------------------------------------------------------------------
ENABLE_COUNTERMEASURE_OVERLAY = True
COUNTERMEASURE_DATA_CSV = INPUT_DATA_DIR / 'countermeasures.csv'
COUNTERMEASURE_ID_COL = 'Location ID'  # join key in countermeasure CSV
COUNTERMEASURE_TEXT_COL = 'Countermeasure'  # text column with countermeasure description (canonical name in CSV)
HOTSPOT_OVERLAY_CSV = 'hotspot_prediction_overlay.csv'
COUNTERMEASURE_COVERAGE_CSV = 'countermeasure_coverage_summary.csv'
COUNTERMEASURE_MIN_OCCURRENCES_LOG = 'countermeasure_occurrence_counts.csv'
# Column used to show more detailed countermeasure metadata where available
# Canonical columns present in the countermeasure CSV: 'Countermeasure Pack',
# 'Countermeasure Summary Group'. Set the preferred detailed column here.
COUNTERMEASURE_DETAIL_COL = 'Countermeasure Pack'

# -----------------------------------------------------------------------------
# 7. Output filenames and directories (per-run)
# -----------------------------------------------------------------------------
FINAL_MARKDOWN_REPORT_FILE = "final_analysis_report.md"
ROAD_EXPLANATION_PREFIX = 'road_'
SEGMENT_EXPLANATION_PREFIX = 'segment_'
EXPLANATION_HTML_REPORT = 'explanations_report.html'
ROAD_EXPLANATION_SUMMARY_CSV = 'road_explanation_summary.csv'
SEGMENT_EXPLANATION_SUMMARY_CSV = 'segment_explanation_summary.csv'
PER_ROAD_METRICS_CSV = 'per_road_hotspot_metrics.csv'
RANKING_METRICS_LONG_CSV = 'per_road_hotspot_metrics_long.csv'
RANKING_METRICS_AGG_JSON = 'hotspot_ranking_metrics_aggregated.json'
PER_ROAD_HOTSPOT_SHAP_CSV = 'per_road_hotspot_shap.csv'

# publication exports
EXPORT_PUBLICATION_TABLES = True
PUBLICATION_EXPORT_DIR = 'publication_exports'
PUB_TOP_FEATURES_CSV = 'pub_ranking_agg.csv'
PUB_RANKING_AGG_CSV = 'pub_ranking_agg.csv'
PUB_COVERAGE_CSV = 'pub_countermeasure_coverage.csv'

# -----------------------------------------------------------------------------
# 8. Visualization & mapping settings
# -----------------------------------------------------------------------------
TOP_N_HOTSPOTS = 3
GENERATE_ROAD_EXPLANATIONS = True
GENERATE_SEGMENT_EXPLANATIONS = True
SAVE_EXPLANATION_HTML_REPORTS = True
ROAD_VALIDATION_MAPS_DIR = 'road_validation_maps'
VALIDATION_MAP_TOP_K = None
GENERATE_ROAD_VALIDATION_MAPS = True  # Generate side-by-side ground truth vs predicted hotspots maps

# -----------------------------------------------------------------------------
# 9. Reproducibility, performance & misc
# -----------------------------------------------------------------------------
WRITE_RUN_MANIFEST = True
RUN_MANIFEST_FILE = 'run_manifest.json'
CAPTURE_LIBRARY_VERSIONS = True
USE_GPU = True
GPU_DEVICE_ID = 0
GPU_ENABLE_CATBOOST = True
GPU_ENABLE_LIGHTGBM = True
GPU_ENABLE_XGBOOST = True
AUTO_DISABLE_GPU_MIN_ROWS = 15000

ENABLE_ENCODER_CACHE = True
ENCODER_CACHE_MAXSIZE = 1000

# -----------------------------------------------------------------------------
# 10. Reporting & testing helpers
# -----------------------------------------------------------------------------
COUNTERMEASURE_OVERLAY_MAP_HTML = 'hotspot_prediction_overlay_map.html'
COUNTERMEASURE_OVERLAY_MAP_PNG = 'hotspot_prediction_overlay_map.png'

# Filename for interactive hotspot overlay map
HOTSPOT_OVERLAY_MAP_HTML = 'hotspot_prediction_overlay_map.html'

# Explanation integrity and alignment controls
# Enforce exact match between stored y_pred and live prediction when generating explanations
ENFORCE_PREDICTION_MATCH = True
# Relative difference tolerance for prediction equality (only used if ENFORCE_PREDICTION_MATCH)
EXPLANATION_PREDICTION_MISMATCH_TOLERANCE = 0.01
# Allow using per-fold artifacts if available; otherwise skip (global proxy mode disabled by default)
ALLOW_GLOBAL_PROXY_EXPLANATIONS = False
# For legacy/debug only: allow positional alignment as a last resort when key joins fail (not recommended)
ALLOW_POSITIONAL_ALIGNMENT = False

# -----------------------------------------------------------------------------
# 11. Scientific extras: IDs, uncertainty, calibration (optional)
# -----------------------------------------------------------------------------
# Enable creation of a collision-proof composite ID (Dataset ID + Location ID)
ENABLE_COMPOSITE_ID = False
COMPOSITE_ID_COL = 'composite_segment_id'
COMPOSITE_ID_FORMAT = '{dataset}:{loc}'  # how to join Dataset ID and Location ID

# Conformal prediction intervals on OOF predictions (regression)
ENABLE_CONFORMAL_INTERVALS = True
CONFORMAL_ALPHA = 0.1  # 90% intervals by default

# Calibration plotting for OOF regression predictions
ENABLE_CALIBRATION_PLOT = True
CALIBRATION_NUM_BINS = 10

# -----------------------------------------------------------------------------
# 12. Regional SHAP Analysis Configuration
# -----------------------------------------------------------------------------
# Dataset to country mapping
# DATASET_COUNTRY_MAP = {

#     #'code': 'country_abbreviation'
# }

DATASET_COUNTRY_MAP = {
    '12008': 'MNE',   # Montenegro
    '1240': 'SLO',    # Slovenia
    '1242': 'SLO',    # Slovenia
    '1246': 'BIH',    # Bosnia and Herzegovina
    '1247': 'BIH',    # Bosnia and Herzegovina
    '12983': 'GRC',   # Greece
    '1398': 'BGR',    # Bulgaria
    '1400': 'ROU',    # Romania
    '1424': 'HRV',    # Croatia
    '1425': 'HRV',    # Croatia
    '1426': 'HRV',    # Croatia
    '980': 'UKR',     # Ukraine
}
# -----------------------------------------------------------------------------
# regional groupings:
# -----------------------------------------------------------------------------
# Rationale: group by broadly similar road design/maintenance regimes and network
# maturity, rather than strict geographic labels.
# REGIONAL_GROUPINGS = {
#         # 'name': {
#         #     'name': 'Ename',
#         #     'datasets': ['id', 'id', ...], 
#         #     'countries': 'country name1, country name2, ...',
#         #     'description': 'as needed'
#         # },
#         # other regions...
# }
REGIONAL_GROUPINGS = {
    'EU_Central_Adriatic': {
        'name': 'EU Central/Adriatic',
        'datasets': ['1240', '1242', '1424', '1425', '1426'],  # SLO + HRV
        'countries': 'Slovenia, Croatia',
        'description': 'SLO (2 datasets), HRV (3 datasets)'
    },
    'NonEU_Western_Balkans': {
        'name': 'Western Balkans (non-EU)',
        'datasets': ['1246', '1247', '12008'],  # BIH + MNE
        'countries': 'Bosnia and Herzegovina, Montenegro',
        'description': 'BIH (2 datasets), MNE (1 dataset)'
    },
    'EU_Southeast_Balkan': {
        'name': 'EU Southeast Europe',
        'datasets': ['1398', '1400', '12983'],  # BGR + ROU + GRC
        'countries': 'Bulgaria, Romania, Greece',
        'description': 'BGR (1 dataset), ROU (1 dataset), GRC (1 dataset)'
    },
    'Eastern_Europe': {
        'name': 'Eastern Europe',
        'datasets': ['980'],  # UKR
        'countries': 'Ukraine',
        'description': 'UKR (1 dataset)'
    },
}


# -----------------------------------------------------------------------------
# Utility functions used by modules (kept minimal)
# -----------------------------------------------------------------------------

def save_plot(fig, filename, directory=OUTPUT_DIR):
    """Saves a matplotlib figure to the specified output directory.

    Accepts either a Path or a string for `directory`.
    Ensures the directory exists before saving.
    """
    try:
        base_dir = Path(directory) if not isinstance(directory, Path) else directory
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / filename
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"   Plot saved to: {path}")
    except Exception as e:
        print(f"   Warning: Could not save plot to {directory}/{filename}. Error: {e}")