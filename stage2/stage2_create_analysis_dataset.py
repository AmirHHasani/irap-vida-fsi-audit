"""
Create analysis_dataset.csv by merging Stage 1 predictions with input features.
This file is needed by downstream Stage 2 analysis scripts.
"""
import pandas as pd
from pathlib import Path
from stage2_config import STAGE1_OOF_PREDICTIONS, OUTPUT_DIR, INPUT_DATA_DIR
from stage2_treatment_codebook import apply_canonical_treatment_mapping, write_mapping_artifact

print("="*70)
print("CREATING ANALYSIS_DATASET.CSV")
print("="*70)

# Load Stage 1 predictions (outcome + segment IDs)
print("\n1. Loading Stage 1 predictions...")
stage1_predictions = pd.read_csv(STAGE1_OOF_PREDICTIONS)
print(f"   [OK] Loaded: {len(stage1_predictions):,} segments")

# Load Phase 2 sensor features (all treatment + control features)
print("\n2. Loading sensor features...")
sensor_data = pd.read_csv(
    INPUT_DATA_DIR / 'segments_unique.csv',
    low_memory=False
)
print(f"   [OK] Loaded: {len(sensor_data):,} segments")

# Merge on segment_id
print("\n3. Merging datasets...")
sensor_data['segment_id'] = sensor_data['Location ID']

# Select columns from Stage 1
stage1_cols = ['segment_id', 'actual_risk', 'predicted_risk']
if 'Dataset ID' in stage1_predictions.columns:
    stage1_cols.append('Dataset ID')

# Merge
analysis_data = sensor_data.merge(
    stage1_predictions[stage1_cols],
    on='segment_id',
    how='inner'
)
print(f"   [OK] Merged: {len(analysis_data):,} segments")

# Handle Dataset ID conflicts
if 'Dataset ID_x' in analysis_data.columns:
    analysis_data['Dataset ID'] = analysis_data['Dataset ID_x']
    analysis_data.drop(['Dataset ID_x', 'Dataset ID_y'], axis=1, inplace=True, errors='ignore')
    print(f"   [OK] Resolved Dataset ID column")

# Canonical treatment mapping (centralized)
print("\n4. Applying canonical treatment mapping...")
from stage2_config import BINARY_TREATMENTS, ORDINAL_TREATMENTS

all_treatments = BINARY_TREATMENTS + ORDINAL_TREATMENTS
mapping_report = apply_canonical_treatment_mapping(
    analysis_data,
    all_treatments,
    strict=True,
    keep_raw=False,
)

mapped = list(mapping_report.get("treatments", {}).keys())
print(f"   [OK] Canonical-mapped {len(mapped)} treatment columns")
for name, info in mapping_report.get("treatments", {}).items():
    print(f"   {name}: raw {info['raw_unique']} -> canonical {info['mapped_unique']}")

# Save to output directory
print("\n5. Saving analysis_dataset.csv...")
data_dir = OUTPUT_DIR / 'data'
data_dir.mkdir(parents=True, exist_ok=True)

# Persist mapping metadata for reproducibility
mapping_path = data_dir / 'treatment_level_mapping_used.json'
write_mapping_artifact(mapping_report, mapping_path)
print(f"   [OK] Saved mapping metadata: {mapping_path}")

output_path = data_dir / 'analysis_dataset.csv'
analysis_data.to_csv(output_path, index=False)
print(f"   [OK] Saved to: {output_path}")
print(f"   Shape: {analysis_data.shape}")

print("\n" + "="*70)
print("[OK] COMPLETE! analysis_dataset.csv is ready for Phase 2")
print("="*70)
print(f"\nFile location: {output_path}")
print(f"Size: {len(analysis_data):,} segments x {analysis_data.shape[1]} features")
print(f"\nYou can now run: python stage2_hierarchical_cf.py --run-id ...")
