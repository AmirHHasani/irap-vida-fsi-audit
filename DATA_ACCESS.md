# Data Access Note

This repository accompanies the paper:

*Auditing iRAP's ViDA Risk Engine: A Two-Stage Surrogate Learning and Orthogonalized Heterogeneity Framework for Modelled Road Safety*

## What is not redistributed

The study uses exported segment-level iRAP ViDA data. Those files are not redistributed here because access is subject to iRAP licensing and project-specific permissions.

This includes the main segment-level and countermeasure exports used by the code.

## Expected local input files

The current code expects the following local files:

- `input_data/segments_unique.csv`
- `input_data/countermeasures.csv`

Some working scripts also refer to large merge-time CSV files under `combine_data/`, but those are not required for a minimal paper-code release.

## Included support files

The repository may include small support metadata files that are part of the code logic, for example:

- `stage2/codes_filled_analysis.csv`
- `stage2/dataset_regional_mapping.csv`

These are code-support files, not substitutes for the licensed ViDA exports.

## What the data represent

In the paper, the primary outcome is the exported **ViDA modelled fatal-and-serious-injury (FSI) estimate** at the 100 m segment level. The repository documentation follows that same interpretation.

## Access route

If you need the underlying iRAP data, access must be requested through iRAP and any relevant project permissions. Please follow the applicable iRAP licensing and acknowledgment requirements.

## Reproducibility note

Because the licensed input data are not redistributed here, full end-to-end reruns require an authorized local copy of the expected input files placed in the paths referenced by:

- `stage1/stage1_config.py`
- `stage2/stage2_config.py`
