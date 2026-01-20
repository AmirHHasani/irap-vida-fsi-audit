# A Two‑Stage Framework for Modelled Road Safety

<p align="center">
  <img src="flowchart.png" alt="Two-stage workflow flowchart" width="500" height="450" />
</p>

This repository implements a two‑stage analysis pipeline for modelled road safety:

- **Stage 1 — Prediction:** predictive modelling and risk estimation (feature engineering, model training, evaluation).
- **Stage 2 — Causal analysis:** creation of analysis datasets, causal forest estimation, and prescription generation.

## Repository structure

```
stage1/
  stage1_main.py
  stage1_config.py
  stage1_data_loader.py
  stage1_feature_engineering.py
  stage1_model_training.py
  stage1_visualizations.py
  stage1_reporting.py
  stage1_utils.py
stage2/
  stage2_config.py
  stage2_create_analysis_dataset.py
  stage2_hierarchical_cf.py
  stage2_cf_prescription_generation.py
  stage2_treatment_codebook.py
README.md
requirements-stage1.txt
requirements-stage2.txt
flowchart.png
```

## Quick start

1. Create and activate a Python environment for the stage you want to run.
2. Install dependencies:

```bash
# Stage 1
pip install -r requirements-stage1.txt

# Stage 2
pip install -r requirements-stage2.txt
```

3. Configure input paths and parameters in `stage1/stage1_config.py` or `stage2/stage2_config.py`.
4. Run the appropriate entry point:

```bash
python stage1/stage1_main.py    # run predictive modelling pipeline
python stage2/stage2_cf_prescription_generation.py    # run causal prescription generation
```

## Usage notes

- Edit the config files (`stage1/stage1_config.py`, `stage2/stage2_config.py`) to point to input data and set runtime options.
- Intermediate files and outputs are written to the locations defined in the config files; check those paths before running long jobs.
- For hyperparameter tuning traces and results, see `input_data/stage1_tuning/` subfolders.

## Data Availability

The source datasets used in this project are provided by the International Road Assessment Programme (iRAP) and are available from https://vida.irap.org/en-gb/home.

## Citation

If you use this repository or methods from it, please cite the corresponding work.
