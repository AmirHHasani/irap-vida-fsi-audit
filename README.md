# Auditing iRAP's ViDA Risk Engine

<p align="center">
  <img src="flowchart.png" alt="Two-stage workflow flowchart" width="500" height="450" />
</p>

This repository accompanies the paper:

*Auditing iRAP's ViDA Risk Engine: A Two-Stage Surrogate Learning and Orthogonalized Heterogeneity Framework for Modelled Road Safety*

The repository is aligned with the latest accepted manuscript used for production.

## What the paper does

The study audits the **modelled fatal-and-serious-injury (FSI) output** produced by iRAP ViDA. It does **not** estimate treatment effects on observed crashes.

The analysis has two stages:

1. **Stage 1: surrogate modelling of the ViDA surface**
   - Gradient-boosted tree models are trained to reproduce ViDA's exported segment-level modelled FSI.
   - Out-of-fold predictions are used to assess emulation fidelity and hotspot retrieval.

2. **Stage 2: orthogonalized heterogeneity analysis**
   - A double-machine-learning causal forest is fit on the exported ViDA surface.
   - The resulting contrasts are interpreted as **conditional associations on the modelled ViDA surface**, not as causal effects on observed crash outcomes.
   - Simple absolute and relative thresholds are then used to convert those associations into **association-based candidate upgrades**.

The comparison with SRIP in the paper is therefore a comparison between:
- rule-based iRAP recommendations, and
- data-driven association-based prescriptions defined on the exported ViDA model output.

## Repository contents

The main code is organized as follows:

```text
stage1/
  stage1_main.py
  stage1_tuning.py
  stage1_reporting.py
  ...

stage2/
  stage2_create_analysis_dataset.py
  stage2_hierarchical_cf.py
  stage2_cf_prescription_generation.py
  stage2_diagnostics.py
  stage2_treatment_codebook.py
  ...

reporters/reporters/
  create_irap_concordance_v27.py
  create_table8_prevalence_v27.py
  create_fig3_cate_hotspots.py
  ...
```

Supporting files at the repository root include:
- `requirements-stage1.txt`
- `requirements-stage1.min.txt`
- `requirements-stage2.txt`
- `requirements-stage2.min.txt`
- `audit_compute.py`
- `audit_srip.py`
- `flowchart.png`

## Recommended workflow

The paper-facing workflow is:

1. Run Stage 1 surrogate modelling

```bash
python stage1/stage1_main.py
```

2. Create the Stage 2 analysis dataset from Stage 1 outputs

```bash
python stage2/stage2_create_analysis_dataset.py
```

3. Run the Stage 2 hierarchical heterogeneity model

```bash
python stage2/stage2_hierarchical_cf.py --run-id <your_run_id>
```

4. Generate association-based prescriptions and agreement summaries

```bash
python stage2/stage2_cf_prescription_generation.py
```

5. Generate paper/report tables and figures as needed from `reporters/reporters/`

## Configuration

Before running the code, review:
- `stage1/stage1_config.py`
- `stage2/stage2_config.py`

These files define:
- input data paths,
- output locations,
- feature lists,
- cross-validation settings,
- treatment/control definitions,
- and model hyperparameters.

## Data and access

This repository does **not** redistribute the underlying iRAP ViDA segment exports used in the study.

The raw and processed survey data are subject to iRAP licensing and access conditions. See [DATA_ACCESS.md](DATA_ACCESS.md) for the expected local input files and the data-access note.

Small support metadata used by the codebase, such as treatment/code mappings, may be included where appropriate.

## Interpretation note

For consistency with the paper:

- the Stage 1 outcome is the exported **ViDA modelled FSI**,
- the Stage 2 quantities are **conditional associations / contrasts** on that modelled surface,
- and the resulting prescriptions are **association-based candidate upgrades**.

They should not be described as direct estimates of crash reduction on observed crash data.

## Citation

If you use this repository, please cite the corresponding paper once publication details are available.
