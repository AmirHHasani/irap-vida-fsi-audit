"""Appendix Table A2: Ordinal step-contrast check for Road condition.

Purpose
- Reviewer-proof check that an ordinal "per-step" interpretation is reasonable.
- Trains on the full eligible segment corpus for each contrast.
- Reports summaries on the 396 Stage-1 candidate hotspots (TP+FP) only.

Contrasts (RAW road_condition coding, as in iRAP survey coding)
- Contrast A: 3→2 using only segments with Road condition ∈ {3,2}
  Treatment T=1{Road condition == 2} (i.e., improved from poor(3) to medium(2)).
- Contrast B: 2→1 using only segments with Road condition ∈ {2,1}
  Treatment T=1{Road condition == 1} (i.e., improved from medium(2) to good(1)).

Outcome scale
- Fits the model on the same outcome scale used in the Stage-1 OOF output (actual_risk).
- Reports implied changes on the natural FSI scale by converting the estimated log-scale
    step effect $\tau$ using:
        ΔFSI = (FSI + eps_y) * (exp(τ) - 1)
    where eps_y = OUTCOME_EPSILON.

Outputs
- stage2_outputs/.../reports/table_A2_road_condition_step_contrasts.csv
- stage2_outputs/.../reports/table_A2_road_condition_step_contrasts.tex

Run
  python create_table_A2_road_condition_step_contrasts.py

Notes
- Requires econml and the Stage-1/Stage-2 data products referenced in stage2_config.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

import reporter_config as rcfg


# econml emits this warning repeatedly during cross-fitting when a discrete
# treatment is modeled with a regressor (which we do to match the main Stage-2
# pipeline). Keep it visible but only once per run.
warnings.filterwarnings(
    action="once",
    message=r"First stage model has discrete target but model is not a classifier!",
    category=UserWarning,
)


def _baseline_fsi_natural(df: pd.DataFrame) -> np.ndarray:
    """Get baseline FSI on natural (non-negative) scale for a subset of rows.

    Preference order:
    1) Use the raw target column if present in the Phase-2 source file.
    2) Otherwise, infer from actual_risk assuming log(FSI + eps_y).
    """
    eps_y = float(getattr(rcfg, "OUTCOME_EPSILON", 1e-4))

    if hasattr(rcfg, "TARGET_COL") and rcfg.TARGET_COL in df.columns:
        baseline = pd.to_numeric(df[rcfg.TARGET_COL], errors="coerce").astype(float).values
        return np.clip(baseline, 0.0, np.inf)

    if "actual_risk" not in df.columns:
        return np.array([], dtype=float)

    y = pd.to_numeric(df["actual_risk"], errors="coerce").astype(float).values
    baseline = np.exp(y) - eps_y
    return np.clip(baseline, 0.0, np.inf)


@dataclass(frozen=True)
class Contrast:
    name: str
    eligible_codes: tuple[int, int]
    treated_code: int


CONTRASTS: list[Contrast] = [
    Contrast(name="3→2 (Poor→Medium)", eligible_codes=(3, 2), treated_code=2),
    Contrast(name="2→1 (Medium→Good)", eligible_codes=(2, 1), treated_code=1),
]


def _normalize_road_condition(series: pd.Series) -> pd.Series:
    # Keep raw codes (1/2/3) but normalize numeric formatting.
    s = pd.to_numeric(series, errors="coerce")
    # Some sources encode as floats like 1.0
    return s.round().astype("Int64")


def _build_features(analysis_data: pd.DataFrame) -> pd.DataFrame:
    dataset_ids_clean = analysis_data["Dataset ID"].astype(str).str.strip()

    dataset_dummies = pd.get_dummies(
        dataset_ids_clean,
        prefix="dataset",
        drop_first=False,
    )

    available_controls = [
        col for col in rcfg.CONTROL_FEATURES
        if col in analysis_data.columns and col != "Dataset ID"
    ]

    X_controls = analysis_data[available_controls].copy()
    X_controls = X_controls.fillna(0)

    # Ensure numeric matrix; one-hot any object/category controls.
    X_controls_numeric = pd.DataFrame(index=analysis_data.index)
    for col in X_controls.columns:
        if X_controls[col].dtype == "object" or X_controls[col].dtype.name == "category":
            dummies = pd.get_dummies(X_controls[col], prefix=col, drop_first=False)
            X_controls_numeric = pd.concat([X_controls_numeric, dummies], axis=1)
        else:
            X_controls_numeric[col] = pd.to_numeric(X_controls[col], errors="coerce")

    X_features = pd.concat([X_controls_numeric, dataset_dummies], axis=1)
    return X_features.astype(float)


def load_merged_data() -> pd.DataFrame:
    """Match raw features with Stage-1 predictions + hotspot labels."""
    stage1_hotspots = pd.read_csv(rcfg.STAGE1_HOTSPOT_OVERLAY_CSV)
    stage1_predictions = pd.read_csv(rcfg.STAGE1_OOF_PREDICTIONS_CSV, low_memory=False)
    original_data = pd.read_csv(rcfg.SEGMENTS_UNIQUE_CSV, low_memory=False)

    original_data["segment_id"] = original_data["Location ID"]

    merge_columns = ["segment_id", "predicted_risk", "actual_risk"]
    for col in ["fold_number", "road_id", "road_canon"]:
        if col in stage1_predictions.columns:
            merge_columns.append(col)
    stage1_subset = stage1_predictions[merge_columns].copy()

    data = original_data.merge(stage1_subset, on="segment_id", how="inner")

    if "class" in stage1_hotspots.columns and "segment_id" in stage1_hotspots.columns:
        data = data.merge(
            stage1_hotspots[["segment_id", "class"]].rename(columns={"class": "hotspot_class"}),
            on="segment_id",
            how="left",
        )

    # Overlay membership
    overlay_ids = (
        stage1_hotspots["segment_id"].unique()
        if "segment_id" in stage1_hotspots.columns
        else stage1_hotspots["Location ID"].unique()
    )
    data["is_hotspot"] = data["segment_id"].isin(overlay_ids)

    # Candidate hotspots (TP+FP only)
    if "hotspot_class" in data.columns:
        data["is_candidate_hotspot"] = data["hotspot_class"].isin(["TP", "FP"])
    else:
        data["is_candidate_hotspot"] = data["is_hotspot"]

    # Region mapping (optional)
    try:
        mapping = pd.read_csv(rcfg.REGIONAL_MAPPING_CSV)
        data = data.merge(
            mapping[["Dataset ID", "Country Name", "Country Code", "Region Name"]],
            on="Dataset ID",
            how="left",
        )
    except Exception:
        if "Country Name" not in data.columns:
            data["Country Name"] = data["Dataset ID"].astype(str)
        if "Region Name" not in data.columns:
            data["Region Name"] = None

    data["Region"] = data.get("Region Name")

    return data


def fit_binary_cf_dml(Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> CausalForestDML:
    model_t = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    cf_model = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=2000,
        max_depth=8,
        min_samples_leaf=10,
        min_var_fraction_leaf=0.1,
        min_var_leaf_on_val=True,
        cv=5,
        mc_iters=4,
        inference=True,
        random_state=42,
        n_jobs=-1,
    )

    cf_model.fit(Y=Y, T=T, X=X, W=None)
    return cf_model


def summarize_contrast(
    contrast: Contrast,
    data: pd.DataFrame,
    X_features: pd.DataFrame,
) -> dict:
    rc = _normalize_road_condition(data["Road condition"]) if "Road condition" in data.columns else pd.Series([pd.NA] * len(data))

    eligible = rc.isin(list(contrast.eligible_codes))
    eligible &= data["actual_risk"].notna() & data["predicted_risk"].notna()

    # For reporting, we want hotspots that *currently need the upgrade*.
    # This is the (worse) control code in the contrast pair.
    needs_upgrade_code = next(
        code for code in contrast.eligible_codes if code != contrast.treated_code
    )

    # Ensure features are finite
    X = X_features.loc[eligible].values
    if X.size == 0:
        return {
            "Contrast": contrast.name,
            "Eligible_segments": 0,
            "N_hotspots": 0,
            "Mean_delta_fsi": np.nan,
            "Pct_negative": np.nan,
        }

    Y = pd.to_numeric(data.loc[eligible, "actual_risk"], errors="coerce").values.astype(float)
    T = (rc.loc[eligible] == contrast.treated_code).astype(int).values

    # Fit on full eligible corpus
    model = fit_binary_cf_dml(Y=Y, T=T, X=X)

    # Report on candidate hotspots (TP+FP) that currently need the upgrade,
    # restricted to the contrast subset.
    hotspot_mask = (
        eligible
        & data["is_candidate_hotspot"].fillna(False)
        & (rc == needs_upgrade_code)
    )
    n_hotspots_needing_upgrade = int(hotspot_mask.sum())
    if n_hotspots_needing_upgrade == 0:
        return {
            "Contrast": contrast.name,
            "Eligible_segments": int(eligible.sum()),
            "Hotspots_needing_upgrade": 0,
            "Mean_baseline_fsi": np.nan,
            "Median_baseline_fsi": np.nan,
            "Mean_delta_fsi": np.nan,
            "Median_delta_fsi": np.nan,
            "Pct_negative": np.nan,
            "Mean_relative_change_pct": np.nan,
            "Share_abs_delta_gt_baseline": np.nan,
        }

    X_hot = X_features.loc[hotspot_mask].values
    # tau is on the same scale as the outcome used in training. Empirically, actual_risk
    # in this project is log(FSI + eps_y), so we convert tau to implied ΔFSI on natural scale.
    tau = model.effect(X_hot).astype(float)
    eps_y = float(getattr(rcfg, "OUTCOME_EPSILON", 1e-4))
    eps_den = 1e-6

    baseline_fsi = _baseline_fsi_natural(data.loc[hotspot_mask])
    delta_fsi = (baseline_fsi + eps_y) * (np.exp(tau) - 1.0)
    rel_change_pct = (delta_fsi / np.maximum(baseline_fsi, eps_den)) * 100.0

    abs_gt_baseline = (np.abs(delta_fsi) > baseline_fsi) & np.isfinite(delta_fsi) & np.isfinite(baseline_fsi)
    share_abs_gt = float(np.nanmean(abs_gt_baseline) * 100.0) if abs_gt_baseline.size else np.nan

    return {
        "Contrast": contrast.name,
        "Eligible_segments": int(eligible.sum()),
        "Hotspots_needing_upgrade": n_hotspots_needing_upgrade,
        "Mean_baseline_fsi": float(np.nanmean(baseline_fsi)),
        "Median_baseline_fsi": float(np.nanmedian(baseline_fsi)),
        "Mean_delta_fsi": float(np.nanmean(delta_fsi)),
        "Median_delta_fsi": float(np.nanmedian(delta_fsi)),
        "Pct_negative": float(np.nanmean(delta_fsi < 0) * 100.0),
        "Mean_relative_change_pct": float(np.nanmean(rel_change_pct)),
        "Share_abs_delta_gt_baseline": share_abs_gt,
    }


def _to_latex_table(df: pd.DataFrame) -> str:
    lines = [
        r"\\begin{table}[t]",
        r"\\centering",
        r"\\small",
        r"\\begin{tabular}{lrrrr}",
        r"\\hline",
        r"Contrast & Hotspots needing upgrade (N) & Mean baseline FSI & Mean $\\Delta$FSI & \\% negative $\\Delta$FSI \\\\ ",
        r"\\hline",
    ]

    def esc(s: str) -> str:
        return str(s).replace("&", r"\\&").replace("%", r"\\%")

    for _, row in df.iterrows():
        contrast = esc(row["Contrast"])
        n_hot = int(row["Hotspots_needing_upgrade"])
        base = row.get("Mean_baseline_fsi")
        base_txt = f"{base:.4f}" if pd.notna(base) else "NA"
        mean = row["Mean_delta_fsi"]
        mean_txt = f"{mean:+.4f}" if pd.notna(mean) else "NA"
        pct = row["Pct_negative"]
        pct_txt = f"{pct:.1f}" if pd.notna(pct) else "NA"
        lines.append(f"{contrast} & {n_hot} & {base_txt} & {mean_txt} & {pct_txt} \\\\ ")

    lines.extend(
        [
            r"\\hline",
            r"\\end{tabular}",
            r"\\caption{Ordinal step-contrast check for Road condition. Models are trained on the full eligible segment corpus for each contrast, and summaries are reported on the 396 candidate hotspots (TP+FP) restricted to the contrast subset. Baseline FSI is on the natural scale; $\\Delta$FSI is computed as $(FSI+\\varepsilon_y)(e^{\\tau}-1)$ where $\\tau$ is the estimated step effect on the model's outcome scale and $\\varepsilon_y$ is OUTCOME\\_EPSILON.}",
            r"\\label{tab:road_condition_step_contrasts}",
            r"\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    print("=" * 80)
    print("APPENDIX TABLE A2: ROAD CONDITION STEP-CONTRAST CHECK")
    print("=" * 80)

    eps_y = float(getattr(rcfg, "OUTCOME_EPSILON", 1e-4))
    print(f"Outcome: using actual_risk from Phase-3 OOF output; reporting implied natural-scale ΔFSI via (FSI+{eps_y:g})*(exp(tau)-1).")

    data = load_merged_data()
    if "actual_risk" in data.columns:
        min_y = pd.to_numeric(data["actual_risk"], errors="coerce").min()
        if pd.notna(min_y) and float(min_y) < 0:
            print("Note: actual_risk contains negatives; treating as log(FSI+eps) for reporting conversion.")
    n_candidates = int(data["is_candidate_hotspot"].sum())
    print(f"Candidate hotspots (TP+FP): {n_candidates}")
    if n_candidates != 396:
        print("Warning: expected 396 candidate hotspots; check STAGE1_HOTSPOT_OVERLAY / labels.")

    if "Road condition" not in data.columns:
        raise ValueError("'Road condition' column not found in SEGMENTS_DATA_CSV.")

    X_features = _build_features(data)

    rows: list[dict] = []
    for c in CONTRASTS:
        print(f"\nFitting contrast: {c.name}")
        row = summarize_contrast(c, data, X_features)
        rows.append(row)

        # Mandatory scale validation diagnostics (hotspots only, within contrast subset)
        if row.get("Hotspots_needing_upgrade", 0) and pd.notna(row.get("Mean_baseline_fsi")):
            print(
                "  Hotspots-only scale check: "
                f"mean_baseline_fsi={row['Mean_baseline_fsi']:.4f}, "
                f"median_baseline_fsi={row['Median_baseline_fsi']:.4f}, "
                f"mean_delta_fsi={row['Mean_delta_fsi']:+.4f}, "
                f"median_delta_fsi={row['Median_delta_fsi']:+.4f}, "
                f"share(|ΔFSI|>baseline)={row['Share_abs_delta_gt_baseline']:.1f}%"
            )
            if row["Mean_baseline_fsi"] < 0:
                print("  WARNING: baseline FSI < 0 after conversion; check baseline source and OUTCOME_EPSILON.")
            if row["Share_abs_delta_gt_baseline"] > 50.0:
                print("  WARNING: Large share(|ΔFSI|>baseline) on natural scale; verify outcome scale and baseline source.")

    out_df = pd.DataFrame(rows)

    # CSV: keep minimal + defensible audit columns (all hotspots-only summaries)
    out_df = out_df[
        [
            "Contrast",
            "Hotspots_needing_upgrade",
            "Eligible_segments",
            "Mean_baseline_fsi",
            "Median_baseline_fsi",
            "Mean_delta_fsi",
            "Median_delta_fsi",
            "Pct_negative",
            "Mean_relative_change_pct",
            "Share_abs_delta_gt_baseline",
        ]
    ]

    reports_dir: Path = rcfg.ensure_reports_dir()

    out_csv = reports_dir / "table_A2_road_condition_step_contrasts.csv"
    out_tex = reports_dir / "table_A2_road_condition_step_contrasts.tex"

    out_df.to_csv(out_csv, index=False)
    out_tex.write_text(_to_latex_table(out_df), encoding="utf-8")

    print("\nOutputs:")
    print(f"- {out_csv}")
    print(f"- {out_tex}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
