"""\
Concordance analysis between Stage 2 prescriptions and iRAP SRIP countermeasures
(Paper v27 definitions).

Run-aware inputs/outputs:
- Preferred: stage2_outputs/runs/<run-id>/...
- Fallback: stage2_config.OUTPUT_DIR/...

Outputs (written to <stage2-output-dir>/reports):
- irap_concordance_metrics_v27.csv (Table 10 driver)
- irap_mapping_audit_v27.csv
- fn_cate_summary_by_treatment_v27.csv

Notes:
- This is NOT a validation that treats iRAP as ground truth.
- Agreement metrics (micro-accuracy, kappa) are computed on the overlap population
    (overlap hotspots × 6 treatments). Coverage stats (including "both silent") are
    reported separately and are not included in overlap-based metrics.
"""

from __future__ import annotations

import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import reporter_config as rcfg


# -------------------------
# Configuration (v27)
# -------------------------

V27_TREATMENTS: list[str] = list(rcfg.ALL_TREATMENTS)


@dataclass(frozen=True)
class Inputs:
    candidates_csv: Path
    prescriptions_csv: Path
    stage1_inventory_csv: Path | None
    irap_countermeasures_csv: Path
    reports_dir: Path
    overlap_label_table_long_csv: Path | None


def _runs_root() -> Path:
    return Path(__file__).resolve().parent / "stage2_outputs" / "runs"


def _resolve_stage2_output_dir(*, stage2_output_dir: str | None, run_id: str | None) -> Path:
    return rcfg.resolve_stage2_root(stage2_output_dir=stage2_output_dir, run_id=run_id)


def _latest_csv(parent: Path, pattern: str) -> Path:
    candidates = sorted(parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No files found for pattern '{pattern}' in {parent}")
    return candidates[0]


def resolve_inputs(*, stage2_root: Path, irap_countermeasures_csv: Path | None = None) -> Inputs:
    run_root = Path(stage2_root).resolve()

    candidates_csv = (
        run_root
        / "hierarchical_cf"
        / "hotspot_level"
        / "hotspot_segments_detailed.csv"
    )
    if not candidates_csv.exists():
        raise FileNotFoundError(f"Missing candidates file: {candidates_csv}")

    stage2_dir = run_root / "stage2_cf_prescriptions"
    prescriptions_csv = _latest_csv(stage2_dir, "stage2_segment_prescriptions_*.csv")

    stage1_inventory_csv = None
    try:
        stage1_inventory_csv = _latest_csv(stage2_dir, "stage1_hotspot_inventory_*.csv")
    except FileNotFoundError:
        stage1_inventory_csv = None

    # iRAP file path: use reporter_config default unless overridden.
    if irap_countermeasures_csv is None:
        irap_countermeasures_csv = rcfg.IRAP_COUNTERMEASURES_CSV

    if not irap_countermeasures_csv.exists():
        raise FileNotFoundError(
            "Missing iRAP countermeasures CSV at expected location: "
            f"{irap_countermeasures_csv}. "
            "If your iRAP file is elsewhere, pass --irap-countermeasures-csv."
        )

    reports_dir = rcfg.ensure_reports_dir(run_root)

    overlap_label_table_long_csv = reports_dir / "stage2_vs_irap_overlap_label_table_long.csv"
    if not overlap_label_table_long_csv.exists():
        overlap_label_table_long_csv = None

    return Inputs(
        candidates_csv=candidates_csv,
        prescriptions_csv=prescriptions_csv,
        stage1_inventory_csv=stage1_inventory_csv,
        irap_countermeasures_csv=irap_countermeasures_csv,
        reports_dir=reports_dir,
        overlap_label_table_long_csv=overlap_label_table_long_csv,
    )


# -------------------------
# Mapping + normalization
# -------------------------

_normalize_re = re.compile(r"[^a-z0-9\s]+")


def normalize_countermeasure(name: str) -> str:
    """Normalize iRAP countermeasure names for robust mapping."""
    if name is None:
        return ""
    s = str(name).lower().strip()
    s = s.replace("_", " ").replace("-", " ")
    s = _normalize_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def create_treatment_mapping_normalized() -> dict[str, str]:
    """Return mapping from normalized iRAP countermeasure name -> treatment class (6-class space)."""

    # Source list is intentionally explicit and auditable.
    raw_mapping: dict[str, list[str]] = {
        "Centreline rumble strips": [
            "Centreline rumble strips",
            "Wide centreline",
            # Common naming variants in iRAP export
            "Centreline rumble strip / flexi-post",
        ],
        "Delineation": [
            "Improve Delineation",
            "Improve delineation",
            # Common variants
            "Improve curve delineation",
        ],
        "Street lighting": [
            "Provide street lighting",
            "Street lighting",
            # Common variants
            "Street lighting (mid-block)",
            "Street lighting (intersection)",
        ],
        "Paved shoulder - driver-side": [
            "Shoulder sealing driver side (>1m)",
            "Shoulder sealing driver side",
            "Widen road shoulder driver side",
        ],
        "Paved shoulder - passenger-side": [
            "Shoulder sealing passenger side (>1m)",
            "Shoulder sealing passenger side",
            "Widen road shoulder passenger side",
        ],
        "Road condition": [
            "Road surface rehabilitation",
            "Improve pavement condition",
        ],
    }

    mapping: dict[str, str] = {}
    for treatment, cm_names in raw_mapping.items():
        for cm in cm_names:
            key = normalize_countermeasure(cm)
            if key in mapping and mapping[key] != treatment:
                raise ValueError(
                    "Ambiguous mapping: normalized countermeasure maps to multiple treatments: "
                    f"'{key}' -> '{mapping[key]}' and '{treatment}'"
                )
            mapping[key] = treatment

    return mapping


# -------------------------
# Core concordance logic
# -------------------------


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_candidates(candidates_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(candidates_csv)
    if "segment_id" not in df.columns:
        raise ValueError("Candidates file missing 'segment_id' column")

    # Paper v27 definition: start from the 396 Stage-1 candidate hotspots
    # In this file, candidates are the rows present (expected 396).
    candidates = df.copy()
    candidates = candidates.drop_duplicates(subset=["segment_id"]).reset_index(drop=True)
    return candidates


def load_stage2_recommendations(prescriptions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(prescriptions_csv)

    required = {"segment_id", "treatment", "current_level", "target_level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prescriptions file missing columns: {sorted(missing)}")

    df = df[df["treatment"].isin(V27_TREATMENTS)].copy()

    cur = _coerce_numeric(df["current_level"])
    tgt = _coerce_numeric(df["target_level"])
    df["_cur"] = cur
    df["_tgt"] = tgt

    # A Stage 2 "prescription" for v27 is a change recommendation (exclude no-change rows).
    df = df[(df["_cur"].notna()) & (df["_tgt"].notna()) & (df["_tgt"] > df["_cur"])].copy()

    # De-duplicate defensively at segment-treatment level (avoid any accidental double counting).
    df = df.drop_duplicates(subset=["segment_id", "treatment"]).reset_index(drop=True)

    return df[["segment_id", "treatment"]]


def load_irap_and_map(
    irap_csv: Path,
    candidate_segment_ids: Iterable[int],
    cm_to_treatment: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load iRAP rows for candidate segments and return mapped + unmapped subsets."""

    # Performance: the raw SRIP file is large; for v27 we only need these columns.
    irap = pd.read_csv(
        irap_csv,
        usecols=["Location ID", "Countermeasure"],
        low_memory=False,
    )
    required = {"Location ID", "Countermeasure"}
    missing = required - set(irap.columns)
    if missing:
        raise ValueError(f"iRAP file missing columns: {sorted(missing)}")

    candidate_ids = set(candidate_segment_ids)
    irap = irap[irap["Location ID"].isin(candidate_ids)].copy()

    irap["countermeasure_original"] = irap["Countermeasure"].astype(str)
    irap["countermeasure_normalized"] = irap["countermeasure_original"].map(normalize_countermeasure)
    irap["mapped_treatment"] = irap["countermeasure_normalized"].map(cm_to_treatment)

    mapped = irap[irap["mapped_treatment"].notna()].copy()
    mapped = mapped[mapped["mapped_treatment"].isin(V27_TREATMENTS)].copy()

    unmapped = irap[irap["mapped_treatment"].isna()].copy()

    return mapped, unmapped


def build_overlap_population(mapped_irap: pd.DataFrame) -> np.ndarray:
    """Overlap hotspots: candidate hotspots with >=1 mapped iRAP countermeasure in 6 classes."""
    overlap_ids = mapped_irap["Location ID"].dropna().unique()
    return overlap_ids


def build_pair_table(
    overlap_segment_ids: np.ndarray,
    stage2_recs: pd.DataFrame,
    mapped_irap: pd.DataFrame,
) -> pd.DataFrame:
    # iRAP recommends: segment-treatment is 1 if ANY mapped countermeasure exists (dedupe strings)
    irap_pairs = (
        mapped_irap[["Location ID", "mapped_treatment"]]
        .drop_duplicates()
        .rename(columns={"Location ID": "segment_id", "mapped_treatment": "treatment"})
    )
    irap_pairs["irap_recommends"] = 1

    s2_pairs = stage2_recs.copy()
    s2_pairs["stage2_recommends"] = 1

    base = pd.MultiIndex.from_product(
        [sorted(overlap_segment_ids.tolist()), V27_TREATMENTS],
        names=["segment_id", "treatment"],
    ).to_frame(index=False)

    merged = base.merge(s2_pairs, on=["segment_id", "treatment"], how="left")
    merged = merged.merge(irap_pairs, on=["segment_id", "treatment"], how="left")
    merged["stage2_recommends"] = merged["stage2_recommends"].fillna(0).astype(int)
    merged["irap_recommends"] = merged["irap_recommends"].fillna(0).astype(int)

    return merged


def confusion_counts(pairs: pd.DataFrame) -> dict[str, int]:
    p = pairs["stage2_recommends"].astype(int)
    t = pairs["irap_recommends"].astype(int)

    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def load_run_overlap_label_table_long(path: Path) -> pd.DataFrame:
    """Load the run-local overlap label table (authoritative for run-mappable overlap metrics)."""
    df = pd.read_csv(path)
    required = {"segment_id", "treatment", "y_stage2", "y_srip"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Overlap label table missing columns: {sorted(missing)}")

    out = df[["segment_id", "treatment", "y_stage2", "y_srip"]].copy()
    out = out.rename(columns={"y_stage2": "stage2_recommends", "y_srip": "irap_recommends"})
    out["segment_id"] = pd.to_numeric(out["segment_id"], errors="coerce").astype("Int64")
    out = out[out["segment_id"].notna()].copy()
    out["segment_id"] = out["segment_id"].astype(int)

    out["stage2_recommends"] = pd.to_numeric(out["stage2_recommends"], errors="coerce")
    out["irap_recommends"] = pd.to_numeric(out["irap_recommends"], errors="coerce")
    out = out[out["stage2_recommends"].notna() & out["irap_recommends"].notna()].copy()
    out["stage2_recommends"] = out["stage2_recommends"].astype(int)
    out["irap_recommends"] = out["irap_recommends"].astype(int)

    # Keep to the v27 6-treatment space
    out = out[out["treatment"].isin(V27_TREATMENTS)].copy()
    return out


def cohens_kappa_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    n = tp + fp + fn + tn
    if n == 0:
        return float("nan")

    po = (tp + tn) / n

    p_pred_yes = (tp + fp) / n
    p_pred_no = (fn + tn) / n
    p_true_yes = (tp + fn) / n
    p_true_no = (fp + tn) / n

    pe = p_pred_yes * p_true_yes + p_pred_no * p_true_no
    if 1 - pe == 0:
        return float("nan")
    return (po - pe) / (1 - pe)


def build_metrics_output(
    candidates_df: pd.DataFrame,
    stage2_recs: pd.DataFrame,
    mapped_irap: pd.DataFrame,
    overlap_ids: np.ndarray,
    pairs: pd.DataFrame,
    counts: dict[str, int],
    extra_rows: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    # Coverage over the 396 candidates
    candidate_ids = candidates_df["segment_id"].unique()

    s2_any = set(stage2_recs["segment_id"].unique())
    irap_any = set(mapped_irap["Location ID"].unique())

    both_silent = [seg for seg in candidate_ids if seg not in s2_any and seg not in irap_any]

    total_pairs = int(len(overlap_ids) * len(V27_TREATMENTS))
    micro_accuracy = (counts["tp"] + counts["tn"]) / total_pairs if total_pairs else 0.0
    kappa = cohens_kappa_from_counts(counts["tp"], counts["fp"], counts["fn"], counts["tn"])

    # Prescription-level overlap (deduped segment-treatment prescriptions)
    denom = counts["tp"] + counts["fp"]
    prescription_overlap = counts["tp"] / denom if denom else 0.0

    rows = [
        {
            "Section": "Overlap",
            "Metric": "Overlapping hotspots",
            "Value": int(len(overlap_ids)),
            "Notes": "Stage 1 candidate hotspots with ≥1 mapped iRAP countermeasure in the 6-class space (overlap evaluation population)",
        },
        {
            "Section": "Overlap",
            "Metric": "Stage 2 prescriptions reviewed",
            "Value": int(stage2_recs[stage2_recs["segment_id"].isin(overlap_ids)][["segment_id", "treatment"]].drop_duplicates().shape[0]),
            "Notes": "Unique (segment,treatment) prescriptions on overlap hotspots; excludes no-change",
        },
        {
            "Section": "Overlap",
            "Metric": "iRAP countermeasures reviewed",
            "Value": int(mapped_irap[mapped_irap["Location ID"].isin(overlap_ids)].shape[0]),
            "Notes": "Row-level iRAP recommendations on overlap hotspots; restricted to mapped countermeasures in 6 classes",
        },
        {
            "Section": "Overlap",
            "Metric": "Exact matches",
            "Value": int(counts["tp"]),
            "Notes": "Matched Stage 2 prescriptions where iRAP recommends the same segment–treatment (deduplicated)",
        },
        {
            "Section": "Overlap",
            "Metric": "Prescription-level overlap",
            "Value": float(prescription_overlap),
            "Notes": "TP / (TP+FP) computed on overlap hotspots × 6 treatments",
        },
        {
            "Section": "Classification",
            "Metric": "True positives (TP)",
            "Value": int(counts["tp"]),
            "Notes": "Segment–treatment pairs recommended by both systems",
        },
        {
            "Section": "Classification",
            "Metric": "False positives (FP)",
            "Value": int(counts["fp"]),
            "Notes": "Stage 2 recommends; iRAP does not (within mapped 6-class space)",
        },
        {
            "Section": "Classification",
            "Metric": "False negatives (FN)",
            "Value": int(counts["fn"]),
            "Notes": "iRAP recommends; Stage 2 does not (within mapped 6-class space)",
        },
        {
            "Section": "Classification",
            "Metric": "True negatives (TN)",
            "Value": int(counts["tn"]),
            "Notes": "Neither system recommends the treatment",
        },
        {
            "Section": "Classification",
            "Metric": "Total segment–treatment pairs",
            "Value": int(total_pairs),
            "Notes": "Overlap hotspots × 6 treatments",
        },
        {
            "Section": "Classification",
            "Metric": "Micro accuracy",
            "Value": float(micro_accuracy),
            "Notes": "(TP+TN)/total over overlap hotspots × 6 treatments",
        },
        {
            "Section": "Classification",
            "Metric": "Cohen kappa",
            "Value": float(kappa),
            "Notes": "Chance-corrected agreement over overlap hotspots × 6 treatments",
        },
        {
            "Section": "Coverage",
            "Metric": "Candidate hotspots",
            "Value": int(len(candidate_ids)),
            "Notes": "Stage 1 candidate hotspot population",
        },
        {
            "Section": "Coverage",
            "Metric": "Hotspots with Stage 2 prescriptions",
            "Value": int(len(s2_any)),
            "Notes": "Any of the 6 treatments prescribed (excludes no-change)",
        },
        {
            "Section": "Coverage",
            "Metric": "Hotspots with mapped iRAP countermeasures",
            "Value": int(len(irap_any)),
            "Notes": "At least one mapped countermeasure in the 6 classes (using the full iRAP countermeasures CSV)",
        },
        {
            "Section": "Coverage",
            "Metric": "Hotspots where both systems silent",
            "Value": int(len(both_silent)),
            "Notes": "Candidates with no Stage 2 prescriptions and no mapped iRAP actions (6 classes); not part of overlap-based agreement metrics",
        },
    ]

    if extra_rows:
        rows.extend(extra_rows)

    return pd.DataFrame(rows)


# -------------------------
# Mapping audit output
# -------------------------


def build_mapping_audit(
    overlap_ids: np.ndarray,
    mapped_irap: pd.DataFrame,
    unmapped_irap: pd.DataFrame,
) -> pd.DataFrame:
    overlap_set = set(overlap_ids.tolist())

    mapped_overlap = mapped_irap[mapped_irap["Location ID"].isin(overlap_set)].copy()
    unmapped_overlap = unmapped_irap[unmapped_irap["Location ID"].isin(overlap_set)].copy()

    mapped_summary = (
        mapped_overlap.groupby("mapped_treatment")
        .size()
        .reindex(V27_TREATMENTS)
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"mapped_treatment": "key", 0: "count"})
    )
    mapped_summary["record_type"] = "mapped_count_by_treatment"
    mapped_summary["original_example"] = ""

    # Top unmapped countermeasures within overlap hotspots (row counts)
    if not unmapped_overlap.empty:
        top_unmapped = (
            unmapped_overlap.groupby(["countermeasure_normalized", "countermeasure_original"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(50)
        )
        top_unmapped["record_type"] = "top_unmapped_countermeasure"
        top_unmapped = top_unmapped.rename(
            columns={
                "countermeasure_normalized": "key",
                "countermeasure_original": "original_example",
            }
        )
    else:
        top_unmapped = pd.DataFrame(columns=["key", "original_example", "count", "record_type"])  # type: ignore[assignment]

    audit = pd.concat(
        [
            mapped_summary[["record_type", "key", "original_example", "count"]],
            top_unmapped[["record_type", "key", "original_example", "count"]],
        ],
        ignore_index=True,
    )
    return audit


# -------------------------
# FN CATE summary
# -------------------------


def _normalize_result_key(name: str) -> str:
    """Match the naming scheme used by stage2_hierarchical_cf.py outputs."""
    return name.replace(" - ", "_").replace(" ", "_").lower()


def _load_contrast_spec(stage2_root: Path) -> list[dict]:
    path = Path(stage2_root).resolve() / "data" / "stage2_contrast_spec.json"
    if not path.exists():
        return []
    return pd.read_json(path).to_dict(orient="records")


def _build_contrast_index(contrast_spec: list[dict]) -> dict[tuple[str, int, int], dict]:
    index: dict[tuple[str, int, int], dict] = {}
    for item in contrast_spec:
        base = str(item.get("treatment_base", "")).strip()
        if not base:
            continue
        try:
            t0 = int(float(item.get("t0")))
            t1 = int(float(item.get("t1")))
        except Exception:
            continue
        index[(base, t0, t1)] = item
    return index


def _pick_fixed_contrast(
    *,
    treatment_base: str,
    contrast_index: dict[tuple[str, int, int], dict],
) -> str | None:
    """Prefer 0->1; otherwise fall back to smallest available upgrade from the spec."""
    item = contrast_index.get((treatment_base, 0, 1))
    if item:
        return str(item.get("contrast") or f"{treatment_base}__0_to_1")

    upgrades = [(t0, t1) for (base, t0, t1) in contrast_index.keys() if base == treatment_base and t1 > t0]
    if not upgrades:
        return None
    t0, t1 = sorted(upgrades, key=lambda x: (x[0], x[1]))[0]
    fallback_item = contrast_index.get((treatment_base, t0, t1), {})
    return str(fallback_item.get("contrast") or f"{treatment_base}__{t0}_to_{t1}")


def fn_cate_summary(
    pairs: pd.DataFrame,
    candidates_df: pd.DataFrame,
    stage2_root: Path,
) -> pd.DataFrame:
    fn_pairs = pairs[(pairs["irap_recommends"] == 1) & (pairs["stage2_recommends"] == 0)].copy()
    if fn_pairs.empty:
        return pd.DataFrame(
            columns=[
                "treatment",
                "fn_pairs",
                "cate_log_mean",
                "cate_log_median",
                "cate_pct_from_log_mean",
                "cate_pct_from_log_median",
                "cate_raw_mean",
                "cate_raw_median",
            ]
        )

    # Long-format CATE table: (segment_id, treatment) -> cate_log, cate_raw
    rows: list[pd.DataFrame] = []

    # Path A (legacy): per-treatment columns exist.
    legacy_cols = {
        "Centreline rumble strips": ("centreline_rumble_strips_cate", "centreline_rumble_strips_cate_raw"),
        "Delineation": ("delineation_cate", "delineation_cate_raw"),
        "Street lighting": ("street_lighting_cate", "street_lighting_cate_raw"),
        "Paved shoulder - driver-side": ("paved_shoulder_driver-side_cate", "paved_shoulder_driver-side_cate_raw"),
        "Paved shoulder - passenger-side": ("paved_shoulder_passenger-side_cate", "paved_shoulder_passenger-side_cate_raw"),
        "Road condition": ("road_condition_cate", "road_condition_cate_raw"),
    }
    if all((c1 in candidates_df.columns and c2 in candidates_df.columns) for (c1, c2) in legacy_cols.values()):
        for treatment, (col_log, col_raw) in legacy_cols.items():
            sub = candidates_df[["segment_id", col_log, col_raw]].copy()
            sub = sub.rename(columns={col_log: "cate_log", col_raw: "cate_raw"})
            sub["treatment"] = treatment
            rows.append(sub)
        cate_long = pd.concat(rows, ignore_index=True)
    else:
        # Path B (current): contrast-based columns. Choose one fixed contrast per treatment.
        contrast_spec = _load_contrast_spec(stage2_root)
        contrast_index = _build_contrast_index(contrast_spec)
        if not contrast_index:
            raise ValueError(
                "Candidates file does not include legacy per-treatment CATE columns, and stage2_contrast_spec.json "
                "was not found (or empty). Cannot compute FN CATE summary."
            )

        # treatment_base values in stage2_contrast_spec.json are the human-readable treatment names.
        for treatment in V27_TREATMENTS:
            contrast = _pick_fixed_contrast(treatment_base=treatment, contrast_index=contrast_index)
            if not contrast:
                continue
            col_log = f"{_normalize_result_key(contrast)}_cate"
            col_raw = f"{_normalize_result_key(contrast)}_cate_raw"
            if col_log not in candidates_df.columns:
                raise ValueError(f"Candidates missing expected CATE column: {col_log}")
            if col_raw not in candidates_df.columns:
                raise ValueError(f"Candidates missing expected CATE raw column: {col_raw}")

            sub = candidates_df[["segment_id", col_log, col_raw]].copy()
            sub = sub.rename(columns={col_log: "cate_log", col_raw: "cate_raw"})
            sub["treatment"] = treatment
            rows.append(sub)

        if not rows:
            raise ValueError("Failed to build any FN CATE inputs from contrast-based candidates file.")
        cate_long = pd.concat(rows, ignore_index=True)

    merged = fn_pairs.merge(cate_long, on=["segment_id", "treatment"], how="left")

    # Transform log-scale cate to percent change if interpreted as log risk ratio change
    # percent_from_log = (exp(cate_log) - 1) * 100
    merged["cate_pct_from_log"] = (np.exp(merged["cate_log"]) - 1.0) * 100.0

    # Use explicit aggregations (avoids pandas groupby.apply deprecation warnings)
    merged["cate_log"] = pd.to_numeric(merged["cate_log"], errors="coerce")
    merged["cate_raw"] = pd.to_numeric(merged["cate_raw"], errors="coerce")
    merged["cate_pct_from_log"] = pd.to_numeric(merged["cate_pct_from_log"], errors="coerce")

    summary = (
        merged.groupby("treatment", as_index=False)
        .agg(
            fn_pairs=("segment_id", "size"),
            cate_log_mean=("cate_log", "mean"),
            cate_log_median=("cate_log", "median"),
            cate_pct_from_log_mean=("cate_pct_from_log", "mean"),
            cate_pct_from_log_median=("cate_pct_from_log", "median"),
            cate_raw_mean=("cate_raw", "mean"),
            cate_raw_median=("cate_raw", "median"),
        )
    )

    summary = summary.set_index("treatment").reindex(V27_TREATMENTS).reset_index()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Concordance analysis vs iRAP SRIP (Table 10 driver, v27).")
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Stage 2 run id under stage2_outputs/runs/<run-id>. "
            "If provided, reads/writes within that run folder."
        ),
    )
    parser.add_argument(
        "--stage2-output-dir",
        default=None,
        help=(
            "Explicit Stage 2 output directory containing hierarchical_cf/... and stage2_cf_prescriptions/... outputs. "
            "Overrides --run-id."
        ),
    )
    parser.add_argument(
        "--irap-countermeasures-csv",
        default=None,
        help="Optional override path to the iRAP countermeasures CSV (SRIP export).",
    )
    parser.add_argument(
        "--overlap-source",
        default="run_mappable",
        choices=["run_mappable", "full_srip"],
        help=(
            "Which overlap population to use for agreement metrics. "
            "'run_mappable' uses <run>/reports/stage2_vs_irap_overlap_label_table_long.csv when available "
            "(consistent with compare_stage2_with_irap_srip.py outputs). "
            "'full_srip' uses any candidate hotspot with ≥1 mapped iRAP countermeasure from the full SRIP CSV."
        ),
    )
    args = parser.parse_args()

    stage2_root = _resolve_stage2_output_dir(stage2_output_dir=args.stage2_output_dir, run_id=args.run_id)
    irap_csv = Path(args.irap_countermeasures_csv).expanduser().resolve() if args.irap_countermeasures_csv else None

    inputs = resolve_inputs(stage2_root=stage2_root, irap_countermeasures_csv=irap_csv)

    candidates_df = load_candidates(inputs.candidates_csv)
    candidate_ids = candidates_df["segment_id"].unique()

    # Stage 2 prescriptions in 6-class space, excluding no-change
    stage2_recs = load_stage2_recommendations(inputs.prescriptions_csv)
    stage2_recs = stage2_recs[stage2_recs["segment_id"].isin(candidate_ids)].copy()

    cm_to_treatment = create_treatment_mapping_normalized()

    mapped_irap, unmapped_irap = load_irap_and_map(
        inputs.irap_countermeasures_csv,
        candidate_ids,
        cm_to_treatment,
    )

    overlap_ids_full = build_overlap_population(mapped_irap)

    extra_rows: list[dict[str, object]] = []

    if args.overlap_source == "run_mappable":
        if inputs.overlap_label_table_long_csv is None:
            print(
                "WARNING: overlap_source=run_mappable requested but run-local overlap label table was not found. "
                "Falling back to full_srip overlap population."
            )
            overlap_ids = overlap_ids_full
            pairs = build_pair_table(overlap_ids, stage2_recs, mapped_irap)
        else:
            pairs = load_run_overlap_label_table_long(inputs.overlap_label_table_long_csv)
            overlap_ids = pairs["segment_id"].dropna().unique()
            extra_rows.append(
                {
                    "Section": "Overlap",
                    "Metric": "Overlapping hotspots (full SRIP mapping)",
                    "Value": int(len(overlap_ids_full)),
                    "Notes": "Candidate hotspots with ≥1 mapped iRAP countermeasure when using the full iRAP SRIP export; may differ from run-mappable overlap",
                }
            )
            extra_rows.append(
                {
                    "Section": "Coverage",
                    "Metric": "Hotspots with mapped iRAP countermeasures (run-mappable)",
                    "Value": int(len(overlap_ids)),
                    "Notes": "Unique hotspot IDs present in the run-local overlap label table used for agreement metrics",
                }
            )
    else:
        overlap_ids = overlap_ids_full
        pairs = build_pair_table(overlap_ids, stage2_recs, mapped_irap)

    counts = confusion_counts(pairs)

    metrics_df = build_metrics_output(
        candidates_df=candidates_df,
        stage2_recs=stage2_recs,
        mapped_irap=mapped_irap,
        overlap_ids=overlap_ids,
        pairs=pairs,
        counts=counts,
        extra_rows=extra_rows,
    )

    audit_df = build_mapping_audit(overlap_ids, mapped_irap, unmapped_irap)
    fn_summary_df = fn_cate_summary(pairs, candidates_df, stage2_root=stage2_root)

    metrics_path = inputs.reports_dir / "irap_concordance_metrics_v27.csv"
    audit_path = inputs.reports_dir / "irap_mapping_audit_v27.csv"
    fn_path = inputs.reports_dir / "fn_cate_summary_by_treatment_v27.csv"

    metrics_df.to_csv(metrics_path, index=False)
    audit_df.to_csv(audit_path, index=False)
    fn_summary_df.to_csv(fn_path, index=False)

    # Keep stdout minimal and not table-number-specific
    print("Wrote metrics:", metrics_path)
    print("Wrote mapping audit:", audit_path)
    print("Wrote FN CATE summary:", fn_path)


if __name__ == "__main__":
    # Basic sanity check: config should still define the same 6 treatments.
    if set(rcfg.ALL_TREATMENTS) != set(V27_TREATMENTS):
        print("WARNING: stage2_config.ALL_TREATMENTS differs from the v27 treatment space used here.")
        print("         This script will still use the fixed 6-treatment list per v27.")

    main()
