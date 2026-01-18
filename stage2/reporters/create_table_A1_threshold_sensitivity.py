"""Create Appendix Table A1: threshold sensitivity grid (Stage 2 prescriptions).

Generates a 3×3 grid of (absolute, relative) thresholds and summarizes:
- N actionable upgrades (unique segment × treatment with code change)
- N hotspots covered (unique segments with ≥1 actionable upgrade)
- Top-2 treatment shares by upgrade count

This script reuses the Stage 2 CF-aligned prescription logic without writing
nine full sets of Stage 2 outputs.

Outputs:
- stage2_outputs/.../reports/table_A1_threshold_sensitivity.csv
- stage2_outputs/.../reports/table_A1_threshold_sensitivity.tex

Run:
  python create_table_A1_threshold_sensitivity.py

Common (run-scoped) usage:
    python create_table_A1_threshold_sensitivity.py --run-id base111
    python create_table_A1_threshold_sensitivity.py --stage2-output-dir stage2_outputs/runs/base111

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

import stage2_config as config
import stage2_cf_prescription_generation as stage2_prescriptions


def _runs_root() -> Path:
    return Path(__file__).resolve().parent / "stage2_outputs" / "runs"


def resolve_stage2_root(*, stage2_output_dir: str | None, run_id: str | None) -> Path:
    """Resolve the Stage 2 output root that contains hierarchical_cf/ and data/."""
    if stage2_output_dir:
        return Path(stage2_output_dir).expanduser().resolve()
    if run_id:
        return (_runs_root() / str(run_id)).resolve()
    return Path(config.OUTPUT_DIR).resolve()


def _load_current_levels(*, stage2_root: Path) -> pd.DataFrame:
    """Load current (canonical) treatment levels for each segment from run-local analysis_dataset."""
    analysis_path = Path(stage2_root).resolve() / "data" / "analysis_dataset.csv"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"analysis_dataset.csv not found: {analysis_path}. "
            "Run create_analysis_dataset.py (or rerun Stage 2) to produce run-local data/."
        )

    usecols = ["Location ID"] + list(config.ALL_TREATMENTS)
    df = pd.read_csv(analysis_path, usecols=usecols, low_memory=False)
    df = df.rename(columns={"Location ID": "segment_id"})

    # Normalise segment IDs to match CF outputs.
    df["segment_id"] = pd.to_numeric(df["segment_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["segment_id"]).copy()
    df["segment_id"] = df["segment_id"].astype(int)

    rename_map = {t: stage2_prescriptions.CURRENT_LEVEL_COLUMN_MAP[t] for t in config.ALL_TREATMENTS if t in df.columns}
    return df.rename(columns=rename_map)


ABS_THRESHOLDS = [0.001, 0.002, 0.004]
PCT_THRESHOLDS = [2.5, 5.0, 10.0]


@dataclass(frozen=True)
class Scenario:
    min_abs: float
    min_pct: float


def _format_top2_shares(counts: pd.Series) -> str:
    if counts.empty or counts.sum() == 0:
        return "None"

    total = float(counts.sum())
    top2 = counts.sort_values(ascending=False).head(2)
    parts: list[str] = []
    for treatment, n in top2.items():
        share = n / total * 100.0 if total else 0.0
        parts.append(f"{treatment} {share:.0f}%")
    return ", ".join(parts)


def _unit_label(population: str) -> str:
    return "Hotspots" if population == "hotspots" else "Segments"


def _normalize_result_key(name: str) -> str:
    """Match the naming scheme used by stage2_hierarchical_cf.py contrast outputs."""
    return name.replace(" - ", "_").replace(" ", "_").lower()


def _load_population(*, population: str, stage2_root: Path) -> pd.DataFrame:
    stage2_root = Path(stage2_root).resolve()
    if population == "hotspots":
        hotspot_path = (
            stage2_root / "hierarchical_cf" / "hotspot_level" / "hotspot_segments_detailed.csv"
        )
        if not hotspot_path.exists():
            raise FileNotFoundError(
                "Hotspot-level CF output not found: "
                f"{hotspot_path}. Run stage2_hierarchical_cf.py first."
            )
        df = pd.read_csv(hotspot_path, low_memory=False)
        if "segment_id" not in df.columns and "Location ID" in df.columns:
            df = df.rename(columns={"Location ID": "segment_id"})
        if "segment_id" not in df.columns:
            raise ValueError(
                "Expected 'segment_id' column in hotspot_segments_detailed.csv (or 'Location ID' to rename)."
            )
        return df

    # Full population (all segments) uses the wide file written by stage2_hierarchical_cf.py.
    all_segments_path = (
        stage2_root / "hierarchical_cf" / "segment_level" / "all_segments_cates_wide.csv"
    )
    if not all_segments_path.exists():
        raise FileNotFoundError(
            "Full segment-level CATE file not found: "
            f"{all_segments_path}. Run stage2_hierarchical_cf.py first."
        )
    df = pd.read_csv(all_segments_path, low_memory=False)
    if "segment_id" not in df.columns and "Location ID" in df.columns:
        df = df.rename(columns={"Location ID": "segment_id"})
    if "segment_id" not in df.columns:
        raise ValueError(
            "Expected 'segment_id' column in all_segments_cates_wide.csv (or 'Location ID' to rename)."
        )
    return df


def _make_latex_table(df: pd.DataFrame, caption: str, population: str) -> str:
    # Keep it compact: one row per scenario.
    # Output columns match the paper table: A, R, Total (N), Upgrades (N), Hotspots (N).
    unit = _unit_label(population)
    cols = [
        ("min_abs", "A"),
        ("min_pct", "R"),
        ("n_total_prescriptions", "Total (N)"),
        ("n_actionable_upgrades", "Upgrades (N)"),
        ("n_units_covered", f"{unit} (N)"),
    ]

    header = " & ".join([latex for _, latex in cols]) + r" \\" + "\n"
    lines = [
        r"\\begin{table}[t]",
        r"\\centering",
        r"\\small",
        r"\\begin{tabular}{rrrrr}",
        r"\\hline",
        header,
        r"\\hline",
    ]

    for _, row in df.iterrows():
        a = f"{row['min_abs']:.3f}"
        r = f"{row['min_pct']:.1f}"
        n_total = int(row["n_total_prescriptions"])
        n_up = int(row["n_actionable_upgrades"])
        n_hs = int(row["n_units_covered"])
        lines.append(f"{a} & {r} & {n_total} & {n_up} & {n_hs} \\\\ ")

    lines.extend(
        [
            r"\\hline",
            r"\\end{tabular}",
            rf"\\caption{{{caption}}}",
            rf"\\label{{tab:threshold_sensitivity_{population}}}",
            r"\\end{table}",
            "",
        ]
    )

    return "\n".join(lines)


def compute_scenario_metrics(df: pd.DataFrame, scenario: Scenario, population: str) -> dict:
    # Vectorized computation (fast for all segments).

    # Stage 2 CF effects (tau) and baseline risk are on the Stage 1 outcome scale.
    # Stage 1 uses a log-offset transform: y = log(FSI + eps). We must compute
    # absolute and relative reductions in *linear FSI units* for thresholding.
    #
    # If baseline_log = log(FSI + eps) and tau = E[y|t1]-E[y|t0], then:
    #   baseline_plus = exp(baseline_log) = FSI + eps
    #   post_plus     = exp(baseline_log + tau) = (FSI + eps) * exp(tau)
    #   abs_red_fsi   = max(0, (baseline_plus - eps) - (post_plus - eps))
    #               = max(0, baseline_plus * (1 - exp(tau)))
    eps_y = float(getattr(config, "OUTCOME_EPSILON", 0.001))
    baseline_log = pd.to_numeric(df.get("actual_risk", 0.0), errors="coerce")
    baseline_plus = np.exp(baseline_log.astype(float))
    baseline_fsi = (baseline_plus - eps_y).clip(lower=float(stage2_prescriptions.EPS))

    n_beneficial = 0
    n_pass_thresholds = 0
    n_total_prescriptions = 0
    n_units_with_any_prescription: set[int] = set()
    n_actionable_upgrades = 0
    n_units_with_any_actionable: set[int] = set()
    per_treatment_actionable_counts: dict[str, int] = {}

    seg_ids = pd.to_numeric(df["segment_id"], errors="coerce")
    seg_ids = seg_ids.astype("Int64")

    # IMPORTANT:
    # We compute two related but different counts:
    # - Total prescriptions (N): for each segment×treatment, take the *best* (largest)
    #   estimated reduction among all available step contrasts exported by the CF
    #   (e.g., 0->1, 1->2, 2->3). This is model-only and does NOT depend on current code.
    # - Actionable upgrades (N): evaluate the adjacent upgrade from the segment's current
    #   level (prescription-aligned), then keep only those that pass thresholds.

    min_abs = float(scenario.min_abs)
    min_pct = float(scenario.min_pct)
    guard = (min_abs / (min_pct / 100.0)) if min_pct > 0 else 0.0

    for treatment in stage2_prescriptions.CF_COLUMN_MAP.keys():
        current_col = stage2_prescriptions.CURRENT_LEVEL_COLUMN_MAP.get(treatment)
        has_current = bool(current_col) and (current_col in df.columns)

        meta = stage2_prescriptions.TREATMENT_LEVEL_METADATA.get(treatment, {})
        order = list(meta.get("order", []))
        best = max(order) if order else None
        if best is None:
            per_treatment_actionable_counts[treatment] = 0
            continue

        key = _normalize_result_key(treatment)
        current = (
            pd.to_numeric(df[current_col], errors="coerce").astype("Int64")
            if has_current
            else pd.Series(pd.NA, index=df.index, dtype="Int64")
        )

        # -------------------------------------------------------------------
        # TOTAL (model-only): best available contrast per segment×treatment.
        # -------------------------------------------------------------------
        contrast_cols = [
            col
            for col in df.columns
            if col.startswith(f"{key}__") and col.endswith("_cate") and "_to_" in col
        ]
        if contrast_cols:
            tau_mat = df[contrast_cols].apply(pd.to_numeric, errors="coerce")
            tau_best = tau_mat.min(axis=1, skipna=True).astype(float)

            valid_total = tau_best.notna() & baseline_fsi.notna() & seg_ids.notna()
            abs_red_total = (baseline_plus * (1.0 - np.exp(tau_best))).clip(lower=0.0)
            beneficial_total = valid_total & (abs_red_total > 0)
            n_beneficial += int(beneficial_total.sum())

            denom = baseline_fsi.clip(lower=float(stage2_prescriptions.EPS))
            pct_red_total = abs_red_total / denom * 100.0
            pct_ok_total = (pct_red_total >= min_pct) | (baseline_fsi < guard) if min_pct > 0 else pd.Series(True, index=df.index)
            passing_total = beneficial_total & (abs_red_total >= min_abs) & pct_ok_total

            n_pass_thresholds += int(passing_total.sum())
            n_total_prescriptions += int(passing_total.sum())
            if passing_total.any():
                n_units_with_any_prescription.update(seg_ids[passing_total].astype(int).tolist())
        else:
            # No model outputs for this treatment in this dataset.
            pass

        # -------------------------------------------------------------------
        # UPGRADES: adjacent upgrade from current level.
        # -------------------------------------------------------------------
        if not has_current:
            per_treatment_actionable_counts[treatment] = 0
            continue

        tau_adj = pd.Series(index=df.index, dtype="float64")
        for lv in order:
            if int(lv) == int(best):
                continue
            col = f"{key}__{int(lv)}_to_{int(lv) + 1}_cate"
            if col not in df.columns:
                continue
            mask = current == int(lv)
            if not mask.any():
                continue
            tau_adj.loc[mask] = pd.to_numeric(df.loc[mask, col], errors="coerce")

        valid_up = tau_adj.notna() & baseline_fsi.notna() & seg_ids.notna()
        abs_red_up = (baseline_plus * (1.0 - np.exp(tau_adj.astype(float)))).clip(lower=0.0)
        beneficial_up = valid_up & (abs_red_up > 0)
        denom = baseline_fsi.clip(lower=float(stage2_prescriptions.EPS))
        pct_red_up = abs_red_up / denom * 100.0
        pct_ok_up = (pct_red_up >= min_pct) | (baseline_fsi < guard) if min_pct > 0 else pd.Series(True, index=df.index)
        passing_up = beneficial_up & (abs_red_up >= min_abs) & pct_ok_up

        actionable_level = current.isin(order) & (current != int(best))
        actionable = passing_up & actionable_level
        count_actionable = int(actionable.sum())
        per_treatment_actionable_counts[treatment] = count_actionable
        n_actionable_upgrades += count_actionable
        if actionable.any():
            n_units_with_any_actionable.update(seg_ids[actionable].astype(int).tolist())

    return {
        "population": population,
        "min_abs": scenario.min_abs,
        "min_pct": scenario.min_pct,
        "n_candidates_beneficial": n_beneficial,
        "n_candidates_passing_thresholds": n_pass_thresholds,
        "n_total_prescriptions": n_total_prescriptions,
        "n_actionable_upgrades": n_actionable_upgrades,
        "n_units_covered": len(n_units_with_any_actionable),
        "n_units_with_any_prescription": len(n_units_with_any_prescription),
    }


def main() -> int:
    print("=" * 70)
    print("TABLE A1: THRESHOLD SENSITIVITY (Stage 2 prescriptions)")
    print("=" * 70)

    parser = argparse.ArgumentParser(description="Table A1 threshold sensitivity")
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Run folder name under stage2_outputs/runs/. "
            "Example: --run-id base111 (reads from stage2_outputs/runs/base111/)"
        ),
    )
    parser.add_argument(
        "--stage2-output-dir",
        default=None,
        help=(
            "Explicit Stage 2 output directory that contains hierarchical_cf/ and data/. "
            "Overrides --run-id when provided."
        ),
    )
    parser.add_argument(
        "--population",
        choices=["hotspots", "all_segments", "both"],
        default="hotspots",
        help="Which population to run on (default: hotspots)",
    )
    args = parser.parse_args()

    stage2_root = resolve_stage2_root(stage2_output_dir=args.stage2_output_dir, run_id=args.run_id)
    print(f"Stage 2 root: {stage2_root}")

    populations = [args.population]
    if args.population == "both":
        populations = ["hotspots", "all_segments"]

    scenarios = [Scenario(a, r) for a in ABS_THRESHOLDS for r in PCT_THRESHOLDS]

    all_rows: list[dict] = []
    for pop in populations:
        print(f"\nLoading population: {pop}")
        df = _load_population(population=pop, stage2_root=stage2_root)
        print(f"✓ Rows loaded: {len(df):,}")

        print("Merging current treatment levels...")
        current_levels = _load_current_levels(stage2_root=stage2_root)
        df = df.merge(current_levels, on="segment_id", how="left")

        current_cols = list(stage2_prescriptions.CURRENT_LEVEL_COLUMN_MAP.values())
        missing_current = int(df[current_cols].isna().any(axis=1).sum())
        if missing_current:
            print(f"⚠️ Missing current treatment levels for {missing_current:,} rows")

        for s in scenarios:
            print(f"- Scenario A≥{s.min_abs:.3f}, R≥{s.min_pct:.1f}%")
            metrics = compute_scenario_metrics(df, s, population=pop)
            print(
                f"    beneficial: {metrics['n_candidates_beneficial']:,}; "
                f"pass: {metrics['n_candidates_passing_thresholds']:,}; "
                f"total: {metrics['n_total_prescriptions']:,}; "
                f"actionable: {metrics['n_actionable_upgrades']:,}; "
                f"units covered: {metrics['n_units_covered']:,}"
            )
            all_rows.append(metrics)

    out_df = pd.DataFrame(all_rows)
    out_df = out_df.sort_values(["population", "min_abs", "min_pct"], ascending=[True, True, True]).reset_index(drop=True)

    # Keep output files minimal (no extra diagnostic columns).
    minimal_cols = [
        "population",
        "min_abs",
        "min_pct",
        "n_total_prescriptions",
        "n_actionable_upgrades",
        "n_units_covered",
    ]
    out_df_min = out_df[minimal_cols].copy()

    # Outputs
    reports_dir: Path = stage2_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    suffix = "" if args.population == "hotspots" else f"_{args.population}"
    out_csv = reports_dir / f"table_A1_threshold_sensitivity{suffix}.csv"
    out_tex = reports_dir / f"table_A1_threshold_sensitivity{suffix}.tex"

    out_df_min.to_csv(out_csv, index=False)

    caption = (
        "Sensitivity of prescriptions to absolute (A) and relative (R) reduction thresholds. "
        "Counts reflect actionable code-level upgrades only (no-change validations excluded)."
    )
    # For TeX, render one table per population to keep headers correct.
    if args.population == "both":
        tex_parts: list[str] = []
        for pop in populations:
            sub = out_df_min[out_df_min["population"] == pop].copy()
            tex_parts.append(_make_latex_table(sub, caption=f"{caption} (Population: {pop}.)", population=pop))
        out_tex.write_text("\n\n".join(tex_parts), encoding="utf-8")
    else:
        tex = _make_latex_table(out_df_min, caption=caption, population=populations[0])
        out_tex.write_text(tex, encoding="utf-8")

    print("\nOutputs:")
    print(f"- {out_csv}")
    print(f"- {out_tex}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
