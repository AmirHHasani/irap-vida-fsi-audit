"""Create Figure 3: Hotspot CATA (association) by treatment (hotspots only).

Reads the Stage 2 hotspot outputs (TP+FP candidates; 396 hotspots) and produces a
clean horizontal bar chart showing distributions of hotspot-level association
estimates (CATA) for the 6 tracked treatments.

Design choice:
- Bar: mean association across hotspots
- Whiskers: 2.5th–97.5th percentiles of hotspot association estimates (robust)

Supports both:
- Legacy outputs with one column per treatment (e.g., centreline_rumble_strips_cate)
- Current contrast-based outputs (e.g., centreline_rumble_strips__0_to_1_cate)
    In this case, we derive ONE actionable contrast per hotspot and treatment using
    the hotspot's current level + stage2_contrast_spec.json (same selection rule as
    Stage 2 prescriptions: adjacent upgrade current->next).

Outputs:
- fig3_cate_hotspots.pdf (vector)
- fig3_cate_hotspots.png
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

from stage2_config import OUTPUT_DIR, ALL_TREATMENTS


# Okabe-Ito-ish palette (matches other publication scripts in this repo)
COLORS = {
    "beneficial": "#009E73",  # green
    "harmful": "#D55E00",  # orange-red
    "stage2": "#0072B2",  # blue
}


TREATMENT_COLUMNS = {
    # Legacy compatibility (older Stage 2 outputs that exported one column per treatment)
    "Centreline rumble strips": "centreline_rumble_strips_cate",
    "Delineation": "delineation_cate",
    "Street lighting": "street_lighting_cate",
    "Paved shoulder (driver-side)": "paved_shoulder_driver-side_cate",
    "Paved shoulder (passenger-side)": "paved_shoulder_passenger-side_cate",
    "Road condition": "road_condition_cate",
}


TREATMENT_LABEL_TO_BASE = {
    "Centreline rumble strips": "Centreline rumble strips",
    "Delineation": "Delineation",
    "Street lighting": "Street lighting",
    "Paved shoulder (driver-side)": "Paved shoulder - driver-side",
    "Paved shoulder (passenger-side)": "Paved shoulder - passenger-side",
    "Road condition": "Road condition",
}


# Match the typical paper ordering used in prior versions (top→bottom)
TREATMENT_ORDER = [
    "Road condition",
    "Paved shoulder (passenger-side)",
    "Paved shoulder (driver-side)",
    "Street lighting",
    "Delineation",
    "Centreline rumble strips",
]

# Plot choices
# - Bar shows the mean hotspot CATE per treatment (as in your reference figure)
# - Whiskers should ideally be robust to outliers; default is 2.5–97.5% interval.
#   Set to "minmax" if you explicitly want full min–max spread.
WHISKERS_MODE = "p95"  # "p95" or "minmax"

# Which per-treatment contrast to plot when Stage 2 outputs are contrast-based.
# - fixed_all_hotspots: plot a fixed, interpretable contrast across ALL 396 hotspots (default; matches paper caption).
# - actionable_by_current: plot the adjacent upgrade current->next per hotspot (matches prescription eligibility; n may be <396).
DEFAULT_CONTRAST_MODE = "fixed_all_hotspots"  # or "actionable_by_current"


def _runs_root() -> Path:
    return Path(__file__).resolve().parent / 'stage2_outputs' / 'runs'


def resolve_stage2_root(*, stage2_output_dir: str | None, run_id: str | None) -> Path:
    """Resolve the Stage 2 output root that contains hierarchical_cf/."""
    if stage2_output_dir:
        return Path(stage2_output_dir).expanduser().resolve()
    if run_id:
        return (_runs_root() / str(run_id)).resolve()
    return Path(OUTPUT_DIR).resolve()


def _normalize_treatment_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "treatment"


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


def _pick_adjacent_upgrade(
    *,
    treatment_base: str,
    current_level: int,
    contrast_index: dict[tuple[str, int, int], dict],
) -> tuple[int, int] | None:
    """Pick the smallest available t1>t0 for this treatment_base at current_level."""
    candidates: list[int] = []
    for (base, t0, t1), _ in contrast_index.items():
        if base == treatment_base and t0 == current_level and t1 > t0:
            candidates.append(t1)
    if not candidates:
        return None
    return current_level, min(candidates)


def _load_current_levels(stage2_root: Path) -> pd.DataFrame:
    """Load current (canonical) levels for each hotspot and treatment from run-local analysis_dataset."""
    path = Path(stage2_root).resolve() / "data" / "analysis_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"analysis_dataset.csv not found: {path}. "
            "Run create_analysis_dataset.py (or rerun Stage 2) to produce run-local data/."
        )
    usecols = ["Location ID"] + list(ALL_TREATMENTS)
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.rename(columns={"Location ID": "segment_id"})
    df["segment_id"] = df["segment_id"].astype(int)

    rename_map = {t: f"current_{_normalize_treatment_name(t)}" for t in ALL_TREATMENTS}
    return df.rename(columns=rename_map)


def _ieee_single_column_figsize(height_in: float) -> tuple[float, float]:
    # IEEE single-column is typically ~3.5 in wide.
    return (3.50, height_in)


def _load_hotspot_cates(*, hotspot_csv: Path, stage2_root: Path, contrast_mode: str) -> pd.DataFrame:
    df = pd.read_csv(hotspot_csv)

    if "segment_id" in df.columns:
        n_segments = int(df["segment_id"].nunique())
        if n_segments != 396:
            print(f"Note: hotspot file has {n_segments} unique segments (expected 396).")

    # Path A: legacy per-treatment columns already exist.
    if all(col in df.columns for col in TREATMENT_COLUMNS.values()):
        long_rows: list[dict[str, object]] = []
        for label, col in TREATMENT_COLUMNS.items():
            values = pd.to_numeric(df[col], errors="coerce")
            for v in values.values:
                if np.isfinite(v):
                    long_rows.append({"treatment": label, "cate": float(v)})

        out = pd.DataFrame(long_rows)
        if out.empty:
            raise ValueError("No finite CATA values found in hotspot file (legacy columns).")
        return out

    # Path B: contrast-based outputs.
    contrast_spec = _load_contrast_spec(stage2_root)
    contrast_index = _build_contrast_index(contrast_spec)
    if not contrast_index:
        raise ValueError(
            "Hotspot file does not include legacy per-treatment CATE columns, and stage2_contrast_spec.json "
            "was not found (or empty). Cannot determine which contrast to plot per hotspot." 
        )

    if "segment_id" not in df.columns:
        raise ValueError("Expected 'segment_id' column in hotspot file for contrast-based plotting.")

    long_rows: list[dict[str, object]] = []
    if contrast_mode == "fixed_all_hotspots":
        # Default: plot a fixed contrast per treatment across all hotspots.
        for label in TREATMENT_COLUMNS.keys():
            treatment_base = TREATMENT_LABEL_TO_BASE[label]

            # Prefer 0->1 for interpretability; otherwise fall back to the smallest available upgrade from the spec.
            item = contrast_index.get((treatment_base, 0, 1))
            if item:
                contrast = str(item.get("contrast") or f"{treatment_base}__0_to_1")
            else:
                upgrades = [(t0, t1) for (base, t0, t1) in contrast_index.keys() if base == treatment_base and t1 > t0]
                if not upgrades:
                    continue
                t0, t1 = sorted(upgrades, key=lambda x: (x[0], x[1]))[0]
                fallback_item = contrast_index.get((treatment_base, t0, t1), {})
                contrast = str(fallback_item.get("contrast") or f"{treatment_base}__{t0}_to_{t1}")

            cate_col = f"{_normalize_result_key(contrast)}_cate"
            if cate_col not in df.columns:
                raise ValueError(f"Expected CATA column not found in hotspot file: {cate_col}")

            values = pd.to_numeric(df[cate_col], errors="coerce")
            for v in values.values:
                if np.isfinite(v):
                    long_rows.append({"treatment": label, "cate": float(v)})
    elif contrast_mode == "actionable_by_current":
        # Prescription-aligned diagnostics: plot the adjacent upgrade current->next per hotspot.
        levels = _load_current_levels(stage2_root)
        merged = df.merge(levels, on="segment_id", how="left", validate="one_to_one")
        if merged.filter(like="current_").isna().all(axis=None):
            raise ValueError(
                "Failed to merge current levels for hotspots; ensure run-local data/analysis_dataset.csv contains canonical levels."
            )

        for label in TREATMENT_COLUMNS.keys():
            treatment_base = TREATMENT_LABEL_TO_BASE[label]
            current_col = f"current_{_normalize_treatment_name(treatment_base)}"
            if current_col not in merged.columns:
                raise ValueError(f"Missing current level column after merge: {current_col}")

            for _, row in merged.iterrows():
                cur = row.get(current_col)
                if cur is None or pd.isna(cur):
                    continue
                try:
                    cur_i = int(cur)
                except Exception:
                    continue

                picked = _pick_adjacent_upgrade(
                    treatment_base=treatment_base,
                    current_level=cur_i,
                    contrast_index=contrast_index,
                )
                if not picked:
                    continue
                t0, t1 = picked
                item = contrast_index.get((treatment_base, t0, t1), {})
                contrast = str(item.get("contrast") or f"{treatment_base}__{t0}_to_{t1}")
                cate_col = f"{_normalize_result_key(contrast)}_cate"
                v = row.get(cate_col)
                if v is None or pd.isna(v):
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    long_rows.append({"treatment": label, "cate": fv})
    else:
        raise ValueError(
            f"Unknown contrast_mode={contrast_mode!r}. Expected 'fixed_all_hotspots' or 'actionable_by_current'."
        )

    out = pd.DataFrame(long_rows)
    if out.empty:
        raise ValueError(
            "No finite CATA values found after deriving contrast-based hotspot associations. "
            "This usually indicates a mismatch between stage2_contrast_spec.json and hotspot CSV column naming." 
        )
    return out


def _summarize_by_treatment(df_long: pd.DataFrame) -> pd.DataFrame:
    def _pct(series: pd.Series, pred) -> float:
        s = series.dropna().astype(float)
        if s.empty:
            return float("nan")
        return 100.0 * float(pred(s).mean())

    grouped = df_long.groupby("treatment")["cate"]
    stats = grouped.agg(
        n="count",
        mean="mean",
        median="median",
        p025=lambda x: float(np.nanpercentile(x, 2.5)),
        p975=lambda x: float(np.nanpercentile(x, 97.5)),
        vmin=lambda x: float(np.nanmin(x)),
        vmax=lambda x: float(np.nanmax(x)),
    )
    stats["pct_beneficial"] = grouped.apply(lambda s: _pct(s, lambda x: x < 0.0))
    stats["pct_harmful"] = grouped.apply(lambda s: _pct(s, lambda x: x > 0.0))
    return stats.reset_index()


def create_fig3_cate_hotspots(
    hotspot_csv: Path | None = None,
    out_dir: Path | None = None,
    *,
    stage2_root: Path | None = None,
    contrast_mode: str = DEFAULT_CONTRAST_MODE,
) -> tuple[Path, Path]:
    if stage2_root is None:
        stage2_root = Path(OUTPUT_DIR).resolve()

    if hotspot_csv is None:
        hotspot_csv = (
            Path(stage2_root)
            / "hierarchical_cf"
            / "hotspot_level"
            / "hotspot_segments_detailed.csv"
        )

    # Report exactly what file we used (helps confirm you're plotting the newest run outputs)
    try:
        mtime = datetime.fromtimestamp(hotspot_csv.stat().st_mtime)
        print(f"Using hotspot association (CATA) file: {hotspot_csv}")
        print(f"Last modified: {mtime.isoformat(sep=' ', timespec='seconds')}")
    except Exception:
        print(f"Using hotspot association (CATA) file: {hotspot_csv}")

    if out_dir is None:
        out_dir = Path(stage2_root) / "reports"

    out_dir.mkdir(parents=True, exist_ok=True)

    df_long = _load_hotspot_cates(
        hotspot_csv=hotspot_csv,
        stage2_root=Path(stage2_root),
        contrast_mode=contrast_mode,
    )

    stats = _summarize_by_treatment(df_long)
    try:
        counts = stats[["treatment", "n"]].sort_values("treatment")
        print(f"Counts used per treatment ({contrast_mode}):")
        for _, r in counts.iterrows():
            print(f"  - {r['treatment']}: n={int(r['n'])}")
    except Exception:
        pass
    # Use fixed paper-style order when available; fall back to mean-sorted
    available = set(stats["treatment"].tolist())
    if all(t in available for t in TREATMENT_ORDER):
        order = TREATMENT_ORDER
        stats = stats.set_index("treatment").loc[order].reset_index()
    else:
        stats = stats.sort_values("mean", ascending=True)
        order = stats["treatment"].tolist()

    # Styling: match existing publication scripts (compact, clean)
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 8,
            "font.weight": "bold",
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        }
    )

    sns.set_style("whitegrid")

    # Figure size: tuned to the clean reference look (still reasonably column-friendly)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    # X-limits based on observed min/max whiskers
    if WHISKERS_MODE == "minmax":
        lo = stats["vmin"].values
        hi = stats["vmax"].values
        xmin = float(stats["vmin"].min())
        xmax = float(stats["vmax"].max())
    else:
        lo = stats["p025"].values
        hi = stats["p975"].values
        xmin = float(stats["p025"].min())
        xmax = float(stats["p975"].max())
    span = (xmax - xmin) if xmax > xmin else 0.2
    pad = 0.12 * span
    xmin -= pad
    xmax += pad

    # Plot per-treatment mean bars + 95% distribution whiskers
    y_pos = np.arange(len(order))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order, fontweight="bold")

    means = stats["mean"].values

    # Horizontal bars from 0 to mean (thicker, cleaner)
    ax.barh(
        y_pos,
        means,
        left=0.0,
        height=0.62,
        color=COLORS["stage2"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.25,
        zorder=2,
    )

    # Whiskers: distribution spread across hotspots (robust by default)
    ax.hlines(
        y=y_pos,
        xmin=lo,
        xmax=hi,
        color="#243447",
        linewidth=1.0,
        alpha=0.95,
        zorder=3,
    )

    # Zero line (clean reference style)
    ax.axvline(0, color="#22313F", linestyle="--", linewidth=1.4, alpha=0.95, zorder=1)

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Mean CATA (Δ log FSI; hotspot segments only)", fontweight="bold")
    ax.set_ylabel("")

    # Clean, non-overlapping decimal ticks
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='x', labelsize=8)

    # Force bold tick labels (matplotlib doesn't always apply rcParams to tick labels)
    for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick.set_fontweight("bold")

    # No title (matches the clean reference figure style)
    ax.set_title("")

    # Grid + spines to match the clean look
    ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    sns.despine(ax=ax, top=True, right=True)

    # Tight layout for column width
    fig.tight_layout(pad=0.5)

    pdf_path = out_dir / "fig3_cate_hotspots.pdf"
    png_path = out_dir / "fig3_cate_hotspots.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create Figure 3: hotspot association (CATA) distributions by treatment. "
            "Run-aware via --run-id (stage2_outputs/runs/<run-id>)."
        )
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stage 2 run id under stage2_outputs/runs/<run-id>. If provided, reads/writes within that run folder.",
    )
    parser.add_argument(
        "--stage2-output-dir",
        default=None,
        help="Explicit Stage 2 output root directory containing hierarchical_cf/. Overrides --run-id.",
    )
    parser.add_argument(
        "--hotspot-csv",
        default=None,
        help="Optional explicit hotspot CSV path. Overrides default <run>/hierarchical_cf/hotspot_level/hotspot_segments_detailed.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory for figures. Defaults to <run>/reports (or OUTPUT_DIR/reports).",
    )
    parser.add_argument(
        "--contrast-mode",
        default=DEFAULT_CONTRAST_MODE,
        choices=["fixed_all_hotspots", "actionable_by_current"],
        help=(
            "How to pick a per-hotspot contrast when Stage 2 outputs are contrast-based. "
            "fixed_all_hotspots matches the paper caption (one fixed contrast per treatment over all 396 hotspots). "
            "actionable_by_current matches prescription eligibility (current->next; n may be <396)."
        ),
    )
    args = parser.parse_args()

    stage2_root = resolve_stage2_root(stage2_output_dir=args.stage2_output_dir, run_id=args.run_id)
    hotspot_csv = Path(args.hotspot_csv).expanduser().resolve() if args.hotspot_csv else None
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None

    pdf_path, png_path = create_fig3_cate_hotspots(
        hotspot_csv=hotspot_csv,
        out_dir=out_dir,
        stage2_root=stage2_root,
        contrast_mode=str(args.contrast_mode),
    )
    print("✓ Saved:", pdf_path)
    print("✓ Saved:", png_path)


if __name__ == "__main__":
    main()
